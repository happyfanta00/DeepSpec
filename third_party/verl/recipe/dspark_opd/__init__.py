"""DSpark-OPD recipe package init.

Runs at import time in BOTH the driver and every Ray worker process (they import this
package to resolve the recipe modules), so it's the right place for the global compat
setups that must happen before verl touches transformers / sglang / builds a rollout:

  1. transformers compat shim: verl 0.7.0 uses `AutoModelForVision2Seq`, removed/renamed
     to `AutoModelForImageTextToText` in transformers 5.10.2. Alias it before any verl
     import chain that pulls verl/utils/model.py. (Mirrors scripts/opd/verl_compat.py in
     the DeepSpec repo; inlined here to avoid a cross-repo path dependency in Ray workers.)
     ⚠️ ORDER: importing sglang HEAD REPLACES the `transformers` module object in sys.modules
     (observed: id() changes), wiping this alias. So we apply the sglang shim (2) FIRST, then
     this alias LAST — otherwise the alias is silently lost before verl's rollout import needs it.

  2. sglang launch-API compat shim (Stage-M2 T3): verl 0.7.0's async sglang server was
     written against sglang 0.5.2, where `http_server._launch_subprocesses(server_args=...)`
     was a MODULE-LEVEL function returning a 3-tuple (tokenizer_manager, template_manager,
     scheduler_info). Our vendored sglang HEAD (third_party/sglang) moved it to
     `Engine._launch_subprocesses` (a classmethod taking 3 launch-func callables and
     returning a 5-tuple). verl's async_sglang_server does `from http_server import
     _launch_subprocesses` at module load + calls it 3-tuple-style, so it breaks on HEAD.
     We inject a module-level 3-tuple-compatible `_launch_subprocesses` into http_server
     so the verl HYBRID rollout path works unchanged. Same spirit as (1): verl-vs-dep
     version skew fixed by a recipe-side shim, not by editing vendored verl core.

  3. custom rollout registration: insert ("dspark","sync") into verl's _ROLLOUT_REGISTRY
     so get_rollout_class(name="dspark", mode="sync") resolves to our DSparkRollout.
     Entries are FQDN strings resolved lazily via importlib, so a plain dict insert works.
"""
from __future__ import annotations


def _ensure_cuda_env() -> None:
    """Make CUDA_HOME + PATH/LD_LIBRARY_PATH self-contained for EVERY process that imports this
    recipe (driver, Ray training workers, the sglang server actor).

    WHY here: importing sglang HEAD pulls the CUDA-heavy srt stack — deep_gemm asserts CUDA_HOME is
    set (deep_gemm/cuda_helpers.py:find_cuda_home) and flashinfer JIT needs nvcc on PATH. The
    training workers launch via `VENV_PY=<abs path>` WITHOUT activating the venv, so their env has
    NO CUDA_HOME and NO venv/bin on PATH. Without this, a worker's first real sglang import (e.g.
    make_sglang_client -> from http_server_engine import ... -> deep_gemm) fails with a bare
    AssertionError. Set it BEFORE _apply_sglang_launch_compat's sglang import below. Idempotent
    (setdefault); the pure-training env (no sglang) is unaffected — nothing here imports sglang.
    Mirrors sglang_server.__init__'s env setup so all three process kinds agree.
    """
    import os
    import sys

    cuda_home = os.environ.setdefault("CUDA_HOME", "/opt/pytorch/cuda")
    venv_bin = os.path.dirname(sys.executable)   # ninja / nvcc-adjacent tools live here
    if venv_bin not in os.environ.get("PATH", "").split(":"):
        os.environ["PATH"] = f"{venv_bin}:{cuda_home}/bin:" + os.environ.get("PATH", "")
    ld = os.environ.get("LD_LIBRARY_PATH", "")
    if f"{cuda_home}/lib64" not in ld:
        os.environ["LD_LIBRARY_PATH"] = f"{cuda_home}/lib:{cuda_home}/lib64:{ld}"


def _apply_transformers_compat() -> list[str]:
    import transformers

    aliases = [("AutoModelForVision2Seq", "AutoModelForImageTextToText")]
    patched = []
    for old, new in aliases:
        if not hasattr(transformers, old) and hasattr(transformers, new):
            setattr(transformers, old, getattr(transformers, new))
            patched.append(old)
    return patched


def _apply_sglang_launch_compat() -> list[str]:
    """Bridge sglang-0.5.2 symbols that verl 0.7 imports but our sglang HEAD relocated.

    verl 0.7's sglang rollout package was written against sglang 0.5.2; a handful of the
    symbols it does `from sglang... import` at module load moved in HEAD. We monkey-patch
    the OLD locations to re-expose them so verl's HYBRID rollout imports work unchanged
    (same spirit as the transformers alias: fix version skew recipe-side, not in vendored
    verl core). Two relocations covered:
      (a) http_server._launch_subprocesses: module-level 3-tuple fn (0.5.2) -> classmethod
          Engine._launch_subprocesses returning a 5-tuple (HEAD). We inject a module-level
          3-tuple-compatible wrapper (scheduler_info == scheduler_init_result.scheduler_infos[0]).
      (b) sglang.srt.utils.{get_open_port,get_local_ip_auto,get_ip}: moved into the
          sglang.srt.utils.network submodule (no longer re-exported from the package). verl
          imports get_open_port BARE (unguarded, sglang_rollout.py:41) so this must be fixed.
          get_ip: HEAD has no get_ip — verl's guarded fallback uses get_local_ip_auto; we
          re-export both names for robustness.

    Guarded on purpose: importing sglang pulls the CUDA-heavy srt stack (needs CUDA_HOME).
    This __init__ also runs in the pure-training env (~/.venv/dspark-opd, no sglang) and in a
    CPU-only driver — there sglang isn't importable and this shim silently no-ops (the current
    non-sglang fused train_step path is unaffected). Where a needed sglang import can't be
    resolved even in the unified env, verl's own `from sglang... import` fails loudly at the
    point of use, which is the right signal. Returns the list of shims actually installed.
    """
    installed: list[str] = []

    # (a) http_server._launch_subprocesses 3-tuple wrapper
    try:
        import sglang.srt.entrypoints.http_server as _hs
    except Exception:  # noqa: BLE001 — sglang absent (pure-training env) or CUDA_HOME unset
        return installed
    if not hasattr(_hs, "_launch_subprocesses"):
        try:
            from sglang.srt.entrypoints.engine import (
                Engine,
                init_tokenizer_manager,
                run_detokenizer_process,
                run_scheduler_process,
            )

            def _launch_subprocesses(server_args, port_args=None):
                # Adapt HEAD's Engine._launch_subprocesses (5-tuple) to 0.5.2's module-level
                # 3-tuple contract. scheduler_info (0.5.2 single Dict) == scheduler_infos[0].
                (tokenizer_manager, template_manager, _port_args,
                 scheduler_init_result, _watchdog) = Engine._launch_subprocesses(
                    server_args=server_args,
                    init_tokenizer_manager_func=init_tokenizer_manager,
                    run_scheduler_process_func=run_scheduler_process,
                    run_detokenizer_process_func=run_detokenizer_process,
                    port_args=port_args,
                )
                return (tokenizer_manager, template_manager,
                        scheduler_init_result.scheduler_infos[0])

            _hs._launch_subprocesses = _launch_subprocesses
            installed.append("_launch_subprocesses")
        except Exception:  # noqa: BLE001 — unexpected sglang layout; leave verl to fail loudly
            pass

    # (b) re-export relocated network helpers into the sglang.srt.utils package namespace
    try:
        import sglang.srt.utils as _u
        from sglang.srt.utils import network as _net

        for name in ("get_open_port", "get_local_ip_auto", "get_ip"):
            if not hasattr(_u, name) and hasattr(_net, name):
                setattr(_u, name, getattr(_net, name))
                installed.append(f"utils.{name}")
    except Exception:  # noqa: BLE001 — network submodule layout changed; verl fails loudly
        pass

    return installed


def _register_rollout() -> None:
    from verl.workers.rollout import base as _rollout_base

    _rollout_base._ROLLOUT_REGISTRY[("dspark", "sync")] = (
        "recipe.dspark_opd.rollout.DSparkRollout"
    )


# CUDA env FIRST — deep_gemm/flashinfer imported transitively by the sglang shim below assert
# CUDA_HOME is set; workers launched via VENV_PY (no venv activate) lack it. Must precede any
# sglang import (the shim, and later make_sglang_client in the workers).
_ensure_cuda_env()
# ORDER MATTERS (see docstring §1): sglang import replaces sys.modules["transformers"], so run
# the sglang launch shim FIRST, then apply the transformers alias LAST so it survives to verl's
# rollout import. In the pure-training env (no sglang) the shim no-ops and order is irrelevant.
_SGLANG_LAUNCH_SHIMMED = _apply_sglang_launch_compat()
_COMPAT_PATCHED = _apply_transformers_compat()
_register_rollout()
