"""DSpark-OPD Stage-M2 T3a — standalone DSPARK sglang server actor (infra only).

WHY a separate Ray actor (not built inside the training worker):
  The dspark actor-rollout workers each run as a Ray actor with num_gpus=1, and the
  recipe does NOT set RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES, so Ray pins each
  training worker to see ONLY its own 1 GPU (single_controller/base/worker.py). A
  tp_size=8 sglang server must see all 8 GPUs, which no training-worker process can.
  So we host the tp=8 server in a DEDICATED Ray actor that gains full-node GPU
  visibility via RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1 + an explicit joined
  CUDA_VISIBLE_DEVICES + NodeAffinity(soft=False) — the same physical placement verl's
  official async path uses (async_sglang_server.py:284-300). We reuse only that
  PLACEMENT trick; we do NOT wrap it in verl's SGLangReplica/AgentLoop orchestration.

The actor holds an sglang HttpServerEngineAdapter(speculative_algorithm="DSPARK", ...)
— the exact same engine class the T2 probe validated (dspark_draft_update_probe.py) —
which launches ONE sglang server subprocess spanning all 8 GPUs (tp=8) and exposes:
  - generate(input_ids, sampling_params)            -> live DSPARK spec-decode, raw dict
  - release() / resume()                            -> memory-occupation sleep/wake
  - update_weights(named, draft_model_only=True)    -> T6 draft-only weight sync (CUDA-IPC)
  - get_address()                                   -> (host, port) for training workers
  - shutdown()                                      -> kill the server subprocess

The actor takes NO Ray GPU resources (num_gpus unset): it SHARES the 8 GPUs already
allocated to the training workers. Colocation is safe because rollout and training are
time-multiplexed (train_step brackets generation with resume/release; §5.0).

See docs/opd/dspark-on-sglang-design.md §5.6a.
"""
from __future__ import annotations

import os
from typing import Any

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy


# ---- attention backend: flashinfer (triton crashes on spec-decode long seqs, §4.5) ----
_DEFAULT_ATTN_BACKEND = "flashinfer"


def make_sglang_client(host: str, port: int, tp_size: int, model_path: str):
    """T3b/T6: a pure HTTP CLIENT to an already-running DSPARK server (no new server launched).

    ⚠️ This submodule's HttpServerEngineAdapter.__init__ ALWAYS launches a server subprocess (line
    61 unconditional launch_server_process) and has NO launch_server=False client mode (that flag is
    verl's adapter, not this sglang one — and it's not a ServerArgs field). So we build the adapter
    via __new__ (bypass __init__/launch) and set server_args ourselves to point at the existing
    server, reusing its exact generate / update_weights_from_tensor / _make_request logic (incl. the
    per-TP-rank serialization that needs the right tp_size). Returns the adapter instance (client).

    model_path is only to satisfy ServerArgs.__post_init__ validation (real path, e.g. the target)
    — the client never loads it; host/port/tp_size are what actually matter.
    """
    from sglang.srt.entrypoints.http_server_engine import HttpServerEngineAdapter
    from sglang.srt.server_args import ServerArgs

    client = HttpServerEngineAdapter.__new__(HttpServerEngineAdapter)   # no __init__ -> no launch
    # host/port to reach the server; tp_size so update_weights_from_tensor serializes one copy per
    # TP rank (server scatters serialized_named_tensors[rank]; a wrong tp_size drops ranks 1..N-1).
    client.server_args = ServerArgs(model_path=model_path, host=host, port=int(port),
                                    tp_size=int(tp_size))
    client.process = None                                              # no owned subprocess
    return client


@ray.remote
class DSparkSglangServer:
    """Ray actor hosting a tp=8 DSPARK sglang server (HttpServerEngineAdapter)."""

    def __init__(
        self,
        *,
        target_path: str,
        draft_path: str,
        tp_size: int,
        cuda_visible_devices: str,
        port: int,
        mem_fraction_static: float = 0.6,
        attention_backend: str = _DEFAULT_ATTN_BACKEND,
        trust_remote_code: bool = True,
    ):
        # Full-node GPU visibility: RAY_EXPERIMENTAL_NOSET_* is set on this actor's
        # runtime_env (see build_dspark_sglang_server), so Ray does NOT overwrite this.
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
        # sgl_kernel / deep_gemm need the CUDA toolkit on PATH/LD_LIBRARY_PATH.
        cuda_home = os.environ.setdefault("CUDA_HOME", "/opt/pytorch/cuda")
        # flashinfer JIT-compiles some attention kernels ON DEMAND via `ninja` (a real subprocess,
        # cpp_ext.run_ninja). ninja lives in this venv's bin (installed as the `ninja` pip pkg), but
        # the actor's spawned scheduler subprocesses may not inherit that on PATH -> FileNotFoundError:
        # 'ninja' during cuda-graph capture (only bites when a not-yet-cached kernel shape must be
        # built — e.g. a new mem_fraction changes the captured prefill batch sizes). Prepend the venv
        # bin (== dir of sys.executable) so ninja is always found.
        import sys as _sys
        venv_bin = os.path.dirname(_sys.executable)
        os.environ["PATH"] = f"{venv_bin}:{cuda_home}/bin:" + os.environ.get("PATH", "")
        ld = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = f"{cuda_home}/lib:{cuda_home}/lib64:{ld}"

        from sglang.srt.entrypoints.http_server_engine import HttpServerEngineAdapter

        self._port = int(port)
        self._host = "127.0.0.1"  # server actor & training workers share the node (NodeAffinity)
        # ONE server subprocess spanning tp_size GPUs. **kwargs -> ServerArgs unmodified,
        # so speculative_* flow through (unlike verl sync's whitelist which drops them).
        self._engine = HttpServerEngineAdapter(
            model_path=target_path,                              # Qwen3-4B (target)
            speculative_algorithm="DSPARK",
            speculative_draft_model_path=draft_path,             # gamma auto-read from ckpt config
            speculative_draft_attention_backend=attention_backend,
            attention_backend=attention_backend,
            tp_size=int(tp_size),
            mem_fraction_static=float(mem_fraction_static),
            enable_memory_saver=True,                            # release/resume depends on this
            # ★ CPU-backup for BOTH target and draft weights. Without these, release_memory_
            # occupation frees the weights' GPU pages and resume remaps FRESH GARBAGE pages
            # (torch_memory_saver only preserves content when enable_cpu_backup=True;
            # load_model_utils.py:218). We verified (T3a smoke): without them, resume corrupts
            # ALL weights (target 290/290 + draft 46/46 changed -> gibberish, accept 8.00/1.00).
            # This mirrors Draft-OPD's composed-DFLASH-student server (its async_sglang_server.py
            # :315-329 sets both for a draft-model spec-decode server exactly like ours).
            enable_weights_cpu_backup=True,
            enable_draft_weights_cpu_backup=True,
            trust_remote_code=trust_remote_code,
            port=self._port,
        )
        print(f"[DSparkSglangServer] up: tp={tp_size} devices={cuda_visible_devices} "
              f"port={self._port} target={target_path} draft={draft_path}", flush=True)

    def get_address(self) -> tuple[str, int]:
        """(host, port) so training workers can build launch_server=False HTTP clients."""
        return self._host, self._port

    def generate(self, input_ids: list[int], sampling_params: dict[str, Any]) -> dict:
        """Live DSPARK speculative decoding. Returns the raw server JSON dict
        (output_ids + meta_info incl. dspark_accept_state)."""
        return self._engine.generate(input_ids=input_ids, sampling_params=sampling_params)

    def resume(self, tags=None):
        """Wake: reclaim GPU memory for `tags` (None = all: KV + weights + cuda_graph).

        For the KV-offload rollout/train time-multiplexing (§5.6c) pass tags=["kv_cache"] to
        rebuild ONLY the KV pool (fresh empty; no CPU copy) before a rollout phase, leaving the
        (large, cpu-backup'd) weights resident the whole time.
        """
        return self._engine.resume_memory_occupation(tags=tags)

    def release(self, tags=None):
        """Sleep: release GPU memory for `tags` (None = all) back to training.

        tags=["kv_cache"] frees ONLY the KV pool (flush_cache + free pages, no CPU stash — KV
        content is discarded, rebuilt empty on the next resume) after a rollout phase, so training
        gets the KV budget back while weights stay put (avoids weight export/import churn + the
        weight-corruption class of bugs, §resume-bug).
        """
        return self._engine.release_memory_occupation(tags=tags)

    def cuda_mem_info(self) -> dict:
        """(free, total, used) bytes on the server's local GPU 0 — for verifying that a KV
        release/resume actually moves the GPU footprint as expected (§5.6c smoke)."""
        import torch
        free, total = torch.cuda.mem_get_info(0)
        return {"free": int(free), "total": int(total), "used": int(total - free)}

    def update_weights(self, named_tensors, draft_model_only: bool = True,
                       flush_cache: bool = True):
        """T6: push freshly-trained draft weights (draft_model_only) via CUDA-IPC."""
        return self._engine.update_weights_from_tensor(
            named_tensors, draft_model_only=draft_model_only, flush_cache=flush_cache)

    def weights_checker(self, action: str = "checksum") -> dict:
        """/weights_checker passthrough (T2/T6 checksum verification)."""
        return self._engine._make_request("weights_checker", {"action": action})

    def shutdown(self):
        """Kill the sglang server subprocess (kill_process_tree)."""
        try:
            self._engine.shutdown()
        except Exception as e:  # noqa: BLE001 — best-effort teardown
            print(f"[DSparkSglangServer] shutdown error (ignored): {e!r}", flush=True)


def build_dspark_sglang_server(
    *,
    worker_group,
    target_path: str,
    draft_path: str,
    tp_size: int = 8,
    port: int = 30000,
    mem_fraction_static: float = 0.6,
    attention_backend: str = _DEFAULT_ATTN_BACKEND,
    trust_remote_code: bool = True,
):
    """Build the DSparkSglangServer actor colocated on the training workers' GPUs.

    Gathers each training worker's (node_id, CUDA_VISIBLE_DEVICES) — exactly like
    verl's SGLangReplica.launch_servers (async_sglang_server.py:261-278) — then pins
    the server actor to that node with full-node GPU visibility.

    Args:
        worker_group: the RayWorkerGroup (trainer.actor_rollout_wg); .workers is a list
                      of Ray actor handles of length == world_size (== tp_size here).
        target_path / draft_path: served target + DSPARK draft ckpt.
        tp_size: sglang tensor-parallel size (== #training GPUs on the node, 8).
    Returns:
        (server_actor, (host, port))
    """
    workers = worker_group.workers
    assert len(workers) >= tp_size, (
        f"need >= tp_size={tp_size} workers to gather 8-GPU visibility, got {len(workers)}"
    )

    # (node_id, CUDA_VISIBLE_DEVICES) per worker — same pattern as verl async.
    worker_infos = ray.get([
        w.__ray_call__.remote(
            lambda self: (
                ray.get_runtime_context().get_node_id(),
                os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            )
        )
        for w in workers[:tp_size]
    ])
    node_ids = [info[0] for info in worker_infos]
    devices = [info[1] for info in worker_infos]
    assert len(set(node_ids)) == 1, (
        f"T3a assumes single-node 8-GPU colocation; workers span nodes {set(node_ids)}"
    )
    node_id = node_ids[0]
    cuda_visible_devices = ",".join(devices)  # e.g. "0,1,2,3,4,5,6,7"

    server = DSparkSglangServer.options(
        scheduling_strategy=NodeAffinitySchedulingStrategy(
            node_id=node_id, soft=False,
        ),
        # NOSET -> Ray won't pin CUDA_VISIBLE_DEVICES; the actor sets it to all 8 itself.
        runtime_env={"env_vars": {"RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1"}},
        name="dspark_sglang_server",
    ).remote(
        target_path=target_path,
        draft_path=draft_path,
        tp_size=tp_size,
        cuda_visible_devices=cuda_visible_devices,
        port=port,
        mem_fraction_static=mem_fraction_static,
        attention_backend=attention_backend,
        trust_remote_code=trust_remote_code,
    )
    host, port = ray.get(server.get_address.remote())
    print(f"[build_dspark_sglang_server] actor on node={node_id} devices={cuda_visible_devices} "
          f"-> http://{host}:{port}", flush=True)
    return server, (host, port)
