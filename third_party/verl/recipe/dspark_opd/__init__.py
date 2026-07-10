"""DSpark-OPD recipe package init.

Runs at import time in BOTH the driver and every Ray worker process (they import this
package to resolve the recipe modules), so it's the right place for two global setups
that must happen before verl touches transformers / builds a rollout:

  1. transformers compat shim: verl 0.7.0 uses `AutoModelForVision2Seq`, removed/renamed
     to `AutoModelForImageTextToText` in transformers 5.10.2. Alias it before any verl
     import chain that pulls verl/utils/model.py. (Mirrors scripts/opd/verl_compat.py in
     the DeepSpec repo; inlined here to avoid a cross-repo path dependency in Ray workers.)

  2. custom rollout registration: insert ("dspark","sync") into verl's _ROLLOUT_REGISTRY
     so get_rollout_class(name="dspark", mode="sync") resolves to our DSparkRollout.
     Entries are FQDN strings resolved lazily via importlib, so a plain dict insert works.
"""
from __future__ import annotations


def _apply_transformers_compat() -> list[str]:
    import transformers

    aliases = [("AutoModelForVision2Seq", "AutoModelForImageTextToText")]
    patched = []
    for old, new in aliases:
        if not hasattr(transformers, old) and hasattr(transformers, new):
            setattr(transformers, old, getattr(transformers, new))
            patched.append(old)
    return patched


def _register_rollout() -> None:
    from verl.workers.rollout import base as _rollout_base

    _rollout_base._ROLLOUT_REGISTRY[("dspark", "sync")] = (
        "recipe.dspark_opd.rollout.DSparkRollout"
    )


_COMPAT_PATCHED = _apply_transformers_compat()
_register_rollout()
