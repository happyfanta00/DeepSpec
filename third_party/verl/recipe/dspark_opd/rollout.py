"""DSpark-OPD rollout — minimal placeholder.

The fused `train_step` (worker.py) drives the sglang DSPARK rollout directly (via the tp=8 server
client), so this rollout object's `generate_sequences` is NEVER called. It exists only because
verl's `_build_rollout` instantiates the registered rollout class (("dspark","sync") ->
DSparkRollout, see __init__.py) during worker init; `set_module` is the attach hook the actor
worker calls. BaseRollout's abstract resume/update_weights/release are async no-ops.
"""
from __future__ import annotations

from verl import DataProto
from verl.workers.rollout.base import BaseRollout


class DSparkRollout(BaseRollout):
    def __init__(self, config, model_config, device_mesh):
        super().__init__(config=config, model_config=model_config, device_mesh=device_mesh)
        self.module = None   # attached by the actor worker after FSDP build (unused; sglang drives rollout)

    async def resume(self, tags):  # noqa: D401
        return None

    async def update_weights(self, weights, **kwargs):  # noqa: D401
        return None

    async def release(self):  # noqa: D401
        return None

    def set_module(self, module):
        """Attach the draft nn.Module (called by the actor worker; unused — sglang does rollout)."""
        self.module = module

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        raise NotImplementedError(
            "DSparkRollout.generate_sequences is not used — the fused train_step drives the "
            "sglang DSPARK rollout directly. This rollout object exists only for verl worker init.")
