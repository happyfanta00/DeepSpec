"""DSpark-OPD rollout (IP-1) — S2: real block-parallel sampling.

`DSparkRollout.generate_sequences` runs the DSpark block-parallel rollout (block_rollout.py)
over a batch of cached samples and returns the anchor×block structured output
(tokens / logp_draft / anchor_positions / block_keep_mask / eval_mask) per
docs/opd/tensor-contract.md §S2.

Rollout needs the draft model + the sample's target_hidden_states/loss_mask. In verl the
rollout object holds a model handle (set by the actor worker); the extra cached tensors
are threaded through the prompts DataProto (DSparkTrainer._get_gen_batch carries them).

BaseRollout abstract methods (resume/update_weights/release) are async no-ops: we don't
use vLLM weight sync (the actor's FSDP module is used directly for generation).
"""
from __future__ import annotations

import torch

from verl import DataProto
from verl.workers.rollout.base import BaseRollout

from recipe.dspark_opd.block_rollout import dspark_block_rollout


class DSparkRollout(BaseRollout):
    def __init__(self, config, model_config, device_mesh):
        super().__init__(config=config, model_config=model_config, device_mesh=device_mesh)
        # the draft module is attached by the actor worker after FSDP build (see worker.py)
        self.module = None
        self.temperature = float(getattr(config, "temperature", 1.0))
        # DSpark rollout knobs live in model.override_config (RolloutConfig rejects unknown keys).
        # rollout count n is verl's rollout.n (batch dim, applied by gen_batch.repeat), NOT here.
        oc = dict(getattr(model_config, "override_config", {}) or {})
        self.num_anchors = int(oc.get("dspark_num_anchors", 32))
        self.log_prob_top_k = int(oc.get("log_prob_top_k", 0))  # K: top-k dense (0 = single)
        self.reward_weight_mode = str(oc.get("reward_weight_mode", "student_p"))
        print(f"[DSparkRollout:S2] num_anchors={self.num_anchors} temperature={self.temperature} "
              f"top_k={self.log_prob_top_k}")

    async def resume(self, tags):  # noqa: D401
        return None

    async def update_weights(self, weights, **kwargs):  # noqa: D401
        return None

    async def release(self):  # noqa: D401
        return None

    def set_module(self, module):
        """Attach the draft nn.Module used for generation (called by actor worker)."""
        self.module = module

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        import os as _os, time as _time
        _timing = _os.environ.get("DSPARK_TIMING", "0") == "1"
        assert self.module is not None, "DSparkRollout.module not set by actor worker"
        b = prompts.batch
        device = next(self.module.parameters()).device
        if _timing:
            torch.cuda.synchronize(); _tc = _time.perf_counter()
        # Actor built in bf16 (§_build_dspark_module); inputs cast to bf16 -> uniform dtype,
        # numerically matches the S2 smoke (all-bf16). We call submethods directly (bypassing
        # FSDP's forward hook), so a uniform-dtype module avoids mixed-dtype matmul errors.
        out = dspark_block_rollout(
            self.module,
            input_ids=b["input_ids"].to(device),
            loss_mask=b["loss_mask"].to(device),
            target_hidden_states=b["target_hidden_states"].to(device).to(torch.bfloat16),
            num_anchors=self.num_anchors,
            temperature=self.temperature,
            top_k=self.log_prob_top_k,
        )
        tensors = {
            "rollout_tokens": out["tokens"],
            "rollout_logp_draft": out["logp_draft"],
            "rollout_anchor_positions": out["anchor_positions"],
            "rollout_block_keep_mask": out["block_keep_mask"].to(torch.long),
            "rollout_eval_mask": out["eval_mask"].to(torch.long),
        }
        if out["student_top_k_ids"] is not None:   # top-k dense (log_prob_top_k > 0)
            tensors["rollout_student_top_k_ids"] = out["student_top_k_ids"]
            tensors["rollout_student_top_k_logp"] = out["student_top_k_logp"]
        if _timing:
            torch.cuda.synchronize()
            print(f"[TIMING] rollout worker compute = {_time.perf_counter()-_tc:.3f}s", flush=True)
        result = DataProto.from_single_dict(tensors)
        result.meta_info["timing"] = {}
        return result
