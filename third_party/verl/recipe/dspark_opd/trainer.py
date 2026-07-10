"""DSpark-OPD trainer: self-contained multi-step fit loop over a FUSED worker step.

verl's RayPPOTrainer.fit() is hardwired to the sequence-level CausalLM pipeline
(_get_gen_batch drops target_hidden_states; compute_log_prob/compute_distillation_reward/
update_actor — none of which DSpark implements). So we own the fit loop; never super().fit().

Each training step is ONE fused worker RPC: actor_rollout_wg.train_step(batch) runs
rollout → teacher scoring → update INSIDE the worker (DSparkActorRolloutRefWorker.train_step),
keeping intermediates in GPU memory and returning only scalar metrics. This replaced the older
3-RPC path (generate_sequences + compute_rm_score + update_dspark_opd) whose per-step Ray
object-store round-trips (~800MB: target_hidden_states ×2 + rollout/teacher outputs) were
measured (py-spy) to be 75-80% of step wall time. The stage-gate mechanism (S0-S5 dev
scaffolding) is removed — those stages are all verified; training runs the fused step directly.
"""
from __future__ import annotations

import os
import time

import numpy as np

from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer

# A/B toggle for the repeat-in-worker optimization: set DSPARK_REPEAT_ON_DRIVER=1 to restore the
# old behavior (repeat rollout.n on the driver before dispatch, inflating the parallel_put payload
# n×). Default (unset/0) repeats worker-side in train_step. Used only for perf comparison; the two
# paths are numerically identical (scripts/opd/repeat_in_worker_smoke.py).
_REPEAT_ON_DRIVER = os.environ.get("DSPARK_REPEAT_ON_DRIVER", "0") == "1"


class DSparkTrainer(RayPPOTrainer):
    """RayPPOTrainer with a self-contained OPD multi-step loop (fused train_step; never super().fit())."""

    def fit(self):
        rollout_n = int(self.config.actor_rollout_ref.rollout.get("n", 1))
        total_steps = int(self.config.trainer.get("total_training_steps", 1) or 1)
        save_freq = int(self.config.trainer.get("save_freq", -1))
        self.global_steps = 0
        print(f"[DSparkTrainer] fit: total_training_steps={total_steps} rollout.n={rollout_n} "
              f"save_freq={save_freq}")
        done = False
        for _epoch in range(self.config.trainer.total_epochs):
            if done:
                break
            for batch_dict in self.train_dataloader:
                self.global_steps += 1
                _t0 = time.perf_counter()
                batch = self._prepare_batch(batch_dict, rollout_n)
                # ONE fused RPC: rollout -> teacher -> update, all in the worker.
                # (rollout.n repeat happens INSIDE train_step, worker-side — see _prepare_batch.)
                loss_out = self.actor_rollout_wg.train_step(batch)
                _step_s = time.perf_counter() - _t0
                lg = loss_out.batch
                metrics = {k: float(lg[k].reshape(-1)[0]) for k in (
                    "actor/loss", "actor/pg_loss", "actor/confidence_loss",
                    "actor/grad_norm", "actor/ppo_kl", "actor/n_micro") if k in lg}
                _mode = "driver-repeat" if _REPEAT_ON_DRIVER else "worker-repeat"
                print(f"[DSparkTrainer] step {self.global_steps}/{total_steps} "
                      f"wall={_step_s:.2f}s [{_mode}] "
                      + " ".join(f"{k}={v:.4f}" for k, v in metrics.items()), flush=True)
                if save_freq > 0 and self.global_steps % save_freq == 0:
                    self._save_checkpoint()
                if self.global_steps >= total_steps:
                    done = True
                    break
        if save_freq > 0 and self.global_steps % save_freq != 0:
            self._save_checkpoint()   # final save
        print(f"[DSparkTrainer] fit done at step {self.global_steps}.")

    def _prepare_batch(self, batch_dict, repeat_n=1):
        """dict -> DataProto (+ uid). NO rollout.n repeat here — deferred to the worker.

        Stock verl repeats by rollout.n on the driver (ray_trainer.py:1042) BEFORE dispatch. For
        DSpark that is pathological: our per-prompt side-tensor target_hidden_states is huge
        ([B,T,12800] bf16 ~ per-prompt MBs), and the n copies of it are BIT-IDENTICAL (same prompt,
        the n rollouts only differ by which anchors they sample). Repeating on the driver inflates
        the dispatch payload n×; parallel_put then serializes it single-core (GIL-bound) — measured
        (py-spy) as ~80% of step wall time (the long GPU-idle windows). So we dispatch only the B
        UNIQUE prompts and let train_step do the repeat_interleave(n) worker-side, on GPU, AFTER the
        cheap dispatch. This is numerically identical to driver-repeat because verl dispatch chunks
        with .chunk(dp_size) contiguously and interleave groups of size n tile exactly within each
        rank's chunk (requires train_batch_size % dp_world == 0). Equivalence is asserted by
        scripts/opd/repeat_in_worker_smoke.py. Each repeated copy still independently samples its
        own anchors/blocks inside train_step's rollout phase (R-in-batch, tensor-contract §S2).
        """
        batch: DataProto = DataProto.from_single_dict(batch_dict)
        if _REPEAT_ON_DRIVER and repeat_n and repeat_n > 1:
            # perf-comparison path only (old verl behavior); worker must NOT re-repeat.
            batch = batch.repeat(repeat_times=repeat_n, interleave=True)
        batch.non_tensor_batch["uid"] = np.array(
            [f"s{self.global_steps}-{i}" for i in range(len(batch.batch))], dtype=object)
        return batch
