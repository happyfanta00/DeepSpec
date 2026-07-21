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


def _fmt_eta(seconds: float) -> str:
    """Human-readable duration, e.g. 45s / 12m34s / 3h07m."""
    s = int(max(0.0, seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{sec:02d}s"
    return f"{sec}s"


class DSparkTrainer(RayPPOTrainer):
    """RayPPOTrainer with a self-contained OPD multi-step loop (fused train_step; never super().fit())."""

    def fit(self):
        rollout_n = int(self.config.actor_rollout_ref.rollout.get("n", 1))
        # Standard verl semantics: epochs are the primary control. The parent's _create_dataloader
        # already set self.total_training_steps = len(train_dataloader) * total_epochs when config
        # trainer.total_training_steps is null (i.e. "train exactly total_epochs epochs"), or to the
        # explicit value when set (steps override; may stop mid-epoch). We reuse that computed value.
        total_epochs = int(self.config.trainer.get("total_epochs", 1))
        steps_per_epoch = len(self.train_dataloader)
        total_steps = int(getattr(self, "total_training_steps", 0)
                          or steps_per_epoch * total_epochs)
        save_freq = int(self.config.trainer.get("save_freq", -1))
        self.global_steps = 0
        print(f"[DSparkTrainer] fit: total_epochs={total_epochs} steps_per_epoch={steps_per_epoch} "
              f"-> total_training_steps={total_steps} rollout.n={rollout_n} save_freq={save_freq}")

        # Stage-M2 T3b: if the sglang-rollout path is enabled, build the tp=8 DSPARK server actor
        # (colocated on the training GPUs) and push its address to every worker. Off by default
        # (DSPARK_SGLANG_ROLLOUT unset) -> the cache-response fused path runs unchanged. Registers
        # an atexit shutdown for crash safety; also shut down explicitly at fit() end below.
        self._maybe_start_sglang_server()

        # Rolling step-time average for throughput / ETA. Skip step 1 (it eats compile + FSDP
        # lazy-init warmup and would bias the estimate) — accumulate from step 2 onward.
        _timed_sum, _timed_n = 0.0, 0
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
                    "actor/loss", "actor/reverse_kl_loss", "actor/forward_kl_loss",
                    "actor/reject_kl_loss", "actor/confidence_loss", "actor/grad_norm",
                    "actor/n_micro", "actor/draft_weights_pushed") if k in lg}
                _mode = "driver-repeat" if _REPEAT_ON_DRIVER else "worker-repeat"
                # throughput + ETA (avg over timed steps; falls back to this step's time on step 1)
                if self.global_steps > 1:
                    _timed_sum += _step_s
                    _timed_n += 1
                _avg = (_timed_sum / _timed_n) if _timed_n else _step_s
                _rate = (1.0 / _avg) if _avg > 0 else 0.0
                _eta = _avg * max(0, total_steps - self.global_steps)
                print(f"[DSparkTrainer] step {self.global_steps}/{total_steps} "
                      f"wall={_step_s:.2f}s avg={_avg:.2f}s ({_rate:.3f} steps/s) "
                      f"ETA {_fmt_eta(_eta)} [{_mode}] "
                      + " ".join(f"{k}={v:.4f}" for k, v in metrics.items()), flush=True)
                if save_freq > 0 and self.global_steps % save_freq == 0:
                    self._save_checkpoint()
                if self.global_steps >= total_steps:
                    done = True
                    break
        if save_freq > 0 and self.global_steps % save_freq != 0:
            self._save_checkpoint()   # final save
        self._shutdown_sglang_server()
        print(f"[DSparkTrainer] fit done at step {self.global_steps} "
              f"(avg {_avg:.2f}s/step, ~{_fmt_eta(_avg * self.global_steps)} total).")

    def _maybe_start_sglang_server(self):
        """T3b: build the tp=8 DSPARK sglang server actor + push its address to all workers.

        No-op unless DSPARK_SGLANG_ROLLOUT=1. Colocates the server on the training workers' GPUs
        (build_dspark_sglang_server: NOSET + joined CUDA_VISIBLE_DEVICES + NodeAffinity), then
        broadcasts (host, port) to every worker's set_sglang_server (ONE_TO_ALL) so they build
        launch_server=False HTTP clients. mem_fraction defaults small (0.15) for the resident
        design (§5.6b); the engine stays resident, no per-step release/resume.
        """
        self._sglang_server = None
        if os.environ.get("DSPARK_SGLANG_ROLLOUT", "0") != "1":
            return
        from recipe.dspark_opd.sglang_server import build_dspark_sglang_server

        arc = self.config.actor_rollout_ref
        oc = dict(arc.model.get("override_config", {}) or {})
        target_path = oc.get("dspark_teacher_path") or oc.get("dspark_tokenizer_path")
        draft_path = arc.model.path
        tp = int(self.config.trainer.n_gpus_per_node) * int(self.config.trainer.get("nnodes", 1))
        # §5.6c KV-offload: a LARGE KV pool (fast rollout) that is released to training each step.
        # Default larger (0.6) when offload is on, else the resident small-pool default (0.15).
        _kv_offload = os.environ.get("DSPARK_SGLANG_KV_OFFLOAD", "0") == "1"
        _default_mf = "0.6" if _kv_offload else "0.15"
        mem_fraction = float(os.environ.get("DSPARK_SGLANG_MEM_FRACTION", _default_mf))
        print(f"[DSparkTrainer] T3b: starting DSPARK sglang server (tp={tp}, "
              f"mem_fraction={mem_fraction}, kv_offload={_kv_offload}, "
              f"target={target_path}, draft={draft_path}) ...")
        self._sglang_server, (host, port) = build_dspark_sglang_server(
            worker_group=self.actor_rollout_wg,
            target_path=target_path, draft_path=draft_path,
            tp_size=tp, mem_fraction_static=mem_fraction,
        )
        # crash-safety: kill the server subprocess if the driver dies unexpectedly.
        import atexit
        atexit.register(self._shutdown_sglang_server)
        # push address + TP to all workers -> they build launch_server=False HTTP clients.
        # tp_size matters for T6: the client serializes one weight copy per TP rank (§set_sglang_server).
        self.actor_rollout_wg.set_sglang_server(host, int(port), tp)
        print(f"[DSparkTrainer] T3b: sglang server ready at http://{host}:{port} (tp={tp}); workers wired.")

    def _shutdown_sglang_server(self):
        """Shut down the T3b sglang server actor (idempotent; safe to call from atexit + fit end)."""
        srv = getattr(self, "_sglang_server", None)
        if srv is None:
            return
        self._sglang_server = None
        import ray
        try:
            ray.get(srv.shutdown.remote())
        except Exception as e:  # noqa: BLE001 — best-effort
            print(f"[DSparkTrainer] sglang server shutdown error (ignored): {e!r}")
        try:
            ray.kill(srv)
        except Exception:  # noqa: BLE001
            pass

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
