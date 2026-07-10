"""DSpark-OPD worker subclasses (IP-2 actor/model, IP-4 teacher).

S2 (current): DSparkActorRolloutRefWorker overrides _build_model_optimizer to build the
non-standard Qwen3DSparkModel actor (verl's AutoModel path would silently load the WRONG
class — config model_type='qwen3' maps AutoModel -> standard Qwen3Model, not our draft),
freezes embed_tokens/lm_head, FSDP-wraps it (fsdp_size=1 -> no param sharding;
use_orig_params=True required by the frozen params; MixedPrecision bf16), and attaches
the actor module to the DSparkRollout for block-parallel sampling.

FSDP config rationale: see docs/DSpark-OPD.md §2.6.5.

S3: DSparkRewardModelWorker overrides _build_model (target) + _forward_micro_batch
    (block-diagonal scoring).
S4: DataParallelPPOActor training-forward + loss.
"""
from __future__ import annotations

import os

import torch
from omegaconf import OmegaConf
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy

from verl import DataProto
from verl.single_controller.base.decorator import Dispatch, make_nd_compute_dataproto_dispatch_fn, register
from verl.workers.fsdp_workers import ActorRolloutRefWorker, RewardModelWorker
from verl.utils.device import get_device_id
from verl.utils.fsdp_utils import get_fsdp_wrap_policy, init_fn


class DSparkActorRolloutRefWorker(ActorRolloutRefWorker):
    """Actor/rollout worker for the DSpark draft model (Qwen3DSparkModel)."""

    def _build_dspark_module(self, model_path):
        """Construct the DSpark draft model + freeze embed/lm_head. Returns nn.Module (bf16).

        Built in bf16 to match DeepSpec training (config precision='bf16' + FSDP
        MixedPrecision(bf16)) and the S2 smoke (all-bf16, which passed the eval-consistency
        checks). We call submethods directly (bypassing FSDP's forward-hook autocast), so a
        uniform bf16 module avoids fp32-weight/bf16-input matmul mismatches.
        """
        from deepspec.modeling.dspark.qwen3 import Qwen3DSparkModel

        # flex_attention: block-parallel path uses BlockMask (training/rollout forward).
        module = Qwen3DSparkModel.from_pretrained(
            model_path, dtype=torch.bfloat16, attn_implementation="flex_attention",
        )
        # freeze embed_tokens / lm_head (copied from target; only train backbone/fc/heads)
        if hasattr(module, "set_embedding_head_trainable"):
            module.set_embedding_head_trainable(False)
        return module

    def _build_model_optimizer(self, model_path, fsdp_config, optim_config, override_model_config,
                               use_remove_padding=False, use_fused_kernels=False,
                               enable_gradient_checkpointing=False, trust_remote_code=False,
                               use_liger=False, role="actor", enable_activation_offload=False):
        """Override: build Qwen3DSparkModel actor (not AutoModel), reuse verl FSDP wrap.

        Returns (actor_module_fsdp, actor_optimizer, actor_lr_scheduler, actor_model_config)
        matching verl's contract (fsdp_workers.py:579).
        """
        from transformers import AutoConfig
        from verl.utils.torch_dtypes import PrecisionType
        from verl.workers.config.optimizer import build_optimizer

        assert role in ("actor", "ref")
        # tokenizer/processor (verl expects these set on the worker). The draft checkpoint
        # has NO tokenizer files -> load tokenizer from the TARGET (draft shares vocab, §2.2)
        # via dspark_tokenizer_path; fall back to model_path if unset.
        from verl.utils import hf_processor, hf_tokenizer
        tok_path = self.config.model.get("override_config", {}).get("dspark_tokenizer_path") or model_path
        self.tokenizer = hf_tokenizer(tok_path, trust_remote_code=trust_remote_code)
        self.processor = hf_processor(tok_path, trust_remote_code=trust_remote_code)

        actor_model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
        # verl's generate_sequences wrapper reads self.generation_config (None -> falls back
        # to tokenizer eos/pad). We don't use HF .generate(), so None is fine.
        self.generation_config = None

        # 1) build our draft model (bf16, matches DeepSpec training + smoke), freeze heads
        actor_module = self._build_dspark_module(model_path)
        if enable_gradient_checkpointing and hasattr(actor_module, "gradient_checkpointing_enable"):
            actor_module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        # use_orig_params=True is REQUIRED: frozen embed/lm_head (mixed requires_grad) is
        # incompatible with FSDP flat-param when False. See §2.6.5.
        self.use_orig_params = True

        # 2) mixed precision (bf16), reuse verl's config plumbing
        mp_cfg = fsdp_config.get("mixed_precision", None)
        if mp_cfg is not None:
            param_dtype = PrecisionType.to_dtype(mp_cfg.get("param_dtype", "bf16"))
            reduce_dtype = PrecisionType.to_dtype(mp_cfg.get("reduce_dtype", "fp32"))
            buffer_dtype = PrecisionType.to_dtype(mp_cfg.get("buffer_dtype", "fp32"))
        else:
            param_dtype, reduce_dtype, buffer_dtype = torch.bfloat16, torch.float32, torch.bfloat16
        mixed_precision = MixedPrecision(param_dtype=param_dtype, reduce_dtype=reduce_dtype,
                                         buffer_dtype=buffer_dtype)

        auto_wrap_policy = get_fsdp_wrap_policy(module=actor_module,
                                                config=fsdp_config.get("wrap_policy", None), is_lora=False)

        # ★ NO_SHARD: "data-parallel, model-not-sharded" — every rank holds FULL actor params
        # (so rollout's direct submethod calls see complete params) AND gradients all-reduce
        # across ranks. We MUST NOT use verl's get_sharding_strategy here: with fsdp_size=1 it
        # returns a degenerate HYBRID_SHARD (ddp=world, fsdp=1) whose replicate-dim gradient
        # all-reduce is NOT wired in this FSDP1 version -> multi-GPU replicas silently diverge.
        # Verified (2-rank, distinct data, params-after-step): NO_SHARD Δ=0 (synced),
        # HYBRID(fsdp_size=1) Δ≈2 (NOT synced). See docs/DSpark-OPD.md §2.6.5. device_mesh=None
        # -> DP over the default global group (NO_SHARD needs no 2D mesh). NOTE: NO_SHARD is
        # deprecated in PyTorch (DDP is the suggested successor); kept for now to preserve the
        # full FSDP scaffolding (checkpoint_manager, clip_grad_norm_, summon_full_params).
        sharding_strategy = ShardingStrategy.NO_SHARD

        # 3) FSDP wrap (strategy 'fsdp' == FSDP1), mirrors verl fsdp_workers.py:496-509
        actor_module_fsdp = FSDP(
            actor_module,
            cpu_offload=None,
            param_init_fn=init_fn,
            auto_wrap_policy=auto_wrap_policy,
            device_id=get_device_id(),
            sharding_strategy=sharding_strategy,
            mixed_precision=mixed_precision,
            sync_module_states=True,
            device_mesh=None,             # NO_SHARD -> DP over global group; no 2D shard mesh
            use_orig_params=self.use_orig_params,
            forward_prefetch=fsdp_config.get("forward_prefetch", False),
        )

        # 4) optimizer + scheduler (only for actor role)
        actor_optimizer = actor_lr_scheduler = None
        if role == "actor" and optim_config is not None:
            from verl.utils.torch_functional import get_constant_schedule_with_warmup
            actor_optimizer = build_optimizer(actor_module_fsdp.parameters(), optim_config)
            total_steps = optim_config.get("total_training_steps", 0)
            num_warmup = int(optim_config.get("lr_warmup_steps", -1))
            if num_warmup < 0:
                num_warmup = int(optim_config.get("lr_warmup_steps_ratio", 0.0) * total_steps)
            actor_lr_scheduler = get_constant_schedule_with_warmup(
                optimizer=actor_optimizer, num_warmup_steps=num_warmup)

        return actor_module_fsdp, actor_optimizer, actor_lr_scheduler, actor_model_config

    def _build_rollout(self, trust_remote_code=False):
        """Reuse verl's rollout build, then attach the actor module to DSparkRollout."""
        super()._build_rollout(trust_remote_code=trust_remote_code)
        # attach the (unwrapped) actor module so DSparkRollout can call its submethods.
        module = getattr(self, "actor_module", None) or getattr(self, "actor_module_fsdp", None)
        if hasattr(self.rollout, "set_module"):
            self.rollout.set_module(module)

    def _build_teacher(self, teacher_path):
        """Build the TARGET (teacher) FSDP module INSIDE the actor worker (fused design).

        Since the fused train_step runs rollout→teacher→update in one worker RPC (no driver
        round-trips), the teacher must be co-resident with the actor. We build it here rather
        than as a separate RewardModel role. FULL_SHARD (ZeRO-3, 1D mesh) + bf16 + sdpa (no
        flash-attn); NO CPUOffload (ample VRAM, avoids per-step CPU<->GPU transfer). Only used
        for no-grad scoring, so no optimizer / use_orig_params=False is fine.
        """
        from transformers import AutoModelForCausalLM
        from verl.workers.fsdp_workers import get_sharding_strategy

        target = AutoModelForCausalLM.from_pretrained(
            teacher_path, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
            trust_remote_code=False)
        target = target.to(get_device_id())
        target.eval()
        for p in target.parameters():
            p.requires_grad_(False)
        mp = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32,
                            buffer_dtype=torch.bfloat16)
        sharding_strategy = get_sharding_strategy(self.device_mesh)   # 1D mesh -> FULL_SHARD
        self.teacher_module = FSDP(
            target,
            cpu_offload=None,                 # keep resident (VRAM ample; avoid per-step transfer)
            auto_wrap_policy=get_fsdp_wrap_policy(module=target, config=None, is_lora=False),
            device_id=get_device_id(),
            sharding_strategy=sharding_strategy,
            mixed_precision=mp,
            sync_module_states=True,
            device_mesh=self.device_mesh,
            use_orig_params=False,
        )
        self.teacher_module.eval()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        """Build actor (+rollout+optimizer via super) THEN the co-resident teacher (fused design).

        Data-path setup by DSPARK_HIDDEN_MODE:
          - "recompute" (default, #5): STANDARD dispatch — the driver sends tokens; the worker
            recomputes target_hidden_states via the teacher. Needs only target_layer_ids (read from
            the draft model); NO cache handle in the worker.
          - "cache" (#4): the driver sends indices; the worker re-reads full samples from a mmap'd
            CacheDataset handle (opened here).
          - "dispatch" (#3): the driver sends full tensors incl. hidden; worker needs neither.
        """
        super().init_model()
        oc = dict(self.config.model.get("override_config", {}) or {})
        # teacher path: explicit dspark_teacher_path, else reuse tokenizer path (== target, §2.2)
        teacher_path = oc.get("dspark_teacher_path") or oc.get("dspark_tokenizer_path")
        assert teacher_path, "override_config.dspark_teacher_path (or dspark_tokenizer_path) required"
        self._build_teacher(teacher_path)

        from recipe.dspark_opd.dataset import _HIDDEN_MODE
        self._hidden_mode = _HIDDEN_MODE
        self._cache = None
        self._target_layer_ids = None
        if _HIDDEN_MODE == "recompute":
            # only need which teacher layers to capture — read from the (unwrapped) draft model.
            draft = getattr(self, "actor_module", None) or self.actor.actor_module
            self._target_layer_ids = [int(x) for x in draft.target_layer_ids]
        elif _HIDDEN_MODE == "cache":
            from deepspec.data.target_cache_dataset import CacheDataset
            cache_path = oc.get("dspark_target_cache_path")
            assert cache_path, "override_config.dspark_target_cache_path required for HIDDEN_MODE=cache"
            self._cache = CacheDataset(cache_dir=str(cache_path))
            self._target_layer_ids = list(self._cache.target_layer_ids)  # [1,9,17,25,33]

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    def update_dspark_opd(self, data: DataProto):
        """S4: one full OPD training step (grad forward -> reward -> PG+confidence loss ->
        backward -> optimizer step), on the DSpark draft. NOT verl's sequence-level
        compute_log_prob/compute_distillation_reward/update_policy trio (all CausalLM-coupled).

        Like rollout/teacher, we own the whole step. Reuses the block-parallel forward
        (loss_bridge, validated by s4_smoke) + verl's token_reward_direct advantage + 3D
        dual-clip compute_policy_loss_vanilla, plus DSpark's own confidence BCE.

        Input DataProto.batch (from DSparkTrainer loss phase; all on device):
          input_ids [B,T], loss_mask [B,T], target_hidden_states [B,T,L*H],
          rollout_tokens [B,A,blk], rollout_anchor_positions [B,A], rollout_block_keep_mask [B,A],
          rollout_student_top_k_ids/logp [B,A,blk,K], rollout_eval_mask [B,A,blk],
          logp_target_on_topk [B,A,blk,K] (from teacher phase).
        Returns DataProto with scalar metrics: actor/pg_loss, actor/confidence_loss,
          actor/loss, actor/grad_norm, actor/ppo_kl.
        """
        from verl.trainer.ppo.core_algos import (
            get_adv_estimator_fn, compute_policy_loss_vanilla)
        from recipe.dspark_opd.loss_bridge import (
            logp_on_topk_ids, build_opd_reward,
            confidence_accept_rate_topk, confidence_bce, block_decay_weight_mask,
            flatten_blocks_to_sequence)

        data = data.to(get_device_id())
        # The FSDP-wrapped draft. We call module(...) (FSDP.forward) — NOT submethods — so the
        # FSDP pre-forward hook registers the cross-rank gradient-reduction hooks; otherwise the
        # root-unit trainable heads' grads would never all-reduce on multi-GPU (see §S4).
        module = self.actor.actor_module
        module.train()

        # knobs (override_config for DSpark-specific; rollout for temperature/weight_mode)
        oc = dict(self.config.model.get("override_config", {}) or {})
        temperature = float(self.config.rollout.get("temperature", 1.0))
        weight_mode = str(self.config.rollout.get("reward_weight_mode",
                                                  oc.get("reward_weight_mode", "student_p")))
        gamma = float(oc.get("loss_decay_gamma", 4.0))            # dspark_qwen3_4b.py
        conf_alpha = float(oc.get("confidence_head_alpha", 1.0))
        cfg_pl = OmegaConf.create({
            "clip_ratio": float(self.config.actor.get("clip_ratio", 0.2)),
            "clip_ratio_low": float(self.config.actor.get("clip_ratio_low", 0.2)),
            "clip_ratio_high": float(self.config.actor.get("clip_ratio_high", 0.2)),
            "clip_ratio_c": float(self.config.actor.get("clip_ratio_c", 3.0)),
        })

        # ---- micro-batch split + gradient accumulation (verl-style, dp_actor.py:801-819) ----
        # The worker receives ONE mini (= one optimizer step; trainer already repeated by
        # rollout.n and dispatch chunked by dp, so B_recv == normalized ppo_mini_batch_size).
        # We split it into micros of ppo_micro_batch_size_per_gpu and accumulate gradients.
        # Each micro's loss is scaled by 1/num_micro (verl's 1/gradient_accumulation) — this is
        # verl's known token-mean-vs-1/M approximation (exact only when micros have equal valid
        # tokens; balance_batch mitigates cross-rank, see §S6). Faithful to verl on purpose.
        micro_bsz = int(self.config.actor.get("ppo_micro_batch_size_per_gpu", 0) or 0)
        micro_batches = data.split(micro_bsz) if micro_bsz > 0 else [data]
        n_micro = len(micro_batches)
        scale = 1.0 / n_micro

        import os as _os, time as _time
        _timing = _os.environ.get("DSPARK_TIMING", "0") == "1"
        if _timing:
            torch.cuda.synchronize(); _tc = _time.perf_counter()

        self.actor.actor_optimizer.zero_grad(set_to_none=True)
        agg = {"actor/pg_loss": 0.0, "actor/confidence_loss": 0.0,
               "actor/loss": 0.0, "actor/ppo_kl": 0.0}
        for mb in micro_batches:
            b = mb.batch
            blk = int(b["rollout_tokens"].shape[2])
            anchor_positions = b["rollout_anchor_positions"]
            block_keep_mask = b["rollout_block_keep_mask"].bool()

            # 1) GRAD training forward THROUGH FSDP.forward (module(...), OPD branch): reproduce
            # rollout's corrected_logits (== draft_logits) + confidence_pred with grad.
            # block_prev_tokens = [anchor_tok, ỹ_1..ỹ_{blk-1}] teacher-forces the markov head on
            # the SAMPLED draft tokens. Going through module(...) (not submethods) is what arms
            # FSDP's post-backward cross-rank gradient all-reduce (§S4).
            anchor_tok = torch.gather(b["input_ids"], 1, anchor_positions)     # [B, A]
            block_prev_tokens = torch.cat(
                [anchor_tok.unsqueeze(-1), b["rollout_tokens"][:, :, : blk - 1]], dim=-1)
            out = module(
                input_ids=b["input_ids"],
                target_hidden_states=b["target_hidden_states"].to(torch.bfloat16),
                loss_mask=b["loss_mask"],
                anchor_positions=anchor_positions,
                block_keep_mask=block_keep_mask,
                block_prev_tokens=block_prev_tokens,
            )
            # draft_logits is already markov-corrected (= rollout's corrected_logits).
            # grad student logp on the FIXED S2 top-k ids.
            S_logp_grad = logp_on_topk_ids(
                out.draft_logits, b["rollout_student_top_k_ids"], temperature=temperature)

            # 2) reward (no-grad) + token_reward_direct advantage, flattened to [B, A*blk, K].
            # response_mask = eval_mask × exp(-pos/γ) decay (§2.5.4: PG reuses SFT
            # loss_decay_gamma); advantage=rm×mask and token-mean both weight by decay,
            # matching SFT's loss_weight_mask (loss.py:_build_loss_weight_mask).
            S_logp_old = b["rollout_student_top_k_logp"]          # S2 no-grad (old_log_prob)
            T_on_S = b["logp_target_on_topk"]                    # teacher phase
            rm = build_opd_reward(T_on_S, S_logp_old, weight_mode=weight_mode)
            eval_mask = b["rollout_eval_mask"].bool()
            decay_mask = block_decay_weight_mask(eval_mask, blk, gamma)
            rm_flat = flatten_blocks_to_sequence(rm)
            mask_flat = flatten_blocks_to_sequence(decay_mask.unsqueeze(-1)).squeeze(-1)
            adv, _ = get_adv_estimator_fn("token_reward_direct")(
                token_level_rewards=rm_flat, response_mask=mask_flat)

            # 3) 3D dual-clip PG loss (on-policy: old == log_prob.detach() -> ratio≈1)
            logp_flat = flatten_blocks_to_sequence(S_logp_grad)
            pg_loss, pg_metrics = compute_policy_loss_vanilla(
                old_log_prob=logp_flat.detach(), log_prob=logp_flat, advantages=adv,
                response_mask=mask_flat, loss_agg_mode="token-mean", config=cfg_pl)

            # 4) confidence BCE (top-k accept-rate target, detached) + total loss
            accept = confidence_accept_rate_topk(S_logp_old, T_on_S)
            conf_loss = confidence_bce(out.confidence_pred, accept, decay_mask)
            loss = pg_loss + conf_alpha * conf_loss

            # 5) scale by 1/num_micro and accumulate gradients
            (loss * scale).backward()
            agg["actor/pg_loss"] += float(pg_loss.detach()) * scale
            agg["actor/confidence_loss"] += float(
                conf_loss.detach() if hasattr(conf_loss, "detach") else conf_loss) * scale
            agg["actor/loss"] += float(loss.detach()) * scale
            agg["actor/ppo_kl"] += float(pg_metrics.get("actor/ppo_kl", 0.0)) * scale

        # 6) one optimizer step per mini (reuse actor's grad-clip/step)
        grad_norm = self.actor._optimizer_step()
        if self.actor_lr_scheduler is not None:
            self.actor_lr_scheduler.step()
        if _timing:
            torch.cuda.synchronize()
            print(f"[TIMING] update worker compute ({n_micro} micro) = "
                  f"{_time.perf_counter()-_tc:.3f}s", flush=True)

        # scalar metrics as shape-[1] tensors (DataProto needs a batch dim; actor-mesh dispatch
        # concatenates across dp workers, here world_size=1 -> returned as-is).
        def _s(v):
            return torch.tensor([float(v.item() if hasattr(v, "item") else v)])
        metrics = {k: _s(v) for k, v in agg.items()}
        metrics["actor/grad_norm"] = _s(grad_norm)
        metrics["actor/n_micro"] = _s(n_micro)
        return DataProto.from_single_dict(metrics)

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    def train_step(self, data: DataProto):
        """★ FUSED OPD step: rollout → teacher scoring → update, all in ONE worker RPC.

        Replaces the old 3-RPC path (generate_sequences + compute_rm_score + update_dspark_opd),
        which round-tripped ~800MB (target_hidden_states ×2 + rollout/teacher outputs) through
        the Ray object store per step — measured (py-spy) to be 75-80% of step wall time.
        Here the whole step runs in the worker; intermediates stay in GPU memory; only scalar
        metrics return to the driver. The three phases reuse the SAME validated pure functions
        (dspark_block_rollout / score_blocks_flat / loss_bridge), so numerics match the separate
        path (verified by fused_step_smoke).

        Input DataProto.batch: input_ids [B,T], loss_mask [B,T], target_hidden_states [B,T,L*H].
        Returns scalar metrics: actor/{loss,pg_loss,confidence_loss,grad_norm,ppo_kl,n_micro}.
        """
        import os as _os
        from verl.trainer.ppo.core_algos import get_adv_estimator_fn, compute_policy_loss_vanilla
        from recipe.dspark_opd.block_rollout import dspark_block_rollout
        from recipe.dspark_opd.teacher_scoring import score_blocks_flat, recompute_target_hidden_states
        from recipe.dspark_opd.dataset import adapt_cache_record, dspark_collate_fn
        from recipe.dspark_opd.loss_bridge import (
            logp_on_topk_ids, build_opd_reward, confidence_accept_rate_topk,
            confidence_bce, block_decay_weight_mask, flatten_blocks_to_sequence)

        rollout_n = int(self.config.rollout.get("n", 1))
        _dev = get_device_id()
        _mode = getattr(self, "_hidden_mode", "dispatch")
        # rollout.n repeat is done worker-side unless DSPARK_REPEAT_ON_DRIVER=1 (A/B perf toggle).
        _repeat = rollout_n > 1 and _os.environ.get("DSPARK_REPEAT_ON_DRIVER", "0") != "1"

        def _rep(t):
            return t.repeat_interleave(rollout_n, dim=0) if _repeat else t

        if _mode == "cache" and "sample_index" in data.batch.keys():
            # ★ Opt #4: the driver dispatched ONLY this rank's DP-shard of sample INDICES (~512B).
            # This is equivalent to the driver reading the full batch and chunking it to this rank
            # (cache is deterministic; dispatch chunks contiguously) — see worker_cache_read_smoke.py.
            # Read each UNIQUE full sample (incl target_hidden_states) from the mmap'd cache once,
            # then repeat_interleave(n) so the n rollout copies share one read.
            idxs = data.batch["sample_index"].reshape(-1).tolist()
            feats = [adapt_cache_record(self._cache[int(i)]) for i in idxs]
            bb = dspark_collate_fn(feats)                     # [B_u, T, ...] incl target_hidden_states
            b = {k: _rep(v).to(_dev) for k, v in bb.items()}
        elif _mode == "recompute":
            # ★ Opt #5 (STANDARD dispatch): the driver read+dispatched this rank's TOKENS
            # (input_ids/loss_mask/attention_mask, ~KB, prefetched by the dataloader's num_workers).
            # We RECOMPUTE target_hidden_states with the co-resident teacher (one prefill forward +
            # layer hooks) — no large hidden read/dispatch. Trades a ~0.1s GPU forward for the cache
            # cold-read + hidden serialization, and matches inference (live-teacher hidden, bf16).
            # Recompute on the UNIQUE sequences (before repeat) so the n rollout copies share one
            # forward; then repeat_interleave(n) everything.
            data = data.to(_dev)
            bd = data.batch
            ii_u = bd["input_ids"]
            am_u = bd.get("attention_mask")
            if am_u is None:                                  # derive from loss_mask/nonzero if absent
                am_u = (ii_u != 0).long()
            th_u = recompute_target_hidden_states(            # [B_u, T, L*H] bf16
                self.teacher_module, input_ids=ii_u, attention_mask=am_u,
                target_layer_ids=self._target_layer_ids)
            b = {
                "input_ids": _rep(ii_u),
                "loss_mask": _rep(bd["loss_mask"]),
                "attention_mask": _rep(am_u),
                "target_hidden_states": _rep(th_u),
            }
        else:
            # Opt #3 (full-tensor dispatch): the driver dispatched the real tensors incl. hidden.
            # rollout.n repeat is done worker-side (NOT on the driver): the driver dispatches only
            # the B UNIQUE prompts; repeating on GPU after the cheap dispatch avoids inflating the
            # parallel_put payload n× with bit-identical copies of the huge target_hidden_states.
            # Numerically identical to driver-repeat (dispatch chunks contiguously; interleave groups
            # of size n tile exactly within each rank's chunk). Verified by repeat_in_worker_smoke.py.
            data = data.to(_dev)
            if _repeat:
                data = data.repeat(repeat_times=rollout_n, interleave=True)
            b = data.batch
        module = self.actor.actor_module          # FSDP root (update goes through module(...))
        unwrapped = getattr(self, "actor_module", None) or module   # for no_grad rollout submethods
        module.train()
        _debug = _os.environ.get("DSPARK_DEBUG", "0") == "1"

        # ★ Prime the ROOT FSDP's lazy-init BEFORE Phase-1 rollout. Rollout calls the per-layer
        # nested-FSDP modules directly (unwrapped._forward_backbone -> for layer in layers:
        # layer(...)); doing that before the root's first forward makes each nested layer grab
        # _is_root=True, and then Phase-3's root module(...) trips
        # "Non-root FSDP instance's _is_root should not have been set yet" (FSDP1 _lazy_init).
        # verl's inherited generate_sequences avoided this because rollout_mode() calls
        # actor_module_fsdp.state_dict(), which primes the root. The fused train_step bypasses
        # that, so we prime the root explicitly (idempotent; no-op after the first step).
        from torch.distributed.fsdp import _runtime_utils as _fsdp_rt
        _fsdp_rt._lazy_init(module, module)

        # knobs
        oc = dict(self.config.model.get("override_config", {}) or {})
        temperature = float(self.config.rollout.get("temperature", 1.0))
        weight_mode = str(self.config.rollout.get("reward_weight_mode",
                                                  oc.get("reward_weight_mode", "student_p")))
        num_anchors = int(oc.get("dspark_num_anchors", 32))
        top_k = int(oc.get("log_prob_top_k", 0))
        gamma = float(oc.get("loss_decay_gamma", 4.0))
        conf_alpha = float(oc.get("confidence_head_alpha", 1.0))
        cfg_pl = OmegaConf.create({
            "clip_ratio": float(self.config.actor.get("clip_ratio", 0.2)),
            "clip_ratio_low": float(self.config.actor.get("clip_ratio_low", 0.2)),
            "clip_ratio_high": float(self.config.actor.get("clip_ratio_high", 0.2)),
            "clip_ratio_c": float(self.config.actor.get("clip_ratio_c", 3.0)),
        })

        # ===== Phase 1: ROLLOUT (no_grad, unwrapped submethods; big transients freed on return) =====
        roll = dspark_block_rollout(
            unwrapped,
            input_ids=b["input_ids"],
            loss_mask=b["loss_mask"],
            target_hidden_states=b["target_hidden_states"].to(torch.bfloat16),
            num_anchors=num_anchors, temperature=temperature, top_k=top_k)
        anchors = roll["anchor_positions"]                    # [B,A]
        keep = roll["block_keep_mask"].bool()                 # [B,A]
        tokens = roll["tokens"]                               # [B,A,blk]
        tki = roll["student_top_k_ids"]                       # [B,A,blk,K]
        S_logp_old = roll["student_top_k_logp"]               # [B,A,blk,K] no-grad old_log_prob
        eval_mask = roll["eval_mask"].bool()                  # [B,A,blk]

        # ===== Phase 2: TEACHER scoring (no_grad, co-resident FULL_SHARD target) =====
        T_on_S = score_blocks_flat(
            self.teacher_module, input_ids=b["input_ids"], tokens=tokens,
            anchor_positions=anchors, block_keep_mask=keep, student_top_k_ids=tki)  # [B,A,blk,K]
        from verl.utils.fsdp_utils import fsdp_version
        if self.world_size > 1 and fsdp_version(self.teacher_module) == 1:
            self.teacher_module._handle.reshard(True)   # drop all-gathered teacher params (FULL_SHARD)

        if _debug:
            self._debug_teacher_signal(tokens, anchors, keep, tki, S_logp_old, T_on_S)

        # ===== Phase 3: UPDATE (grad forward via module(...), micro-batch + grad accum) =====
        blk = int(tokens.shape[2])
        # split the B rollout+teacher tensors into micro-batches (same as update_dspark_opd)
        idx = torch.arange(b["input_ids"].shape[0], device=b["input_ids"].device)
        micro_bsz = int(self.config.actor.get("ppo_micro_batch_size_per_gpu", 0) or 0)
        chunks = idx.split(micro_bsz) if micro_bsz > 0 else [idx]
        n_micro = len(chunks)
        scale = 1.0 / n_micro
        adv_fn = get_adv_estimator_fn("token_reward_direct")

        self.actor.actor_optimizer.zero_grad(set_to_none=True)
        agg = {"actor/pg_loss": 0.0, "actor/confidence_loss": 0.0, "actor/loss": 0.0,
               "actor/ppo_kl": 0.0}
        for sl in chunks:
            ii = b["input_ids"][sl]
            anchor_positions = anchors[sl]
            anchor_tok = torch.gather(ii, 1, anchor_positions)
            block_prev = torch.cat([anchor_tok.unsqueeze(-1), tokens[sl][:, :, : blk - 1]], dim=-1)
            out = module(
                input_ids=ii,
                target_hidden_states=b["target_hidden_states"][sl].to(torch.bfloat16),
                loss_mask=b["loss_mask"][sl],
                anchor_positions=anchor_positions,
                block_keep_mask=keep[sl],
                block_prev_tokens=block_prev)
            S_grad = logp_on_topk_ids(out.draft_logits, tki[sl], temperature=temperature)
            rm = build_opd_reward(T_on_S[sl], S_logp_old[sl], weight_mode=weight_mode)
            dmask = block_decay_weight_mask(eval_mask[sl], blk, gamma)
            mask_flat = flatten_blocks_to_sequence(dmask.unsqueeze(-1)).squeeze(-1)
            adv, _ = adv_fn(token_level_rewards=flatten_blocks_to_sequence(rm), response_mask=mask_flat)
            lp = flatten_blocks_to_sequence(S_grad)
            pg_loss, pg_metrics = compute_policy_loss_vanilla(
                old_log_prob=lp.detach(), log_prob=lp, advantages=adv,
                response_mask=mask_flat, loss_agg_mode="token-mean", config=cfg_pl)
            accept = confidence_accept_rate_topk(S_logp_old[sl], T_on_S[sl])
            conf_loss = confidence_bce(out.confidence_pred, accept, dmask)
            loss = pg_loss + conf_alpha * conf_loss
            (loss * scale).backward()
            agg["actor/pg_loss"] += float(pg_loss.detach()) * scale
            agg["actor/confidence_loss"] += float(
                conf_loss.detach() if hasattr(conf_loss, "detach") else conf_loss) * scale
            agg["actor/loss"] += float(loss.detach()) * scale
            agg["actor/ppo_kl"] += float(pg_metrics.get("actor/ppo_kl", 0.0)) * scale

        grad_norm = self.actor._optimizer_step()
        if self.actor_lr_scheduler is not None:
            self.actor_lr_scheduler.step()

        def _s(v):
            return torch.tensor([float(v.item() if hasattr(v, "item") else v)])
        metrics = {k: _s(v) for k, v in agg.items()}
        metrics["actor/grad_norm"] = _s(grad_norm)
        metrics["actor/n_micro"] = _s(n_micro)
        return DataProto.from_single_dict(metrics)

    def _debug_teacher_signal(self, tokens, anchors, keep, tki, S_logp_old, T_on_S):
        """Print target-vs-draft top-k signal for sample 0's first valid block (DSPARK_DEBUG)."""
        tok = self.tokenizer
        k0 = next((i for i in range(keep.shape[1]) if bool(keep[0, i])), None)
        if k0 is None:
            return
        ids = tki[0, k0, 0]
        lp_t, lp_d = T_on_S[0, k0, 0], S_logp_old[0, k0, 0]
        print(f"\n[DSPARK_DEBUG] teacher vs draft top-k (sample0 block{k0}@anchor{int(anchors[0,k0])} pos0)")
        for k in range(min(ids.shape[0], 8)):
            piece = tok.convert_ids_to_tokens([int(ids[k])])[0]
            t, d = float(lp_t[k]), float(lp_d[k])
            print(f"   {piece!r:>16}  T={t:>8.3f} D={d:>8.3f} Δ={t-d:>+7.3f}", flush=True)


class DSparkRewardModelWorker(RewardModelWorker):
    """Teacher (target) reward worker.

    S2: only need it to init without flash-attn (we don't install flash-attn; verl's
    reward _build_model hardcodes attn_implementation='flash_attention_2', fsdp_workers.py
    :1758). Force sdpa by patching AutoModelForCausalLM.from_pretrained's default during
    the parent build.
    S3: override compute_rm_score entirely (verl's is coupled to a sequence-level
    old_log_probs/responses layout) to run block-diagonal target scoring via score_blocks_flat.
    """

    def _build_model(self, config):
        import transformers
        import torch.distributed.fsdp as _fsdp

        orig_fp = transformers.AutoModelForCausalLM.from_pretrained
        orig_cpuoffload = _fsdp.CPUOffload

        def _patched_fp(*args, **kwargs):
            kwargs["attn_implementation"] = "sdpa"  # we don't install flash-attn (§2.6)
            return orig_fp(*args, **kwargs)

        # verl hardcodes cpu_offload=CPUOffload(offload_params=True) for the reward FSDP wrap
        # (fsdp_workers.py:1786) with no config knob. That offloads the 4B teacher params to CPU
        # and re-gathers them to GPU on EVERY compute_rm_score forward — a big CPU<->GPU transfer
        # that stalls the GPU (idle windows) every step. We have ample VRAM (teacher is ~1GB/rank
        # sharded across 8), so keep params resident: patch CPUOffload to force offload_params=False.
        # _build_model does a LOCAL `from torch.distributed.fsdp import CPUOffload`, so patching the
        # module attribute is picked up by that re-import.
        def _no_offload_cpuoffload(*args, **kwargs):
            kwargs["offload_params"] = False
            return orig_cpuoffload(*args, **kwargs)

        transformers.AutoModelForCausalLM.from_pretrained = staticmethod(_patched_fp)
        _fsdp.CPUOffload = _no_offload_cpuoffload
        try:
            return super()._build_model(config)
        finally:
            transformers.AutoModelForCausalLM.from_pretrained = orig_fp
            _fsdp.CPUOffload = orig_cpuoffload

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="reward"))
    def compute_rm_score(self, data: DataProto, kl_estimator="k1"):
        """S3: block-diagonal teacher scoring (route A), NOT verl's sequence-level path.

        verl's stock compute_rm_score assumes a causal, sequence-level layout
        (old_log_probs [B,resp_len], `responses`), which does not match DSpark's
        anchor×block structure. So — like the actor's _build_model_optimizer — we override
        the whole method and call score_blocks_flat (one target forward, 4D block-diagonal
        causal mask) on the rollout's per-anchor blocks + top-K candidates.

        Input DataProto.batch (from DSparkTrainer teacher phase; all on device):
          input_ids [B,T], rollout_tokens [B,A,blk], rollout_anchor_positions [B,A],
          rollout_block_keep_mask [B,A] (long 0/1), rollout_student_top_k_ids [B,A,blk,K].
        Returns DataProto with:
          logp_target_on_topk [B,A,blk,K] (float) — teacher logπ on student top-K candidates.
        """
        from recipe.dspark_opd.teacher_scoring import score_blocks_flat

        data = data.to(get_device_id())
        b = data.batch
        # FSDP-wrapped target. NOTE: unlike the actor (NO_SHARD), the reward worker inherits
        # verl's default fsdp_size=-1 -> 1D FULL_SHARD (ZeRO-3) + CPUOffload. score_blocks_flat
        # calls target(...) i.e. FSDP.forward, whose pre-forward all-gathers the sharded params
        # (so scoring is correct on multi-GPU). It reads next(target.parameters()).dtype only for
        # the additive-mask dtype (valid on a sharded flat param).
        target = self.reward_module

        import os as _os, time as _time
        _timing = _os.environ.get("DSPARK_TIMING", "0") == "1"
        if _timing:
            torch.cuda.synchronize(); _tc = _time.perf_counter()
        logp_target_on_topk = score_blocks_flat(
            target,
            input_ids=b["input_ids"],
            tokens=b["rollout_tokens"],
            anchor_positions=b["rollout_anchor_positions"],
            block_keep_mask=b["rollout_block_keep_mask"].bool(),
            student_top_k_ids=b["rollout_student_top_k_ids"],
        )  # [B,A,blk,K] float, zeroed on invalid blocks
        if _timing:
            torch.cuda.synchronize()
            print(f"[TIMING] teacher worker compute = {_time.perf_counter()-_tc:.3f}s", flush=True)

        # Reshard the ROOT FSDP module after the (no-grad, no-backward) forward — mirrors verl
        # (fsdp_workers.py:2751-2754). FSDP1 auto-reshards nested units post-forward but leaves
        # the root flat-param all-gathered (it expects a backward that never comes for the
        # teacher). Without this, under FULL_SHARD+CPUOffload the root params stay GPU-resident
        # every step -> memory bloat / OOM on multi-GPU. No-op on single-GPU.
        from verl.utils.fsdp_utils import fsdp_version
        if self.world_size > 1 and fsdp_version(self.reward_module) == 1:
            self.reward_module._handle.reshard(True)

        out = DataProto.from_single_dict({"logp_target_on_topk": logp_target_on_topk})
        out.meta_info["timing"] = {}
        return out
