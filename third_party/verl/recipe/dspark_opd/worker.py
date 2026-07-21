"""DSpark-OPD worker (sglang DSPARK rollout, fused train_step).

DSparkActorRolloutRefWorker builds the non-standard Qwen3DSparkModel actor (verl's AutoModel path
would silently load the WRONG class — config model_type='qwen3' maps AutoModel -> standard Qwen3Model,
not our draft), freezes embed_tokens/lm_head, FSDP-wraps it (NO_SHARD: per-card full params,
use_orig_params=True for the frozen heads, MixedPrecision bf16), and co-hosts a FULL_SHARD teacher.

The single training entry is the fused `train_step` (sglang rollout → merged teacher forward →
dual-stream forward/reverse KL loss → draft weight sync). FSDP config rationale: docs/DSpark-OPD.md §2.6.5.
"""
from __future__ import annotations

import os

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy

from verl import DataProto
from verl.single_controller.base.decorator import Dispatch, make_nd_compute_dataproto_dispatch_fn, register
from verl.workers.fsdp_workers import ActorRolloutRefWorker
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
        """Build actor (+rollout+optimizer via super) THEN the co-resident teacher (fused design)."""
        super().init_model()
        oc = dict(self.config.model.get("override_config", {}) or {})
        # teacher path: explicit dspark_teacher_path, else reuse tokenizer path (== target, §2.2)
        teacher_path = oc.get("dspark_teacher_path") or oc.get("dspark_tokenizer_path")
        assert teacher_path, "override_config.dspark_teacher_path (or dspark_tokenizer_path) required"
        self._build_teacher(teacher_path)

        # Step-1 loss decoupling: build the typed OPD loss config ONCE (drives which self-
        # contained operators run + how they aggregate). See docs/opd/loss-refactor-design.md.
        from recipe.dspark_opd.loss_config import DSparkOPDLossConfig
        self._loss_cfg = DSparkOPDLossConfig.from_override_config(oc)

        # which teacher layers to capture for the draft's cross-attn context — read from the draft.
        draft = getattr(self, "actor_module", None) or self.actor.actor_module
        self._target_layer_ids = [int(x) for x in draft.target_layer_ids]

        # sglang rollout client handle (set by the trainer via set_sglang_server once the tp=8
        # DSPARK server actor is up). None until then.
        self._sglang_client = None
        # §5.6c KV-offload time-multiplexing: when DSPARK_SGLANG_KV_OFFLOAD=1 the server runs with a
        # LARGE KV pool for fast rollout, then releases ONLY the KV cache (weights stay resident) so
        # training reclaims that budget. False = server just launched with its KV pool present, so
        # the FIRST rollout must NOT resume (nothing was released yet). Flips to True after each
        # post-rollout release. Unused unless KV-offload is on.
        self._sglang_kv_released = False

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def set_sglang_server(self, host: str, port: int, tp_size: int = 8):
        """T3b: build this worker's pure HTTP CLIENT to the (already-running) T3a DSPARK server.

        Called by the trainer after build_dspark_sglang_server. make_sglang_client builds a
        no-launch client (this sglang's HttpServerEngineAdapter has no launch_server=False mode —
        it always launches; make_sglang_client bypasses __init__). All 8 workers point at the one
        tp=8 server; each sends its own DP-shard's prompts (server does continuous batching).

        ⚠️ tp_size MUST match the server's TP (T6): update_weights_from_tensor serializes ONE copy
        per tp rank (server scatters serialized_named_tensors[rank] to each TP worker,
        io_struct.py:1596). A client left at tp_size=1 would send only 1 copy -> ranks 1..7 get no
        weights -> broken draft sync. make_sglang_client sets server_args.tp_size so it matches.
        """
        from recipe.dspark_opd.sglang_server import make_sglang_client
        oc = dict(self.config.model.get("override_config", {}) or {})
        # model_path only satisfies ServerArgs validation (real path); client never loads it.
        target_path = oc.get("dspark_teacher_path") or oc.get("dspark_tokenizer_path") \
            or self.config.model.path
        self._sglang_client = make_sglang_client(host, int(port), int(tp_size), target_path)
        print(f"[DSparkActorRollout] rank={self.rank} sglang client -> http://{host}:{port} "
              f"(tp={tp_size})", flush=True)
        return True

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    def train_step(self, data: DataProto):
        """★ FUSED OPD step (sglang DSPARK rollout): rollout → merged teacher forward → dual-stream
        update → draft weight sync, all in ONE worker RPC.

        Pipeline (docs/opd/t5-train-step-tensor-contract.md):
          1. sglang rollout: prompts -> tp=8 DSPARK server live spec-decode -> fresh responses +
             per-token accept_state stream (KV-offload brackets it with resume/release).
          2. block plan: reconstruct anchors + dual-stream slot_type (accept/reject) from accept_state.
          3. MERGED teacher forward: ONE plain-causal forward yields BOTH the mid-layer hidden (draft
             cross-attn context) AND the teacher top-K at block positions (teacher_causal_hidden_and_topk).
          4. draft forward + dual-stream loss: accept stream = forward KL (teacher top-K, mass-covering),
             reject stream = reverse KL (draft top-K), + confidence BCE. Micro-batched + grad accum.
          5. T6: push freshly-trained draft weights back into the server (CUDA-IPC, draft-only).

        Requires DSPARK_SGLANG_ROLLOUT=1 + a set_sglang_server client + a prompt dataset (yields
        the T1 dspark_accept_state stream). Returns scalar metrics only (intermediates stay on GPU).
        """
        import os as _os
        import time as _time
        from verl.utils.fsdp_utils import fsdp_version
        from recipe.dspark_opd.block_plan_reconstruct import reconstruct_block_plan
        from recipe.dspark_opd.teacher_scoring import teacher_causal_hidden_and_topk
        from recipe.dspark_opd.sglang_rollout_bridge import (
            GOLDEN_SAMPLING_PARAMS, extract_prompts, rebuild_padded_batch, sglang_generate_batch,
        )
        from recipe.dspark_opd.loss_bridge import (
            align_teacher_to_draft, block_decay_weight_mask, draft_topk_logp, logp_on_ids_lse)
        from recipe.dspark_opd.losses import DSparkLossContext, compose_dspark_loss

        assert self._sglang_client is not None, (
            "train_step requires an sglang client (set_sglang_server); DSPARK_SGLANG_ROLLOUT path only.")

        rollout_n = int(self.config.rollout.get("n", 1))
        _dev = get_device_id()
        _timing = _os.environ.get("DSPARK_TIMING", "0") == "1" and self.rank == 0
        _viz = _os.environ.get("DSPARK_VIZ", "0") == "1" and self.rank == 0
        oc = dict(self.config.model.get("override_config", {}) or {})
        temperature = float(self.config.rollout.get("temperature", 1.0))
        top_k = int(oc.get("log_prob_top_k", 0))
        loss_cfg = self._loss_cfg              # typed OPD loss config (built in init_model)
        gamma = loss_cfg.loss_decay_gamma

        def _tick(_prev, _label, _extra=""):
            if not _timing:
                return None
            torch.cuda.synchronize()
            _now = _time.perf_counter()
            if _prev is not None:
                print(f"[T5-TIMING] {_label}: {_now - _prev:.3f}s {_extra}", flush=True)
            return _now

        _tk = _tick(None, "start")

        # ===== 1. sglang DSPARK rollout: prompts -> fresh responses + accept_state stream =====
        # The prompt dataset (DSparkPromptDataset) dispatched the B UNIQUE prompts. We extract them,
        # have the tp=8 server generate rollout_n INDEPENDENT responses each (B*n total), and rebuild
        # a right-padded (prompt++response) batch carrying the T1 accept_state stream.
        data = data.to(_dev)
        db = data.batch
        prompts = extract_prompts(db["input_ids"], db["loss_mask"], attention_mask=db.get("attention_mask"))
        sp = dict(GOLDEN_SAMPLING_PARAMS)
        sp["temperature"] = temperature
        # ★ TRAIN rollout may shorten max_new_tokens (dspark_rollout_max_new_tokens) below the eval
        # GOLDEN 2048 to cut the long-response tail that dominates rollout wall time. This DEVIATES
        # from eval caliber — GOLDEN_SAMPLING_PARAMS (eval) stays 2048; verify accept length at 2048
        # before finalizing. 0/absent -> keep the golden 2048.
        _rmnt = int(oc.get("dspark_rollout_max_new_tokens", 0) or 0)
        if _rmnt > 0:
            sp["max_new_tokens"] = _rmnt
        # §5.6c KV-offload: resume(kv) rebuilds the KV pool before generate; release(kv) frees it back
        # to training right after (weights stay resident).
        self._sglang_kv_resume()
        gen_outs = sglang_generate_batch(self._sglang_client, prompts, rollout_n, sampling_params=sp)
        self._sglang_kv_release()
        if _timing:
            _resp = sorted(len(o.get("output_ids") or []) for o in gen_outs)
            _n = len(_resp)

            def _pct(q):                              # nearest-rank percentile on sorted lengths
                return _resp[min(_n - 1, int(q * _n))] if _n else 0
            _stats = (f"n_req={_n} min={_resp[0] if _n else 0} p50={_pct(0.50)} "
                      f"p90={_pct(0.90)} p99={_pct(0.99)} max={_resp[-1] if _n else 0} "
                      f"mean={sum(_resp) / _n:.0f}" if _n else "n_req=0")
            _tk = _tick(_tk, "rollout(resume+generate+release)", f"| resp_len {_stats}")
        if _viz:
            self._viz_rollout(prompts, gen_outs, rollout_n)
        b = rebuild_padded_batch(prompts, gen_outs, rollout_n, device=_dev)

        module = self.actor.actor_module          # FSDP root (update goes through module(...))
        unwrapped = getattr(self, "actor_module", None) or module
        module.train()
        # Prime the ROOT FSDP's lazy-init before the grad forward (the fused step bypasses verl's
        # rollout_mode state_dict() call that normally primes it; idempotent after step 1).
        from torch.distributed.fsdp import _runtime_utils as _fsdp_rt
        _fsdp_rt._lazy_init(module, module)

        # ===== 2. block plan: anchors + dual-stream slot_type from the accept_state trajectory =====
        blk = int(unwrapped.block_size)
        teacher_top_k = int(oc.get("dspark_teacher_top_k", 64))
        # A (max_anchors) = # decode rounds, DATA-DRIVEN. Optional hard cap dspark_max_anchors_cap
        # (0 = unlimited; caps the [B,A,blk,V] tensor size — tail rounds dropped with a loud log).
        _acc = b["dspark_accept_state"]                       # [B,R] pad=-1
        _rl = b["response_lengths"]
        _mr = 1
        for _i in range(_acc.shape[0]):
            _li = int(_rl[_i])
            if _li > 0:
                _mr = max(_mr, int((_acc[_i, :_li] == 1).sum().item()) + 1)
        _cap = int(oc.get("dspark_max_anchors_cap", 0) or 0)
        max_anchors = min(_mr, _cap) if _cap > 0 else _mr
        plan = reconstruct_block_plan(
            input_ids=b["input_ids"], accept_state=b["dspark_accept_state"],
            response_lengths=b["response_lengths"], prompt_lengths=b["prompt_lengths"],
            block_size=blk, max_anchors=max_anchors)
        if _viz:
            self._viz_block_plan(plan, b["input_ids"], b["prompt_lengths"], b["response_lengths"])
        _tk = _tick(_tk, "reconstruct", f"A={max_anchors}")
        anchors = plan["anchor_positions"]                 # [B,A] input_ids idx (real trajectory)
        keep = plan["block_keep_mask"].bool()              # [B,A]
        tokens = plan["tokens"]                            # [B,A,blk] real accept+reject tokens
        slot_type = plan["slot_type"]                      # [B,A,blk] int8: 0=acc 1=rej -1=none
        accept_mask = (slot_type == 0)                     # [B,A,blk] accept stream
        reject_mask = (slot_type == 1)                     # [B,A,blk] reject/boundary stream
        eval_mask = (slot_type >= 0)                       # [B,A,blk] accept ∪ reject

        # ===== 3. MERGED teacher forward: hidden (for draft) + teacher top-K (for loss), micro-batched
        # ONE plain-causal forward per micro yields the mid-layer hidden AND the block-position top-K.
        # Micro-batch over the batch dim (the [micro,T,V] logits peak is the memory point).
        _tmb = int(self.config.actor.get("ppo_micro_batch_size_per_gpu", 0) or 0)
        _bsz = int(b["input_ids"].shape[0])
        _tchunks = torch.arange(_bsz, device=b["input_ids"].device).split(_tmb) if _tmb > 0 else [
            torch.arange(_bsz, device=b["input_ids"].device)]
        _th_parts, _ids_parts, _logp_parts = [], [], []
        for _tsl in _tchunks:
            _m = teacher_causal_hidden_and_topk(
                self.teacher_module, input_ids=b["input_ids"][_tsl],
                attention_mask=b["attention_mask"][_tsl], target_layer_ids=self._target_layer_ids,
                anchor_positions=anchors[_tsl], block_keep_mask=keep[_tsl],
                block_size=blk, teacher_top_k=teacher_top_k)
            _th_parts.append(_m["target_hidden_states"])
            _ids_parts.append(_m["t_ids"])
            _logp_parts.append(_m["t_logp"])
        b["target_hidden_states"] = torch.cat(_th_parts, dim=0)   # [B,T,L*H] (draft cross-attn ctx)
        t64_ids = torch.cat(_ids_parts, dim=0)             # [B,A,blk,Kt]
        t64_logp = torch.cat(_logp_parts, dim=0)           # [B,A,blk,Kt]
        _tk = _tick(_tk, "merged-teacher-fwd",
                    f"B={_bsz} T={int(b['input_ids'].shape[1])} nchunk={len(_tchunks)}")

        if self.world_size > 1 and fsdp_version(self.teacher_module) == 1:
            self.teacher_module._handle.reshard(True)   # drop all-gathered teacher params (FULL_SHARD)

        # ===== 4. draft forward + dual-stream loss (micro-batch + grad accum) =====
        idx = torch.arange(b["input_ids"].shape[0], device=b["input_ids"].device)
        chunks = idx.split(_tmb) if _tmb > 0 else [idx]
        n_micro = len(chunks)
        scale = 1.0 / n_micro
        # forward KL (accept stream) enabled? If so, per-micro also gather draft logπ on the TEACHER
        # top-K ids (S_on_T) + carry teacher top-K logπ (T_topk). K = log_prob_top_k, same as reverse.
        _fwd_on = any(n == "forward_kl" for n, _ in loss_cfg.enabled_terms())

        self.actor.actor_optimizer.zero_grad(set_to_none=True)
        agg: dict[str, float] = {}
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
            # candidate algo: draft's OWN top-K (grad, true full-vocab logπ) is the reject-stream
            # support; teacher logπ on those ids = teacher top-K value if present else min (align).
            K = max(1, top_k)
            d_ids, S_grad = draft_topk_logp(out.draft_logits, K, temperature=temperature)
            T_sl = align_teacher_to_draft(d_ids, t64_ids[sl], t64_logp[sl])   # [B,A,blk,K] no-grad
            S_on_T_sl = None
            T_topk_sl = None
            if _fwd_on:
                # forward KL (accept stream): teacher top-K = teacher top-Kt's first K; draft logπ on
                # those ids via logsumexp (true full-vocab, grad). §4b.
                _tk_ids = t64_ids[sl][..., :K]                               # [B,A,blk,K] teacher top-K ids
                T_topk_sl = t64_logp[sl][..., :K]                            # [B,A,blk,K] teacher top-K logπ
                S_on_T_sl = logp_on_ids_lse(out.draft_logits, _tk_ids, temperature=temperature)
            if _viz and int(sl[0]) == 0:                     # first micro-batch holds sample 0
                self._viz_teacher_draft(d_ids, S_grad, T_sl, eval_mask[sl], tokens[sl])
                self._viz_reject(d_ids, S_grad, T_sl, reject_mask[sl], tokens[sl])
            # dual-stream loss: accept -> forward KL, reject -> reverse KL, + confidence BCE.
            ctx = DSparkLossContext(
                S_grad=S_grad,
                T_on_S=T_sl,
                S_logp_old=None,
                eval_mask=eval_mask[sl],
                decay_mask=block_decay_weight_mask(eval_mask[sl], blk, gamma),
                confidence_pred=out.confidence_pred,
                block_size=blk,
                accept_mask=accept_mask[sl],
                reject_mask=reject_mask[sl],
                loss_decay_gamma=gamma,
                S_on_T=S_on_T_sl,
                T_topk=T_topk_sl,
            )
            loss, loss_metrics = compose_dspark_loss(ctx, loss_cfg)
            (loss * scale).backward()
            for k, v in loss_metrics.items():
                agg[k] = agg.get(k, 0.0) + float(v.item() if hasattr(v, "item") else v) * scale

        _tk = _tick(_tk, "draft-fwd+loss+backward", f"n_micro={n_micro}")
        grad_norm = self.actor._optimizer_step()
        if self.actor_lr_scheduler is not None:
            self.actor_lr_scheduler.step()
        _tk = _tick(_tk, "optimizer-step")

        # ===== 5. T6: push freshly-trained DRAFT weights back into the sglang server (rank0) =====
        # NO_SHARD: every DP rank holds an IDENTICAL draft after the synced step; push from rank 0
        # only (draft_model_only=True, target frozen). Other ranks barrier so the next rollout waits.
        n_draft_pushed = 0
        if self.rank == 0:
            n_draft_pushed = self._push_draft_weights_to_sglang()
        if self.world_size > 1:
            torch.distributed.barrier()
        _tk = _tick(_tk, "T6-weight-push")

        def _s(v):
            return torch.tensor([float(v.item() if hasattr(v, "item") else v)])
        metrics = {k: _s(v) for k, v in agg.items()}
        metrics["actor/grad_norm"] = _s(grad_norm)
        metrics["actor/n_micro"] = _s(n_micro)
        metrics["actor/draft_weights_pushed"] = _s(n_draft_pushed)
        return DataProto.from_single_dict(metrics)

    # ==================== §5.6c KV-offload rollout/train time-multiplexing ====================
    # When DSPARK_SGLANG_KV_OFFLOAD=1, the tp=8 server runs with a LARGE KV pool
    # (DSPARK_SGLANG_MEM_FRACTION, e.g. 0.6) for high-throughput rollout. Each step brackets the
    # sglang generation with resume(kv)/release(kv): resume rebuilds a FRESH EMPTY KV pool before
    # generating; release frees it (flush_cache + free pages, NO CPU copy — KV content is discarded)
    # after generating, returning the budget to training. WEIGHTS stay resident the whole time
    # (never released) — so there is no per-step weight export/import churn and none of the weight-
    # corruption risk from full release/resume (§resume-bug). release/resume are GLOBAL server ops,
    # so only rank 0 drives them; the other ranks barrier so no worker generates before the KV pool
    # is back (resume) and none proceeds to training before the KV pool is freed (release).

    def _sglang_kv_offload_on(self) -> bool:
        return (os.environ.get("DSPARK_SGLANG_KV_OFFLOAD", "0") == "1"
                and self._sglang_client is not None)

    def _sglang_kv_resume(self):
        """Rebuild the server's KV pool (fresh empty) before a rollout. rank0-driven + barrier.
        Skips the very first rollout (server launched with its KV pool already present).

        ★ EVERY rank runs aggressive_empty_cache BEFORE resume. The KV pool is rebuilt by each GPU's
        sglang scheduler via torch_memory_saver's cu_mem_create, which asks the CUDA DRIVER directly
        for physical pages — it does NOT see PyTorch's caching allocator reserve. After the previous
        step's teacher/draft forward, the freed activations sit in PyTorch's reserved-but-unallocated
        cache (not returned to the driver), so cu_mem_create fails with `CUresult error: 2 (out of
        memory)` even though torch itself has room. empty_cache returns that reserve to the driver so
        the KV pool fits. This is exactly what verl's fsdp_sglang sharding manager does around its
        own resume_memory_occupation (aggressive_empty_cache before/after, fsdp_sglang.py:169-195).
        All ranks must do it (each GPU hosts one TP scheduler that allocates on THAT card)."""
        if not self._sglang_kv_offload_on():
            return
        if self._sglang_kv_released:
            from verl.utils.memory_utils import aggressive_empty_cache
            aggressive_empty_cache(force_sync=True)          # all ranks: free reserve to the driver
            if self.world_size > 1:
                torch.distributed.barrier()                  # ensure every card freed before alloc
            if self.rank == 0:
                self._sglang_client.resume_memory_occupation(tags=["kv_cache"])
        self._sglang_kv_released = False
        if self.world_size > 1:
            torch.distributed.barrier()

    def _sglang_kv_release(self):
        """Free the server's KV pool after a rollout (weights stay resident). rank0-driven + barrier.
        Barrier FIRST so every worker has finished generating (server idle) before rank0 releases —
        release_memory_occupation asserts the scheduler is fully idle."""
        if not self._sglang_kv_offload_on():
            return
        if self.world_size > 1:
            torch.distributed.barrier()
        if self.rank == 0:
            self._sglang_client.release_memory_occupation(tags=["kv_cache"])
        self._sglang_kv_released = True
        if self.world_size > 1:
            torch.distributed.barrier()

    def _push_draft_weights_to_sglang(self) -> int:
        """T6: export draft trainable weights + push draft-only into the sglang server (CUDA-IPC).

        Reuses verl's rollout_mode idiom (fsdp_workers.py:674-709): state_dict -> convert_weight_keys
        (strip FSDP prefixes) -> .full_tensor() if DTensor. Drops the frozen embed/lm_head/rotary
        (Qwen3DSparkModel.load_weights skips them — the same 47/47-CLEAN set as §4.0/T2). Names
        already match Qwen3DSparkModel.load_weights (NO draft_model. prefix — that is Draft-OPD's
        composed-student wrapper, not ours). Pushes via the HTTP client's
        update_weights_from_tensor(draft_model_only=True) — CUDA-IPC, target frozen (T2-verified,
        dspark_draft_update_probe.py:220/362). Returns #tensors pushed (0 -> raise).
        """
        from verl.utils.model import convert_weight_keys

        fsdp_module = self.actor.actor_module
        unwrapped = getattr(fsdp_module, "_fsdp_wrapped_module", fsdp_module)
        sd = convert_weight_keys(fsdp_module.state_dict(), unwrapped)
        _dev = get_device_id()
        try:
            from torch.distributed.tensor import DTensor
        except Exception:  # noqa: BLE001
            from torch.distributed._tensor import DTensor

        named = []
        for name, param in sd.items():
            # drop frozen heads (load_weights ignores them; keep the trained backbone/fc/markov/conf)
            if name.startswith(("embed_tokens.", "lm_head.", "rotary_emb.")):
                continue
            t = param
            if isinstance(t, DTensor):
                t = t.full_tensor()
            named.append((name, t.to(_dev, non_blocking=True).to(torch.bfloat16).contiguous()))

        if not named:
            raise RuntimeError("[T6] draft weight export produced 0 tensors — check state_dict/prefix rules")
        # flush_cache clears radix/KV so the next rollout recomputes with the new draft. But under
        # KV-offload the KV pool is ALREADY released here (freed after this step's rollout) and the
        # next rollout's resume(kv) rebuilds a FRESH EMPTY pool anyway — so a flush now is both
        # redundant AND unsafe (flush_cache -> token_to_kv_pool_allocator.clear() on freed pages).
        # Skip it in that case; the resume-rebuilt empty pool is the flush.
        flush = not (self._sglang_kv_offload_on() and self._sglang_kv_released)
        self._sglang_client.update_weights_from_tensor(
            named, draft_model_only=True, flush_cache=flush)
        return len(named)

    # ==================== DSPARK_VIZ: T5 sglang-path intermediate visualization ====================
    # Gated on DSPARK_VIZ=1 + rank 0 (avoid 8-worker interleaving). Three probes trace one train_step
    # end-to-end for sample 0: (A) the live sglang DSPARK rollout text + its T1 accept_state stream,
    # (B) the block plan reconstructed from that accept_state, (C) teacher-vs-draft logπ/p on the SAME
    # draft top-K candidate ids at the same in-block token positions. Purely diagnostic (no grad, no
    # state mutation); bounded output (first few samples/blocks/positions).

    def _viz_rollout(self, prompts, gen_outs, rollout_n, max_samples=2):
        """(A) sglang DSPARK rollout: decoded prompt tail + response + T1 accept_state (sample 0..N)."""
        from recipe.dspark_opd.sglang_rollout_bridge import _accept_state_of
        tok = self.tokenizer
        n = min(len(gen_outs), int(max_samples))
        print(f"\n{'=' * 84}\n[VIZ-A] sglang DSPARK rollout: {len(gen_outs)} responses, showing {n}", flush=True)
        for i in range(n):
            prompt = prompts[i // int(rollout_n)]
            resp = [int(t) for t in (gen_outs[i].get("output_ids") or [])]
            st = _accept_state_of(gen_outs[i])
            nb = sum(1 for s in st if s == 1)                     # #COMMIT_BOUNDARY == #decode rounds
            al = (len(st) / nb) if nb else float(len(st))         # accept_len ≈ tokens / rounds
            p_txt = tok.decode(prompt[-80:], skip_special_tokens=False)
            r_txt = tok.decode(resp[:120], skip_special_tokens=False)
            print(f"\n-- response[{i}]  P={len(prompt)}  R={len(resp)}  rounds={nb}  accept_len≈{al:.2f}", flush=True)
            print(f"   prompt tail : ...{p_txt!r}", flush=True)
            print(f"   response    : {r_txt!r}{' ...(trunc)' if len(resp) > 120 else ''}", flush=True)
            print(f"   accept_state[:48] (2=seed 0=accept 1=boundary): {st[:48]}", flush=True)

    def _viz_block_plan(self, plan, input_ids, prompt_lengths, response_lengths, max_blocks=6):
        """(B) block plan reconstructed from accept_state (sample 0): anchor/n_accept/tokens per block."""
        tok = self.tokenizer
        anchors = plan["anchor_positions"][0]
        keep = plan["block_keep_mask"][0]
        tokens = plan["tokens"][0]
        eval_mask = plan["eval_mask"][0]
        plen, rlen = int(prompt_lengths[0]), int(response_lengths[0])
        n_valid = int(keep.sum().item())
        print(f"\n[VIZ-B] reconstructed block plan (sample 0): P={plen} R={rlen} "
              f"valid_blocks={n_valid}/{int(keep.shape[0])} block_size={int(tokens.shape[1])}", flush=True)
        shown = 0
        for r in range(int(keep.shape[0])):
            if not bool(keep[r]):
                continue
            if shown >= int(max_blocks):
                print(f"   ... ({n_valid - shown} more valid blocks)", flush=True)
                break
            a = int(anchors[r])
            n_acc = int(eval_mask[r].sum().item())               # ACCEPT tokens (option B eval span)
            blk_ids = [int(t) for t in tokens[r].tolist()]
            show_n = min(len(blk_ids), n_acc + 2)                # accepts + boundary/reject slot
            piece = tok.decode(blk_ids[:show_n], skip_special_tokens=False)
            anchor_piece = tok.convert_ids_to_tokens([int(input_ids[0, a])])[0]
            print(f"   block{r:>3}: anchor@input_pos={a} (resp_idx={a - plen}) anchor_tok={anchor_piece!r} "
                  f"n_accept={n_acc} committed->{piece!r}", flush=True)
            shown += 1

    def _viz_teacher_draft(self, d_ids, S_grad, T_sl, eval_mask, tokens, topn=8):
        """(C) teacher-vs-draft logπ/p on the SAME draft top-K ids, sample 0's first valid block,
        first eval position. '*' marks the candidate == the REAL accepted token."""
        import math
        tok = self.tokenizer
        for r in range(int(eval_mask.shape[1])):
            pos_list = torch.nonzero(eval_mask[0, r], as_tuple=False).flatten().tolist()
            if not pos_list:
                continue
            pos = pos_list[0]                                    # first ACCEPT position of first block
            ids = d_ids[0, r, pos].detach().tolist()
            s_lp = S_grad[0, r, pos].detach().float().tolist()   # draft true full-vocab logπ (grad→detached)
            t_lp = T_sl[0, r, pos].detach().float().tolist()     # teacher logπ aligned to draft ids
            real = int(tokens[0, r, pos])
            print(f"\n[VIZ-C] teacher-vs-draft top-K logπ (sample0 block{r} pos{pos}, K={len(ids)}); "
                  f"real accepted tok={tok.convert_ids_to_tokens([real])[0]!r}", flush=True)
            print(f"     {'token':>16} {'draft_logπ':>11} {'draft_p':>8} {'tchr_logπ':>11} "
                  f"{'tchr_p':>8} {'Δlogπ':>7}", flush=True)
            order = sorted(range(len(s_lp)), key=lambda k: s_lp[k], reverse=True)
            for k in order[:int(topn)]:
                piece = tok.convert_ids_to_tokens([int(ids[k])])[0]
                mark = " *" if int(ids[k]) == real else "  "
                print(f"  {mark}{piece!r:>14} {s_lp[k]:>11.3f} {math.exp(s_lp[k]):>8.4f} "
                      f"{t_lp[k]:>11.3f} {math.exp(t_lp[k]):>8.4f} {t_lp[k] - s_lp[k]:>+7.3f}", flush=True)
            return                                               # one block/position is enough

    def _viz_reject(self, d_ids, S_grad, T_sl, reject_mask, tokens, max_blocks=3, topn=8,
                    kl_thresh=2.0):
        """(R) HIGH-reverse-KL REJECT-position teacher-vs-draft logπ/p on the draft top-K ids (the
        reverse-KL support). Prints only reject slots whose per-token reverse KL > kl_thresh (default
        2.0), up to max_blocks — to inspect WHETHER a large reject KL is expected: the reject token is
        the target's CORRECTION (draft's proposal was rejected), so draft should predict it poorly
        while teacher predicts it well. Scans ALL samples in the (micro-)batch, not just sample 0, so
        high-KL slots are actually found. Shows the top-K candidates' draft_p/tchr_p + the slot's
        reverse KL Σ_j softmax_K(S)_j·(S_j−T_j). '*' marks the candidate == the REAL correction token."""
        import math
        tok = self.tokenizer
        B, Anum = int(reject_mask.shape[0]), int(reject_mask.shape[1])
        shown = 0
        for bi in range(B):
            for r in range(Anum):
                pos_list = torch.nonzero(reject_mask[bi, r], as_tuple=False).flatten().tolist()
                if not pos_list:
                    continue
                pos = pos_list[0]                                # the reject/boundary slot of block r
                s_lp = S_grad[bi, r, pos].detach().float()       # draft logπ on its top-K (detached)
                t_lp = T_sl[bi, r, pos].detach().float()         # teacher logπ aligned to those ids
                # per-token reverse KL exactly as the loss computes it: Σ_j softmax_K(S)_j·(S_j−T_j).
                p = torch.softmax(s_lp, dim=-1)
                rev_kl = float((p * (s_lp - t_lp)).sum())
                if rev_kl <= float(kl_thresh):                   # only high-KL reject slots
                    continue
                # top-K-support entropy (nats): H(softmax_K(·)) over the SAME K candidates the KL
                # uses (NOT full-vocab — we only have the top-K logπ here). Low H = peaked/confident.
                q_t = torch.softmax(t_lp, dim=-1)
                ent_d = float(-(p * torch.log(p.clamp_min(1e-12))).sum())
                ent_t = float(-(q_t * torch.log(q_t.clamp_min(1e-12))).sum())
                ids = d_ids[bi, r, pos].detach().tolist()
                real = int(tokens[bi, r, pos])
                real_in = real in ids
                print(f"\n[VIZ-R] HIGH-KL REJECT slot (sample{bi} block{r} pos{pos}, K={len(ids)}); "
                      f"real correction tok={tok.convert_ids_to_tokens([real])[0]!r} "
                      f"in_draft_topK={real_in}; reverse_kl@slot={rev_kl:+.4f} (>{kl_thresh}); "
                      f"topK-entropy draft={ent_d:.3f} tchr={ent_t:.3f} nats", flush=True)
                print(f"     {'token':>16} {'draft_logπ':>11} {'draft_p':>8} {'tchr_logπ':>11} "
                      f"{'tchr_p':>8} {'Δlogπ':>7}", flush=True)
                s_l, t_l = s_lp.tolist(), t_lp.tolist()
                order = sorted(range(len(s_l)), key=lambda k: s_l[k], reverse=True)
                for k in order[:int(topn)]:
                    piece = tok.convert_ids_to_tokens([int(ids[k])])[0]
                    mark = " *" if int(ids[k]) == real else "  "
                    print(f"  {mark}{piece!r:>14} {s_l[k]:>11.3f} {math.exp(s_l[k]):>8.4f} "
                          f"{t_l[k]:>11.3f} {math.exp(t_l[k]):>8.4f} {t_l[k] - s_l[k]:>+7.3f}", flush=True)
                shown += 1
                if shown >= int(max_blocks):
                    return
        if shown == 0:
            print(f"\n[VIZ-R] no reject slot with reverse_kl > {kl_thresh} in this micro-batch", flush=True)

