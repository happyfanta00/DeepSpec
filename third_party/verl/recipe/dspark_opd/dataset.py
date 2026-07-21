"""DSpark-OPD dataset — prompt-only corpus for live sglang DSPARK rollout.

`DSparkPromptDataset` reads a RAW prompt corpus (user-only conversations), applies the eval-golden
chat template, and yields PROMPT token ids. Upstream DSPARK sglang generates a FRESH response at
train time (on-policy), so no pre-generated response / target cache is read. `dspark_collate_fn`
right-pads the variable-length prompts (verl's default stack needs equal shapes).
"""
from __future__ import annotations

import os

import torch
from torch.utils.data import Dataset


class DSparkPromptDataset(Dataset):
    """Prompt-only dataset — reads a RAW prompt corpus (user-only conversations), applies the
    eval-golden chat template, and yields PROMPT token ids for live sglang rollout.

    Yields ONLY the prompt so upstream DSPARK sglang generates a FRESH response at train time
    (on-policy). The prompt is built EXACTLY like the eval golden (encode_chat_messages(
    add_generation_prompt=True, enable_thinking=False), §2.4) so the rollout distribution matches
    training/eval — this keeps accept length in the ~6.2 gsm8k ballpark (a cache-reverse-sliced
    prompt drops the enable_thinking=False `<think></think>` shell and tanks accept length; §5.6b).

    Source jsonl (data.dspark.prompt_jsonl_path): one JSON/line with `conversations: [{role, content}]`
    (user turn(s), NO assistant) — e.g. train_datasets/perfectblend_train.jsonl.

    __getitem__ -> {input_ids [P] long (the prompt), attention_mask [P] long (all 1),
                    loss_mask [P] long (all 0 — no response yet; sglang fills it live)}.
    dspark_collate_fn right-pads these like any 1-D keys. The sglang-rollout path
    (worker.train_step) extracts the prompt via attention_mask and replaces loss_mask with the
    new response's span after generation.
    """

    def __init__(self, data_files=None, tokenizer=None, processor=None, config=None,
                 max_samples: int = -1):
        from deepspec.data.jsonl_dataset import JsonLineDataset

        assert tokenizer is not None, "DSparkPromptDataset needs a tokenizer (from TARGET, §2.2)"
        self.tokenizer = tokenizer
        d = {}
        try:
            d = dict(config.get("dspark", {})) if config is not None else {}
        except Exception:  # noqa: BLE001
            d = {}
        path = d.get("prompt_jsonl_path", None)
        if not path:
            raise ValueError(
                "data.dspark.prompt_jsonl_path is required for DSparkPromptDataset (T3b). "
                "Set it to e.g. train_datasets/perfectblend_train.jsonl (user-only conversations)."
            )
        # run.sh cd's into third_party/verl, so a relative path resolves wrong. Resolve a relative
        # path against DEEPSPEC_DIR (or its default) so both the config and CLI overrides work.
        path = str(path)
        if not os.path.isabs(path) and not os.path.exists(path):
            root = os.environ.get("DEEPSPEC_DIR", "/home/ec2-user/efs_data/workspace/DeepSpec")
            cand = os.path.join(root, path)
            if os.path.exists(cand):
                path = cand
        self.max_length = int(d.get("prompt_max_length", 4096))
        self.enable_thinking = bool(d.get("enable_thinking", False))  # golden default False (§2.4)
        self.jsonl = JsonLineDataset(data_paths=[str(path)])
        self.n_samples = len(self.jsonl)
        cap = int(d.get("n_samples", -1) or -1)
        if cap > 0:
            self.n_samples = min(self.n_samples, cap)
        if max_samples and max_samples > 0:
            self.n_samples = min(self.n_samples, int(max_samples))
        print(f"[DSparkPromptDataset:T3b] prompt_jsonl={path} len={len(self.jsonl)} "
              f"using={self.n_samples} enable_thinking={self.enable_thinking}")

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, index: int) -> dict:
        from deepspec.data.parser import encode_chat_messages

        rec = self.jsonl[int(index)]
        # keep only user (+system) turns — drop any assistant if present (we generate it live).
        convs = rec.get("conversations") or rec.get("messages") or []
        messages = [{"role": c["role"], "content": c["content"]}
                    for c in convs if c.get("role") in ("system", "user")]
        ids = encode_chat_messages(
            self.tokenizer, messages,
            add_generation_prompt=True, enable_thinking=self.enable_thinking,
        )[0]  # [P] (golden prompt incl. the enable_thinking=False <think></think> shell)
        if ids.shape[0] > self.max_length:
            ids = ids[: self.max_length]
        ids = ids.to(torch.long)
        P = int(ids.shape[0])
        return {
            "input_ids": ids,
            "attention_mask": torch.ones(P, dtype=torch.long),   # all real prompt tokens
            "loss_mask": torch.zeros(P, dtype=torch.long),       # no response yet (sglang fills)
        }


def dspark_collate_fn(features: list[dict]) -> dict:
    """Right-pad the variable-length prompts into a batch (verl's default stack needs equal shapes).

    1-D keys (input_ids/attention_mask/loss_mask) -> (B, T_max) padded 0. attention_mask marks the
    real prompt tokens.
    """
    keys_1d = ("input_ids", "attention_mask", "loss_mask")
    bsz = len(features)
    t_max = max(int(f["input_ids"].shape[0]) for f in features)

    batch: dict = {}
    for key in keys_1d:
        if key not in features[0]:
            continue
        dtype = features[0][key].dtype
        out = torch.zeros((bsz, t_max), dtype=dtype)
        for i, f in enumerate(features):
            n = int(f[key].shape[0])
            out[i, :n] = f[key]
        batch[key] = out
    return batch
