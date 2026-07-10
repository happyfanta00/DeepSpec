"""verl 0.7.0 × transformers 5.10.2 兼容 shim。

verl 0.7.0 基于较老的 transformers 编写，用到 `AutoModelForVision2Seq`，
而 transformers 5.10.2 已将其更名/合并为 `AutoModelForImageTextToText`
（纯视觉多模态路径；DeepSpec 纯文本用不到，仅为让 verl import 链通过）。

用法：在**任何 verl import 之前**调用一次 `apply()`。例如在 recipe 入口
`recipe/dspark_opd/main.py` 顶部、或自检脚本中：

    from scripts.opd.verl_compat import apply as apply_verl_compat
    apply_verl_compat()
    # 之后再 import verl.*

设计原则：不改 verl 源码（保持 fork 干净）、不降 transformers（保持 DeepSpec
的 Qwen3 draft modeling 可用）。仅在运行时给 transformers 命名空间补别名。

已验证（transformers 5.10.2 + verl 0.7.0）：打此 shim 后
core_algos / fsdp_workers / rollout.base / trainer.main_ppo 全部 import 通过，
且无其它缺失符号（verl 顶层 from-transformers import 仅此一个符号缺失）。
"""
from __future__ import annotations

# (旧名, 新名) 别名表：旧名缺失且新名存在时补别名。
_ALIASES = [
    ("AutoModelForVision2Seq", "AutoModelForImageTextToText"),
]

_applied = False


def apply() -> list[str]:
    """给 transformers 命名空间补 verl 所需的旧符号别名。

    返回本次实际补齐的别名旧名列表（用于日志/自检）。幂等：重复调用安全。
    """
    global _applied
    import transformers

    patched: list[str] = []
    for old, new in _ALIASES:
        if not hasattr(transformers, old) and hasattr(transformers, new):
            setattr(transformers, old, getattr(transformers, new))
            patched.append(old)
    _applied = True
    return patched


if __name__ == "__main__":
    done = apply()
    print(f"[verl_compat] applied aliases: {done or '(none needed)'}")
