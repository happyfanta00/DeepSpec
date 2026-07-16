from .config import build_draft_config

# Gemma4 modeling depends on transformers.models.gemma4, which is absent on some
# transformers versions. Degrade gracefully to None so the Qwen3 path stays
# importable. Zero effect on envs where gemma4 is available (the import succeeds).
try:
    from .modeling import Gemma4Eagle3Model
except ImportError:
    Gemma4Eagle3Model = None

__all__ = [
    "Gemma4Eagle3Model",
    "build_draft_config",
]
