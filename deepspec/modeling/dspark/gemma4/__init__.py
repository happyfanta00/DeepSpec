from .config import build_draft_config

# Gemma4 modeling depends on transformers.models.gemma4, which is absent on some
# transformers versions. Degrade gracefully to None so the Qwen3 path stays
# importable. Zero effect on envs where gemma4 is available (the import succeeds).
try:
    from .modeling import Gemma4DSparkModel
except ImportError:
    Gemma4DSparkModel = None

__all__ = [
    "Gemma4DSparkModel",
    "build_draft_config",
]
