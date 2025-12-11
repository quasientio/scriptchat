"""Manually curated context window limits for models not in upstream data.

This file is NOT overwritten by scripts/update_model_defaults.py.
Add newer models here that aren't yet in the upstream repository.

These take priority over model_defaults.py when looking up context limits.
"""

# Model name -> context window size in tokens
# Use lowercase keys for case-insensitive matching
MODEL_CONTEXT_LIMITS_EXTRA: dict[str, int] = {
    # OpenAI models (newer, not in upstream)
    "gpt-5": 400000,
    "gpt-5-mini": 400000,
    "gpt-5-nano": 400000,
    "gpt-5.1": 400000,
    "accounts/fireworks/models/gpt-oss-120b": 128000,
    "accounts/fireworks/models/kimi-k2-thinking": 256000,
    "accounts/fireworks/models/deepseek-v3p2": 160000
}
