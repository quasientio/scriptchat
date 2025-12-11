"""Default context window limits for well-known LLM models.

This module provides built-in context limits for common models so that
context usage % can be displayed in the status bar without requiring
explicit configuration.

Data sourced from: https://github.com/taylorwilsdon/llm-context-limits

To update these defaults from upstream, run:
    python scripts/update_model_defaults.py

For models not in upstream, add them to model_defaults_extra.py instead.
That file is never overwritten by the update script.

Precedence: User config > model_defaults_extra > this file
"""

from .model_defaults_extra import MODEL_CONTEXT_LIMITS_EXTRA

# Model name -> context window size in tokens
# Use lowercase keys for case-insensitive matching
MODEL_CONTEXT_LIMITS: dict[str, int] = {
    # OpenAI models
    "gpt-4.1": 1048000,
    "gpt-4.1-mini": 1048000,
    "gpt-4.1-nano": 1048000,
    "gpt-4o": 128000,
    "gpt-4o-mini": 128000,
    "gpt-4-turbo": 128000,
    "gpt-4": 128000,
    "gpt-3.5-turbo": 16000,
    "o3-mini": 200000,
    "o4-mini": 128000,
    "o1": 200000,
    "o1-mini": 128000,
    "o1-preview": 128000,
    "o1-pro": 200000,
    "o3": 128000,

    # Anthropic models
    "claude-opus-4-5": 200000,
    "claude-opus-4": 200000,
    "claude-sonnet-4": 200000,
    "claude-3.7-sonnet": 200000,
    "claude-3.5-sonnet": 200000,
    "claude-3.5-haiku": 200000,
    "claude-3-opus": 200000,
    "claude-3-sonnet": 200000,
    "claude-3-haiku": 200000,

    # DeepSeek models
    "deepseek-chat": 64000,
    "deepseek-reasoner": 64000,

    # Mistral models
    "mistral-large": 32000,
    "mistral-small": 32000,
    "mistral-medium": 32000,
    "mistral-nemo": 128000,
    "mistral-7b-instruct": 32000,

    # Gemini models (via OpenAI-compatible API)
    "gemini-2.0-flash": 1048000,
    "gemini-2.5-pro": 1048000,
    "gemini-1.5-pro": 1048000,
    "gemini-1.5-flash": 1048000,
    "gemma-3": 128000,

    # Qwen models
    "qwen2.5-coder-32b": 131072,
    "qwen2.5-72b-instruct": 131072,
    "qwen2.5-3b": 32000,
    "qwq": 32000,

    # Llama models
    "llama-3.3-70b": 131072,
    "llama3.3": 131072,
    "llama3.2": 128000,
    "llama3.1": 128000,
    "llama3": 8000,

    # Other models
    "phi4": 16000,
}


def get_default_context_limit(model_name: str) -> int | None:
    """Get the default context limit for a model.

    Checks model_defaults_extra first, then this file's defaults.
    Uses fuzzy matching: first tries exact match, then prefix match.
    For example, 'gpt-4o-2024-08-06' will match 'gpt-4o'.

    Args:
        model_name: The model name to look up

    Returns:
        Context limit in tokens, or None if unknown
    """
    name_lower = model_name.lower()

    # Check both dicts: extras first (higher priority), then upstream defaults
    for limits_dict in (MODEL_CONTEXT_LIMITS_EXTRA, MODEL_CONTEXT_LIMITS):
        # Try exact match first
        if name_lower in limits_dict:
            return limits_dict[name_lower]

    # Try prefix match across both dicts (longest match wins)
    best_match = None
    best_length = 0

    for limits_dict in (MODEL_CONTEXT_LIMITS_EXTRA, MODEL_CONTEXT_LIMITS):
        for known_model, limit in limits_dict.items():
            if name_lower.startswith(known_model) and len(known_model) > best_length:
                best_match = limit
                best_length = len(known_model)

    return best_match
