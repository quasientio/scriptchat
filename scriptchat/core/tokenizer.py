# Copyright 2024 ScriptChat contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Token estimation for various LLM providers.

Supports accurate counting with optional dependencies (tiktoken, transformers)
and falls back to character-based estimation when libraries are unavailable.
"""

import logging
from functools import lru_cache
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Try to import optional tokenizer libraries
try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False

try:
    import os
    import sys
    import warnings
    # Suppress transformers warnings about missing PyTorch/TensorFlow
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    _stderr = sys.stderr
    sys.stderr = open(os.devnull, 'w')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import transformers
            transformers.logging.set_verbosity_error()
    finally:
        sys.stderr.close()
        sys.stderr = _stderr
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


# OpenAI model to encoding mapping
OPENAI_ENCODINGS = {
    # GPT-4o and newer
    "gpt-4o": "o200k_base",
    "gpt-4o-mini": "o200k_base",
    "o1": "o200k_base",
    "o1-mini": "o200k_base",
    "o1-preview": "o200k_base",
    "o3-mini": "o200k_base",
    # GPT-4 and GPT-3.5
    "gpt-4": "cl100k_base",
    "gpt-4-turbo": "cl100k_base",
    "gpt-3.5-turbo": "cl100k_base",
}

# DeepSeek tokenizer path (inside package data)
DEEPSEEK_TOKENIZER_PATH = Path(__file__).parent.parent / "data" / "deepseek_tokenizer"

# Model patterns for tokenizer selection
DEEPSEEK_PATTERNS = ["deepseek"]
LLAMA_PATTERNS = ["llama", "llama2", "llama3"]
QWEN_PATTERNS = ["qwen"]
MISTRAL_PATTERNS = ["mistral", "mixtral"]


@lru_cache(maxsize=8)
def _get_tiktoken_encoding(encoding_name: str):
    """Get cached tiktoken encoding."""
    if not HAS_TIKTOKEN:
        return None
    try:
        return tiktoken.get_encoding(encoding_name)
    except Exception as e:
        logger.debug(f"Failed to load tiktoken encoding {encoding_name}: {e}")
        return None


@lru_cache(maxsize=4)
def _get_transformers_tokenizer(tokenizer_path_or_name: str):
    """Get cached transformers tokenizer."""
    if not HAS_TRANSFORMERS:
        return None
    try:
        return transformers.AutoTokenizer.from_pretrained(
            tokenizer_path_or_name,
            trust_remote_code=True
        )
    except Exception as e:
        logger.debug(f"Failed to load transformers tokenizer {tokenizer_path_or_name}: {e}")
        return None


def _match_model_pattern(model_name: str, patterns: list[str]) -> bool:
    """Check if model name matches any of the patterns."""
    model_lower = model_name.lower()
    return any(p in model_lower for p in patterns)


def _get_openai_encoding_name(model_name: str) -> str:
    """Get tiktoken encoding name for an OpenAI model."""
    model_lower = model_name.lower()

    # Check exact matches first
    for model, encoding in OPENAI_ENCODINGS.items():
        if model_lower.startswith(model):
            return encoding

    # Default to cl100k_base for unknown OpenAI models
    return "cl100k_base"


def estimate_tokens(
    text: str,
    provider_id: Optional[str] = None,
    model_name: Optional[str] = None
) -> tuple[int, str]:
    """Estimate token count for text.

    Args:
        text: The text to count tokens for
        provider_id: Provider ID (e.g., "openai", "anthropic", "ollama", "deepseek")
        model_name: Model name for more accurate tokenizer selection

    Returns:
        Tuple of (token_count, method) where method describes how the count was obtained:
        - "tiktoken:<encoding>" - Used tiktoken with specific encoding
        - "tiktoken:<encoding>~" - Used tiktoken as approximation (~ suffix)
        - "transformers:<tokenizer>" - Used transformers tokenizer
        - "estimate" - Used character-based estimation (~4 chars/token)
    """
    if not text:
        return (0, "empty")

    provider_lower = (provider_id or "").lower()
    model_lower = (model_name or "").lower()

    # 1. Try DeepSeek tokenizer first (by model name, any provider)
    if HAS_TRANSFORMERS and _match_model_pattern(model_lower, DEEPSEEK_PATTERNS):
        if DEEPSEEK_TOKENIZER_PATH.exists():
            tokenizer = _get_transformers_tokenizer(str(DEEPSEEK_TOKENIZER_PATH))
            if tokenizer:
                try:
                    tokens = tokenizer.encode(text)
                    return (len(tokens), "transformers:deepseek-v3")
                except Exception as e:
                    logger.debug(f"DeepSeek tokenizer encode failed: {e}")

    # 2. Try tiktoken for OpenAI models (by model name, any provider)
    #    Be specific to avoid false matches like "gpt-oss" (not an OpenAI model)
    if HAS_TIKTOKEN:
        openai_patterns = ["gpt-3", "gpt-4", "o1-", "o3-", "davinci", "curie", "babbage", "ada"]
        if any(m in model_lower for m in openai_patterns):
            encoding_name = _get_openai_encoding_name(model_name or "")
            encoding = _get_tiktoken_encoding(encoding_name)
            if encoding:
                try:
                    tokens = encoding.encode(text)
                    return (len(tokens), f"tiktoken:{encoding_name}")
                except Exception as e:
                    logger.debug(f"tiktoken encode failed: {e}")

    # 3. Provider-specific fallbacks using tiktoken as approximation
    if HAS_TIKTOKEN:
        # Anthropic - cl100k_base is reasonable approximation
        if provider_lower == "anthropic":
            encoding = _get_tiktoken_encoding("cl100k_base")
            if encoding:
                try:
                    tokens = encoding.encode(text)
                    return (len(tokens), "tiktoken:cl100k_base~")
                except Exception as e:
                    logger.debug(f"tiktoken encode failed for Anthropic: {e}")

        # Ollama and other open-source models - cl100k_base as approximation
        # Most are Llama/Mistral-based with similar BPE tokenization
        if provider_lower in ("ollama", "openai-compatible"):
            encoding = _get_tiktoken_encoding("cl100k_base")
            if encoding:
                try:
                    tokens = encoding.encode(text)
                    return (len(tokens), "tiktoken:cl100k_base~")
                except Exception as e:
                    logger.debug(f"tiktoken encode failed for {provider_lower}: {e}")

    # Fallback: character-based estimation
    # Average is ~4 characters per token for English text
    char_count = len(text)
    estimated_tokens = (char_count + 3) // 4  # Round up
    return (estimated_tokens, "estimate")


def get_available_methods() -> dict[str, bool]:
    """Get available tokenization methods.

    Returns:
        Dict with availability status for each method
    """
    return {
        "tiktoken": HAS_TIKTOKEN,
        "transformers": HAS_TRANSFORMERS,
        "deepseek_local": HAS_TRANSFORMERS and DEEPSEEK_TOKENIZER_PATH.exists(),
        "estimate": True,  # Always available
    }
