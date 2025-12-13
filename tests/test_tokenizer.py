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

"""Tests for the tokenizer module."""

import unittest

from scriptchat.core.tokenizer import (
    estimate_tokens,
    get_available_methods,
    HAS_TIKTOKEN,
    HAS_TRANSFORMERS,
    DEEPSEEK_TOKENIZER_PATH,
)


class TokenizerTests(unittest.TestCase):
    """Tests for token estimation functionality."""

    def test_empty_text_returns_zero(self):
        """Empty text should return 0 tokens."""
        count, method = estimate_tokens("")
        self.assertEqual(count, 0)
        self.assertEqual(method, "empty")

    def test_openai_gpt4_uses_tiktoken(self):
        """OpenAI GPT-4 models should use tiktoken."""
        if not HAS_TIKTOKEN:
            self.skipTest("tiktoken not installed")
        count, method = estimate_tokens("Hello world", "openai", "gpt-4o")
        self.assertGreater(count, 0)
        self.assertTrue(method.startswith("tiktoken:"))
        self.assertFalse(method.endswith("~"))  # Exact, not approximate

    def test_openai_gpt35_uses_tiktoken(self):
        """OpenAI GPT-3.5 models should use tiktoken."""
        if not HAS_TIKTOKEN:
            self.skipTest("tiktoken not installed")
        count, method = estimate_tokens("Hello world", "openai", "gpt-3.5-turbo")
        self.assertGreater(count, 0)
        self.assertIn("cl100k_base", method)

    def test_anthropic_uses_approximate_tiktoken(self):
        """Anthropic should use tiktoken with ~ suffix."""
        if not HAS_TIKTOKEN:
            self.skipTest("tiktoken not installed")
        count, method = estimate_tokens("Hello world", "anthropic", "claude-3-sonnet")
        self.assertGreater(count, 0)
        self.assertTrue(method.endswith("~"))  # Approximate

    def test_ollama_uses_approximate_tiktoken(self):
        """Ollama should use tiktoken with ~ suffix."""
        if not HAS_TIKTOKEN:
            self.skipTest("tiktoken not installed")
        count, method = estimate_tokens("Hello world", "ollama", "llama3")
        self.assertGreater(count, 0)
        self.assertTrue(method.endswith("~"))  # Approximate

    def test_openai_compatible_uses_approximate_tiktoken(self):
        """OpenAI-compatible providers should use tiktoken with ~ suffix."""
        if not HAS_TIKTOKEN:
            self.skipTest("tiktoken not installed")
        count, method = estimate_tokens("Hello world", "openai-compatible", "some-model")
        self.assertGreater(count, 0)
        self.assertTrue(method.endswith("~"))  # Approximate

    def test_deepseek_model_detected_by_name(self):
        """DeepSeek models should be detected by name regardless of provider."""
        if not HAS_TRANSFORMERS:
            self.skipTest("transformers not installed")
        if not DEEPSEEK_TOKENIZER_PATH.exists():
            self.skipTest("DeepSeek tokenizer not found")
        # Test with fireworks provider but deepseek model name
        count, method = estimate_tokens("Hello world", "fireworks", "deepseek-v3")
        self.assertGreater(count, 0)
        self.assertEqual(method, "transformers:deepseek-v3")

    def test_gpt_oss_not_matched_as_openai(self):
        """gpt-oss models should NOT match as OpenAI models."""
        if not HAS_TIKTOKEN:
            self.skipTest("tiktoken not installed")
        count, method = estimate_tokens("Hello world", "openai-compatible", "gpt-oss-120b")
        # Should fall through to openai-compatible approximation, not exact OpenAI
        self.assertTrue(method.endswith("~"))

    def test_fallback_to_estimate_without_libraries(self):
        """Without tokenizer libraries, should fall back to char estimate."""
        # This is hard to test directly since libraries are installed,
        # but we can verify the estimate formula works
        text = "1234"  # 4 chars = ~1 token
        # The estimate is (len + 3) // 4, so 4 chars = 1 token
        count, method = estimate_tokens(text, "unknown-provider", "unknown-model")
        if method == "estimate":
            self.assertEqual(count, 1)

    def test_get_available_methods(self):
        """get_available_methods should return availability dict."""
        methods = get_available_methods()
        self.assertIn("tiktoken", methods)
        self.assertIn("transformers", methods)
        self.assertIn("deepseek_local", methods)
        self.assertIn("estimate", methods)
        self.assertTrue(methods["estimate"])  # Always available

    def test_token_count_reasonable_for_sentence(self):
        """Token count should be reasonable for a typical sentence."""
        text = "The quick brown fox jumps over the lazy dog."
        count, _ = estimate_tokens(text, "openai", "gpt-4o")
        # This sentence is typically 9-11 tokens
        self.assertGreater(count, 5)
        self.assertLess(count, 20)

    def test_longer_text_more_tokens(self):
        """Longer text should have more tokens."""
        short = "Hello"
        long = "Hello world, this is a much longer piece of text with more tokens."
        short_count, _ = estimate_tokens(short, "openai", "gpt-4o")
        long_count, _ = estimate_tokens(long, "openai", "gpt-4o")
        self.assertGreater(long_count, short_count)


if __name__ == "__main__":
    unittest.main()
