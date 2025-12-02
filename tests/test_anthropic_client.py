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

"""Tests for Anthropic client."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from scriptchat.core.anthropic_client import AnthropicChatClient, THINKING_BUDGET_PRESETS
from scriptchat.core.config import Config, ModelConfig, ProviderConfig
from scriptchat.core.conversations import Conversation, Message


def make_config(tmp_path: Path):
    provider = ProviderConfig(
        id="anthropic",
        type="anthropic",
        api_url="https://api.anthropic.com",
        api_key="test-key",
        models=[ModelConfig(name="claude-3-5-sonnet", contexts=[200000])],
        streaming=True,
        headers={},
        default_model="claude-3-5-sonnet",
    )
    return Config(
        api_url="https://api.anthropic.com",
        api_key="test-key",
        conversations_dir=tmp_path,
        exports_dir=None,
        enable_streaming=True,
        system_prompt=None,
        default_provider="anthropic",
        default_model="claude-3-5-sonnet",
        default_temperature=0.7,
        timeout=30,
        log_level="INFO",
        log_file=None,
        providers=[provider],
        file_confirm_threshold_bytes=40_000,
    )


class AnthropicClientTests(unittest.TestCase):
    """Tests for AnthropicChatClient."""

    def test_client_initialization(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = make_config(Path(tmpdir))
            provider = config.providers[0]
            client = AnthropicChatClient(config, provider, timeout=30)

            self.assertEqual(client.provider.id, "anthropic")
            self.assertIn("x-api-key", client.session.headers)
            self.assertIn("anthropic-version", client.session.headers)

    def test_thinking_budget_presets(self):
        self.assertEqual(THINKING_BUDGET_PRESETS["low"], 4000)
        self.assertEqual(THINKING_BUDGET_PRESETS["medium"], 16000)
        self.assertEqual(THINKING_BUDGET_PRESETS["high"], 32000)
        self.assertEqual(THINKING_BUDGET_PRESETS["max"], 55000)

    def test_get_thinking_budget_from_explicit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = make_config(Path(tmpdir))
            provider = config.providers[0]
            client = AnthropicChatClient(config, provider, timeout=30)

            convo = Conversation(
                id=None,
                provider_id="anthropic",
                model_name="claude-3-5-sonnet",
                temperature=0.7,
                thinking_budget=12000,
            )
            budget = client._get_thinking_budget(convo)
            self.assertEqual(budget, 12000)

    def test_get_thinking_budget_from_reasoning_level(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = make_config(Path(tmpdir))
            provider = config.providers[0]
            client = AnthropicChatClient(config, provider, timeout=30)

            convo = Conversation(
                id=None,
                provider_id="anthropic",
                model_name="claude-3-5-sonnet",
                temperature=0.7,
                reasoning_level="high",
            )
            budget = client._get_thinking_budget(convo)
            self.assertEqual(budget, 32000)

    def test_get_thinking_budget_explicit_overrides_level(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = make_config(Path(tmpdir))
            provider = config.providers[0]
            client = AnthropicChatClient(config, provider, timeout=30)

            convo = Conversation(
                id=None,
                provider_id="anthropic",
                model_name="claude-3-5-sonnet",
                temperature=0.7,
                reasoning_level="low",
                thinking_budget=50000,
            )
            budget = client._get_thinking_budget(convo)
            self.assertEqual(budget, 50000)  # Explicit takes precedence

    def test_get_thinking_budget_none_when_disabled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = make_config(Path(tmpdir))
            provider = config.providers[0]
            client = AnthropicChatClient(config, provider, timeout=30)

            convo = Conversation(
                id=None,
                provider_id="anthropic",
                model_name="claude-3-5-sonnet",
                temperature=0.7,
            )
            budget = client._get_thinking_budget(convo)
            self.assertIsNone(budget)

    def test_extract_content_simple(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = make_config(Path(tmpdir))
            provider = config.providers[0]
            client = AnthropicChatClient(config, provider, timeout=30)

            data = {
                "content": [
                    {"type": "text", "text": "Hello, "},
                    {"type": "text", "text": "world!"}
                ]
            }
            content = client._extract_content(data)
            self.assertEqual(content, "Hello, world!")

    def test_extract_content_with_thinking_blocks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = make_config(Path(tmpdir))
            provider = config.providers[0]
            client = AnthropicChatClient(config, provider, timeout=30)

            data = {
                "content": [
                    {"type": "thinking", "thinking": "Let me think..."},
                    {"type": "text", "text": "The answer is 42."}
                ]
            }
            content = client._extract_content(data)
            self.assertEqual(content, "The answer is 42.")  # Thinking excluded

    @patch('scriptchat.core.anthropic_client.requests.Session')
    def test_chat_single_builds_correct_payload(self, mock_session_class):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = make_config(Path(tmpdir))
            provider = config.providers[0]

            # Mock session and response
            mock_session = MagicMock()
            mock_session_class.return_value = mock_session

            mock_response = MagicMock()
            mock_response.json.return_value = {
                "content": [{"type": "text", "text": "Hello!"}],
                "usage": {"input_tokens": 10, "output_tokens": 5}
            }
            mock_response.raise_for_status = MagicMock()
            mock_session.post.return_value = mock_response

            client = AnthropicChatClient(config, provider, timeout=30)

            convo = Conversation(
                id=None,
                provider_id="anthropic",
                model_name="claude-3-5-sonnet",
                temperature=0.7,
                system_prompt="Be helpful",
                messages=[Message(role="system", content="Be helpful")],
            )

            result = client.chat(convo, "Hi there", streaming=False)

            # Verify payload
            call_args = mock_session.post.call_args
            payload = call_args.kwargs.get('json') or call_args[1].get('json')

            self.assertEqual(payload["model"], "claude-3-5-sonnet")
            self.assertEqual(payload["system"], "Be helpful")
            self.assertEqual(payload["temperature"], 0.7)
            self.assertEqual(len(payload["messages"]), 1)  # Only user message
            self.assertEqual(payload["messages"][0]["content"], "Hi there")

            self.assertEqual(result, "Hello!")
            self.assertEqual(convo.tokens_in, 10)
            self.assertEqual(convo.tokens_out, 5)

    @patch('scriptchat.core.anthropic_client.requests.Session')
    def test_chat_with_thinking_includes_thinking_param(self, mock_session_class):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = make_config(Path(tmpdir))
            provider = config.providers[0]

            mock_session = MagicMock()
            mock_session_class.return_value = mock_session

            mock_response = MagicMock()
            mock_response.json.return_value = {
                "content": [{"type": "text", "text": "Thought about it!"}],
                "usage": {"input_tokens": 100, "output_tokens": 50}
            }
            mock_response.raise_for_status = MagicMock()
            mock_session.post.return_value = mock_response

            client = AnthropicChatClient(config, provider, timeout=30)

            convo = Conversation(
                id=None,
                provider_id="anthropic",
                model_name="claude-3-5-sonnet",
                temperature=0.5,
                thinking_budget=16000,
            )

            client.chat(convo, "Think hard", streaming=False)

            call_args = mock_session.post.call_args
            payload = call_args.kwargs.get('json') or call_args[1].get('json')

            self.assertIn("thinking", payload)
            self.assertEqual(payload["thinking"]["type"], "enabled")
            self.assertEqual(payload["thinking"]["budget_tokens"], 16000)


if __name__ == "__main__":
    unittest.main()
