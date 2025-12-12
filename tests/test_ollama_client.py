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

import unittest
from pathlib import Path
from unittest.mock import patch

import requests

from scriptchat.core.config import Config, ModelConfig, ProviderConfig
from scriptchat.core.conversations import Conversation, Message
from scriptchat.core.ollama_client import OllamaChatClient, check_ollama_running


class OllamaClientTests(unittest.TestCase):
    def make_config(self, context=1024):
        provider = ProviderConfig(
            id="ollama",
            type="ollama",
            api_url="http://localhost:11434/api",
            api_key="",
            models=[ModelConfig(name="llama3", context=context)],
            streaming=True,
            headers={},
            default_model="llama3",
        )
        return Config(
            api_url="http://localhost:11434/api",
            api_key="",
            conversations_dir=Path("."),
            exports_dir=None,
            enable_streaming=False,
            system_prompt=None,
            default_provider="ollama",
            default_model="llama3",
            default_temperature=0.7,
            timeout=5,
            file_confirm_threshold_bytes=40_000,
            log_level="INFO",
            log_file=None,
            providers=[provider],
        )

    @patch('scriptchat.core.ollama_client.check_ollama_running', return_value=True)
    def test_chat_sends_request_and_updates_conversation(self, mock_check):
        cfg = self.make_config(context=4096)
        client = OllamaChatClient(cfg, "http://localhost:11434/api", interactive=False)
        convo = Conversation(
            id=None,
            provider_id="ollama",
            model_name="llama3",
            temperature=0.3,
            messages=[Message(role="user", content="hi")],
            tokens_in=0,
            tokens_out=0,
        )

        class FakeResponse:
            status_code = 200
            reason = "OK"

            def __init__(self):
                self.text = ""

            def json(self):
                return {
                    "message": {"content": "hello"},
                    "prompt_eval_count": 5,
                    "eval_count": 2,
                }

            def raise_for_status(self):
                return None

        class FakeSession:
            def __init__(self):
                self.posts = []

            def post(self, url, json=None, timeout=None, stream=False):
                self.posts.append({"url": url, "json": json, "timeout": timeout, "stream": stream})
                return FakeResponse()

        client.session = FakeSession()

        reply = client.chat(convo, "hello?", streaming=False)
        self.assertEqual(reply, "hello")
        self.assertEqual(convo.tokens_in, 5)
        self.assertEqual(convo.tokens_out, 2)
        self.assertEqual(convo.messages[-1].role, "assistant")
        self.assertEqual(convo.context_length_configured, 4096)
        self.assertEqual(convo.context_length_used, 5)
        # Verify num_ctx was sent in request
        self.assertEqual(client.session.posts[0]["json"]["options"]["num_ctx"], 4096)

    @patch('scriptchat.core.ollama_client.check_ollama_running', return_value=True)
    def test_chat_streaming_aggregates_chunks_and_tokens(self, mock_check):
        cfg = self.make_config(context=1024)
        client = OllamaChatClient(cfg, "http://localhost:11434/api", interactive=False)
        convo = Conversation(
            id=None,
            provider_id="ollama",
            model_name="llama3",
            temperature=0.7,
            messages=[],
            tokens_in=0,
            tokens_out=0,
        )

        class StreamResponse:
            status_code = 200
            reason = "OK"

            def __init__(self, lines):
                self.lines = lines

            def iter_lines(self, decode_unicode=True):
                return iter(self.lines)

            def raise_for_status(self):
                return None

            def json(self):
                return {}

        class StreamSession:
            def __init__(self, lines):
                self.lines = lines
                self.posts = []

            def post(self, url, json=None, timeout=None, stream=False):
                self.posts.append({"url": url, "json": json})
                return StreamResponse(self.lines)

        lines = [
            '{"message": {"content": "Hi "}, "done": false}',
            '{"message": {"content": "there"}, "done": false}',
            '{"done": true, "prompt_eval_count": 7, "eval_count": 3}',
        ]
        client.session = StreamSession(lines)

        chunks = []

        def on_chunk(text):
            chunks.append(text)

        content = client.chat(convo, "go", streaming=True, on_chunk=on_chunk)
        self.assertEqual(content, "Hi there")
        self.assertEqual(chunks[-1], "Hi there")
        self.assertEqual(convo.tokens_in, 7)
        self.assertEqual(convo.tokens_out, 3)
        self.assertEqual(convo.context_length_configured, 1024)
        # Verify num_ctx was sent in request
        self.assertEqual(client.session.posts[0]["json"]["options"]["num_ctx"], 1024)

    @patch('scriptchat.core.ollama_client.check_ollama_running', return_value=True)
    def test_invalid_provider_id_raises(self, mock_check):
        cfg = self.make_config()
        client = OllamaChatClient(cfg, "http://localhost:11434/api", interactive=False)
        convo = Conversation(
            id=None,
            provider_id="other",
            model_name="llama3",
            temperature=0.7,
            messages=[],
            tokens_in=0,
            tokens_out=0,
        )
        with self.assertRaises(ValueError):
            client.chat(convo, "hi")

    @patch('scriptchat.core.ollama_client.check_ollama_running', return_value=True)
    def test_chat_single_http_error_and_unload(self, mock_check):
        cfg = self.make_config()
        client = OllamaChatClient(cfg, "http://localhost:11434/api", interactive=False)
        convo = Conversation(
            id=None,
            provider_id="ollama",
            model_name="llama3",
            temperature=0.7,
            messages=[],
            tokens_in=0,
            tokens_out=0,
        )

        class FailResponse:
            status_code = 500
            reason = "oops"
            text = "error body"

            def json(self):
                return {"detail": "boom"}

            def raise_for_status(self):
                raise requests.HTTPError("fail", response=self)

        class FailSession:
            def __init__(self):
                self.unload_calls = 0

            def post(self, *args, **kwargs):
                return FailResponse()

        client.session = FailSession()

        with self.assertRaises(RuntimeError):
            client.chat(convo, "hi")

        # Unload should post even if errors are ignored
        client.current_model = "llama3"
        calls = []

        class UnloadSession:
            def post(self, url, json=None, timeout=None):
                calls.append((url, json))
                return FailResponse()

        client.session = UnloadSession()
        client.unload_model()
        self.assertTrue(calls)

    def test_check_ollama_running_returns_true_on_200(self):
        with patch('scriptchat.core.ollama_client.requests.get') as mock_get:
            mock_get.return_value.status_code = 200
            result = check_ollama_running("http://localhost:11434/api")
            self.assertTrue(result)
            mock_get.assert_called_once_with("http://localhost:11434/api/tags", timeout=2.0)

    def test_check_ollama_running_returns_false_on_error(self):
        with patch('scriptchat.core.ollama_client.requests.get') as mock_get:
            mock_get.side_effect = requests.ConnectionError("refused")
            result = check_ollama_running("http://localhost:11434/api")
            self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
