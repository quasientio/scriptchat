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

import json
import os
import tempfile
import unittest
from pathlib import Path

import requests

from scriptchat.core.config import Config, ProviderConfig, load_config
from scriptchat.core.conversations import Conversation
from scriptchat.core.openai_client import OpenAIChatClient


class OpenAIClientSmokeTest(unittest.TestCase):
    def test_provider_id_mismatch_raises(self):
        config_text = """
[general]
default_provider = "openai"
default_model = "gpt-4o"
conversations_dir = "{conv}"

[[providers]]
id = "openai"
type = "openai-compatible"
api_url = "https://api.openai.com"
api_key = "sk-test"
models = "gpt-4o"
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["HOME"] = tmpdir
            conv_dir = Path(tmpdir) / "conversations"
            cfg_dir = Path(tmpdir) / ".scriptchat"
            cfg_dir.mkdir()
            (cfg_dir / "config.toml").write_text(config_text.format(conv=conv_dir.as_posix()), encoding="utf-8")
            cfg = load_config()
            provider = cfg.get_provider("openai")
            client = OpenAIChatClient(cfg, provider, timeout=1)
            convo = Conversation(
                id=None,
                provider_id="other",
                model_name="gpt-4o",
                temperature=0.7,
                messages=[],
                tokens_in=0,
                tokens_out=0
            )
            with self.assertRaises(ValueError):
                client.chat(convo, "hi")

    def test_retries_without_temperature_on_temperature_error(self):
        provider = ProviderConfig(
            id="openai",
            type="openai-compatible",
            api_url="https://api.openai.com",
            api_key="sk-test",
            models=[],
            streaming=True,
            headers={},
            default_model="gpt-4o",
        )
        cfg = Config(
            api_url="https://api.openai.com",
            api_key="sk-test",
            conversations_dir=Path("."),
            exports_dir=None,
            enable_streaming=False,
            system_prompt=None,
            default_provider="openai",
            default_model="gpt-4o",
            default_temperature=0.7,
            timeout=1,
            file_confirm_threshold_bytes=40_000,
            log_level="INFO",
            log_file=None,
            providers=[provider],
        )

        class FakeResponse:
            def __init__(self, status_code, payload, reason=""):
                self.status_code = status_code
                self._payload = payload
                self.reason = reason
                self.text = json.dumps(payload)

            def json(self):
                return self._payload

            def raise_for_status(self):
                if self.status_code >= 400:
                    raise requests.HTTPError("err", response=self)

            def iter_lines(self):
                return iter([])

        class FakeSession:
            def __init__(self):
                self.calls = []

            def post(self, url, json=None, timeout=None, stream=False):
                self.calls.append({"url": url, "json": json, "stream": stream})
                if len(self.calls) == 1:
                    return FakeResponse(400, {"error": {"param": "temperature", "message": "bad temperature"}}, reason="Bad Request")
                return FakeResponse(
                    200,
                    {
                        "choices": [{"message": {"content": "ok"}}],
                        "usage": {"prompt_tokens": 3, "completion_tokens": 2},
                    },
                    reason="OK",
                )

        convo = Conversation(
            id=None,
            provider_id="openai",
            model_name="gpt-4o",
            temperature=0.7,
            messages=[],
            tokens_in=0,
            tokens_out=0,
        )
        client = OpenAIChatClient(cfg, provider, timeout=1)
        client.session = FakeSession()

        reply = client.chat(convo, "hello")
        self.assertEqual(reply, "ok")
        self.assertEqual(len(client.session.calls), 2)  # retried without temperature
        self.assertEqual(convo.tokens_in, 3)
        self.assertEqual(convo.tokens_out, 2)
        # Second payload should omit temperature
        self.assertNotIn("temperature", client.session.calls[-1]["json"])

    def test_http_error_without_temperature_retry_surfaces(self):
        provider = ProviderConfig(
            id="openai",
            type="openai-compatible",
            api_url="https://api.openai.com",
            api_key="sk-test",
            models=[],
            streaming=True,
            headers={},
            default_model="gpt-4o",
        )
        cfg = Config(
            api_url="https://api.openai.com",
            api_key="sk-test",
            conversations_dir=Path("."),
            exports_dir=None,
            enable_streaming=False,
            system_prompt=None,
            default_provider="openai",
            default_model="gpt-4o",
            default_temperature=0.7,
            timeout=1,
            file_confirm_threshold_bytes=40_000,
            log_level="INFO",
            log_file=None,
            providers=[provider],
        )

        class FailSession:
            def post(self, url, json=None, timeout=None, stream=False):
                class Resp:
                    status_code = 500
                    reason = "Server Error"
                    text = "fail"

                    def json(self_inner):
                        return {"error": {"message": "boom"}}

                    def raise_for_status(self_inner):
                        raise requests.HTTPError("fail", response=self_inner)

                    def iter_lines(self_inner):
                        return iter([])

                return Resp()

        convo = Conversation(
            id=None,
            provider_id="openai",
            model_name="gpt-4o",
            temperature=0.7,
            messages=[],
            tokens_in=0,
            tokens_out=0,
        )
        client = OpenAIChatClient(cfg, provider, timeout=1)
        client.session = FailSession()

        with self.assertRaises(RuntimeError):
            client.chat(convo, "hi")

    def test_streaming_appends_chunks_and_usage(self):
        provider = ProviderConfig(
            id="openai",
            type="openai-compatible",
            api_url="https://api.openai.com",
            api_key="sk-test",
            models=[],
            streaming=True,
            headers={},
            default_model="gpt-4o",
        )
        cfg = Config(
            api_url="https://api.openai.com",
            api_key="sk-test",
            conversations_dir=Path("."),
            exports_dir=None,
            enable_streaming=False,
            system_prompt=None,
            default_provider="openai",
            default_model="gpt-4o",
            default_temperature=0.7,
            timeout=1,
            file_confirm_threshold_bytes=40_000,
            log_level="INFO",
            log_file=None,
            providers=[provider],
        )

        class StreamResponse:
            status_code = 200
            reason = "OK"

            def __init__(self, lines):
                self._lines = lines
                self.text = ""

            def raise_for_status(self):
                return None

            def iter_lines(self):
                return iter(self._lines)

            def json(self):
                return {}

        class StreamSession:
            def __init__(self, lines):
                self.lines = lines

            def post(self, url, json=None, timeout=None, stream=False):
                return StreamResponse(self.lines)

        lines = [
            b'data: {"choices": [{"delta": {"content": "Hel"}}], "usage": {"prompt_tokens": 1, "completion_tokens": 0}}\n',
            b'data: {"choices": [{"delta": {"content": "lo"}}], "usage": {"prompt_tokens": 1, "completion_tokens": 2}}\n',
            b"[DONE]",
        ]

        convo = Conversation(
            id=None,
            provider_id="openai",
            model_name="gpt-4o",
            temperature=0.7,
            messages=[],
            tokens_in=0,
            tokens_out=0,
        )
        client = OpenAIChatClient(cfg, provider, timeout=1)
        client.session = StreamSession(lines)

        collected = []

        def on_chunk(text):
            collected.append(text)

        content = client.chat(convo, "stream me", streaming=True, on_chunk=on_chunk)
        self.assertEqual(content, "Hello")
        self.assertEqual(collected[-1], "Hello")
        # Each chunk included usage; total prompt/completion accumulate
        self.assertEqual(convo.tokens_in, 2)
        self.assertEqual(convo.tokens_out, 2)

    def test_reasoning_effort_included_when_set(self):
        provider = ProviderConfig(
            id="openai",
            type="openai-compatible",
            api_url="https://api.openai.com",
            api_key="sk-test",
            models=[],
            streaming=True,
            headers={},
            default_model="gpt-5.1",
        )
        cfg = Config(
            api_url="https://api.openai.com",
            api_key="sk-test",
            conversations_dir=Path("."),
            exports_dir=None,
            enable_streaming=False,
            system_prompt=None,
            default_provider="openai",
            default_model="gpt-5.1",
            default_temperature=0.7,
            timeout=1,
            file_confirm_threshold_bytes=40_000,
            log_level="INFO",
            log_file=None,
            providers=[provider],
        )

        class CaptureSession:
            def __init__(self):
                self.calls = []

            def post(self, url, json=None, timeout=None, stream=False):
                self.calls.append({"url": url, "json": json, "timeout": timeout, "stream": stream})

                class Resp:
                    status_code = 200
                    reason = "OK"
                    text = "{}"

                    def raise_for_status(self_inner):
                        return None

                    def json(self_inner):
                        return {
                            "choices": [{"message": {"content": "done"}}],
                            "usage": {"prompt_tokens": 1, "completion_tokens": 2},
                        }

                    def iter_lines(self_inner):
                        return iter([])

                return Resp()

        convo = Conversation(
            id=None,
            provider_id="openai",
            model_name="gpt-5.1",
            temperature=0.7,
            reasoning_level="medium",
            messages=[],
            tokens_in=0,
            tokens_out=0,
        )
        client = OpenAIChatClient(cfg, provider, timeout=1)
        client.session = CaptureSession()

        reply = client.chat(convo, "hello")
        self.assertEqual(reply, "done")
        self.assertIn("reasoning", client.session.calls[0]["json"])
        self.assertEqual(client.session.calls[0]["json"]["reasoning"], {"effort": "medium"})
        self.assertTrue(client.session.calls[0]["url"].endswith("/v1/responses"))

    def test_responses_api_streaming_and_nonstream_content_extraction(self):
        provider = ProviderConfig(
            id="openai",
            type="openai-compatible",
            api_url="https://api.openai.com",
            api_key="sk-test",
            models=[],
            streaming=True,
            headers={},
            default_model="gpt-5.1",
        )
        cfg = Config(
            api_url="https://api.openai.com",
            api_key="sk-test",
            conversations_dir=Path("."),
            exports_dir=None,
            enable_streaming=True,
            system_prompt=None,
            default_provider="openai",
            default_model="gpt-5.1",
            default_temperature=0.7,
            timeout=1,
            file_confirm_threshold_bytes=40_000,
            log_level="INFO",
            log_file=None,
            providers=[provider],
        )

        class StreamResponse:
            status_code = 200
            reason = "OK"

            def __init__(self, lines):
                self.lines = lines
                self.text = ""

            def raise_for_status(self):
                return None

            def iter_lines(self):
                return iter(self.lines)

            def json(self):
                return {}

        class StreamSession:
            def __init__(self, lines):
                self.lines = lines
                self.calls = []

            def post(self, url, json=None, timeout=None, stream=False):
                self.calls.append({"url": url, "json": json, "stream": stream})
                if stream:
                    return StreamResponse(self.lines)

                class Resp:
                    status_code = 200
                    reason = "OK"
                    text = "{}"

                    def raise_for_status(self_inner):
                        return None

                    def json(self_inner):
                        return {
                            "output_text": ["nonstream text"],
                            "usage": {"prompt_tokens": 1, "completion_tokens": 2},
                        }

                    def iter_lines(self_inner):
                        return iter([])

                return Resp()

        # Streaming path
        lines = [
            b'data: {"output_text": ["Hel"]}\n',
            b'data: {"output_text": ["lo"]}\n',
            b"[DONE]",
        ]

        convo = Conversation(
            id=None,
            provider_id="openai",
            model_name="gpt-5.1",
            temperature=0.7,
            messages=[],
            tokens_in=0,
            tokens_out=0,
        )
        client = OpenAIChatClient(cfg, provider, timeout=1)
        client.session = StreamSession(lines)

        collected = []

        def on_chunk(text):
            collected.append(text)

        streamed = client.chat(convo, "hi", streaming=True, on_chunk=on_chunk)
        self.assertEqual(streamed, "Hello")
        self.assertEqual(collected[-1], "Hello")

        # Non-streaming path uses same session with a fresh conversation
        convo2 = Conversation(
            id=None,
            provider_id="openai",
            model_name="gpt-5.1",
            temperature=0.7,
            messages=[],
            tokens_in=0,
            tokens_out=0,
        )
        non_stream = client.chat(convo2, "hi", streaming=False)
        self.assertEqual(non_stream, "nonstream text")

    def test_runtime_timeout_override_used_for_requests(self):
        provider = ProviderConfig(
            id="openai",
            type="openai-compatible",
            api_url="https://api.openai.com",
            api_key="sk-test",
            models=[],
            streaming=True,
            headers={},
            default_model="gpt-4o",
        )
        cfg = Config(
            api_url="https://api.openai.com",
            api_key="sk-test",
            conversations_dir=Path("."),
            exports_dir=None,
            enable_streaming=False,
            system_prompt=None,
            default_provider="openai",
            default_model="gpt-4o",
            default_temperature=0.7,
            timeout=1,
            file_confirm_threshold_bytes=40_000,
            log_level="INFO",
            log_file=None,
            providers=[provider],
        )

        class RecordingSession:
            def __init__(self):
                self.timeouts = []

            def post(self, url, json=None, timeout=None, stream=False):
                self.timeouts.append(timeout)

                class Resp:
                    status_code = 200
                    reason = "OK"
                    text = ""

                    def json(self_inner):
                        return {"choices": [{"message": {"content": "ok"}}], "usage": {}}

                    def raise_for_status(self_inner):
                        return None

                    def iter_lines(self_inner):
                        return iter([])

                return Resp()

        convo = Conversation(
            id=None,
            provider_id="openai",
            model_name="gpt-4o",
            temperature=0.7,
            messages=[],
            tokens_in=0,
            tokens_out=0,
        )
        client = OpenAIChatClient(cfg, provider, timeout=1)
        client.session = RecordingSession()

        cfg.timeout = 7

        reply = client.chat(convo, "hi")
        self.assertEqual(reply, "ok")
        self.assertEqual(client.session.timeouts, [7])

    def test_timeout_errors_are_wrapped(self):
        provider = ProviderConfig(
            id="openai",
            type="openai-compatible",
            api_url="https://api.openai.com",
            api_key="sk-test",
            models=[],
            streaming=True,
            headers={},
            default_model="gpt-4o",
        )
        cfg = Config(
            api_url="https://api.openai.com",
            api_key="sk-test",
            conversations_dir=Path("."),
            exports_dir=None,
            enable_streaming=False,
            system_prompt=None,
            default_provider="openai",
            default_model="gpt-4o",
            default_temperature=0.7,
            timeout=2,
            file_confirm_threshold_bytes=40_000,
            log_level="INFO",
            log_file=None,
            providers=[provider],
        )

        class TimeoutSession:
            def post(self, url, json=None, timeout=None, stream=False):
                raise requests.Timeout("boom")

        convo = Conversation(
            id=None,
            provider_id="openai",
            model_name="gpt-4o",
            temperature=0.7,
            messages=[],
            tokens_in=0,
            tokens_out=0,
        )
        client = OpenAIChatClient(cfg, provider, timeout=1)
        client.session = TimeoutSession()

        with self.assertRaises(TimeoutError) as ctx:
            client.chat(convo, "hi")
        self.assertIn("Request timed out after 2 seconds", str(ctx.exception))

    def test_connection_errors_are_wrapped(self):
        provider = ProviderConfig(
            id="openai",
            type="openai-compatible",
            api_url="https://api.openai.com",
            api_key="sk-test",
            models=[],
            streaming=True,
            headers={},
            default_model="gpt-4o",
        )
        cfg = Config(
            api_url="https://api.openai.com",
            api_key="sk-test",
            conversations_dir=Path("."),
            exports_dir=None,
            enable_streaming=False,
            system_prompt=None,
            default_provider="openai",
            default_model="gpt-4o",
            default_temperature=0.7,
            timeout=2,
            file_confirm_threshold_bytes=40_000,
            log_level="INFO",
            log_file=None,
            providers=[provider],
        )

        class FailSession:
            def post(self, url, json=None, timeout=None, stream=False):
                raise requests.ConnectionError("nope")

        convo = Conversation(
            id=None,
            provider_id="openai",
            model_name="gpt-4o",
            temperature=0.7,
            messages=[],
            tokens_in=0,
            tokens_out=0,
        )
        client = OpenAIChatClient(cfg, provider, timeout=1)
        client.session = FailSession()

        with self.assertRaises(ConnectionError) as ctx:
            client.chat(convo, "hi")
        self.assertIn("Failed to connect to provider at https://api.openai.com", str(ctx.exception))
        self.assertIn("/timeout", str(ctx.exception))


    def test_api_format_responses_uses_responses_endpoint_with_store_false(self):
        """When api_format='responses', should use /v1/responses with store=false."""
        provider = ProviderConfig(
            id="fireworks",
            type="openai-compatible",
            api_url="https://api.fireworks.ai/inference",
            api_key="test-key",
            models=[],
            streaming=True,
            headers={},
            default_model="test-model",
            api_format="responses",
        )
        cfg = Config(
            api_url="https://api.fireworks.ai/inference",
            api_key="test-key",
            conversations_dir=Path("."),
            exports_dir=None,
            enable_streaming=False,
            system_prompt=None,
            default_provider="fireworks",
            default_model="test-model",
            default_temperature=0.7,
            timeout=1,
            file_confirm_threshold_bytes=40_000,
            log_level="INFO",
            log_file=None,
            providers=[provider],
        )

        class CaptureSession:
            def __init__(self):
                self.calls = []

            def post(self, url, json=None, timeout=None, stream=False):
                self.calls.append({"url": url, "json": json})

                class Resp:
                    status_code = 200
                    reason = "OK"
                    text = "{}"

                    def raise_for_status(self_inner):
                        return None

                    def json(self_inner):
                        return {
                            "output_text": ["response"],
                            "usage": {"input_tokens": 1, "output_tokens": 2},
                        }

                    def iter_lines(self_inner):
                        return iter([])

                return Resp()

        convo = Conversation(
            id=None,
            provider_id="fireworks",
            model_name="test-model",
            temperature=0.7,
            messages=[],
            tokens_in=0,
            tokens_out=0,
        )
        client = OpenAIChatClient(cfg, provider, timeout=1)
        client.session = CaptureSession()

        client.chat(convo, "hello")

        self.assertTrue(client.session.calls[0]["url"].endswith("/v1/responses"))
        self.assertIn("store", client.session.calls[0]["json"])
        self.assertEqual(client.session.calls[0]["json"]["store"], False)
        self.assertIn("input", client.session.calls[0]["json"])

    def test_api_format_chat_uses_chat_completions_endpoint(self):
        """When api_format='chat', should use /v1/chat/completions without store."""
        provider = ProviderConfig(
            id="fireworks",
            type="openai-compatible",
            api_url="https://api.fireworks.ai/inference",
            api_key="test-key",
            models=[],
            streaming=True,
            headers={},
            default_model="test-model",
            api_format="chat",
        )
        cfg = Config(
            api_url="https://api.fireworks.ai/inference",
            api_key="test-key",
            conversations_dir=Path("."),
            exports_dir=None,
            enable_streaming=False,
            system_prompt=None,
            default_provider="fireworks",
            default_model="test-model",
            default_temperature=0.7,
            timeout=1,
            file_confirm_threshold_bytes=40_000,
            log_level="INFO",
            log_file=None,
            providers=[provider],
        )

        class CaptureSession:
            def __init__(self):
                self.calls = []

            def post(self, url, json=None, timeout=None, stream=False):
                self.calls.append({"url": url, "json": json})

                class Resp:
                    status_code = 200
                    reason = "OK"
                    text = "{}"

                    def raise_for_status(self_inner):
                        return None

                    def json(self_inner):
                        return {
                            "choices": [{"message": {"content": "response"}}],
                            "usage": {"prompt_tokens": 1, "completion_tokens": 2},
                        }

                    def iter_lines(self_inner):
                        return iter([])

                return Resp()

        convo = Conversation(
            id=None,
            provider_id="fireworks",
            model_name="test-model",
            temperature=0.7,
            messages=[],
            tokens_in=0,
            tokens_out=0,
        )
        client = OpenAIChatClient(cfg, provider, timeout=1)
        client.session = CaptureSession()

        client.chat(convo, "hello")

        self.assertTrue(client.session.calls[0]["url"].endswith("/v1/chat/completions"))
        self.assertNotIn("store", client.session.calls[0]["json"])
        self.assertIn("messages", client.session.calls[0]["json"])

    def test_api_format_unset_non_openai_uses_chat_completions(self):
        """When api_format is not set and id != 'openai', should use chat completions."""
        provider = ProviderConfig(
            id="deepseek",
            type="openai-compatible",
            api_url="https://api.deepseek.com",
            api_key="test-key",
            models=[],
            streaming=True,
            headers={},
            default_model="deepseek-chat",
            # api_format not set
        )
        cfg = Config(
            api_url="https://api.deepseek.com",
            api_key="test-key",
            conversations_dir=Path("."),
            exports_dir=None,
            enable_streaming=False,
            system_prompt=None,
            default_provider="deepseek",
            default_model="deepseek-chat",
            default_temperature=0.7,
            timeout=1,
            file_confirm_threshold_bytes=40_000,
            log_level="INFO",
            log_file=None,
            providers=[provider],
        )

        class CaptureSession:
            def __init__(self):
                self.calls = []

            def post(self, url, json=None, timeout=None, stream=False):
                self.calls.append({"url": url, "json": json})

                class Resp:
                    status_code = 200
                    reason = "OK"
                    text = "{}"

                    def raise_for_status(self_inner):
                        return None

                    def json(self_inner):
                        return {
                            "choices": [{"message": {"content": "response"}}],
                            "usage": {"prompt_tokens": 1, "completion_tokens": 2},
                        }

                    def iter_lines(self_inner):
                        return iter([])

                return Resp()

        convo = Conversation(
            id=None,
            provider_id="deepseek",
            model_name="deepseek-chat",
            temperature=0.7,
            messages=[],
            tokens_in=0,
            tokens_out=0,
        )
        client = OpenAIChatClient(cfg, provider, timeout=1)
        client.session = CaptureSession()

        client.chat(convo, "hello")

        self.assertTrue(client.session.calls[0]["url"].endswith("/v1/chat/completions"))
        self.assertNotIn("store", client.session.calls[0]["json"])

    def test_prompt_cache_false_adds_prompt_cache_max_len_zero(self):
        """When prompt_cache=false at provider level, should add prompt_cache_max_len=0 to payload."""
        provider = ProviderConfig(
            id="fireworks",
            type="openai-compatible",
            api_url="https://api.fireworks.ai/inference",
            api_key="test-key",
            models=[],
            streaming=True,
            headers={},
            default_model="test-model",
            prompt_cache=False,
        )
        cfg = Config(
            api_url="https://api.fireworks.ai/inference",
            api_key="test-key",
            conversations_dir=Path("."),
            exports_dir=None,
            enable_streaming=False,
            system_prompt=None,
            default_provider="fireworks",
            default_model="test-model",
            default_temperature=0.7,
            timeout=1,
            file_confirm_threshold_bytes=40_000,
            log_level="INFO",
            log_file=None,
            providers=[provider],
        )

        class CaptureSession:
            def __init__(self):
                self.calls = []

            def post(self, url, json=None, timeout=None, stream=False):
                self.calls.append({"url": url, "json": json})

                class Resp:
                    status_code = 200
                    reason = "OK"
                    text = "{}"

                    def raise_for_status(self_inner):
                        return None

                    def json(self_inner):
                        return {
                            "choices": [{"message": {"content": "response"}}],
                            "usage": {"prompt_tokens": 1, "completion_tokens": 2},
                        }

                    def iter_lines(self_inner):
                        return iter([])

                return Resp()

        convo = Conversation(
            id=None,
            provider_id="fireworks",
            model_name="test-model",
            temperature=0.7,
            messages=[],
            tokens_in=0,
            tokens_out=0,
        )
        client = OpenAIChatClient(cfg, provider, timeout=1)
        client.session = CaptureSession()

        client.chat(convo, "hello")

        self.assertIn("prompt_cache_max_len", client.session.calls[0]["json"])
        self.assertEqual(client.session.calls[0]["json"]["prompt_cache_max_len"], 0)

    def test_prompt_cache_true_default_does_not_add_prompt_cache_max_len(self):
        """When prompt_cache=true (default), should not add prompt_cache_max_len."""
        provider = ProviderConfig(
            id="fireworks",
            type="openai-compatible",
            api_url="https://api.fireworks.ai/inference",
            api_key="test-key",
            models=[],
            streaming=True,
            headers={},
            default_model="test-model",
            # prompt_cache defaults to True
        )
        cfg = Config(
            api_url="https://api.fireworks.ai/inference",
            api_key="test-key",
            conversations_dir=Path("."),
            exports_dir=None,
            enable_streaming=False,
            system_prompt=None,
            default_provider="fireworks",
            default_model="test-model",
            default_temperature=0.7,
            timeout=1,
            file_confirm_threshold_bytes=40_000,
            log_level="INFO",
            log_file=None,
            providers=[provider],
        )

        class CaptureSession:
            def __init__(self):
                self.calls = []

            def post(self, url, json=None, timeout=None, stream=False):
                self.calls.append({"url": url, "json": json})

                class Resp:
                    status_code = 200
                    reason = "OK"
                    text = "{}"

                    def raise_for_status(self_inner):
                        return None

                    def json(self_inner):
                        return {
                            "choices": [{"message": {"content": "response"}}],
                            "usage": {"prompt_tokens": 1, "completion_tokens": 2},
                        }

                    def iter_lines(self_inner):
                        return iter([])

                return Resp()

        convo = Conversation(
            id=None,
            provider_id="fireworks",
            model_name="test-model",
            temperature=0.7,
            messages=[],
            tokens_in=0,
            tokens_out=0,
        )
        client = OpenAIChatClient(cfg, provider, timeout=1)
        client.session = CaptureSession()

        client.chat(convo, "hello")

        self.assertNotIn("prompt_cache_max_len", client.session.calls[0]["json"])

    def test_streaming_handles_empty_choices_gracefully(self):
        """When streaming response has empty choices array, should not raise IndexError."""
        provider = ProviderConfig(
            id="fireworks",
            type="openai-compatible",
            api_url="https://api.fireworks.ai/inference",
            api_key="test-key",
            models=[],
            streaming=True,
            headers={},
            default_model="test-model",
        )
        cfg = Config(
            api_url="https://api.fireworks.ai/inference",
            api_key="test-key",
            conversations_dir=Path("."),
            exports_dir=None,
            enable_streaming=False,
            system_prompt=None,
            default_provider="fireworks",
            default_model="test-model",
            default_temperature=0.7,
            timeout=1,
            file_confirm_threshold_bytes=40_000,
            log_level="INFO",
            log_file=None,
            providers=[provider],
        )

        class StreamResponse:
            status_code = 200
            reason = "OK"
            text = ""

            def __init__(self, lines):
                self._lines = lines

            def raise_for_status(self):
                return None

            def iter_lines(self):
                return iter(self._lines)

            def json(self):
                return {}

        class StreamSession:
            def __init__(self, lines):
                self.lines = lines

            def post(self, url, json=None, timeout=None, stream=False):
                return StreamResponse(self.lines)

        # Simulate a response with empty choices (which can happen with some providers)
        lines = [
            b'data: {"choices": [{"delta": {"content": "Hello"}}]}\n',
            b'data: {"choices": []}\n',  # Empty choices - should be handled gracefully
            b'data: {"choices": [{"delta": {"content": " world"}}]}\n',
            b'data: {"usage": {"prompt_tokens": 10, "completion_tokens": 5}}\n',
            b"[DONE]",
        ]

        convo = Conversation(
            id=None,
            provider_id="fireworks",
            model_name="test-model",
            temperature=0.7,
            messages=[],
            tokens_in=0,
            tokens_out=0,
        )
        client = OpenAIChatClient(cfg, provider, timeout=1)
        client.session = StreamSession(lines)

        # This should not raise IndexError
        content = client.chat(convo, "hello", streaming=True)
        self.assertEqual(content, "Hello world")


class ExtractThinkTagsTests(unittest.TestCase):
    """Tests for _extract_think_tags method."""

    def _make_client(self):
        """Create a minimal client for testing."""
        provider = ProviderConfig(
            id="test",
            type="openai-compatible",
            api_url="https://api.test.com",
            api_key="test-key",
            models=[],
            streaming=True,
            headers={},
            default_model="test-model",
        )
        cfg = Config(
            api_url="https://api.test.com",
            api_key="test-key",
            conversations_dir=Path("."),
            exports_dir=None,
            enable_streaming=False,
            system_prompt=None,
            default_provider="test",
            default_model="test-model",
            default_temperature=0.7,
            timeout=1,
            file_confirm_threshold_bytes=40_000,
            log_level="INFO",
            log_file=None,
            providers=[provider],
        )
        return OpenAIChatClient(cfg, provider, timeout=1)

    def test_extract_think_tags_deepseek_format(self):
        """Test extraction of <think>...</think> tags (DeepSeek R1 format)."""
        client = self._make_client()
        content = "<think>I need to analyze this carefully.</think>\n\nThe answer is 42."
        result, thinking = client._extract_think_tags(content)
        self.assertEqual(result, "The answer is 42.")
        self.assertEqual(thinking, "I need to analyze this carefully.")

    def test_extract_thinking_tags_kimi_format(self):
        """Test extraction of <thinking>...</thinking> tags (Kimi format)."""
        client = self._make_client()
        content = "<thinking>Let me reason through this step by step.</thinking>\n\n# Plan\n\n1. First step\n2. Second step"
        result, thinking = client._extract_think_tags(content)
        self.assertEqual(result, "# Plan\n\n1. First step\n2. Second step")
        self.assertEqual(thinking, "Let me reason through this step by step.")

    def test_extract_think_tags_multiline(self):
        """Test extraction with multiline thinking content."""
        client = self._make_client()
        content = """<thinking>
Line 1 of thinking
Line 2 of thinking
Line 3 of thinking
</thinking>

Here is the response."""
        result, thinking = client._extract_think_tags(content)
        self.assertEqual(result, "Here is the response.")
        self.assertIn("Line 1 of thinking", thinking)
        self.assertIn("Line 3 of thinking", thinking)

    def test_extract_think_tags_no_tags(self):
        """Test that content without think tags is returned unchanged."""
        client = self._make_client()
        content = "Just a regular response without thinking."
        result, thinking = client._extract_think_tags(content)
        self.assertEqual(result, content)
        self.assertIsNone(thinking)

    def test_extract_think_tags_case_insensitive(self):
        """Test that tag matching is case insensitive."""
        client = self._make_client()
        content = "<THINKING>Uppercase tags</THINKING>\n\nResponse."
        result, thinking = client._extract_think_tags(content)
        self.assertEqual(result, "Response.")
        self.assertEqual(thinking, "Uppercase tags")

    def test_extract_think_tags_empty_thinking(self):
        """Test extraction with empty thinking block."""
        client = self._make_client()
        content = "<think></think>Response."
        result, thinking = client._extract_think_tags(content)
        self.assertEqual(result, "Response.")
        self.assertEqual(thinking, "")


if __name__ == "__main__":
    unittest.main()
