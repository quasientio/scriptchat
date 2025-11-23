import json
import os
import tempfile
import unittest
from pathlib import Path

import requests

from litechat.core.config import Config, ProviderConfig, load_config
from litechat.core.conversations import Conversation
from litechat.core.openai_client import OpenAIChatClient


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
            cfg_dir = Path(tmpdir) / ".lite-chat"
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
        self.assertEqual(convo.tokens_in, 1)
        self.assertEqual(convo.tokens_out, 2)

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


if __name__ == "__main__":
    unittest.main()
