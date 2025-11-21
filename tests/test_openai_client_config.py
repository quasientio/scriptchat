import os
import tempfile
import unittest
from pathlib import Path

from litechat.core.config import load_config
from litechat.core.conversations import Conversation
from litechat.core.openai_client import OpenAIChatClient
from litechat.core.config import ProviderConfig


class DummyProvider:
    def __init__(self):
        self.id = "openai"
        self.type = "openai-compatible"
        self.api_url = "https://api.openai.com"
        self.api_key = "sk-test"
        self.models = []
        self.streaming = True
        self.headers = {}
        self.default_model = None


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


if __name__ == "__main__":
    unittest.main()
