import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from litechat import __main__
from litechat.core.config import Config, ModelConfig, ProviderConfig
from litechat.core.conversations import Conversation


def make_config(tmp_path: Path):
    provider = ProviderConfig(
        id="ollama",
        type="ollama",
        api_url="http://localhost:11434/api",
        api_key="",
        models=[ModelConfig(name="llama3", contexts=[512])],
        streaming=True,
        headers={},
        default_model="llama3",
    )
    return Config(
        api_url="http://localhost:11434/api",
        api_key="",
        conversations_dir=tmp_path,
        exports_dir=tmp_path,
        enable_streaming=False,
        system_prompt=None,
        default_provider="ollama",
        default_model="llama3",
        default_temperature=0.7,
        timeout=10,
        log_level="INFO",
        log_file=None,
        providers=[provider],
    )


class DummyDispatcher:
    def __init__(self, clients=None):
        self.clients = clients or {}
        self.cleaned = False

    def cleanup(self):
        self.cleaned = True


class MainEntrypointTests(unittest.TestCase):
    def test_main_runs_with_mocked_dependencies(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = make_config(Path(tmpdir))
            # patch load_config, run_ui, ProviderDispatcher, OllamaChatClient
            with (
                mock.patch.object(__main__, "load_config", return_value=cfg),
                mock.patch.object(__main__, "run_ui", return_value=None),
                mock.patch.object(__main__, "OllamaChatClient") as mock_ollama_client,
                mock.patch.object(__main__, "ProviderDispatcher") as mock_dispatcher,
            ):
                mock_ollama_client.return_value = object()
                dispatcher = DummyDispatcher()
                mock_dispatcher.return_value = dispatcher

                argv = sys.argv
                sys.argv = ["litechat"]  # no --run
                try:
                    __main__.main()
                finally:
                    sys.argv = argv

                self.assertTrue(dispatcher.cleaned or dispatcher.clients == {})

    def test_main_errors_on_missing_default_provider(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = make_config(Path(tmpdir))
            cfg.default_provider = "missing"
            with mock.patch.object(__main__, "load_config", return_value=cfg):
                argv = sys.argv
                sys.argv = ["litechat"]
                with self.assertRaises(SystemExit) as exc:
                    __main__.main()
                sys.argv = argv
                self.assertEqual(exc.exception.code, 1)


if __name__ == "__main__":
    unittest.main()
