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

import sys
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scriptchat import __main__
from scriptchat.core.config import Config, ModelConfig, ProviderConfig
from scriptchat.core.conversations import Conversation


def make_config(tmp_path: Path):
    provider = ProviderConfig(
        id="ollama",
        type="ollama",
        api_url="http://localhost:11434/api",
        api_key="",
        models=[ModelConfig(name="llama3", context=512)],
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
        file_confirm_threshold_bytes=40_000,
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
                mock.patch.object(sys.stdin, "isatty", return_value=True),
            ):
                mock_ollama_client.return_value = object()
                dispatcher = DummyDispatcher()
                mock_dispatcher.return_value = dispatcher

                argv = sys.argv
                sys.argv = ["scriptchat"]  # no --run
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
                sys.argv = ["scriptchat"]
                with self.assertRaises(SystemExit) as exc:
                    __main__.main()
                sys.argv = argv
                self.assertEqual(exc.exception.code, 1)

    def test_main_runs_batch_from_stdin(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = make_config(Path(tmpdir))
            with (
                mock.patch.object(__main__, "load_config", return_value=cfg),
                mock.patch.object(__main__, "run_ui") as mock_run_ui,
                mock.patch.object(__main__, "run_batch_lines", return_value=0) as mock_run_lines,
                mock.patch.object(__main__, "OllamaChatClient") as mock_ollama_client,
                mock.patch.object(__main__, "ProviderDispatcher") as mock_dispatcher,
                mock.patch.object(sys.stdin, "isatty", return_value=True),
            ):
                mock_ollama_client.return_value = object()
                dispatcher = DummyDispatcher()
                mock_dispatcher.return_value = dispatcher

                argv = sys.argv
                stdin_backup = sys.stdin
                sys.argv = ["scriptchat"]
                sys.stdin = io.StringIO("hello\n")
                try:
                    with self.assertRaises(SystemExit) as exc:
                        __main__.main()
                finally:
                    sys.argv = argv
                    sys.stdin = stdin_backup

                self.assertEqual(exc.exception.code, 0)
                mock_run_ui.assert_not_called()
                mock_run_lines.assert_called_once()


if __name__ == "__main__":
    unittest.main()
