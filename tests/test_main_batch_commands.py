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

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from scriptchat.__main__ import handle_batch_command
from scriptchat.core.commands import AppState
from scriptchat.core.config import Config, ModelConfig, ProviderConfig
from scriptchat.core.conversations import Conversation


def make_state(tmp_path: Path):
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
    cfg = Config(
        api_url="http://localhost:11434/api",
        api_key="",
        conversations_dir=tmp_path,
        exports_dir=tmp_path,
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
    # compatibility for /model index lookup
    cfg.models = provider.models
    convo = Conversation(
        id=None,
        provider_id="ollama",
        model_name="llama3",
        temperature=0.7,
        messages=[],
        tokens_in=0,
        tokens_out=0,
    )
    return AppState(config=cfg, current_conversation=convo, client=None, conversations_root=tmp_path, file_registry={})


class MainBatchCommandBranches(unittest.TestCase):
    def test_invalid_temp_stream_and_export(self):
        with tempfile.TemporaryDirectory() as tmpdir, io.StringIO() as buf, redirect_stdout(buf):
            state = make_state(Path(tmpdir))
            handle_batch_command("/temp not-a-number", state, 1)
            handle_batch_command("/stream maybe", state, 2)
            handle_batch_command("/export txt", state, 3)
            handle_batch_command("/assert-not", state, 4)
            output = buf.getvalue()
        self.assertIn("Invalid temperature", output)
        self.assertIn("expects 'on' or 'off'", output)
        self.assertIn("format must be 'md', 'json', or 'html'", output)
        self.assertIn("requires a pattern", output)

    def test_prompt_and_sleep_branches(self):
        with tempfile.TemporaryDirectory() as tmpdir, io.StringIO() as buf, redirect_stdout(buf):
            state = make_state(Path(tmpdir))
            handle_batch_command("/prompt", state, 1)
            handle_batch_command("/sleep -1", state, 2)
            output = buf.getvalue()
        self.assertIn("requires an argument", output)
        self.assertIn("must be positive", output)

    def test_file_permission_and_invalid_command(self):
        with tempfile.TemporaryDirectory() as tmpdir, io.StringIO() as buf, redirect_stdout(buf):
            state = make_state(Path(tmpdir))
            protected = Path(tmpdir) / "protected.txt"
            protected.write_text("content", encoding="utf-8")
            try:
                protected.chmod(0o000)
                handle_batch_command(f"/file {protected}", state, 1)
            finally:
                # Restore permissions so tmpdir cleanup succeeds
                protected.chmod(0o600)
            handle_batch_command("/unknowncmd", state, 2)
            output = buf.getvalue()
        self.assertIn("Permission denied", output)
        self.assertIn("not supported", output)

    def test_timeout_command_branch(self):
        with tempfile.TemporaryDirectory() as tmpdir, io.StringIO() as buf, redirect_stdout(buf):
            state = make_state(Path(tmpdir))
            handle_batch_command("/timeout 9", state, 1)
            handle_batch_command("/timeout", state, 2)
            output = buf.getvalue()
        self.assertEqual(state.config.timeout, 9)
        self.assertIn("Timeout set to 9", output)
        self.assertIn("Current timeout: 9", output)

    def test_export_json_succeeds(self):
        with tempfile.TemporaryDirectory() as tmpdir, io.StringIO() as buf, redirect_stdout(buf):
            state = make_state(Path(tmpdir))
            handle_batch_command("/export json", state, 1)
            output = buf.getvalue()
            files = list(Path(tmpdir).glob("*.json"))
            self.assertTrue(files)
            self.assertIn("Exported to:", output)

    def test_export_html_succeeds(self):
        with tempfile.TemporaryDirectory() as tmpdir, io.StringIO() as buf, redirect_stdout(buf):
            state = make_state(Path(tmpdir))
            handle_batch_command("/export html", state, 1)
            output = buf.getvalue()
            files = list(Path(tmpdir).glob("*.html"))
            self.assertTrue(files)
            self.assertIn("Exported to:", output)

    def test_import_command_branch(self):
        with tempfile.TemporaryDirectory() as tmpdir, io.StringIO() as buf, redirect_stdout(buf):
            state = make_state(Path(tmpdir))
            # create an export to import
            export_path = Path(tmpdir) / "exp.json"
            export_path.write_text('{"model": "llama3", "provider_id": "ollama", "temperature": 0.7, "messages": [{"role":"user","content":"hi"}]}', encoding="utf-8")
            handle_batch_command(f"/import {export_path}", state, 1)
            output = buf.getvalue()
            self.assertIn("Imported conversation as:", output)
            # ensure directory created
            dirs = [p for p in Path(tmpdir).iterdir() if p.is_dir()]
            self.assertTrue(dirs)

    def test_profile_branch(self):
        with tempfile.TemporaryDirectory() as tmpdir, io.StringIO() as buf, redirect_stdout(buf):
            state = make_state(Path(tmpdir))
            handle_batch_command("/profile", state, 1)
            output = buf.getvalue()
            self.assertIn("Provider:", output)
            self.assertIn("Model:", output)

    def test_retry_branch(self):
        with tempfile.TemporaryDirectory() as tmpdir, io.StringIO() as buf, redirect_stdout(buf):
            state = make_state(Path(tmpdir))
            state.current_conversation.messages.append(
                type("M", (), {"role": "user", "content": "hi again"})()
            )
            state.current_conversation.messages.append(
                type("M", (), {"role": "assistant", "content": "last"})()
            )
            cont, msg, _ = handle_batch_command("/retry", state, 1)
            self.assertTrue(cont)
            self.assertEqual(msg, "hi again")

    def test_tag_command_batch(self):
        with tempfile.TemporaryDirectory() as tmpdir, io.StringIO() as buf, redirect_stdout(buf):
            state = make_state(Path(tmpdir))
            handle_batch_command("/tag topic=science", state, 1)
            output = buf.getvalue()
        self.assertIn("Tag set: topic=science", output)
        self.assertEqual(state.current_conversation.tags.get("topic"), "science")


if __name__ == "__main__":
    unittest.main()
