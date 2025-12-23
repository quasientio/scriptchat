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
from pathlib import Path

from scriptchat.core.commands import AppState, create_new_conversation
from scriptchat.core.config import Config, ModelConfig, ProviderConfig
from scriptchat.core.conversations import Conversation, Message
from scriptchat.ui.command_handlers import CommandHandlers
from scriptchat.__main__ import parse_script_lines


def make_config(root: Path, system_prompt: str | None = None):
    provider = ProviderConfig(
        id="ollama",
        type="ollama",
        api_url="http://localhost:11434/api",
        api_key="",
        models=[ModelConfig(name="llama3", context=1024)],
        streaming=True,
        headers={},
        default_model="llama3",
    )
    return Config(
        api_url="http://localhost:11434/api",
        api_key="",
        conversations_dir=root,
        exports_dir=root,
        enable_streaming=False,
        system_prompt=system_prompt,
        default_provider="ollama",
        default_model="llama3",
        default_temperature=0.7,
        timeout=30,
        log_level="INFO",
        log_file=None,
        providers=[provider],
        file_confirm_threshold_bytes=40_000,
    )


class FakeApp:
    def __init__(self, state: AppState):
        self.state = state
        self.prompt_message = ""
        self.messages = []
        self.last_callback = None
        self.last_user_message = None
        self.script_queue = None
        self.updated = False

    def add_system_message(self, msg: str):
        self.messages.append(msg)

    def update_conversation_display(self):
        self.updated = True

    def prompt_for_input(self, callback, expect_single_key: bool = False):
        self.last_callback = callback

    def handle_user_message(self, msg: str):
        self.last_user_message = msg

    def parse_script_lines(self, lines):
        return parse_script_lines(lines)

    def run_script_queue(self, lines):
        self.script_queue = lines


def make_state(tmp_path: Path, system_prompt: str | None = None):
    cfg = make_config(tmp_path, system_prompt=system_prompt)
    convo = Conversation(
        id=None,
        provider_id="ollama",
        model_name="llama3",
        temperature=0.5,
        system_prompt=system_prompt,
        messages=[],
        tokens_in=0,
        tokens_out=0,
    )
    return AppState(config=cfg, current_conversation=convo, client=None, conversations_root=tmp_path, file_registry={}, folder_registry={})


class CommandHandlersUiTests(unittest.TestCase):
    def test_echo_and_save_load_flow(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_state(Path(tmpdir), system_prompt="sys")
            app = FakeApp(state)
            handlers = CommandHandlers(app)

            handlers.handle_command("/echo hello")
            self.assertEqual(state.current_conversation.messages[-1].role, "echo")
            self.assertTrue(app.updated)

            handlers.handle_save("greet")
            self.assertIsNotNone(state.current_conversation.id)
            self.assertIn("saved as", app.messages[-1].lower())

            handlers.handle_load("0")
            self.assertIn("loaded", app.messages[-1].lower())
            self.assertEqual(state.current_conversation.messages[0].role, "system")

    def test_branch_and_rename_and_stream_toggle(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            state = make_state(root)
            app = FakeApp(state)
            handlers = CommandHandlers(app)

            handlers.handle_save("base")
            old_id = state.current_conversation.id
            handlers.handle_branch("fork")
            self.assertNotEqual(old_id, state.current_conversation.id)

            branched_id = state.current_conversation.id
            handlers.handle_rename("renamed")
            self.assertNotEqual(branched_id, state.current_conversation.id)
            self.assertTrue((root / state.current_conversation.id).exists())
            self.assertIn("renamed", app.messages[-1].lower())

            handlers.handle_stream("")
            self.assertTrue(state.config.enable_streaming)
            self.assertIn("streaming", app.messages[-1].lower())

    def test_prompt_temp_and_run_script(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_state(Path(tmpdir))
            app = FakeApp(state)
            handlers = CommandHandlers(app)

            handlers.handle_prompt("be nice")
            self.assertEqual(state.current_conversation.system_prompt, "be nice")
            self.assertEqual(state.current_conversation.messages[0].role, "system")

            handlers.handle_prompt("clear")
            self.assertIsNone(state.current_conversation.system_prompt)

            handlers.handle_temp("1.2")
            self.assertAlmostEqual(state.current_conversation.temperature, 1.2)

            # No-arg temp prompts for input
            handlers.handle_temp("")
            self.assertEqual(app.prompt_message, "New temperature (0.0-2.0):")
            self.assertIsNotNone(app.last_callback)

            script = Path(tmpdir) / "script.txt"
            script.write_text("# comment\nhello\n/send there\n", encoding="utf-8")
            handlers.handle_run(str(script))
            self.assertEqual(app.script_queue, ["hello", "/send there"])

    def test_clear_deletes_current_and_saved(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            state = make_state(root, system_prompt="sys")
            app = FakeApp(state)
            handlers = CommandHandlers(app)

            handlers.handle_save("keep")
            saved_dir = root / state.current_conversation.id
            self.assertTrue(saved_dir.exists())

            handlers.handle_clear("")
            # simulate user confirming clear
            app.last_callback("y")

            self.assertFalse(saved_dir.exists())
            self.assertIsNone(state.current_conversation.id)
            self.assertIn("cleared", " ".join(app.messages).lower())

    def test_send_and_export(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            state = make_state(root)
            app = FakeApp(state)
            handlers = CommandHandlers(app)

            handlers.handle_send("hi there")
            self.assertEqual(app.last_user_message, "hi there")

            handlers.handle_export("md")
            # export writes a file into exports_dir (root)
            exported = list(Path(root).glob("*.md"))
            self.assertTrue(exported)
            self.assertIn("exported", app.messages[-1].lower())

    def test_export_all_exports_all_saved_conversations(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            state = make_state(root)
            app = FakeApp(state)
            handlers = CommandHandlers(app)

            # Save first conversation
            state.current_conversation.messages.append(Message(role="user", content="first"))
            handlers.handle_save("conv1")

            # Create and save second conversation
            state.current_conversation = Conversation(
                id=None,
                provider_id="ollama",
                model_name="llama3",
                temperature=0.5,
                system_prompt=None,
                messages=[Message(role="user", content="second")],
                tokens_in=0,
                tokens_out=0,
            )
            handlers.handle_save("conv2")

            # Export all to JSON
            handlers.handle_export_all("json")

            # Should have 2 JSON files
            exported = list(Path(root).glob("*.json"))
            self.assertEqual(len(exported), 2)
            self.assertIn("exported 2 conversation(s)", app.messages[-1].lower())

    def test_export_all_prompts_for_format_when_not_provided(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            state = make_state(root)
            app = FakeApp(state)
            handlers = CommandHandlers(app)

            handlers.handle_export_all("")
            self.assertIn("export format", app.prompt_message.lower())
            self.assertIsNotNone(app.last_callback)

    def test_export_all_rejects_invalid_format(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            state = make_state(root)
            app = FakeApp(state)
            handlers = CommandHandlers(app)

            handlers.handle_export_all("invalid")
            self.assertIn("unsupported format", app.messages[-1].lower())

    def test_export_all_with_no_saved_conversations(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            state = make_state(root)
            app = FakeApp(state)
            handlers = CommandHandlers(app)

            handlers.handle_export_all("md")
            self.assertIn("no saved conversations", app.messages[-1].lower())

    def test_load_registers_file_references(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            file_path = root / "note.txt"
            file_path.write_text("hello world", encoding="utf-8")
            convo = Conversation(
                id=None,
                provider_id="ollama",
                model_name="llama3",
                temperature=0.7,
                system_prompt=None,
                messages=[Message(role="user", content=f"See @{file_path}")],
                tokens_in=0,
                tokens_out=0,
                file_references={str(file_path): str(file_path)}
            )
            from scriptchat.core.conversations import save_conversation
            saved = save_conversation(root, convo, save_name="withfile", system_prompt=None)

            state = make_state(root)
            app = FakeApp(state)
            handlers = CommandHandlers(app)

            handlers.handle_load("0")

            self.assertEqual(state.current_conversation.id, saved.id)
            self.assertIn(str(file_path), state.file_registry)
            self.assertIn(file_path.name, state.file_registry)
            self.assertEqual(state.file_registry[str(file_path)]["content"], "hello world")

    def test_load_warns_missing_file_references(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            convo = Conversation(
                id=None,
                provider_id="ollama",
                model_name="llama3",
                temperature=0.7,
                system_prompt=None,
                messages=[Message(role="user", content="Missing @{absent.txt}")],
                tokens_in=0,
                tokens_out=0,
                file_references={"absent.txt": str(root / "absent.txt")},
            )
            from scriptchat.core.conversations import save_conversation
            save_conversation(root, convo, save_name="missing", system_prompt=None)

            state = make_state(root)
            app = FakeApp(state)
            handlers = CommandHandlers(app)

            handlers.handle_load("0")

            combined = " ".join(app.messages).lower()
            self.assertIn("missing file", combined)
            self.assertIn("absent.txt", combined)
            self.assertIn("absent.txt", state.file_registry)
            self.assertTrue(state.file_registry["absent.txt"]["missing"])

    def test_model_and_prompt_callbacks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_state(Path(tmpdir))
            # Add second provider with default_model but no models list to hit branch
            other_provider = ProviderConfig(
                id="remote",
                type="openai-compatible",
                api_url="https://api.fake",
                api_key="",
                models=[],
                streaming=True,
                headers={},
                default_model="gpt-x",
            )
            state.config.providers.append(other_provider)
            app = FakeApp(state)
            handlers = CommandHandlers(app)

            # handle_model with provider/model string
            handlers.handle_model("remote/gpt-x")
            self.assertIn("provider: remote", app.messages[-1])

            # Unknown index falls back to current provider/model
            handlers.handle_model("999")
            self.assertIn("provider:", app.messages[-1])

            # prompt via callback
            handlers.handle_prompt("")
            self.assertEqual(app.prompt_message, "New system prompt (empty to clear):")
            cb = app.last_callback
            cb("new prompt")
            self.assertEqual(state.current_conversation.system_prompt, "new prompt")

    def test_load_by_display_name(self):
        """Load a conversation by its simple save name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            state = make_state(root)
            app = FakeApp(state)
            handlers = CommandHandlers(app)

            handlers.handle_save("my-chat")
            saved_id = state.current_conversation.id

            # Create a new conversation
            state.current_conversation = Conversation(
                id=None, provider_id="ollama", model_name="llama3",
                temperature=0.5, system_prompt=None, messages=[],
                tokens_in=0, tokens_out=0,
            )

            # Load by display name
            handlers.handle_load("my-chat")
            self.assertEqual(state.current_conversation.id, saved_id)
            self.assertIn("loaded", app.messages[-1].lower())

    def test_load_by_full_dir_name(self):
        """Load a conversation by its full directory name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            state = make_state(root)
            app = FakeApp(state)
            handlers = CommandHandlers(app)

            handlers.handle_save("another-chat")
            saved_id = state.current_conversation.id

            # Create a new conversation
            state.current_conversation = Conversation(
                id=None, provider_id="ollama", model_name="llama3",
                temperature=0.5, system_prompt=None, messages=[],
                tokens_in=0, tokens_out=0,
            )

            # Load by full dir_name
            handlers.handle_load(saved_id)
            self.assertEqual(state.current_conversation.id, saved_id)
            self.assertIn("loaded", app.messages[-1].lower())

    def test_load_by_name_not_found(self):
        """Error message when conversation name doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            state = make_state(root)
            app = FakeApp(state)
            handlers = CommandHandlers(app)

            handlers.handle_save("exists")

            handlers.handle_load("nonexistent")
            self.assertIn("not found", app.messages[-1].lower())

    def test_load_by_name_multiple_matches(self):
        """Error when multiple conversations match the same name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            state = make_state(root)
            app = FakeApp(state)
            handlers = CommandHandlers(app)

            # Manually create two directories with same display_name but different timestamps
            import json
            for ts in ["202412010000", "202412020000"]:
                dir_name = f"{ts}_llama3_samename"
                conv_dir = root / dir_name
                conv_dir.mkdir()
                meta = {"model": "llama3", "provider_id": "ollama", "temperature": 0.7}
                (conv_dir / "meta.json").write_text(json.dumps(meta))

            # Try to load by name - should fail with multiple matches
            handlers.handle_load("samename")
            self.assertIn("multiple", app.messages[-1].lower())


if __name__ == "__main__":
    unittest.main()
