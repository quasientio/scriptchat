# Copyright 2024 lite-chat contributors
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

import tempfile
import unittest
from pathlib import Path

import json

from litechat.core.commands import AppState, CommandResult, create_new_conversation, handle_command, set_model, set_temperature, resolve_placeholders
from litechat.core.config import Config, ModelConfig, ProviderConfig
from litechat.core.conversations import Conversation, Message
from litechat.core.provider_dispatcher import ProviderDispatcher


def make_config(tmp_path: Path, system_prompt: str | None = "system says"):
    provider = ProviderConfig(
        id="ollama",
        type="ollama",
        api_url="http://localhost:11434/api",
        api_key="",
        models=[ModelConfig(name="llama3", contexts=[2048])],
        streaming=True,
        headers={},
        default_model="llama3",
    )
    return Config(
        api_url="http://localhost:11434/api",
        api_key="",
        conversations_dir=tmp_path,
        exports_dir=None,
        enable_streaming=False,
        system_prompt=system_prompt,
        default_provider="ollama",
        default_model="llama3",
        default_temperature=0.7,
        timeout=30,
        file_confirm_threshold_bytes=40_000,
        log_level="INFO",
        log_file=None,
        providers=[provider],
    )


def make_state(tmpdir: Path, system_prompt: str | None = "system says"):
    cfg = make_config(tmpdir, system_prompt=system_prompt)
    convo = Conversation(
        id="existing",
        provider_id="ollama",
        model_name="llama3",
        temperature=0.5,
        system_prompt=system_prompt,
        messages=[],
        tokens_in=3,
        tokens_out=4,
    )
    return AppState(config=cfg, current_conversation=convo, client=None, conversations_root=tmpdir, file_registry={})


class CommandTests(unittest.TestCase):
    def test_handle_new_injects_system_prompt(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_state(Path(tmpdir))
            result = handle_command("/new", state)
            self.assertIsInstance(result, CommandResult)
            self.assertEqual(result.message, "Started new conversation")
            self.assertIsNone(state.current_conversation.id)
            self.assertEqual(state.current_conversation.provider_id, "ollama")
            self.assertEqual([m.role for m in state.current_conversation.messages], ["system"])

    def test_handle_exit_sets_flag(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_state(Path(tmpdir))
            result = handle_command("/exit", state)
            self.assertTrue(result.should_exit)
            self.assertIn("Exiting", result.message)

    def test_file_command_reads_content(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_state(Path(tmpdir))
            file_path = Path(tmpdir) / "note.txt"
            file_path.write_text("hello", encoding="utf-8")

            result = handle_command(f"/file {file_path}", state)
            self.assertIsNone(result.file_content)
            self.assertIn("Registered", result.message)
            entry = state.file_registry[str(file_path)]
            self.assertEqual(entry["content"], "hello")

    def test_file_command_requires_force_above_threshold(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_state(Path(tmpdir))
            # set a tiny threshold
            state.config.file_confirm_threshold_bytes = 1
            big = Path(tmpdir) / "big.txt"
            big.write_text("123456", encoding="utf-8")

            result = handle_command(f"/file {big}", state)
            self.assertIn("exceeds", result.message)
            self.assertNotIn(str(big), state.file_registry)

            force = handle_command(f"/file --force {big}", state)
            self.assertIn("Registered", force.message)
            self.assertIn(str(big), state.file_registry)

    def test_files_command_lists_entries(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_state(Path(tmpdir))
            file_path = Path(tmpdir) / "note.txt"
            file_path.write_text("hello", encoding="utf-8")

            handle_command(f"/file {file_path}", state)
            result = handle_command("/files", state)

            lines = (result.message or "").splitlines()
            self.assertEqual(lines[0], "Registered files:")
            expected_full = str(file_path.resolve())
            expected_lines = {
                f"@{file_path} -> {expected_full}",
                f"@{file_path.name} -> {expected_full}",
            }
            self.assertSetEqual(set(lines[1:]), expected_lines)

    def test_files_command_long_format(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_state(Path(tmpdir))
            file_path = Path(tmpdir) / "note.txt"
            file_path.write_text("hi", encoding="utf-8")

            handle_command(f"/file {file_path}", state)
            result = handle_command("/files --long", state)

            lines = (result.message or "").splitlines()
            self.assertEqual(lines[0], "Registered files:")
            expected_full = str(file_path.resolve())
            # content is "hi" -> len 2 and known sha256
            digest = "8f434346648f6b96df89dda901c5176b10a6d83961dd3c1ac88b59b2dc327aa4"
            self.assertIn(f"@{file_path} -> {expected_full} (2 bytes) sha256:{digest}", lines[1:])
            self.assertIn(f"@{file_path.name} -> {expected_full} (2 bytes) sha256:{digest}", lines[1:])

    def test_set_model_resets_token_counters_and_validates(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_state(Path(tmpdir))
            state.current_conversation.tokens_in = 10
            state.current_conversation.tokens_out = 5

            ok = set_model(state, "llama3")
            self.assertEqual(ok.message, "Switched to model: llama3")
            self.assertEqual(state.current_conversation.tokens_in, 0)
            self.assertEqual(state.current_conversation.tokens_out, 0)

            missing = set_model(state, "unknown")
            self.assertIn("not found", missing.message)
            self.assertEqual(state.current_conversation.model_name, "llama3")

    def test_set_model_switches_provider_if_model_found_elsewhere(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_state(Path(tmpdir))
            other = ProviderConfig(
                id="remote",
                type="openai-compatible",
                api_url="http://example",
                api_key="",
                models=[ModelConfig(name="gpt-5.1", contexts=[1024])],
                streaming=True,
                headers={},
                default_model="gpt-5.1",
            )
            state.config.providers.append(other)
            result = set_model(state, "gpt-5.1")
            self.assertIn("provider: remote", result.message)
            self.assertEqual(state.current_conversation.provider_id, "remote")
            self.assertEqual(state.current_conversation.model_name, "gpt-5.1")

    def test_set_temperature_clamps_range(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_state(Path(tmpdir))
            high = set_temperature(state, 3.5)
            self.assertEqual(state.current_conversation.temperature, 2.0)
            self.assertIn("2.00", high.message)

            low = set_temperature(state, -1)
            self.assertEqual(state.current_conversation.temperature, 0.0)
            self.assertIn("0.00", low.message)

    def test_unknown_and_send_usage_messages(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_state(Path(tmpdir))
            unknown = handle_command("/doesnotexist", state)
            self.assertIn("Unknown command", unknown.message)

            send_usage = handle_command("/send", state)
            self.assertIn("Usage: /send", send_usage.message)

    def test_assert_command_pass_fail(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_state(Path(tmpdir))
            state.current_conversation.messages.append(Message(role="assistant", content="The capital is Paris"))

            ok = handle_command("/assert Paris", state)
            self.assertTrue(ok.assert_passed)
            self.assertIn("PASSED", ok.message)

            fail = handle_command("/assert London", state)
            self.assertFalse(fail.assert_passed)
            self.assertIn("FAILED", fail.message)

            not_found = handle_command("/assert-not london", state)
            self.assertTrue(not_found.assert_passed)

            not_fail = handle_command("/assert-not paris", state)
            self.assertFalse(not_fail.assert_passed)

    def test_assert_without_assistant_and_non_commands(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_state(Path(tmpdir))
            missing = handle_command("/assert anything", state)
            self.assertFalse(missing.assert_passed)
            self.assertIn("No assistant response", missing.message)

            not_command = handle_command("hello", state)
            self.assertIn("Commands must start with /", not_command.message)

            empty = handle_command("/", state)
            self.assertIn("Empty command", empty.message)

    def test_profile_command_outputs_snapshot(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_state(Path(tmpdir))
            result = handle_command("/profile", state)
            msg = result.message or ""
            self.assertIn("Provider: ollama", msg)
            self.assertIn("Model: llama3", msg)
            self.assertIn("Tokens:", msg)
            self.assertIn("Timeout:", msg)
            self.assertIn("convs)", msg.lower())

    def test_profile_shows_reasoning_unavailable_when_not_configured(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_state(Path(tmpdir))
            # Model has no reasoning_levels configured
            result = handle_command("/profile", state)
            msg = result.message or ""
            self.assertIn("Reasoning: (unavailable)", msg)

    def test_profile_shows_context_length_when_configured(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_state(Path(tmpdir))
            # Model already has contexts=[2048] from make_config
            result = handle_command("/profile", state)
            msg = result.message or ""
            self.assertIn("Context: 2048", msg)

    def test_profile_shows_context_not_configured_when_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_state(Path(tmpdir))
            # Remove contexts from model
            state.config.providers[0].models[0].contexts = None
            result = handle_command("/profile", state)
            msg = result.message or ""
            self.assertIn("Context: (not configured)", msg)

    def test_profile_shows_reasoning_default_when_available(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_state(Path(tmpdir))
            # Enable reasoning on the model
            state.config.providers[0].models[0].reasoning_levels = ["low", "medium", "high"]
            result = handle_command("/profile", state)
            msg = result.message or ""
            self.assertIn("Reasoning: (default)", msg)

    def test_profile_shows_reasoning_level_when_set(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_state(Path(tmpdir))
            state.config.providers[0].models[0].reasoning_levels = ["low", "medium", "high"]
            state.current_conversation.reasoning_level = "high"
            result = handle_command("/profile", state)
            msg = result.message or ""
            self.assertIn("Reasoning: high", msg)

    def test_profile_system_prompt_trimmed_and_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            long_prompt = "p" * 150
            state = make_state(Path(tmpdir), system_prompt=long_prompt)

            trimmed_result = handle_command("/profile", state)
            msg = trimmed_result.message or ""
            self.assertIn("System prompt:", msg)
            self.assertIn("...", msg)
            self.assertLess(len(msg), 400)  # ensure trimming applied

            state.current_conversation.system_prompt = None
            none_result = handle_command("/profile", state)
            self.assertIn("(none)", none_result.message)

    def test_timeout_command_updates_config_and_clients(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_state(Path(tmpdir))
            client = type("C", (), {"timeout": 5})()
            dispatcher = ProviderDispatcher({"ollama": client})
            state.client = dispatcher

            result = handle_command("/timeout 45", state)
            self.assertIn("Timeout set to 45", result.message)
            self.assertEqual(state.config.timeout, 45)
            self.assertEqual(dispatcher.clients["ollama"].timeout, 45)

            current = handle_command("/timeout", state)
            self.assertIn("Current timeout", current.message)

            invalid = handle_command("/timeout 0", state)
            self.assertIn("greater than zero", invalid.message)
            self.assertEqual(state.config.timeout, 45)

    def test_undo_removes_exchanges(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_state(Path(tmpdir))
            state.current_conversation.messages = [
                Message(role="user", content="u1"),
                Message(role="assistant", content="a1"),
                Message(role="user", content="u2"),
                Message(role="assistant", content="a2"),
                Message(role="user", content="u3"),
                Message(role="assistant", content="a3"),
            ]
            res = handle_command("/undo", state)
            self.assertIn("Removed 1", res.message)
            self.assertEqual([m.content for m in state.current_conversation.messages], ["u1", "a1", "u2", "a2"])

            res2 = handle_command("/undo 2", state)
            self.assertIn("Removed 2", res2.message)
            self.assertEqual(state.current_conversation.messages, [])

    def test_undo_skips_non_chat_roles(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_state(Path(tmpdir))
            state.current_conversation.messages = [
                Message(role="system", content="sys"),
                Message(role="user", content="u1"),
                Message(role="assistant", content="a1"),
                Message(role="echo", content="note"),
            ]
            res = handle_command("/undo", state)
            self.assertIn("Removed 1", res.message)
            self.assertEqual([m.role for m in state.current_conversation.messages], ["system"])

    def test_undo_repeated_single_steps(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_state(Path(tmpdir))
            state.current_conversation.messages = [
                Message(role="user", content="u1"),
                Message(role="assistant", content="a1"),
                Message(role="user", content="u2"),
                Message(role="assistant", content="a2"),
                Message(role="user", content="u3"),
                Message(role="assistant", content="a3"),
            ]

            res1 = handle_command("/undo 1", state)
            self.assertIn("Removed 1", res1.message)
            self.assertEqual([m.content for m in state.current_conversation.messages], ["u1", "a1", "u2", "a2"])

            res2 = handle_command("/undo 1", state)
            self.assertIn("Removed 1", res2.message)
            self.assertEqual([m.content for m in state.current_conversation.messages], ["u1", "a1"])

            res3 = handle_command("/undo 1", state)
            self.assertIn("Removed 1", res3.message)
            self.assertEqual(state.current_conversation.messages, [])

    def test_undo_default_equals_one(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_state(Path(tmpdir))
            state.current_conversation.messages = [
                Message(role="user", content="u1"),
                Message(role="assistant", content="a1"),
            ]
            res_default = handle_command("/undo", state)
            self.assertIn("Removed 1", res_default.message)
            # Reset and compare with explicit 1
            state.current_conversation.messages = [
                Message(role="user", content="u1"),
                Message(role="assistant", content="a1"),
            ]
            res_one = handle_command("/undo 1", state)
            self.assertIn("Removed 1", res_one.message)

    def test_tag_sets_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_state(Path(tmpdir))
            result = handle_command("/tag topic=science", state)
            self.assertIn("Tag set", result.message)
            self.assertEqual(state.current_conversation.tags.get("topic"), "science")

    def test_untag_removes_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_state(Path(tmpdir))
            handle_command("/tag topic=science", state)
            result = handle_command("/untag topic", state)
            self.assertIn("removed", result.message.lower())
            self.assertNotIn("topic", state.current_conversation.tags)

    def test_untag_missing_key(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_state(Path(tmpdir))
            result = handle_command("/untag missing", state)
            self.assertIn("not found", result.message.lower())

    def test_tags_lists_current_tags(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_state(Path(tmpdir))
            no_tags = handle_command("/tags", state)
            self.assertIn("no tags", no_tags.message.lower())
            handle_command("/tag topic=science", state)
            handle_command("/tag owner=alice", state)
            listed = handle_command("/tags", state)
            self.assertIn("Tags:", listed.message)
            self.assertIn("owner=alice", listed.message)
            self.assertIn("topic=science", listed.message)

    def test_file_command_persists_references_in_meta(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            state = make_state(root)
            file_path = root / "note.txt"
            file_path.write_text("hello", encoding="utf-8")

            handle_command(f"/file {file_path}", state)
            from litechat.core.conversations import save_conversation
            saved = save_conversation(root, state.current_conversation, save_name="save", system_prompt=None)
            meta = json.loads((root / saved.id / "meta.json").read_text(encoding="utf-8"))
            self.assertIn(str(file_path), meta.get("file_references", {}))

    def test_file_command_stores_sha256_hash(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            state = make_state(root)
            file_path = root / "note.txt"
            file_path.write_text("hello", encoding="utf-8")

            handle_command(f"/file {file_path}", state)
            from litechat.core.conversations import save_conversation
            saved = save_conversation(root, state.current_conversation, save_name="save", system_prompt=None)
            meta = json.loads((root / saved.id / "meta.json").read_text(encoding="utf-8"))

            file_ref = meta.get("file_references", {}).get(str(file_path))
            self.assertIsInstance(file_ref, dict)
            self.assertIn("path", file_ref)
            self.assertIn("sha256", file_ref)
            # sha256 of "hello"
            import hashlib
            expected_hash = hashlib.sha256("hello".encode("utf-8")).hexdigest()
            self.assertEqual(file_ref["sha256"], expected_hash)

    def test_file_command_updates_basename_hash_on_reregister(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            state = make_state(root)
            file_path = root / "note.txt"
            file_path.write_text("hello", encoding="utf-8")

            # Register file first time
            handle_command(f"/file {file_path}", state)

            import hashlib
            hash1 = hashlib.sha256("hello".encode("utf-8")).hexdigest()
            self.assertEqual(state.current_conversation.file_references[str(file_path)]["sha256"], hash1)
            self.assertEqual(state.current_conversation.file_references["note.txt"]["sha256"], hash1)

            # Modify file and re-register
            file_path.write_text("world", encoding="utf-8")
            handle_command(f"/file {file_path}", state)

            hash2 = hashlib.sha256("world".encode("utf-8")).hexdigest()
            # Both full path and basename should have updated hash
            self.assertEqual(state.current_conversation.file_references[str(file_path)]["sha256"], hash2)
            self.assertEqual(state.current_conversation.file_references["note.txt"]["sha256"], hash2)

    def test_tag_auto_saves_when_conversation_has_id(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            state = make_state(root)
            # Save conversation to assign id and persist
            from litechat.core.conversations import save_conversation
            saved = save_conversation(root, state.current_conversation, save_name="test", system_prompt=state.current_conversation.system_prompt)
            state.current_conversation = saved

            result = handle_command("/tag topic=science", state)
            self.assertIn("saved", result.message)
            meta = (root / saved.id / "meta.json").read_text(encoding="utf-8")
            self.assertIn("topic", meta)

    def test_log_level_command(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_state(Path(tmpdir))
            res = handle_command("/log-level debug", state)
            self.assertIn("Log level set to DEBUG", res.message)
            res_bad = handle_command("/log-level nope", state)
            self.assertIn("Usage", res_bad.message)

    def test_log_level_all_levels(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_state(Path(tmpdir))
            levels = ["debug", "info", "warn", "error", "critical"]
            for lvl in levels:
                res = handle_command(f"/log-level {lvl}", state)
                self.assertIn("Log level set", res.message)

    def test_reason_command_sets_and_clears_level(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_state(Path(tmpdir))
            # Enable reasoning on the configured model
            state.config.providers[0].models[0].reasoning_levels = ["none", "medium", "high"]

            show = handle_command("/reason", state)
            self.assertIn("Available", show.message)

            bad = handle_command("/reason low", state)
            self.assertIn("Unsupported", bad.message)
            self.assertIsNone(state.current_conversation.reasoning_level)

            res = handle_command("/reason medium", state)
            self.assertIn("Reasoning level set", res.message)
            self.assertEqual(state.current_conversation.reasoning_level, "medium")

            cleared = handle_command("/reason clear", state)
            self.assertIsNone(state.current_conversation.reasoning_level)

    def test_reason_command_not_available_when_unsupported(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_state(Path(tmpdir))
            res = handle_command("/reason", state)
            self.assertIn("not available", res.message.lower())

    def test_set_model_applies_reasoning_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_state(Path(tmpdir))
            # Add another model with a default reasoning level
            state.config.providers[0].models.append(
                ModelConfig(name="llama-reason", contexts=None, reasoning_levels=["none", "high"], reasoning_default="high")
            )

            res = set_model(state, "llama-reason")
            self.assertIn("Switched to model", res.message)
            self.assertEqual(state.current_conversation.model_name, "llama-reason")
            self.assertEqual(state.current_conversation.reasoning_level, "high")


class ResolvePlaceholdersTests(unittest.TestCase):
    """Tests for the resolve_placeholders function."""

    def test_expands_registered_file_reference(self):
        registry = {"myfile.txt": {"content": "file contents here", "full_path": "/tmp/myfile.txt"}}
        result, err = resolve_placeholders("Please review @myfile.txt for issues", registry)
        self.assertIsNone(err)
        self.assertEqual(result, "Please review file contents here for issues")

    def test_expands_braced_reference(self):
        registry = {"path/to/file.py": {"content": "def foo(): pass", "full_path": "/home/user/path/to/file.py"}}
        result, err = resolve_placeholders("Check @{path/to/file.py} now", registry)
        self.assertIsNone(err)
        self.assertEqual(result, "Check def foo(): pass now")

    def test_leaves_java_annotations_untouched(self):
        registry = {}
        text = """public class Foo {
    @Override
    public String toString() { return "foo"; }

    @Nullable
    private String name;
}"""
        result, err = resolve_placeholders(text, registry)
        self.assertIsNone(err)
        self.assertEqual(result, text)

    def test_leaves_unregistered_at_tokens_untouched(self):
        registry = {"registered.txt": {"content": "CONTENT", "full_path": "/tmp/registered.txt"}}
        text = "See @registered.txt and also @unregistered.txt"
        result, err = resolve_placeholders(text, registry)
        self.assertIsNone(err)
        self.assertEqual(result, "See CONTENT and also @unregistered.txt")

    def test_multiple_registered_references(self):
        registry = {
            "a.txt": {"content": "AAA", "full_path": "/a.txt"},
            "b.txt": {"content": "BBB", "full_path": "/b.txt"},
        }
        result, err = resolve_placeholders("First @a.txt then @b.txt", registry)
        self.assertIsNone(err)
        self.assertEqual(result, "First AAA then BBB")

    def test_mixed_registered_and_annotations(self):
        registry = {"code.java": {"content": "System.out.println();", "full_path": "/code.java"}}
        text = "File @code.java has @Override annotation"
        result, err = resolve_placeholders(text, registry)
        self.assertIsNone(err)
        self.assertEqual(result, "File System.out.println(); has @Override annotation")

    def test_empty_registry_leaves_all_untouched(self):
        registry = {}
        text = "Email me @user@example.com or check @Override"
        result, err = resolve_placeholders(text, registry)
        self.assertIsNone(err)
        self.assertEqual(result, text)

    def test_no_at_symbols_unchanged(self):
        registry = {"file.txt": {"content": "X", "full_path": "/file.txt"}}
        text = "No references here"
        result, err = resolve_placeholders(text, registry)
        self.assertIsNone(err)
        self.assertEqual(result, text)


if __name__ == "__main__":
    unittest.main()
