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

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import json

from scriptchat.core.commands import AppState, CommandResult, create_new_conversation, handle_command, set_model, set_temperature, resolve_placeholders, expand_variables
from scriptchat.core.config import Config, ModelConfig, ProviderConfig
from scriptchat.core.conversations import Conversation, Message
from scriptchat.core.provider_dispatcher import ProviderDispatcher


def make_config(tmp_path: Path, system_prompt: str | None = "system says"):
    provider = ProviderConfig(
        id="ollama",
        type="ollama",
        api_url="http://localhost:11434/api",
        api_key="",
        models=[ModelConfig(name="llama3", context=2048)],
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
            self.assertIn("5 chars", result.message)
            self.assertIn("tokens", result.message)  # Token estimate included
            self.assertIn("% ctx", result.message)  # Context percentage included
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

    def test_unfile_command_removes_file_and_aliases(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_state(Path(tmpdir))
            file_path = Path(tmpdir) / "note.txt"
            file_path.write_text("hello", encoding="utf-8")

            # Register the file
            handle_command(f"/file {file_path}", state)
            self.assertIn(str(file_path), state.file_registry)
            self.assertIn("note.txt", state.file_registry)

            # Unregister by full path
            result = handle_command(f"/unfile {file_path}", state)
            self.assertIn("Unregistered", result.message)
            self.assertNotIn(str(file_path), state.file_registry)
            self.assertNotIn("note.txt", state.file_registry)

    def test_unfile_command_removes_by_basename(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_state(Path(tmpdir))
            file_path = Path(tmpdir) / "note.txt"
            file_path.write_text("hello", encoding="utf-8")

            # Register the file
            handle_command(f"/file {file_path}", state)

            # Unregister by basename
            result = handle_command("/unfile note.txt", state)
            self.assertIn("Unregistered", result.message)
            self.assertNotIn(str(file_path), state.file_registry)
            self.assertNotIn("note.txt", state.file_registry)

    def test_unfile_command_not_found(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_state(Path(tmpdir))
            result = handle_command("/unfile nonexistent.txt", state)
            self.assertIn("not registered", result.message)

    def test_models_command_lists_providers_and_models(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_state(Path(tmpdir))
            # Add another provider with alias and reasoning
            other = ProviderConfig(
                id="openai",
                type="openai-compatible",
                api_url="http://example",
                api_key="",
                models=[
                    ModelConfig(name="gpt-5", context=400000, alias="g5", reasoning_levels=["low", "high"], reasoning_default="low"),
                ],
                streaming=True,
                headers={},
                default_model=None,
            )
            state.config.providers.append(other)

            result = handle_command("/models", state)
            self.assertIn("Models by provider:", result.message)
            # Check ollama provider
            self.assertIn("[ollama]", result.message)
            self.assertIn("llama3", result.message)
            # Check openai provider
            self.assertIn("[openai]", result.message)
            self.assertIn("gpt-5", result.message)
            self.assertIn("alias: **g5**", result.message)
            self.assertIn("ctx: 400,000", result.message)
            self.assertIn("reasoning: low/high", result.message)

    @patch('scriptchat.core.commands.check_ollama_running', return_value=True)
    def test_set_model_resets_token_counters_and_validates(self, mock_check):
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
                models=[ModelConfig(name="gpt-5.1", context=1024)],
                streaming=True,
                headers={},
                default_model="gpt-5.1",
            )
            state.config.providers.append(other)
            result = set_model(state, "gpt-5.1")
            self.assertIn("provider: remote", result.message)
            self.assertEqual(state.current_conversation.provider_id, "remote")
            self.assertEqual(state.current_conversation.model_name, "gpt-5.1")

    def test_set_model_requires_model_name_not_just_provider(self):
        """Switching to just a provider without a model is not supported."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_state(Path(tmpdir))
            other = ProviderConfig(
                id="remote",
                type="openai-compatible",
                api_url="http://example",
                api_key="",
                models=[ModelConfig(name="gpt-5.1", context=1024)],
                streaming=True,
                headers={},
                default_model=None,
            )
            state.config.providers.append(other)
            # Trying to switch to just a provider name should fail
            result = set_model(state, "remote")
            self.assertIn("not found", result.message)
            # Should stay on original provider/model
            self.assertEqual(state.current_conversation.provider_id, "ollama")
            self.assertEqual(state.current_conversation.model_name, "llama3")

    def test_set_model_resolves_alias(self):
        """Test that set_model resolves model aliases to provider/model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_state(Path(tmpdir))
            # Add a provider with aliased models
            other = ProviderConfig(
                id="fireworks",
                type="openai-compatible",
                api_url="http://example",
                api_key="",
                models=[
                    ModelConfig(name="accounts/fireworks/models/deepseek-v3", context=1024, alias="dsv3"),
                    ModelConfig(name="accounts/fireworks/models/deepseek-r1", context=1024, alias="dsr1"),
                ],
                streaming=True,
                headers={},
                default_model=None,
            )
            state.config.providers.append(other)

            # Switch using alias
            result = set_model(state, "dsv3")
            self.assertIn("Switched to model", result.message)
            self.assertIn("alias: dsv3", result.message)
            self.assertEqual(state.current_conversation.provider_id, "fireworks")
            self.assertEqual(state.current_conversation.model_name, "accounts/fireworks/models/deepseek-v3")

            # Switch to another alias
            result = set_model(state, "dsr1")
            self.assertIn("alias: dsr1", result.message)
            self.assertEqual(state.current_conversation.model_name, "accounts/fireworks/models/deepseek-r1")

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
            # Model already has context=2048 from make_config
            result = handle_command("/profile", state)
            msg = result.message or ""
            self.assertIn("Context: 2048", msg)

    def test_profile_shows_context_not_configured_when_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_state(Path(tmpdir))
            # Use a model name that has no defaults in model_defaults.py
            state.current_conversation.model_name = "unknown-test-model-xyz"
            state.config.providers[0].models[0].name = "unknown-test-model-xyz"
            state.config.providers[0].models[0].context = None
            result = handle_command("/profile", state)
            msg = result.message or ""
            self.assertIn("Context: (not configured)", msg)

    def test_profile_shows_reasoning_off_when_available_but_not_set(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_state(Path(tmpdir))
            # Enable reasoning on the model but don't set a level
            state.config.providers[0].models[0].reasoning_levels = ["low", "medium", "high"]
            result = handle_command("/profile", state)
            msg = result.message or ""
            self.assertIn("Reasoning: (off)", msg)

    def test_profile_shows_reasoning_level_when_set(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_state(Path(tmpdir))
            state.config.providers[0].models[0].reasoning_levels = ["low", "medium", "high"]
            state.current_conversation.reasoning_level = "high"
            result = handle_command("/profile", state)
            msg = result.message or ""
            self.assertIn("Reasoning: high (32000 tokens)", msg)

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

            # Test disabling with 0
            disable_zero = handle_command("/timeout 0", state)
            self.assertIn("Timeout disabled", disable_zero.message)
            self.assertIsNone(state.config.timeout)
            self.assertIsNone(dispatcher.clients["ollama"].timeout)

            # Show disabled state
            disabled_status = handle_command("/timeout", state)
            self.assertIn("disabled", disabled_status.message)

            # Test disabling with off
            handle_command("/timeout 60", state)  # re-enable first
            disable_off = handle_command("/timeout off", state)
            self.assertIn("Timeout disabled", disable_off.message)
            self.assertIsNone(state.config.timeout)

            # Test invalid value (negative)
            handle_command("/timeout 30", state)  # re-enable first
            invalid = handle_command("/timeout -5", state)
            self.assertIn("greater than zero", invalid.message)
            self.assertEqual(state.config.timeout, 30)

    def test_note_command_adds_message(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_state(Path(tmpdir))
            initial_count = len(state.current_conversation.messages)

            result = handle_command("/note Remember to test edge cases", state)
            self.assertIn("[Note]", result.message)
            self.assertIn("Remember to test edge cases", result.message)

            # Check that a note message was added
            self.assertEqual(len(state.current_conversation.messages), initial_count + 1)
            note_msg = state.current_conversation.messages[-1]
            self.assertEqual(note_msg.role, "note")
            self.assertEqual(note_msg.content, "Remember to test edge cases")

            # Usage error when no text provided
            usage = handle_command("/note", state)
            self.assertIn("Usage:", usage.message)

    def test_history_command(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_state(Path(tmpdir))

            # Empty history
            empty = handle_command("/history", state)
            self.assertIn("No messages", empty.message)

            # Add some messages
            state.current_conversation.messages = [
                Message(role="user", content="Hello"),
                Message(role="assistant", content="Hi there!"),
                Message(role="user", content="How are you?"),
                Message(role="assistant", content="I'm doing well."),
            ]

            # Default (last 10, but we only have 2 user messages)
            result = handle_command("/history", state)
            self.assertIn("2 of 2", result.message)
            self.assertIn("Hello", result.message)
            self.assertIn("How are you?", result.message)
            # Should NOT include assistant messages
            self.assertNotIn("Hi there!", result.message)
            self.assertNotIn("doing well", result.message)

            # Specific count
            result2 = handle_command("/history 1", state)
            self.assertIn("1 of 2", result2.message)
            self.assertIn("How are you?", result2.message)
            self.assertNotIn("Hello", result2.message)

            # All
            result_all = handle_command("/history all", state)
            self.assertIn("2 of 2", result_all.message)

            # Invalid
            invalid = handle_command("/history abc", state)
            self.assertIn("Usage:", invalid.message)

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
            from scriptchat.core.conversations import save_conversation
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
            from scriptchat.core.conversations import save_conversation
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
            from scriptchat.core.conversations import save_conversation
            saved = save_conversation(root, state.current_conversation, save_name="test", system_prompt=state.current_conversation.system_prompt)
            state.current_conversation = saved

            result = handle_command("/tag topic=science", state)
            self.assertIn("saved", result.message)
            meta = (root / saved.id / "meta.json").read_text(encoding="utf-8")
            self.assertIn("topic", meta)

    def test_log_level_command(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_state(Path(tmpdir))
            # Without args, delegates to UI for selection menu
            res_no_args = handle_command("/log-level", state)
            self.assertTrue(res_no_args.needs_ui_interaction)
            self.assertEqual(res_no_args.command_type, 'log-level')
            # With valid level
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

            # Without args, it delegates to UI for selection menu
            show = handle_command("/reason", state)
            self.assertTrue(show.needs_ui_interaction)
            self.assertEqual(show.command_type, 'reason')

            bad = handle_command("/reason low", state)
            self.assertIn("Unsupported", bad.message)
            self.assertIsNone(state.current_conversation.reasoning_level)

            res = handle_command("/reason medium", state)
            self.assertIn("Reasoning level set", res.message)
            self.assertEqual(state.current_conversation.reasoning_level, "medium")

            cleared = handle_command("/reason clear", state)
            self.assertIsNone(state.current_conversation.reasoning_level)

    def test_reason_command_not_available_when_unsupported(self):
        # When /reason is called with no args, it delegates to UI for selection menu.
        # The UI handler then checks if reasoning is available.
        # Here we test that calling with an invalid level on unsupported model
        # returns the "not available" message.
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_state(Path(tmpdir))
            # Without args, it returns UI interaction request
            res = handle_command("/reason", state)
            self.assertTrue(res.needs_ui_interaction)
            self.assertEqual(res.command_type, 'reason')
            # With an invalid level on unsupported model
            res = handle_command("/reason high", state)
            self.assertIn("not available", res.message.lower())

    def test_set_model_applies_reasoning_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_state(Path(tmpdir))
            # Add another model with a default reasoning level
            state.config.providers[0].models.append(
                ModelConfig(name="llama-reason", context=None, reasoning_levels=["none", "high"], reasoning_default="high")
            )

            res = set_model(state, "llama-reason")
            self.assertIn("Switched to model", res.message)
            self.assertEqual(state.current_conversation.model_name, "llama-reason")
            self.assertEqual(state.current_conversation.reasoning_level, "high")

    def test_thinking_command_sets_budget(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_state(Path(tmpdir))
            res = handle_command("/thinking 8000", state)
            self.assertIn("8000 tokens", res.message)
            self.assertEqual(state.current_conversation.thinking_budget, 8000)
            self.assertIsNone(state.current_conversation.reasoning_level)

    def test_thinking_command_shows_current(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_state(Path(tmpdir))
            state.current_conversation.thinking_budget = 16000
            res = handle_command("/thinking", state)
            self.assertIn("16000 tokens", res.message)

    def test_thinking_command_validates_range(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_state(Path(tmpdir))
            res_low = handle_command("/thinking 500", state)
            self.assertIn("Minimum", res_low.message)
            self.assertIsNone(state.current_conversation.thinking_budget)

            res_high = handle_command("/thinking 200000", state)
            self.assertIn("Maximum", res_high.message)
            self.assertIsNone(state.current_conversation.thinking_budget)

    def test_thinking_command_clears_budget(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_state(Path(tmpdir))
            state.current_conversation.thinking_budget = 8000
            res = handle_command("/thinking off", state)
            self.assertIn("disabled", res.message.lower())
            self.assertIsNone(state.current_conversation.thinking_budget)

    def test_thinking_and_reason_clear_each_other(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_state(Path(tmpdir))
            state.config.providers[0].models[0].reasoning_levels = ["low", "medium", "high", "max"]

            # Set thinking budget, then reason level should clear it
            handle_command("/thinking 10000", state)
            self.assertEqual(state.current_conversation.thinking_budget, 10000)

            handle_command("/reason medium", state)
            self.assertIsNone(state.current_conversation.thinking_budget)
            self.assertEqual(state.current_conversation.reasoning_level, "medium")

            # Set reason level, then thinking budget should clear it
            handle_command("/thinking 20000", state)
            self.assertIsNone(state.current_conversation.reasoning_level)
            self.assertEqual(state.current_conversation.thinking_budget, 20000)


class HelpCommandTests(unittest.TestCase):
    """Tests for the /help command."""

    def test_help_shows_all_categories(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_state(Path(tmpdir))
            result = handle_command("/help", state)
            # Should show category headers
            self.assertIn("Conversation:", result.message)
            self.assertIn("Export/Import:", result.message)
            self.assertIn("Model & Settings:", result.message)
            self.assertIn("Files:", result.message)
            self.assertIn("Tags:", result.message)
            self.assertIn("System:", result.message)

    def test_help_shows_commands(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_state(Path(tmpdir))
            result = handle_command("/help", state)
            # Should list various commands
            self.assertIn("/save", result.message)
            self.assertIn("/export", result.message)
            self.assertIn("/model", result.message)
            self.assertIn("/help", result.message)

    def test_help_specific_command_shows_details(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_state(Path(tmpdir))
            result = handle_command("/help save", state)
            # Should show usage and examples
            self.assertIn("/save", result.message)
            self.assertIn("Examples:", result.message)
            self.assertIn("my-chat", result.message)

    def test_help_specific_command_with_slash(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_state(Path(tmpdir))
            result = handle_command("/help /export", state)
            # Should work with leading slash
            self.assertIn("Export current conversation", result.message)

    def test_help_search_finds_matching_commands(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_state(Path(tmpdir))
            result = handle_command("/help conversation", state)
            # Should find commands with "conversation" in description
            self.assertIn("matching", result.message.lower())
            self.assertIn("/save", result.message)
            self.assertIn("/new", result.message)

    def test_help_search_no_matches(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_state(Path(tmpdir))
            result = handle_command("/help xyznonexistent", state)
            self.assertIn("No commands found", result.message)

    def test_help_export_all_included(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_state(Path(tmpdir))
            result = handle_command("/help export-all", state)
            self.assertIn("Export all saved conversations", result.message)

    def test_keys_command_shows_shortcuts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_state(Path(tmpdir))
            result = handle_command("/keys", state)
            self.assertIn("Keyboard Shortcuts", result.message)
            self.assertIn("Ctrl+Up", result.message)
            self.assertIn("Escape", result.message)
            self.assertIn("Tab", result.message)


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

    def test_expands_inside_markdown_code_blocks(self):
        """Test that @references inside backticks are expanded correctly."""
        registry = {"README.md": {"content": "# My Project", "full_path": "/README.md"}}
        # Triple backticks adjacent to @reference
        text = "what do you think? ```@README.md```"
        result, err = resolve_placeholders(text, registry)
        self.assertIsNone(err)
        self.assertEqual(result, "what do you think? ```# My Project```")

    def test_expands_inside_single_backticks(self):
        """Test that @references inside single backticks are expanded."""
        registry = {"file.py": {"content": "code here", "full_path": "/file.py"}}
        text = "Look at `@file.py` please"
        result, err = resolve_placeholders(text, registry)
        self.assertIsNone(err)
        self.assertEqual(result, "Look at `code here` please")


class ScriptVariablesTests(unittest.TestCase):
    """Tests for /set, /vars commands and variable expansion."""

    def test_expand_variables_simple(self):
        """Test basic variable expansion."""
        variables = {"name": "Alice", "greeting": "Hello"}
        result = expand_variables("${greeting}, ${name}!", variables)
        self.assertEqual(result, "Hello, Alice!")

    def test_expand_variables_unknown_left_unchanged(self):
        """Unknown variables are left as-is."""
        variables = {"known": "value"}
        result = expand_variables("${known} and ${unknown}", variables)
        self.assertEqual(result, "value and ${unknown}")

    def test_expand_variables_empty_dict(self):
        """Empty variables dict leaves all references unchanged."""
        result = expand_variables("${foo} ${bar}", {})
        self.assertEqual(result, "${foo} ${bar}")

    def test_expand_variables_no_references(self):
        """Text without variable references is unchanged."""
        variables = {"foo": "bar"}
        result = expand_variables("plain text", variables)
        self.assertEqual(result, "plain text")

    def test_expand_variables_underscore_names(self):
        """Variable names can start with underscore."""
        variables = {"_private": "secret", "var_name": "value"}
        result = expand_variables("${_private} ${var_name}", variables)
        self.assertEqual(result, "secret value")

    def test_set_command_creates_variable(self):
        """Test /set command creates a variable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_state(Path(tmpdir))
            result = handle_command("/set foo=bar", state)
            self.assertIn("foo", result.message)
            self.assertEqual(state.variables.get("foo"), "bar")

    def test_set_command_with_equals_in_value(self):
        """Test /set with value containing = sign."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_state(Path(tmpdir))
            result = handle_command("/set equation=a=b+c", state)
            self.assertEqual(state.variables.get("equation"), "a=b+c")

    def test_set_command_invalid_name(self):
        """Test /set rejects invalid variable names."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_state(Path(tmpdir))
            result = handle_command("/set 123invalid=value", state)
            self.assertIn("Invalid", result.message)
            self.assertNotIn("123invalid", state.variables)

    def test_set_command_missing_value(self):
        """Test /set requires name=value format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_state(Path(tmpdir))
            result = handle_command("/set novalue", state)
            self.assertIn("Usage", result.message)

    def test_vars_command_empty(self):
        """Test /vars with no variables defined."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_state(Path(tmpdir))
            result = handle_command("/vars", state)
            self.assertIn("No variables", result.message)

    def test_vars_command_lists_variables(self):
        """Test /vars lists all defined variables."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_state(Path(tmpdir))
            state.variables = {"alpha": "1", "beta": "2"}
            result = handle_command("/vars", state)
            self.assertIn("alpha", result.message)
            self.assertIn("beta", result.message)

    def test_variable_expansion_in_command(self):
        """Test that variables are expanded in command arguments."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_state(Path(tmpdir))
            state.variables = {"msg": "hello world"}
            result = handle_command("/echo ${msg}", state)
            self.assertEqual(result.message, "hello world")

    def test_set_overwrites_existing_variable(self):
        """Overwriting a variable replaces its value."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_state(Path(tmpdir))
            handle_command("/set foo=original", state)
            self.assertEqual(state.variables.get("foo"), "original")
            handle_command("/set foo=updated", state)
            self.assertEqual(state.variables.get("foo"), "updated")

    def test_set_empty_value(self):
        """Setting a variable to empty string is allowed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_state(Path(tmpdir))
            result = handle_command("/set empty=", state)
            self.assertEqual(state.variables.get("empty"), "")
            self.assertIn("empty", result.message)

    def test_set_value_with_spaces(self):
        """Value can contain spaces."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_state(Path(tmpdir))
            handle_command("/set msg=hello world with spaces", state)
            self.assertEqual(state.variables.get("msg"), "hello world with spaces")

    def test_set_value_with_special_chars(self):
        """Value can contain special characters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_state(Path(tmpdir))
            handle_command("/set special=!@#$%^&*()", state)
            self.assertEqual(state.variables.get("special"), "!@#$%^&*()")

    def test_set_invalid_name_with_hyphen(self):
        """Variable names cannot contain hyphens."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_state(Path(tmpdir))
            result = handle_command("/set my-var=value", state)
            self.assertIn("Invalid", result.message)
            self.assertNotIn("my-var", state.variables)

    def test_set_invalid_name_with_space(self):
        """Variable names cannot contain spaces."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_state(Path(tmpdir))
            # "my var=value" splits as name="my var" which is invalid
            result = handle_command("/set my var=value", state)
            self.assertIn("Invalid", result.message)

    def test_expand_adjacent_variables(self):
        """Adjacent variable references expand correctly."""
        variables = {"a": "hello", "b": "world"}
        result = expand_variables("${a}${b}", variables)
        self.assertEqual(result, "helloworld")

    def test_expand_variable_in_middle_of_text(self):
        """Variable in middle of text expands correctly."""
        variables = {"name": "Alice"}
        result = expand_variables("Hello ${name}, welcome!", variables)
        self.assertEqual(result, "Hello Alice, welcome!")

    def test_expand_same_variable_multiple_times(self):
        """Same variable referenced multiple times expands each occurrence."""
        variables = {"x": "foo"}
        result = expand_variables("${x} and ${x} again", variables)
        self.assertEqual(result, "foo and foo again")

    def test_expand_preserves_unclosed_brace(self):
        """Unclosed ${name doesn't match and is left alone."""
        variables = {"name": "value"}
        result = expand_variables("${name and ${name}", variables)
        self.assertEqual(result, "${name and value")

    def test_expand_empty_braces_unchanged(self):
        """${} is not a valid variable reference and is left unchanged."""
        variables = {"": "empty"}
        result = expand_variables("test ${} here", variables)
        self.assertEqual(result, "test ${} here")

    def test_expand_nested_syntax_unchanged(self):
        """Nested ${${x}} doesn't do recursive expansion."""
        variables = {"x": "inner", "inner": "value"}
        result = expand_variables("${${x}}", variables)
        # Only outer braces match, inner ${x} becomes "inner", result is ${inner}
        # which doesn't get re-expanded
        self.assertEqual(result, "${inner}")

    def test_variables_are_case_sensitive(self):
        """Variable names are case-sensitive."""
        variables = {"Name": "upper", "name": "lower"}
        result = expand_variables("${Name} vs ${name}", variables)
        self.assertEqual(result, "upper vs lower")

    def test_expand_with_newlines_in_value(self):
        """Variables can contain newlines."""
        variables = {"multi": "line1\nline2"}
        result = expand_variables("${multi}", variables)
        self.assertEqual(result, "line1\nline2")

    def test_vars_shows_sorted_output(self):
        """Variables are listed in sorted order."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_state(Path(tmpdir))
            state.variables = {"zebra": "z", "alpha": "a", "beta": "b"}
            result = handle_command("/vars", state)
            # Check that alpha appears before beta appears before zebra
            alpha_pos = result.message.find("alpha")
            beta_pos = result.message.find("beta")
            zebra_pos = result.message.find("zebra")
            self.assertLess(alpha_pos, beta_pos)
            self.assertLess(beta_pos, zebra_pos)

    def test_set_then_use_in_same_session(self):
        """Set a variable and use it in the same session."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_state(Path(tmpdir))
            handle_command("/set prefix=DEBUG", state)
            result = handle_command("/echo ${prefix}: message", state)
            self.assertEqual(result.message, "DEBUG: message")

    def test_variable_in_set_value_expands(self):
        """Variables in /set value are expanded (since expansion happens first)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_state(Path(tmpdir))
            state.variables = {"base": "/home/user"}
            handle_command("/set path=${base}/docs", state)
            self.assertEqual(state.variables.get("path"), "/home/user/docs")

    def test_expand_env_var_fallback(self):
        """Environment variables are used when script variable not set."""
        import os
        os.environ["SCRIPTCHAT_TEST_VAR"] = "from_env"
        try:
            result = expand_variables("Value: ${SCRIPTCHAT_TEST_VAR}", {})
            self.assertEqual(result, "Value: from_env")
        finally:
            del os.environ["SCRIPTCHAT_TEST_VAR"]

    def test_expand_script_var_shadows_env(self):
        """Script variables take precedence over environment variables."""
        import os
        os.environ["SCRIPTCHAT_TEST_VAR"] = "from_env"
        try:
            variables = {"SCRIPTCHAT_TEST_VAR": "from_script"}
            result = expand_variables("Value: ${SCRIPTCHAT_TEST_VAR}", variables)
            self.assertEqual(result, "Value: from_script")
        finally:
            del os.environ["SCRIPTCHAT_TEST_VAR"]

    def test_expand_env_var_not_found_unchanged(self):
        """Unknown variables (not in script or env) are left unchanged."""
        import os
        # Ensure var doesn't exist
        os.environ.pop("SCRIPTCHAT_NONEXISTENT_VAR", None)
        result = expand_variables("${SCRIPTCHAT_NONEXISTENT_VAR}", {})
        self.assertEqual(result, "${SCRIPTCHAT_NONEXISTENT_VAR}")

    def test_expand_mixed_script_and_env_vars(self):
        """Mix of script and environment variables in same text."""
        import os
        os.environ["ENV_VAR"] = "env_value"
        try:
            variables = {"SCRIPT_VAR": "script_value"}
            result = expand_variables("${SCRIPT_VAR} and ${ENV_VAR}", variables)
            self.assertEqual(result, "script_value and env_value")
        finally:
            del os.environ["ENV_VAR"]

    def test_expand_blocks_sensitive_env_vars_by_default(self):
        """Default blocklist prevents expansion of sensitive env vars like *_KEY."""
        import os
        os.environ["OPENAI_API_KEY"] = "sk-secret123"
        os.environ["MY_SECRET"] = "hidden"
        os.environ["AUTH_TOKEN"] = "bearer-xyz"
        try:
            # These should NOT be expanded (blocked by default patterns)
            result = expand_variables("${OPENAI_API_KEY}", {})
            self.assertEqual(result, "${OPENAI_API_KEY}")

            result = expand_variables("${MY_SECRET}", {})
            self.assertEqual(result, "${MY_SECRET}")

            result = expand_variables("${AUTH_TOKEN}", {})
            self.assertEqual(result, "${AUTH_TOKEN}")
        finally:
            del os.environ["OPENAI_API_KEY"]
            del os.environ["MY_SECRET"]
            del os.environ["AUTH_TOKEN"]

    def test_expand_allows_non_sensitive_env_vars(self):
        """Non-sensitive env vars are expanded normally."""
        import os
        os.environ["LANGUAGE"] = "Python"
        os.environ["MY_VAR"] = "hello"
        try:
            result = expand_variables("${LANGUAGE} ${MY_VAR}", {})
            self.assertEqual(result, "Python hello")
        finally:
            del os.environ["LANGUAGE"]
            del os.environ["MY_VAR"]

    def test_expand_env_disabled(self):
        """When env_expand=False, env vars are not expanded."""
        import os
        os.environ["SAFE_VAR"] = "value"
        try:
            result = expand_variables("${SAFE_VAR}", {}, env_expand=False)
            self.assertEqual(result, "${SAFE_VAR}")
        finally:
            del os.environ["SAFE_VAR"]

    def test_expand_custom_blocklist_overrides_defaults(self):
        """Custom blocklist replaces defaults entirely."""
        import os
        os.environ["MY_PRIVATE_DATA"] = "sensitive"
        os.environ["OPENAI_API_KEY"] = "sk-secret"  # Normally blocked by default
        try:
            # Custom blocklist only blocks MY_PRIVATE_*, not *_KEY
            result = expand_variables(
                "${MY_PRIVATE_DATA} ${OPENAI_API_KEY}",
                {},
                env_blocklist=["MY_PRIVATE_*"]
            )
            # MY_PRIVATE_DATA blocked, but OPENAI_API_KEY now allowed (not in custom list)
            self.assertEqual(result, "${MY_PRIVATE_DATA} sk-secret")
        finally:
            del os.environ["MY_PRIVATE_DATA"]
            del os.environ["OPENAI_API_KEY"]

    def test_expand_empty_blocklist_allows_all(self):
        """Empty blocklist allows all env vars including sensitive ones."""
        import os
        os.environ["OPENAI_API_KEY"] = "sk-secret"
        try:
            result = expand_variables("${OPENAI_API_KEY}", {}, env_blocklist=[])
            self.assertEqual(result, "sk-secret")
        finally:
            del os.environ["OPENAI_API_KEY"]

    def test_expand_script_var_overrides_blocked_env(self):
        """Script variables can still be set even if env var name is blocked."""
        import os
        os.environ["OPENAI_API_KEY"] = "env-secret"
        try:
            # Script var takes precedence and is not subject to blocklist
            variables = {"OPENAI_API_KEY": "explicit-value"}
            result = expand_variables("${OPENAI_API_KEY}", variables)
            self.assertEqual(result, "explicit-value")
        finally:
            del os.environ["OPENAI_API_KEY"]


if __name__ == "__main__":
    unittest.main()
