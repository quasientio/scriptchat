import tempfile
import unittest
from pathlib import Path

from litechat.core.commands import AppState, CommandResult, create_new_conversation, handle_command, set_model, set_temperature
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
    return AppState(config=cfg, current_conversation=convo, client=None, conversations_root=tmpdir)


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
            self.assertEqual(result.file_content, "hello")
            self.assertIn("Loaded file", result.message)

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
            self.assertEqual(missing.message, "Switched to model: unknown")
            self.assertEqual(state.current_conversation.model_name, "unknown")

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


if __name__ == "__main__":
    unittest.main()
