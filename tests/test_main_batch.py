import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from litechat.__main__ import handle_batch_command, parse_script_lines, run_batch, run_batch_lines
from litechat.core.commands import AppState
from litechat.core.config import Config, ModelConfig, ProviderConfig
from litechat.core.conversations import Conversation, Message


def make_config(root: Path):
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
        conversations_dir=root,
        exports_dir=root,
        enable_streaming=False,
        system_prompt=None,
        default_provider="ollama",
        default_model="llama3",
        default_temperature=0.7,
        timeout=5,
        log_level="INFO",
        log_file=None,
        providers=[provider],
    )
    # Support legacy /model index handling
    cfg.models = provider.models
    return cfg


def make_state(tmp_path: Path):
    cfg = make_config(tmp_path)
    convo = Conversation(
        id=None,
        provider_id="ollama",
        model_name="llama3",
        temperature=0.7,
        messages=[],
        tokens_in=0,
        tokens_out=0,
    )
    return AppState(config=cfg, current_conversation=convo, client=None, conversations_root=tmp_path)


class FakeClient:
    def __init__(self):
        self.calls = []

    def chat(self, convo, new_user_message: str, streaming: bool = False, on_chunk=None):
        self.calls.append(new_user_message)
        convo.messages.append(Message(role="assistant", content="ACK"))
        return "ACK"


class MainBatchTests(unittest.TestCase):
    def test_parse_script_lines_strips_comments(self):
        lines = ["# comment\n", " say\n", "\n", "/send hi\n"]
        parsed = parse_script_lines(lines)
        self.assertEqual(parsed, ["say", "/send hi"])

    def test_handle_batch_command_stream_prompt_send(self):
        with tempfile.TemporaryDirectory() as tmpdir, io.StringIO() as buf, redirect_stdout(buf):
            state = make_state(Path(tmpdir))
            # stream toggle
            should_continue, msg, exit_code = handle_batch_command("/stream on", state, 1)
            self.assertTrue(should_continue)
            self.assertIsNone(exit_code)
            self.assertTrue(state.config.enable_streaming)
            # prompt set
            handle_batch_command("/prompt hello", state, 2)
            self.assertEqual(state.current_conversation.system_prompt, "hello")
            self.assertEqual(state.current_conversation.messages[0].role, "system")
            # prompt clear
            handle_batch_command("/prompt clear", state, 3)
            self.assertIsNone(state.current_conversation.system_prompt)
            # model by index
            handle_batch_command("/model 0", state, 4)
            self.assertEqual(state.current_conversation.model_name, "llama3")
            # send returns message
            cont, send_msg, _ = handle_batch_command("/send hi", state, 5)
            self.assertTrue(cont)
            self.assertEqual(send_msg, "hi")
            # unsupported
            handle_batch_command("/unknown", state, 6)
            out = buf.getvalue()
            self.assertIn("Streaming enabled", out)
            self.assertIn("System prompt set", out)
            self.assertIn("System prompt cleared", out)
            self.assertIn("unknown", out)

    def test_run_batch_executes_messages(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            script = root / "run.txt"
            script.write_text("hello\n/send hi\n/exit\n", encoding="utf-8")

            state = make_state(root)
            state.client = FakeClient()

            with io.StringIO() as buf, redirect_stdout(buf):
                exit_code = run_batch(str(script), state)
                output = buf.getvalue()

            self.assertEqual(exit_code, 0)
            self.assertIn("[User]: hello", output)
            self.assertIn("[Assistant]: ACK", output)
            self.assertEqual(state.client.calls, ["hello", "hi"])

    def test_run_batch_assert_pass_fail(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            script = root / "assert.txt"
            script.write_text("Q\n/assert fail\n", encoding="utf-8")
            state = make_state(root)
            state.client = FakeClient()
            with io.StringIO() as buf, redirect_stdout(buf):
                code = run_batch(str(script), state)
                out = buf.getvalue()
            self.assertEqual(code, 1)
            self.assertIn("FAILED", out)

            # Passing assertion
            script.write_text("Q\n/assert ACK\n/assert-not nope\n", encoding="utf-8")
            with io.StringIO() as buf, redirect_stdout(buf):
                code = run_batch(str(script), state)
            self.assertEqual(code, 0)

    def test_run_batch_continue_on_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            script = root / "assert.txt"
            script.write_text("hi\n/assert nope\n/assert-not ACK\n/exit\n", encoding="utf-8")
            state = make_state(root)
            state.client = FakeClient()
            with io.StringIO() as buf, redirect_stdout(buf):
                code = run_batch(str(script), state, continue_on_error=True)
                out = buf.getvalue()
            self.assertEqual(code, 1)
            self.assertIn("FAILED", out)

    def test_run_batch_errors_and_file_command(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            missing = run_batch(str(root / "missing.txt"), make_state(root))
            self.assertEqual(missing, 1)

            comments = root / "comments.txt"
            comments.write_text("# only comments\n# still comments\n", encoding="utf-8")
            no_lines = run_batch(str(comments), make_state(root))
            self.assertEqual(no_lines, 1)

            # file command should print load error for missing file
            state = make_state(root)
            state.client = FakeClient()
            script = root / "file_script.txt"
            script.write_text("/file missing.txt\n/exit\n", encoding="utf-8")
            with io.StringIO() as buf, redirect_stdout(buf):
                exit_code = run_batch(str(script), state)
                output = buf.getvalue()
            self.assertEqual(exit_code, 0)
            self.assertIn("File not found", output)

    def test_run_batch_lines_handles_stdin_flow(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_state(Path(tmpdir))
            state.client = FakeClient()
            lines = ["hello", "/send hi", "/assert ACK"]
            with io.StringIO() as buf, redirect_stdout(buf):
                code = run_batch_lines(lines, state, continue_on_error=False, source_label="<stdin>")
                output = buf.getvalue()
            self.assertEqual(code, 0)
            self.assertIn("[User]: hello", output)
            self.assertIn("ACK", output)


if __name__ == "__main__":
    unittest.main()
