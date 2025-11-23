import tempfile
import unittest
from pathlib import Path

from prompt_toolkit.document import Document

from litechat.core.commands import AppState
from litechat.core.config import Config, ModelConfig, ProviderConfig
from litechat.core.conversations import Conversation, Message
from litechat.ui.app import AnsiLexer, LiteChatUI


def make_state():
    provider = ProviderConfig(
        id="ollama",
        type="ollama",
        api_url="http://localhost:11434/api",
        api_key="",
        models=[ModelConfig(name="llama3", contexts=[1000])],
        streaming=True,
        headers={},
        default_model="llama3",
    )
    cfg = Config(
        api_url="http://localhost:11434/api",
        api_key="",
        conversations_dir=Path("."),
        exports_dir=None,
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
    convo = Conversation(
        id="202401010101_llama3_chat",
        provider_id="ollama",
        model_name="llama3",
        temperature=0.7,
        system_prompt=None,
        messages=[
            Message(role="system", content="sys"),
            Message(role="user", content="hi"),
            Message(role="assistant", content="hello"),
            Message(role="echo", content="note"),
        ],
        tokens_in=10,
        tokens_out=5,
        context_length_configured=1000,
        context_length_used=100,
    )
    return AppState(config=cfg, current_conversation=convo, client=None, conversations_root=Path("."))


class AppHelperTests(unittest.TestCase):
    def test_status_bar_and_prompt_prefix(self):
        state = make_state()
        ui = object.__new__(LiteChatUI)
        ui.state = state
        ui.thinking = True
        ui.thinking_dots = 1
        ui.prompt_message = "Enter"

        status = ui._get_status_bar()
        self.assertIn("ollama/llama3", status[0][1])
        self.assertIn("Thinking", status[0][1])
        self.assertEqual(ui._get_prompt_prefix(), "Enter ")

    def test_build_conversation_text_and_lexer(self):
        state = make_state()
        ui = object.__new__(LiteChatUI)
        ui.state = state

        text = ui._build_conversation_text()
        self.assertIn("[user]", text)
        self.assertIn("[assistant]", text)
        self.assertIn("note", text)

        lexer = AnsiLexer()
        doc = Document(text)
        line_fn = lexer.lex_document(doc)
        # ensure at least first line returns formatted text list
        self.assertTrue(line_fn(0))

    def test_history_navigation_and_apply(self):
        ui = object.__new__(LiteChatUI)
        ui.input_history = []
        ui.input_history_index = None

        class FakeBuffer:
            def __init__(self):
                self.document = None

            def set_document(self, document):
                self.document = document

        ui.input_buffer = FakeBuffer()

        ui._append_history("first")
        ui._append_history("second")
        self.assertEqual(ui.input_history, ["first", "second"])

        ui._history_previous()
        self.assertEqual(ui.input_history_index, 1)
        self.assertEqual(ui.input_buffer.document.text, "second")

        ui._history_previous()
        self.assertEqual(ui.input_buffer.document.text, "first")

        ui._history_next()
        self.assertEqual(ui.input_buffer.document.text, "second")

    def test_litechat_ui_initialization_and_cleanup(self):
        state = make_state()

        class DummyClient:
            def cleanup(self):
                self.cleaned = True

        state.client = DummyClient()

        # Instantiate full UI to exercise layout/key setup
        ui = LiteChatUI(state)
        ui.add_system_message("note")
        self.assertIn("note", ui.conversation_buffer.text)

        ui._cleanup()
        self.assertTrue(hasattr(state.client, "cleaned"))

    def test_input_history_persistence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            conv_dir = root / "conversations"
            conv_dir.mkdir(parents=True, exist_ok=True)
            state = make_state()
            state.config.conversations_dir = conv_dir
            ui = LiteChatUI(state)
            ui._append_history("first")
            ui._append_history("second")

            # Reload UI and ensure history was persisted
            ui2 = LiteChatUI(state)
            self.assertEqual(ui2.input_history, ["first", "second"])


if __name__ == "__main__":
    unittest.main()
