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

"""Tests for the /search command."""

import tempfile
import unittest
from pathlib import Path

from scriptchat.core.commands import (
    AppState,
    CommandResult,
    handle_command,
    search_conversation,
    _extract_snippet,
)
from scriptchat.core.config import Config, ModelConfig, ProviderConfig
from scriptchat.core.conversations import Conversation, Message


def make_config(tmp_path: Path):
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
        system_prompt="system says",
        default_provider="ollama",
        default_model="llama3",
        default_temperature=0.7,
        timeout=30,
        file_confirm_threshold_bytes=40_000,
        log_level="INFO",
        log_file=None,
        providers=[provider],
    )


def make_state(tmpdir: Path):
    cfg = make_config(tmpdir)
    convo = Conversation(
        id="test",
        provider_id="ollama",
        model_name="llama3",
        temperature=0.5,
        system_prompt="system says",
        messages=[],
        tokens_in=0,
        tokens_out=0,
    )
    return AppState(
        config=cfg,
        current_conversation=convo,
        client=None,
        conversations_root=tmpdir,
        file_registry={},
        folder_registry={},
    )


class TestSearchCommand(unittest.TestCase):
    def test_search_command_requires_pattern(self):
        """Test that /search without pattern shows usage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_state(Path(tmpdir))
            result = handle_command("/search", state)
            self.assertIsInstance(result, CommandResult)
            self.assertIn("Usage:", result.message)

    def test_search_command_needs_ui_interaction(self):
        """Test that /search with pattern requests UI interaction."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_state(Path(tmpdir))
            result = handle_command("/search hello", state)
            self.assertTrue(result.needs_ui_interaction)
            self.assertEqual(result.command_type, "search")

    def test_search_conversation_literal(self):
        """Test literal substring search in conversation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_state(Path(tmpdir))
            state.current_conversation.messages = [
                Message(role="user", content="Hello world"),
                Message(role="assistant", content="Hi there! How can I help?"),
                Message(role="user", content="Tell me about Python"),
                Message(role="assistant", content="Python is a programming language"),
            ]

            matches = search_conversation(state.current_conversation, "Python")
            self.assertEqual(len(matches), 2)
            # Check that both messages containing "Python" are found
            self.assertEqual(matches[0][0], 2)  # User message at index 2
            self.assertEqual(matches[0][1], "user")
            self.assertEqual(matches[1][0], 3)  # Assistant message at index 3
            self.assertEqual(matches[1][1], "assistant")

    def test_search_conversation_case_insensitive(self):
        """Test that search is case-insensitive."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_state(Path(tmpdir))
            state.current_conversation.messages = [
                Message(role="user", content="Hello WORLD"),
                Message(role="assistant", content="hi there"),
            ]

            matches = search_conversation(state.current_conversation, "hello")
            self.assertEqual(len(matches), 1)
            self.assertEqual(matches[0][0], 0)

            matches = search_conversation(state.current_conversation, "HI")
            self.assertEqual(len(matches), 1)
            self.assertEqual(matches[0][0], 1)

    def test_search_conversation_regex(self):
        """Test regex pattern search."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_state(Path(tmpdir))
            state.current_conversation.messages = [
                Message(role="user", content="Error 404: not found"),
                Message(role="assistant", content="Error 500: server error"),
                Message(role="user", content="Success!"),
            ]

            # Search for "Error" followed by digits
            matches = search_conversation(state.current_conversation, r"Error \d+")
            self.assertEqual(len(matches), 2)
            self.assertEqual(matches[0][0], 0)
            self.assertEqual(matches[1][0], 1)

    def test_search_conversation_no_matches(self):
        """Test search with no matches returns empty list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_state(Path(tmpdir))
            state.current_conversation.messages = [
                Message(role="user", content="Hello world"),
            ]

            matches = search_conversation(state.current_conversation, "Python")
            self.assertEqual(len(matches), 0)

    def test_search_skips_non_content_messages(self):
        """Test that search skips echo, note, status messages."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_state(Path(tmpdir))
            state.current_conversation.messages = [
                Message(role="user", content="Hello"),
                Message(role="echo", content="Echo hello"),
                Message(role="note", content="Note hello"),
                Message(role="status", content="Status hello"),
            ]

            matches = search_conversation(state.current_conversation, "hello")
            # Should only match the user message, not echo/note/status
            self.assertEqual(len(matches), 1)
            self.assertEqual(matches[0][0], 0)
            self.assertEqual(matches[0][1], "user")

    def test_extract_snippet_with_context(self):
        """Test snippet extraction with context."""
        text = "The quick brown fox jumps over the lazy dog. This is a test sentence."
        # Match "fox" at position 16-19
        snippet = _extract_snippet(text, 16, 19, context_chars=10)
        self.assertIn("fox", snippet)
        self.assertIn("brown", snippet)
        self.assertIn("jumps", snippet)
        self.assertTrue(snippet.startswith("..."))
        self.assertTrue(snippet.endswith("..."))

    def test_extract_snippet_at_start(self):
        """Test snippet extraction at text start (no leading ellipsis)."""
        text = "Hello world! This is a test."
        snippet = _extract_snippet(text, 0, 5, context_chars=10)
        self.assertFalse(snippet.startswith("..."))
        self.assertTrue(snippet.endswith("..."))

    def test_extract_snippet_at_end(self):
        """Test snippet extraction at text end (no trailing ellipsis)."""
        text = "This is a test sentence."
        snippet = _extract_snippet(text, 19, 24, context_chars=10)
        self.assertTrue(snippet.startswith("..."))
        self.assertFalse(snippet.endswith("..."))

    def test_extract_snippet_short_text(self):
        """Test snippet extraction from short text (no ellipsis)."""
        text = "Hello world"
        snippet = _extract_snippet(text, 0, 5, context_chars=50)
        self.assertFalse(snippet.startswith("..."))
        self.assertFalse(snippet.endswith("..."))
        self.assertEqual(snippet, "Hello world")

    def test_extract_snippet_newlines_replaced(self):
        """Test that newlines are replaced with spaces in snippet."""
        text = "Line one\nLine two\nLine three"
        snippet = _extract_snippet(text, 9, 17, context_chars=5)
        self.assertNotIn("\n", snippet)
        self.assertIn("Line two", snippet)


if __name__ == "__main__":
    unittest.main()
