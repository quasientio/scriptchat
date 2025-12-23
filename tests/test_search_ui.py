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

"""UI tests for the /search command."""

import tempfile
import unittest
from pathlib import Path

from scriptchat.core.commands import AppState
from scriptchat.core.config import Config, ModelConfig, ProviderConfig
from scriptchat.core.conversations import Conversation, Message
from scriptchat.ui.test_harness import UITestHarness


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


class TestSearchUI(unittest.TestCase):
    def test_search_shows_selection_menu_with_matches(self):
        """Test that /search shows selection menu with matching results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_state(Path(tmpdir))
            state.current_conversation.messages = [
                Message(role="user", content="Hello world"),
                Message(role="assistant", content="Hi there! How can I help?"),
                Message(role="user", content="Tell me about Python programming"),
                Message(role="assistant", content="Python is a powerful programming language"),
            ]

            harness = UITestHarness(state)
            harness.setup_component_mode()

            # Execute search command
            harness.execute_command("/search Python")

            # Check that selection menu is visible
            ui_state = harness.get_state()
            self.assertTrue(ui_state.selection_menu_visible)

            # Check that we have 2 matches (indices 2 and 3)
            self.assertEqual(len(ui_state.selection_menu_items), 2)

            # Verify menu items contain expected data
            items = ui_state.selection_menu_items
            # First match should be at index 2 (user message)
            self.assertEqual(items[0][0], 2)
            self.assertIn("#2", items[0][1])
            self.assertIn("user", items[0][1])
            self.assertIn("Python", items[0][1])

            # Second match should be at index 3 (assistant message)
            self.assertEqual(items[1][0], 3)
            self.assertIn("#3", items[1][1])
            self.assertIn("assistant", items[1][1])
            self.assertIn("Python", items[1][1])

    def test_search_no_matches_shows_message(self):
        """Test that /search with no matches shows appropriate message."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_state(Path(tmpdir))
            state.current_conversation.messages = [
                Message(role="user", content="Hello world"),
                Message(role="assistant", content="Hi there!"),
            ]

            harness = UITestHarness(state)
            harness.setup_component_mode()

            # Execute search command with pattern that won't match
            harness.execute_command("/search nonexistent")

            # Check that no selection menu is shown
            ui_state = harness.get_state()
            self.assertFalse(ui_state.selection_menu_visible)

            # Check that a message is shown
            harness.assert_conversation_contains("No matches found for: nonexistent")

    def test_search_regex_pattern(self):
        """Test /search with regex pattern."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_state(Path(tmpdir))
            state.current_conversation.messages = [
                Message(role="user", content="Error 404: not found"),
                Message(role="assistant", content="Let me help with that error"),
                Message(role="user", content="Error 500: server error"),
            ]

            harness = UITestHarness(state)
            harness.setup_component_mode()

            # Search for "Error" followed by digits
            harness.execute_command(r"/search Error \d+")

            # Check that selection menu is visible with 2 matches
            ui_state = harness.get_state()
            self.assertTrue(ui_state.selection_menu_visible)
            self.assertEqual(len(ui_state.selection_menu_items), 2)

            # Verify matches are at indices 0 and 2
            items = ui_state.selection_menu_items
            self.assertEqual(items[0][0], 0)
            self.assertEqual(items[1][0], 2)

    def test_search_case_insensitive(self):
        """Test that /search is case-insensitive."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_state(Path(tmpdir))
            state.current_conversation.messages = [
                Message(role="user", content="Hello WORLD"),
                Message(role="assistant", content="hi there"),
            ]

            harness = UITestHarness(state)
            harness.setup_component_mode()

            # Search for lowercase "hello" should match uppercase "HELLO"
            harness.execute_command("/search hello")

            ui_state = harness.get_state()
            self.assertTrue(ui_state.selection_menu_visible)
            self.assertEqual(len(ui_state.selection_menu_items), 1)

            # Search for uppercase "HI" should match lowercase "hi"
            harness.execute_command("/search HI")

            ui_state = harness.get_state()
            self.assertTrue(ui_state.selection_menu_visible)
            self.assertEqual(len(ui_state.selection_menu_items), 1)

    def test_search_multiple_occurrences_in_one_message(self):
        """Test that search finds only first occurrence per message."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_state(Path(tmpdir))
            state.current_conversation.messages = [
                Message(role="user", content="test test test"),
                Message(role="assistant", content="no match here"),
                Message(role="user", content="another test"),
            ]

            harness = UITestHarness(state)
            harness.setup_component_mode()

            harness.execute_command("/search test")

            ui_state = harness.get_state()
            self.assertTrue(ui_state.selection_menu_visible)
            # Should find 2 messages (indices 0 and 2), not 4 occurrences
            self.assertEqual(len(ui_state.selection_menu_items), 2)


if __name__ == "__main__":
    unittest.main()
