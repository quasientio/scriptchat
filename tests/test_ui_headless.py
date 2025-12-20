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

"""Tests for headless UI mode."""

import tempfile
import time
import unittest
from pathlib import Path

from scriptchat.core.commands import AppState
from scriptchat.core.config import Config, ModelConfig, ProviderConfig
from scriptchat.core.conversations import Conversation
from scriptchat.ui.test_harness import UITestHarness, MockLLMClient


def make_test_state(tmp_path: Path):
    """Create test AppState."""
    provider = ProviderConfig(
        id="test",
        type="ollama",
        api_url="http://localhost:11434/api",
        api_key="",
        models=[ModelConfig(name="test-model", context=1024)],
        streaming=False,
        headers={},
        default_model="test-model",
    )
    cfg = Config(
        api_url="http://localhost:11434/api",
        api_key="",
        conversations_dir=tmp_path,
        exports_dir=tmp_path,
        enable_streaming=False,
        system_prompt=None,
        default_provider="test",
        default_model="test-model",
        default_temperature=0.7,
        timeout=30,
        log_level="INFO",
        log_file=None,
        providers=[provider],
        file_confirm_threshold_bytes=40_000,
    )
    convo = Conversation(
        id=None,
        provider_id="test",
        model_name="test-model",
        temperature=0.7,
        messages=[],
        tokens_in=0,
        tokens_out=0,
    )
    return AppState(
        config=cfg,
        current_conversation=convo,
        client=None,
        conversations_root=tmp_path,
        file_registry={}
    )


class HeadlessModeTests(unittest.TestCase):
    """Tests for headless UI mode."""

    def test_start_and_stop(self):
        """Test basic start/stop lifecycle."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_test_state(Path(tmpdir))

            with UITestHarness(state) as harness:
                harness.start_headless()

                # Should be running
                self.assertTrue(harness._running)
                self.assertIsNotNone(harness.ui)

    def test_type_text(self):
        """Test typing text into input."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_test_state(Path(tmpdir))

            with UITestHarness(state) as harness:
                harness.start_headless()

                harness.send_text("hello")
                harness.wait_for_idle(timeout=1)

                # Text should be in input buffer
                self.assertIn("hello", harness.get_input_text())

    def test_send_message_and_get_response(self):
        """Test sending a message and receiving mock response."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_test_state(Path(tmpdir))

            mock = MockLLMClient(
                responses={"hello": "Hi there!"},
                default_response="I don't understand"
            )

            with UITestHarness(state) as harness:
                harness.start_headless(mock_client=mock)

                harness.send_message("hello", wait_for_response=True)

                # Should have user and assistant messages
                harness.assert_message_count(3)  # user + assistant + status (thought for X secs)
                harness.assert_conversation_contains("hello")
                harness.assert_conversation_contains("Hi there!")

    def test_command_execution(self):
        """Test executing a command."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_test_state(Path(tmpdir))

            with UITestHarness(state) as harness:
                harness.start_headless()

                harness.type_command("/echo test message")

                harness.assert_conversation_contains("test message")

    @unittest.skip("Escape key handling is unreliable in headless mode due to prompt_toolkit's escape sequence processing")
    def test_escape_clears_input(self):
        """Test that Escape clears input.

        Note: Escape handling is tricky in headless mode because prompt_toolkit
        waits to see if ESC is a prefix for another escape sequence (like arrow
        keys). This test is skipped but the functionality works in real terminal.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_test_state(Path(tmpdir))

            with UITestHarness(state) as harness:
                harness.start_headless()

                harness.send_text("some text")
                time.sleep(0.1)  # Wait for text to be processed
                self.assertIn("some text", harness.get_input_text())

                harness.send_key('escape', flush_delay=0.15)
                time.sleep(0.1)  # Wait for escape to be processed

                self.assertEqual("", harness.get_input_text())

    def test_context_manager(self):
        """Test that context manager properly stops harness."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_test_state(Path(tmpdir))

            harness = UITestHarness(state)
            with harness:
                harness.start_headless()
                self.assertTrue(harness._running)

            # After exiting context, should be stopped
            self.assertFalse(harness._running)


class MockClientTests(unittest.TestCase):
    """Tests for MockLLMClient."""

    def test_pattern_matching(self):
        """Test response pattern matching."""
        mock = MockLLMClient(
            responses={
                "weather": "It's sunny!",
                "time": "It's noon.",
            },
            default_response="I don't know"
        )

        convo = Conversation(
            id=None, provider_id="test", model_name="test",
            temperature=0.7, messages=[], tokens_in=0, tokens_out=0
        )

        # Pattern match
        resp = mock.chat(convo, "What's the weather?")
        self.assertEqual(resp, "It's sunny!")

        # Different pattern
        resp = mock.chat(convo, "What time is it?")
        self.assertEqual(resp, "It's noon.")

        # No match - default
        resp = mock.chat(convo, "Random question")
        self.assertEqual(resp, "I don't know")

    def test_call_tracking(self):
        """Test that calls are tracked."""
        mock = MockLLMClient()

        convo = Conversation(
            id=None, provider_id="test", model_name="test",
            temperature=0.7, messages=[], tokens_in=0, tokens_out=0
        )

        mock.chat(convo, "Hello")
        mock.chat(convo, "World")

        self.assertEqual(len(mock.calls), 2)
        self.assertEqual(mock.calls[0]['message'], "Hello")
        self.assertEqual(mock.calls[1]['message'], "World")

    def test_messages_added_to_conversation(self):
        """Test that responses are added to conversation."""
        mock = MockLLMClient(default_response="Test response")

        convo = Conversation(
            id=None, provider_id="test", model_name="test",
            temperature=0.7, messages=[], tokens_in=0, tokens_out=0
        )

        mock.chat(convo, "Hello")

        self.assertEqual(len(convo.messages), 1)
        self.assertEqual(convo.messages[0].role, "assistant")
        self.assertEqual(convo.messages[0].content, "Test response")

    def test_token_counting(self):
        """Test that tokens are estimated."""
        mock = MockLLMClient(default_response="Short response")

        convo = Conversation(
            id=None, provider_id="test", model_name="test",
            temperature=0.7, messages=[], tokens_in=0, tokens_out=0
        )

        mock.chat(convo, "Hello world")

        # Should have some token estimates
        self.assertGreater(convo.tokens_in, 0)
        self.assertGreater(convo.tokens_out, 0)


class KeySequenceTests(unittest.TestCase):
    """Tests for key sequence mapping."""

    def test_basic_keys(self):
        """Test that basic keys are mapped."""
        from scriptchat.ui.test_harness import KEY_SEQUENCES

        self.assertIn('enter', KEY_SEQUENCES)
        self.assertIn('escape', KEY_SEQUENCES)
        self.assertIn('tab', KEY_SEQUENCES)
        self.assertIn('up', KEY_SEQUENCES)
        self.assertIn('down', KEY_SEQUENCES)

    def test_ctrl_keys(self):
        """Test that ctrl keys are mapped."""
        from scriptchat.ui.test_harness import KEY_SEQUENCES

        self.assertIn('ctrl-c', KEY_SEQUENCES)
        self.assertIn('ctrl-d', KEY_SEQUENCES)
        self.assertIn('ctrl-a', KEY_SEQUENCES)

    def test_alt_keys(self):
        """Test that alt keys are mapped."""
        from scriptchat.ui.test_harness import KEY_SEQUENCES

        self.assertIn('alt-enter', KEY_SEQUENCES)


class HeadlessInputTests(unittest.TestCase):
    """Tests for headless input methods."""

    def test_send_key_validates_key(self):
        """Test that send_key validates key names."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_test_state(Path(tmpdir))

            with UITestHarness(state) as harness:
                harness.start_headless()

                # Invalid key should raise
                with self.assertRaises(ValueError) as ctx:
                    harness.send_key('invalid-key')

                self.assertIn('Unknown key', str(ctx.exception))

    def test_type_and_enter(self):
        """Test type_and_enter helper."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_test_state(Path(tmpdir))

            with UITestHarness(state) as harness:
                harness.start_headless()

                harness.type_command("/echo hello")
                harness.assert_conversation_contains("hello")


class HeadlessSynchronizationTests(unittest.TestCase):
    """Tests for synchronization methods."""

    def test_wait_for_idle_timeout(self):
        """Test that wait_for_idle times out properly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_test_state(Path(tmpdir))

            with UITestHarness(state) as harness:
                harness.start_headless()

                # Manually set thinking to True
                harness.ui.thinking = True

                # Should timeout
                with self.assertRaises(TimeoutError):
                    harness.wait_for_idle(timeout=0.2)

    def test_wait_for_menu_timeout(self):
        """Test that wait_for_menu times out properly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_test_state(Path(tmpdir))

            with UITestHarness(state) as harness:
                harness.start_headless()

                # Menu is not visible
                with self.assertRaises(TimeoutError):
                    harness.wait_for_menu(timeout=0.2)


if __name__ == "__main__":
    unittest.main()
