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

"""Tests for the UI test harness and event system."""

import tempfile
import unittest
from pathlib import Path

from scriptchat.core.commands import AppState
from scriptchat.core.config import Config, ModelConfig, ProviderConfig
from scriptchat.core.conversations import Conversation, Message
from scriptchat.ui.events import UIEventType, UIEventEmitter, UIEvent, parse_ui_events_from_log
from scriptchat.ui.test_harness import UITestHarness, run_ui_test


def make_test_state(tmp_path: Path):
    """Create a test AppState."""
    provider = ProviderConfig(
        id="test",
        type="ollama",
        api_url="http://localhost:11434/api",
        api_key="",
        models=[ModelConfig(name="test-model", context=1024)],
        streaming=True,
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


class UIEventEmitterTests(unittest.TestCase):
    """Tests for UIEventEmitter."""

    def test_emit_and_get_events(self):
        emitter = UIEventEmitter()
        emitter.emit(UIEventType.CONVERSATION_UPDATED, message_count=5)
        emitter.emit(UIEventType.THINKING_STARTED)

        all_events = emitter.get_events()
        self.assertEqual(len(all_events), 2)

        conv_events = emitter.get_events(UIEventType.CONVERSATION_UPDATED)
        self.assertEqual(len(conv_events), 1)
        self.assertEqual(conv_events[0].data['message_count'], 5)

    def test_get_last_event(self):
        emitter = UIEventEmitter()
        emitter.emit(UIEventType.INPUT_CHANGED, text="a")
        emitter.emit(UIEventType.INPUT_CHANGED, text="ab")
        emitter.emit(UIEventType.INPUT_CHANGED, text="abc")

        last = emitter.get_last_event(UIEventType.INPUT_CHANGED)
        self.assertEqual(last.data['text'], "abc")

    def test_listener_notification(self):
        emitter = UIEventEmitter()
        received = []

        def listener(event):
            received.append(event)

        emitter.add_listener(listener)
        emitter.emit(UIEventType.THINKING_STARTED)

        self.assertEqual(len(received), 1)
        self.assertEqual(received[0].type, UIEventType.THINKING_STARTED)

    def test_remove_listener(self):
        emitter = UIEventEmitter()
        received = []

        def listener(event):
            received.append(event)

        emitter.add_listener(listener)
        emitter.emit(UIEventType.THINKING_STARTED)
        self.assertEqual(len(received), 1)

        emitter.remove_listener(listener)
        emitter.emit(UIEventType.THINKING_STOPPED)
        self.assertEqual(len(received), 1)  # Still 1, no new events

    def test_event_to_log_line(self):
        event = UIEvent(
            type=UIEventType.MESSAGE_SENT,
            timestamp="2024-01-01T12:00:00",
            data={"content": "hello"}
        )
        line = event.to_log_line()
        self.assertIn("UI_EVENT|", line)
        self.assertIn("MESSAGE_SENT", line)
        self.assertIn("hello", line)

    def test_parse_event_from_log_line(self):
        line = 'UI_EVENT|2024-01-01T12:00:00|MESSAGE_SENT|{"content": "hello"}'
        event = UIEvent.from_log_line(line)
        self.assertIsNotNone(event)
        self.assertEqual(event.type, UIEventType.MESSAGE_SENT)
        self.assertEqual(event.data['content'], "hello")

    def test_parse_invalid_log_line(self):
        event = UIEvent.from_log_line("not a valid event line")
        self.assertIsNone(event)

    def test_parse_ui_events_from_log(self):
        log_content = """
2024-01-01 12:00:00 INFO UI_EVENT|2024-01-01T12:00:00|THINKING_STARTED|{}
2024-01-01 12:00:01 INFO Some other log line
2024-01-01 12:00:02 INFO UI_EVENT|2024-01-01T12:00:02|THINKING_STOPPED|{}
"""
        events = parse_ui_events_from_log(log_content)
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0].type, UIEventType.THINKING_STARTED)
        self.assertEqual(events[1].type, UIEventType.THINKING_STOPPED)

    def test_clear_events(self):
        emitter = UIEventEmitter()
        emitter.emit(UIEventType.THINKING_STARTED)
        emitter.emit(UIEventType.THINKING_STOPPED)
        self.assertEqual(len(emitter.get_events()), 2)

        emitter.clear()
        self.assertEqual(len(emitter.get_events()), 0)

    def test_max_log_size_limit(self):
        emitter = UIEventEmitter()
        emitter._max_log_size = 5  # Set a small limit for testing

        for i in range(10):
            emitter.emit(UIEventType.INPUT_CHANGED, text=str(i))

        events = emitter.get_events()
        self.assertEqual(len(events), 5)
        # Should have the last 5 events
        self.assertEqual(events[0].data['text'], "5")
        self.assertEqual(events[4].data['text'], "9")


class UITestHarnessTests(unittest.TestCase):
    """Tests for UITestHarness."""

    def test_component_mode_setup(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_test_state(Path(tmpdir))
            harness = UITestHarness(state)
            harness.setup_component_mode()

            self.assertIsNotNone(harness.ui)
            self.assertFalse(harness.ui.thinking)

    def test_get_state(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_test_state(Path(tmpdir))
            harness = UITestHarness(state)
            harness.setup_component_mode()

            ui_state = harness.get_state()
            self.assertEqual(ui_state.model_name, "test-model")
            self.assertFalse(ui_state.thinking)
            self.assertEqual(ui_state.conversation_message_count, 0)

    def test_add_message_updates_display(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_test_state(Path(tmpdir))
            harness = UITestHarness(state)
            harness.setup_component_mode()

            harness.add_message('user', 'hello world')

            # Check message was added
            self.assertEqual(len(state.current_conversation.messages), 1)

            # Check display was updated
            conv_text = harness.get_conversation_text()
            self.assertIn('hello world', conv_text)

    def test_set_and_get_input(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_test_state(Path(tmpdir))
            harness = UITestHarness(state)
            harness.setup_component_mode()

            harness.set_input("test input")
            self.assertEqual(harness.get_input_text(), "test input")

            harness.clear_input()
            self.assertEqual(harness.get_input_text(), "")

    def test_selection_menu_navigation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_test_state(Path(tmpdir))
            harness = UITestHarness(state)
            harness.setup_component_mode()

            selected_value = [None]
            def on_select(value):
                selected_value[0] = value

            harness.show_selection_menu([
                ("a", "Option A"),
                ("b", "Option B"),
                ("c", "Option C"),
            ], on_select)

            self.assertTrue(harness.ui.selection_menu.is_visible)
            self.assertEqual(harness.ui.selection_menu.selected_index, 0)

            harness.navigate_menu_down()
            self.assertEqual(harness.ui.selection_menu.selected_index, 1)

            harness.select_menu_item()
            self.assertFalse(harness.ui.selection_menu.is_visible)
            self.assertEqual(selected_value[0], "b")

    def test_assertions_pass(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_test_state(Path(tmpdir))
            harness = UITestHarness(state)
            harness.setup_component_mode()

            harness.add_message('user', 'hello')
            harness.add_message('assistant', 'hi there')

            harness.assert_conversation_contains('hello')
            harness.assert_conversation_contains('hi there')
            harness.assert_conversation_not_contains('goodbye')
            harness.assert_thinking(False)
            harness.assert_message_count(2)

    def test_assertions_fail_correctly(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_test_state(Path(tmpdir))
            harness = UITestHarness(state)
            harness.setup_component_mode()

            with self.assertRaises(AssertionError):
                harness.assert_conversation_contains('nonexistent')

    def test_events_emitted(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_test_state(Path(tmpdir))
            harness = UITestHarness(state)
            harness.setup_component_mode()

            harness.add_message('user', 'test')

            events = harness.ui.events.get_events(UIEventType.CONVERSATION_UPDATED)
            self.assertGreater(len(events), 0)

    def test_simulate_exchange(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_test_state(Path(tmpdir))
            harness = UITestHarness(state)
            harness.setup_component_mode()

            harness.simulate_user_message_flow("What is 2+2?", "4")

            harness.assert_message_count(2)
            harness.assert_conversation_contains("2+2")
            harness.assert_conversation_contains("4")

            # Check events
            harness.assert_event_occurred(UIEventType.MESSAGE_SENT)
            harness.assert_event_occurred(UIEventType.THINKING_STARTED)
            harness.assert_event_occurred(UIEventType.THINKING_STOPPED)
            harness.assert_event_occurred(UIEventType.RESPONSE_COMPLETE)

    def test_get_conversation_messages(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_test_state(Path(tmpdir))
            harness = UITestHarness(state)
            harness.setup_component_mode()

            harness.add_message('user', 'hello')
            harness.add_message('assistant', 'hi')

            messages = harness.get_conversation_messages()
            self.assertEqual(len(messages), 2)
            self.assertEqual(messages[0].role, 'user')
            self.assertEqual(messages[1].role, 'assistant')

    def test_cancel_menu(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_test_state(Path(tmpdir))
            harness = UITestHarness(state)
            harness.setup_component_mode()

            cancelled = [False]
            def on_cancel():
                cancelled[0] = True

            harness.ui.selection_menu.show(
                items=[("a", "A"), ("b", "B")],
                on_select=lambda x: None,
                on_cancel=on_cancel
            )

            harness.cancel_menu()
            self.assertFalse(harness.ui.selection_menu.is_visible)
            self.assertTrue(cancelled[0])


class RunUITestTests(unittest.TestCase):
    """Tests for run_ui_test helper."""

    def test_passing_test(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_test_state(Path(tmpdir))

            def test_fn(harness):
                harness.add_message('user', 'hello')
                harness.assert_message_count(1)

            result = run_ui_test("test_hello", state, test_fn)
            self.assertTrue(result.passed)
            self.assertEqual(result.test_name, "test_hello")
            self.assertIsNone(result.error)

    def test_failing_test(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_test_state(Path(tmpdir))

            def test_fn(harness):
                harness.assert_message_count(99)  # Will fail

            result = run_ui_test("test_fail", state, test_fn)
            self.assertFalse(result.passed)
            self.assertIsNotNone(result.error)

    def test_result_includes_events(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_test_state(Path(tmpdir))

            def test_fn(harness):
                harness.add_message('user', 'hello')

            result = run_ui_test("test_events", state, test_fn)
            self.assertIsNotNone(result.events)
            self.assertGreater(len(result.events), 0)

    def test_result_includes_duration(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_test_state(Path(tmpdir))

            def test_fn(harness):
                pass

            result = run_ui_test("test_duration", state, test_fn)
            self.assertGreaterEqual(result.duration_ms, 0)


class ThinkingContentDisplayTests(unittest.TestCase):
    """Tests for thinking/reasoning content display."""

    def test_thinking_content_displayed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_test_state(Path(tmpdir))
            harness = UITestHarness(state)
            harness.setup_component_mode()

            harness.add_message(
                'assistant',
                'The answer is 4.',
                thinking='Let me calculate: 2 + 2 = 4'
            )

            conv_text = harness.get_conversation_text()
            self.assertIn('<thinking>', conv_text)
            self.assertIn('Let me calculate', conv_text)
            self.assertIn('The answer is 4', conv_text)

    def test_thinking_content_in_gray(self):
        """Thinking content should use gray ANSI codes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_test_state(Path(tmpdir))
            harness = UITestHarness(state)
            harness.setup_component_mode()

            harness.add_message('assistant', 'response', thinking='reasoning')

            conv_text = harness.get_conversation_text()
            # Gray ANSI code
            self.assertIn('\033[90m', conv_text)


class MarkdownRenderingTests(unittest.TestCase):
    """Tests for markdown rendering in conversation display."""

    def test_bold_rendering(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_test_state(Path(tmpdir))
            harness = UITestHarness(state)
            harness.setup_component_mode()

            harness.add_message('assistant', 'This is **bold** text')

            conv_text = harness.get_conversation_text()
            # Bold ANSI code
            self.assertIn('\033[1m', conv_text)
            self.assertIn('bold', conv_text)
            self.assertNotIn('**', conv_text)

    def test_code_rendering(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_test_state(Path(tmpdir))
            harness = UITestHarness(state)
            harness.setup_component_mode()

            harness.add_message('assistant', 'Use `code` here')

            conv_text = harness.get_conversation_text()
            # Cyan ANSI code
            self.assertIn('\033[96m', conv_text)
            self.assertIn('code', conv_text)
            self.assertNotIn('`', conv_text)

    def test_header_rendering(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_test_state(Path(tmpdir))
            harness = UITestHarness(state)
            harness.setup_component_mode()

            harness.add_message('assistant', '## Section Header')

            conv_text = harness.get_conversation_text()
            self.assertIn('\033[1m', conv_text)  # Bold for headers
            self.assertIn('Section Header', conv_text)
            self.assertNotIn('##', conv_text)


class SelectionMenuEventTests(unittest.TestCase):
    """Tests for selection menu event emissions."""

    def test_menu_show_emits_event(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_test_state(Path(tmpdir))
            harness = UITestHarness(state)
            harness.setup_component_mode()

            harness.show_selection_menu([("a", "Item A"), ("b", "Item B")])

            events = harness.ui.events.get_events(UIEventType.SELECTION_MENU_SHOWN)
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0].data['item_count'], 2)

    def test_menu_navigate_emits_event(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_test_state(Path(tmpdir))
            harness = UITestHarness(state)
            harness.setup_component_mode()

            harness.show_selection_menu([("a", "Item A"), ("b", "Item B")])
            harness.navigate_menu_down()

            events = harness.ui.events.get_events(UIEventType.SELECTION_MENU_NAVIGATED)
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0].data['direction'], 'down')
            self.assertEqual(events[0].data['index'], 1)

    def test_menu_select_emits_event(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_test_state(Path(tmpdir))
            harness = UITestHarness(state)
            harness.setup_component_mode()

            harness.show_selection_menu([("a", "Item A"), ("b", "Item B")])
            harness.select_menu_item()

            events = harness.ui.events.get_events(UIEventType.SELECTION_MENU_SELECTED)
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0].data['label'], 'Item A')

    def test_menu_hide_emits_event(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_test_state(Path(tmpdir))
            harness = UITestHarness(state)
            harness.setup_component_mode()

            harness.show_selection_menu([("a", "Item A")])
            harness.cancel_menu()

            events = harness.ui.events.get_events(UIEventType.SELECTION_MENU_HIDDEN)
            self.assertGreater(len(events), 0)


if __name__ == "__main__":
    unittest.main()
