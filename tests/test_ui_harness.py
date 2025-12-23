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
        file_registry={},
        folder_registry={}
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


class SelectionMenuScrollingTests(unittest.TestCase):
    """Tests for selection menu viewport scrolling."""

    def test_scroll_down_through_long_list(self):
        """Test scrolling down through a list longer than max_visible."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_test_state(Path(tmpdir))
            harness = UITestHarness(state)
            harness.setup_component_mode()

            # Create more items than max_visible (default 10)
            items = [(str(i), f"Item {i}") for i in range(15)]
            harness.show_selection_menu(items)

            # Initially at index 0, viewport_start 0
            self.assertEqual(harness.ui.selection_menu.selected_index, 0)
            self.assertEqual(harness.ui.selection_menu.viewport_start, 0)

            # Navigate down past viewport
            for _ in range(12):
                harness.navigate_menu_down()

            # Should be at index 12, viewport should have scrolled
            self.assertEqual(harness.ui.selection_menu.selected_index, 12)
            self.assertGreater(harness.ui.selection_menu.viewport_start, 0)

    def test_scroll_up_through_long_list(self):
        """Test scrolling up after scrolling down."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_test_state(Path(tmpdir))
            harness = UITestHarness(state)
            harness.setup_component_mode()

            items = [(str(i), f"Item {i}") for i in range(15)]
            harness.show_selection_menu(items)

            # Scroll down first
            for _ in range(12):
                harness.navigate_menu_down()

            # Now scroll back up
            for _ in range(10):
                harness.navigate_menu_up()

            self.assertEqual(harness.ui.selection_menu.selected_index, 2)

    def test_scroll_bounds(self):
        """Test that scrolling respects list bounds."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_test_state(Path(tmpdir))
            harness = UITestHarness(state)
            harness.setup_component_mode()

            items = [(str(i), f"Item {i}") for i in range(5)]
            harness.show_selection_menu(items)

            # Try to scroll up from start - should stay at 0
            harness.navigate_menu_up()
            self.assertEqual(harness.ui.selection_menu.selected_index, 0)

            # Go to end
            for _ in range(10):  # More than needed
                harness.navigate_menu_down()

            # Should be at last item
            self.assertEqual(harness.ui.selection_menu.selected_index, 4)

    def test_menu_item_styling_classes(self):
        """Test that menu items have correct style classes for highlighting."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_test_state(Path(tmpdir))
            harness = UITestHarness(state)
            harness.setup_component_mode()

            items = [("a", "Item A"), ("b", "Item B"), ("c", "Item C")]
            harness.show_selection_menu(items)

            # Get the formatted text from the menu
            menu_text = harness.ui.selection_menu._get_menu_text()

            # Extract style classes for menu items (skip border lines)
            item_styles = [
                (style, text) for style, text in menu_text
                if style in ('class:menu-selected', 'class:menu-item')
            ]

            # First item (index 0) should be selected
            self.assertEqual(len(item_styles), 3)
            self.assertEqual(item_styles[0][0], 'class:menu-selected')
            self.assertEqual(item_styles[1][0], 'class:menu-item')
            self.assertEqual(item_styles[2][0], 'class:menu-item')

            # Navigate down - now second item should be selected
            harness.navigate_menu_down()
            menu_text = harness.ui.selection_menu._get_menu_text()
            item_styles = [
                (style, text) for style, text in menu_text
                if style in ('class:menu-selected', 'class:menu-item')
            ]

            self.assertEqual(item_styles[0][0], 'class:menu-item')
            self.assertEqual(item_styles[1][0], 'class:menu-selected')
            self.assertEqual(item_styles[2][0], 'class:menu-item')

    def test_menu_border_and_hint_styling(self):
        """Test that menu borders and hints have correct style classes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_test_state(Path(tmpdir))
            harness = UITestHarness(state)
            harness.setup_component_mode()

            items = [("a", "Item A"), ("b", "Item B")]
            harness.show_selection_menu(items)

            menu_text = harness.ui.selection_menu._get_menu_text()
            styles_used = {style for style, text in menu_text}

            # Should have border and hint styles
            self.assertIn('class:menu-border', styles_used)
            self.assertIn('class:menu-hint', styles_used)

    def test_menu_initial_index_selects_correct_item(self):
        """Test that initial_index parameter selects the correct item."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_test_state(Path(tmpdir))
            harness = UITestHarness(state)
            harness.setup_component_mode()

            items = [("a", "Item A"), ("b", "Item B"), ("c", "Item C"), ("d", "Item D")]

            # Show menu starting at index 2 (Item C)
            harness.show_selection_menu(items, initial_index=2)

            # Third item should be selected
            self.assertEqual(harness.ui.selection_menu.selected_index, 2)

            # Check styling reflects selection
            menu_text = harness.ui.selection_menu._get_menu_text()
            item_styles = [
                (style, text) for style, text in menu_text
                if style in ('class:menu-selected', 'class:menu-item')
            ]

            self.assertEqual(len(item_styles), 4)
            self.assertEqual(item_styles[0][0], 'class:menu-item')  # Item A
            self.assertEqual(item_styles[1][0], 'class:menu-item')  # Item B
            self.assertEqual(item_styles[2][0], 'class:menu-selected')  # Item C
            self.assertEqual(item_styles[3][0], 'class:menu-item')  # Item D

    def test_menu_initial_index_clamps_to_bounds(self):
        """Test that initial_index is clamped to valid range."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_test_state(Path(tmpdir))
            harness = UITestHarness(state)
            harness.setup_component_mode()

            items = [("a", "Item A"), ("b", "Item B"), ("c", "Item C")]

            # Test with index beyond bounds
            harness.show_selection_menu(items, initial_index=100)
            self.assertEqual(harness.ui.selection_menu.selected_index, 2)  # Last item

            # Test with negative index
            harness.ui.selection_menu.hide()
            harness.show_selection_menu(items, initial_index=-5)
            self.assertEqual(harness.ui.selection_menu.selected_index, 0)  # First item

    def test_menu_initial_index_adjusts_viewport(self):
        """Test that viewport scrolls to show initially selected item."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_test_state(Path(tmpdir))
            harness = UITestHarness(state)
            harness.setup_component_mode()

            # Create menu with more items than visible (default max_visible=10)
            items = [(str(i), f"Item {i}") for i in range(15)]

            # Select item 12 (beyond initial viewport)
            harness.show_selection_menu(items, initial_index=12)

            # Viewport should have scrolled
            menu = harness.ui.selection_menu
            self.assertEqual(menu.selected_index, 12)
            # Viewport start should be adjusted so item 12 is visible
            self.assertGreater(menu.viewport_start, 0)
            self.assertLessEqual(menu.viewport_start, 12)
            self.assertGreater(menu.viewport_start + menu.max_visible, 12)


class HistoryNavigationComponentTests(unittest.TestCase):
    """Tests for input history navigation in component mode."""

    def test_history_previous_sets_input(self):
        """Test navigating to previous history entry."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_test_state(Path(tmpdir))
            harness = UITestHarness(state)
            harness.setup_component_mode()

            # Add some history
            harness.ui.input_history = ["first", "second", "third"]
            harness.ui.input_history_index = None

            # Navigate up
            harness.ui._history_previous()

            # Should show most recent
            self.assertEqual(harness.get_input_text(), "third")
            self.assertEqual(harness.ui.input_history_index, 2)

    def test_history_next_cycles_forward(self):
        """Test navigating forward through history."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_test_state(Path(tmpdir))
            harness = UITestHarness(state)
            harness.setup_component_mode()

            harness.ui.input_history = ["first", "second", "third"]
            harness.ui.input_history_index = None

            # Navigate up twice
            harness.ui._history_previous()
            harness.ui._history_previous()
            self.assertEqual(harness.get_input_text(), "second")

            # Navigate forward
            harness.ui._history_next()
            self.assertEqual(harness.get_input_text(), "third")

    def test_history_next_past_end_clears(self):
        """Test that navigating past newest clears input."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_test_state(Path(tmpdir))
            harness = UITestHarness(state)
            harness.setup_component_mode()

            harness.ui.input_history = ["first", "second"]
            harness.ui.input_history_index = 1  # At "second"
            harness.ui.input_buffer.text = "second"

            # Navigate forward past end
            harness.ui._history_next()

            self.assertEqual(harness.get_input_text(), "")
            self.assertIsNone(harness.ui.input_history_index)

    def test_empty_history_no_crash(self):
        """Test history navigation with empty history."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_test_state(Path(tmpdir))
            harness = UITestHarness(state)
            harness.setup_component_mode()

            harness.ui.input_history = []

            # Should not crash
            harness.ui._history_previous()
            harness.ui._history_next()

            self.assertEqual(harness.get_input_text(), "")


class MessageRoleRenderingTests(unittest.TestCase):
    """Tests for different message role rendering."""

    def test_echo_message_rendering(self):
        """Test echo messages render in yellow without prefix."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_test_state(Path(tmpdir))
            harness = UITestHarness(state)
            harness.setup_component_mode()

            harness.add_message('echo', 'Echo content here')

            conv_text = harness.get_conversation_text()
            # Yellow ANSI code
            self.assertIn('\033[93m', conv_text)
            self.assertIn('Echo content here', conv_text)
            # No [echo] prefix
            self.assertNotIn('[echo]', conv_text)

    def test_note_message_rendering(self):
        """Test note messages render in magenta with prefix."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_test_state(Path(tmpdir))
            harness = UITestHarness(state)
            harness.setup_component_mode()

            harness.add_message('note', 'Note content here')

            conv_text = harness.get_conversation_text()
            # Magenta ANSI code
            self.assertIn('\033[95m', conv_text)
            self.assertIn('[note]', conv_text)
            self.assertIn('Note content here', conv_text)

    def test_system_error_rendering(self):
        """Test system messages starting with Error are red."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_test_state(Path(tmpdir))
            harness = UITestHarness(state)
            harness.setup_component_mode()

            harness.add_message('system', 'Error: Something went wrong')

            conv_text = harness.get_conversation_text()
            # Red ANSI code
            self.assertIn('\033[91m', conv_text)

    def test_status_error_rendering(self):
        """Test status messages starting with Error are red."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_test_state(Path(tmpdir))
            harness = UITestHarness(state)
            harness.setup_component_mode()

            harness.add_message('status', 'Error: Connection failed')

            conv_text = harness.get_conversation_text()
            # Red ANSI code
            self.assertIn('\033[91m', conv_text)

    def test_user_message_rendering(self):
        """Test user messages render in cyan."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_test_state(Path(tmpdir))
            harness = UITestHarness(state)
            harness.setup_component_mode()

            harness.add_message('user', 'User input')

            conv_text = harness.get_conversation_text()
            # Cyan ANSI code
            self.assertIn('\033[96m', conv_text)
            self.assertIn('[user]', conv_text)

    def test_assistant_message_rendering(self):
        """Test assistant messages render in green."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_test_state(Path(tmpdir))
            harness = UITestHarness(state)
            harness.setup_component_mode()

            harness.add_message('assistant', 'Assistant response')

            conv_text = harness.get_conversation_text()
            # Green ANSI code
            self.assertIn('\033[92m', conv_text)
            self.assertIn('[assistant]', conv_text)


class UIStateSnapshotTests(unittest.TestCase):
    """Tests for UIState snapshot completeness."""

    def test_state_captures_all_fields(self):
        """Test that get_state captures all UI state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_test_state(Path(tmpdir))
            harness = UITestHarness(state)
            harness.setup_component_mode()

            # Set various state values
            harness.set_input("test input")
            harness.add_message('user', 'hello')
            harness.ui.thinking = True
            harness.ui.multiline_mode = True
            harness.ui.prompt_message = "Confirm?"

            ui_state = harness.get_state()

            # Verify all fields
            self.assertEqual(ui_state.input_text, "test input")
            self.assertEqual(ui_state.conversation_message_count, 1)
            self.assertTrue(ui_state.thinking)
            self.assertTrue(ui_state.multiline_mode)
            self.assertEqual(ui_state.prompt_message, "Confirm?")
            self.assertEqual(ui_state.model_name, "test-model")
            self.assertEqual(ui_state.provider_id, "test")

    def test_state_captures_selection_menu(self):
        """Test that state captures selection menu state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_test_state(Path(tmpdir))
            harness = UITestHarness(state)
            harness.setup_component_mode()

            items = [("a", "A"), ("b", "B"), ("c", "C")]
            harness.show_selection_menu(items)
            harness.navigate_menu_down()

            ui_state = harness.get_state()

            self.assertTrue(ui_state.selection_menu_visible)
            self.assertEqual(ui_state.selection_menu_index, 1)
            self.assertEqual(len(ui_state.selection_menu_items), 3)


class UIEventEmitterAdvancedTests(unittest.TestCase):
    """Tests for advanced UIEventEmitter features."""

    def test_wait_for_event_immediate(self):
        """Test wait_for_event returns immediately if event fires."""
        import threading

        emitter = UIEventEmitter()

        def emit_soon():
            import time
            time.sleep(0.05)
            emitter.emit(UIEventType.THINKING_STOPPED)

        threading.Thread(target=emit_soon, daemon=True).start()

        event = emitter.wait_for_event(UIEventType.THINKING_STOPPED, timeout=1.0)
        self.assertIsNotNone(event)
        self.assertEqual(event.type, UIEventType.THINKING_STOPPED)

    def test_wait_for_event_timeout(self):
        """Test wait_for_event returns None on timeout."""
        emitter = UIEventEmitter()

        event = emitter.wait_for_event(UIEventType.THINKING_STOPPED, timeout=0.1)
        self.assertIsNone(event)

    def test_wait_for_event_with_predicate(self):
        """Test wait_for_event with predicate filter."""
        import threading

        emitter = UIEventEmitter()

        def emit_events():
            import time
            time.sleep(0.05)
            emitter.emit(UIEventType.INPUT_CHANGED, text="a")
            time.sleep(0.05)
            emitter.emit(UIEventType.INPUT_CHANGED, text="abc")

        threading.Thread(target=emit_events, daemon=True).start()

        # Wait for input with text length > 2
        event = emitter.wait_for_event(
            UIEventType.INPUT_CHANGED,
            timeout=1.0,
            predicate=lambda e: len(e.data.get('text', '')) > 2
        )
        self.assertIsNotNone(event)
        self.assertEqual(event.data['text'], "abc")

    def test_get_events_with_since_filter(self):
        """Test filtering events by timestamp."""
        from datetime import datetime
        import time

        emitter = UIEventEmitter()

        emitter.emit(UIEventType.THINKING_STARTED)
        time.sleep(0.01)
        cutoff = datetime.now().isoformat()
        time.sleep(0.01)
        emitter.emit(UIEventType.THINKING_STOPPED)

        # Get events since cutoff
        events = emitter.get_events(since=cutoff)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].type, UIEventType.THINKING_STOPPED)

    def test_listener_exception_doesnt_crash(self):
        """Test that listener exceptions don't crash emit."""
        emitter = UIEventEmitter()
        received = []

        def bad_listener(event):
            raise ValueError("Intentional error")

        def good_listener(event):
            received.append(event)

        emitter.add_listener(bad_listener)
        emitter.add_listener(good_listener)

        # Should not raise
        emitter.emit(UIEventType.THINKING_STARTED)

        # Good listener should still receive event
        self.assertEqual(len(received), 1)


class MarkdownAdvancedRenderingTests(unittest.TestCase):
    """Tests for edge cases in markdown rendering."""

    def test_multiple_bold_in_line(self):
        """Test multiple bold segments in one line."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_test_state(Path(tmpdir))
            harness = UITestHarness(state)
            harness.setup_component_mode()

            harness.add_message('assistant', 'This is **bold** and **also bold**')

            conv_text = harness.get_conversation_text()
            self.assertIn('bold', conv_text)
            self.assertIn('also bold', conv_text)
            self.assertNotIn('**', conv_text)

    def test_code_not_spanning_newlines(self):
        """Test that code backticks don't span newlines."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_test_state(Path(tmpdir))
            harness = UITestHarness(state)
            harness.setup_component_mode()

            # Backticks on different lines should not match
            harness.add_message('assistant', 'Start `code\nend` here')

            conv_text = harness.get_conversation_text()
            # The backticks should remain since they span newlines
            self.assertIn('`', conv_text)

    def test_headers_at_different_levels(self):
        """Test headers from h1 to h6."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state = make_test_state(Path(tmpdir))
            harness = UITestHarness(state)
            harness.setup_component_mode()

            harness.add_message('assistant', '# H1\n## H2\n### H3')

            conv_text = harness.get_conversation_text()
            self.assertIn('H1', conv_text)
            self.assertIn('H2', conv_text)
            self.assertIn('H3', conv_text)
            # No hash marks
            self.assertNotIn('#', conv_text)


class ResolveDelTargetTests(unittest.TestCase):
    """Tests for resolve_del_target_from_args helper."""

    def test_no_args_deletes_current(self):
        """Test /del with no args targets current conversation."""
        from scriptchat.ui.app import resolve_del_target_from_args

        with tempfile.TemporaryDirectory() as tmpdir:
            conversations_root = Path(tmpdir)

            target_id, prompt, error, _, target_is_current = resolve_del_target_from_args(
                "",
                conversations_root,
                "current-convo-id"
            )

            self.assertEqual(target_id, "current-convo-id")
            self.assertTrue(target_is_current)
            self.assertIsNone(error)
            self.assertIn("current-convo-id", prompt)

    def test_invalid_index_returns_error(self):
        """Test that invalid index returns error message."""
        from scriptchat.ui.app import resolve_del_target_from_args

        with tempfile.TemporaryDirectory() as tmpdir:
            conversations_root = Path(tmpdir)

            target_id, prompt, error, _, _ = resolve_del_target_from_args(
                "999",  # No conversations exist
                conversations_root,
                None
            )

            self.assertIsNone(target_id)
            self.assertIsNotNone(error)

    def test_non_numeric_returns_usage_error(self):
        """Test that non-numeric arg returns usage error."""
        from scriptchat.ui.app import resolve_del_target_from_args

        with tempfile.TemporaryDirectory() as tmpdir:
            conversations_root = Path(tmpdir)

            target_id, prompt, error, _, _ = resolve_del_target_from_args(
                "abc",
                conversations_root,
                None
            )

            self.assertIsNone(target_id)
            self.assertIn("Usage", error)


if __name__ == "__main__":
    unittest.main()
