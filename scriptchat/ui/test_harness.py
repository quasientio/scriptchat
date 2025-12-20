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

"""UI test harness for automated testing without a real terminal."""

import logging
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

from ..core.commands import AppState
from ..core.conversations import Message
from .events import UIEvent, UIEventEmitter, UIEventType, UIState

logger = logging.getLogger(__name__)


class UITestHarness:
    """Test harness for driving and inspecting the UI programmatically.

    This harness can operate in two modes:
    1. Component testing: Test UI logic without running the full app
    2. Integration testing: Run the app with mock I/O (headless mode)

    Example usage (component testing):
        harness = UITestHarness(state)
        harness.setup_component_mode()

        # Test conversation display
        state.current_conversation.messages.append(
            Message(role='user', content='hello')
        )
        harness.ui.update_conversation_display()

        assert 'hello' in harness.get_conversation_text()

    Example usage (integration testing):
        harness = UITestHarness(state)
        harness.start_headless()

        harness.type_and_enter("hello")
        harness.wait_for_event(UIEventType.RESPONSE_COMPLETE, timeout=30)

        assert 'hello' in harness.get_conversation_text()
        harness.stop()
    """

    def __init__(self, state: AppState):
        self.state = state
        self.ui = None
        self._app_thread: Optional[threading.Thread] = None
        self._running = False

    def setup_component_mode(self):
        """Set up for component testing (no app loop)."""
        # Import here to avoid circular imports
        from .app import ScriptChatUI

        # Create UI object without running __init__
        self.ui = object.__new__(ScriptChatUI)

        # Initialize essential attributes
        self.ui.state = self.state
        self.ui.events = UIEventEmitter(log_events=True)
        self.ui.thinking = False
        self.ui.thinking_dots = 0
        self.ui.multiline_mode = False
        self.ui.prompt_message = ""
        self.ui.pending_callback = None
        self.ui.expecting_single_key = False
        self.ui.inference_cancelled = False
        self.ui.cancel_requested_at = None
        self.ui.input_history = []
        self.ui.input_history_index = None
        self.ui.message_queue = []
        self.ui.script_queue = []
        self.ui.running_script = False

        # Create mock buffers
        from prompt_toolkit.buffer import Buffer
        from prompt_toolkit.document import Document

        self.ui.input_buffer = Buffer()
        self.ui.conversation_buffer = Buffer()

        # Create mock selection menu
        from .selection_menu import SelectionMenu

        class MockApp:
            def invalidate(self):
                pass

        mock_app = MockApp()
        self.ui.app = mock_app
        self.ui.selection_menu = SelectionMenu(self.ui)

        # Import command handlers
        from .command_handlers import CommandHandlers
        self.ui.handlers = CommandHandlers(self.ui)

        return self

    def start_headless(self):
        """Start the UI in headless mode for integration testing."""
        # This requires more complex setup with prompt_toolkit mock I/O
        # For now, use component mode for most testing
        raise NotImplementedError(
            "Headless integration testing not yet implemented. "
            "Use setup_component_mode() for component testing."
        )

    def stop(self):
        """Stop the UI if running."""
        self._running = False
        if self.ui and hasattr(self.ui, 'app') and self.ui.app:
            try:
                self.ui.app.exit()
            except Exception:
                pass
        if self._app_thread:
            self._app_thread.join(timeout=2)

    # State queries

    def get_state(self) -> UIState:
        """Get complete UI state snapshot."""
        if not self.ui:
            raise RuntimeError("UI not initialized. Call setup_component_mode() first.")

        convo = self.state.current_conversation

        return UIState(
            # Conversation
            conversation_text=self.ui.conversation_buffer.text,
            conversation_message_count=len(convo.messages),

            # Status
            status_bar_text=self._get_status_text(),
            provider_id=convo.provider_id,
            model_name=convo.model_name,
            tokens_in=convo.tokens_in,
            tokens_out=convo.tokens_out,
            conversation_id=convo.id,

            # Input
            input_text=self.ui.input_buffer.text,
            input_cursor_position=self.ui.input_buffer.cursor_position,

            # Modes
            thinking=self.ui.thinking,
            multiline_mode=self.ui.multiline_mode,
            streaming=self.state.config.enable_streaming,

            # Selection menu
            selection_menu_visible=self.ui.selection_menu.is_visible,

            # Prompts
            prompt_message=self.ui.prompt_message,
            has_pending_callback=self.ui.pending_callback is not None,
            expecting_single_key=self.ui.expecting_single_key,

            # History
            history_index=self.ui.input_history_index,
            history_length=len(self.ui.input_history),

            # Script state
            running_script=self.ui.running_script,
            script_queue_length=len(self.ui.script_queue),
            message_queue_length=len(self.ui.message_queue),

            # Fields with defaults
            selection_menu_items=list(self.ui.selection_menu.items),
            selection_menu_index=self.ui.selection_menu.selected_index,
        )

    def _get_status_text(self) -> str:
        """Get status bar text without ANSI codes."""
        try:
            parts = self.ui._get_status_bar()
            return ''.join(text for style, text in parts)
        except Exception:
            return ""

    def get_conversation_text(self) -> str:
        """Get raw conversation buffer text."""
        return self.ui.conversation_buffer.text if self.ui else ""

    def get_conversation_messages(self) -> list[Message]:
        """Get conversation messages."""
        return list(self.state.current_conversation.messages)

    def get_input_text(self) -> str:
        """Get current input buffer text."""
        return self.ui.input_buffer.text if self.ui else ""

    # Actions

    def set_input(self, text: str):
        """Set input buffer content."""
        from prompt_toolkit.document import Document
        self.ui.input_buffer.set_document(
            Document(text=text, cursor_position=len(text))
        )
        self.ui.events.emit(UIEventType.INPUT_CHANGED, text=text)

    def clear_input(self):
        """Clear input buffer."""
        self.set_input("")
        self.ui.events.emit(UIEventType.INPUT_CLEARED)

    def add_message(self, role: str, content: str, thinking: str = None):
        """Add a message to conversation and update display."""
        self.state.current_conversation.messages.append(
            Message(role=role, content=content, thinking=thinking)
        )
        self.ui.update_conversation_display()

    def execute_command(self, command: str):
        """Execute a command through the UI handlers."""
        self.ui.handlers.handle_command(command)

    def simulate_user_message_flow(self, message: str, response: str):
        """Simulate a complete user message -> response flow.

        This is useful for testing UI state after an exchange.
        """
        # Add user message
        self.add_message('user', message)
        self.ui.events.emit(UIEventType.MESSAGE_SENT, content=message)

        # Simulate thinking
        self.ui.thinking = True
        self.ui.events.emit(UIEventType.THINKING_STARTED)

        # Add response
        self.add_message('assistant', response)

        # Stop thinking
        self.ui.thinking = False
        self.ui.events.emit(UIEventType.THINKING_STOPPED)
        self.ui.events.emit(UIEventType.RESPONSE_COMPLETE, content=response)

    # Selection menu helpers

    def show_selection_menu(self, items: list[tuple], on_select: Callable = None):
        """Show selection menu with items."""
        self.ui.selection_menu.show(
            items=items,
            on_select=on_select or (lambda x: None),
            on_cancel=lambda: None
        )

    def navigate_menu_down(self):
        """Navigate selection menu down."""
        self.ui.selection_menu.move_down()

    def navigate_menu_up(self):
        """Navigate selection menu up."""
        self.ui.selection_menu.move_up()

    def select_menu_item(self):
        """Select current menu item."""
        self.ui.selection_menu.select_current()

    def cancel_menu(self):
        """Cancel selection menu."""
        self.ui.selection_menu.cancel()

    # Assertions

    def assert_conversation_contains(self, text: str, msg: str = None):
        """Assert conversation display contains text."""
        content = self.get_conversation_text()
        assert text in content, msg or f"Expected '{text}' in conversation:\n{content[:500]}"

    def assert_conversation_not_contains(self, text: str, msg: str = None):
        """Assert conversation display does not contain text."""
        content = self.get_conversation_text()
        assert text not in content, msg or f"Unexpected '{text}' found in conversation"

    def assert_status_contains(self, text: str, msg: str = None):
        """Assert status bar contains text."""
        status = self._get_status_text()
        assert text in status, msg or f"Expected '{text}' in status: {status}"

    def assert_thinking(self, expected: bool, msg: str = None):
        """Assert thinking state."""
        actual = self.ui.thinking
        assert actual == expected, msg or f"Expected thinking={expected}, got {actual}"

    def assert_selection_menu_visible(self, expected: bool = True, msg: str = None):
        """Assert selection menu visibility."""
        actual = self.ui.selection_menu.is_visible
        assert actual == expected, msg or f"Expected menu visible={expected}, got {actual}"

    def assert_input_equals(self, expected: str, msg: str = None):
        """Assert input buffer content."""
        actual = self.get_input_text()
        assert actual == expected, msg or f"Expected input '{expected}', got '{actual}'"

    def assert_event_occurred(self, event_type: UIEventType, msg: str = None):
        """Assert that an event occurred."""
        events = self.ui.events.get_events(event_type)
        assert len(events) > 0, msg or f"Expected event {event_type.name} did not occur"

    def assert_message_count(self, expected: int, msg: str = None):
        """Assert number of messages in conversation."""
        actual = len(self.state.current_conversation.messages)
        assert actual == expected, msg or f"Expected {expected} messages, got {actual}"


@dataclass
class UITestResult:
    """Result of a UI test."""
    passed: bool
    test_name: str
    error: Optional[str] = None
    events: list[UIEvent] = None
    duration_ms: float = 0


def run_ui_test(
    name: str,
    state: AppState,
    test_fn: Callable[[UITestHarness], None]
) -> UITestResult:
    """Run a UI test function with a fresh harness.

    Example:
        def test_message_display(harness):
            harness.add_message('user', 'hello')
            harness.assert_conversation_contains('[user] hello')

        result = run_ui_test("message_display", state, test_message_display)
    """
    harness = UITestHarness(state)
    harness.setup_component_mode()

    start = time.time()
    try:
        test_fn(harness)
        duration = (time.time() - start) * 1000
        return UITestResult(
            passed=True,
            test_name=name,
            events=harness.ui.events.get_events(),
            duration_ms=duration
        )
    except AssertionError as e:
        duration = (time.time() - start) * 1000
        return UITestResult(
            passed=False,
            test_name=name,
            error=str(e),
            events=harness.ui.events.get_events(),
            duration_ms=duration
        )
    except Exception as e:
        duration = (time.time() - start) * 1000
        return UITestResult(
            passed=False,
            test_name=name,
            error=f"Unexpected error: {e}",
            events=harness.ui.events.get_events(),
            duration_ms=duration
        )
