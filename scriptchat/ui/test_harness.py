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
from typing import Any, Callable, Optional

from ..core.commands import AppState
from ..core.conversations import Conversation, Message
from .events import UIEvent, UIEventEmitter, UIEventType, UIState

logger = logging.getLogger(__name__)


# Escape sequences for special keys
KEY_SEQUENCES = {
    'enter': '\r',
    'escape': '\x1b',
    'tab': '\t',
    'shift-tab': '\x1b[Z',
    'backspace': '\x7f',
    'delete': '\x1b[3~',
    'up': '\x1b[A',
    'down': '\x1b[B',
    'left': '\x1b[D',
    'right': '\x1b[C',
    'home': '\x1b[H',
    'end': '\x1b[F',
    'page-up': '\x1b[5~',
    'page-down': '\x1b[6~',
    'ctrl-a': '\x01',
    'ctrl-b': '\x02',
    'ctrl-c': '\x03',
    'ctrl-d': '\x04',
    'ctrl-e': '\x05',
    'ctrl-f': '\x06',
    'ctrl-j': '\n',
    'ctrl-k': '\x0b',
    'ctrl-l': '\x0c',
    'ctrl-n': '\x0e',
    'ctrl-p': '\x10',
    'ctrl-u': '\x15',
    'ctrl-w': '\x17',
    'ctrl-up': '\x1b[1;5A',
    'ctrl-down': '\x1b[1;5B',
    'ctrl-left': '\x1b[1;5D',
    'ctrl-right': '\x1b[1;5C',
    'ctrl-home': '\x1b[1;5H',
    'ctrl-end': '\x1b[1;5F',
    'alt-enter': '\x1b\r',
    'alt-up': '\x1b[1;3A',
    'alt-down': '\x1b[1;3B',
}


class MockLLMClient:
    """Mock LLM client for testing without real API calls."""

    def __init__(
        self,
        responses: dict[str, str] | None = None,
        default_response: str = "Mock response",
        thinking_responses: dict[str, str] | None = None,
        response_delay: float | None = None
    ):
        """Initialize mock client.

        Args:
            responses: Dict mapping user message patterns to responses
            default_response: Response when no pattern matches
            thinking_responses: Dict mapping patterns to thinking content
            response_delay: Delay in seconds before responding (overrides default)
        """
        self.responses = responses or {}
        self.default_response = default_response
        self.thinking_responses = thinking_responses or {}
        self.calls: list[dict] = []
        self.delay: float = response_delay if response_delay is not None else 0.1

    def chat(
        self,
        convo: Conversation,
        new_user_message: str,
        streaming: bool = False,
        on_chunk: Callable = None,
        expanded_history: list = None
    ) -> str:
        """Mock chat method."""
        self.calls.append({
            'message': new_user_message,
            'streaming': streaming,
            'model': convo.model_name,
        })

        # Simulate thinking time
        time.sleep(self.delay)

        # Find matching response and thinking
        response = self.default_response
        thinking = None
        matched_pattern = None

        for pattern, resp in self.responses.items():
            if pattern.lower() in new_user_message.lower():
                response = resp
                matched_pattern = pattern
                break

        # Check for matching thinking content
        if matched_pattern and matched_pattern in self.thinking_responses:
            thinking = self.thinking_responses[matched_pattern]

        # Add to conversation
        msg = Message(role='assistant', content=response)
        if thinking:
            msg.thinking = thinking
        convo.messages.append(msg)
        convo.tokens_in += len(new_user_message.split()) * 2  # Rough estimate
        convo.tokens_out += len(response.split()) * 2

        # Handle streaming
        if streaming and on_chunk:
            for i in range(0, len(response), 10):
                on_chunk(response[:i+10])
                time.sleep(0.01)

        return response

    def cleanup(self):
        """Mock cleanup."""
        pass


class UITestHarness:
    """Test harness for driving and inspecting the UI programmatically.

    Supports two modes:

    1. **Component mode**: Test UI logic without running the app event loop.
       Fast, synchronous, good for unit testing display logic.

    2. **Headless mode**: Run the full app with mock I/O.
       Slower, async, good for integration testing key bindings and flows.

    Example (component mode):
        harness = UITestHarness(state)
        harness.setup_component_mode()
        harness.add_message('user', 'hello')
        harness.assert_conversation_contains('hello')

    Example (headless mode):
        harness = UITestHarness(state)
        harness.start_headless(mock_client=MockLLMClient())
        harness.type_and_enter("hello")
        harness.wait_for_idle()
        harness.assert_conversation_contains('hello')
        harness.stop()
    """

    def __init__(self, state: AppState):
        self.state = state
        self.ui = None
        self._pipe_input = None
        self._pipe_input_context = None
        self._app_thread: Optional[threading.Thread] = None
        self._running = False
        self._headless = False
        self._startup_complete = threading.Event()
        self._error: Optional[Exception] = None

    # =========================================================================
    # Component Mode (synchronous, no event loop)
    # =========================================================================

    def setup_component_mode(self) -> 'UITestHarness':
        """Set up for component testing (no app event loop).

        Creates UI components that can be tested synchronously.
        """
        from prompt_toolkit.buffer import Buffer
        from .app import ScriptChatUI
        from .selection_menu import SelectionMenu
        from .command_handlers import CommandHandlers

        # Create UI object without full init
        self.ui = object.__new__(ScriptChatUI)

        # Initialize essential attributes
        self.ui.state = self.state
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
        self.ui.history_path = self.state.config.conversations_dir.parent / "history.txt"

        # Create buffers
        self.ui.input_buffer = Buffer()
        self.ui.conversation_buffer = Buffer()

        # Create mock app for invalidation
        class MockApp:
            is_running = False
            def invalidate(self): pass
            def exit(self): pass

        self.ui.app = MockApp()

        # Create events emitter
        self.ui.events = UIEventEmitter(log_events=True)

        # Create selection menu
        self.ui.selection_menu = SelectionMenu(self.ui)

        # Create command handlers
        self.ui.handlers = CommandHandlers(self.ui)

        # Bind methods from the real class
        self.ui._build_conversation_text = lambda: ScriptChatUI._build_conversation_text(self.ui)
        self.ui._get_status_bar = lambda: ScriptChatUI._get_status_bar(self.ui)
        self.ui._get_prompt_prefix = lambda: ScriptChatUI._get_prompt_prefix(self.ui)
        self.ui._markdown_to_ansi = lambda text: ScriptChatUI._markdown_to_ansi(self.ui, text)
        self.ui.update_conversation_display = lambda: self._component_update_display()
        self.ui.add_system_message = lambda text: self._component_add_system_message(text)

        self._headless = False
        return self

    def _component_update_display(self):
        """Update conversation display in component mode."""
        from prompt_toolkit.document import Document
        text = self.ui._build_conversation_text()
        self.ui.conversation_buffer.reset(
            document=Document(text=text, cursor_position=len(text))
        )
        if self.ui.events:
            self.ui.events.emit(UIEventType.CONVERSATION_UPDATED,
                              message_count=len(self.state.current_conversation.messages))

    def _component_add_system_message(self, text: str):
        """Add system message in component mode."""
        self.state.current_conversation.messages.append(
            Message(role='status', content=text)
        )
        self._component_update_display()
        if self.ui.events:
            self.ui.events.emit(UIEventType.SYSTEM_MESSAGE_ADDED, message=text)

    # =========================================================================
    # Headless Mode (full app with mock I/O)
    # =========================================================================

    def start_headless(
        self,
        mock_client: Optional[MockLLMClient] = None,
        timeout: float = 5.0
    ) -> 'UITestHarness':
        """Start the UI in headless mode for integration testing.

        Args:
            mock_client: Mock LLM client. If None, a default mock is created.
            timeout: Seconds to wait for app startup.

        Returns:
            self for chaining
        """
        from prompt_toolkit.input import create_pipe_input
        from prompt_toolkit.output import DummyOutput
        from .app import ScriptChatUI

        # Set up mock client
        if mock_client is None:
            mock_client = MockLLMClient()
        self.state.client = mock_client

        # Create pipe input for sending keystrokes
        # create_pipe_input returns a context manager, we need to enter it
        self._pipe_input_context = create_pipe_input()
        self._pipe_input = self._pipe_input_context.__enter__()

        # Create the UI with headless I/O
        self.ui = ScriptChatUI(
            self.state,
            log_ui_events=True,
            input=self._pipe_input,
            output=DummyOutput(),
        )

        # Run in background thread
        self._running = True
        self._headless = True
        self._startup_complete.clear()
        self._error = None

        self._app_thread = threading.Thread(
            target=self._run_app_loop,
            daemon=True,
            name="headless-ui"
        )
        self._app_thread.start()

        # Wait for app to be running
        if not self._startup_complete.wait(timeout):
            self.stop()
            if self._error:
                raise self._error
            raise TimeoutError(f"App did not start within {timeout}s")

        if self._error:
            raise self._error

        return self

    def _run_app_loop(self):
        """Run the app event loop in background thread."""
        try:
            # Signal that we're starting
            # The app.run() call is blocking, so we signal ready
            # after a brief delay to let the event loop start
            def signal_ready():
                time.sleep(0.1)
                self._startup_complete.set()

            threading.Thread(target=signal_ready, daemon=True).start()

            self.ui.run()
        except Exception as e:
            logger.exception(f"Headless app error: {e}")
            self._error = e
            self._startup_complete.set()
        finally:
            self._running = False

    def stop(self):
        """Stop the UI if running in headless mode."""
        if not self._headless:
            return

        self._running = False

        # Exit the app
        if self.ui and hasattr(self.ui, 'app') and self.ui.app:
            try:
                self.ui.app.exit()
            except Exception:
                pass

        # Close the pipe input context
        if self._pipe_input_context:
            try:
                self._pipe_input_context.__exit__(None, None, None)
            except Exception:
                pass

        # Wait for thread
        if self._app_thread and self._app_thread.is_alive():
            self._app_thread.join(timeout=2)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.stop()

    # =========================================================================
    # Input Methods (headless mode)
    # =========================================================================

    def send_text(self, text: str, flush_delay: float = 0.02):
        """Send raw text to the input.

        Args:
            text: Text to send
            flush_delay: Time to wait for processing
        """
        if not self._headless:
            # In component mode, just set the buffer
            self.set_input(self.get_input_text() + text)
            return

        if not self._pipe_input:
            raise RuntimeError("Not in headless mode")

        self._pipe_input.send_text(text)
        time.sleep(flush_delay)

    def send_key(self, key: str, flush_delay: float = 0.02):
        """Send a named key.

        Args:
            key: Key name (e.g., 'enter', 'escape', 'ctrl-c', 'up')
            flush_delay: Time to wait for processing

        Supported keys:
            enter, escape, tab, shift-tab, backspace, delete,
            up, down, left, right, home, end, page-up, page-down,
            ctrl-a through ctrl-w, ctrl-up, ctrl-down, ctrl-left, ctrl-right,
            ctrl-home, ctrl-end, alt-enter, alt-up, alt-down
        """
        key_lower = key.lower()
        seq = KEY_SEQUENCES.get(key_lower)

        if seq is None:
            raise ValueError(f"Unknown key: {key}. Supported: {list(KEY_SEQUENCES.keys())}")

        self.send_text(seq, flush_delay)

    def type_and_enter(self, text: str):
        """Type text and press Enter."""
        self.send_text(text)
        self.send_key('enter')

    def type_command(self, command: str, wait_for_complete: bool = True):
        """Type a command and press Enter.

        Args:
            command: Command to type (e.g., '/model ollama/llama3')
            wait_for_complete: Wait for command processing
        """
        self.type_and_enter(command)
        if wait_for_complete:
            self.wait_for_idle(timeout=2.0)

    def send_message(self, message: str, wait_for_response: bool = True, timeout: float = 30.0):
        """Send a user message and optionally wait for response.

        Args:
            message: Message to send
            wait_for_response: Wait for LLM response
            timeout: Max seconds to wait
        """
        self.type_and_enter(message)
        if wait_for_response:
            self.wait_for_idle(timeout=timeout)

    def press_escape(self, times: int = 1):
        """Press Escape key."""
        for _ in range(times):
            self.send_key('escape')

    def navigate_menu(self, direction: str, times: int = 1):
        """Navigate selection menu.

        Args:
            direction: 'up' or 'down'
            times: Number of times to navigate
        """
        for _ in range(times):
            self.send_key(direction)

    def select_menu_item(self):
        """Select current menu item (press Enter or Tab)."""
        if self._headless:
            self.send_key('enter')
        else:
            # In component mode, call the selection menu directly
            self.ui.selection_menu.select_current()

    # =========================================================================
    # State Queries
    # =========================================================================

    def get_state(self) -> UIState:
        """Get complete UI state snapshot."""
        if not self.ui:
            raise RuntimeError("UI not initialized")

        convo = self.state.current_conversation

        return UIState(
            conversation_text=self.get_conversation_text(),
            conversation_message_count=len(convo.messages),
            status_bar_text=self._get_status_text(),
            provider_id=convo.provider_id,
            model_name=convo.model_name,
            tokens_in=convo.tokens_in,
            tokens_out=convo.tokens_out,
            conversation_id=convo.id,
            input_text=self.get_input_text(),
            input_cursor_position=self.ui.input_buffer.cursor_position,
            thinking=self.ui.thinking,
            multiline_mode=self.ui.multiline_mode,
            streaming=self.state.config.enable_streaming,
            selection_menu_visible=self.ui.selection_menu.is_visible,
            selection_menu_items=list(self.ui.selection_menu.items),
            selection_menu_index=self.ui.selection_menu.selected_index,
            prompt_message=self.ui.prompt_message,
            has_pending_callback=self.ui.pending_callback is not None,
            expecting_single_key=self.ui.expecting_single_key,
            history_index=self.ui.input_history_index,
            history_length=len(self.ui.input_history),
            running_script=self.ui.running_script,
            script_queue_length=len(self.ui.script_queue),
            message_queue_length=len(self.ui.message_queue),
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

    def is_thinking(self) -> bool:
        """Check if LLM is currently processing."""
        return self.ui.thinking if self.ui else False

    def is_menu_visible(self) -> bool:
        """Check if selection menu is visible."""
        return self.ui.selection_menu.is_visible if self.ui else False

    def get_menu_items(self) -> list[tuple[Any, str]]:
        """Get current selection menu items."""
        return list(self.ui.selection_menu.items) if self.ui else []

    def get_menu_index(self) -> int:
        """Get current selection menu index."""
        return self.ui.selection_menu.selected_index if self.ui else 0

    # =========================================================================
    # Synchronization
    # =========================================================================

    def wait_for_idle(self, timeout: float = 10.0):
        """Wait until UI is idle (not thinking, no queued messages).

        Args:
            timeout: Max seconds to wait

        Raises:
            TimeoutError: If not idle within timeout
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            if not self.ui:
                raise RuntimeError("UI not initialized")

            # Check if idle
            is_idle = (
                not self.ui.thinking and
                not self.ui.message_queue and
                not self.ui.script_queue
            )

            if is_idle:
                return

            time.sleep(0.05)

        raise TimeoutError(f"UI not idle within {timeout}s (thinking={self.ui.thinking})")

    def wait_for_thinking(self, timeout: float = 5.0):
        """Wait for thinking to start.

        Args:
            timeout: Max seconds to wait
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.ui and self.ui.thinking:
                return
            time.sleep(0.05)
        raise TimeoutError(f"Thinking did not start within {timeout}s")

    def wait_for_menu(self, timeout: float = 2.0):
        """Wait for selection menu to appear.

        Args:
            timeout: Max seconds to wait
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.is_menu_visible():
                return
            time.sleep(0.05)
        raise TimeoutError(f"Menu did not appear within {timeout}s")

    def wait_for_menu_hidden(self, timeout: float = 2.0):
        """Wait for selection menu to hide.

        Args:
            timeout: Max seconds to wait
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            if not self.is_menu_visible():
                return
            time.sleep(0.05)
        raise TimeoutError(f"Menu did not hide within {timeout}s")

    def wait_for_event(
        self,
        event_type: UIEventType,
        timeout: float = 5.0,
        predicate: Callable[[UIEvent], bool] = None
    ) -> Optional[UIEvent]:
        """Wait for a specific UI event.

        Args:
            event_type: Type of event to wait for
            timeout: Max seconds to wait
            predicate: Optional filter function

        Returns:
            The event, or None if timeout
        """
        if not self.ui or not self.ui.events:
            raise RuntimeError("Events not available")

        return self.ui.events.wait_for_event(event_type, timeout, predicate)

    # =========================================================================
    # Actions (component mode helpers)
    # =========================================================================

    def set_input(self, text: str):
        """Set input buffer content (component mode)."""
        from prompt_toolkit.document import Document
        self.ui.input_buffer.set_document(
            Document(text=text, cursor_position=len(text))
        )
        if self.ui.events:
            self.ui.events.emit(UIEventType.INPUT_CHANGED, text=text)

    def clear_input(self):
        """Clear input buffer."""
        self.set_input("")
        if self.ui.events:
            self.ui.events.emit(UIEventType.INPUT_CLEARED)

    def add_message(self, role: str, content: str, thinking: str = None):
        """Add a message to conversation and update display."""
        self.state.current_conversation.messages.append(
            Message(role=role, content=content, thinking=thinking)
        )
        self.ui.update_conversation_display()

    def execute_command(self, command: str):
        """Execute a command through the UI handlers (component mode)."""
        self.ui.handlers.handle_command(command)

    def simulate_user_message_flow(self, message: str, response: str, thinking: str = None):
        """Simulate a complete user message -> response flow (component mode).

        This is useful for testing UI state after an exchange without
        actually calling an LLM.
        """
        # Add user message
        self.add_message('user', message)
        if self.ui.events:
            self.ui.events.emit(UIEventType.MESSAGE_SENT, content=message)

        # Simulate thinking
        self.ui.thinking = True
        if self.ui.events:
            self.ui.events.emit(UIEventType.THINKING_STARTED)

        # Add response
        self.add_message('assistant', response, thinking=thinking)

        # Stop thinking
        self.ui.thinking = False
        if self.ui.events:
            self.ui.events.emit(UIEventType.THINKING_STOPPED)
            self.ui.events.emit(UIEventType.RESPONSE_COMPLETE, content=response)

    def show_selection_menu(
        self,
        items: list[tuple[Any, str]],
        on_select: Callable[[Any], None] = None
    ):
        """Show selection menu with items (component mode)."""
        self.ui.selection_menu.show(
            items=items,
            on_select=on_select or (lambda x: None),
            on_cancel=lambda: None
        )

    def navigate_menu_down(self):
        """Navigate selection menu down (component mode)."""
        self.ui.selection_menu.move_down()

    def navigate_menu_up(self):
        """Navigate selection menu up (component mode)."""
        self.ui.selection_menu.move_up()

    def cancel_menu(self):
        """Cancel selection menu (component mode)."""
        self.ui.selection_menu.cancel()

    # =========================================================================
    # Assertions
    # =========================================================================

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
        actual = self.is_thinking()
        assert actual == expected, msg or f"Expected thinking={expected}, got {actual}"

    def assert_selection_menu_visible(self, expected: bool = True, msg: str = None):
        """Assert selection menu visibility."""
        actual = self.is_menu_visible()
        assert actual == expected, msg or f"Expected menu visible={expected}, got {actual}"

    def assert_menu_visible(self, expected: bool = True, msg: str = None):
        """Assert selection menu visibility (alias)."""
        return self.assert_selection_menu_visible(expected, msg)

    def assert_input_equals(self, expected: str, msg: str = None):
        """Assert input buffer content."""
        actual = self.get_input_text()
        assert actual == expected, msg or f"Expected input '{expected}', got '{actual}'"

    def assert_input_contains(self, text: str, msg: str = None):
        """Assert input buffer contains text."""
        actual = self.get_input_text()
        assert text in actual, msg or f"Expected '{text}' in input: {actual}"

    def assert_message_count(self, expected: int, msg: str = None):
        """Assert number of messages in conversation."""
        actual = len(self.state.current_conversation.messages)
        assert actual == expected, msg or f"Expected {expected} messages, got {actual}"

    def assert_last_message(self, role: str, content_contains: str = None, msg: str = None):
        """Assert properties of the last message."""
        messages = self.get_conversation_messages()
        assert messages, msg or "No messages in conversation"

        last = messages[-1]
        assert last.role == role, msg or f"Expected last message role '{role}', got '{last.role}'"

        if content_contains:
            assert content_contains in last.content, \
                msg or f"Expected '{content_contains}' in last message: {last.content[:100]}"

    def assert_event_occurred(self, event_type: UIEventType, msg: str = None):
        """Assert that an event occurred."""
        if not self.ui.events:
            raise RuntimeError("Events not available")
        events = self.ui.events.get_events(event_type)
        assert len(events) > 0, msg or f"Expected event {event_type.name} did not occur"

    def assert_no_error(self):
        """Assert no error messages in conversation."""
        content = self.get_conversation_text().lower()
        assert 'error' not in content or 'error handling' in content, \
            f"Found error in conversation: {content[:200]}"


# =========================================================================
# Test Runner
# =========================================================================

@dataclass
class UITestResult:
    """Result of a UI test."""
    passed: bool
    test_name: str
    error: Optional[str] = None
    events: list = None
    duration_ms: float = 0


def run_ui_test(
    name: str,
    state: AppState,
    test_fn: Callable[[UITestHarness], None],
    mode: str = 'component'
) -> UITestResult:
    """Run a UI test function with a fresh harness.

    Args:
        name: Test name for reporting
        state: AppState to use
        test_fn: Test function that takes a harness
        mode: 'component' or 'headless'

    Example:
        def test_message_display(harness):
            harness.add_message('user', 'hello')
            harness.assert_conversation_contains('hello')

        result = run_ui_test("message_display", state, test_message_display)
    """
    harness = UITestHarness(state)

    if mode == 'component':
        harness.setup_component_mode()
    else:
        harness.start_headless()

    start = time.time()
    try:
        test_fn(harness)
        duration = (time.time() - start) * 1000
        return UITestResult(
            passed=True,
            test_name=name,
            events=harness.ui.events.get_events() if harness.ui.events else [],
            duration_ms=duration
        )
    except AssertionError as e:
        duration = (time.time() - start) * 1000
        return UITestResult(
            passed=False,
            test_name=name,
            error=str(e),
            events=harness.ui.events.get_events() if harness.ui.events else [],
            duration_ms=duration
        )
    except Exception as e:
        duration = (time.time() - start) * 1000
        return UITestResult(
            passed=False,
            test_name=name,
            error=f"Unexpected error: {e}",
            events=harness.ui.events.get_events() if harness.ui.events else [],
            duration_ms=duration
        )
    finally:
        harness.stop()
