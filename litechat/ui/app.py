# Copyright 2024 lite-chat contributors
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

"""Terminal UI for lite-chat using prompt_toolkit."""

import shutil
import threading
import logging
from pathlib import Path
from typing import Optional

from prompt_toolkit import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.document import Document
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout, HSplit, VSplit, Window, FormattedTextControl, FloatContainer, Float
from prompt_toolkit.layout.containers import WindowAlign
from prompt_toolkit.layout.controls import BufferControl
from prompt_toolkit.layout.menus import CompletionsMenu
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.filters import Condition
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.lexers import Lexer
from prompt_toolkit.formatted_text import ANSI, to_formatted_text
from prompt_toolkit.mouse_events import MouseEventType

from ..core.commands import AppState, handle_command, set_model, set_temperature
from ..core.conversations import (
    list_conversations, load_conversation, save_conversation,
    branch_conversation, delete_conversation, Conversation, Message
)
from .command_handlers import CommandHandlers

logger = logging.getLogger(__name__)


def resolve_clear_target_from_args(
    args: str,
    conversations_root: Path,
    current_conversation_id: Optional[str]
) -> tuple[Optional[str], Optional[str], Optional[str], list, bool]:
    """Resolve target id and prompt for /clear.

    Returns (target_id, prompt_message, error_message, summaries_used, target_is_current).
    """
    arg = args.strip()
    summaries_used = list_conversations(conversations_root)

    if arg:
        try:
            idx = int(arg)
            if idx < 0:
                raise ValueError
            if idx >= len(summaries_used):
                return None, None, "Invalid conversation index.", summaries_used, False

            target_id = summaries_used[idx].dir_name
            prompt = f"Clear and delete conversation #{idx} ({summaries_used[idx].display_name})? (y/N)"
            target_is_current = (current_conversation_id is not None and target_id == current_conversation_id)
            return target_id, prompt, None, summaries_used, target_is_current
        except ValueError:
            return None, None, "Usage: /clear [index]", summaries_used, False

    # No arg: clear current
    target_id = current_conversation_id
    target_label = target_id or "current (unsaved) conversation"
    prompt = f"Clear and delete {target_label}? (y/N)"
    target_is_current = True
    return target_id, prompt, None, summaries_used, target_is_current


class AnsiLexer(Lexer):
    """Lexer that interprets ANSI color codes in buffer text."""

    def lex_document(self, document):
        """Lex the document and return formatted text for each line.

        Args:
            document: The document to lex

        Returns:
            Function that takes line number and returns formatted text
        """
        lines = document.text.split('\n')

        def get_line(lineno):
            if lineno < len(lines):
                # Parse ANSI codes and convert to list of formatted text tuples
                return to_formatted_text(ANSI(lines[lineno]))
            return []

        return get_line


class LiteChatUI:
    """Terminal user interface for lite-chat."""

    def __init__(self, state: AppState):
        """Initialize the UI.

        Args:
            state: Application state
        """
        self.state = state
        self.multiline_mode = False
        self.multiline_buffer = []
        self.prompt_message = ""  # Current prompt message for user input
        self.pending_callback = None  # Callback waiting for user input
        self.expecting_single_key = False
        self.thinking = False  # Track if LLM is processing
        self.thinking_dots = 0  # Animation counter for thinking indicator
        self.current_inference_thread = None  # Track current LLM thread for cancellation
        self.cancel_requested_at = None  # Timestamp of first ESC press for double-ESC cancel
        self.input_history: list[str] = []
        self.input_history_index: Optional[int] = None
        self.message_queue: list[str] = []  # Pending user messages to send when LLM is free
        self.script_queue: list[str] = []  # Pending script lines to execute
        self.running_script: bool = False
        self.history_path = self.state.config.conversations_dir.expanduser().parent / "history.txt"

        # Create command handlers
        self.handlers = CommandHandlers(self)

        # Create command completer
        command_completer = WordCompleter(
            ['/new', '/save', '/load', '/branch', '/rename', '/chats', '/send', '/export', '/export-all', '/import', '/stream', '/prompt', '/run', '/model', '/temp', '/reason', '/timeout', '/profile', '/log-level', '/files', '/clear', '/file', '/echo', '/tag', '/untag', '/tags', '/assert', '/assert-not', '/undo', '/retry', '/exit'],
            ignore_case=True,
            sentence=True
        )

        # Create buffers
        self.input_buffer = Buffer(
            multiline=False,
            history=InMemoryHistory(),
            completer=command_completer,
            complete_while_typing=True
        )

        # Create buffer for conversation display (not read-only so we can scroll)
        self.conversation_buffer = Buffer()

        # Create layout
        self.layout = self._create_layout()

        # Create key bindings
        self.kb = self._create_key_bindings()

        # Create application
        self.app = Application(
            layout=self.layout,
            key_bindings=self.kb,
            full_screen=True,
            mouse_support=False  # Disabled to allow terminal-native mouse selection
        )

        # Initialize conversation display (scroll to bottom initially)
        self.update_conversation_display()
        self._load_history()

    def _create_layout(self) -> Layout:  # pragma: no cover - UI layout wiring
        """Create the application layout.

        Returns:
            Layout object with conversation, status, and input panes
        """
        # Conversation pane (top) - using buffer for scroll control with ANSI lexer
        self.conversation_window = Window(
            content=BufferControl(
                buffer=self.conversation_buffer,
                lexer=AnsiLexer(),
                focusable=True  # Allow focus for scrolling
            ),
            wrap_lines=True
        )

        # Status bar (middle)
        status_window = Window(
            content=FormattedTextControl(text=self._get_status_bar),
            height=1
        )

        # Separator line between status and input
        separator_window = Window(
            char='─',
            height=1
        )

        # Input pane (bottom) - with prompt prefix
        prompt_window = Window(
            content=FormattedTextControl(text=self._get_prompt_prefix),
            width=lambda: len(self._get_prompt_prefix()),
            dont_extend_width=True
        )

        def get_input_height():
            text = self.input_buffer.text
            if not text:
                return 1
            term_width = shutil.get_terminal_size().columns
            prompt_len = len(self._get_prompt_prefix())
            available_width = max(1, term_width - prompt_len)
            return max(1, (len(text) + available_width - 1) // available_width)

        self.input_window = Window(
            content=BufferControl(buffer=self.input_buffer),
            wrap_lines=True
        )

        input_container = VSplit([
            prompt_window,
            self.input_window
        ], height=get_input_height)

        # Main container
        root_container = HSplit([
            self.conversation_window,
            status_window,
            separator_window,
            input_container
        ])

        # Wrap in FloatContainer to support completion menu popup
        float_container = FloatContainer(
            content=root_container,
            floats=[
                Float(
                    xcursor=True,
                    ycursor=True,
                    content=CompletionsMenu(max_height=10)
                )
            ]
        )

        return Layout(
            float_container,
            focused_element=self.input_window
        )

    def _create_key_bindings(self) -> KeyBindings:  # pragma: no cover - interactive key handling
        """Create key bindings for the application.

        Returns:
            KeyBindings object
        """
        kb = KeyBindings()

        @kb.add('enter')
        def handle_enter(event):
            """Handle Enter key press."""
            text = self.input_buffer.text.strip()

            # Check if we're waiting for callback input
            if self.pending_callback:
                self.input_buffer.text = ''
                self.prompt_message = ""
                callback = self.pending_callback
                self.expecting_single_key = False
                self.pending_callback = None
                callback(text)
                return

            if self.multiline_mode:
                # Check if this is the closing delimiter
                if text == '"""':
                    # Send the collected multiline message
                    full_message = '\n'.join(self.multiline_buffer)
                    self.multiline_mode = False
                    self.multiline_buffer = []
                    self.input_buffer.text = ''
                    if full_message.strip():
                        self._append_history(full_message)

                    if full_message.strip():
                        self.handle_user_message(full_message)
                else:
                    # Add line to buffer and continue
                    self.multiline_buffer.append(text)
                    self.input_buffer.text = ''
            else:
                # Check if starting multiline mode
                if text.startswith('"""'):
                    self.multiline_mode = True
                    self.multiline_buffer = []
                    self.input_buffer.text = ''
                    self.add_system_message("[Multi-line mode active. Type '\"\"\"' on a new line to send]")
                elif text.startswith('/'):
                    # Command
                    if text:
                        self._append_history(text)
                    self.input_buffer.text = ''
                    self._handle_command(text)
                elif text:
                    # Regular user message
                    self._append_history(text)
                    self.input_buffer.text = ''
                    event.app.invalidate()  # Force redraw to show cleared input
                    self.handle_user_message(text)

        @kb.add('escape')
        def handle_escape(event):
            """Handle Escape to cancel inference or return focus to input."""
            import time

            # If LLM is thinking, handle cancellation with confirmation
            if self.thinking:
                current_time = time.time()

                # Check if ESC was recently pressed (within 2 seconds)
                if self.cancel_requested_at and (current_time - self.cancel_requested_at) < 2.0:
                    # Double ESC - proceed with cancellation
                    self.cancel_requested_at = None
                    self._cancel_inference()
                else:
                    # First ESC - ask for confirmation
                    self.cancel_requested_at = current_time
                    self.add_system_message("⚠ Press ESC again within 2 seconds to cancel inference")
                return

            # Otherwise, return focus to input
            event.app.layout.focus(self.input_window)

        @kb.add('tab')
        def handle_tab(event):
            """Handle Tab key for completion or focus switch."""
            buff = event.current_buffer

            # If in input buffer, handle completion
            if buff == self.input_buffer:
                # If there are completions available, cycle through them
                if buff.complete_state:
                    buff.complete_next()
                else:
                    # Start completion - show menu
                    buff.start_completion(select_first=True)
            # If in conversation buffer, switch to input
            else:
                event.app.layout.focus(self.input_window)

        @kb.add('s-tab')
        def handle_shift_tab(event):
            """Handle Shift+Tab for reverse completion."""
            buff = event.current_buffer
            if buff.complete_state:
                buff.complete_previous()

        @kb.add('y', filter=Condition(lambda: self.pending_callback is not None and self.expecting_single_key))
        @kb.add('n', filter=Condition(lambda: self.pending_callback is not None and self.expecting_single_key))
        def handle_yes_no(event):
            """Handle single-key confirmations when expected."""
            key = event.key_sequence[0].key
            self.input_buffer.text = ''
            self.prompt_message = ""
            callback = self.pending_callback
            self.pending_callback = None
            self.expecting_single_key = False
            callback(key)
            return

        @kb.add('up')
        def handle_up(event):
            """Recall previous history entry when in input."""
            buff = event.current_buffer
            if buff == self.input_buffer and not self.multiline_mode and not self.pending_callback:
                if buff.complete_state:
                    buff.complete_previous()
                else:
                    self._history_previous()

        @kb.add('down')
        def handle_down(event):
            """Recall next history entry when in input."""
            buff = event.current_buffer
            if buff == self.input_buffer and not self.multiline_mode and not self.pending_callback:
                if buff.complete_state:
                    buff.complete_next()
                else:
                    self._history_next()

        @kb.add('c-up')
        def handle_ctrl_up(event):
            """Handle Ctrl+Up to focus conversation for scrolling."""
            # Focus conversation and position cursor in middle for scrolling
            event.app.layout.focus(self.conversation_buffer)
            doc = self.conversation_buffer.document
            # Position cursor in middle or near bottom to allow scrolling up
            self.conversation_buffer.cursor_position = max(0, len(doc.text) - 500)

        @kb.add('c-home')
        def handle_ctrl_home(event):
            """Handle Ctrl+Home to jump to start of conversation."""
            event.app.layout.focus(self.conversation_buffer)
            self.conversation_buffer.cursor_position = 0

        @kb.add('c-end')
        def handle_ctrl_end(event):
            """Handle Ctrl+End to jump to end of conversation."""
            event.app.layout.focus(self.conversation_buffer)
            doc = self.conversation_buffer.document
            # Position slightly before end to allow scrolling
            self.conversation_buffer.cursor_position = max(0, len(doc.text) - 1)

        @kb.add('up', filter=Condition(lambda: self.app.layout.has_focus(self.conversation_buffer)))
        def handle_up(event):
            """Scroll conversation up with Up arrow when focused."""
            buff = self.conversation_buffer
            buff.cursor_up(count=1)

        @kb.add('down', filter=Condition(lambda: self.app.layout.has_focus(self.conversation_buffer)))
        def handle_down(event):
            """Scroll conversation down with Down arrow when focused."""
            buff = self.conversation_buffer
            buff.cursor_down(count=1)

        # Prevent typing in conversation buffer
        @kb.add('<any>', filter=Condition(lambda: self.app.layout.has_focus(self.conversation_buffer)))
        def handle_conversation_input(event):
            """Prevent editing in conversation buffer - only allow navigation."""
            # Allow navigation keys, block everything else
            pass  # Key will be ignored for non-navigation keys

        @kb.add('c-c')
        @kb.add('c-d')
        def handle_exit(event):
            """Handle Ctrl+C or Ctrl+D to exit."""
            event.app.exit()

        return kb


    def _get_status_bar(self):
        """Get status bar text with styling.

        Returns:
            Formatted status bar with gray styling
        """
        convo = self.state.current_conversation
        conv_id = convo.id if convo.id else '<unsaved>'

        # Add thinking indicator if LLM is processing
        thinking_indicator = ""
        if self.thinking:
            dots = "." * ((self.thinking_dots % 3) + 1)
            thinking_indicator = f" | Thinking{dots}"

        # Build context usage display
        context_display = ""
        if convo.context_length_configured is not None and convo.context_length_used is not None:
            percentage = (convo.context_length_used / convo.context_length_configured * 100) if convo.context_length_configured > 0 else 0
            context_display = f" | {convo.context_length_used}/{convo.context_length_configured} ({percentage:.1f}%)"

        reasoning_display = f" ({convo.reasoning_level})" if getattr(convo, "reasoning_level", None) else ""

        text = (
            f"{convo.provider_id}/{convo.model_name}{reasoning_display} | "
            f"{convo.tokens_in} in / {convo.tokens_out} out{context_display} | "
            f"{conv_id}{thinking_indicator}"
        )

        # Return with reverse video (inverted colors)
        return [('reverse', text)]

    def _get_prompt_prefix(self) -> str:
        """Get prompt prefix text."""
        return (self.prompt_message + " ") if self.prompt_message else "> "

    def _build_conversation_text(self) -> str:
        """Build conversation text with ANSI color codes.

        Returns:
            String with ANSI color codes for display
        """
        lines = []

        # ANSI color codes
        GRAY = '\033[90m'
        RED = '\033[91m'
        CYAN = '\033[96m'
        GREEN = '\033[92m'
        YELLOW = '\033[93m'
        RESET = '\033[0m'

        for msg in self.state.current_conversation.messages:
            if msg.role == 'system':
                content = msg.content or ""
                is_error = content.strip().lower().startswith("error")
                color = RED if is_error else GRAY
                lines.append(f"{color}[system] {content}{RESET}")
            elif msg.role == 'user':
                # User messages in cyan
                lines.append(f"{CYAN}[user]{RESET} {msg.content}")
            elif msg.role == 'assistant':
                # Assistant messages in green
                lines.append(f"{GREEN}[assistant]{RESET} {msg.content}")
            elif msg.role == 'echo':
                # Echo messages in yellow, no prefix
                lines.append(f"{YELLOW}{msg.content}{RESET}")

        return '\n'.join(lines)

    def update_conversation_display(self):
        """Update the conversation display and scroll to bottom."""
        # Build text with ANSI colors
        text = self._build_conversation_text()

        # Update buffer with new content, cursor at end for auto-scroll
        self.conversation_buffer.reset(
            document=Document(text=text, cursor_position=len(text))
        )

        # Force redraw
        self.app.invalidate()

    def add_system_message(self, text: str):
        """Add a system message to the conversation display.

        Args:
            text: System message text
        """
        from ..core.conversations import Message
        self.state.current_conversation.messages.append(
            Message(role='system', content=text)
        )
        self.update_conversation_display()

    def handle_user_message(self, message: str):
        """Handle a user message by sending it to the LLM.

        Args:
            message: User message text
        """
        if self.thinking or self.message_queue:
            self._enqueue_message(message)
            return

        self._send_message_now(message)

    def _send_message_now(self, message: str):  # pragma: no cover - threaded UI flow
        """Send a user message immediately (assumes LLM is free)."""
        from ..core.conversations import Message
        from ..core.commands import resolve_placeholders

        # Add user message to conversation and show it (store original)
        self.state.current_conversation.messages.append(
            Message(role='user', content=message)
        )
        self.update_conversation_display()

        # Expand placeholders in current history and new message for sending
        expanded_messages = []
        for msg in self.state.current_conversation.messages:
            if msg.role in ('user', 'system'):
                expanded, err = resolve_placeholders(msg.content, self.state.file_registry)
                if err:
                    self.add_system_message(err)
                    self.state.current_conversation.messages.pop()
                    return
                expanded_messages.append(Message(role=msg.role, content=expanded))
            else:
                expanded_messages.append(msg)
        # Note: assistant messages are passed through unchanged
        # Start thinking animation
        self.thinking = True
        self.thinking_dots = 0

        def animate_thinking():
            import time
            while self.thinking:
                self.thinking_dots += 1
                self.app.invalidate()
                time.sleep(0.5)

        threading.Thread(target=animate_thinking, daemon=True).start()

        def call_llm():
            import time
            start_time = time.time()

            try:
                streaming = bool(self.state.config.enable_streaming)

                def on_chunk(_):
                    self.update_conversation_display()
                    self.app.invalidate()

                self.state.client.chat(
                    self.state.current_conversation,
                    expanded_messages[-1].content,
                    streaming=streaming,
                    on_chunk=on_chunk if streaming else None,
                    expanded_history=expanded_messages[:-1]
                )

                end_time = time.time()
                duration_secs = int(end_time - start_time)
                mins = duration_secs // 60
                secs = duration_secs % 60
                duration_msg = f"Thought for {mins} mins {secs} secs" if mins > 0 else f"Thought for {secs} secs"

                self.thinking = False
                self.cancel_requested_at = None
                self.update_conversation_display()
                self.add_system_message(duration_msg)
                self.app.invalidate()
            except Exception as e:
                logger.exception("Chat error")
                self.thinking = False
                self.cancel_requested_at = None
                self.add_system_message(f"Error: {str(e)}")
                self.app.invalidate()
            finally:
                # Process any queued messages
                self._process_queue()

        threading.Thread(target=call_llm, daemon=True).start()

    def _enqueue_message(self, message: str):
        """Queue a user message to send later."""
        self.message_queue.append(message)
        position = len(self.message_queue)
        self.add_system_message(f"Message queued (#{position}).")

    def _process_queue(self):  # pragma: no cover - threaded UI flow
        """Send next queued message if available and not already thinking."""
        if self.thinking:
            return
        if not self.message_queue:
            # If no messages pending, try to advance any running script
            self._process_script_queue()
            return
        next_message = self.message_queue.pop(0)
        self._send_message_now(next_message)

    def _handle_command(self, command_line: str):  # pragma: no cover - UI dispatch shim
        """Handle a command.

        Args:
            command_line: Command line starting with /
        """
        # Delegate to command handlers
        self.handlers.handle_command(command_line)

    def parse_script_lines(self, lines: list[str]) -> list[str]:
        """Parse script lines, stripping comments/empties."""
        parsed = []
        for raw in lines:
            line = raw.strip()
            if not line or line.startswith('#'):
                continue
            parsed.append(line)
        return parsed

    def run_script_queue(self, lines: list[str]):
        """Initialize and start running a script queue."""
        self.script_queue = list(lines)
        self.running_script = True
        self._process_script_queue()

    def prompt_for_input(self, callback, expect_single_key: bool = False):
        """Prompt for user input and call callback with result.

        Args:
            callback: Function to call with user input
            expect_single_key: If True, accept a single keypress without Enter
        """
        # Store callback - will be called when user presses Enter
        self.pending_callback = callback
        self.expecting_single_key = expect_single_key

    def _process_script_queue(self):
        """Process pending script lines if idle and no queued messages."""
        if self.thinking or self.message_queue:
            return
        if not self.script_queue:
            if self.running_script:
                self.running_script = False
                self.add_system_message("Script completed.")
            return

        # Consume as many immediate lines as possible without blocking
        while not self.thinking and not self.message_queue and self.script_queue:
            line = self.script_queue.pop(0)
            self._execute_script_line(line)
            if self.thinking or self.message_queue:
                break

        # If done and nothing pending, announce completion
        if not self.thinking and not self.message_queue and not self.script_queue and self.running_script:
            self.running_script = False
            self.add_system_message("Script completed.")

    def _execute_script_line(self, line: str):
        """Execute a single script line as command or message."""
        if line.startswith('/'):
            self._handle_command(line)
        else:
            self.handle_user_message(line)

    def _append_history(self, entry: str):
        """Append a line to the input history."""
        entry = entry.strip()
        if not entry:
            return

        self.input_history.append(entry)
        self.input_history_index = None
        try:
            self.history_path.parent.mkdir(parents=True, exist_ok=True)
            # Keep file trimmed by rewriting if it grows large
            if len(self.input_history) > 500:
                self.input_history = self.input_history[-500:]
            self.history_path.write_text("\n".join(self.input_history) + "\n", encoding="utf-8")
        except Exception:
            # Ignore persistence errors to avoid interrupting the session
            pass

    def _load_history(self):
        """Load persisted input history."""
        try:
            if self.history_path.exists():
                lines = [line.strip() for line in self.history_path.read_text(encoding="utf-8").splitlines() if line.strip()]
                self.input_history = lines[-500:]
        except Exception:
            # Ignore load errors
            self.input_history = []

    def _history_previous(self):
        """Move to previous history item."""
        if not self.input_history:
            return

        if self.input_history_index is None:
            self.input_history_index = len(self.input_history) - 1
        else:
            self.input_history_index = max(0, self.input_history_index - 1)

        self._apply_history_at_index()

    def _history_next(self):
        """Move to next history item."""
        if not self.input_history:
            return

        if self.input_history_index is None:
            return

        if self.input_history_index >= len(self.input_history) - 1:
            # Past the newest entry -> clear
            self.input_history_index = None
            self.input_buffer.text = ''
            self.input_buffer.cursor_position = 0
        else:
            self.input_history_index += 1
            self._apply_history_at_index()

    def _apply_history_at_index(self):
        """Apply the current history index to the input buffer."""
        if self.input_history_index is None:
            return
        entry = self.input_history[self.input_history_index]
        # Replace buffer content with history entry and move cursor to end
        self.input_buffer.set_document(Document(text=entry, cursor_position=len(entry)))

    def _cancel_inference(self):  # pragma: no cover - interactive cancel path
        """Cancel the current LLM inference."""
        if not self.thinking:
            return

        logger.info("Cancelling inference")

        # Show cancelling message immediately
        self.add_system_message("Cancelling...")
        self.app.invalidate()

        # Stop Ollama server to kill the running inference
        # Only do this for Ollama provider
        if self.state.current_conversation.provider_id == 'ollama':
            try:
                # Access Ollama client from dispatcher
                ollama_client = self.state.client.clients.get('ollama')
                if ollama_client and hasattr(ollama_client, 'server_manager'):
                    ollama_client.server_manager.stop()
                    self.add_system_message("⚠ Inference cancelled (Ollama stopped)")
                else:
                    self.add_system_message("⚠ Inference cancelled (soft)")
            except Exception as e:
                logger.error(f"Error stopping Ollama: {e}")
                self.add_system_message(f"⚠ Error cancelling: {e}")
        else:
            # For non-Ollama providers, just set thinking to false
            # The request will complete in background but we ignore it
            self.add_system_message("⚠ Inference cancelled (request continues in background)")

        self.thinking = False
        self.cancel_requested_at = None
        self.app.invalidate()

    def _cleanup(self):
        """Cleanup resources on exit."""
        try:
            print("\nCleaning up providers...")
            if hasattr(self.state.client, "cleanup"):
                self.state.client.cleanup()
            print("Cleanup complete.")
        except Exception as e:
            print(f"Error during cleanup: {e}")
            pass  # Ignore errors during cleanup

    def run(self):
        """Run the application."""
        try:
            self.app.run()
        finally:
            # Ensure Ollama server is stopped
            self._cleanup()


def run_ui(state: AppState):  # pragma: no cover - interactive UI loop
    """Run the terminal UI.

    Args:
        state: Application state
    """
    ui = LiteChatUI(state)
    ui.run()
