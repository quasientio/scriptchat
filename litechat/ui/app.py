"""Terminal UI for lite-chat using prompt_toolkit."""

import threading

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

from ..core.commands import AppState, handle_command, set_model, set_temperature
from ..core.conversations import (
    list_conversations, load_conversation, save_conversation,
    branch_conversation, delete_conversation, Conversation, Message
)


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
        self.thinking = False  # Track if LLM is processing
        self.thinking_dots = 0  # Animation counter for thinking indicator

        # Create command completer
        command_completer = WordCompleter(
            ['/new', '/save', '/load', '/branch', '/rename', '/chats', '/export', '/stream', '/model', '/temp', '/clear', '/file', '/exit'],
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
            mouse_support=True  # Enable mouse scrolling
        )

        # Initialize conversation display (scroll to bottom initially)
        self._update_conversation_display()

    def _create_layout(self) -> Layout:
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

        # Prompt line (shown when asking for input)
        prompt_line_window = Window(
            content=FormattedTextControl(text=self._get_prompt_line),
            height=lambda: 1 if self.prompt_message else 0
        )

        # Separator line between status and input
        separator_window = Window(
            char='─',
            height=1
        )

        # Input pane (bottom) - with prompt prefix
        prompt_window = Window(
            content=FormattedTextControl(text='> '),
            width=2,
            dont_extend_width=True
        )

        self.input_window = Window(
            content=BufferControl(buffer=self.input_buffer),
            height=1
        )

        input_container = VSplit([
            prompt_window,
            self.input_window
        ], height=1)

        # Main container
        root_container = HSplit([
            self.conversation_window,
            status_window,
            separator_window,
            prompt_line_window,
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

    def _create_key_bindings(self) -> KeyBindings:
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
                        self._handle_user_message(full_message)
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
                    self._add_system_message("[Multi-line mode active. Type '\"\"\"' on a new line to send]")
                elif text.startswith('/'):
                    # Command
                    self.input_buffer.text = ''
                    self._handle_command(text)
                elif text:
                    # Regular user message
                    self.input_buffer.text = ''
                    event.app.invalidate()  # Force redraw to show cleared input
                    self._handle_user_message(text)

        @kb.add('escape')
        def handle_escape(event):
            """Handle Escape to return focus to input."""
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

        text = (
            f"{convo.model_name} | "
            f"{convo.tokens_in} in / {convo.tokens_out} out{context_display} | "
            f"{conv_id}{thinking_indicator}"
        )

        # Return with reverse video (inverted colors)
        return [('reverse', text)]

    def _get_prompt_line(self) -> str:
        """Get prompt line text.

        Returns:
            Current prompt message or empty string
        """
        return self.prompt_message

    def _build_conversation_text(self) -> str:
        """Build conversation text with ANSI color codes.

        Returns:
            String with ANSI color codes for display
        """
        lines = []

        # ANSI color codes
        GRAY = '\033[90m'
        CYAN = '\033[96m'
        GREEN = '\033[92m'
        RESET = '\033[0m'

        for msg in self.state.current_conversation.messages:
            if msg.role == 'system':
                # System messages in dim gray
                lines.append(f"{GRAY}[system] {msg.content}{RESET}")
            elif msg.role == 'user':
                # User messages in cyan
                lines.append(f"{CYAN}[user]{RESET} {msg.content}")
            elif msg.role == 'assistant':
                # Assistant messages in green
                lines.append(f"{GREEN}[assistant]{RESET} {msg.content}")

        return '\n'.join(lines)

    def _update_conversation_display(self):
        """Update the conversation display and scroll to bottom."""
        # Build text with ANSI colors
        text = self._build_conversation_text()

        # Update buffer with new content, cursor at end for auto-scroll
        self.conversation_buffer.reset(
            document=Document(text=text, cursor_position=len(text))
        )

        # Force redraw
        self.app.invalidate()

    def _add_system_message(self, text: str):
        """Add a system message to the conversation display.

        Args:
            text: System message text
        """
        from ..core.conversations import Message
        self.state.current_conversation.messages.append(
            Message(role='system', content=text)
        )
        self._update_conversation_display()

    def _handle_user_message(self, message: str):
        """Handle a user message by sending it to the LLM.

        Args:
            message: User message text
        """
        # Add user message to conversation immediately
        from ..core.conversations import Message
        self.state.current_conversation.messages.append(
            Message(role='user', content=message)
        )

        # Update display to show user message right away
        self._update_conversation_display()

        # Start thinking animation
        self.thinking = True
        self.thinking_dots = 0

        # Animate the thinking indicator
        def animate_thinking():
            import time
            while self.thinking:
                self.thinking_dots += 1
                self.app.invalidate()
                time.sleep(0.5)

        animation_thread = threading.Thread(target=animate_thinking, daemon=True)
        animation_thread.start()

        # Call LLM in a background thread so UI stays responsive
        def call_llm():
            import time
            start_time = time.time()

            try:
                # Send to LLM (this will add the assistant response)
                streaming = bool(self.state.config.enable_streaming)

                def on_chunk(_):
                    # Update display as content streams in
                    self._update_conversation_display()
                    self.app.invalidate()

                response = self.state.client.chat(
                    self.state.current_conversation,
                    message,
                    streaming=streaming,
                    on_chunk=on_chunk if streaming else None
                )

                # Calculate duration
                end_time = time.time()
                duration_secs = int(end_time - start_time)
                mins = duration_secs // 60
                secs = duration_secs % 60

                # Format duration message
                if mins > 0:
                    duration_msg = f"Thought for {mins} mins {secs} secs"
                else:
                    duration_msg = f"Thought for {secs} secs"

                # Stop thinking animation
                self.thinking = False

                # Update display again with assistant response
                self._update_conversation_display()

                # Add timing system message
                self._add_system_message(duration_msg)

                self.app.invalidate()

            except Exception as e:
                self.thinking = False
                self._add_system_message(f"Error: {str(e)}")
                self.app.invalidate()

        thread = threading.Thread(target=call_llm, daemon=True)
        thread.start()

    def _handle_command(self, command_line: str):
        """Handle a command.

        Args:
            command_line: Command line starting with /
        """
        # Parse command and arguments
        parts = command_line[1:].split(maxsplit=1)
        cmd = parts[0].lower() if parts else ""
        args = parts[1] if len(parts) > 1 else ""

        result = handle_command(command_line, self.state)

        if result.message:
            self._add_system_message(result.message)

        if result.should_exit:
            self.app.exit()

        # Handle file content - send it as a user message
        if result.file_content:
            self._handle_user_message(result.file_content)
            return

        if result.needs_ui_interaction:
            # Handle commands that need user interaction
            if result.command_type == 'model':
                self._handle_model(args)
            elif result.command_type == 'save':
                self._handle_save(args)
            elif result.command_type == 'load':
                self._handle_load(args)
            elif result.command_type == 'branch':
                self._handle_branch(args)
            elif result.command_type == 'rename':
                self._handle_rename(args)
            elif result.command_type == 'chats':
                self._handle_chats()
            elif result.command_type == 'export':
                self._handle_export(args)
            elif result.command_type == 'stream':
                self._handle_stream(args)
            elif result.command_type == 'temp':
                self._handle_temp(args)
            elif result.command_type == 'clear':
                self._handle_clear()

    def _handle_model(self, args: str = ""):
        """Handle /model command.

        Args:
            args: Optional arguments (model index)
        """
        # If index provided as argument, use it directly
        if args.strip():
            self._model_callback(args.strip())
            return

        # Display model list
        lines = ["Available models:"]
        for i, model in enumerate(self.state.config.models):
            contexts_str = ', '.join(str(c) for c in model.contexts)
            lines.append(f"  [{i}] {model.name} (contexts: {contexts_str})")

        self._add_system_message('\n'.join(lines))
        self.prompt_message = "Enter model index:"

        # Get user input
        self._prompt_for_input(self._model_callback)

    def _model_callback(self, index_str: str):
        """Callback for model input.

        Args:
            index_str: User input (model index)
        """
        try:
            index = int(index_str)
            if 0 <= index < len(self.state.config.models):
                model_name = self.state.config.models[index].name
                result = set_model(self.state, model_name)
                self._add_system_message(result.message)
            else:
                self._add_system_message("Invalid model index")
        except ValueError:
            self._add_system_message("Invalid input. Please enter a number.")

    def _handle_save(self, save_name: str = ""):
        """Handle /save command.

        Args:
            save_name: Optional save name from command argument
        """
        if self.state.current_conversation.id:
            # Already saved, just update
            save_conversation(
                self.state.conversations_root,
                self.state.current_conversation,
                system_prompt=self.state.config.system_prompt
            )
            self._add_system_message("Conversation saved")
        else:
            # Need save name
            if save_name:
                # Name provided as argument
                self._save_callback(save_name)
            else:
                # Prompt for name
                self.prompt_message = "Save as (short name, used in folder name):"
                self._prompt_for_input(self._save_callback)

    def _save_callback(self, save_name: str):
        """Callback for save input.

        Args:
            save_name: User-provided save name
        """
        if not save_name.strip():
            self._add_system_message("Save cancelled (empty name)")
            return

        try:
            save_conversation(
                self.state.conversations_root,
                self.state.current_conversation,
                save_name=save_name,
                system_prompt=self.state.config.system_prompt
            )
            self._add_system_message(f"Conversation saved as: {self.state.current_conversation.id}")
        except Exception as e:
            self._add_system_message(f"Error saving: {str(e)}")

    def _handle_load(self, args: str = ""):
        """Handle /load command.

        Args:
            args: Optional arguments (conversation index)
        """
        summaries = list_conversations(self.state.conversations_root)

        if not summaries:
            self._add_system_message("No saved conversations found")
            return

        # If index provided as argument, use it directly
        if args.strip():
            self._load_callback(args.strip(), summaries)
            return

        lines = ["Saved conversations:"]
        for i, summary in enumerate(summaries):
            lines.append(
                f"  [{i}] {summary.display_name} "
                f"(model: {summary.model_name}, created: {summary.created_at[:10]})"
            )

        self._add_system_message('\n'.join(lines))
        self.prompt_message = "Enter conversation index:"
        self._prompt_for_input(lambda idx: self._load_callback(idx, summaries))

    def _load_callback(self, index_str: str, summaries: list):
        """Callback for load input.

        Args:
            index_str: User input (conversation index)
            summaries: List of conversation summaries
        """
        try:
            index = int(index_str)
            if 0 <= index < len(summaries):
                summary = summaries[index]
                convo = load_conversation(self.state.conversations_root, summary.dir_name)
                self.state.current_conversation = convo
                self._update_conversation_display()
                self._add_system_message(f"Loaded conversation: {summary.display_name}")
            else:
                self._add_system_message("Invalid conversation index")
        except ValueError:
            self._add_system_message("Invalid input. Please enter a number.")
        except Exception as e:
            self._add_system_message(f"Error loading: {str(e)}")

    def _handle_branch(self, args: str = ""):
        """Handle /branch command.

        Args:
            args: Optional arguments (new save name)
        """
        try:
            # Use args as new save name if provided, otherwise None (auto-generate)
            new_save_name = args if args else None
            new_convo = branch_conversation(
                self.state.conversations_root,
                self.state.current_conversation,
                new_save_name=new_save_name,
                system_prompt=self.state.config.system_prompt
            )
            self.state.current_conversation = new_convo
            self._add_system_message(f"Branched to: {new_convo.id}")
        except Exception as e:
            self._add_system_message(f"Error branching: {str(e)}")

    def _handle_chats(self):
        """Handle /chats command (list saved conversations)."""
        summaries = list_conversations(self.state.conversations_root)

        if not summaries:
            self._add_system_message("No saved conversations found")
            return

        lines = ["Saved conversations:"]
        for i, summary in enumerate(summaries):
            lines.append(
                f"  [{i}] {summary.display_name} "
                f"(model: {summary.model_name}, created: {summary.created_at[:10]})"
            )

        self._add_system_message('\n'.join(lines))

    def _handle_export(self, args: str = ""):
        """Handle /export command."""
        format_arg = args.strip().lower()

        if format_arg and format_arg != 'md':
            self._add_system_message("Unsupported format. Available: md")
            return

        if format_arg == 'md':
            self._export_md()
        else:
            self.prompt_message = "Export format (available: md):"
            self._prompt_for_input(self._export_format_callback)

    def _export_format_callback(self, fmt: str):
        """Callback for export format prompt."""
        fmt = fmt.strip().lower()
        if not fmt:
            self._add_system_message("Export cancelled (no format selected).")
            return
        if fmt != 'md':
            self._add_system_message("Unsupported format. Available: md")
            return
        self._export_md()

    def _export_md(self):
        """Export conversation to Markdown."""
        from pathlib import Path
        from ..core.conversations import export_conversation_md

        target_dir = self.state.config.exports_dir or Path.cwd()

        try:
            path = export_conversation_md(self.state.current_conversation, target_dir)
            self._add_system_message(f"Exported to: {path}")
        except Exception as e:
            self._add_system_message(f"Error exporting: {str(e)}")

    def _handle_stream(self, args: str = ""):
        """Handle /stream command (toggle or set on/off)."""
        arg = args.strip().lower()
        if arg in ('on', 'off'):
            self.state.config.enable_streaming = (arg == 'on')
        elif not arg:
            # Toggle when no arg
            self.state.config.enable_streaming = not self.state.config.enable_streaming
        else:
            self._add_system_message("Usage: /stream [on|off]")
            return

        status = "enabled" if self.state.config.enable_streaming else "disabled"
        self._add_system_message(f"Streaming {status}.")

    def _handle_rename(self, args: str = ""):
        """Handle /rename command."""
        if not self.state.current_conversation.id:
            self._add_system_message("Save the conversation first before renaming (/save).")
            return

        if args.strip():
            self._rename_callback(args.strip())
        else:
            self.prompt_message = "New name for this conversation:"
            self._prompt_for_input(self._rename_callback)

    def _rename_callback(self, new_name: str):
        """Callback for rename input."""
        from ..core.conversations import rename_conversation

        if not new_name.strip():
            self._add_system_message("Rename cancelled (empty name).")
            return

        try:
            rename_conversation(self.state.conversations_root, self.state.current_conversation, new_name)
            self._add_system_message(f"Conversation renamed to: {self.state.current_conversation.id}")
        except FileExistsError as e:
            self._add_system_message(str(e))
        except Exception as e:
            self._add_system_message(f"Error renaming: {str(e)}")

    def _handle_temp(self, args: str = ""):
        """Handle /temp command.

        Args:
            args: Optional arguments (temperature value)
        """
        if args:
            # Temperature provided as argument
            self._temp_callback(args)
        else:
            # Prompt for temperature
            self.prompt_message = "New temperature (0.0-2.0):"
            self._prompt_for_input(self._temp_callback)

    def _temp_callback(self, temp_str: str):
        """Callback for temp input.

        Args:
            temp_str: User input (temperature)
        """
        try:
            temp = float(temp_str)
            result = set_temperature(self.state, temp)
            self._add_system_message(result.message)
        except ValueError:
            self._add_system_message("Invalid temperature. Please enter a number.")

    def _handle_clear(self):
        """Handle /clear command."""
        self.prompt_message = "Clear and delete this conversation? (y/N):"
        self._prompt_for_input(self._clear_callback)

    def _clear_callback(self, confirm: str):
        """Callback for clear confirmation.

        Args:
            confirm: User input (y/N)
        """
        if confirm.lower() == 'y':
            # Delete directory if it exists
            if self.state.current_conversation.id:
                try:
                    delete_conversation(
                        self.state.conversations_root,
                        self.state.current_conversation.id
                    )
                except Exception as e:
                    self._add_system_message(f"Error deleting: {str(e)}")

            # Create new conversation
            from ..core.commands import create_new_conversation
            self.state.current_conversation = create_new_conversation(self.state)
            self._update_conversation_display()
            self._add_system_message("Conversation cleared")
        else:
            self._add_system_message("Clear cancelled")

    def _prompt_for_input(self, callback):
        """Prompt for user input and call callback with result.

        Args:
            callback: Function to call with user input
        """
        # Store callback - will be called when user presses Enter
        self.pending_callback = callback

    def _cleanup(self):
        """Cleanup resources on exit."""
        try:
            print("\nUnloading model...")
            self.state.client.unload_model()

            print("Stopping Ollama server...")
            self.state.client.server_manager.stop()
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


def run_ui(state: AppState):
    """Run the terminal UI.

    Args:
        state: Application state
    """
    ui = LiteChatUI(state)
    ui.run()
