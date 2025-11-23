"""Command handlers for lite-chat UI."""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from ..core.commands import handle_command, set_model, set_temperature
from ..core.conversations import (
    list_conversations, load_conversation, save_conversation,
    branch_conversation, delete_conversation
)
from ..core.exports import (
    export_conversation_md,
    export_conversation_json,
    export_conversation_html,
    import_conversation_from_file,
)

if TYPE_CHECKING:
    from .app import LiteChatUI

logger = logging.getLogger(__name__)


class CommandHandlers:
    """Handles all command processing for the UI."""

    def __init__(self, app: 'LiteChatUI'):
        """Initialize command handlers.

        Args:
            app: Reference to the main LiteChatUI instance
        """
        self.app = app

        # Handler-specific state
        self.last_model_options: list[tuple[str, str]] = []
        self.pending_clear_index: Optional[int] = None
        self.pending_clear_target_id: Optional[str] = None
        self.pending_clear_is_current: bool = False

    def handle_command(self, command_line: str):
        """Handle a command.

        Args:
            command_line: Command line starting with /
        """
        # Parse command and arguments
        parts = command_line[1:].split(maxsplit=1)
        cmd = parts[0].lower() if parts else ""
        args = parts[1] if len(parts) > 1 else ""

        # Handle /echo specially - display without adding to conversation
        if cmd == 'echo':
            message = args if args else ""
            # Display echo message directly in conversation pane
            if message:
                # Temporarily add to display, but mark it somehow
                from ..core.conversations import Message
                # Add as a special message that won't be sent to LLM
                # We'll use a marker to identify echo messages
                self.app.state.current_conversation.messages.append(
                    Message(role='echo', content=message)
                )
                self.app.update_conversation_display()
            return

        result = handle_command(command_line, self.app.state)

        if result.message:
            self.app.add_system_message(result.message)

        if result.should_exit:
            self.app.app.exit()

        # Handle file content - send it as a user message
        if result.file_content:
            self.app.handle_user_message(result.file_content)
            return

        if result.needs_ui_interaction:
            # Handle commands that need user interaction
            if result.command_type == 'model':
                self.handle_model(args)
            elif result.command_type == 'save':
                self.handle_save(args)
            elif result.command_type == 'load':
                self.handle_load(args)
            elif result.command_type == 'branch':
                self.handle_branch(args)
            elif result.command_type == 'rename':
                self.handle_rename(args)
            elif result.command_type == 'chats':
                self.handle_chats()
            elif result.command_type == 'send':
                self.handle_send(args)
            elif result.command_type == 'export':
                self.handle_export(args)
            elif result.command_type == 'import':
                self.handle_import(args)
            elif result.command_type == 'stream':
                self.handle_stream(args)
            elif result.command_type == 'prompt':
                self.handle_prompt(args)
            elif result.command_type == 'run':
                self.handle_run(args)
            elif result.command_type == 'temp':
                self.handle_temp(args)
            elif result.command_type == 'clear':
                self.handle_clear(args)

    def handle_model(self, args: str = ""):
        """Handle /model command.

        Args:
            args: Optional arguments (model index or provider/model)
        """
        # If index provided as argument, use it directly
        if args.strip():
            self._model_callback(args.strip())
            return

        # Build combined model list across providers
        options = []
        lines = ["Available models (all providers):"]
        idx = 0
        for provider in self.app.state.config.providers:
            names = [m.name for m in (provider.models or [])]
            if not names and provider.default_model:
                names = [provider.default_model]
            for name in names:
                ctx_display = ""
                models = provider.models or []
                for m in models:
                    if m.name == name and m.contexts:
                        ctx_display = f" (contexts: {', '.join(str(c) for c in m.contexts)})"
                        break
                lines.append(f"  [{idx}] {provider.id}/{name}{ctx_display}")
                options.append((provider.id, name))
                idx += 1

        if not options:
            lines.append("  (no models listed; enter provider/model manually)")

        self.last_model_options = options

        self.app.add_system_message('\n'.join(lines))
        self.app.prompt_message = "Enter model index or provider/model:"

        # Get user input
        self.app.prompt_for_input(self._model_callback)

    def _model_callback(self, index_str: str):
        """Callback for model input.

        Args:
            index_str: User input (model index or name)
        """
        # Try index
        try:
            index = int(index_str)
            if 0 <= index < len(self.last_model_options):
                provider_id, model_name = self.last_model_options[index]
                self._apply_model_selection(provider_id, model_name)
                return
        except ValueError:
            pass

        # Try provider/model
        if '/' in index_str:
            provider_id, model_name = index_str.split('/', 1)
            provider_id = provider_id.strip()
            model_name = model_name.strip()
            self._apply_model_selection(provider_id, model_name)
            return

        # Fallback: model name on current provider
        model_name = index_str.strip()
        if model_name:
            self._apply_model_selection(self.app.state.current_conversation.provider_id, model_name)
        else:
            self.app.add_system_message("Invalid input. Enter model index or provider/model.")

    def _apply_model_selection(self, provider_id: str, model_name: str):
        """Apply a model selection, switching provider if needed."""
        try:
            # Switch provider and set model
            self.app.state.current_conversation.provider_id = provider_id
            result = set_model(self.app.state, model_name)
            self.app.add_system_message(f"{result.message} (provider: {provider_id})")
        except Exception as e:
            self.app.add_system_message(str(e))

    def handle_save(self, save_name: str = ""):
        """Handle /save command.

        Args:
            save_name: Optional save name from command argument
        """
        if self.app.state.current_conversation.id:
            # Already saved, just update
            save_conversation(
                self.app.state.conversations_root,
                self.app.state.current_conversation,
                system_prompt=self.app.state.current_conversation.system_prompt
            )
            self.app.add_system_message("Conversation saved")
        else:
            # Need save name
            if save_name:
                # Name provided as argument
                self._save_callback(save_name)
            else:
                # Prompt for name
                self.app.prompt_message = "Save conversation as:"
                self.app.prompt_for_input(self._save_callback)

    def _save_callback(self, save_name: str):
        """Callback for save input.

        Args:
            save_name: User-provided save name
        """
        if not save_name.strip():
            self.app.add_system_message("Save cancelled (empty name)")
            return

        try:
            save_conversation(
                self.app.state.conversations_root,
                self.app.state.current_conversation,
                save_name=save_name,
                system_prompt=self.app.state.current_conversation.system_prompt
            )
            self.app.add_system_message(f"Conversation saved as: {save_name}")
        except Exception as e:
            self.app.add_system_message(f"Error saving: {str(e)}")

    def handle_load(self, args: str = ""):
        """Handle /load command.

        Args:
            args: Optional arguments (conversation index)
        """
        # List available conversations
        summaries = list_conversations(self.app.state.conversations_root)

        if not summaries:
            self.app.add_system_message("No saved conversations found")
            return

        # If index provided as argument
        if args.strip():
            self._load_callback(args.strip())
            return

        # Display list
        lines = ["Saved conversations:"]
        for i, summary in enumerate(summaries):
            lines.append(f"  [{i}] {summary.display_name}")

        self.app.add_system_message('\n'.join(lines))
        self.app.prompt_message = "Enter conversation index:"

        # Get user input
        self.app.prompt_for_input(self._load_callback)

    def _load_callback(self, index_str: str):
        """Callback for load input.

        Args:
            index_str: User input (conversation index)
        """
        try:
            index = int(index_str)
            summaries = list_conversations(self.app.state.conversations_root)

            if 0 <= index < len(summaries):
                conversation = load_conversation(
                    self.app.state.conversations_root,
                    summaries[index].dir_name
                )
                self.app.state.current_conversation = conversation
                self.app.update_conversation_display()
                self.app.add_system_message(f"Loaded: {summaries[index].display_name}")
            else:
                self.app.add_system_message("Invalid conversation index")
        except ValueError:
            self.app.add_system_message("Invalid index. Please enter a number.")

    def handle_branch(self, args: str = ""):
        """Handle /branch command.

        Args:
            args: Optional arguments (new save name)
        """
        try:
            # Use args as new save name if provided, otherwise None (auto-generate)
            new_save_name = args if args else None
            new_convo = branch_conversation(
                self.app.state.conversations_root,
                self.app.state.current_conversation,
                new_save_name=new_save_name,
                system_prompt=self.app.state.current_conversation.system_prompt
            )
            self.app.state.current_conversation = new_convo
            self.app.add_system_message(f"Branched to: {new_convo.id}")
        except Exception as e:
            self.app.add_system_message(f"Error branching: {str(e)}")

    def handle_rename(self, args: str = ""):
        """Handle /rename command.

        Args:
            args: Optional new name
        """
        if not self.app.state.current_conversation.id:
            self.app.add_system_message("Save the conversation first before renaming (/save).")
            return

        if args.strip():
            self._rename_callback(args.strip())
        else:
            self.app.prompt_message = "New name:"
            self.app.prompt_for_input(self._rename_callback)

    def _rename_callback(self, new_name: str):
        """Callback for rename input.

        Args:
            new_name: New conversation name
        """
        if not new_name.strip():
            self.app.add_system_message("Rename cancelled (empty name)")
            return

        try:
            old_dir = self.app.state.conversations_root / self.app.state.current_conversation.id
            new_dir_name = f"{self.app.state.current_conversation.model_name}_{new_name}"
            new_dir = self.app.state.conversations_root / new_dir_name

            if new_dir.exists():
                self.app.add_system_message(f"Conversation already exists: {new_name}")
                return

            old_dir.rename(new_dir)
            self.app.state.current_conversation.id = new_dir_name
            self.app.add_system_message(f"Renamed to: {new_name}")
        except Exception as e:
            self.app.add_system_message(f"Error renaming: {str(e)}")

    def handle_chats(self):
        """Handle /chats command - list all conversations."""
        summaries = list_conversations(self.app.state.conversations_root)

        if not summaries:
            self.app.add_system_message("No saved conversations found")
            return

        lines = ["Saved conversations:"]
        for i, summary in enumerate(summaries):
            lines.append(f"  [{i}] {summary.display_name}")

        self.app.add_system_message('\n'.join(lines))

    def handle_send(self, args: str):
        """Handle /send command - enqueue or send immediately."""
        msg = args.strip()
        if not msg:
            self.app.add_system_message("Usage: /send <message>")
            return
        self.app.handle_user_message(msg)

    def handle_run(self, args: str):
        """Handle /run command to execute scripted commands/messages from a file."""
        path = args.strip()
        if not path:
            self.app.add_system_message("Usage: /run <path>")
            return

        try:
            with open(Path(path).expanduser(), 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except FileNotFoundError:
            self.app.add_system_message(f"File not found: {path}")
            return
        except Exception as e:
            self.app.add_system_message(f"Error reading {path}: {e}")
            return

        script_lines = self.app.parse_script_lines(lines)
        if not script_lines:
            self.app.add_system_message(f"No runnable lines in {path} (comments/empty only).")
            return

        self.app.add_system_message(f"Running script: {path} ({len(script_lines)} lines)")
        self.app.run_script_queue(script_lines)

    def handle_export(self, args: str = ""):
        """Handle /export command."""
        format_arg = args.strip().lower()

        if format_arg and format_arg not in ('md', 'json', 'html'):
            self.app.add_system_message("Unsupported format. Available: md, json, html")
            return

        if format_arg == 'md':
            self._export_md()
        elif format_arg == 'json':
            self._export_json()
        elif format_arg == 'html':
            self._export_html()
        else:
            self.app.prompt_message = "Export format (available: md, json, html):"
            self.app.prompt_for_input(self._export_format_callback)

    def _export_format_callback(self, fmt: str):
        """Callback for export format prompt."""
        fmt = fmt.strip().lower()
        if not fmt:
            self.app.add_system_message("Export cancelled (no format selected).")
            return
        if fmt not in ('md', 'json', 'html'):
            self.app.add_system_message("Unsupported format. Available: md, json, html")
            return
        if fmt == 'md':
            self._export_md()
        else:
            if fmt == 'json':
                self._export_json()
            else:
                self._export_html()

    def _export_md(self):
        """Export conversation to Markdown."""
        target_dir = self.app.state.config.exports_dir or Path.cwd()

        try:
            path = export_conversation_md(self.app.state.current_conversation, target_dir)
            self.app.add_system_message(f"Exported to: {path}")
        except Exception as e:
            self.app.add_system_message(f"Error exporting: {str(e)}")

    def _export_json(self):
        """Export conversation to JSON."""
        target_dir = self.app.state.config.exports_dir or Path.cwd()

        try:
            path = export_conversation_json(self.app.state.current_conversation, target_dir)
            self.app.add_system_message(f"Exported to: {path}")
        except Exception as e:
            self.app.add_system_message(f"Error exporting: {str(e)}")

    def _export_html(self):
        """Export conversation to HTML."""
        target_dir = self.app.state.config.exports_dir or Path.cwd()

        try:
            path = export_conversation_html(self.app.state.current_conversation, target_dir)
            self.app.add_system_message(f"Exported to: {path}")
        except Exception as e:
            self.app.add_system_message(f"Error exporting: {str(e)}")

    def handle_import(self, args: str):
        """Handle /import command (load from exported file)."""
        path = args.strip()
        if not path:
            self.app.add_system_message("Usage: /import <path>")
            return

        try:
            imported = import_conversation_from_file(Path(path), self.app.state.conversations_root)
            self.app.state.current_conversation = imported
            self.app.add_system_message(f"Imported conversation as: {imported.id}")
            self.app.update_conversation_display()
        except Exception as e:
            self.app.add_system_message(f"Error importing: {e}")

    def handle_stream(self, args: str = ""):
        """Handle /stream command (toggle or set on/off)."""
        arg = args.strip().lower()
        if arg in ('on', 'off'):
            self.app.state.config.enable_streaming = (arg == 'on')
        elif not arg:
            # Toggle when no arg
            self.app.state.config.enable_streaming = not self.app.state.config.enable_streaming
        else:
            self.app.add_system_message("Usage: /stream [on|off]")
            return

        status = "enabled" if self.app.state.config.enable_streaming else "disabled"
        self.app.add_system_message(f"Streaming {status}.")

    def handle_prompt(self, args: str = ""):
        """Handle /prompt command (set/clear system prompt for this conversation)."""
        arg = args.strip()

        if arg:
            if arg.lower() == 'clear':
                self._set_system_prompt(None)
                self.app.add_system_message("System prompt cleared for this conversation.")
            else:
                self._set_system_prompt(arg)
                self.app.add_system_message("System prompt set for this conversation.")
            return

        self.app.prompt_message = "New system prompt (empty to clear):"
        self.app.prompt_for_input(self._prompt_callback)

    def _prompt_callback(self, prompt_text: str):
        """Callback for system prompt input."""
        text = prompt_text.strip()
        if text:
            self._set_system_prompt(text)
            self.app.add_system_message("System prompt set for this conversation.")
        else:
            self._set_system_prompt(None)
            self.app.add_system_message("System prompt cleared for this conversation.")

    def _set_system_prompt(self, prompt: Optional[str]):
        """Set or clear the system prompt for the current conversation."""
        # Remove existing leading system message if present
        if self.app.state.current_conversation.messages and self.app.state.current_conversation.messages[0].role == 'system':
            self.app.state.current_conversation.messages.pop(0)

        self.app.state.current_conversation.system_prompt = prompt if prompt else None

        if prompt:
            from ..core.conversations import Message
            self.app.state.current_conversation.messages.insert(0, Message(role='system', content=prompt))

        self.app.update_conversation_display()

    def handle_temp(self, args: str = ""):
        """Handle /temp command."""
        if args.strip():
            self._temp_callback(args.strip())
        else:
            current_temp = self.app.state.current_conversation.temperature
            self.app.add_system_message(f"Current temperature: {current_temp:.2f}")
            self.app.prompt_message = "New temperature (0.0-2.0):"
            self.app.prompt_for_input(self._temp_callback)

    def _temp_callback(self, temp_str: str):
        """Callback for temp input.

        Args:
            temp_str: User input (temperature)
        """
        try:
            temp = float(temp_str)
            result = set_temperature(self.app.state, temp)
            self.app.add_system_message(result.message)
        except ValueError:
            self.app.add_system_message("Invalid temperature. Please enter a number.")

    def handle_clear(self, args: str = ""):
        """Handle /clear command."""
        from .app import resolve_clear_target_from_args

        self.pending_clear_index = None
        target_id, prompt, error, summaries_used, is_current = resolve_clear_target_from_args(
            args,
            self.app.state.conversations_root,
            self.app.state.current_conversation.id
        )

        if error:
            self.app.add_system_message(error)
            return

        if args.strip():
            try:
                self.pending_clear_index = int(args.strip())
            except ValueError:
                self.pending_clear_index = None
        self.pending_clear_target_id = target_id
        self.pending_clear_is_current = is_current
        self.app.prompt_message = prompt

        self.app.prompt_for_input(self._clear_callback, expect_single_key=True)

    def _clear_callback(self, confirm: str):
        """Callback for clear confirmation.

        Args:
            confirm: User input (y/N)
        """
        if confirm.lower() != 'y':
            self.app.add_system_message("Clear cancelled")
            self.pending_clear_index = None
            self.pending_clear_target_id = None
            self.pending_clear_is_current = False
            return

        target_id = self.pending_clear_target_id
        is_current = self.pending_clear_is_current

        # If target is current conversation, create new one
        if is_current:
            from ..core.commands import create_new_conversation
            self.app.state.current_conversation = create_new_conversation(self.app.state)
            self.app.update_conversation_display()

        # Delete saved conversation if it exists
        if target_id:
            try:
                delete_conversation(self.app.state.conversations_root, target_id)
            except Exception as e:
                logger.error(f"Error deleting conversation: {e}")
                self.app.add_system_message(f"Error deleting: {str(e)}")
                return

        if is_current:
            self.app.add_system_message("Conversation cleared")
        else:
            self.app.add_system_message(f"Conversation deleted: {target_id}")

        self.pending_clear_index = None
        self.pending_clear_target_id = None
        self.pending_clear_is_current = False
