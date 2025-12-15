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

"""Command handlers for ScriptChat UI."""

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from ..core.commands import handle_command, set_model, set_temperature
from ..core.conversations import (
    list_conversations, load_conversation, save_conversation,
    branch_conversation, delete_conversation, rename_conversation, ConversationSummary,
    archive_conversation, unarchive_conversation
)
from ..core.exports import (
    export_conversation_md,
    export_conversation_json,
    export_conversation_html,
    import_conversation_from_file,
    generate_html_index,
)

if TYPE_CHECKING:
    from .app import ScriptChatUI

logger = logging.getLogger(__name__)


class CommandHandlers:
    """Handles all command processing for the UI."""

    def __init__(self, app: 'ScriptChatUI'):
        """Initialize command handlers.

        Args:
            app: Reference to the main ScriptChatUI instance
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
                from ..core.conversations import Message
                self.app.state.current_conversation.messages.append(
                    Message(role='echo', content=message)
                )
                self.app.update_conversation_display()
            return

        result = handle_command(command_line, self.app.state)

        if result.message:
            self.app.add_system_message(result.message)

        # Handle resend (retry)
        if result.resend_message:
            self.app.handle_user_message(result.resend_message)
            return

        if result.should_exit:
            self.app.app.exit()

        # Handle file content - send it as a user message
        if result.file_content:
            self.app.handle_user_message(result.file_content)
            return

        if result.needs_ui_interaction:
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
                self.handle_chats(args)
            elif result.command_type == 'archive':
                self.handle_archive(args)
            elif result.command_type == 'unarchive':
                self.handle_unarchive(args)
            elif result.command_type == 'send':
                self.handle_send(args)
            elif result.command_type == 'export':
                self.handle_export(args)
            elif result.command_type == 'export-all':
                self.handle_export_all(args)
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
            elif result.command_type == 'profile':
                pass
            elif result.command_type == 'undo':
                self.app.update_conversation_display()
            elif result.command_type == 'tag':
                self.app.update_conversation_display()
            elif result.command_type == 'files':
                pass
            elif result.command_type == 'reason':
                self.handle_reason(args)
            elif result.command_type == 'log-level':
                self.handle_log_level(args)

    def handle_model(self, args: str = ""):
        # If name provided directly, apply it (for scripting)
        if args.strip():
            arg = args.strip()
            if '/' in arg:
                provider_id, model_name = arg.split('/', 1)
                self._apply_model_selection(provider_id.strip(), model_name.strip())
            else:
                # Just model name - use current provider
                self._apply_model_selection(
                    self.app.state.current_conversation.provider_id,
                    arg
                )
            return

        # Build options list for selection menu
        options = []
        for provider in self.app.state.config.providers:
            names = [m.name for m in (provider.models or [])]
            if not names and provider.default_model:
                names = [provider.default_model]
            for name in names:
                ctx_display = ""
                models = provider.models or []
                for m in models:
                    if m.name == name and m.context:
                        ctx_display = f" (ctx: {m.context})"
                        break
                display = f"{provider.id}/{name}{ctx_display}"
                options.append(((provider.id, name), display))

        if not options:
            self.app.add_system_message("No models configured")
            return

        # Show selection menu
        self.app.selection_menu.show(
            items=options,
            on_select=self._on_model_selected,
            on_cancel=lambda: self.app.add_system_message("Model selection cancelled")
        )

    def _on_model_selected(self, value: tuple):
        """Handle model selection from menu."""
        provider_id, model_name = value
        self._apply_model_selection(provider_id, model_name)

    def _apply_model_selection(self, provider_id: str, model_name: str):
        try:
            self.app.state.current_conversation.provider_id = provider_id
            result = set_model(self.app.state, model_name)
            msg = result.message or ""
            # Only append provider if set_model didn't already include it
            # (set_model includes provider info for aliases and cross-provider switches)
            if provider_id and "provider:" not in msg:
                msg = f"{msg} (provider: {provider_id})"
            self.app.add_system_message(msg.strip())
        except Exception as e:
            self.app.add_system_message(f"Error setting model: {e}")
        self.app.update_conversation_display()

    def handle_save(self, args: str = ""):
        save_name = args.strip()
        # If conversation already has an id and no new name provided, save in place
        if self.app.state.current_conversation.id and not save_name:
            try:
                save_conversation(
                    self.app.state.conversations_root,
                    self.app.state.current_conversation,
                    save_name=None,
                    system_prompt=self.app.state.current_conversation.system_prompt
                )
                self.app.add_system_message(f"Conversation saved: {self.app.state.current_conversation.id}")
            except Exception as e:
                self.app.add_system_message(f"Error saving: {str(e)}")
            return

        if not save_name:
            self.app.prompt_message = "Save name:"
            self.app.prompt_for_input(self._save_callback)
        else:
            self._save_callback(save_name)

    def _save_callback(self, save_name: str):
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
        # Parse --archived flag
        args = args.strip()
        from_archived = False
        if args.startswith("--archived"):
            from_archived = True
            args = args[10:].strip()  # Remove --archived prefix

        filter_mode = "archived" if from_archived else "active"
        summaries = list_conversations(self.app.state.conversations_root, filter=filter_mode)
        label = "archived" if from_archived else "saved"

        if not summaries:
            self.app.add_system_message(f"No {label} conversations found")
            return
        if args:
            self._load_by_name(args, from_archived=from_archived)
            return

        # Build options for selection menu
        options = []
        for summary in summaries:
            meta_parts = [summary.model_name]
            if summary.last_modified:
                meta_parts.append(summary.last_modified.split("T", 1)[0])
            display = f"{summary.display_name} ({' - '.join(meta_parts)})"
            options.append((summary.dir_name, display))

        # Store from_archived state for callback
        self._load_from_archived = from_archived

        # Show selection menu
        self.app.selection_menu.show(
            items=options,
            on_select=self._on_load_selected,
            on_cancel=lambda: self.app.add_system_message("Load cancelled")
        )

    def _on_load_selected(self, dir_name: str):
        """Handle conversation selection from menu."""
        from_archived = getattr(self, '_load_from_archived', False)
        self._load_by_name(dir_name, from_archived=from_archived)

    def _load_by_name(self, name: str, from_archived: bool = False):
        from ..core.conversations import ARCHIVE_DIR

        filter_mode = "archived" if from_archived else "active"
        summaries = list_conversations(self.app.state.conversations_root, filter=filter_mode)

        # Match by name (display_name or dir_name)
        name = name.strip()
        matches = [
            s for s in summaries
            if s.display_name == name or s.dir_name == name
            or name in s.display_name or name in s.dir_name
        ]

        if len(matches) == 0:
            self.app.add_system_message(f"Conversation not found: {name}")
            return
        if len(matches) > 1:
            match_names = [m.display_name for m in matches[:5]]
            hint = ", ".join(match_names)
            if len(matches) > 5:
                hint += f", ... ({len(matches)} total)"
            self.app.add_system_message(
                f"Multiple conversations match '{name}': {hint}"
            )
            return

        target = matches[0].dir_name
        display = matches[0].display_name

        # Determine load directory
        if from_archived:
            load_root = self.app.state.conversations_root / ARCHIVE_DIR
        else:
            load_root = self.app.state.conversations_root

        conversation = load_conversation(load_root, target)
        self.app.state.current_conversation = conversation
        # Rehydrate file registry for placeholder expansion
        self.app.state.file_registry = getattr(conversation, "file_registry", {})
        missing = [
            key for key, entry in self.app.state.file_registry.items()
            if isinstance(entry, dict) and entry.get("missing")
        ]
        self.app.update_conversation_display()
        self.app.add_system_message(f"Loaded: {display}")
        if missing:
            self.app.add_system_message(
                "Warning: missing file(s) referenced: " + ", ".join(sorted(missing))
            )

    def handle_branch(self, args: str = ""):
        if args.strip():
            # Name provided, branch immediately
            self._branch_callback(args.strip())
        else:
            # Prompt for name with default
            default_name = self._get_default_branch_name()
            self.app.prompt_message = f"Branch name [{default_name}]:"
            self.app.prompt_for_input(lambda name: self._branch_callback(name or default_name))

    def _get_default_branch_name(self) -> str:
        """Generate default branch name based on current conversation."""
        convo = self.app.state.current_conversation
        if convo.id:
            parts = convo.id.split('_', 2)
            if len(parts) >= 3:
                base_name = parts[2]
            else:
                base_name = 'untitled'
        else:
            base_name = 'untitled'

        # Find next available branch number
        root = self.app.state.conversations_root
        branch_num = 1
        while True:
            candidate = f"{base_name}-branch-{branch_num}"
            # Check if any conversation already uses this name
            exists = False
            if root.exists():
                for d in root.iterdir():
                    if d.is_dir() and d.name.endswith(f"_{candidate}"):
                        exists = True
                        break
            if not exists:
                return candidate
            branch_num += 1

    def _branch_callback(self, new_save_name: str):
        try:
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
        try:
            new_name = args.strip()
            if new_name:
                self._rename_callback(new_name)
                return
        except Exception:
            pass
        self.app.prompt_message = "New name:"
        self.app.prompt_for_input(self._rename_callback)

    def _rename_callback(self, new_name: str):
        if not new_name.strip():
            self.app.add_system_message("Rename cancelled (empty name)")
            return
        try:
            self.app.state.current_conversation = rename_conversation(
                self.app.state.conversations_root,
                self.app.state.current_conversation,
                new_name
            )
            self.app.add_system_message(f"Renamed to: {new_name}")
        except FileExistsError:
            self.app.add_system_message(f"Conversation already exists: {new_name}")
        except Exception as e:
            self.app.add_system_message(f"Error renaming: {str(e)}")

    def handle_chats(self, args: str = ""):
        # Parse flags
        args = args.strip()
        if args == "--archived":
            filter_mode = "archived"
            label = "archived"
        elif args == "--all":
            filter_mode = "all"
            label = "all"
        elif args:
            self.app.add_system_message("Usage: /chats [--archived|--all]")
            return
        else:
            filter_mode = "active"
            label = "saved"

        summaries = list_conversations(self.app.state.conversations_root, filter=filter_mode)
        if not summaries:
            self.app.add_system_message(f"No {label} conversations found")
            return
        lines = format_conversation_list(summaries)
        self.app.add_system_message('\n'.join(lines))

    def handle_reason(self, args: str = ""):
        """Handle /reason command with selection menu."""
        from ..core.commands import set_reasoning_level
        from ..core.config import reasoning_levels_for_model

        if args.strip():
            # Direct argument provided - use existing logic
            result = set_reasoning_level(self.app.state, args.strip())
            self.app.add_system_message(result.message)
            return

        # Check if reasoning is available for current model
        available = reasoning_levels_for_model(
            self.app.state.config,
            self.app.state.current_conversation.provider_id,
            self.app.state.current_conversation.model_name
        )

        if not available:
            self.app.add_system_message("Reasoning controls not available for this model/provider.")
            return

        # Build options
        options = [(level, level.capitalize()) for level in available]
        options.append(("clear", "Clear (provider default)"))

        self.app.selection_menu.show(
            items=options,
            on_select=self._on_reason_selected,
            on_cancel=lambda: self.app.add_system_message("Reason selection cancelled")
        )

    def _on_reason_selected(self, level: str):
        """Handle reason level selection from menu."""
        from ..core.commands import set_reasoning_level
        result = set_reasoning_level(self.app.state, level)
        self.app.add_system_message(result.message)

    def handle_log_level(self, args: str = ""):
        """Handle /log-level command with selection menu."""
        if args.strip():
            # Direct argument provided
            level = args.strip().upper()
            if level == "WARN":
                level = "WARNING"
            if level not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
                self.app.add_system_message("Invalid level. Options: debug, info, warn, error, critical")
                return
            import logging
            root_logger = logging.getLogger()
            root_logger.setLevel(level)
            for handler in root_logger.handlers:
                handler.setLevel(level)
            self.app.state.config.log_level = level
            self.app.add_system_message(f"Log level set to {level}")
            return

        # Show selection menu
        levels = [
            ("DEBUG", "Debug - Detailed diagnostic information"),
            ("INFO", "Info - General operational messages"),
            ("WARNING", "Warning - Potential issues"),
            ("ERROR", "Error - Error conditions"),
            ("CRITICAL", "Critical - Severe errors"),
        ]

        self.app.selection_menu.show(
            items=levels,
            on_select=self._on_log_level_selected,
            on_cancel=lambda: self.app.add_system_message("Log level selection cancelled")
        )

    def _on_log_level_selected(self, level: str):
        """Handle log level selection from menu."""
        import logging
        root_logger = logging.getLogger()
        root_logger.setLevel(level)
        for handler in root_logger.handlers:
            handler.setLevel(level)
        self.app.state.config.log_level = level
        self.app.add_system_message(f"Log level set to {level}")

    def handle_archive(self, args: str):
        """Archive conversations by index, name, range, or tag filter."""
        self._handle_archive_operation(args, is_archive=True)

    def handle_unarchive(self, args: str):
        """Unarchive conversations by index, name, range, or tag filter."""
        self._handle_archive_operation(args, is_archive=False)

    def _handle_archive_operation(self, args: str, is_archive: bool):
        """Common handler for archive/unarchive operations."""
        args = args.strip()
        cmd = "archive" if is_archive else "unarchive"
        source_filter = "active" if is_archive else "archived"
        op_func = archive_conversation if is_archive else unarchive_conversation

        if not args:
            self.app.add_system_message(f"Usage: /{cmd} [index|name|range] [--tag key=value]")
            return

        # Get source conversation list
        summaries = list_conversations(self.app.state.conversations_root, filter=source_filter)
        if not summaries:
            label = "active" if is_archive else "archived"
            self.app.add_system_message(f"No {label} conversations found")
            return

        # Check for --tag flag
        if args.startswith("--tag "):
            tag_arg = args[6:].strip()
            if "=" not in tag_arg:
                self.app.add_system_message(f"Usage: /{cmd} --tag key=value")
                return
            key, value = tag_arg.split("=", 1)
            key, value = key.strip(), value.strip()

            # Filter by tag
            matches = [s for s in summaries if s.tags.get(key) == value]
            if not matches:
                self.app.add_system_message(f"No conversations found with tag {key}={value}")
                return

            self._perform_archive_batch(matches, op_func, cmd)
            return

        # Check for range (e.g., "1-5")
        range_match = re.match(r'^(\d+)-(\d+)$', args)
        if range_match:
            start, end = int(range_match.group(1)), int(range_match.group(2))
            if start > end:
                start, end = end, start
            if start < 0 or end >= len(summaries):
                self.app.add_system_message(f"Index range out of bounds (0-{len(summaries)-1})")
                return
            matches = summaries[start:end+1]
            self._perform_archive_batch(matches, op_func, cmd)
            return

        # Check for single index
        if args.isdigit():
            idx = int(args)
            if idx < 0 or idx >= len(summaries):
                self.app.add_system_message(f"Index out of range (0-{len(summaries)-1})")
                return
            matches = [summaries[idx]]
            self._perform_archive_batch(matches, op_func, cmd)
            return

        # Try to match by name (partial match)
        matches = [s for s in summaries if args.lower() in s.display_name.lower() or args.lower() in s.dir_name.lower()]
        if not matches:
            self.app.add_system_message(f"No conversation found matching '{args}'")
            return
        if len(matches) > 1:
            self.app.add_system_message(f"Multiple matches for '{args}'. Use index or be more specific:")
            for i, s in enumerate(summaries):
                if s in matches:
                    self.app.add_system_message(f"  {i}: {s.display_name}")
            return

        self._perform_archive_batch(matches, op_func, cmd)

    def _perform_archive_batch(self, summaries: list, op_func, cmd: str):
        """Perform archive/unarchive operation on a batch of conversations."""
        success = 0
        errors = []
        for s in summaries:
            try:
                op_func(self.app.state.conversations_root, s.dir_name)
                success += 1
            except Exception as e:
                errors.append(f"{s.display_name}: {e}")

        past_tense = "archived" if cmd == "archive" else "unarchived"
        if success:
            self.app.add_system_message(f"{past_tense.capitalize()} {success} conversation(s)")
        for err in errors:
            self.app.add_system_message(f"Error: {err}")

    def handle_send(self, args: str):
        msg = args.strip()
        if not msg:
            self.app.add_system_message("Usage: /send <message>")
            return
        self.app.handle_user_message(msg)

    def handle_run(self, args: str):
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
        fmt = args.strip().lower()
        if fmt and fmt not in ('md', 'json', 'html'):
            self.app.add_system_message("Unsupported format. Available: md, json, html")
            return
        if fmt == 'md':
            self._export_md()
        elif fmt == 'json':
            self._export_json()
        elif fmt == 'html':
            self._export_html()
        else:
            self.app.prompt_message = "Export format (available: md, json, html):"
            self.app.prompt_for_input(self._export_format_callback)

    def _export_format_callback(self, fmt: str):
        fmt = fmt.strip().lower()
        if not fmt:
            self.app.add_system_message("Export cancelled (no format selected).")
            return
        if fmt not in ('md', 'json', 'html'):
            self.app.add_system_message("Unsupported format. Available: md, json, html")
            return
        if fmt == 'md':
            self._export_md()
        elif fmt == 'json':
            self._export_json()
        else:
            self._export_html()

    def _export_md(self):
        target_dir = self.app.state.config.exports_dir or Path.cwd()
        try:
            path = export_conversation_md(self.app.state.current_conversation, target_dir)
            self.app.add_system_message(f"Exported to: {path}")
        except Exception as e:
            self.app.add_system_message(f"Error exporting: {str(e)}")

    def _export_json(self):
        target_dir = self.app.state.config.exports_dir or Path.cwd()
        try:
            path = export_conversation_json(self.app.state.current_conversation, target_dir)
            self.app.add_system_message(f"Exported to: {path}")
        except Exception as e:
            self.app.add_system_message(f"Error exporting: {str(e)}")

    def _export_html(self):
        target_dir = self.app.state.config.exports_dir or Path.cwd()
        try:
            path = export_conversation_html(self.app.state.current_conversation, target_dir)
            # Regenerate index.html
            generate_html_index(target_dir, self.app.state.conversations_root)
            self.app.add_system_message(f"Exported to: {path}")
        except Exception as e:
            self.app.add_system_message(f"Error exporting: {str(e)}")

    def handle_export_all(self, args: str = ""):
        fmt = args.strip().lower()
        if fmt and fmt not in ('md', 'json', 'html'):
            self.app.add_system_message("Unsupported format. Available: md, json, html")
            return
        if fmt:
            self._export_all(fmt)
        else:
            self.app.prompt_message = "Export format (available: md, json, html):"
            self.app.prompt_for_input(self._export_all_format_callback)

    def _export_all_format_callback(self, fmt: str):
        fmt = fmt.strip().lower()
        if not fmt:
            self.app.add_system_message("Export cancelled (no format selected).")
            return
        if fmt not in ('md', 'json', 'html'):
            self.app.add_system_message("Unsupported format. Available: md, json, html")
            return
        self._export_all(fmt)

    def _export_all(self, fmt: str):
        target_dir = self.app.state.config.exports_dir or Path.cwd()
        conversations = list_conversations(self.app.state.conversations_root)
        if not conversations:
            self.app.add_system_message("No saved conversations to export.")
            return

        exported = 0
        errors = 0
        for summary in conversations:
            try:
                conv = load_conversation(self.app.state.conversations_root, summary.dir_name)
                if fmt == 'md':
                    export_conversation_md(conv, target_dir)
                elif fmt == 'json':
                    export_conversation_json(conv, target_dir)
                else:
                    export_conversation_html(conv, target_dir)
                exported += 1
            except Exception as e:
                logger.error(f"Error exporting {summary.dir_name}: {e}")
                errors += 1

        # Regenerate index.html if HTML format
        if fmt == 'html':
            generate_html_index(target_dir, self.app.state.conversations_root)

        msg = f"Exported {exported} conversation(s) to {fmt.upper()}"
        if errors:
            msg += f" ({errors} error(s))"
        self.app.add_system_message(msg)

    def handle_import(self, args: str):
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
        arg = args.strip().lower()
        if arg in ('on', 'off'):
            self.app.state.config.enable_streaming = (arg == 'on')
        elif not arg:
            self.app.state.config.enable_streaming = not self.app.state.config.enable_streaming
        else:
            self.app.add_system_message("Usage: /stream [on|off]")
            return
        status = "enabled" if self.app.state.config.enable_streaming else "disabled"
        self.app.add_system_message(f"Streaming {status}.")

    def handle_prompt(self, args: str = ""):
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
        text = prompt_text.strip()
        if text:
            self._set_system_prompt(text)
            self.app.add_system_message("System prompt set for this conversation.")
        else:
            self._set_system_prompt(None)
            self.app.add_system_message("System prompt cleared for this conversation.")

    def _set_system_prompt(self, prompt: Optional[str]):
        if self.app.state.current_conversation.messages and self.app.state.current_conversation.messages[0].role == 'system':
            self.app.state.current_conversation.messages.pop(0)
        self.app.state.current_conversation.system_prompt = prompt if prompt else None
        if prompt:
            from ..core.conversations import Message
            self.app.state.current_conversation.messages.insert(0, Message(role='system', content=prompt))
        self.app.update_conversation_display()

    def handle_temp(self, args: str = ""):
        if args.strip():
            self._temp_callback(args.strip())
        else:
            current_temp = self.app.state.current_conversation.temperature
            self.app.add_system_message(f"Current temperature: {current_temp:.2f}")
            self.app.prompt_message = "New temperature (0.0-2.0):"
            self.app.prompt_for_input(self._temp_callback)

    def _temp_callback(self, temp_str: str):
        try:
            temp = float(temp_str)
            result = set_temperature(self.app.state, temp)
            self.app.add_system_message(result.message)
        except ValueError:
            self.app.add_system_message("Invalid temperature. Please enter a number.")

    def handle_clear(self, args: str = ""):
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
                idx = int(args.strip())
                self.pending_clear_index = idx
            except ValueError:
                self.app.add_system_message("Usage: /clear [index]")
                return
        self.pending_clear_target_id = target_id
        self.pending_clear_is_current = is_current
        self.app.prompt_message = prompt
        self.app.prompt_for_input(self._clear_callback, expect_single_key=True)

    def _clear_callback(self, confirm: str):
        if confirm.lower() != 'y':
            self.app.add_system_message("Clear cancelled")
            self.pending_clear_index = None
            self.pending_clear_target_id = None
            self.pending_clear_is_current = False
            return
        target_id = self.pending_clear_target_id
        is_current = self.pending_clear_is_current
        if is_current:
            from ..core.commands import create_new_conversation
            self.app.state.current_conversation = create_new_conversation(self.app.state)
            self.app.update_conversation_display()
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


def format_conversation_list(summaries: list[ConversationSummary]) -> list[str]:
    """Format conversation summaries with tags and branch hierarchy."""
    lines = ["Saved conversations:"]

    # Build parent->children map and identify root conversations
    children_map: dict[str, list[int]] = {}  # parent_id -> list of summary indices
    for i, summary in enumerate(summaries):
        if summary.parent_id:
            if summary.parent_id not in children_map:
                children_map[summary.parent_id] = []
            children_map[summary.parent_id].append(i)

    # Track which summaries have been rendered (to avoid duplicates)
    rendered = set()

    def format_summary(idx: int, indent: str = "  ") -> list[str]:
        """Format a single summary and its children recursively."""
        if idx in rendered:
            return []
        rendered.add(idx)

        summary = summaries[idx]
        result = []

        tag_str = ""
        if getattr(summary, "tags", None):
            tag_str = " tags: " + ", ".join(f"{k}={v}" for k, v in summary.tags.items())
        last_mod_raw = summary.last_modified or ""
        last_mod = last_mod_raw.split("T", 1)[0] if last_mod_raw else ""
        model = summary.model_name
        meta_parts = [model]
        if last_mod:
            meta_parts.append(last_mod)
        if tag_str:
            meta_parts.append(tag_str.strip())
        meta_text = " - ".join(meta_parts)
        result.append(f"{indent}[{idx}] {summary.display_name} ({meta_text})")

        # Render children (branches) indented below
        if summary.dir_name in children_map:
            for child_idx in children_map[summary.dir_name]:
                result.extend(format_summary(child_idx, indent + "  └─ "))

        return result

    # Render root conversations (those without parents, or whose parent doesn't exist)
    existing_dirs = {s.dir_name for s in summaries}
    for i, summary in enumerate(summaries):
        # Root if no parent or parent doesn't exist in current list
        if not summary.parent_id or summary.parent_id not in existing_dirs:
            lines.extend(format_summary(i))

    return lines
