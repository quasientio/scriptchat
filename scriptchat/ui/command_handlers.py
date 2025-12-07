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
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from ..core.commands import handle_command, set_model, set_temperature
from ..core.conversations import (
    list_conversations, load_conversation, save_conversation,
    branch_conversation, delete_conversation, rename_conversation, ConversationSummary
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
                self.handle_chats()
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

    def handle_model(self, args: str = ""):
        if args.strip():
            self._model_callback(args.strip())
            return
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
        self.app.prompt_for_input(self._model_callback)

    def _model_callback(self, index_str: str):
        try:
            index = int(index_str)
            if 0 <= index < len(self.last_model_options):
                provider_id, model_name = self.last_model_options[index]
                self._apply_model_selection(provider_id, model_name)
                return
        except ValueError:
            pass
        if '/' in index_str:
            provider_id, model_name = index_str.split('/', 1)
            self._apply_model_selection(provider_id.strip(), model_name.strip())
            return
        model_name = index_str.strip()
        if model_name:
            self._apply_model_selection(self.app.state.current_conversation.provider_id, model_name)
        else:
            self.app.add_system_message("Invalid input. Enter model index or provider/model.")

    def _apply_model_selection(self, provider_id: str, model_name: str):
        try:
            self.app.state.current_conversation.provider_id = provider_id
            result = set_model(self.app.state, model_name)
            msg = result.message or ""
            if provider_id:
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
        summaries = list_conversations(self.app.state.conversations_root)
        if not summaries:
            self.app.add_system_message("No saved conversations found")
            return
        if args.strip():
            self._load_callback(args.strip())
            return
        lines = format_conversation_list(summaries)
        self.app.add_system_message('\n'.join(lines))
        self.app.prompt_message = "Enter conversation index:"
        self.app.prompt_for_input(self._load_callback)

    def _load_callback(self, index_or_name: str):
        summaries = list_conversations(self.app.state.conversations_root)
        target = None
        display = None

        # Try parsing as index first
        try:
            index = int(index_or_name)
            if 0 <= index < len(summaries):
                target = summaries[index].dir_name
                display = summaries[index].display_name
        except ValueError:
            # Try matching by name (display_name or dir_name)
            name = index_or_name.strip()
            matches = [
                s for s in summaries
                if s.display_name == name or s.dir_name == name
            ]
            if len(matches) == 1:
                target = matches[0].dir_name
                display = matches[0].display_name
            elif len(matches) > 1:
                self.app.add_system_message(
                    f"Multiple conversations match '{name}'. Use index or full name."
                )
                return

        if target is None:
            self.app.add_system_message(f"Conversation not found: {index_or_name}")
            return

        conversation = load_conversation(self.app.state.conversations_root, target)
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

    def handle_chats(self):
        summaries = list_conversations(self.app.state.conversations_root)
        if not summaries:
            self.app.add_system_message("No saved conversations found")
            return
        lines = format_conversation_list(summaries)
        self.app.add_system_message('\n'.join(lines))

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
