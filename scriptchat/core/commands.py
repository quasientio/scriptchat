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

"""Command parsing and handling for ScriptChat."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import re

from .config import Config, reasoning_levels_for_model, reasoning_default_for_model
from .conversations import Conversation, Message, save_conversation
from .ollama_client import OllamaChatClient
from .provider_dispatcher import ProviderDispatcher


# Command registry with metadata for help system
COMMAND_REGISTRY = {
    # Conversation management
    "new": {
        "category": "Conversation",
        "usage": "/new",
        "description": "Start a new conversation, preserving current model and settings.",
        "examples": ["/new"],
    },
    "save": {
        "category": "Conversation",
        "usage": "/save [name]",
        "description": "Save the current conversation. Prompts for name if not provided.",
        "examples": ["/save", "/save my-chat"],
    },
    "load": {
        "category": "Conversation",
        "usage": "/load [--archived] [index|name]",
        "description": "Load a saved conversation by index or name. Use --archived to load from archive.",
        "examples": ["/load", "/load 0", "/load my-chat", "/load --archived 0"],
    },
    "branch": {
        "category": "Conversation",
        "usage": "/branch [name]",
        "description": "Create a branch (copy) of the current conversation.",
        "examples": ["/branch", "/branch experiment"],
    },
    "rename": {
        "category": "Conversation",
        "usage": "/rename [new-name]",
        "description": "Rename the current saved conversation.",
        "examples": ["/rename better-name"],
    },
    "chats": {
        "category": "Conversation",
        "usage": "/chats [--archived|--all]",
        "description": "List saved conversations. Use --archived for archived only, --all for both.",
        "examples": ["/chats", "/chats --archived", "/chats --all"],
    },
    "clear": {
        "category": "Conversation",
        "usage": "/clear [index]",
        "description": "Clear current conversation or delete a saved one by index.",
        "examples": ["/clear", "/clear 2"],
    },
    "archive": {
        "category": "Conversation",
        "usage": "/archive [index|name|range] [--tag key=value]",
        "description": "Archive conversations. Supports index (3), name, range (1-5), or --tag filter.",
        "examples": ["/archive 0", "/archive 1-5", "/archive --tag imported_from=chatgpt"],
    },
    "unarchive": {
        "category": "Conversation",
        "usage": "/unarchive [index|name|range] [--tag key=value]",
        "description": "Restore archived conversations. Same syntax as /archive.",
        "examples": ["/unarchive 0", "/unarchive 1-3", "/unarchive --tag imported_from=chatgpt"],
    },
    # Export/Import
    "export": {
        "category": "Export/Import",
        "usage": "/export [format]",
        "description": "Export current conversation. Formats: md, json, html.",
        "examples": ["/export", "/export md", "/export json", "/export html"],
    },
    "export-all": {
        "category": "Export/Import",
        "usage": "/export-all [format]",
        "description": "Export all saved conversations in the given format.",
        "examples": ["/export-all md", "/export-all json", "/export-all html"],
    },
    "import": {
        "category": "Export/Import",
        "usage": "/import <path>",
        "description": "Import a conversation from a JSON or Markdown file.",
        "examples": ["/import ~/backup/chat.json"],
    },
    "import-chatgpt": {
        "category": "Export/Import",
        "usage": "/import-chatgpt [--dry-run] <zip-path>",
        "description": "Import conversations from a ChatGPT export ZIP file. Use --dry-run to preview without saving.",
        "examples": ["/import-chatgpt ~/downloads/chatgpt-export.zip", "/import-chatgpt --dry-run ~/downloads/export.zip"],
    },
    # Model & Settings
    "model": {
        "category": "Model & Settings",
        "usage": "/model [index|provider/model]",
        "description": "Switch to a different model or list available models.",
        "examples": ["/model", "/model 2", "/model ollama/llama3"],
    },
    "temp": {
        "category": "Model & Settings",
        "usage": "/temp [value]",
        "description": "Set temperature (0.0-2.0). Lower = more deterministic.",
        "examples": ["/temp", "/temp 0.7", "/temp 1.5"],
    },
    "reason": {
        "category": "Model & Settings",
        "usage": "/reason [level]",
        "description": "Set reasoning level: low, medium, high, max (presets for thinking budget).",
        "examples": ["/reason", "/reason low", "/reason high", "/reason max"],
    },
    "thinking": {
        "category": "Model & Settings",
        "usage": "/thinking [tokens|off]",
        "description": "Set exact thinking budget in tokens (Anthropic Claude only). 1024-128000, or off to disable.",
        "examples": ["/thinking", "/thinking 8000", "/thinking 32000", "/thinking off"],
    },
    "timeout": {
        "category": "Model & Settings",
        "usage": "/timeout <seconds|0|off>",
        "description": "Set the request timeout in seconds, or disable with 0/off.",
        "examples": ["/timeout 60", "/timeout 120", "/timeout off"],
    },
    "stream": {
        "category": "Model & Settings",
        "usage": "/stream [on|off]",
        "description": "Toggle streaming mode for responses.",
        "examples": ["/stream", "/stream on", "/stream off"],
    },
    "prompt": {
        "category": "Model & Settings",
        "usage": "/prompt [text|clear]",
        "description": "Set or clear the system prompt.",
        "examples": ["/prompt Be concise", "/prompt clear"],
    },
    # Files
    "file": {
        "category": "Files",
        "usage": "/file [--force] <path>",
        "description": "Register a file for @reference in messages. Use --force for large files.",
        "examples": ["/file src/main.py", "/file --force large.txt"],
    },
    "files": {
        "category": "Files",
        "usage": "/files [--long]",
        "description": "List registered files. Use --long for details.",
        "examples": ["/files", "/files --long"],
    },
    # Tags
    "tag": {
        "category": "Tags",
        "usage": "/tag <key>=<value>",
        "description": "Add a tag to the current conversation.",
        "examples": ["/tag project=demo", "/tag status=done"],
    },
    "untag": {
        "category": "Tags",
        "usage": "/untag <key>",
        "description": "Remove a tag from the current conversation.",
        "examples": ["/untag project"],
    },
    "tags": {
        "category": "Tags",
        "usage": "/tags",
        "description": "Show all tags on the current conversation.",
        "examples": ["/tags"],
    },
    # Messaging
    "send": {
        "category": "Messaging",
        "usage": "/send <message>",
        "description": "Send a message (useful in scripts).",
        "examples": ["/send Hello, how are you?"],
    },
    "history": {
        "category": "Messaging",
        "usage": "/history [n|all]",
        "description": "Show recent user messages in the current conversation (persists if saved/loaded). Default: last 10.",
        "examples": ["/history", "/history 5", "/history all"],
    },
    "undo": {
        "category": "Messaging",
        "usage": "/undo [n]",
        "description": "Remove the last n user/assistant exchanges (default 1).",
        "examples": ["/undo", "/undo 2"],
    },
    "retry": {
        "category": "Messaging",
        "usage": "/retry",
        "description": "Remove last response and re-send the user message.",
        "examples": ["/retry"],
    },
    # Testing & Debug
    "assert": {
        "category": "Testing & Debug",
        "usage": "/assert <pattern>",
        "description": "Assert last response contains pattern (regex or substring).",
        "examples": ["/assert hello", "/assert \\d{4}"],
    },
    "assert-not": {
        "category": "Testing & Debug",
        "usage": "/assert-not <pattern>",
        "description": "Assert last response does NOT contain pattern.",
        "examples": ["/assert-not error", "/assert-not fail"],
    },
    "echo": {
        "category": "Testing & Debug",
        "usage": "/echo <message>",
        "description": "Display a message without sending to the model (not saved).",
        "examples": ["/echo Step 1 complete"],
    },
    "note": {
        "category": "Messaging",
        "usage": "/note <text>",
        "description": "Add a note to the conversation (saved, visible, but not sent to model).",
        "examples": ["/note Remember to ask about error handling", "/note TODO: revisit this"],
    },
    "log-level": {
        "category": "Testing & Debug",
        "usage": "/log-level <level>",
        "description": "Set log level: debug, info, warn, error, critical.",
        "examples": ["/log-level debug", "/log-level info", "/log-level warn"],
    },
    "profile": {
        "category": "Testing & Debug",
        "usage": "/profile [--full]",
        "description": "Show current session profile (model, tokens, settings). Use --full to show complete system prompt.",
        "examples": ["/profile", "/profile --full"],
    },
    # Scripting
    "run": {
        "category": "Scripting",
        "usage": "/run <path>",
        "description": "Run commands from a script file.",
        "examples": ["/run tests/scenario.txt"],
    },
    "sleep": {
        "category": "Scripting",
        "usage": "/sleep <seconds>",
        "description": "Pause execution for the specified duration (scripts/batch mode only).",
        "examples": ["/sleep 1", "/sleep 0.5", "/sleep 10"],
    },
    "set": {
        "category": "Scripting",
        "usage": "/set <name>=<value>",
        "description": "Set a variable for use in scripts. Use ${name} to reference.",
        "examples": ["/set model=llama3", "/set prompt=Be concise", "/send ${prompt}"],
    },
    "vars": {
        "category": "Scripting",
        "usage": "/vars",
        "description": "List all defined variables.",
        "examples": ["/vars"],
    },
    # System
    "help": {
        "category": "System",
        "usage": "/help [command|search]",
        "description": "Show help. Search commands by name or keyword.",
        "examples": ["/help", "/help export", "/help save"],
    },
    "keys": {
        "category": "System",
        "usage": "/keys",
        "description": "Show keyboard shortcuts.",
        "examples": ["/keys"],
    },
    "exit": {
        "category": "System",
        "usage": "/exit",
        "description": "Exit ScriptChat.",
        "examples": ["/exit"],
    },
}

# Category display order
CATEGORY_ORDER = [
    "Conversation",
    "Export/Import",
    "Model & Settings",
    "Files",
    "Tags",
    "Messaging",
    "Testing & Debug",
    "Scripting",
    "System",
]


def format_help_all() -> str:
    """Format help for all commands, organized by category."""
    lines = ["", "Commands:"]

    # Group by category
    by_category = {}
    for cmd, info in COMMAND_REGISTRY.items():
        cat = info["category"]
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append((cmd, info))

    # Display in order
    for cat in CATEGORY_ORDER:
        if cat not in by_category:
            continue
        lines.append(f"\n  {cat}:")
        for cmd, info in sorted(by_category[cat], key=lambda x: x[0]):
            lines.append(f"    {info['usage']:<24} {info['description']}")

    lines.append("\nUse /help <command> for details, or /help <keyword> to search.")
    return "\n".join(lines)


def format_help_command(cmd: str) -> str:
    """Format detailed help for a specific command."""
    info = COMMAND_REGISTRY.get(cmd)
    if not info:
        return None

    lines = [
        "",
        f"  {info['usage']}",
        "",
        f"  {info['description']}",
        "",
        "  Examples:",
    ]
    for ex in info["examples"]:
        lines.append(f"    {ex}")

    return "\n".join(lines)


def search_commands(query: str) -> str:
    """Search commands by name or description."""
    query_lower = query.lower()
    matches = []

    for cmd, info in COMMAND_REGISTRY.items():
        # Match command name or description
        if query_lower in cmd or query_lower in info["description"].lower():
            matches.append((cmd, info))

    if not matches:
        return f"No commands found matching '{query}'."

    lines = [f"", f"Commands matching '{query}':"]
    for cmd, info in sorted(matches, key=lambda x: x[0]):
        lines.append(f"  {info['usage']:<24} {info['description']}")

    return "\n".join(lines)


def format_keyboard_shortcuts() -> str:
    """Format keyboard shortcuts for display."""
    return """
Keyboard Shortcuts:

  Navigation:
    Ctrl+Up          Focus conversation pane for scrolling
    Ctrl+Home        Jump to start of conversation
    Ctrl+End         Jump to end of conversation
    Escape           Return focus to input pane
    Tab              Return to input (when in conversation pane)

  In conversation pane:
    Up/Down          Scroll line by line

  In input pane:
    Up/Down          Navigate command history
    Tab              Command completion
    Shift+Tab        Reverse completion cycling

  General:
    Ctrl+C, Ctrl+D   Exit ScriptChat
    Escape (2x)      Cancel ongoing inference (within 2 seconds)
"""


@dataclass
class AppState:
    """Global application state."""
    config: Config
    current_conversation: Conversation
    client: object
    conversations_root: Path
    file_registry: dict = field(default_factory=dict)  # key -> {"content": str, "full_path": str}
    variables: dict = field(default_factory=dict)  # script variables: name -> value


@dataclass
class CommandResult:
    """Result of a command execution."""
    message: Optional[str] = None
    should_exit: bool = False
    needs_ui_interaction: bool = False
    command_type: Optional[str] = None  # For complex commands that need UI handling
    file_content: Optional[str] = None  # For /file command - content to send as user message
    assert_passed: Optional[bool] = None  # For /assert results
    resend_message: Optional[str] = None  # For /retry to re-send a user message


def create_new_conversation(state: AppState) -> Conversation:
    """Create a new conversation with current settings.

    Args:
        state: Application state

    Returns:
        New Conversation object
    """
    messages = []

    # Add system prompt if configured
    if state.config.system_prompt:
        messages.append(Message(
            role='system',
            content=state.config.system_prompt
        ))

    return Conversation(
        id=None,
        provider_id=state.current_conversation.provider_id if state.current_conversation else state.config.default_provider,
        model_name=state.current_conversation.model_name,
        temperature=state.current_conversation.temperature,
        reasoning_level=getattr(state.current_conversation, "reasoning_level", None) if state.current_conversation else None,
        messages=messages,
        system_prompt=state.current_conversation.system_prompt or state.config.system_prompt,
        tokens_in=0,
        tokens_out=0,
        tags=state.current_conversation.tags.copy() if state.current_conversation else {},
        file_references={}
    )


def assert_last_response(convo: Conversation, pattern: str, negate: bool = False) -> tuple[bool, str]:
    """Check whether the last assistant message matches (or does not match) the pattern."""
    last_assistant = None
    for msg in reversed(convo.messages):
        if msg.role == 'assistant':
            last_assistant = msg.content
            break

    if last_assistant is None:
        return False, "No assistant response available to assert against."

    try:
        matched = re.search(pattern, last_assistant, flags=re.IGNORECASE) is not None
    except re.error:
        # Fallback to simple substring if regex is invalid
        matched = pattern.lower() in last_assistant.lower()

    if negate:
        if matched:
            return False, f"Assertion FAILED: pattern '{pattern}' should NOT appear in last response."
        return True, f"Assertion PASSED: pattern '{pattern}' not found."
    else:
        if matched:
            return True, f"Assertion PASSED: found pattern '{pattern}'."
        return False, f"Assertion FAILED: pattern '{pattern}' not found in last response."


def handle_command(line: str, state: AppState) -> CommandResult:
    """Parse and handle a command.

    Args:
        line: Command line (starting with /)
        state: Application state

    Returns:
        CommandResult with execution result
    """
    line = line.strip()

    # Expand variables (${name}) before processing
    line = expand_variables(
        line,
        state.variables,
        env_expand=state.config.env_expand_from_environment,
        env_blocklist=state.config.env_var_blocklist,
    )

    if not line.startswith('/'):
        return CommandResult(message="Commands must start with /")

    # Split command and arguments
    parts = line[1:].split(maxsplit=1)
    if not parts:
        return CommandResult(message="Empty command")

    command = parts[0].lower()

    # Handle simple commands
    if command == 'exit':
        return CommandResult(
            message="Exiting ScriptChat...",
            should_exit=True
        )

    elif command == 'help':
        arg = parts[1].strip() if len(parts) > 1 else ""
        if not arg:
            return CommandResult(message=format_help_all())
        # Check for exact command match first
        cmd_help = format_help_command(arg.lstrip('/'))
        if cmd_help:
            return CommandResult(message=cmd_help)
        # Otherwise search
        return CommandResult(message=search_commands(arg))

    elif command == 'keys':
        return CommandResult(message=format_keyboard_shortcuts())

    elif command == 'new':
        # Create new conversation
        state.current_conversation = create_new_conversation(state)
        return CommandResult(message="Started new conversation")

    # Commands that need UI interaction
    elif command == 'model':
        return CommandResult(
            needs_ui_interaction=True,
            command_type='model'
        )

    elif command == 'save':
        return CommandResult(
            needs_ui_interaction=True,
            command_type='save'
        )

    elif command == 'load':
        return CommandResult(
            needs_ui_interaction=True,
            command_type='load'
        )

    elif command == 'branch':
        return CommandResult(
            needs_ui_interaction=True,
            command_type='branch'
        )

    elif command == 'rename':
        return CommandResult(
            needs_ui_interaction=True,
            command_type='rename'
        )

    elif command == 'chats':
        return CommandResult(
            needs_ui_interaction=True,
            command_type='chats'
        )

    elif command == 'archive':
        return CommandResult(
            needs_ui_interaction=True,
            command_type='archive'
        )

    elif command == 'unarchive':
        return CommandResult(
            needs_ui_interaction=True,
            command_type='unarchive'
        )

    elif command == 'send':
        if len(parts) < 2 or not parts[1].strip():
            return CommandResult(message="Usage: /send <message>")
        return CommandResult(
            needs_ui_interaction=True,
            command_type='send'
        )

    elif command == 'history':
        messages = state.current_conversation.messages
        # Show only user messages (like shell history shows commands, not output)
        user_msgs = [m for m in messages if m.role == 'user']

        if not user_msgs:
            return CommandResult(message="No messages in history.")

        # Parse count argument
        arg = parts[1].strip().lower() if len(parts) > 1 else ""
        if arg == "all" or arg == "":
            count = len(user_msgs) if arg == "all" else min(10, len(user_msgs))
        else:
            try:
                count = int(arg)
                if count <= 0:
                    return CommandResult(message="Count must be positive.")
            except ValueError:
                return CommandResult(message="Usage: /history [n|all]")

        # Get last n messages
        recent = user_msgs[-count:] if count < len(user_msgs) else user_msgs
        lines = []
        for i, msg in enumerate(recent):
            # Truncate long messages
            content = msg.content
            if len(content) > 200:
                content = content[:200] + "..."
            lines.append(content)

        header = f"Last {len(recent)} of {len(user_msgs)} messages:"
        return CommandResult(message=header + "\n" + "\n".join(lines))

    elif command == 'export':
        return CommandResult(
            needs_ui_interaction=True,
            command_type='export'
        )

    elif command == 'export-all':
        return CommandResult(
            needs_ui_interaction=True,
            command_type='export-all'
        )

    elif command == 'stream':
        return CommandResult(
            needs_ui_interaction=True,
            command_type='stream'
        )

    elif command == 'prompt':
        return CommandResult(
            needs_ui_interaction=True,
            command_type='prompt'
        )

    elif command == 'run':
        if len(parts) < 2 or not parts[1].strip():
            return CommandResult(message="Usage: /run <path>")
        return CommandResult(
            needs_ui_interaction=True,
            command_type='run'
        )

    elif command == 'sleep':
        # Sleep is only useful in batch/script mode - direct users there
        return CommandResult(message="/sleep is only available in scripts (/run) or batch mode.")

    elif command == 'set':
        if len(parts) < 2 or '=' not in parts[1]:
            return CommandResult(message="Usage: /set <name>=<value>")
        name, value = parts[1].split('=', 1)
        name = name.strip()
        # Validate variable name
        import re
        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', name):
            return CommandResult(message="Invalid variable name. Use letters, numbers, and underscores (cannot start with number).")
        state.variables[name] = value
        return CommandResult(message=f"Set ${{{name}}} = {value}")

    elif command == 'vars':
        if not state.variables:
            return CommandResult(message="No variables defined. Use /set name=value to define.")
        lines = [f"${{{k}}} = {v}" for k, v in sorted(state.variables.items())]
        return CommandResult(message="Variables:\n" + "\n".join(lines))

    elif command == 'temp':
        return CommandResult(
            needs_ui_interaction=True,
            command_type='temp'
        )

    elif command == 'reason':
        level_arg = parts[1].strip() if len(parts) > 1 else ""
        return set_reasoning_level(state, level_arg)

    elif command == 'thinking':
        budget_arg = parts[1].strip() if len(parts) > 1 else ""
        return set_thinking_budget(state, budget_arg)

    elif command == 'import':
        if len(parts) < 2 or not parts[1].strip():
            return CommandResult(message="Usage: /import <path>")
        return CommandResult(
            needs_ui_interaction=True,
            command_type='import'
        )

    elif command == 'import-chatgpt':
        if len(parts) < 2 or not parts[1].strip():
            return CommandResult(message="Usage: /import-chatgpt [--dry-run] <zip-path>")

        # Parse args
        args = parts[1].strip().split()
        dry_run = False
        zip_arg = None
        for arg in args:
            if arg == '--dry-run':
                dry_run = True
            elif not zip_arg:
                zip_arg = arg

        if not zip_arg:
            return CommandResult(message="Usage: /import-chatgpt [--dry-run] <zip-path>")

        zip_path = Path(zip_arg).expanduser()
        if not zip_path.exists():
            return CommandResult(message=f"File not found: {zip_path}")
        if not zip_path.suffix.lower() == '.zip':
            return CommandResult(message="File must be a ZIP file")
        try:
            from .conversations import import_chatgpt_export, parse_chatgpt_export

            if dry_run:
                # Preview mode - parse without saving
                parsed = parse_chatgpt_export(zip_path, state.current_conversation.provider_id)
                if not parsed:
                    return CommandResult(message="No conversations found in export")
                lines = [f"Dry run: {len(parsed)} conversation(s) would be imported:"]
                for convo, title, timestamp in parsed:
                    msg_count = len(convo.messages)
                    date_str = timestamp.strftime("%Y-%m-%d") if timestamp else "unknown"
                    lines.append(f"  - {title} ({convo.model_name}, {msg_count} messages, {date_str})")
                return CommandResult(message="\n".join(lines))
            else:
                imported = import_chatgpt_export(
                    zip_path,
                    state.conversations_root,
                    default_provider_id=state.current_conversation.provider_id
                )
                if not imported:
                    return CommandResult(message="No conversations found in export")
                titles = [c.id.split('_', 2)[-1] if c.id else 'untitled' for c in imported]
                return CommandResult(message=f"Imported {len(imported)} conversation(s):\n" + "\n".join(f"  - {t}" for t in titles))
        except ValueError as e:
            return CommandResult(message=f"Import error: {e}")
        except Exception as e:
            return CommandResult(message=f"Import failed: {e}")

    elif command == 'timeout':
        if len(parts) < 2 or not parts[1].strip():
            current = state.config.timeout
            if current is None:
                return CommandResult(message="Current timeout: disabled\nUsage: /timeout <seconds|0|off>")
            return CommandResult(message=f"Current timeout: {current} seconds\nUsage: /timeout <seconds|0|off>")
        arg = parts[1].strip().lower()
        if arg in ("off", "0"):
            return set_timeout(state, None)
        try:
            seconds = float(arg)
        except ValueError:
            return CommandResult(message="Invalid timeout. Usage: /timeout <seconds|0|off>")
        return set_timeout(state, seconds)

    elif command == 'clear':
        return CommandResult(
            needs_ui_interaction=True,
            command_type='clear'
        )

    elif command == 'file':
        if len(parts) < 2:
            return CommandResult(message="Usage: /file [--force] <path>")

        arg_str = parts[1].strip()
        force = False
        file_path = arg_str
        if arg_str.startswith("--force "):
            force = True
            file_path = arg_str[len("--force "):].strip()
        elif arg_str == "--force":
            return CommandResult(message="Usage: /file [--force] <path>")

        expanded_path = Path(file_path).expanduser()
        try:
            content = expanded_path.read_text(encoding='utf-8')
            full_path_str = str(expanded_path.resolve())
        except FileNotFoundError:
            return CommandResult(message=f"File not found: {file_path}")
        except PermissionError:
            return CommandResult(message=f"Permission denied: {file_path}")
        except UnicodeDecodeError:
            return CommandResult(message=f"Cannot read file (not UTF-8 text): {file_path}")
        except Exception as e:
            return CommandResult(message=f"Error reading file: {str(e)}")

        size = len(content.encode("utf-8"))
        threshold = getattr(state.config, "file_confirm_threshold_bytes", 40_000)
        if not force and size > threshold:
            return CommandResult(
                message=f"File is {size} bytes; exceeds {threshold}. Re-run with /file --force {file_path} to register."
            )

        # Compute hash for the file content
        import hashlib
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

        # Register by original path token
        state.file_registry[file_path] = {"content": content, "full_path": full_path_str}
        # Also register by basename if not already present
        basename = expanded_path.name
        alt = ""
        if basename not in state.file_registry:
            state.file_registry[basename] = {"content": content, "full_path": full_path_str}
            alt = f" and @{basename}"

        # Persist references on the conversation for future loads (with hash)
        if not hasattr(state.current_conversation, "file_references") or state.current_conversation.file_references is None:
            state.current_conversation.file_references = {}
        file_ref_entry = {"path": full_path_str, "sha256": content_hash}
        state.current_conversation.file_references[file_path] = file_ref_entry
        # Update basename entry if it points to the same file, or create if not present
        existing_basename_ref = state.current_conversation.file_references.get(basename)
        if existing_basename_ref is None:
            state.current_conversation.file_references[basename] = file_ref_entry
        elif isinstance(existing_basename_ref, dict) and existing_basename_ref.get("path") == full_path_str:
            # Same file, update the hash
            state.current_conversation.file_references[basename] = file_ref_entry
        elif isinstance(existing_basename_ref, str) and existing_basename_ref == full_path_str:
            # Old format, same file, update to new format with hash
            state.current_conversation.file_references[basename] = file_ref_entry
        # Auto-save meta if conversation already saved
        try:
            if state.current_conversation.id:
                save_conversation(
                    state.conversations_root,
                    state.current_conversation,
                    system_prompt=state.current_conversation.system_prompt
                )
        except Exception:
            pass

        return CommandResult(message=f"Registered @{file_path}{alt} ({len(content)} chars)")

    elif command == 'echo':
        # Echo command - print message without sending to LLM
        message = parts[1] if len(parts) > 1 else ""
        return CommandResult(message=message)

    elif command == 'note':
        # Note command - add a note to conversation (saved, visible, not sent to model)
        if len(parts) < 2 or not parts[1].strip():
            return CommandResult(message="Usage: /note <text>")
        note_text = parts[1].strip()
        state.current_conversation.messages.append(Message(role='note', content=note_text))
        # Auto-save if conversation has an ID
        if state.current_conversation.id:
            save_conversation(
                state.conversations_root,
                state.current_conversation,
                system_prompt=state.config.system_prompt
            )
        return CommandResult(message=f"[Note] {note_text}")

    elif command == 'tag':
        if len(parts) < 2 or '=' not in parts[1]:
            return CommandResult(message="Usage: /tag key=value")
        key, value = parts[1].split('=', 1)
        key = key.strip()
        value = value.strip()
        if not key:
            return CommandResult(message="Usage: /tag key=value")
        if state.current_conversation.tags is None:
            state.current_conversation.tags = {}
        state.current_conversation.tags[key] = value
        # Auto-save if conversation has an id
        saved_note = ""
        try:
            if state.current_conversation.id:
                save_conversation(
                    state.conversations_root,
                    state.current_conversation,
                    system_prompt=state.current_conversation.system_prompt
                )
                saved_note = " (saved)"
        except Exception:
            saved_note = " (not saved)"
        return CommandResult(message=f"Tag set: {key}={value}{saved_note}")

    elif command == 'tags':
        tags = state.current_conversation.tags or {}
        if not tags:
            return CommandResult(message="No tags set.")
        items = [f"{k}={v}" for k, v in sorted(tags.items())]
        return CommandResult(message="Tags: " + ", ".join(items))

    elif command == 'untag':
        if len(parts) < 2 or not parts[1].strip():
            return CommandResult(message="Usage: /untag <key>")
        key = parts[1].strip()
        if not state.current_conversation.tags or key not in state.current_conversation.tags:
            return CommandResult(message=f"Tag not found: {key}")
        try:
            del state.current_conversation.tags[key]
        except Exception:
            state.current_conversation.tags = {k: v for k, v in state.current_conversation.tags.items() if k != key}

        saved_note = ""
        try:
            if state.current_conversation.id:
                save_conversation(
                    state.conversations_root,
                    state.current_conversation,
                    system_prompt=state.current_conversation.system_prompt
                )
                saved_note = " (saved)"
        except Exception:
            saved_note = " (not saved)"
        return CommandResult(message=f"Tag removed: {key}{saved_note}")

    elif command == 'undo':
        count = 1
        if len(parts) > 1 and parts[1].strip():
            try:
                count = max(1, int(parts[1].strip()))
            except ValueError:
                return CommandResult(message="Usage: /undo [n]")
        removed = undo_last_exchanges(state.current_conversation, count)
        if removed == 0:
            return CommandResult(message="Nothing to undo.")
        plural = "exchange" if removed == 1 else "exchanges"
        return CommandResult(message=f"Removed {removed} {plural}.")

    elif command == 'retry':
        msg_to_resend, info = retry_last_user_message(state.current_conversation)
        if msg_to_resend is None:
            return CommandResult(message=info)
        return CommandResult(message=info, resend_message=msg_to_resend)

    elif command == 'log-level':
        if len(parts) < 2 or not parts[1].strip():
            return CommandResult(message="Usage: /log-level <debug|info|warn>")
        level = parts[1].strip().upper()
        if level == "WARN":
            level = "WARNING"
        if level not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            return CommandResult(message="Usage: /log-level <debug|info|warn|error|critical>")
        import logging
        root_logger = logging.getLogger()
        root_logger.setLevel(level)
        # Also update handler levels so messages actually get output
        for handler in root_logger.handlers:
            handler.setLevel(level)
        state.config.log_level = level
        return CommandResult(message=f"Log level set to {level}")

    elif command == 'files':
        long = False
        if len(parts) > 1 and parts[1].strip():
            long = parts[1].strip() in ("--long", "-l")
        if not state.file_registry:
            return CommandResult(message="No files registered.")
        lines = ["Registered files:"]
        for key, entry in state.file_registry.items():
            content = entry["content"] if isinstance(entry, dict) else ""
            full_path = entry["full_path"] if isinstance(entry, dict) else str(key)
            size = len(content)
            if long:
                import hashlib
                h = hashlib.sha256(content.encode("utf-8")).hexdigest() if content else ""
                lines.append(f"@{key} -> {full_path} ({size} bytes) sha256:{h}")
            else:
                lines.append(f"@{key} -> {full_path}")
        return CommandResult(message="\n".join(lines))

    elif command == 'profile':
        full = False
        if len(parts) > 1 and parts[1].strip():
            full = parts[1].strip() in ("--full", "-f")
        convo = state.current_conversation
        cfg = state.config
        conv_count = 0
        exp_count = 0
        try:
            if state.conversations_root.exists():
                conv_count = sum(1 for p in state.conversations_root.iterdir() if p.is_dir())
        except Exception:
            conv_count = 0
        exports_dir = cfg.exports_dir or Path.cwd()
        try:
            if exports_dir.exists():
                exp_count = sum(1 for p in exports_dir.iterdir() if p.is_file())
        except Exception:
            exp_count = 0

        file_keys = list(state.file_registry.keys()) if hasattr(state, "file_registry") else []
        available_reasoning = reasoning_levels_for_model(cfg, convo.provider_id, convo.model_name)
        if convo.thinking_budget:
            reasoning_display = f"{convo.thinking_budget} tokens (custom)"
        elif convo.reasoning_level:
            budget = THINKING_BUDGET_PRESETS.get(convo.reasoning_level)
            if budget:
                reasoning_display = f"{convo.reasoning_level} ({budget} tokens)"
            else:
                reasoning_display = convo.reasoning_level
        elif available_reasoning:
            reasoning_display = "(off)"
        else:
            reasoning_display = "(unavailable)"

        # Get context length from model config
        try:
            model_cfg = cfg.get_model(convo.provider_id, convo.model_name)
            if model_cfg.contexts:
                context_display = ", ".join(str(c) for c in model_cfg.contexts)
            else:
                context_display = "(not configured)"
        except ValueError:
            context_display = "(not configured)"

        lines = [
            "",  # Start on new line after [system] prefix
            f"Provider: {convo.provider_id}",
            f"Model: {convo.model_name}",
            f"Context: {context_display}",
            f"Reasoning: {reasoning_display}",
            f"Temperature: {convo.temperature:.2f}",
            f"Tokens: {convo.tokens_in} in / {convo.tokens_out} out",
            f"Streaming: {'on' if cfg.enable_streaming else 'off'}",
            f"Timeout: {cfg.timeout}s",
            f"Log level: {cfg.log_level}",
            f"Conversations dir: {cfg.conversations_dir} ({conv_count} convs)",
        ]
        if cfg.exports_dir:
            lines.append(f"Exports dir: {cfg.exports_dir} ({exp_count} files)")
        if convo.system_prompt:
            if full:
                lines.append(f"System prompt:\n{convo.system_prompt}")
            else:
                trimmed = convo.system_prompt.strip().replace("\n", " ")
                if len(trimmed) > 100:
                    trimmed = trimmed[:97] + "..."
                lines.append(f"System prompt: {trimmed}")
        else:
            lines.append("System prompt: (none)")
        if file_keys:
            # Show unique full paths
            full_paths = []
            seen = set()
            for entry in state.file_registry.values():
                full_path = entry["full_path"] if isinstance(entry, dict) else str(entry)
                if full_path not in seen:
                    seen.add(full_path)
                    full_paths.append(full_path)
            lines.append("Files: " + ", ".join(full_paths))

        # Show branch info
        if convo.parent_id:
            lines.append(f"Parent: {convo.parent_id}")
        # Count children (branches of this conversation)
        if convo.id:
            child_count = 0
            try:
                for d in state.conversations_root.iterdir():
                    if d.is_dir():
                        meta_path = d / "meta.json"
                        if meta_path.exists():
                            import json
                            meta = json.loads(meta_path.read_text())
                            if meta.get("parent_id") == convo.id:
                                child_count += 1
            except Exception:
                pass
            if child_count > 0:
                lines.append(f"Branches: {child_count}")

        return CommandResult(message="\n".join(lines))

    elif command == 'assert':
        if len(parts) < 2 or not parts[1].strip():
            return CommandResult(message="Usage: /assert <pattern>")
        passed, msg = assert_last_response(state.current_conversation, parts[1].strip())
        return CommandResult(message=msg, assert_passed=passed)

    elif command == 'assert-not':
        if len(parts) < 2 or not parts[1].strip():
            return CommandResult(message="Usage: /assert-not <pattern>")
        passed, msg = assert_last_response(state.current_conversation, parts[1].strip(), negate=True)
        return CommandResult(message=msg, assert_passed=passed)

    else:
        return CommandResult(
            message=f"Unknown command: /{command}. Type /help for available commands."
        )


def set_model(state: AppState, model_name: str) -> CommandResult:
    """Set the current model.

    Args:
        state: Application state
        model_name: Name of model to switch to

    Returns:
        CommandResult with execution result
    """
    try:
        provider = state.config.get_provider(state.current_conversation.provider_id)
    except ValueError as e:
        return CommandResult(message=str(e))

    model_name = model_name.strip()
    available_models = [m.name for m in (provider.models or [])]
    if provider.default_model and provider.default_model not in available_models:
        available_models.append(provider.default_model)

    def _apply_reasoning_default():
        default_reason = reasoning_default_for_model(state.config, state.current_conversation.provider_id, state.current_conversation.model_name)
        state.current_conversation.reasoning_level = default_reason

    if available_models and model_name not in available_models:
        # Try to find the model in other providers
        matches = []
        for p in state.config.providers:
            names = [m.name for m in (p.models or [])]
            if p.default_model and p.default_model not in names:
                names.append(p.default_model)
            if model_name in names:
                matches.append(p.id)
        if len(matches) == 1:
            # Switch provider and apply
            state.current_conversation.provider_id = matches[0]
            state.current_conversation.model_name = model_name
            state.current_conversation.tokens_in = 0
            state.current_conversation.tokens_out = 0
            _apply_reasoning_default()
            return CommandResult(message=f"Switched to model: {model_name} (provider: {matches[0]})")
        options = available_models if available_models else []
        return CommandResult(
            message=f"Model '{model_name}' not found for provider '{provider.id}'. Options: {options}"
        )

    state.current_conversation.model_name = model_name
    state.current_conversation.tokens_in = 0
    state.current_conversation.tokens_out = 0
    _apply_reasoning_default()
    return CommandResult(message=f"Switched to model: {model_name}")


def set_temperature(state: AppState, temperature: float) -> CommandResult:
    """Set the conversation temperature.

    Args:
        state: Application state
        temperature: Temperature value (0.0-2.0)

    Returns:
        CommandResult with execution result
    """
    # Clamp to valid range
    temperature = max(0.0, min(2.0, temperature))
    state.current_conversation.temperature = temperature
    return CommandResult(message=f"Temperature set to {temperature:.2f}")


def set_reasoning_level(state: AppState, level: str) -> CommandResult:
    """Set or show the reasoning level for the current model/provider."""
    convo = state.current_conversation
    available = reasoning_levels_for_model(state.config, convo.provider_id, convo.model_name)
    if not level.strip():
        if not available:
            return CommandResult(message="Reasoning controls not available for this model/provider.")
        current = convo.reasoning_level or "(default)"
        budget_info = ""
        if convo.thinking_budget:
            budget_info = f" (custom: {convo.thinking_budget} tokens)"
        return CommandResult(
            message=f"Reasoning level: {current}{budget_info}\nAvailable: {', '.join(available)}\nUsage: /reason <level>"
        )

    desired = level.strip().lower()
    if desired in ("default", "clear", "reset"):
        convo.reasoning_level = None
        convo.thinking_budget = None
        return CommandResult(message="Reasoning level cleared (provider default).")

    if not available:
        return CommandResult(message="Reasoning controls not available for this model/provider.")

    if desired not in available:
        return CommandResult(message=f"Unsupported reasoning level for {convo.model_name}. Available: {', '.join(available)}")

    convo.reasoning_level = desired
    convo.thinking_budget = None  # Clear custom budget when using preset
    return CommandResult(message=f"Reasoning level set to {desired}")


# Thinking budget presets (used by Anthropic client)
# Values chosen to fit within Claude Sonnet 4.5's 64000 max_tokens limit
THINKING_BUDGET_PRESETS = {
    "low": 4000,
    "medium": 16000,
    "high": 32000,
    "max": 55000,
}


def set_thinking_budget(state: AppState, budget: str) -> CommandResult:
    """Set or show the thinking budget for Anthropic Claude models."""
    convo = state.current_conversation

    if not budget.strip():
        # Show current setting
        current_budget = convo.thinking_budget
        current_level = convo.reasoning_level
        if current_budget:
            return CommandResult(message=f"Thinking budget: {current_budget} tokens\nUsage: /thinking <tokens> (1024-128000)")
        elif current_level and current_level in THINKING_BUDGET_PRESETS:
            preset_budget = THINKING_BUDGET_PRESETS[current_level]
            return CommandResult(message=f"Thinking budget: {preset_budget} tokens (from /reason {current_level})\nUsage: /thinking <tokens> (1024-128000)")
        else:
            return CommandResult(message="Thinking: disabled\nUsage: /thinking <tokens> (1024-128000)")

    budget_str = budget.strip().lower()

    # Handle clear/disable
    if budget_str in ("clear", "off", "disable", "0"):
        convo.thinking_budget = None
        convo.reasoning_level = None
        return CommandResult(message="Thinking disabled.")

    # Parse token count
    try:
        tokens = int(budget_str)
    except ValueError:
        return CommandResult(message="Invalid budget. Usage: /thinking <tokens> (1024-128000)")

    # Validate range
    if tokens < 1024:
        return CommandResult(message="Minimum thinking budget is 1024 tokens.")
    if tokens > 128000:
        return CommandResult(message="Maximum thinking budget is 128000 tokens.")

    convo.thinking_budget = tokens
    convo.reasoning_level = None  # Clear preset when using custom budget
    return CommandResult(message=f"Thinking budget set to {tokens} tokens")


def set_timeout(state: AppState, seconds: float | None) -> CommandResult:
    """Set the request timeout in seconds.

    Args:
        state: Application state
        seconds: Timeout duration in seconds, or None to disable

    Returns:
        CommandResult with execution result
    """
    if seconds is None:
        # Disable timeout
        state.config.timeout = None
        client = state.client
        if isinstance(client, ProviderDispatcher):
            for provider_client in client.clients.values():
                if hasattr(provider_client, "timeout"):
                    provider_client.timeout = None
                if hasattr(provider_client, "config"):
                    try:
                        provider_client.config.timeout = None
                    except Exception:
                        pass
        return CommandResult(message="Timeout disabled")

    try:
        timeout_val = float(seconds)
    except (TypeError, ValueError):
        return CommandResult(message="Invalid timeout. Please provide a number of seconds.")

    if timeout_val <= 0:
        return CommandResult(message="Timeout must be greater than zero seconds.")

    timeout_int = int(timeout_val)
    if timeout_int <= 0:
        timeout_int = 1

    state.config.timeout = timeout_int

    # Propagate to provider clients that cache the timeout
    client = state.client
    if isinstance(client, ProviderDispatcher):
        for provider_client in client.clients.values():
            if hasattr(provider_client, "timeout"):
                provider_client.timeout = timeout_int
            if hasattr(provider_client, "config"):
                try:
                    provider_client.config.timeout = timeout_int
                except Exception:
                    pass

    return CommandResult(message=f"Timeout set to {timeout_int} seconds")


def undo_last_exchanges(convo: Conversation, count: int) -> int:
    """Remove the last N user+assistant exchanges from the conversation.

    Args:
        convo: Conversation to mutate
        count: Number of exchanges to remove

    Returns:
        Number of exchanges actually removed
    """
    removed = 0
    for _ in range(count):
        # Find last assistant
        idx = None
        for i in range(len(convo.messages) - 1, -1, -1):
            if convo.messages[i].role == 'assistant':
                idx = i
                break
        if idx is None:
            break

        # Remove assistant
        convo.messages.pop(idx)

        # Find preceding user before idx (after removal idx is now position of next item)
        user_idx = None
        for j in range(idx - 1, -1, -1):
            if convo.messages[j].role == 'user':
                user_idx = j
                break
        if user_idx is not None:
            convo.messages.pop(user_idx)

        # Remove trailing non-chat roles (e.g., echo) that were after the assistant
        while convo.messages and convo.messages[-1].role not in ('user', 'assistant', 'system'):
            convo.messages.pop()

        removed += 1
    return removed


def retry_last_user_message(convo: Conversation) -> tuple[Optional[str], str]:
    """Drop the last assistant message and preceding user message, return user content to resend."""
    # Find last assistant index
    last_assistant = None
    for i in range(len(convo.messages) - 1, -1, -1):
        if convo.messages[i].role == 'assistant':
            last_assistant = i
            break
    if last_assistant is None:
        return None, "No assistant response to retry."

    # Find the user message before that assistant
    last_user_idx = None
    last_user_content = None
    for j in range(last_assistant - 1, -1, -1):
        if convo.messages[j].role == 'user':
            last_user_idx = j
            last_user_content = convo.messages[j].content
            break

    if last_user_content is None:
        return None, "No user message found to retry."

    # Remove assistant message first (higher index), then user message
    convo.messages.pop(last_assistant)
    convo.messages.pop(last_user_idx)
    return last_user_content, "Retrying last user message."


# Default patterns for sensitive env vars that should not be expanded
DEFAULT_ENV_VAR_BLOCKLIST = [
    "*_KEY",
    "*_SECRET",
    "*_TOKEN",
    "*_PASSWORD",
    "*_CREDENTIAL",
    "*_CREDENTIALS",
    "*_API_KEY",
]


def _matches_blocklist(var_name: str, blocklist: list[str]) -> bool:
    """Check if a variable name matches any blocklist pattern.

    Patterns support * as wildcard (e.g., *_KEY matches OPENAI_API_KEY).
    """
    import fnmatch
    var_upper = var_name.upper()
    for pattern in blocklist:
        if fnmatch.fnmatch(var_upper, pattern.upper()):
            return True
    return False


def expand_variables(
    text: str,
    variables: dict,
    env_expand: bool = True,
    env_blocklist: list[str] | None = None,
) -> str:
    """Expand ${name} variable references in text.

    Checks script variables first, then falls back to environment variables
    (if env_expand is True and the variable name doesn't match the blocklist).

    Args:
        text: Text containing variable references
        variables: Dict mapping variable names to values
        env_expand: Whether to fall back to environment variables (default: True)
        env_blocklist: Patterns to block (overrides defaults if provided; use [] to allow all)

    Returns:
        Text with variables expanded. Unknown variables are left as-is.
    """
    import os
    import re

    # Use custom blocklist if provided, otherwise use defaults
    full_blocklist = env_blocklist if env_blocklist is not None else DEFAULT_ENV_VAR_BLOCKLIST

    def replace_var(match):
        var_name = match.group(1)
        # Check script variables first
        if var_name in variables:
            return variables[var_name]
        # Fall back to environment if enabled and not blocked
        if env_expand:
            if not _matches_blocklist(var_name, full_blocklist):
                env_value = os.environ.get(var_name)
                if env_value is not None:
                    return env_value
        return match.group(0)  # Leave as-is if not found

    # Match ${name} where name is alphanumeric with underscores
    return re.sub(r'\$\{([a-zA-Z_][a-zA-Z0-9_]*)\}', replace_var, text)


def resolve_placeholders(text: str, registry: dict) -> tuple[Optional[str], Optional[str]]:
    """Expand @path placeholders using the file registry.

    Supports @path or @{path}. Only expands references that exist in the registry;
    other @-prefixed tokens (e.g., Java annotations like @Override) are left untouched.
    """
    import re
    pattern = re.compile(r'@(?:\{([^}]+)\}|(\S+))')

    def repl(match):
        key = match.group(1) or match.group(2)
        if key not in registry:
            # Not a registered file reference; leave as-is
            return match.group(0)
        entry = registry[key]
        return entry["content"] if isinstance(entry, dict) else entry

    expanded = pattern.sub(repl, text)
    return expanded, None
