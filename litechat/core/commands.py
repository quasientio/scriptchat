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

"""Command parsing and handling for lite-chat."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import re

from .config import Config, reasoning_levels_for_model, reasoning_default_for_model
from .conversations import Conversation, Message, save_conversation
from .ollama_client import OllamaChatClient
from .provider_dispatcher import ProviderDispatcher


@dataclass
class AppState:
    """Global application state."""
    config: Config
    current_conversation: Conversation
    client: object
    conversations_root: Path
    file_registry: dict = field(default_factory=dict)  # key -> {"content": str, "full_path": str}


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
        tags=state.current_conversation.tags.copy() if state.current_conversation else {}
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
            message="Exiting lite-chat...",
            should_exit=True
        )

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

    elif command == 'send':
        if len(parts) < 2 or not parts[1].strip():
            return CommandResult(message="Usage: /send <message>")
        return CommandResult(
            needs_ui_interaction=True,
            command_type='send'
        )

    elif command == 'export':
        return CommandResult(
            needs_ui_interaction=True,
            command_type='export'
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

    elif command == 'temp':
        return CommandResult(
            needs_ui_interaction=True,
            command_type='temp'
        )

    elif command == 'reason':
        level_arg = parts[1].strip() if len(parts) > 1 else ""
        return set_reasoning_level(state, level_arg)

    elif command == 'import':
        if len(parts) < 2 or not parts[1].strip():
            return CommandResult(message="Usage: /import <path>")
        return CommandResult(
            needs_ui_interaction=True,
            command_type='import'
        )

    elif command == 'timeout':
        if len(parts) < 2 or not parts[1].strip():
            return CommandResult(message=f"Current timeout: {state.config.timeout} seconds\nUsage: /timeout <seconds>")
        try:
            seconds = float(parts[1].strip())
        except ValueError:
            return CommandResult(message="Invalid timeout. Usage: /timeout <seconds>")
        return set_timeout(state, seconds)

    elif command == 'clear':
        return CommandResult(
            needs_ui_interaction=True,
            command_type='clear'
        )

    elif command == 'file':
        if len(parts) < 2:
            return CommandResult(message="Usage: /file <path> [--force]")

        arg_str = parts[1].strip()
        force = False
        file_path = arg_str
        if arg_str.startswith("--force "):
            force = True
            file_path = arg_str[len("--force "):].strip()
        elif arg_str == "--force":
            return CommandResult(message="Usage: /file <path> [--force]")

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

        # Register by original path token
        state.file_registry[file_path] = {"content": content, "full_path": full_path_str}
        # Also register by basename if not already present
        basename = expanded_path.name
        alt = ""
        if basename not in state.file_registry:
            state.file_registry[basename] = {"content": content, "full_path": full_path_str}
            alt = f" and @{basename}"

        return CommandResult(message=f"Registered @{file_path}{alt} ({len(content)} chars)")

    elif command == 'echo':
        # Echo command - print message without sending to LLM
        message = parts[1] if len(parts) > 1 else ""
        return CommandResult(message=message)

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
        logging.getLogger().setLevel(level)
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
        lines = [
            f"Provider: {convo.provider_id}",
            f"Model: {convo.model_name}",
            f"Reasoning: {convo.reasoning_level or '(default)'}",
            f"Temperature: {convo.temperature:.2f}",
            f"Tokens: {convo.tokens_in} in / {convo.tokens_out} out",
            f"Streaming: {'on' if cfg.enable_streaming else 'off'}",
            f"Timeout: {cfg.timeout}s",
            f"Conversations dir: {cfg.conversations_dir} ({conv_count} convs)",
        ]
        if cfg.exports_dir:
            lines.append(f"Exports dir: {cfg.exports_dir} ({exp_count} files)")
        if convo.system_prompt:
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
            message=f"Unknown command: /{command}\n"
                    "Available commands: /new, /save, /load, /branch, /rename, /chats, /send, /export, /import, /stream, /prompt, /run, /model, /temp, /reason, /timeout, /profile, /clear, /file, /echo, /tag, /untag, /tags, /assert, /assert-not, /exit"
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
        return CommandResult(
            message=f"Reasoning level: {current}\nAvailable: {', '.join(available)}\nUsage: /reason <level>"
        )

    desired = level.strip().lower()
    if desired in ("default", "clear", "reset"):
        convo.reasoning_level = None
        return CommandResult(message="Reasoning level cleared (provider default).")

    if not available:
        return CommandResult(message="Reasoning controls not available for this model/provider.")

    if desired not in available:
        return CommandResult(message=f"Unsupported reasoning level for {convo.model_name}. Available: {', '.join(available)}")

    convo.reasoning_level = desired
    return CommandResult(message=f"Reasoning level set to {desired}")


def set_timeout(state: AppState, seconds: float) -> CommandResult:
    """Set the request timeout in seconds.

    Args:
        state: Application state
        seconds: Timeout duration in seconds

    Returns:
        CommandResult with execution result
    """
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
    """Drop the last assistant message and return the preceding user message to resend."""
    # Find last assistant index
    last_assistant = None
    for i in range(len(convo.messages) - 1, -1, -1):
        if convo.messages[i].role == 'assistant':
            last_assistant = i
            break
    if last_assistant is None:
        return None, "No assistant response to retry."

    # Find the user message before that assistant
    last_user_content = None
    for j in range(last_assistant - 1, -1, -1):
        if convo.messages[j].role == 'user':
            last_user_content = convo.messages[j].content
            break

    if last_user_content is None:
        return None, "No user message found to retry."

    # Remove the assistant message
    convo.messages.pop(last_assistant)
    return last_user_content, "Retrying last user message."


def resolve_placeholders(text: str, registry: dict) -> tuple[Optional[str], Optional[str]]:
    """Expand @path placeholders using the file registry.

    Supports @path or @{path}. If a placeholder is missing, returns (None, error).
    """
    import re
    pattern = re.compile(r'@(?:\{([^}]+)\}|(\S+))')
    missing = []

    def repl(match):
        key = match.group(1) or match.group(2)
        if key not in registry:
            missing.append(key)
            return match.group(0)
        entry = registry[key]
        return entry["content"] if isinstance(entry, dict) else entry

    expanded = pattern.sub(repl, text)
    if missing:
        return None, f"Unregistered file reference(s): {', '.join(missing)}"
    return expanded, None
