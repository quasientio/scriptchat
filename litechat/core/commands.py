"""Command parsing and handling for lite-chat."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import re

from .config import Config
from .conversations import Conversation, Message
from .ollama_client import OllamaChatClient
from .provider_dispatcher import ProviderDispatcher


@dataclass
class AppState:
    """Global application state."""
    config: Config
    current_conversation: Conversation
    client: object
    conversations_root: Path


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
        messages=messages,
        system_prompt=state.current_conversation.system_prompt or state.config.system_prompt,
        tokens_in=0,
        tokens_out=0
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
        # Load file contents
        if len(parts) < 2:
            return CommandResult(message="Usage: /file <path>")

        file_path = parts[1].strip()
        try:
            # Expand user path (e.g., ~/file.txt)
            expanded_path = Path(file_path).expanduser()

            # Read file contents
            with open(expanded_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Return the file contents to be sent as a user message
            return CommandResult(
                file_content=content,
                message=f"Loaded file: {expanded_path}"
            )
        except FileNotFoundError:
            return CommandResult(message=f"File not found: {file_path}")
        except PermissionError:
            return CommandResult(message=f"Permission denied: {file_path}")
        except UnicodeDecodeError:
            return CommandResult(message=f"Cannot read file (not UTF-8 text): {file_path}")
        except Exception as e:
            return CommandResult(message=f"Error reading file: {str(e)}")

    elif command == 'echo':
        # Echo command - print message without sending to LLM
        message = parts[1] if len(parts) > 1 else ""
        return CommandResult(message=message)

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

        lines = [
            f"Provider: {convo.provider_id}",
            f"Model: {convo.model_name}",
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
                    "Available commands: /new, /save, /load, /branch, /rename, /chats, /send, /export, /import, /stream, /prompt, /run, /model, /temp, /timeout, /profile, /clear, /file, /echo, /assert, /assert-not, /exit"
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
        state.config.get_model(state.current_conversation.provider_id, model_name)
        state.current_conversation.model_name = model_name
        # Reset token counters when changing model
        state.current_conversation.tokens_in = 0
        state.current_conversation.tokens_out = 0
        return CommandResult(message=f"Switched to model: {model_name}")
    except ValueError as e:
        return CommandResult(message=str(e))


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
