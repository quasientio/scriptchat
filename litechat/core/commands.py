"""Command parsing and handling for lite-chat."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .config import Config
from .conversations import Conversation, Message
from .ollama_client import OllamaChatClient


@dataclass
class AppState:
    """Global application state."""
    config: Config
    current_conversation: Conversation
    client: OllamaChatClient
    conversations_root: Path


@dataclass
class CommandResult:
    """Result of a command execution."""
    message: Optional[str] = None
    should_exit: bool = False
    needs_ui_interaction: bool = False
    command_type: Optional[str] = None  # For complex commands that need UI handling
    file_content: Optional[str] = None  # For /file command - content to send as user message


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
        model_name=state.current_conversation.model_name,
        temperature=state.current_conversation.temperature,
        messages=messages,
        system_prompt=state.config.system_prompt,
        tokens_in=0,
        tokens_out=0
    )


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

    else:
        return CommandResult(
            message=f"Unknown command: /{command}\n"
                    "Available commands: /new, /save, /load, /branch, /rename, /chats, /send, /export, /stream, /prompt, /run, /model, /temp, /clear, /file, /echo, /exit"
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
        model_cfg = state.config.get_model(model_name)
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
