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
    elif command == 'setmodel':
        return CommandResult(
            needs_ui_interaction=True,
            command_type='setmodel'
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

    elif command == 'settemp':
        return CommandResult(
            needs_ui_interaction=True,
            command_type='settemp'
        )

    elif command == 'clear':
        return CommandResult(
            needs_ui_interaction=True,
            command_type='clear'
        )

    else:
        return CommandResult(
            message=f"Unknown command: /{command}\n"
                    "Available commands: /new, /save, /load, /branch, /setModel, /setTemp, /clear, /exit"
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
