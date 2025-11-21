"""Main entry point for lite-chat."""

import argparse
import logging
import sys
from pathlib import Path

from .core.config import load_config
from .core.conversations import Conversation, Message, save_conversation
from .core.ollama_client import OllamaServerManager, OllamaChatClient
from .core.openai_client import OpenAIChatClient
from .core.provider_dispatcher import ProviderDispatcher
from .core.commands import AppState, handle_command, set_model, set_temperature
from .ui.app import run_ui

logger = logging.getLogger(__name__)


def parse_script_lines(lines: list[str]) -> list[str]:
    """Parse script lines, stripping comments and empty lines.

    Args:
        lines: Raw lines from script file

    Returns:
        List of executable lines
    """
    parsed = []
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        parsed.append(line)
    return parsed


def handle_batch_command(line: str, state: AppState, line_num: int) -> tuple[bool, str | None]:
    """Handle a command in batch mode.

    Args:
        line: Command line (starting with /)
        state: Application state
        line_num: Line number for error messages

    Returns:
        Tuple of (should_continue, message_to_send)
        - should_continue: False if script should exit
        - message_to_send: Message to send to chat (for /send and /file commands)
    """
    parts = line[1:].split(maxsplit=1)
    if not parts:
        print(f"[{line_num}] Empty command")
        return True, None

    command = parts[0].lower()
    args = parts[1] if len(parts) > 1 else ""

    # Commands that work in batch mode
    if command == 'exit':
        logger.info("Script requested exit via /exit command")
        return False, None

    elif command == 'new':
        # Create new conversation using handle_command
        result = handle_command(line, state)
        print(f"[{line_num}] {result.message}")
        return True, None

    elif command == 'model':
        if not args:
            print(f"[{line_num}] Error: /model requires an argument (model index or name)")
            return True, None

        # Try as index first, then as name
        try:
            index = int(args)
            if 0 <= index < len(state.config.models):
                model_name = state.config.models[index].name
                result = set_model(state, model_name)
                print(f"[{line_num}] {result.message}")
            else:
                print(f"[{line_num}] Error: Invalid model index: {index}")
        except ValueError:
            # Try as model name
            result = set_model(state, args)
            print(f"[{line_num}] {result.message}")
        return True, None

    elif command == 'temp':
        if not args:
            print(f"[{line_num}] Error: /temp requires a temperature value")
            return True, None

        try:
            temp = float(args)
            result = set_temperature(state, temp)
            print(f"[{line_num}] {result.message}")
        except ValueError:
            print(f"[{line_num}] Error: Invalid temperature value: {args}")
        return True, None

    elif command == 'stream':
        arg = args.strip().lower()
        if arg in ('on', 'off'):
            state.config.enable_streaming = (arg == 'on')
            status = "enabled" if state.config.enable_streaming else "disabled"
            print(f"[{line_num}] Streaming {status}")
        elif not arg:
            # Toggle
            state.config.enable_streaming = not state.config.enable_streaming
            status = "enabled" if state.config.enable_streaming else "disabled"
            print(f"[{line_num}] Streaming {status}")
        else:
            print(f"[{line_num}] Error: /stream expects 'on' or 'off'")
        return True, None

    elif command == 'prompt':
        if not args:
            print(f"[{line_num}] Error: /prompt requires an argument")
            return True, None

        if args.lower() == 'clear':
            # Clear system prompt
            if state.current_conversation.messages and state.current_conversation.messages[0].role == 'system':
                state.current_conversation.messages.pop(0)
            state.current_conversation.system_prompt = None
            print(f"[{line_num}] System prompt cleared")
        else:
            # Set system prompt
            if state.current_conversation.messages and state.current_conversation.messages[0].role == 'system':
                state.current_conversation.messages.pop(0)
            state.current_conversation.system_prompt = args
            state.current_conversation.messages.insert(0, Message(role='system', content=args))
            print(f"[{line_num}] System prompt set")
        return True, None

    elif command == 'save':
        if not args:
            print(f"[{line_num}] Error: /save requires a save name in batch mode")
            return True, None

        try:
            save_conversation(
                state.conversations_root,
                state.current_conversation,
                save_name=args,
                system_prompt=state.current_conversation.system_prompt
            )
            print(f"[{line_num}] Conversation saved as: {args}")
        except Exception as e:
            print(f"[{line_num}] Error saving conversation: {e}")
        return True, None

    elif command == 'send':
        if not args:
            print(f"[{line_num}] Error: /send requires a message")
            return True, None
        # Return the message to send
        return True, args

    elif command == 'file':
        result = handle_command(line, state)
        if result.file_content:
            print(f"[{line_num}] {result.message}")
            return True, result.file_content
        else:
            print(f"[{line_num}] {result.message}")
            return True, None

    elif command == 'export':
        # Export conversation to file
        format_arg = args.strip().lower()
        if not format_arg or format_arg != 'md':
            print(f"[{line_num}] Error: /export requires 'md' format (only supported format)")
            return True, None

        from .core.conversations import export_conversation_md
        target_dir = state.config.exports_dir or Path.cwd()

        try:
            path = export_conversation_md(state.current_conversation, target_dir)
            print(f"[{line_num}] Exported to: {path}")
        except Exception as e:
            print(f"[{line_num}] Error exporting: {e}")
        return True, None

    elif command == 'echo':
        # Print message without sending to LLM
        if not args:
            # Empty echo is valid (prints blank line)
            print()
        else:
            print(args)
        return True, None

    else:
        print(f"[{line_num}] Error: Command '{command}' not supported in batch mode or unknown")
        print(f"[{line_num}] Supported commands: /new, /exit, /model, /temp, /stream, /prompt, /save, /send, /file, /export, /echo")
        return True, None


def run_batch(script_path: str, state: AppState) -> int:
    """Run a script file in batch mode (non-interactively).

    Args:
        script_path: Path to script file
        state: Application state

    Returns:
        Exit code (0 for success, 1 for error)
    """
    try:
        path = Path(script_path).expanduser()
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: File not found: {script_path}", file=sys.stderr)
        return 1
    except PermissionError:
        print(f"Error: Permission denied: {script_path}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error reading {script_path}: {e}", file=sys.stderr)
        return 1

    script_lines = parse_script_lines(lines)
    if not script_lines:
        print(f"Error: No runnable lines in {script_path} (comments/empty only).", file=sys.stderr)
        return 1

    logger.info(f"Running script: {script_path} ({len(script_lines)} lines)")

    # Execute each line
    for i, line in enumerate(script_lines, 1):
        logger.debug(f"Executing line {i}: {line}")

        message_to_send = None

        if line.startswith('/'):
            # Handle as command
            should_continue, msg = handle_batch_command(line, state, i)

            if not should_continue:
                return 0

            if msg:
                # Command returned a message to send (e.g., /send or /file)
                message_to_send = msg
            else:
                # Command handled, continue to next line
                continue
        else:
            # Regular message line
            message_to_send = line

        # Send message to chat
        if message_to_send:
            print(f"\n[User]: {message_to_send}")

            # Add user message to conversation
            state.current_conversation.messages.append(Message(
                role='user',
                content=message_to_send
            ))

            # Get response from model (non-streaming)
            try:
                response = state.client.chat(
                    convo=state.current_conversation,
                    new_user_message=message_to_send,
                    streaming=False
                )

                print(f"[Assistant]: {response}\n")

            except Exception as e:
                print(f"Error getting response: {e}", file=sys.stderr)
                logger.exception(f"Error during chat: {e}")
                return 1

    logger.info("Script completed successfully")
    return 0


def main():
    """Main entry point for the application."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="lite-chat: A lightweight terminal UI for Ollama",
        prog="python -m litechat"
    )
    parser.add_argument(
        '--run',
        metavar='PATH',
        help='Run a script file in batch mode (non-interactively) and exit'
    )
    args = parser.parse_args()

    try:
        # Load configuration (this also initializes logging)
        if not args.run:
            print("Loading configuration from ~/.lite-chat/config.toml...")
        config = load_config()

        # Log startup information
        logger.info("=== lite-chat starting ===")
        logger.info(f"Configuration loaded: api_url={config.api_url}, default_model={config.default_model}")

        # Inform user where logs are being written (only in interactive mode)
        if not args.run:
            if config.log_file:
                print(f"Logging to: {config.log_file} (level: {config.log_level})")
            else:
                print(f"Logging to stderr (level: {config.log_level})")

        # Initialize provider clients
        provider_clients = {}
        for p in config.providers:
            if p.type == "ollama":
                logger.debug("Initializing Ollama provider '%s'", p.id)
                server_manager = OllamaServerManager(p.api_url)
                provider_clients[p.id] = OllamaChatClient(config, server_manager, base_url=p.api_url)
            elif p.type == "openai-compatible":
                logger.debug("Initializing OpenAI-compatible provider '%s'", p.id)
                provider_clients[p.id] = OpenAIChatClient(config, p, timeout=config.timeout)
            else:
                logger.warning("Provider type '%s' not supported yet (id=%s)", p.type, p.id)

        dispatcher = ProviderDispatcher(provider_clients)

        if config.default_provider not in provider_clients:
            available = ", ".join(provider_clients.keys())
            raise ValueError(f"Default provider '{config.default_provider}' not configured. Available: {available}")

        # Create initial conversation
        messages = []
        if config.system_prompt:
            messages.append(Message(
                role='system',
                content=config.system_prompt
            ))

        # Pick initial model: config.default_model, provider.default_model, or first provider model
        initial_model = config.default_model
        try:
            default_provider_obj = config.get_provider(config.default_provider)
            if not initial_model:
                if default_provider_obj.default_model:
                    initial_model = default_provider_obj.default_model
                elif default_provider_obj.models:
                    initial_model = default_provider_obj.models[0].name
        except ValueError:
            pass

        initial_conversation = Conversation(
            id=None,
            provider_id=config.default_provider,
            model_name=initial_model,
            temperature=config.default_temperature,
            messages=messages,
            system_prompt=config.system_prompt,
            tokens_in=0,
            tokens_out=0
        )

        # Create application state
        state = AppState(
            config=config,
            current_conversation=initial_conversation,
            client=dispatcher,
            conversations_root=config.conversations_dir
        )

        # Check if running in batch mode
        if args.run:
            # Run script in batch mode
            exit_code = run_batch(args.run, state)
            # Cleanup
            try:
                client.unload_model()
                client.server_manager.stop()
            except Exception as e:
                logger.error(f"Error during cleanup: {e}")
            sys.exit(exit_code)

        # Run the UI
        print("Starting lite-chat...")
        print("Press Ctrl+C or Ctrl+D to exit, or use /exit command")
        run_ui(state)

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        print("\nPlease create a configuration file at ~/.lite-chat/config.toml", file=sys.stderr)
        print("See the example configuration for reference.", file=sys.stderr)
        sys.exit(1)

    except ValueError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        # Note: logger may not be initialized if config loading failed
        try:
            logger.error(f"Configuration error: {e}")
        except:
            pass
        sys.exit(1)

    except KeyboardInterrupt:
        print("\nExiting lite-chat...")
        logger.info("Application terminated by user (Ctrl+C)")
        sys.exit(0)

    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        # Log the full traceback
        try:
            logger.exception(f"Unexpected error: {e}")
        except:
            pass
        sys.exit(1)


if __name__ == '__main__':
    main()
