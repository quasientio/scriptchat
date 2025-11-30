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

"""Main entry point for ScriptChat."""

import argparse
import logging
import sys
from pathlib import Path

from .core.config import load_config
from .core.conversations import Conversation, Message, save_conversation
from .core.exports import (
    export_conversation_md,
    export_conversation_json,
    export_conversation_html,
    import_conversation_from_file,
    generate_html_index,
)
from .core.ollama_client import OllamaServerManager, OllamaChatClient
from .core.openai_client import OpenAIChatClient
from .core.provider_dispatcher import ProviderDispatcher
from .core.commands import AppState, assert_last_response, handle_command, set_model, set_temperature, set_timeout, retry_last_user_message, resolve_placeholders
from .core.config import reasoning_default_for_model
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


def handle_batch_command(
    line: str,
    state: AppState,
    line_num: int,
    continue_on_error: bool = False
) -> tuple[bool, str | None, int | None]:
    """Handle a command in batch mode.

    Returns (should_continue, message_to_send, exit_code_if_stopping)."""
    parts = line[1:].split(maxsplit=1)
    if not parts:
        print(f"[{line_num}] Empty command")
        return True, None, None

    command = parts[0].lower()
    args = parts[1] if len(parts) > 1 else ""

    # Commands that work in batch mode
    if command == 'exit':
        logger.info("Script requested exit via /exit command")
        return False, None, 0

    if command == 'assert':
        pattern = args.strip()
        if not pattern:
            print(f"[{line_num}] Error: /assert requires a pattern")
            return True, None, None
        passed, msg = assert_last_response(state.current_conversation, pattern)
        print(f"[{line_num}] {msg}")
        if passed:
            return True, None, None
        return (True, None, 1) if continue_on_error else (False, None, 1)

    if command == 'assert-not':
        pattern = args.strip()
        if not pattern:
            print(f"[{line_num}] Error: /assert-not requires a pattern")
            return True, None, None
        passed, msg = assert_last_response(state.current_conversation, pattern, negate=True)
        print(f"[{line_num}] {msg}")
        if passed:
            return True, None, None
        return (True, None, 1) if continue_on_error else (False, None, 1)

    if command == 'new':
        result = handle_command(line, state)
        print(f"[{line_num}] {result.message}")
        return True, None, None

    if command == 'model':
        if not args:
            print(f"[{line_num}] Error: /model requires an argument (model index or name)")
            return True, None, None
        try:
            index = int(args)
            if 0 <= index < len(state.config.models):
                model_name = state.config.models[index].name
                result = set_model(state, model_name)
                print(f"[{line_num}] {result.message}")
            else:
                print(f"[{line_num}] Error: Invalid model index: {index}")
        except ValueError:
            # Accept provider/model or just model name
            if '/' in args:
                provider_id, model_name = args.split('/', 1)
                provider_id = provider_id.strip()
                model_name = model_name.strip()
                if provider_id:
                    state.current_conversation.provider_id = provider_id
                result = set_model(state, model_name)
                print(f"[{line_num}] {result.message} (provider: {provider_id})")
            else:
                result = set_model(state, args)
                print(f"[{line_num}] {result.message}")
            print(f"[{line_num}] {result.message}")
        return True, None, None

    if command == 'temp':
        if not args:
            print(f"[{line_num}] Error: /temp requires a temperature value")
            return True, None, None
        try:
            temp = float(args)
            result = set_temperature(state, temp)
            print(f"[{line_num}] {result.message}")
        except ValueError:
            print(f"[{line_num}] Error: Invalid temperature value: {args}")
        return True, None, None

    if command == 'reason':
        result = handle_command(line, state)
        if result.message:
            print(f"[{line_num}] {result.message}")
        return True, None, None

    if command == 'timeout':
        if not args:
            print(f"[{line_num}] Current timeout: {state.config.timeout} seconds")
            return True, None, None
        try:
            seconds = float(args)
            result = set_timeout(state, seconds)
            print(f"[{line_num}] {result.message}")
        except ValueError:
            print(f"[{line_num}] Error: Invalid timeout value: {args}")
        return True, None, None

    if command == 'stream':
        arg = args.strip().lower()
        if arg in ('on', 'off'):
            state.config.enable_streaming = (arg == 'on')
            status = "enabled" if state.config.enable_streaming else "disabled"
            print(f"[{line_num}] Streaming {status}")
        elif not arg:
            state.config.enable_streaming = not state.config.enable_streaming
            status = "enabled" if state.config.enable_streaming else "disabled"
            print(f"[{line_num}] Streaming {status}")
        else:
            print(f"[{line_num}] Error: /stream expects 'on' or 'off'")
        return True, None, None

    if command == 'prompt':
        if not args:
            print(f"[{line_num}] Error: /prompt requires an argument")
            return True, None, None
        if args.lower() == 'clear':
            if state.current_conversation.messages and state.current_conversation.messages[0].role == 'system':
                state.current_conversation.messages.pop(0)
            state.current_conversation.system_prompt = None
            print(f"[{line_num}] System prompt cleared")
        else:
            if state.current_conversation.messages and state.current_conversation.messages[0].role == 'system':
                state.current_conversation.messages.pop(0)
            state.current_conversation.system_prompt = args
            state.current_conversation.messages.insert(0, Message(role='system', content=args))
            print(f"[{line_num}] System prompt set")
        return True, None, None

    if command == 'save':
        if not args:
            print(f"[{line_num}] Error: /save requires a save name in batch mode")
            return True, None, None
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
        return True, None, None

    if command == 'import':
        if not args:
            print(f"[{line_num}] Error: /import requires a path")
            return True, None, None
        try:
            imported = import_conversation_from_file(Path(args), state.conversations_root)
            state.current_conversation = imported
            print(f"[{line_num}] Imported conversation as: {imported.id}")
        except Exception as e:
            print(f"[{line_num}] Error importing: {e}")
        return True, None, None

    if command == 'profile':
        result = handle_command(line, state)
        if result.message:
            print(f"[{line_num}] {result.message}")
        return True, None, None

    if command == 'undo':
        result = handle_command(line, state)
        if result.message:
            print(f"[{line_num}] {result.message}")
        return True, None, None

    if command == 'send':
        if not args:
            print(f"[{line_num}] Error: /send requires a message")
            return True, None, None
        return True, args, None

    if command == 'file':
        result = handle_command(line, state)
        if result.file_content:
            print(f"[{line_num}] {result.message}")
            return True, result.file_content, None
        else:
            print(f"[{line_num}] {result.message}")
            return True, None, None

    if command == 'export':
        format_arg = args.strip().lower() or "md"
        if format_arg not in ('md', 'json', 'html'):
            print(f"[{line_num}] Error: /export format must be 'md', 'json', or 'html'")
            return True, None, None
        target_dir = state.config.exports_dir or Path.cwd()
        try:
            if format_arg == 'md':
                path = export_conversation_md(state.current_conversation, target_dir)
            elif format_arg == 'json':
                path = export_conversation_json(state.current_conversation, target_dir)
            else:
                path = export_conversation_html(state.current_conversation, target_dir)
                # Regenerate index.html for HTML exports
                generate_html_index(target_dir, state.conversations_root)
            print(f"[{line_num}] Exported to: {path}")
        except Exception as e:
            print(f"[{line_num}] Error exporting: {e}")
        return True, None, None

    if command == 'echo':
        if not args:
            print()
        else:
            print(args)
        return True, None, None

    if command == 'sleep':
        if not args:
            print(f"[{line_num}] Error: /sleep requires duration in seconds")
            return True, None, None
        try:
            import time
            seconds = float(args)
            if seconds < 0:
                print(f"[{line_num}] Error: Sleep duration must be positive")
                return True, None, None
            print(f"[{line_num}] Sleeping for {seconds} seconds...")
            time.sleep(seconds)
            print(f"[{line_num}] Sleep complete")
        except ValueError:
            print(f"[{line_num}] Error: Invalid sleep duration: {args}")
        return True, None, None

    if command in ('retry', 'tag', 'log-level', 'files'):
        result = handle_command(line, state)
        if result.message:
            print(f"[{line_num}] {result.message}")
        return True, result.resend_message if command == 'retry' else None, None

    print(f"[{line_num}] Error: Command '{command}' not supported in batch mode or unknown")
    print(f"[{line_num}] Supported commands: /new, /exit, /model, /temp, /timeout, /profile, /log-level, /files, /stream, /prompt, /save, /send, /file, /export, /import, /echo, /sleep, /assert, /undo, /retry, /tag")
    return True, None, None


def run_batch_lines(
    lines: list[str],
    state: AppState,
    continue_on_error: bool = False,
    source_label: str | None = None
) -> int:
    """Run already-loaded script lines in batch mode."""
    script_lines = parse_script_lines(lines)
    label = source_label or "<stdin>"
    if not script_lines:
        print(f"Error: No runnable lines in {label} (comments/empty only).", file=sys.stderr)
        return 1

    logger.info(f"Running script: {label} ({len(script_lines)} lines)")

    pass_failures = 0

    # Execute each line
    for i, line in enumerate(script_lines, 1):
        logger.debug(f"Executing line {i}: {line}")

        message_to_send = None
        exit_code = None

        if line.startswith('/'):
            # Handle as command
            result = handle_batch_command(line, state, i, continue_on_error=continue_on_error)
            should_continue, msg, exit_code = result

            if not should_continue:
                if exit_code is not None:
                    return 1 if (exit_code == 0 and pass_failures > 0) else exit_code
                return 1 if pass_failures > 0 else 0

            if msg:
                # Command returned a message to send (e.g., /send or /file)
                message_to_send = msg
            else:
                if exit_code:
                    pass_failures += 1
                # Command handled, continue to next line
                continue
        else:
            # Regular message line
            message_to_send = line

        # Send message to chat
        if message_to_send:
            print(f"\n[User]: {message_to_send}")

            # Add user message to conversation (store original with @ refs)
            state.current_conversation.messages.append(Message(
                role='user',
                content=message_to_send
            ))

            # Expand placeholders for sending
            expanded_messages = []
            for msg in state.current_conversation.messages:
                if msg.role in ('user', 'system'):
                    expanded, err = resolve_placeholders(msg.content, state.file_registry)
                    if err:
                        print(f"[{i}] {err}", file=sys.stderr)
                        # remove the appended user message on error
                        state.current_conversation.messages.pop()
                        return 1
                    expanded_messages.append(Message(role=msg.role, content=expanded))
                else:
                    expanded_messages.append(msg)

            # Get response from model (non-streaming)
            try:
                response = state.client.chat(
                    convo=state.current_conversation,
                    new_user_message=expanded_messages[-1].content,
                    expanded_history=expanded_messages[:-1],
                    streaming=False
                )

                print(f"[Assistant]: {response}\n")

            except Exception as e:
                print(f"Error getting response: {e}", file=sys.stderr)
                logger.exception(f"Error during chat: {e}")
                return 1

        if exit_code:
            # Assert failed but continue_on_error allowed
            pass_failures += 1

    logger.info("Script completed successfully")
    return 0 if pass_failures == 0 else 1


def run_batch(script_path: str, state: AppState, continue_on_error: bool = False) -> int:
    """Run a script file in batch mode (non-interactively)."""
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

    return run_batch_lines(lines, state, continue_on_error=continue_on_error, source_label=str(script_path))


def main():  # pragma: no cover - interactive entrypoint not exercised in unit tests
    """Main entry point for the application."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="ScriptChat: A scriptable terminal chat client for LLMs",
        prog="python -m scriptchat"
    )
    parser.add_argument(
        '--run',
        metavar='PATH',
        help='Run a script file in batch mode (non-interactively) and exit'
    )
    parser.add_argument(
        '--continue-on-error',
        action='store_true',
        help='In batch mode, continue running after assertion failures (exit code 1 if any fail)'
    )
    args = parser.parse_args()

    try:
        # Load configuration (this also initializes logging)
        if not args.run:
            print("Loading configuration from ~/.scriptchat/config.toml...")
        config = load_config()

        # Log startup information
        logger.info("=== ScriptChat starting ===")
        logger.info(f"Configuration loaded: api_url={config.api_url}, default_model={config.default_model}")

        # Inform user where logs are being written (only in interactive mode)
        if not args.run:
            if config.log_file:
                print(f"Logging to: {config.log_file} (level: {config.log_level})")
            else:
                print(f"Logging to stderr (level: {config.log_level})")

        # Initialize provider clients
        is_interactive = not args.run
        provider_clients = {}
        for p in config.providers:
            if p.type == "ollama":
                logger.debug("Initializing Ollama provider '%s'", p.id)
                server_manager = OllamaServerManager(p.api_url, interactive=is_interactive)
                provider_clients[p.id] = OllamaChatClient(config, server_manager, base_url=p.api_url)
            elif p.type == "openai-compatible":
                logger.debug("Initializing OpenAI-compatible provider '%s'", p.id)
                provider_clients[p.id] = OpenAIChatClient(config, p, timeout=config.timeout)
            elif p.type == "anthropic":
                logger.debug("Initializing Anthropic provider '%s'", p.id)
                from .core.anthropic_client import AnthropicChatClient
                provider_clients[p.id] = AnthropicChatClient(config, p, timeout=config.timeout)
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
            tokens_out=0,
            reasoning_level=reasoning_default_for_model(config, config.default_provider, initial_model) if initial_model else None,
        )

        # Create application state
        state = AppState(
            config=config,
            current_conversation=initial_conversation,
            client=dispatcher,
            conversations_root=config.conversations_dir,
            file_registry={}
        )

        # Check if running in batch mode
        if args.run:
            # Run script in batch mode
            exit_code = run_batch(args.run, state, continue_on_error=args.continue_on_error)
            # Cleanup
            try:
                client.unload_model()
                client.server_manager.stop()
            except Exception as e:
                logger.error(f"Error during cleanup: {e}")
            sys.exit(exit_code)

        # If stdin is not a TTY and no --run provided, treat stdin as a script
        if not sys.stdin.isatty():
            stdin_lines = sys.stdin.read().splitlines()
            exit_code = run_batch_lines(stdin_lines, state, continue_on_error=args.continue_on_error, source_label="<stdin>")
            try:
                client.unload_model()
                client.server_manager.stop()
            except Exception as e:
                logger.error(f"Error during cleanup: {e}")
            sys.exit(exit_code)

        # Run the UI
        print("Starting ScriptChat...")
        print("Press Ctrl+C or Ctrl+D to exit, or use /exit command")
        run_ui(state)

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        print("\nPlease create a configuration file at ~/.scriptchat/config.toml", file=sys.stderr)
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
        print("\nExiting ScriptChat...")
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
