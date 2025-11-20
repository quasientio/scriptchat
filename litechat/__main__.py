"""Main entry point for lite-chat."""

import logging
import sys

from .core.config import load_config
from .core.conversations import Conversation, Message
from .core.ollama_client import OllamaServerManager, OllamaChatClient
from .core.commands import AppState
from .ui.app import run_ui

logger = logging.getLogger(__name__)


def main():
    """Main entry point for the application."""
    try:
        # Load configuration (this also initializes logging)
        print("Loading configuration from ~/.lite-chat/config.toml...")
        config = load_config()

        # Log startup information
        logger.info("=== lite-chat starting ===")
        logger.info(f"Configuration loaded: api_url={config.api_url}, default_model={config.default_model}")

        # Inform user where logs are being written
        if config.log_file:
            print(f"Logging to: {config.log_file} (level: {config.log_level})")
        else:
            print(f"Logging to stderr (level: {config.log_level})")

        # Initialize server manager and client
        logger.debug("Initializing Ollama server manager and client")
        server_manager = OllamaServerManager(config.api_url)
        client = OllamaChatClient(config, server_manager)

        # Create initial conversation
        messages = []
        if config.system_prompt:
            messages.append(Message(
                role='system',
                content=config.system_prompt
            ))

        initial_conversation = Conversation(
            id=None,
            model_name=config.default_model,
            temperature=config.default_temperature,
            messages=messages,
            tokens_in=0,
            tokens_out=0
        )

        # Create application state
        state = AppState(
            config=config,
            current_conversation=initial_conversation,
            client=client,
            conversations_root=config.conversations_dir
        )

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
