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

"""Ollama API client and server management for ScriptChat."""

import json
import logging
import os
import subprocess
import time
from typing import Optional

import requests

from .config import Config
from .conversations import Conversation, Message

logger = logging.getLogger(__name__)


class OllamaServerManager:
    """Manages the Ollama server process with context length configuration."""

    def __init__(self, api_url: str, interactive: bool = True):
        """Initialize the server manager.

        Args:
            api_url: Base URL for the Ollama API
            interactive: Whether to prompt user for input (False for batch mode)
        """
        self.api_url = api_url
        self.interactive = interactive
        self.current_process: Optional[subprocess.Popen] = None
        self.current_context_length: Optional[int] = None
        self.using_external_instance = False
        self._alternative_port: Optional[int] = None

        # Check for existing Ollama instance and handle it
        self._handle_existing_instance()

    def _parse_port_from_url(self) -> int:
        """Extract port number from API URL."""
        from urllib.parse import urlparse
        parsed = urlparse(self.api_url)
        return parsed.port or 11434  # Default Ollama port

    def _check_ollama_running(self, port: Optional[int] = None) -> bool:
        """Check if Ollama is running on the specified port.

        Args:
            port: Port to check (uses configured URL if None)

        Returns:
            True if Ollama is responding, False otherwise
        """
        if port is None:
            check_url = f"{self.api_url.rstrip('/')}/tags"
        else:
            from urllib.parse import urlparse
            parsed = urlparse(self.api_url)
            check_url = f"{parsed.scheme}://{parsed.hostname}:{port}/api/tags"

        try:
            response = requests.get(check_url, timeout=2)
            return response.status_code == 200
        except Exception:
            return False

    def _handle_existing_instance(self) -> None:  # pragma: no cover - requires user input
        """Check for existing Ollama instance and prompt user if found."""
        if not self._check_ollama_running():
            # No existing instance, will start our own
            return

        configured_port = self._parse_port_from_url()
        logger.info(f"Detected existing Ollama instance on port {configured_port}")

        if not self.interactive:
            # In batch mode, use the existing instance silently
            logger.info("Batch mode: using existing Ollama instance")
            self.using_external_instance = True
            return

        # Interactive mode: prompt user
        print(f"\nAn Ollama instance is already running on port {configured_port}.")
        print("Options:")
        print("  [1] Use the existing instance (recommended)")
        print("  [2] Start a new instance on a different port")
        print("  [3] Exit and let me stop the existing instance first")

        while True:
            try:
                choice = input("\nChoice [1/2/3]: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting...")
                raise SystemExit(0)

            if choice == '1':
                logger.info("User chose to use existing Ollama instance")
                self.using_external_instance = True
                print("Using existing Ollama instance.")
                return

            elif choice == '2':
                # Find an available alternative port
                alt_port = self._find_alternative_port(configured_port)
                if alt_port:
                    logger.info(f"User chose to start new instance on port {alt_port}")
                    self._alternative_port = alt_port
                    print(f"Will start new Ollama instance on port {alt_port}.")
                    return
                else:
                    print("Could not find an available port. Please stop the existing instance.")
                    continue

            elif choice == '3':
                print("Please stop the existing Ollama instance and restart ScriptChat.")
                raise SystemExit(0)

            else:
                print("Invalid choice. Please enter 1, 2, or 3.")

    def _find_alternative_port(self, base_port: int) -> Optional[int]:
        """Find an available port for a new Ollama instance.

        Args:
            base_port: The port that's already in use

        Returns:
            An available port number, or None if none found
        """
        import socket
        # Try ports in range base_port+1 to base_port+10
        for port in range(base_port + 1, base_port + 11):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('localhost', port))
                    return port
            except OSError:
                continue
        return None

    def _get_effective_api_url(self) -> str:
        """Get the API URL, accounting for alternative port if set."""
        if self._alternative_port is None:
            return self.api_url
        from urllib.parse import urlparse
        parsed = urlparse(self.api_url)
        return f"{parsed.scheme}://{parsed.hostname}:{self._alternative_port}/api"

    def ensure_running(self, context_length: int) -> None:  # pragma: no cover - spawns external server
        """Ensure Ollama server is running with the specified context length.

        If using an external instance, this is a no-op.
        If we manage the server and it's running with a different context length,
        it will be stopped and restarted.

        Args:
            context_length: Context length to set via OLLAMA_CONTEXT_LENGTH
        """
        # If using external instance, don't manage the server
        if self.using_external_instance:
            logger.debug("Using external Ollama instance, skipping server management")
            return

        # If already running with the same context length, do nothing
        if (self.current_process is not None and
                self.current_context_length == context_length and
                self.current_process.poll() is None):
            logger.debug(f"Ollama server already running with context_length={context_length}")
            return

        logger.info(f"Starting Ollama server with context_length={context_length}")

        # Stop existing process if running
        self.stop()

        # Start new process with environment variable
        env = os.environ.copy()
        env['OLLAMA_CONTEXT_LENGTH'] = str(context_length)
        env['OLLAMA_MODELS'] = '/var/lib/ollama'  # Use system models

        # Set alternative port if needed
        if self._alternative_port is not None:
            env['OLLAMA_HOST'] = f"127.0.0.1:{self._alternative_port}"
            logger.info(f"Starting Ollama on alternative port {self._alternative_port}")

        try:
            self.current_process = subprocess.Popen(
                ['ollama', 'serve'],
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            self.current_context_length = context_length

            # Wait for Ollama to be ready (up to 10 seconds)
            effective_url = self._get_effective_api_url()
            health_check_url = f"{effective_url.rstrip('/')}/tags"
            for _ in range(20):
                time.sleep(0.5)
                try:
                    requests.get(health_check_url, timeout=1)
                    break  # Server is ready
                except Exception:
                    continue

        except FileNotFoundError:
            raise FileNotFoundError(
                "Ollama executable not found. Please ensure Ollama is installed and in your PATH."
            )

    def stop(self) -> None:  # pragma: no cover - stops external server
        """Stop the managed Ollama server process gracefully."""
        if self.current_process is None:
            return

        # Check if process is still running
        if self.current_process.poll() is not None:
            # Process already exited
            logger.debug("Ollama process already stopped")
            self.current_process = None
            self.current_context_length = None
            return

        logger.info("Stopping Ollama server process")

        # Try graceful termination first
        try:
            self.current_process.terminate()

            # Wait up to 5 seconds for process to terminate
            self.current_process.wait(timeout=5)
            logger.info("Ollama server stopped gracefully")
        except subprocess.TimeoutExpired:
            # Force kill if still alive
            logger.warning("Ollama server did not stop gracefully, forcing kill")
            self.current_process.kill()
            self.current_process.wait()
            logger.info("Ollama server killed")
        except Exception as e:
            logger.error(f"Error stopping Ollama process: {e}")

        self.current_process = None
        self.current_context_length = None


class OllamaChatClient:
    """Client for communicating with the Ollama chat API."""

    def __init__(self, config: Config, server_manager: OllamaServerManager, base_url: Optional[str] = None):
        """Initialize the chat client.

        Args:
            config: Application configuration
            server_manager: Server manager instance
            base_url: Optional override for API URL
        """
        self.config = config
        self.server_manager = server_manager
        self.current_model = None  # Track current model for cleanup
        self._base_url = base_url or config.api_url

        # Create session with authorization header if api_key is present
        self.session = requests.Session()
        if config.api_key:
            self.session.headers['Authorization'] = f'Bearer {config.api_key}'

    @property
    def api_url(self) -> str:
        """Get effective API URL, accounting for alternative port if set."""
        return self.server_manager._get_effective_api_url()

    def chat(
        self,
        convo: Conversation,
        new_user_message: str,
        streaming: bool = False,
        on_chunk=None,
        expanded_history: list = None
    ) -> str:
        """Send a chat message and get response from Ollama.

        Args:
            convo: Current conversation
            new_user_message: New message from user
            streaming: Whether to stream the response
            on_chunk: Optional callback accepting partial text (called for streaming)

        Returns:
            Assistant's response text

        Raises:
            ValueError: If model not found in config
            requests.RequestException: If HTTP request fails
        """
        if convo.provider_id != 'ollama':
            raise ValueError(f"Provider '{convo.provider_id}' not supported by Ollama client")
        # Look up model configuration
        model_cfg = self.config.get_model(convo.provider_id, convo.model_name)
        context_length = model_cfg.context or 8192  # fallback if not provided

        # Track current model for cleanup
        self.current_model = convo.model_name

        # Ensure server is running with correct context length
        self.server_manager.ensure_running(context_length)

        # Build messages array from conversation
        # Filter out 'echo', 'note', and 'status' messages - they're display-only
        messages = []
        history = expanded_history if expanded_history is not None else convo.messages
        for msg in history:
            if msg.role not in ('echo', 'note', 'status'):
                messages.append({
                    'role': msg.role,
                    'content': msg.content
                })

        # Add new user message
        messages.append({
            'role': 'user',
            'content': new_user_message
        })

        # Build request payload
        payload = {
            'model': convo.model_name,
            'messages': messages,
            'stream': False,
            'options': {
                'temperature': convo.temperature,
                'num_ctx': context_length  # Set context length via API
            }
        }

        # Send POST request
        url = f"{self.api_url.rstrip('/')}/chat"

        logger.debug(
            "POST %s model=%s stream=%s messages=%s temp=%s num_ctx=%s",
            url,
            payload.get('model'),
            payload.get('stream'),
            len(payload.get('messages', [])),
            payload['options'].get('temperature'),
            payload['options'].get('num_ctx'),
        )

        if streaming:
            payload['stream'] = True
            return self._chat_stream(url, payload, convo, context_length, on_chunk)
        else:
            return self._chat_single(url, payload, convo, context_length)

    def _chat_single(self, url, payload, convo, context_length):
        """Handle non-streaming chat."""
        try:
            response = self.session.post(url, json=payload, timeout=self.config.timeout)
            response.raise_for_status()
            logger.debug("Received successful response from Ollama API")
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Connection error to Ollama API at {url}: {e}")
            if self.server_manager.using_external_instance:
                raise ConnectionError(
                    f"Failed to connect to Ollama API at {url}. "
                    "The external Ollama instance may have stopped. "
                    "Please check that Ollama is still running."
                )
            else:
                raise ConnectionError(
                    f"Failed to connect to Ollama API at {url}. "
                    "The Ollama server may have failed to start. "
                    "Check that Ollama is installed and the port is available."
                )
        except requests.exceptions.Timeout as e:
            logger.error(f"Timeout error after {self.config.timeout} seconds: {e}")
            raise TimeoutError(
                f"Request to Ollama API timed out after {self.config.timeout} seconds. "
                "The model may be taking too long to respond. "
                "You can increase the timeout in your config.toml file or via /timeout."
            )
        except requests.exceptions.HTTPError as e:
            self._raise_http_error(e, response, url, payload)

        data = response.json()

        assistant_content = data.get('message', {}).get('content', '')
        tokens_in = data.get('prompt_eval_count', 0)
        tokens_out = data.get('eval_count', 0)

        logger.info(f"Response received: tokens_in={tokens_in}, tokens_out={tokens_out}, "
                    f"response_length={len(assistant_content)} chars")

        # Update conversation
        convo.tokens_in += tokens_in
        convo.tokens_out += tokens_out

        # Update context length tracking
        convo.context_length_configured = context_length
        convo.context_length_used = convo.tokens_in

        logger.debug(f"Conversation totals: tokens_in={convo.tokens_in}, "
                     f"tokens_out={convo.tokens_out}, context_used={convo.context_length_used}")

        # Append only assistant message (user message should already be in conversation)
        convo.messages.append(Message(role='assistant', content=assistant_content))

        return assistant_content

    def _chat_stream(self, url, payload, convo, context_length, on_chunk):
        """Handle streaming chat responses."""
        try:
            response = self.session.post(url, json=payload, timeout=self.config.timeout, stream=True)
            response.raise_for_status()
            logger.debug("Streaming response initiated from Ollama API")
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Connection error to Ollama API at {url}: {e}")
            if self.server_manager.using_external_instance:
                raise ConnectionError(
                    f"Failed to connect to Ollama API at {url}. "
                    "The external Ollama instance may have stopped. "
                    "Please check that Ollama is still running."
                )
            else:
                raise ConnectionError(
                    f"Failed to connect to Ollama API at {url}. "
                    "The Ollama server may have failed to start. "
                    "Check that Ollama is installed and the port is available."
                )
        except requests.exceptions.Timeout as e:
            logger.error(f"Timeout error after {self.config.timeout} seconds: {e}")
            raise TimeoutError(
                f"Request to Ollama API timed out after {self.config.timeout} seconds. "
                "The model may be taking too long to respond. "
                "You can increase the timeout in your config.toml file or via /timeout."
            )
        except requests.exceptions.HTTPError as e:
            self._raise_http_error(e, response, url, payload)

        assistant_msg = Message(role='assistant', content="")
        convo.messages.append(assistant_msg)

        total_tokens_in = 0
        total_tokens_out = 0

        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                logger.debug(f"Skipping non-JSON line from stream: {line}")
                continue

            content = data.get('message', {}).get('content', '')
            if content:
                assistant_msg.content += content
                if on_chunk:
                    try:
                        on_chunk(assistant_msg.content)
                    except Exception:
                        logger.debug("on_chunk callback raised but streaming will continue")

            if data.get('done'):
                total_tokens_in = data.get('prompt_eval_count', total_tokens_in)
                total_tokens_out = data.get('eval_count', total_tokens_out)
                break

        convo.tokens_in += total_tokens_in
        convo.tokens_out += total_tokens_out
        convo.context_length_configured = context_length
        convo.context_length_used = convo.tokens_in

        logger.info(f"Streaming response complete: tokens_in={total_tokens_in}, tokens_out={total_tokens_out}, "
                    f"response_length={len(assistant_msg.content)} chars")

        return assistant_msg.content

    def _raise_http_error(self, exc, response, url, payload):
        """Raise enriched HTTP error for Ollama responses."""
        error_details = {
            'status_code': response.status_code,
            'reason': response.reason,
            'url': url,
            'request_payload': payload
        }
        try:
            error_details['response_body'] = response.text
            response_json = response.json()
            error_details['response_json'] = response_json
        except Exception:
            error_details['response_body'] = response.text if hasattr(response, 'text') else 'Unable to read response'

        logger.error(f"HTTP error from Ollama API: {exc}")
        logger.error(f"Error details: {error_details}")

        error_msg = f"Ollama API returned an error: {exc}"
        if 'response_body' in error_details and error_details['response_body']:
            error_msg += f"\nResponse: {error_details['response_body']}"

        raise RuntimeError(error_msg)

    def unload_model(self) -> None:
        """Unload the current model from Ollama to free memory."""
        if not self.current_model:
            return

        logger.info(f"Unloading model: {self.current_model}")

        try:
            # Use the /api/generate endpoint with keep_alive=0 to unload
            url = f"{self.config.api_url.rstrip('/')}/generate"
            payload = {
                'model': self.current_model,
                'keep_alive': 0  # Unload immediately
            }
            self.session.post(url, json=payload, timeout=5)
            logger.info(f"Successfully unloaded model: {self.current_model}")
        except Exception as e:
            logger.warning(f"Error unloading model {self.current_model}: {e}")
            pass  # Ignore errors during cleanup
