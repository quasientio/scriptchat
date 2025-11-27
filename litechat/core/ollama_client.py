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

"""Ollama API client and server management for lite-chat."""

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

    def __init__(self, api_url: str):
        """Initialize the server manager.

        Args:
            api_url: Base URL for the Ollama API
        """
        self.api_url = api_url
        self.current_process: Optional[subprocess.Popen] = None
        self.current_context_length: Optional[int] = None

    def ensure_running(self, context_length: int) -> None:  # pragma: no cover - spawns external server
        """Ensure Ollama server is running with the specified context length.

        If the server is already running with a different context length,
        it will be stopped and restarted.

        Args:
            context_length: Context length to set via OLLAMA_CONTEXT_LENGTH
        """
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

        try:
            self.current_process = subprocess.Popen(
                ['ollama', 'serve'],
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            self.current_context_length = context_length

            # Wait for Ollama to be ready (up to 10 seconds)
            import requests
            health_check_url = f"{self.api_url.rstrip('/')}/tags"
            for _ in range(20):
                time.sleep(0.5)
                try:
                    requests.get(health_check_url, timeout=1)
                    break  # Server is ready
                except Exception:
                    continue

        except FileNotFoundError:
            raise FileNotFoundError(
                "ollama executable not found. Please ensure Ollama is installed and in your PATH."
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
        self.api_url = base_url or config.api_url

        # Create session with authorization header if api_key is present
        self.session = requests.Session()
        if config.api_key:
            self.session.headers['Authorization'] = f'Bearer {config.api_key}'

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
        context_length = model_cfg.contexts[0] if model_cfg.contexts else None
        if context_length is None:
            context_length = 8192  # fallback if not provided

        # Track current model for cleanup
        self.current_model = convo.model_name

        # Ensure server is running with correct context length
        self.server_manager.ensure_running(context_length)

        # Build messages array from conversation
        # Filter out 'echo' messages - they're display-only
        messages = []
        history = expanded_history if expanded_history is not None else convo.messages
        for msg in history:
            if msg.role != 'echo':
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
            raise ConnectionError(
                f"Failed to connect to Ollama API at {url}. "
                "Ensure Ollama is running and accessible."
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
            raise ConnectionError(
                f"Failed to connect to Ollama API at {url}. "
                "Ensure Ollama is running and accessible."
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
