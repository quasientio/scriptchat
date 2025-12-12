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

"""Ollama API client for ScriptChat."""

import json
import logging
from typing import Optional

import requests

from .config import Config
from .conversations import Conversation, Message

logger = logging.getLogger(__name__)


def check_ollama_running(api_url: str, timeout: float = 2.0) -> bool:
    """Check if Ollama is running at the given URL.

    Args:
        api_url: Base URL for the Ollama API
        timeout: Request timeout in seconds

    Returns:
        True if Ollama is responding, False otherwise
    """
    check_url = f"{api_url.rstrip('/')}/tags"
    try:
        response = requests.get(check_url, timeout=timeout)
        return response.status_code == 200
    except Exception:
        return False


class OllamaChatClient:
    """Client for communicating with the Ollama chat API."""

    def __init__(self, config: Config, api_url: str, interactive: bool = True):
        """Initialize the chat client.

        Args:
            config: Application configuration
            api_url: Base URL for Ollama API
            interactive: Whether running in interactive mode (shows warnings)
        """
        self.config = config
        self.api_url = api_url
        self.current_model = None  # Track current model for cleanup

        # Create session with authorization header if api_key is present
        self.session = requests.Session()
        if config.api_key:
            self.session.headers['Authorization'] = f'Bearer {config.api_key}'

        # Check if Ollama is running on startup (warn but don't block)
        if interactive and not check_ollama_running(api_url):
            print(f"\nWarning: Ollama is not running at {api_url}")
            print("Start it with: ollama serve\n")

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
            expanded_history: Optional pre-expanded message history

        Returns:
            Assistant's response text

        Raises:
            ValueError: If provider doesn't match
            ConnectionError: If Ollama is not reachable
            TimeoutError: If request times out
            RuntimeError: If HTTP error occurs
        """
        if convo.provider_id != 'ollama':
            raise ValueError(f"Provider '{convo.provider_id}' not supported by Ollama client")

        # Look up model configuration
        model_cfg = self.config.get_model(convo.provider_id, convo.model_name)
        context_length = model_cfg.context or 8192  # fallback if not provided

        # Track current model for cleanup
        self.current_model = convo.model_name

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

        # Build request payload with num_ctx to set context length per-request
        payload = {
            'model': convo.model_name,
            'messages': messages,
            'stream': False,
            'options': {
                'temperature': convo.temperature,
                'num_ctx': context_length
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
                "Please check that Ollama is running (ollama serve)."
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
                "Please check that Ollama is running (ollama serve)."
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
            url = f"{self.api_url.rstrip('/')}/generate"
            payload = {
                'model': self.current_model,
                'keep_alive': 0  # Unload immediately
            }
            self.session.post(url, json=payload, timeout=5)
            logger.info(f"Successfully unloaded model: {self.current_model}")
        except Exception as e:
            logger.warning(f"Error unloading model {self.current_model}: {e}")
            pass  # Ignore errors during cleanup

    def stop(self) -> None:
        """No-op for compatibility. Ollama server is managed externally."""
        pass
