"""Ollama API client and server management for lite-chat."""

import os
import subprocess
import time
from typing import Optional

import requests

from .config import Config
from .conversations import Conversation, Message


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

    def ensure_running(self, context_length: int) -> None:
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
            return

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

    def stop(self) -> None:
        """Stop the managed Ollama server process gracefully."""
        if self.current_process is None:
            return

        # Check if process is still running
        if self.current_process.poll() is not None:
            # Process already exited
            self.current_process = None
            self.current_context_length = None
            return

        # Try graceful termination first
        try:
            self.current_process.terminate()

            # Wait up to 5 seconds for process to terminate
            self.current_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            # Force kill if still alive
            self.current_process.kill()
            self.current_process.wait()
        except Exception as e:
            print(f"Error stopping Ollama process: {e}")

        self.current_process = None
        self.current_context_length = None


class OllamaChatClient:
    """Client for communicating with the Ollama chat API."""

    def __init__(self, config: Config, server_manager: OllamaServerManager):
        """Initialize the chat client.

        Args:
            config: Application configuration
            server_manager: Server manager instance
        """
        self.config = config
        self.server_manager = server_manager
        self.current_model = None  # Track current model for cleanup

        # Create session with authorization header if api_key is present
        self.session = requests.Session()
        if config.api_key:
            self.session.headers['Authorization'] = f'Bearer {config.api_key}'

    def chat(self, convo: Conversation, new_user_message: str) -> str:
        """Send a chat message and get response from Ollama.

        Args:
            convo: Current conversation
            new_user_message: New message from user

        Returns:
            Assistant's response text

        Raises:
            ValueError: If model not found in config
            requests.RequestException: If HTTP request fails
        """
        # Look up model configuration
        model_cfg = self.config.get_model(convo.model_name)
        context_length = model_cfg.contexts[0]

        # Track current model for cleanup
        self.current_model = convo.model_name

        # Ensure server is running with correct context length
        self.server_manager.ensure_running(context_length)

        # Build messages array from conversation
        messages = []
        for msg in convo.messages:
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
        url = f"{self.config.api_url.rstrip('/')}/chat"

        try:
            response = self.session.post(url, json=payload, timeout=120)
            response.raise_for_status()
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Failed to connect to Ollama API at {url}. "
                "Ensure Ollama is running and accessible."
            )
        except requests.exceptions.Timeout:
            raise TimeoutError(
                "Request to Ollama API timed out. The model may be taking too long to respond."
            )
        except requests.exceptions.HTTPError as e:
            raise RuntimeError(f"Ollama API returned an error: {e}")

        # Parse response
        data = response.json()

        assistant_content = data.get('message', {}).get('content', '')
        tokens_in = data.get('prompt_eval_count', 0)
        tokens_out = data.get('eval_count', 0)

        # Update conversation
        convo.tokens_in += tokens_in
        convo.tokens_out += tokens_out

        # Append only assistant message (user message should already be in conversation)
        convo.messages.append(Message(role='assistant', content=assistant_content))

        return assistant_content

    def unload_model(self) -> None:
        """Unload the current model from Ollama to free memory."""
        if not self.current_model:
            return

        try:
            # Use the /api/generate endpoint with keep_alive=0 to unload
            url = f"{self.config.api_url.rstrip('/')}/generate"
            payload = {
                'model': self.current_model,
                'keep_alive': 0  # Unload immediately
            }
            self.session.post(url, json=payload, timeout=5)
            print(f"Unloaded model: {self.current_model}")
        except Exception as e:
            print(f"Error unloading model: {e}")
            pass  # Ignore errors during cleanup
