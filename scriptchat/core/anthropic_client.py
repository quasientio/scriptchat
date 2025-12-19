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

"""Anthropic Claude chat client."""

import json
import logging
import os
from typing import Optional

import requests

from .config import Config, ProviderConfig
from .conversations import Conversation, Message
from .model_defaults import get_default_context_limit

logger = logging.getLogger(__name__)


def _resolve_api_key(provider: ProviderConfig) -> Optional[str]:
    """Resolve API key from config or environment variable.

    Checks in order:
    1. provider.api_key from config
    2. {PROVIDER_ID}_API_KEY env var (e.g., ANTHROPIC_API_KEY)
    """
    if provider.api_key:
        return provider.api_key

    env_var = f"{provider.id.upper()}_API_KEY"
    env_key = os.environ.get(env_var)
    if env_key:
        logger.debug("Using API key from %s environment variable", env_var)
        return env_key

    return None

# Mapping from reasoning levels to thinking budget tokens
# Values chosen to fit within Claude Sonnet 4.5's 64000 max_tokens limit
# (thinking budget + output tokens must not exceed model's max_tokens)
THINKING_BUDGET_PRESETS = {
    "low": 4000,
    "medium": 16000,
    "high": 32000,
    "max": 55000,
}


class AnthropicChatClient:
    """Client for Anthropic Claude API."""

    def __init__(self, config: Config, provider: ProviderConfig, timeout: int):
        self.config = config
        self.provider = provider
        self.timeout = timeout
        self.session = requests.Session()
        headers = {
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }
        api_key = _resolve_api_key(provider)
        if api_key:
            headers["x-api-key"] = api_key
        if provider.headers:
            headers.update(provider.headers)
        self.session.headers.update(headers)

    def chat(
        self,
        convo: Conversation,
        new_user_message: str,
        streaming: bool = False,
        on_chunk=None,
        expanded_history: list = None
    ) -> str:
        if convo.provider_id != self.provider.id:
            raise ValueError(f"Provider mismatch: expected {self.provider.id} got {convo.provider_id}")

        # Build messages array - system prompt handled separately in Anthropic
        messages_payload = []
        system_text = None
        history = expanded_history if expanded_history is not None else convo.messages
        include_thinking = getattr(self.config, 'include_thinking_in_history', False)

        for msg in history:
            if msg.role == 'system':
                # Capture system prompt, don't add to messages
                system_text = msg.content
            elif msg.role in ('user', 'assistant'):
                content = msg.content
                # Optionally include thinking content in history (disabled by default)
                if include_thinking and msg.thinking:
                    content = f"<thinking>\n{msg.thinking}\n</thinking>\n\n{content}"
                messages_payload.append({
                    "role": msg.role,
                    "content": content
                })
            # Skip 'echo', 'note', 'status', and other non-chat roles

        messages_payload.append({"role": "user", "content": new_user_message})

        # Use conversation's system_prompt if no system message in history
        if system_text is None and convo.system_prompt:
            system_text = convo.system_prompt

        # Build payload
        payload = {
            "model": convo.model_name,
            "max_tokens": 8192,  # Default max output tokens
            "messages": messages_payload,
        }

        # Add system prompt if present
        if system_text:
            payload["system"] = system_text

        # Handle extended thinking
        thinking_budget = self._get_thinking_budget(convo)
        if thinking_budget:
            payload["thinking"] = {
                "type": "enabled",
                "budget_tokens": thinking_budget
            }
            # When thinking is enabled, max_tokens must be > budget_tokens
            payload["max_tokens"] = max(payload["max_tokens"], thinking_budget + 4096)
            # Anthropic requires temperature=1 when thinking is enabled
            payload["temperature"] = 1.0
        else:
            # Add temperature (Anthropic supports 0.0-1.0, we clamp if needed)
            payload["temperature"] = min(1.0, convo.temperature)

        url = f"{self.provider.api_url.rstrip('/')}/v1/messages"

        if streaming and self.provider.streaming:
            payload["stream"] = True
            return self._chat_stream(url, payload, convo, on_chunk)
        return self._chat_single(url, payload, convo)

    def _get_thinking_budget(self, convo: Conversation) -> Optional[int]:
        """Get thinking budget from conversation settings."""
        # Check for explicit thinking_budget first
        thinking_budget = getattr(convo, 'thinking_budget', None)
        if thinking_budget:
            return thinking_budget

        # Map reasoning_level to preset budget
        reasoning_level = getattr(convo, 'reasoning_level', None)
        if reasoning_level and reasoning_level in THINKING_BUDGET_PRESETS:
            return THINKING_BUDGET_PRESETS[reasoning_level]

        return None

    def _http_error_with_body(self, resp: requests.Response, exc: requests.HTTPError) -> RuntimeError:
        """Build informative error from HTTP response."""
        body = ""
        error_type = None
        error_msg = None
        try:
            body = resp.text
            data = resp.json()
            err = data.get("error", {})
            error_type = err.get("type")
            error_msg = err.get("message")
        except Exception:
            pass

        msg_parts = [f"HTTP error {resp.status_code}: {resp.reason}"]
        if error_msg:
            msg_parts.append(f"Message: {error_msg}")
        if error_type:
            msg_parts.append(f"Type: {error_type}")
        if body and not error_msg:
            msg_parts.append(f"Body: {body[:500]}")

        err = RuntimeError("; ".join(msg_parts))
        try:
            err.__cause__ = exc
        except Exception:
            pass
        return err

    def _log_response_metadata(self, resp, data: dict, usage: dict) -> None:
        """Log response metadata without message content."""
        headers = getattr(resp, 'headers', None) or {}
        interesting_headers = {
            k: v for k, v in headers.items()
            if any(x in k.lower() for x in ['x-request', 'anthropic', 'cf-ray', 'retry'])
        }
        metadata = {
            "status": getattr(resp, 'status_code', None),
            "model": data.get("model"),
            "id": data.get("id"),
            "type": data.get("type"),
            "usage": usage,
            "stop_reason": data.get("stop_reason"),
            "headers": interesting_headers,
        }
        metadata = {k: v for k, v in metadata.items() if v}
        logger.debug("Response metadata: %s", metadata)

    def _chat_single(self, url: str, payload: dict, convo: Conversation) -> str:
        """Non-streaming chat request."""
        timeout = getattr(self.config, "timeout", None) or self.timeout
        resp = None

        try:
            logger.debug(
                "POST %s model=%s messages=%d thinking=%s temp=%s",
                url,
                payload.get("model"),
                len(payload.get("messages", [])),
                payload.get("thinking"),
                payload.get("temperature"),
            )
            resp = self.session.post(url, json=payload, timeout=timeout)
            resp.raise_for_status()
        except requests.Timeout as e:
            raise TimeoutError(
                f"Request timed out after {timeout} seconds. "
                "Increase the timeout in config.toml or with /timeout."
            ) from e
        except requests.ConnectionError as e:
            raise ConnectionError(
                f"Failed to connect to Anthropic API at {self.provider.api_url}. "
                "Check your network or API URL."
            ) from e
        except requests.HTTPError as e:
            raise self._http_error_with_body(resp, e)

        data = resp.json()

        # Extract content from Anthropic response format
        content, thinking_content = self._extract_content(data)
        if thinking_content:
            logger.debug("Captured %d chars of thinking content", len(thinking_content))

        # Track tokens
        usage = data.get("usage", {})

        self._log_response_metadata(resp, data, usage)

        # Update conversation
        convo.messages.append(Message(role='assistant', content=content, thinking=thinking_content))
        convo.tokens_in += usage.get("input_tokens", 0)
        convo.tokens_out += usage.get("output_tokens", 0)

        # Update context tracking (config override > built-in defaults)
        model_cfg = self.config.get_model(convo.provider_id, convo.model_name)
        context_length = model_cfg.context or get_default_context_limit(convo.model_name)
        if context_length:
            convo.context_length_configured = context_length
            convo.context_length_used = convo.tokens_in

        return content

    def _chat_stream(self, url: str, payload: dict, convo: Conversation, on_chunk=None) -> str:
        """Streaming chat request with SSE parsing."""
        timeout = getattr(self.config, "timeout", None) or self.timeout
        resp = None

        try:
            logger.debug(
                "POST %s model=%s messages=%d thinking=%s temp=%s stream=True",
                url,
                payload.get("model"),
                len(payload.get("messages", [])),
                payload.get("thinking"),
                payload.get("temperature"),
            )
            resp = self.session.post(url, json=payload, timeout=timeout, stream=True)
            resp.raise_for_status()
        except requests.Timeout as e:
            raise TimeoutError(
                f"Request timed out after {timeout} seconds. "
                "Increase the timeout in config.toml or with /timeout."
            ) from e
        except requests.ConnectionError as e:
            raise ConnectionError(
                f"Failed to connect to Anthropic API at {self.provider.api_url}. "
                "Check your network or API URL."
            ) from e
        except requests.HTTPError as e:
            raise self._http_error_with_body(resp, e)

        assistant_msg = Message(role='assistant', content="")
        convo.messages.append(assistant_msg)

        total_input_tokens = 0
        total_output_tokens = 0
        current_event = None
        thinking_content = ""  # Accumulate thinking blocks

        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue

            # Parse SSE format
            if line.startswith('event: '):
                current_event = line[7:]
                continue

            if not line.startswith('data: '):
                continue

            data_str = line[6:]
            try:
                data = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            event_type = data.get('type', current_event)

            if event_type == 'message_start':
                # Initial message with usage info
                message = data.get('message', {})
                usage = message.get('usage', {})
                total_input_tokens = usage.get('input_tokens', 0)

            elif event_type == 'content_block_start':
                # Log content block starts to identify thinking blocks
                block = data.get('content_block', {})
                logger.debug("content_block_start type=%s", block.get('type'))

            elif event_type == 'content_block_delta':
                # Text content delta
                delta = data.get('delta', {})
                delta_type = delta.get('type')
                # Log all delta types to help debug thinking capture
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug("content_block_delta type=%s keys=%s", delta_type, list(delta.keys()))
                if delta_type == 'text_delta':
                    text = delta.get('text', '')
                    if text:
                        assistant_msg.content += text
                        if on_chunk:
                            try:
                                on_chunk(assistant_msg.content)
                            except Exception:
                                pass
                # Handle thinking blocks (extended thinking)
                elif delta_type == 'thinking_delta':
                    thinking_text = delta.get('thinking', '')
                    if thinking_text:
                        thinking_content += thinking_text
                        # Update message thinking in real-time for streaming display
                        assistant_msg.thinking = thinking_content
                        if on_chunk:
                            try:
                                on_chunk(assistant_msg.content)
                            except Exception:
                                pass

            elif event_type == 'message_delta':
                # Final usage stats
                usage = data.get('usage', {})
                total_output_tokens = usage.get('output_tokens', 0)

            elif event_type == 'message_stop':
                # Stream complete
                break

            elif event_type == 'error':
                error = data.get('error', {})
                raise RuntimeError(f"Anthropic API error: {error.get('message', 'Unknown error')}")

        convo.tokens_in += total_input_tokens
        convo.tokens_out += total_output_tokens

        # Set thinking content if captured
        if thinking_content:
            assistant_msg.thinking = thinking_content
            logger.debug("Captured %d chars of thinking content", len(thinking_content))

        # Update context tracking (config override > built-in defaults)
        model_cfg = self.config.get_model(convo.provider_id, convo.model_name)
        context_length = model_cfg.context or get_default_context_limit(convo.model_name)
        if context_length:
            convo.context_length_configured = context_length
            convo.context_length_used = convo.tokens_in

        # Log response metadata from streaming
        final_usage = {"input_tokens": total_input_tokens, "output_tokens": total_output_tokens}
        self._log_response_metadata(resp, {}, final_usage)

        return assistant_msg.content

    def _extract_content(self, data: dict) -> tuple[str, str | None]:
        """Extract text and thinking content from Anthropic response.

        Returns:
            Tuple of (text_content, thinking_content or None)
        """
        content_blocks = data.get('content', [])
        text_parts = []
        thinking_parts = []

        for block in content_blocks:
            if block.get('type') == 'text':
                text_parts.append(block.get('text', ''))
            elif block.get('type') == 'thinking':
                thinking_parts.append(block.get('thinking', ''))

        thinking = ''.join(thinking_parts) if thinking_parts else None
        return ''.join(text_parts), thinking
