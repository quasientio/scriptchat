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

"""OpenAI-compatible chat client."""

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
    2. {PROVIDER_ID}_API_KEY env var (e.g., OPENAI_API_KEY, DEEPSEEK_API_KEY)
    """
    if provider.api_key:
        return provider.api_key

    env_var = f"{provider.id.upper()}_API_KEY"
    env_key = os.environ.get(env_var)
    if env_key:
        logger.debug("Using API key from %s environment variable", env_var)
        return env_key

    return None


class OpenAIChatClient:
    """Client for OpenAI-compatible chat completions."""

    def __init__(self, config: Config, provider: ProviderConfig, timeout: int):
        self.config = config
        self.provider = provider
        self.timeout = timeout  # fallback; config.timeout preferred
        self.session = requests.Session()
        headers = {
            "Content-Type": "application/json",
        }
        api_key = _resolve_api_key(provider)
        if api_key:
            auth_format = getattr(provider, 'auth_format', 'bearer')
            if auth_format == "api-key":
                headers["Authorization"] = f"Api-Key {api_key}"
            else:
                headers["Authorization"] = f"Bearer {api_key}"
        if provider.headers:
            headers.update(provider.headers)
        self.session.headers.update(headers)

    # Common stop tokens that may leak through from various models
    STOP_TOKENS = [
        "<|im_end|>",
        "<|endoftext|>",
        "<|im_start|>",
        "<|end|>",
        "</s>",
        "<|eot_id|>",
    ]

    def _strip_stop_tokens(self, content: str) -> str:
        """Strip common stop tokens from the end of model output."""
        if not content:
            return content
        for token in self.STOP_TOKENS:
            if content.endswith(token):
                content = content[:-len(token)].rstrip()
        return content

    def _extract_think_tags(self, content: str) -> tuple[str, str | None]:
        """Extract and strip <think>...</think> tags from content (DeepSeek R1 format).

        Args:
            content: The response content that may contain think tags

        Returns:
            Tuple of (content_without_thinking, thinking_content or None)
        """
        import re
        # Match <think>...</think> block (case insensitive, dotall for multiline)
        pattern = re.compile(r'<think>(.*?)</think>\s*', re.IGNORECASE | re.DOTALL)
        match = pattern.search(content)
        if match:
            thinking = match.group(1).strip()
            # Remove the think block from content
            content_clean = pattern.sub('', content).strip()
            return content_clean, thinking
        return content, None

    def _build_url(self, endpoint: str) -> str:
        """Build URL for API endpoint, avoiding duplicate version prefixes.

        If api_url already ends with a version (e.g., /v1, /v2), and endpoint
        starts with the same version, don't duplicate it.

        Args:
            endpoint: The endpoint path (e.g., "/v1/chat/completions")

        Returns:
            Full URL with api_url and endpoint properly joined
        """
        import re
        base = self.provider.api_url.rstrip('/')
        # Check if base ends with version like /v1, /v2, etc.
        version_match = re.search(r'/v\d+$', base)
        if version_match:
            base_version = version_match.group()  # e.g., "/v1"
            if endpoint.startswith(base_version):
                # Remove version from endpoint to avoid duplication
                endpoint = endpoint[len(base_version):]
        return f"{base}{endpoint}"

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

        # Use Responses API if explicitly configured, or default for "openai" provider
        use_responses_api = (
            self.provider.api_format == "responses" or
            (self.provider.api_format is None and self.provider.id == "openai")
        )

        # Build messages array, filtering out display-only messages
        messages_payload = []
        history = expanded_history if expanded_history is not None else convo.messages
        include_thinking = getattr(self.config, 'include_thinking_in_history', False)
        for msg in history:
            if msg.role not in ('echo', 'note', 'status'):
                content = msg.content
                # Optionally include thinking content in history (disabled by default)
                if include_thinking and msg.thinking:
                    content = f"<thinking>\n{msg.thinking}\n</thinking>\n\n{content}"
                messages_payload.append({
                    "role": msg.role,
                    "content": content
                })
        messages_payload.append({"role": "user", "content": new_user_message})

        if use_responses_api:
            payload = {
                "model": convo.model_name,
                "input": messages_payload,
                "stream": streaming and self.provider.streaming,
                "temperature": convo.temperature,
                "store": False,
            }
            if getattr(convo, "reasoning_level", None):
                payload["reasoning"] = {"effort": convo.reasoning_level}
            url = self._build_url("/v1/responses")
        else:
            payload = {
                "model": convo.model_name,
                "messages": messages_payload,
                "stream": streaming and self.provider.streaming,
                "temperature": convo.temperature,
            }
            if getattr(convo, "reasoning_level", None):
                # Fireworks and other providers use reasoning_effort as top-level param
                # OpenAI Responses API uses reasoning.effort (handled above)
                payload["reasoning_effort"] = convo.reasoning_level
            url = self._build_url("/v1/chat/completions")

        # Set max_tokens from model config (important for thinking models)
        model_cfg = self.config.get_model(convo.provider_id, convo.model_name)
        if model_cfg.max_tokens:
            payload["max_tokens"] = model_cfg.max_tokens
            # Some providers (e.g. Fireworks) require streaming for high max_tokens
            if model_cfg.max_tokens > 4096 and not streaming:
                streaming = True
                payload["stream"] = True
                logger.debug("Forcing streaming due to max_tokens > 4096")

        # Disable prompt caching if configured (for privacy)
        # Skip if model doesn't support the parameter (e.g., Kimi)
        model_cfg = self.config.get_model(convo.provider_id, convo.model_name)
        if not self.provider.prompt_cache and not model_cfg.skip_prompt_cache_param:
            payload["prompt_cache_max_len"] = 0

        if streaming:
            return self._chat_stream(url, payload, convo, on_chunk)
        return self._chat_single(url, payload, convo)

    def _extract_responses_text(self, data: dict) -> tuple[str, str | None]:
        """Extract text and reasoning from Responses API payloads.

        Returns:
            Tuple of (text_content, reasoning_content or None)
        """
        if not isinstance(data, dict):
            return "", None

        text_parts: list[str] = []
        reasoning = None

        # Log top-level keys to help debug reasoning extraction
        if logger.isEnabledFor(logging.DEBUG):
            interesting_keys = [k for k in data.keys() if k not in ('id', 'created', 'object')]
            if interesting_keys:
                logger.debug("Responses API payload keys: %s", interesting_keys)

        # Extract reasoning.summary if present (OpenAI Responses API format)
        reasoning_obj = data.get("reasoning")
        if isinstance(reasoning_obj, dict):
            summary_value = reasoning_obj.get("summary")
            logger.debug("Found reasoning obj: keys=%s, summary=%r", list(reasoning_obj.keys()), summary_value[:100] if summary_value else summary_value)
            if summary_value:
                reasoning = summary_value
                logger.debug("Extracted reasoning summary: %d chars", len(reasoning))

        # Prefer aggregated output_text when present
        if data.get("output_text"):
            try:
                return "".join(data.get("output_text") or []), reasoning
            except Exception:
                pass

        # Some payloads embed content in output -> content list
        for item in data.get("output", []) or []:
            for content in item.get("content", []) or []:
                if isinstance(content, dict):
                    text = content.get("text")
                    if text:
                        text_parts.append(str(text))

        # Streaming deltas may arrive under "delta" similarly
        delta = data.get("delta")
        if isinstance(delta, dict):
            for content in delta.get("content", []) or []:
                if isinstance(content, dict):
                    text = content.get("text")
                    if text:
                        text_parts.append(str(text))
        elif isinstance(delta, str):
            # Direct string delta (e.g., response.output_text.delta event)
            text_parts.append(delta)

        return "".join(text_parts), reasoning

    def _http_error_with_body(self, resp, exc: requests.HTTPError) -> RuntimeError:
        body = ""
        code = None
        param = None
        detail_msg = None
        try:
            body = resp.text
            data = resp.json()
            err = data.get("error", {})
            detail_msg = err.get("message")
            code = err.get("code")
            param = err.get("param")
        except Exception:
            pass

        msg_parts = [f"HTTP error {resp.status_code}: {resp.reason}"]
        if detail_msg:
            msg_parts.append(f"Message: {detail_msg}")
        if code:
            msg_parts.append(f"Code: {code}")
        if param:
            msg_parts.append(f"Param: {param}")
        if body:
            msg_parts.append(f"Body: {body}")

        err = RuntimeError("; ".join(msg_parts))
        try:
            err.__cause__ = exc
        except Exception:
            pass
        return err

    def _log_response_metadata(self, resp: requests.Response, data: dict, usage: dict) -> None:
        """Log response metadata without message content."""
        # Filter interesting headers (rate limits, request IDs, timing)
        headers = getattr(resp, 'headers', None) or {}
        interesting_headers = {
            k: v for k, v in headers.items()
            if any(x in k.lower() for x in ['x-request', 'x-ratelimit', 'openai', 'cf-ray', 'x-envoy'])
        }
        # Build metadata dict excluding actual content
        metadata = {
            "status": getattr(resp, 'status_code', None),
            "model": data.get("model"),
            "id": data.get("id"),
            "created": data.get("created"),
            "usage": usage,
            "system_fingerprint": data.get("system_fingerprint"),
            "headers": interesting_headers,
        }
        # Remove None/empty values
        metadata = {k: v for k, v in metadata.items() if v}
        logger.debug("Response metadata: %s", metadata)

    def _chat_single(self, url: str, payload: dict, convo: Conversation) -> str:
        resp = self._post_with_temperature_retry(url, payload, stream=False, convo=convo)
        data = resp.json()
        thinking_content = None
        if "choices" in data and data["choices"]:
            message = data["choices"][0]["message"]
            content = message.get("content") or ""
            # Capture reasoning content - some providers use 'reasoning_content', others use 'reasoning'
            reasoning_content = message.get("reasoning_content") or message.get("reasoning")
            if reasoning_content:
                thinking_content = reasoning_content
                logger.debug("Captured %d chars of thinking content", len(reasoning_content))
            # Log message keys when content is empty (helps debug)
            if not content:
                logger.warning("Empty content in response. Message keys: %s", list(message.keys()))
                logger.debug("Full message structure: %s", message)
            usage = data.get("usage", {})
        else:
            # Responses API
            content, responses_reasoning = self._extract_responses_text(data)
            if responses_reasoning and not thinking_content:
                thinking_content = responses_reasoning
                logger.debug("Captured %d chars of reasoning from Responses API", len(responses_reasoning))
            usage = data.get("usage", {})

        self._log_response_metadata(resp, data, usage)

        # Strip any leaked stop tokens
        content = self._strip_stop_tokens(content)

        # Extract <think>...</think> tags (DeepSeek R1 format) if no reasoning_content
        if not thinking_content:
            content, think_tag_content = self._extract_think_tags(content)
            if think_tag_content:
                thinking_content = think_tag_content
                logger.debug("Extracted %d chars of thinking from <think> tags", len(think_tag_content))

        # Update conversation
        convo.messages.append(Message(role='assistant', content=content, thinking=thinking_content))
        # Chat Completions API uses prompt_tokens/completion_tokens
        # Responses API uses input_tokens/output_tokens
        convo.tokens_in += usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0)
        convo.tokens_out += usage.get("completion_tokens", 0) or usage.get("output_tokens", 0)

        # Update context tracking (config override > built-in defaults)
        model_cfg = self.config.get_model(convo.provider_id, convo.model_name)
        context_length = model_cfg.context or get_default_context_limit(convo.model_name)
        if context_length:
            convo.context_length_configured = context_length
            convo.context_length_used = convo.tokens_in

        return content

    def _chat_stream(self, url: str, payload: dict, convo: Conversation, on_chunk=None) -> str:
        resp = self._post_with_temperature_retry(
            url,
            payload,
            stream=bool(payload.get("stream", False)),
            convo=convo,
        )
        assistant_msg = Message(role='assistant', content="")
        convo.messages.append(assistant_msg)

        last_data = None
        total_prompt = 0
        total_completion = 0
        usage_added = False
        thinking_content = ""  # Accumulate reasoning_content separately
        for line in resp.iter_lines():
            if not line:
                continue
            if line.startswith(b"data: "):
                line = line[len(b"data: "):]
            if line == b"[DONE]":
                break
            try:
                data = json.loads(line)
                last_data = data
            except json.JSONDecodeError:
                continue
            # Log streaming event type for debugging
            event_type = data.get("type") or data.get("object")
            if event_type and logger.isEnabledFor(logging.DEBUG):
                logger.debug("Stream event type: %s", event_type)

            if "choices" in data and data["choices"]:
                delta = data["choices"][0].get("delta", {})
                content_piece = delta.get("content", "")
                # Capture reasoning content - some providers use 'reasoning_content', others use 'reasoning'
                reasoning_piece = delta.get("reasoning_content") or delta.get("reasoning") or ""
                if reasoning_piece:
                    thinking_content += reasoning_piece
                    # Update message thinking in real-time for streaming display
                    assistant_msg.thinking = thinking_content
                    if on_chunk:
                        try:
                            on_chunk(assistant_msg.content)
                        except Exception:
                            pass
                # Debug log delta keys to help identify thinking model formats
                if delta and logger.isEnabledFor(logging.DEBUG):
                    delta_keys = list(delta.keys())
                    # Log full delta if it has unexpected keys (helps identify new thinking formats)
                    unexpected = set(delta_keys) - {'role', 'content', 'reasoning_content', 'reasoning'}
                    if unexpected:
                        logger.debug("Stream delta with unexpected keys %s: %s", unexpected, delta)
                    elif delta_keys:
                        logger.debug("Stream delta keys: %s", delta_keys)
                usage = data.get("usage") or {}
                # Chat Completions API uses prompt_tokens/completion_tokens
                total_prompt += usage.get("prompt_tokens", 0)
                total_completion += usage.get("completion_tokens", 0)
                if usage:
                    usage_added = True
            else:
                # Responses API streaming - uses input_tokens/output_tokens
                content_piece, responses_reasoning = self._extract_responses_text(data)
                if responses_reasoning:
                    thinking_content += responses_reasoning
                usage = data.get("usage") or {}
                total_prompt += usage.get("input_tokens", 0)
                total_completion += usage.get("output_tokens", 0)
                if usage:
                    usage_added = True
            if content_piece:
                assistant_msg.content += content_piece
                if on_chunk:
                    try:
                        on_chunk(assistant_msg.content)
                    except Exception:
                        pass

        if not usage_added and last_data:
            usage = last_data.get("usage", {})
            # Handle both Chat Completions and Responses API token keys
            total_prompt += usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0)
            total_completion += usage.get("completion_tokens", 0) or usage.get("output_tokens", 0)
        convo.tokens_in += total_prompt
        convo.tokens_out += total_completion

        # Update context tracking (config override > built-in defaults)
        model_cfg = self.config.get_model(convo.provider_id, convo.model_name)
        context_length = model_cfg.context or get_default_context_limit(convo.model_name)
        if context_length:
            convo.context_length_configured = context_length
            convo.context_length_used = convo.tokens_in

        # Log response metadata from streaming (use last_data for model/id info)
        final_usage = {"prompt_tokens": total_prompt, "completion_tokens": total_completion}
        self._log_response_metadata(resp, last_data or {}, final_usage)

        # Strip any leaked stop tokens from final content
        assistant_msg.content = self._strip_stop_tokens(assistant_msg.content)

        # Extract <think>...</think> tags (DeepSeek R1 format) if no reasoning_content
        if not thinking_content:
            original_content = assistant_msg.content
            assistant_msg.content, think_tag_content = self._extract_think_tags(assistant_msg.content)
            if think_tag_content:
                thinking_content = think_tag_content
                logger.debug("Extracted %d chars of thinking from <think> tags", len(think_tag_content))
                logger.debug("Content after think extraction: %d chars", len(assistant_msg.content))
                if not assistant_msg.content:
                    logger.warning("Content is empty after <think> extraction. Original had %d chars", len(original_content))

        # Set thinking content if captured
        if thinking_content:
            assistant_msg.thinking = thinking_content
            logger.debug("Captured %d chars of thinking content", len(thinking_content))
            logger.debug("Message thinking attr after set: %s (msg id: %s)",
                        len(assistant_msg.thinking) if assistant_msg.thinking else "None",
                        id(assistant_msg))
            logger.debug("convo.messages[-1] thinking: %s (msg id: %s)",
                        len(convo.messages[-1].thinking) if convo.messages[-1].thinking else "None",
                        id(convo.messages[-1]))

        # Warn if both content and thinking are empty
        if not assistant_msg.content and not thinking_content:
            logger.warning("Both content and thinking are empty after processing")

        # Log first 200 chars of content to help identify thinking markers
        logger.debug("Final content starts with: %r", assistant_msg.content[:200] if len(assistant_msg.content) > 200 else assistant_msg.content)

        return assistant_msg.content

    def _post_with_temperature_retry(self, url: str, payload: dict, stream: bool, convo: Conversation | None) -> requests.Response:
        """Post chat payload; retry once without temperature if model rejects it."""
        timeout = getattr(self.config, "timeout", None) or self.timeout
        resp = None
        try:
            logger.debug(
                "POST %s model=%s stream=%s messages=%s reasoning=%s reasoning_effort=%s temp=%s",
                url,
                payload.get("model"),
                payload.get("stream"),
                len(payload.get("messages", []) or payload.get("input", [])),
                payload.get("reasoning"),
                payload.get("reasoning_effort"),
                payload.get("temperature"),
            )
            send_payload = dict(payload)
            send_payload.pop("_convo_ref", None)
            resp = self.session.post(url, json=send_payload, timeout=timeout, stream=stream)
            resp.raise_for_status()
            return resp
        except requests.Timeout as e:
            raise TimeoutError(
                f"Request timed out after {timeout} seconds. "
                "Increase the timeout in config.toml or with /timeout."
            ) from e
        except requests.ConnectionError as e:
            raise ConnectionError(
                f"Failed to connect to provider at {self.provider.api_url}. "
                "Check your network or provider URL, or increase the timeout with /timeout."
            ) from e
        except requests.HTTPError as e:
            # Check for temperature error, retry without temperature
            if resp is not None and self._is_temperature_error(resp):
                payload_no_temp = dict(payload)
                payload_no_temp.pop("temperature", None)
                logger.info("Retrying without temperature for model %s due to temperature error", payload.get("model"))
                logger.debug(
                    "POST %s model=%s stream=%s messages=%s reasoning=%s temp=%s (retry sans temp)",
                    url,
                    payload_no_temp.get("model"),
                    payload_no_temp.get("stream"),
                    len(payload_no_temp.get("messages", [])),
                    payload_no_temp.get("reasoning"),
                    payload_no_temp.get("temperature"),
                )
                resp = self.session.post(url, json=payload_no_temp, timeout=timeout, stream=stream)
                resp.raise_for_status()
                return resp
            raise self._http_error_with_body(resp, e)

    def _is_temperature_error(self, resp: requests.Response) -> bool:
        try:
            data = resp.json()
            err = data.get("error", {})
            if err.get("param") == "temperature":
                return True
        except Exception:
            return False
        return False
