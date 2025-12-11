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
            headers["Authorization"] = f"Bearer {api_key}"
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

        # Use Responses API if explicitly configured, or default for "openai" provider
        use_responses_api = (
            self.provider.api_format == "responses" or
            (self.provider.api_format is None and self.provider.id == "openai")
        )

        # Build messages array, filtering out display-only messages
        messages_payload = []
        history = expanded_history if expanded_history is not None else convo.messages
        for msg in history:
            if msg.role not in ('echo', 'note', 'status'):
                messages_payload.append({
                    "role": msg.role,
                    "content": msg.content
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
            url = f"{self.provider.api_url.rstrip('/')}/v1/responses"
        else:
            payload = {
                "model": convo.model_name,
                "messages": messages_payload,
                "stream": streaming and self.provider.streaming,
                "temperature": convo.temperature,
            }
            if getattr(convo, "reasoning_level", None):
                payload["reasoning"] = {"effort": convo.reasoning_level}
            url = f"{self.provider.api_url.rstrip('/')}/v1/chat/completions"

        # Disable prompt caching if configured (for privacy)
        if not self.provider.prompt_cache:
            payload["prompt_cache_max_len"] = 0

        if streaming:
            return self._chat_stream(url, payload, convo, on_chunk)
        return self._chat_single(url, payload, convo)

    def _extract_responses_text(self, data: dict) -> str:
        """Extract text from Responses API payloads (non-stream or stream chunk)."""
        if not isinstance(data, dict):
            return ""
        # Prefer aggregated output_text when present
        if data.get("output_text"):
            try:
                return "".join(data.get("output_text") or [])
            except Exception:
                pass
        text_parts: list[str] = []
        # Some payloads embed content in output -> content list
        for item in data.get("output", []) or []:
            for content in item.get("content", []) or []:
                if isinstance(content, dict):
                    text = content.get("text")
                    if text:
                        text_parts.append(str(text))
        # Streaming deltas may arrive under "delta" similarly
        delta = data.get("delta") or {}
        if isinstance(delta, dict):
            for content in delta.get("content", []) or []:
                if isinstance(content, dict):
                    text = content.get("text")
                    if text:
                        text_parts.append(str(text))
        elif isinstance(delta, str):
            text_parts.append(delta)
        return "".join(text_parts)

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
        if "choices" in data:
            content = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
        else:
            # Responses API
            content = self._extract_responses_text(data)
            usage = data.get("usage", {})

        self._log_response_metadata(resp, data, usage)

        # Update conversation
        convo.messages.append(Message(role='assistant', content=content))
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
            if "choices" in data:
                delta = data.get("choices", [{}])[0].get("delta", {})
                content_piece = delta.get("content", "")
                usage = data.get("usage") or {}
                # Chat Completions API uses prompt_tokens/completion_tokens
                total_prompt += usage.get("prompt_tokens", 0)
                total_completion += usage.get("completion_tokens", 0)
                if usage:
                    usage_added = True
            else:
                # Responses API streaming - uses input_tokens/output_tokens
                content_piece = self._extract_responses_text(data)
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

        return assistant_msg.content

    def _post_with_temperature_retry(self, url: str, payload: dict, stream: bool, convo: Conversation | None) -> requests.Response:
        """Post chat payload; retry once without temperature if model rejects it."""
        timeout = getattr(self.config, "timeout", None) or self.timeout
        resp = None
        try:
            logger.debug(
                "POST %s model=%s stream=%s messages=%s reasoning=%s temp=%s",
                url,
                payload.get("model"),
                payload.get("stream"),
                len(payload.get("messages", []) or payload.get("input", [])),
                payload.get("reasoning"),
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
