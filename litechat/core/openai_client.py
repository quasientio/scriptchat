"""OpenAI-compatible chat client."""

import json
import logging
from typing import Optional

import requests

from .config import Config, ProviderConfig
from .conversations import Conversation, Message

logger = logging.getLogger(__name__)


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
        if provider.api_key:
            headers["Authorization"] = f"Bearer {provider.api_key}"
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

        messages_payload = []
        history = expanded_history if expanded_history is not None else convo.messages
        for msg in history:
            messages_payload.append({
                "role": msg.role,
                "content": msg.content
            })
        messages_payload.append({"role": "user", "content": new_user_message})

        payload = {
            "model": convo.model_name,
            "messages": messages_payload,
            "stream": streaming and self.provider.streaming,
            "temperature": convo.temperature,
        }

        url = f"{self.provider.api_url.rstrip('/')}/v1/chat/completions"

        if streaming:
            return self._chat_stream(url, payload, convo, on_chunk)
        return self._chat_single(url, payload, convo)

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

    def _chat_single(self, url: str, payload: dict, convo: Conversation) -> str:
        resp = self._post_with_temperature_retry(url, payload, stream=False)
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})

        # Update conversation
        convo.messages.append(Message(role='assistant', content=content))
        convo.tokens_in += usage.get("prompt_tokens", 0)
        convo.tokens_out += usage.get("completion_tokens", 0)
        return content

    def _chat_stream(self, url: str, payload: dict, convo: Conversation, on_chunk=None) -> str:
        resp = self._post_with_temperature_retry(
            url,
            payload,
            stream=bool(payload.get("stream", False))
        )
        assistant_msg = Message(role='assistant', content="")
        convo.messages.append(assistant_msg)

        last_data = None
        for line in resp.iter_lines():
            if not line:
                continue
            if line.startswith(b"data: "):
                line = line[len(b"data: "):]
            if line == b"[DONE]":
                # Capture usage if present in last chunk
                if last_data:
                    usage = last_data.get("usage", {})
                    convo.tokens_in += usage.get("prompt_tokens", 0)
                    convo.tokens_out += usage.get("completion_tokens", 0)
                break
            try:
                data = json.loads(line)
                last_data = data
            except json.JSONDecodeError:
                continue
            delta = data.get("choices", [{}])[0].get("delta", {})
            content_piece = delta.get("content", "")
            if content_piece:
                assistant_msg.content += content_piece
                if on_chunk:
                    try:
                        on_chunk(assistant_msg.content)
                    except Exception:
                        pass

        return assistant_msg.content

    def _post_with_temperature_retry(self, url: str, payload: dict, stream: bool) -> requests.Response:
        """Post chat payload; retry once without temperature if model rejects it."""
        timeout = getattr(self.config, "timeout", None) or self.timeout
        resp = None
        try:
            resp = self.session.post(url, json=payload, timeout=timeout, stream=stream)
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
