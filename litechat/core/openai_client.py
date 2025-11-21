"""OpenAI-compatible chat client."""

import json
from typing import Optional

import requests

from .config import Config, ProviderConfig
from .conversations import Conversation, Message


class OpenAIChatClient:
    """Client for OpenAI-compatible chat completions."""

    def __init__(self, config: Config, provider: ProviderConfig, timeout: int):
        self.config = config
        self.provider = provider
        self.timeout = timeout
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
        on_chunk=None
    ) -> str:
        if convo.provider_id != self.provider.id:
            raise ValueError(f"Provider mismatch: expected {self.provider.id} got {convo.provider_id}")

        messages_payload = []
        for msg in convo.messages:
            messages_payload.append({
                "role": msg.role,
                "content": msg.content
            })
        messages_payload.append({"role": "user", "content": new_user_message})

        payload = {
            "model": convo.model_name,
            "messages": messages_payload,
            "stream": streaming,
            "temperature": convo.temperature,
        }

        url = f"{self.provider.api_url.rstrip('/')}/v1/chat/completions"

        if streaming:
            return self._chat_stream(url, payload, convo, on_chunk)
        return self._chat_single(url, payload, convo)

    def _chat_single(self, url: str, payload: dict, convo: Conversation) -> str:
        resp = self.session.post(url, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})

        # Update conversation
        convo.messages.append(Message(role='assistant', content=content))
        convo.tokens_in += usage.get("prompt_tokens", 0)
        convo.tokens_out += usage.get("completion_tokens", 0)
        return content

    def _chat_stream(self, url: str, payload: dict, convo: Conversation, on_chunk=None) -> str:
        resp = self.session.post(url, json=payload, timeout=self.timeout, stream=True)
        resp.raise_for_status()
        assistant_msg = Message(role='assistant', content="")
        convo.messages.append(assistant_msg)

        for line in resp.iter_lines():
            if not line:
                continue
            if line.startswith(b"data: "):
                line = line[len(b"data: "):]
            if line == b"[DONE]":
                # Capture usage if present in last chunk
                try:
                    end_data = data
                    usage = end_data.get("usage", {})
                    convo.tokens_in += usage.get("prompt_tokens", 0)
                    convo.tokens_out += usage.get("completion_tokens", 0)
                except Exception:
                    pass
                break
            try:
                data = json.loads(line)
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
