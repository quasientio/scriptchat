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

"""Provider dispatcher for routing chat requests to the correct backend."""

from typing import Optional

from .conversations import Conversation


class ProviderDispatcher:
    """Dispatch chat calls to provider-specific clients."""

    def __init__(self, clients: dict[str, object]):
        self.clients = clients

    def chat(self, convo: Conversation, new_user_message: str, streaming: bool = False, on_chunk=None, expanded_history: list | None = None) -> str:
        client = self.clients.get(convo.provider_id)
        if client is None:
            raise ValueError(f"No client configured for provider '{convo.provider_id}'")
        return client.chat(convo, new_user_message, streaming=streaming, on_chunk=on_chunk, expanded_history=expanded_history)

    def cleanup(self):
        """Call cleanup on all clients if available."""
        for client in self.clients.values():
            if hasattr(client, "unload_model"):
                try:
                    client.unload_model()
                except Exception:
                    pass
            if hasattr(client, "server_manager"):
                try:
                    client.server_manager.stop()
                except Exception:
                    pass
