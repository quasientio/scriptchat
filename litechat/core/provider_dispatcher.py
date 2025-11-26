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
