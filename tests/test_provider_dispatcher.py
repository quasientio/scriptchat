import unittest

from litechat.core.conversations import Conversation
from litechat.core.provider_dispatcher import ProviderDispatcher


class DummyClient:
    def __init__(self, response: str = "ok", raise_on_unload: bool = False):
        self.response = response
        self.calls = []
        self.unload_called = False
        self.raise_on_unload = raise_on_unload
        self.server_manager = self
        self.stop_called = False

    def chat(self, convo, message, streaming=False, on_chunk=None):
        self.calls.append((convo.id, message, streaming))
        return self.response

    def unload_model(self):
        self.unload_called = True
        if self.raise_on_unload:
            raise RuntimeError("boom")

    def stop(self):
        self.stop_called = True


class ProviderDispatcherTests(unittest.TestCase):
    def test_dispatches_to_matching_client(self):
        convo = Conversation(id=None, provider_id="p1", model_name="m", temperature=0.7, messages=[], tokens_in=0, tokens_out=0)
        client = DummyClient(response="answer")
        dispatcher = ProviderDispatcher(clients={"p1": client})

        result = dispatcher.chat(convo, "hi")
        self.assertEqual(result, "answer")
        self.assertEqual(client.calls[0][1], "hi")

    def test_missing_client_raises(self):
        convo = Conversation(id=None, provider_id="unknown", model_name="m", temperature=0.7, messages=[], tokens_in=0, tokens_out=0)
        dispatcher = ProviderDispatcher(clients={})
        with self.assertRaises(ValueError):
            dispatcher.chat(convo, "hi")

    def test_cleanup_calls_unload_and_stop_even_on_error(self):
        good = DummyClient()
        flaky = DummyClient(raise_on_unload=True)
        dispatcher = ProviderDispatcher(clients={"good": good, "flaky": flaky})

        dispatcher.cleanup()

        self.assertTrue(good.unload_called)
        self.assertTrue(good.stop_called)
        self.assertTrue(flaky.unload_called)
        self.assertTrue(flaky.stop_called)


if __name__ == "__main__":
    unittest.main()
