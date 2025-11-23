import json
import tempfile
import unittest
from pathlib import Path

from litechat.core.conversations import (
    Conversation,
    Message,
    branch_conversation,
    export_conversation_md,
    list_conversations,
    load_conversation,
    rename_conversation,
    save_conversation,
    export_conversation_json,
)


class ConversationIoTests(unittest.TestCase):
    def test_save_and_load_round_trip_with_system_prompt(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            convo = Conversation(
                id=None,
                provider_id="ollama",
                model_name="llama3",
                temperature=0.5,
                system_prompt="be concise",
                messages=[
                    Message(role="user", content="hi"),
                    Message(role="assistant", content="hello"),
                ],
                tokens_in=0,
                tokens_out=0,
            )

            saved = save_conversation(root, convo, save_name="greeting", system_prompt=convo.system_prompt)
            loaded = load_conversation(root, saved.id)

            self.assertEqual(loaded.id, saved.id)
            self.assertEqual(loaded.system_prompt, "be concise")
            self.assertEqual([m.role for m in loaded.messages], ["system", "user", "assistant"])
            self.assertEqual([m.content for m in loaded.messages[1:]], ["hi", "hello"])

    def test_branch_creates_copy_with_new_id(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            convo = Conversation(
                id=None,
                provider_id="ollama",
                model_name="llama3",
                temperature=0.7,
                system_prompt=None,
                messages=[Message(role="user", content="first")],
                tokens_in=0,
                tokens_out=0,
            )

            original = save_conversation(root, convo, save_name="origin")
            branched = branch_conversation(root, original, new_save_name="fork")

            self.assertNotEqual(original.id, branched.id)
            self.assertTrue((root / branched.id).exists())
            self.assertEqual(branched.messages[0].content, "first")

    def test_rename_preserves_prefix_segments(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            # Pre-create a directory that matches expected saved format
            dir_name = "202401010101_modelname_old"
            (root / dir_name).mkdir()
            convo = Conversation(
                id=dir_name,
                provider_id="ollama",
                model_name="modelname",
                temperature=0.7,
                system_prompt=None,
                messages=[],
                tokens_in=0,
                tokens_out=0,
            )

            renamed = rename_conversation(root, convo, "New Name!!")
            parts_old = dir_name.split("_", 2)[:2]
            parts_new = renamed.id.split("_", 2)[:2]

            self.assertEqual(parts_old, parts_new)  # timestamp/model slug preserved
            self.assertTrue((root / renamed.id).exists())
            self.assertFalse((root / dir_name).exists())

    def test_list_conversations_returns_newest_first(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "202401010900_model_a").mkdir()
            (root / "202401011000_model_b").mkdir()

            summaries = list_conversations(root)
            self.assertEqual([s.dir_name for s in summaries], ["202401011000_model_b", "202401010900_model_a"])

    def test_export_conversation_skips_system_messages(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            convo = Conversation(
                id=None,
                provider_id="ollama",
                model_name="llama3",
                temperature=0.7,
                system_prompt="keep it short",
                messages=[
                    Message(role="system", content="keep it short"),
                    Message(role="user", content="ping"),
                    Message(role="assistant", content="pong"),
                ],
                tokens_in=0,
                tokens_out=0,
            )

            export_path = export_conversation_md(convo, Path(tmpdir))
            text = export_path.read_text(encoding="utf-8")

            self.assertTrue(export_path.name.startswith("chat_"))
            self.assertIn("## User", text)
            self.assertNotIn("keep it short", text)  # system prompt omitted
            self.assertIn("pong", text)

    def test_export_conversation_json_includes_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            convo = Conversation(
                id="202401010101_llama3_test",
                provider_id="ollama",
                model_name="llama3",
                temperature=0.7,
                system_prompt="keep it short",
                messages=[
                    Message(role="system", content="keep it short"),
                    Message(role="user", content="ping"),
                    Message(role="assistant", content="pong"),
                ],
                tokens_in=12,
                tokens_out=34,
            )

            export_path = export_conversation_json(convo, Path(tmpdir))
            data = json.loads(export_path.read_text(encoding="utf-8"))

            self.assertEqual(export_path.name, "202401010101_llama3_test.json")
            self.assertEqual(data["id"], convo.id)
            self.assertEqual(data["provider_id"], "ollama")
            self.assertEqual(data["model"], "llama3")
            self.assertEqual(data["temperature"], 0.7)
            self.assertEqual(data["system_prompt"], "keep it short")
            self.assertEqual(data["tokens_in"], 12)
            self.assertEqual(data["tokens_out"], 34)
            self.assertEqual([m["role"] for m in data["messages"]], ["system", "user", "assistant"])
            self.assertEqual(data["messages"][1]["content"], "ping")
            self.assertTrue(data["exported_at"])


if __name__ == "__main__":
    unittest.main()
