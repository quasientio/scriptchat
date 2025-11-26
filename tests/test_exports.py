# Copyright 2024 lite-chat contributors
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

import json
import tempfile
import unittest
from pathlib import Path

from litechat.core.conversations import Conversation, Message, load_conversation
from litechat.core.exports import (
    export_conversation_md,
    export_conversation_json,
    export_conversation_html,
    import_conversation_from_file,
)


class ExportImportTests(unittest.TestCase):
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

    def test_export_conversation_html_contains_messages(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            convo = Conversation(
                id=None,
                provider_id="ollama",
                model_name="llama3",
                temperature=0.7,
                system_prompt=None,
                messages=[
                    Message(role="user", content="# Heading\n\nhello **bold** and `code`\n\n* item1\n* item2\n\n| a | b |\n| --- | --- |\n| 1 | 2 |\n\nSee [link](http://example.com)"),
                    Message(role="assistant", content="hi there"),
                ],
                tokens_in=0,
                tokens_out=0,
            )

            export_path = export_conversation_html(convo, root)
            html = export_path.read_text(encoding="utf-8")

            self.assertTrue(export_path.name.endswith(".html"))
            self.assertIn("<h1>Heading</h1>", html)
            self.assertIn("hello", html)
            self.assertIn("<strong>bold</strong>", html)
            self.assertIn("<code>code</code>", html)
            self.assertIn("<table>", html)
            self.assertIn("<a href=\"http://example.com\">link</a>", html)
            self.assertIn("hi there", html)

    def test_import_conversation_from_json(self):
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
                tokens_in=5,
                tokens_out=6,
            )

            export_path = export_conversation_json(convo, root)

            import_root = root / "imported"
            imported = import_conversation_from_file(export_path, import_root)

            self.assertTrue((import_root / imported.id).exists())

            loaded = load_conversation(import_root, imported.id)
            self.assertEqual(loaded.provider_id, "ollama")
            self.assertEqual(loaded.model_name, "llama3")
            self.assertEqual(loaded.temperature, 0.7)
            self.assertEqual(loaded.system_prompt, "keep it short")
            self.assertEqual(loaded.tokens_in, 0)  # token counts not persisted in meta files
            self.assertEqual(loaded.tokens_out, 0)
            self.assertEqual([m.role for m in loaded.messages], ["system", "user", "assistant"])
            self.assertEqual(loaded.messages[2].content, "pong")

    def test_import_conversation_from_md(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            convo = Conversation(
                id=None,
                provider_id="ollama",
                model_name="llama3",
                temperature=0.5,
                system_prompt=None,
                messages=[
                    Message(role="user", content="hello"),
                    Message(role="assistant", content="hi there"),
                ],
                tokens_in=0,
                tokens_out=0,
            )

            export_path = export_conversation_md(convo, root)

            import_root = root / "imported"
            imported = import_conversation_from_file(export_path, import_root)

            self.assertTrue((import_root / imported.id).exists())

            loaded = load_conversation(import_root, imported.id)
            self.assertEqual(loaded.system_prompt, None)
            self.assertEqual([m.role for m in loaded.messages], ["user", "assistant"])
            self.assertEqual(loaded.messages[0].content, "hello")

    def test_import_md_multiple_turns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            convo = Conversation(
                id=None,
                provider_id="ollama",
                model_name="llama3",
                temperature=0.9,
                system_prompt=None,
                messages=[
                    Message(role="user", content="turn1"),
                    Message(role="assistant", content="reply1"),
                    Message(role="user", content="turn2"),
                    Message(role="assistant", content="reply2"),
                ],
                tokens_in=0,
                tokens_out=0,
            )

            export_path = export_conversation_md(convo, root)
            imported = import_conversation_from_file(export_path, root / "imported")

            loaded = load_conversation(root / "imported", imported.id)
            self.assertEqual([m.role for m in loaded.messages], ["user", "assistant", "user", "assistant"])
            self.assertEqual([m.content for m in loaded.messages], ["turn1", "reply1", "turn2", "reply2"])


if __name__ == "__main__":
    unittest.main()
