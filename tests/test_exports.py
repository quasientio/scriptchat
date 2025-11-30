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

import json
import tempfile
import unittest
from pathlib import Path

from scriptchat.core.conversations import Conversation, Message, load_conversation
from scriptchat.core.exports import (
    export_conversation_md,
    export_conversation_json,
    export_conversation_html,
    import_conversation_from_file,
    generate_html_index,
)
from scriptchat.core.conversations import save_conversation, branch_conversation


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

    def test_export_conversation_html_nested_lists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            nested_list_md = """- Level 1 item A
  - Level 2 item A1
  - Level 2 item A2
    - Level 3 item A2a
- Level 1 item B
  - Level 2 item B1"""
            convo = Conversation(
                id=None,
                provider_id="ollama",
                model_name="llama3",
                temperature=0.7,
                system_prompt=None,
                messages=[
                    Message(role="user", content=nested_list_md),
                ],
                tokens_in=0,
                tokens_out=0,
            )

            export_path = export_conversation_html(convo, root)
            html = export_path.read_text(encoding="utf-8")

            # Should have nested <ul> elements
            self.assertIn("<ul>", html)
            # Check for nested structure: <li>...<ul>...</ul></li>
            self.assertIn("<li>Level 1 item A<ul>", html)
            self.assertIn("<li>Level 2 item A2<ul>", html)
            self.assertIn("<li>Level 3 item A2a</li>", html)
            # Count nested ul tags - should have more than 1
            ul_count = html.count("<ul>")
            self.assertGreaterEqual(ul_count, 3)  # At least 3 levels

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


class HtmlIndexTests(unittest.TestCase):
    def test_generate_html_index_creates_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            export_dir = Path(tmpdir) / "exports"
            convos_dir = Path(tmpdir) / "conversations"
            export_dir.mkdir()
            convos_dir.mkdir()

            # Create a conversation and export it
            convo = Conversation(
                id=None,
                provider_id="ollama",
                model_name="llama3",
                temperature=0.7,
                system_prompt=None,
                messages=[Message(role="user", content="hello")],
                tokens_in=0,
                tokens_out=0,
            )
            saved = save_conversation(convos_dir, convo, save_name="test")
            export_conversation_html(saved, export_dir)

            # Generate index
            index_path = generate_html_index(export_dir, convos_dir)

            self.assertEqual(index_path, export_dir / "index.html")
            self.assertTrue(index_path.exists())

            content = index_path.read_text(encoding="utf-8")
            self.assertIn("Conversation Exports", content)
            self.assertIn("test", content)  # display name
            self.assertIn(".html", content)  # link to export

    def test_generate_html_index_shows_tags(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            export_dir = Path(tmpdir) / "exports"
            convos_dir = Path(tmpdir) / "conversations"
            export_dir.mkdir()
            convos_dir.mkdir()

            convo = Conversation(
                id=None,
                provider_id="ollama",
                model_name="llama3",
                temperature=0.7,
                system_prompt=None,
                messages=[Message(role="user", content="hello")],
                tokens_in=0,
                tokens_out=0,
                tags={"topic": "science", "project": "alpha"},
            )
            saved = save_conversation(convos_dir, convo, save_name="tagged")
            export_conversation_html(saved, export_dir)

            index_path = generate_html_index(export_dir, convos_dir)
            content = index_path.read_text(encoding="utf-8")

            # Should have tag sections
            self.assertIn("topic=science", content)
            self.assertIn("project=alpha", content)
            # Conversation should appear in tag sections
            self.assertIn("<h2>project=alpha</h2>", content)
            self.assertIn("<h2>topic=science</h2>", content)

    def test_generate_html_index_shows_branches(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            export_dir = Path(tmpdir) / "exports"
            convos_dir = Path(tmpdir) / "conversations"
            export_dir.mkdir()
            convos_dir.mkdir()

            # Create parent and child conversations
            convo = Conversation(
                id=None,
                provider_id="ollama",
                model_name="llama3",
                temperature=0.7,
                system_prompt=None,
                messages=[Message(role="user", content="hello")],
                tokens_in=0,
                tokens_out=0,
            )
            parent = save_conversation(convos_dir, convo, save_name="parent")
            child = branch_conversation(convos_dir, parent, new_save_name="child")

            # Export both
            export_conversation_html(parent, export_dir)
            export_conversation_html(child, export_dir)

            index_path = generate_html_index(export_dir, convos_dir)
            content = index_path.read_text(encoding="utf-8")

            # Both should be in the index
            self.assertIn("parent", content)
            self.assertIn("child", content)
            # Child should be in a branches section (nested under parent)
            self.assertIn('class="branches"', content)

    def test_generate_html_index_cleans_orphaned_entries(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            export_dir = Path(tmpdir) / "exports"
            convos_dir = Path(tmpdir) / "conversations"
            export_dir.mkdir()
            convos_dir.mkdir()

            # Create and export a conversation
            convo = Conversation(
                id=None,
                provider_id="ollama",
                model_name="llama3",
                temperature=0.7,
                system_prompt=None,
                messages=[Message(role="user", content="hello")],
                tokens_in=0,
                tokens_out=0,
            )
            saved = save_conversation(convos_dir, convo, save_name="test")
            export_path = export_conversation_html(saved, export_dir)

            # Generate index - should have the export
            index_path = generate_html_index(export_dir, convos_dir)
            content = index_path.read_text(encoding="utf-8")
            self.assertIn("test", content)

            # Delete the export file
            export_path.unlink()

            # Regenerate index - orphaned entry should be gone
            index_path = generate_html_index(export_dir, convos_dir)
            content = index_path.read_text(encoding="utf-8")
            self.assertNotIn(export_path.name, content)


if __name__ == "__main__":
    unittest.main()
