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

from scriptchat.core.conversations import (
    ARCHIVE_DIR,
    Conversation,
    Message,
    archive_conversation,
    branch_conversation,
    import_chatgpt_export,
    list_conversations,
    load_conversation,
    parse_chatgpt_export,
    rename_conversation,
    save_conversation,
    unarchive_conversation,
    _parse_chatgpt_conversation,
    _walk_chatgpt_messages,
    _extract_chatgpt_content,
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

    def test_load_conversation_handles_old_file_references_format(self):
        """Backwards compatibility: old file_references stored path as string, new stores dict."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            # Create a file to reference
            test_file = root / "test.txt"
            test_file.write_text("content", encoding="utf-8")

            # Create conversation directory with old-format file_references (string path)
            conv_dir = root / "202401010101_model_test"
            conv_dir.mkdir()
            meta = {
                "display_name": "test",
                "provider_id": "ollama",
                "model_name": "model",
                "temperature": 0.7,
                "file_references": {
                    "test.txt": str(test_file)  # Old format: just string path
                }
            }
            (conv_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

            # Load should succeed and rehydrate file registry
            loaded = load_conversation(root, "202401010101_model_test")
            self.assertIsNotNone(loaded)
            # Check that file_registry was rehydrated
            self.assertTrue(hasattr(loaded, "file_registry"))
            self.assertIn("test.txt", loaded.file_registry)
            self.assertEqual(loaded.file_registry["test.txt"]["content"], "content")

    def test_branch_sets_parent_id(self):
        """Branch creates new conversation with parent_id pointing to original."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
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

            # Save original
            original = save_conversation(root, convo, save_name="parent")

            # Branch it
            branched = branch_conversation(root, original, new_save_name="child")

            # Verify parent_id is set
            self.assertEqual(branched.parent_id, original.id)
            self.assertIsNotNone(branched.branched_at)

            # Verify it's persisted in meta.json
            meta_path = root / branched.id / "meta.json"
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            self.assertEqual(meta.get("parent_id"), original.id)
            self.assertIsNotNone(meta.get("branched_at"))

    def test_branch_preserves_messages(self):
        """Branch copies all messages from parent."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            convo = Conversation(
                id=None,
                provider_id="ollama",
                model_name="llama3",
                temperature=0.7,
                system_prompt=None,
                messages=[
                    Message(role="user", content="hello"),
                    Message(role="assistant", content="hi there"),
                ],
                tokens_in=10,
                tokens_out=5,
            )

            original = save_conversation(root, convo, save_name="parent")
            branched = branch_conversation(root, original, new_save_name="child")

            # Messages should be copied
            self.assertEqual(len(branched.messages), len(original.messages))
            self.assertEqual(branched.messages[0].content, "hello")
            self.assertEqual(branched.messages[1].content, "hi there")

    def test_list_conversations_includes_parent_id(self):
        """list_conversations returns parent_id in summaries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            convo = Conversation(
                id=None,
                provider_id="ollama",
                model_name="llama3",
                temperature=0.7,
                system_prompt=None,
                messages=[],
                tokens_in=0,
                tokens_out=0,
            )

            original = save_conversation(root, convo, save_name="parent")
            branched = branch_conversation(root, original, new_save_name="child")

            summaries = list_conversations(root)
            # Find the branched conversation summary
            branch_summary = next(s for s in summaries if s.dir_name == branched.id)
            self.assertEqual(branch_summary.parent_id, original.id)

            # Parent should have no parent_id
            parent_summary = next(s for s in summaries if s.dir_name == original.id)
            self.assertIsNone(parent_summary.parent_id)

    def test_recursive_branching(self):
        """Branching from a branch creates correct parent chain."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            convo = Conversation(
                id=None,
                provider_id="ollama",
                model_name="llama3",
                temperature=0.7,
                system_prompt=None,
                messages=[Message(role="user", content="root message")],
                tokens_in=0,
                tokens_out=0,
            )

            # Create: root -> branch1 -> branch2
            root_convo = save_conversation(root, convo, save_name="root")
            root_convo.messages.append(Message(role="assistant", content="root reply"))
            save_conversation(root, root_convo)

            branch1 = branch_conversation(root, root_convo, new_save_name="branch1")
            branch1.messages.append(Message(role="user", content="branch1 message"))
            save_conversation(root, branch1)

            branch2 = branch_conversation(root, branch1, new_save_name="branch2")

            # Verify parent chain
            self.assertIsNone(root_convo.parent_id)
            self.assertEqual(branch1.parent_id, root_convo.id)
            self.assertEqual(branch2.parent_id, branch1.id)  # Parent is branch1, not root

            # Verify branch2 has all messages from branch1
            self.assertEqual(len(branch2.messages), 3)  # root msg + root reply + branch1 msg

            # Verify loaded conversations maintain parent chain
            loaded_branch2 = load_conversation(root, branch2.id)
            self.assertEqual(loaded_branch2.parent_id, branch1.id)


class ChatGptImportTests(unittest.TestCase):
    def test_extract_chatgpt_content_text(self):
        """Extract text content from simple text type."""
        content = {"content_type": "text", "parts": ["Hello world"]}
        result = _extract_chatgpt_content(content)
        self.assertEqual(result, "Hello world")

    def test_extract_chatgpt_content_multipart(self):
        """Extract text from multiple parts."""
        content = {"content_type": "text", "parts": ["Line 1", "Line 2"]}
        result = _extract_chatgpt_content(content)
        self.assertEqual(result, "Line 1\nLine 2")

    def test_extract_chatgpt_content_empty(self):
        """Handle empty parts gracefully."""
        content = {"content_type": "text", "parts": []}
        result = _extract_chatgpt_content(content)
        self.assertEqual(result, "")

    def test_extract_chatgpt_content_dict_parts(self):
        """Extract text from dict parts (multimodal)."""
        content = {
            "content_type": "multimodal_text",
            "parts": [{"text": "Caption for image"}]
        }
        result = _extract_chatgpt_content(content)
        self.assertEqual(result, "Caption for image")

    def test_walk_chatgpt_messages_simple(self):
        """Walk a simple linear conversation."""
        mapping = {
            "root": {"id": "root", "message": None, "parent": None, "children": ["msg1"]},
            "msg1": {
                "id": "msg1",
                "message": {
                    "author": {"role": "user"},
                    "content": {"content_type": "text", "parts": ["Hello"]}
                },
                "parent": "root",
                "children": ["msg2"]
            },
            "msg2": {
                "id": "msg2",
                "message": {
                    "author": {"role": "assistant"},
                    "content": {"content_type": "text", "parts": ["Hi there!"]}
                },
                "parent": "msg1",
                "children": []
            }
        }
        messages = _walk_chatgpt_messages(mapping, "msg2")
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0].role, "user")
        self.assertEqual(messages[0].content, "Hello")
        self.assertEqual(messages[1].role, "assistant")
        self.assertEqual(messages[1].content, "Hi there!")

    def test_walk_chatgpt_messages_skips_tool_role(self):
        """Tool messages should be skipped."""
        mapping = {
            "root": {"id": "root", "message": None, "parent": None, "children": ["msg1"]},
            "msg1": {
                "id": "msg1",
                "message": {
                    "author": {"role": "user"},
                    "content": {"content_type": "text", "parts": ["Search for X"]}
                },
                "parent": "root",
                "children": ["msg2"]
            },
            "msg2": {
                "id": "msg2",
                "message": {
                    "author": {"role": "tool"},
                    "content": {"content_type": "text", "parts": ["Tool result"]}
                },
                "parent": "msg1",
                "children": ["msg3"]
            },
            "msg3": {
                "id": "msg3",
                "message": {
                    "author": {"role": "assistant"},
                    "content": {"content_type": "text", "parts": ["Based on search..."]}
                },
                "parent": "msg2",
                "children": []
            }
        }
        messages = _walk_chatgpt_messages(mapping, "msg3")
        self.assertEqual(len(messages), 2)  # user and assistant only
        self.assertEqual(messages[0].role, "user")
        self.assertEqual(messages[1].role, "assistant")

    def test_parse_chatgpt_conversation(self):
        """Parse a complete ChatGPT conversation structure."""
        data = {
            "title": "Test Conversation",
            "default_model_slug": "gpt-4o",
            "create_time": 1234567890.0,
            "current_node": "msg2",
            "mapping": {
                "root": {"id": "root", "message": None, "parent": None, "children": ["msg1"]},
                "msg1": {
                    "id": "msg1",
                    "message": {
                        "author": {"role": "user"},
                        "content": {"content_type": "text", "parts": ["Question?"]}
                    },
                    "parent": "root",
                    "children": ["msg2"]
                },
                "msg2": {
                    "id": "msg2",
                    "message": {
                        "author": {"role": "assistant"},
                        "content": {"content_type": "text", "parts": ["Answer!"]}
                    },
                    "parent": "msg1",
                    "children": []
                }
            }
        }
        result = _parse_chatgpt_conversation(data, "openai")
        self.assertIsNotNone(result)
        convo, timestamp = result
        self.assertEqual(convo.model_name, "gpt-4o")
        self.assertEqual(convo.provider_id, "openai")
        self.assertEqual(len(convo.messages), 2)
        self.assertEqual(convo.messages[0].content, "Question?")
        self.assertEqual(convo.messages[1].content, "Answer!")
        self.assertEqual(convo.tags.get("imported_from"), "chatgpt")
        self.assertIsNotNone(timestamp)
        self.assertEqual(timestamp.year, 2009)  # 1234567890 = 2009-02-13

    def test_parse_chatgpt_conversation_with_system_prompt(self):
        """System prompt should be extracted separately."""
        data = {
            "title": "Test",
            "default_model_slug": "gpt-4",
            "current_node": "msg2",
            "mapping": {
                "root": {"id": "root", "message": None, "parent": None, "children": ["sys"]},
                "sys": {
                    "id": "sys",
                    "message": {
                        "author": {"role": "system"},
                        "content": {"content_type": "text", "parts": ["You are helpful."]}
                    },
                    "parent": "root",
                    "children": ["msg1"]
                },
                "msg1": {
                    "id": "msg1",
                    "message": {
                        "author": {"role": "user"},
                        "content": {"content_type": "text", "parts": ["Hi"]}
                    },
                    "parent": "sys",
                    "children": ["msg2"]
                },
                "msg2": {
                    "id": "msg2",
                    "message": {
                        "author": {"role": "assistant"},
                        "content": {"content_type": "text", "parts": ["Hello!"]}
                    },
                    "parent": "msg1",
                    "children": []
                }
            }
        }
        result = _parse_chatgpt_conversation(data, "openai")
        convo, timestamp = result
        self.assertEqual(convo.system_prompt, "You are helpful.")
        self.assertEqual(len(convo.messages), 2)  # Only user and assistant

    def test_import_chatgpt_export_zip(self):
        """Test importing from a ZIP file."""
        import zipfile
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            # Create a mock ChatGPT export ZIP
            zip_path = root / "export.zip"
            conversations_data = [{
                "title": "Test Chat",
                "default_model_slug": "gpt-4",
                "current_node": "msg2",
                "mapping": {
                    "root": {"id": "root", "message": None, "parent": None, "children": ["msg1"]},
                    "msg1": {
                        "id": "msg1",
                        "message": {
                            "author": {"role": "user"},
                            "content": {"content_type": "text", "parts": ["Hello"]}
                        },
                        "parent": "root",
                        "children": ["msg2"]
                    },
                    "msg2": {
                        "id": "msg2",
                        "message": {
                            "author": {"role": "assistant"},
                            "content": {"content_type": "text", "parts": ["Hi!"]}
                        },
                        "parent": "msg1",
                        "children": []
                    }
                }
            }]

            with zipfile.ZipFile(zip_path, 'w') as zf:
                zf.writestr('conversations.json', json.dumps(conversations_data))

            # Import
            convos_dir = root / "conversations"
            convos_dir.mkdir()
            imported = import_chatgpt_export(zip_path, convos_dir, "openai")

            self.assertEqual(len(imported), 1)
            self.assertEqual(imported[0].model_name, "gpt-4")
            self.assertEqual(len(imported[0].messages), 2)

            # Verify it was saved
            saved_dirs = list(convos_dir.iterdir())
            self.assertEqual(len(saved_dirs), 1)

    def test_import_chatgpt_export_missing_conversations_json(self):
        """Raise error if ZIP doesn't contain conversations.json."""
        import zipfile
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            zip_path = root / "bad.zip"

            with zipfile.ZipFile(zip_path, 'w') as zf:
                zf.writestr('other.json', '{}')

            with self.assertRaises(ValueError) as ctx:
                import_chatgpt_export(zip_path, root, "openai")

            self.assertIn("conversations.json", str(ctx.exception))

    def test_parse_chatgpt_export_dry_run(self):
        """parse_chatgpt_export returns parsed conversations without saving."""
        import zipfile
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            # Create a mock ChatGPT export ZIP
            zip_path = root / "export.zip"
            conversations_data = [
                {
                    "title": "First Chat",
                    "default_model_slug": "gpt-4",
                    "current_node": "msg2",
                    "mapping": {
                        "root": {"id": "root", "message": None, "parent": None, "children": ["msg1"]},
                        "msg1": {
                            "id": "msg1",
                            "message": {
                                "author": {"role": "user"},
                                "content": {"content_type": "text", "parts": ["Hello"]}
                            },
                            "parent": "root",
                            "children": ["msg2"]
                        },
                        "msg2": {
                            "id": "msg2",
                            "message": {
                                "author": {"role": "assistant"},
                                "content": {"content_type": "text", "parts": ["Hi!"]}
                            },
                            "parent": "msg1",
                            "children": []
                        }
                    }
                },
                {
                    "title": "Second Chat",
                    "default_model_slug": "o3",
                    "current_node": "msg2",
                    "mapping": {
                        "root": {"id": "root", "message": None, "parent": None, "children": ["msg1"]},
                        "msg1": {
                            "id": "msg1",
                            "message": {
                                "author": {"role": "user"},
                                "content": {"content_type": "text", "parts": ["Question"]}
                            },
                            "parent": "root",
                            "children": ["msg2"]
                        },
                        "msg2": {
                            "id": "msg2",
                            "message": {
                                "author": {"role": "assistant"},
                                "content": {"content_type": "text", "parts": ["Answer"]}
                            },
                            "parent": "msg1",
                            "children": []
                        }
                    }
                }
            ]

            with zipfile.ZipFile(zip_path, 'w') as zf:
                zf.writestr('conversations.json', json.dumps(conversations_data))

            # Parse without saving
            parsed = parse_chatgpt_export(zip_path, "openai")

            self.assertEqual(len(parsed), 2)
            convo1, title1, ts1 = parsed[0]
            convo2, title2, ts2 = parsed[1]
            self.assertEqual(title1, "First Chat")
            self.assertEqual(convo1.model_name, "gpt-4")
            self.assertEqual(title2, "Second Chat")
            self.assertEqual(convo2.model_name, "o3")

            # Verify nothing was saved
            convos_dir = root / "conversations"
            self.assertFalse(convos_dir.exists())


class ArchiveTests(unittest.TestCase):
    """Tests for archive/unarchive functionality."""

    def _create_test_convo(self, root: Path, name: str, tags: dict = None) -> Conversation:
        """Helper to create and save a test conversation."""
        convo = Conversation(
            id=None,
            provider_id="ollama",
            model_name="test-model",
            temperature=0.7,
            messages=[Message(role="user", content="test")],
            tags=tags or {},
        )
        return save_conversation(root, convo, name)

    def test_archive_conversation_moves_to_archive_dir(self):
        """Test archiving moves conversation to .archive directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            convo = self._create_test_convo(root, "test-convo")

            # Verify it exists in root
            self.assertTrue((root / convo.id).exists())

            # Archive it
            archive_conversation(root, convo.id)

            # Verify moved to archive
            self.assertFalse((root / convo.id).exists())
            self.assertTrue((root / ARCHIVE_DIR / convo.id).exists())

    def test_archive_conversation_not_found_raises(self):
        """Test archiving non-existent conversation raises FileNotFoundError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with self.assertRaises(FileNotFoundError):
                archive_conversation(root, "nonexistent")

    def test_archive_already_archived_raises(self):
        """Test archiving already archived conversation raises FileExistsError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            convo = self._create_test_convo(root, "test-convo")

            archive_conversation(root, convo.id)
            # Try to archive again
            with self.assertRaises(FileNotFoundError):
                archive_conversation(root, convo.id)

    def test_unarchive_conversation_restores_to_root(self):
        """Test unarchiving moves conversation back to root."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            convo = self._create_test_convo(root, "test-convo")
            dir_name = convo.id

            archive_conversation(root, dir_name)
            self.assertFalse((root / dir_name).exists())

            unarchive_conversation(root, dir_name)
            self.assertTrue((root / dir_name).exists())
            self.assertFalse((root / ARCHIVE_DIR / dir_name).exists())

    def test_unarchive_not_found_raises(self):
        """Test unarchiving non-existent conversation raises FileNotFoundError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with self.assertRaises(FileNotFoundError):
                unarchive_conversation(root, "nonexistent")

    def test_list_conversations_filter_active(self):
        """Test list_conversations with filter='active' excludes archived."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            convo1 = self._create_test_convo(root, "active-convo")
            convo2 = self._create_test_convo(root, "archived-convo")
            archive_conversation(root, convo2.id)

            summaries = list_conversations(root, filter="active")
            dir_names = [s.dir_name for s in summaries]

            self.assertIn(convo1.id, dir_names)
            self.assertNotIn(convo2.id, dir_names)

    def test_list_conversations_filter_archived(self):
        """Test list_conversations with filter='archived' shows only archived."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            convo1 = self._create_test_convo(root, "active-convo")
            convo2 = self._create_test_convo(root, "archived-convo")
            archive_conversation(root, convo2.id)

            summaries = list_conversations(root, filter="archived")
            dir_names = [s.dir_name for s in summaries]

            self.assertNotIn(convo1.id, dir_names)
            self.assertIn(convo2.id, dir_names)
            # Archived display name should have prefix
            self.assertTrue(summaries[0].display_name.startswith("[archived]"))

    def test_list_conversations_filter_all(self):
        """Test list_conversations with filter='all' shows both."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            convo1 = self._create_test_convo(root, "active-convo")
            convo2 = self._create_test_convo(root, "archived-convo")
            archive_conversation(root, convo2.id)

            summaries = list_conversations(root, filter="all")
            dir_names = [s.dir_name for s in summaries]

            self.assertIn(convo1.id, dir_names)
            self.assertIn(convo2.id, dir_names)

    def test_list_conversations_default_filter_is_active(self):
        """Test list_conversations defaults to active filter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            convo1 = self._create_test_convo(root, "active-convo")
            convo2 = self._create_test_convo(root, "archived-convo")
            archive_conversation(root, convo2.id)

            # Default (no filter arg)
            summaries = list_conversations(root)
            dir_names = [s.dir_name for s in summaries]

            self.assertIn(convo1.id, dir_names)
            self.assertNotIn(convo2.id, dir_names)

    def test_archive_preserves_conversation_data(self):
        """Test that archived conversations can still be loaded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            convo = Conversation(
                id=None,
                provider_id="ollama",
                model_name="test-model",
                temperature=0.7,
                messages=[
                    Message(role="user", content="hello"),
                    Message(role="assistant", content="hi there"),
                ],
                tags={"key": "value"},
            )
            saved = save_conversation(root, convo, "test-convo")
            dir_name = saved.id

            archive_conversation(root, dir_name)

            # Load from archive location
            archive_dir = root / ARCHIVE_DIR
            loaded = load_conversation(archive_dir, dir_name)

            self.assertEqual(loaded.model_name, "test-model")
            self.assertEqual(len(loaded.messages), 2)
            self.assertEqual(loaded.tags, {"key": "value"})


if __name__ == "__main__":
    unittest.main()
