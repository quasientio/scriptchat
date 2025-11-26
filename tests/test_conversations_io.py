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

from litechat.core.conversations import (
    Conversation,
    Message,
    branch_conversation,
    list_conversations,
    load_conversation,
    rename_conversation,
    save_conversation,
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



if __name__ == "__main__":
    unittest.main()
