import tempfile
import unittest
from pathlib import Path

from litechat.ui.app import resolve_clear_target_from_args
from litechat.core.conversations import ConversationSummary


def make_convo_dir(root: Path, name: str):
    d = root / name
    d.mkdir()
    meta = d / "meta.json"
    meta.write_text('{"created_at": "2025-01-01", "model": "m"}')
    return d


class ClearTargetTests(unittest.TestCase):
    def test_clear_uses_fresh_list_on_index(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            # create two conversations with sortable names (list_conversations sorts reverse)
            c1 = make_convo_dir(root, "202401010900_model_a")
            c2 = make_convo_dir(root, "202401011000_model_b")  # newest -> index 0

            target_id, prompt, err, summaries, is_current = resolve_clear_target_from_args(
                "1", root, None
            )

            self.assertIsNone(err)
            # Because of reverse sort, index 0 -> later timestamp, index 1 -> earlier
            self.assertEqual(target_id, c1.name)
            self.assertIn("conversation #1", prompt)
            self.assertEqual(len(summaries), 2)
            self.assertFalse(is_current)

    def test_clear_defaults_to_current_without_index(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            make_convo_dir(root, "202401010900_model_a")
            target_id, prompt, err, summaries, is_current = resolve_clear_target_from_args(
                "", root, "current-id"
            )
            self.assertIsNone(err)
            self.assertEqual(target_id, "current-id")
            self.assertIn("current-id", prompt)
            self.assertTrue(is_current)

    def test_clear_marks_current_when_ids_match(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            make_convo_dir(root, "202401010900_model_a")
            target_id, prompt, err, summaries, is_current = resolve_clear_target_from_args(
                "0", root, "202401010900_model_a"
            )

            self.assertTrue(is_current)
            self.assertIsNone(err)
            self.assertEqual(target_id, "202401010900_model_a")


if __name__ == "__main__":
    unittest.main()
