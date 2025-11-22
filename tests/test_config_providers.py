import os
import tempfile
import unittest
from pathlib import Path

from litechat.core.config import load_config


def write_config(text: str, dirpath: Path):
    cfg_dir = dirpath / ".lite-chat"
    cfg_dir.mkdir()
    (cfg_dir / "config.toml").write_text(text, encoding="utf-8")


class ConfigProviderTests(unittest.TestCase):
    def test_loads_providers_and_default_provider(self):
        toml_text = """
[general]
default_provider = "ollama"
default_model = "llama3.1"
conversations_dir = "{conv}"

[ollama]
api_url = "http://localhost:11434/api"
default_model = "llama3.1"

[[providers]]
id = "ollama"
type = "ollama"
api_url = "http://localhost:11434/api"
models = "llama3.1,phi3"
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["HOME"] = tmpdir
            conv_dir = Path(tmpdir) / "conversations"
            text = toml_text.format(conv=conv_dir.as_posix())
            write_config(text, Path(tmpdir))
            cfg = load_config()
            self.assertEqual(cfg.default_provider, "ollama")
            self.assertIsNotNone(cfg.get_provider("ollama"))
            models = cfg.list_models("ollama")
            self.assertTrue(any(m.name == "llama3.1" for m in models))

    def test_missing_default_provider_raises(self):
        toml_text = """
[general]
default_provider = "missing"
conversations_dir = "{conv}"

[[providers]]
id = "ollama"
type = "ollama"
api_url = "http://localhost:11434/api"
models = "llama3"
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["HOME"] = tmpdir
            conv_dir = Path(tmpdir) / "conversations"
            text = toml_text.format(conv=conv_dir.as_posix())
            write_config(text, Path(tmpdir))
            with self.assertRaises(ValueError):
                load_config()

    def test_invalid_provider_entry_and_legacy_model_contexts(self):
        # Missing required provider fields
        toml_text = """
[general]
default_provider = "ollama"
conversations_dir = "{conv}"

[[providers]]
id = "ollama"
type = "ollama"
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["HOME"] = tmpdir
            conv_dir = Path(tmpdir) / "conversations"
            write_config(toml_text.format(conv=conv_dir.as_posix()), Path(tmpdir))
            with self.assertRaises(ValueError):
                load_config()

        # Legacy models missing contexts should raise
        legacy_text = """
[general]
default_provider = "ollama"
conversations_dir = "{conv}"

[ollama]
api_url = "http://localhost:11434/api"

[[models]]
name = "llama3"
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["HOME"] = tmpdir
            conv_dir = Path(tmpdir) / "conversations"
            write_config(legacy_text.format(conv=conv_dir.as_posix()), Path(tmpdir))
            with self.assertRaises(ValueError):
                load_config()


if __name__ == "__main__":
    unittest.main()
