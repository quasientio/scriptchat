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

import os
import tempfile
import unittest
from pathlib import Path

from scriptchat.core.config import load_config


def write_config(text: str, dirpath: Path):
    cfg_dir = dirpath / ".scriptchat"
    cfg_dir.mkdir()
    (cfg_dir / "config.toml").write_text(text, encoding="utf-8")


class ConfigProviderTests(unittest.TestCase):
    def test_loads_providers_and_default_provider(self):
        toml_text = """
[general]
default_model = "ollama/llama3.1"
conversations_dir = "{conv}"

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
        # default_model with non-existent provider should raise
        toml_text = """
[general]
default_model = "missing/somemodel"
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
        # Missing required provider fields (api_url)
        toml_text = """
[general]
default_model = "ollama/llama3"
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

        # Legacy models missing context should raise
        legacy_text = """
[general]
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

    def test_model_alias_parsing(self):
        """Test that model aliases are parsed correctly."""
        toml_text = """
[general]
default_model = "fireworks/deepseek-v3"
conversations_dir = "{conv}"

[[providers]]
id = "fireworks"
type = "openai-compatible"
api_url = "https://api.fireworks.ai/inference"
models = [
  {{ name = "deepseek-v3", alias = "dsv3" }},
  {{ name = "deepseek-r1", alias = "dsr1" }}
]
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["HOME"] = tmpdir
            conv_dir = Path(tmpdir) / "conversations"
            text = toml_text.format(conv=conv_dir.as_posix())
            write_config(text, Path(tmpdir))
            cfg = load_config()
            models = cfg.list_models("fireworks")
            self.assertEqual(models[0].alias, "dsv3")
            self.assertEqual(models[1].alias, "dsr1")

    def test_model_alias_resolve(self):
        """Test that resolve_alias returns the correct provider/model."""
        toml_text = """
[general]
default_model = "fireworks/deepseek-v3"
conversations_dir = "{conv}"

[[providers]]
id = "fireworks"
type = "openai-compatible"
api_url = "https://api.fireworks.ai/inference"
models = [
  {{ name = "deepseek-v3", alias = "dsv3" }}
]

[[providers]]
id = "ollama"
type = "ollama"
api_url = "http://localhost:11434/api"
models = [
  {{ name = "llama3", alias = "ll3" }}
]
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["HOME"] = tmpdir
            conv_dir = Path(tmpdir) / "conversations"
            text = toml_text.format(conv=conv_dir.as_posix())
            write_config(text, Path(tmpdir))
            cfg = load_config()
            # Resolve alias to provider/model
            result = cfg.resolve_alias("dsv3")
            self.assertEqual(result, ("fireworks", "deepseek-v3"))
            result = cfg.resolve_alias("ll3")
            self.assertEqual(result, ("ollama", "llama3"))
            # Unknown alias returns None
            result = cfg.resolve_alias("unknown")
            self.assertIsNone(result)

    def test_model_alias_invalid_format_raises(self):
        """Test that aliases with invalid characters raise ValueError."""
        toml_text = """
[general]
default_model = "fireworks/deepseek-v3"
conversations_dir = "{conv}"

[[providers]]
id = "fireworks"
type = "openai-compatible"
api_url = "https://api.fireworks.ai/inference"
models = [
  {{ name = "deepseek-v3", alias = "ds/v3" }}
]
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["HOME"] = tmpdir
            conv_dir = Path(tmpdir) / "conversations"
            text = toml_text.format(conv=conv_dir.as_posix())
            write_config(text, Path(tmpdir))
            with self.assertRaises(ValueError) as ctx:
                load_config()
            self.assertIn("Invalid alias", str(ctx.exception))
            self.assertIn("ds/v3", str(ctx.exception))

    def test_model_alias_duplicate_raises(self):
        """Test that duplicate aliases raise ValueError."""
        toml_text = """
[general]
default_model = "fireworks/deepseek-v3"
conversations_dir = "{conv}"

[[providers]]
id = "fireworks"
type = "openai-compatible"
api_url = "https://api.fireworks.ai/inference"
models = [
  {{ name = "deepseek-v3", alias = "ds" }},
  {{ name = "deepseek-r1", alias = "ds" }}
]
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["HOME"] = tmpdir
            conv_dir = Path(tmpdir) / "conversations"
            text = toml_text.format(conv=conv_dir.as_posix())
            write_config(text, Path(tmpdir))
            with self.assertRaises(ValueError) as ctx:
                load_config()
            self.assertIn("Duplicate alias", str(ctx.exception))
            self.assertIn("ds", str(ctx.exception))

    def test_model_alias_duplicate_across_providers_raises(self):
        """Test that duplicate aliases across providers raise ValueError."""
        toml_text = """
[general]
default_model = "fireworks/deepseek-v3"
conversations_dir = "{conv}"

[[providers]]
id = "fireworks"
type = "openai-compatible"
api_url = "https://api.fireworks.ai/inference"
models = [
  {{ name = "deepseek-v3", alias = "mymodel" }}
]

[[providers]]
id = "ollama"
type = "ollama"
api_url = "http://localhost:11434/api"
models = [
  {{ name = "llama3", alias = "mymodel" }}
]
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["HOME"] = tmpdir
            conv_dir = Path(tmpdir) / "conversations"
            text = toml_text.format(conv=conv_dir.as_posix())
            write_config(text, Path(tmpdir))
            with self.assertRaises(ValueError) as ctx:
                load_config()
            self.assertIn("Duplicate alias", str(ctx.exception))
            self.assertIn("mymodel", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
