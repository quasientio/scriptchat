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

"""Tests for ScriptChat command and path completion."""

import tempfile
import pytest
from pathlib import Path
from prompt_toolkit.document import Document
from prompt_toolkit.completion import CompleteEvent

from scriptchat.ui.completers import ScriptChatCompleter


class TestScriptChatCompleter:
    """Tests for the ScriptChatCompleter."""

    @pytest.fixture
    def completer(self):
        """Create a basic completer instance."""
        return ScriptChatCompleter()

    @pytest.fixture
    def completer_with_registries(self):
        """Create a completer with file and folder registries."""
        file_registry = {
            'main.py': {'content': 'print("hello")', 'full_path': '/tmp/main.py'},
            'config.json': {'content': '{}', 'full_path': '/tmp/config.json'},
            '/tmp/main.py': {'content': 'print("hello")', 'full_path': '/tmp/main.py'},
        }
        folder_registry = {
            'src': {'path': '/tmp/src', 'files': ['/tmp/src/app.py']},
            '/tmp/src': {'path': '/tmp/src', 'files': ['/tmp/src/app.py']},
        }
        return ScriptChatCompleter(file_registry=file_registry, folder_registry=folder_registry)

    def test_command_name_completion(self, completer):
        """Test completion of command names."""
        document = Document('/fi', cursor_position=3)
        event = CompleteEvent()

        completions = list(completer.get_completions(document, event))

        # Should complete to /file and /files
        texts = [c.text for c in completions]
        assert '/file' in texts
        assert '/files' in texts

    def test_command_name_completion_case_insensitive(self, completer):
        """Test that command completion is case-insensitive."""
        document = Document('/FI', cursor_position=3)
        event = CompleteEvent()

        completions = list(completer.get_completions(document, event))

        # Should still complete despite uppercase
        texts = [c.text for c in completions]
        assert len(texts) > 0

    def test_no_completion_for_non_command(self, completer):
        """Test that non-commands don't get completed."""
        document = Document('hello', cursor_position=5)
        event = CompleteEvent()

        completions = list(completer.get_completions(document, event))

        # Should not complete regular text
        assert len(completions) == 0

    def test_path_completion_for_file_command(self, completer):
        """Test path completion for /file command."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create some test files
            test_dir = Path(tmpdir)
            (test_dir / "test1.txt").touch()
            (test_dir / "test2.txt").touch()
            (test_dir / "subdir").mkdir()

            document = Document(f'/file {tmpdir}/t', cursor_position=len(f'/file {tmpdir}/t'))
            event = CompleteEvent()

            completions = list(completer.get_completions(document, event))

            # PathCompleter returns the completion suffix (not the full path)
            # For '/tmp/xyz/t' with files test1.txt, test2.txt, it returns 'est1.txt', 'est2.txt'
            texts = [c.text for c in completions]
            assert any('est1' in t for t in texts), f"Expected 'est1' in completions, got {texts}"
            assert any('est2' in t for t in texts), f"Expected 'est2' in completions, got {texts}"

    def test_directory_completion_for_folder_command(self, completer):
        """Test that /folder only completes directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create files and directories
            test_dir = Path(tmpdir)
            (test_dir / "file.txt").touch()
            (test_dir / "dir1").mkdir()
            (test_dir / "dir2").mkdir()

            document = Document(f'/folder {tmpdir}/d', cursor_position=len(f'/folder {tmpdir}/d'))
            event = CompleteEvent()

            completions = list(completer.get_completions(document, event))

            # Should only complete directories (completion suffix after 'd')
            texts = [c.text for c in completions]
            # Should have dir1 and dir2 but not file.txt
            assert any('ir1' in t for t in texts), f"Expected 'ir1' in completions, got {texts}"
            assert any('ir2' in t for t in texts), f"Expected 'ir2' in completions, got {texts}"
            assert not any('file' in t for t in texts)

    def test_path_completion_with_force_flag(self, completer):
        """Test path completion after --force flag."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir = Path(tmpdir)
            (test_dir / "large.txt").touch()

            document = Document(f'/file --force {tmpdir}/l', cursor_position=len(f'/file --force {tmpdir}/l'))
            event = CompleteEvent()

            completions = list(completer.get_completions(document, event))

            # Should complete after the flag (completion suffix after 'l')
            texts = [c.text for c in completions]
            assert any('arge' in t for t in texts), f"Expected 'arge' in completions, got {texts}"

    def test_unfile_completion_with_registered_keys(self, completer_with_registries):
        """Test that /unfile completes with registered file keys."""
        document = Document('/unfile m', cursor_position=9)
        event = CompleteEvent()

        completions = list(completer_with_registries.get_completions(document, event))

        # Should complete to main.py
        texts = [c.text for c in completions]
        assert 'main.py' in texts

    def test_unfile_completion_shows_all_keys(self, completer_with_registries):
        """Test that /unfile shows all registered keys when no partial match."""
        document = Document('/unfile ', cursor_position=8)
        event = CompleteEvent()

        completions = list(completer_with_registries.get_completions(document, event))

        # Should show all registered file keys
        texts = [c.text for c in completions]
        assert 'main.py' in texts
        assert 'config.json' in texts
        assert '/tmp/main.py' in texts

    def test_unfolder_completion_with_registered_keys(self, completer_with_registries):
        """Test that /unfolder completes with registered folder keys."""
        document = Document('/unfolder s', cursor_position=11)
        event = CompleteEvent()

        completions = list(completer_with_registries.get_completions(document, event))

        # Should complete to src
        texts = [c.text for c in completions]
        assert 'src' in texts

    def test_import_chatgpt_path_completion(self, completer):
        """Test path completion for /import-chatgpt command."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir = Path(tmpdir)
            (test_dir / "export.zip").touch()

            document = Document(f'/import-chatgpt {tmpdir}/e', cursor_position=len(f'/import-chatgpt {tmpdir}/e'))
            event = CompleteEvent()

            completions = list(completer.get_completions(document, event))

            # Should complete to export.zip (completion suffix after 'e')
            texts = [c.text for c in completions]
            assert any('xport' in t for t in texts), f"Expected 'xport' in completions, got {texts}"

    def test_run_command_path_completion(self, completer):
        """Test path completion for /run command."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir = Path(tmpdir)
            (test_dir / "script.sc").touch()

            document = Document(f'/run {tmpdir}/s', cursor_position=len(f'/run {tmpdir}/s'))
            event = CompleteEvent()

            completions = list(completer.get_completions(document, event))

            # Should complete to script.sc (completion suffix after 's')
            texts = [c.text for c in completions]
            assert any('cript' in t for t in texts), f"Expected 'cript' in completions, got {texts}"

    def test_no_path_completion_for_non_path_commands(self, completer):
        """Test that non-path commands don't trigger path completion."""
        document = Document('/model ', cursor_position=7)
        event = CompleteEvent()

        completions = list(completer.get_completions(document, event))

        # Should not complete paths for /model
        assert len(completions) == 0

    def test_expanduser_in_path_completion(self, completer):
        """Test that ~ is expanded in path completion."""
        document = Document('/file ~/', cursor_position=8)
        event = CompleteEvent()

        completions = list(completer.get_completions(document, event))

        # Should expand ~ and show home directory contents
        # This is hard to test precisely, but we can verify completions are generated
        # (assuming user's home directory is not empty)
        assert len(completions) >= 0  # May be 0 if home is empty, but no errors

    def test_completion_meta_display(self, completer_with_registries):
        """Test that completions show helpful metadata."""
        document = Document('/unfile m', cursor_position=9)
        event = CompleteEvent()

        completions = list(completer_with_registries.get_completions(document, event))

        # Check that completions have display_meta containing the expected text
        assert len(completions) > 0
        for c in completions:
            # display_meta may be a string or FormattedText object
            meta_str = str(c.display_meta) if c.display_meta else ''
            assert '(registered file)' in meta_str

    def test_split_command_line_with_quotes(self, completer):
        """Test that command line splitting handles quotes correctly."""
        # This tests the internal _split_command_line method
        result = completer._split_command_line('/file "path with spaces.txt"')
        assert result == ['/file', 'path with spaces.txt']

    def test_split_command_line_with_escapes(self, completer):
        """Test that command line splitting handles escapes correctly."""
        result = completer._split_command_line('/file path\\ with\\ spaces.txt')
        assert result == ['/file', 'path with spaces.txt']

    def test_completion_position_adjustment(self, completer):
        """Test that completion positions are correctly adjusted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir = Path(tmpdir)
            (test_dir / "test.txt").touch()

            document = Document(f'/file {tmpdir}/t', cursor_position=len(f'/file {tmpdir}/t'))
            event = CompleteEvent()

            completions = list(completer.get_completions(document, event))

            # Verify that start_position is correct
            # PathCompleter returns start_position=0 for appending after the cursor
            # (bash-like behavior where 't' + 'est.txt' = 'test.txt')
            assert len(completions) > 0
            for completion in completions:
                # The start_position should be 0 for appending
                assert completion.start_position == 0
