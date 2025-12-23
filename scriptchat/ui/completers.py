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

"""Custom completers for ScriptChat commands."""

from typing import Iterable, TYPE_CHECKING

from prompt_toolkit.completion import Completer, Completion, PathCompleter, WordCompleter
from prompt_toolkit.document import Document

if TYPE_CHECKING:
    from prompt_toolkit.completion import CompleteEvent


# Commands that take file paths as arguments
PATH_COMMANDS = {
    '/file': {'only_directories': False, 'after_flags': ['--force']},
    '/folder': {'only_directories': True, 'after_flags': ['--force']},
    '/import': {'only_directories': False},
    '/import-chatgpt': {'only_directories': False},
    '/run': {'only_directories': False},
}


class ScriptChatCompleter(Completer):
    """Custom completer that provides command completion and context-aware path completion.

    This completer:
    - Completes command names when typing at the start of input (e.g., /fi -> /file)
    - Completes file paths when inside path-taking commands (e.g., /file ~/doc -> ~/documents/)
    - Supports bash-like TAB behavior: stops at common prefix, double-TAB shows options
    - Completes registered file/folder keys for /unfile and /unfolder
    """

    def __init__(self, file_registry: dict = None, folder_registry: dict = None):
        """Initialize the completer.

        Args:
            file_registry: Dict mapping file keys to file info (for /unfile completion)
            folder_registry: Dict mapping folder keys to folder info (for /unfolder completion)
        """
        self.file_registry = file_registry or {}
        self.folder_registry = folder_registry or {}

        # Command name completer
        self.command_completer = WordCompleter(
            ['/new', '/save', '/open', '/branch', '/rename', '/chats', '/archive',
             '/unarchive', '/send', '/history', '/export', '/export-all', '/import',
             '/import-chatgpt', '/stream', '/prompt', '/run', '/sleep', '/model',
             '/models', '/temp', '/reason', '/thinking', '/think-history', '/timeout',
             '/profile', '/log-level', '/files', '/clear', '/file', '/unfile', '/folder',
             '/unfolder', '/echo', '/note', '/tag', '/untag', '/tags', '/set', '/vars',
             '/assert', '/assert-not', '/undo', '/retry', '/help', '/keys', '/exit'],
            ignore_case=True,
            sentence=True
        )

        # Path completers for different scenarios
        self.path_completer = PathCompleter(expanduser=True)
        self.dir_completer = PathCompleter(only_directories=True, expanduser=True)

    def get_completions(self, document: 'Document', complete_event: 'CompleteEvent') -> 'Iterable[Completion]':
        """Generate completions based on the current input context.

        Args:
            document: Current document state
            complete_event: Completion event

        Yields:
            Completion objects
        """
        text = document.text_before_cursor

        # If empty or not a command, no completion
        if not text or not text.startswith('/'):
            return

        # Split into words, respecting quotes and handling flags
        words = self._split_command_line(text)

        if not words:
            return

        command = words[0].lower()

        # If we're still typing the command name (no space after it yet)
        if len(words) == 1 and not text.endswith(' '):
            # Complete command names
            yield from self.command_completer.get_completions(document, complete_event)
            return

        # Handle path completion for path-taking commands
        if command in PATH_COMMANDS:
            yield from self._complete_path_argument(document, words, command, complete_event)
            return

        # Handle /unfile completion (complete with registered file keys)
        if command == '/unfile':
            yield from self._complete_file_key(document, words, complete_event)
            return

        # Handle /unfolder completion (complete with registered folder keys)
        if command == '/unfolder':
            yield from self._complete_folder_key(document, words, complete_event)
            return

    def _split_command_line(self, text: str) -> list[str]:
        """Split command line into words, respecting quotes.

        Args:
            text: Command line text

        Returns:
            List of words
        """
        import shlex
        try:
            # Use shlex to properly handle quotes and escapes
            return shlex.split(text)
        except ValueError:
            # If there's an unclosed quote, split on spaces
            return text.split()

    def _complete_path_argument(
        self,
        document: 'Document',
        words: list[str],
        command: str,
        complete_event: 'CompleteEvent'
    ) -> 'Iterable[Completion]':
        """Complete path arguments for path-taking commands.

        Args:
            document: Current document
            words: Split command words
            command: Command name
            complete_event: Completion event

        Yields:
            Completion objects for paths
        """
        cmd_config = PATH_COMMANDS[command]
        after_flags = cmd_config.get('after_flags', [])
        only_directories = cmd_config.get('only_directories', False)

        # We should complete paths if:
        # 1. We have just the command + space (len(words) == 1)
        # 2. We're typing a path argument (len(words) >= 2)
        # 3. Or we have command + flag + space (len(words) == 2 and words[1] in after_flags)

        # Don't complete if we only have the command with no space after
        if len(words) == 1:
            return

        # Find where the path text starts in the original document
        text_before = document.text_before_cursor

        # Calculate the position where path completion should start
        # Start after the command name
        path_start = len(command)

        # Skip whitespace after command
        while path_start < len(text_before) and text_before[path_start] == ' ':
            path_start += 1

        # If there's a flag, skip it and following whitespace
        if len(words) >= 2 and words[1] in after_flags:
            # Skip the flag
            path_start += len(words[1])
            # Skip whitespace after flag
            while path_start < len(text_before) and text_before[path_start] == ' ':
                path_start += 1

        # Extract the path text being typed
        path_text = text_before[path_start:]

        # Create a document for just the path portion
        path_document = Document(path_text, cursor_position=len(path_text))

        # Use appropriate path completer
        completer = self.dir_completer if only_directories else self.path_completer

        # Get completions and yield them directly (they already have correct positions)
        for completion in completer.get_completions(path_document, complete_event):
            yield completion

    def _complete_file_key(
        self,
        document: 'Document',
        words: list[str],
        complete_event: 'CompleteEvent'
    ) -> 'Iterable[Completion]':
        """Complete registered file keys for /unfile command.

        Args:
            document: Current document
            words: Split command words
            complete_event: Completion event

        Yields:
            Completion objects for registered file keys
        """
        # Need at least command + space to start completing
        if len(words) < 1:
            return

        # Get the partial key being typed (everything after the command)
        text_before = document.text_before_cursor
        command = '/unfile'

        # Find where the key argument starts
        key_start = len(command)
        while key_start < len(text_before) and text_before[key_start] == ' ':
            key_start += 1

        partial = text_before[key_start:]

        # Find all matching keys
        for key in self.file_registry.keys():
            if key.startswith(partial):
                yield Completion(
                    text=key,
                    start_position=-len(partial),
                    display=key,
                    display_meta='(registered file)'
                )

    def _complete_folder_key(
        self,
        document: 'Document',
        words: list[str],
        complete_event: 'CompleteEvent'
    ) -> 'Iterable[Completion]':
        """Complete registered folder keys for /unfolder command.

        Args:
            document: Current document
            words: Split command words
            complete_event: Completion event

        Yields:
            Completion objects for registered folder keys
        """
        # Need at least command + space to start completing
        if len(words) < 1:
            return

        # Get the partial key being typed (everything after the command)
        text_before = document.text_before_cursor
        command = '/unfolder'

        # Find where the key argument starts
        key_start = len(command)
        while key_start < len(text_before) and text_before[key_start] == ' ':
            key_start += 1

        partial = text_before[key_start:]

        # Find all matching keys
        for key in self.folder_registry.keys():
            if key.startswith(partial):
                yield Completion(
                    text=key,
                    start_position=-len(partial),
                    display=key,
                    display_meta='(registered folder)'
                )
