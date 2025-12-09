# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Types of changes:
- **Added** for new features.
- **Changed** for changes in existing functionality.
- **Deprecated** for soon-to-be removed features.
- **Removed** for now removed features.
- **Fixed** for any bug fixes.
- **Security** in case of vulnerabilities.


## [Unreleased]

### Fixed
- /retry and saving works now, removing last response and without user message duplicate
- /undo and saving deletes stale message files, saved state matches in-memory state

## [0.2.0] - 2025-12-07

### Added
- This CHANGELOG file
- A CONTRIBUTING.md guide
- `/keys` to display keyboard shortcuts
- `/history` command to show recent user messages in the current conversation
- `/note <text>` to add a persistent note to the conversation, NOT sent to the model
- New `examples/` folder with runnable ScriptChat scripts:
  - Interactive: `prompt-engineering.sc`, `code-review.sc`, `research-reasoning.sc`
  - Batch: `ci-prompt-testing.sc`, `batch-testing.sc`, `run-batch-testing.sh`
  - Sample files for code review demo in `examples/files/`
- Environment variable fallback for `${VAR}` syntax (checks script variables first,
  then env vars)

### Changed
- `/timeout 0|off` disables the timeout
- `/load` now takes also as argument a conversation name (full or simple)
- `/profile --full` option to show complete prompt - no trimming applied

### Fixed
- `/rename` now preserves timestamp prefix in conversation directory names
- `/rename` now updates `parent_id` references in child conversations (branches)
- `/save` now adds numeric suffix instead of overwriting when names collide (same name within same minute)
- """-delimited input now handled correctly when pasted in

### Security
- Environment variable expansion blocks sensitive patterns (`*_KEY`,
  `*_SECRET`, `*_TOKEN`, `*_PASSWORD`, `*_CREDENTIAL`) by default. Configure via
  `env_var_blocklist` in config.toml.

## [0.1.0] - 2024-11-30

Initial public release.

### Added

#### Core
- Full-screen terminal UI with conversation history, status bar, and input pane
- Multi-line message support (triple quotes `"""`)
- Token usage tracking
- Temperature adjustment
- Input history recall with UP/DOWN keys
- ESC (twice) to cancel running inference

#### Providers
- Ollama support for local LLMs with automatic server management
- OpenAI-compatible API support (OpenAI, DeepSeek, etc.)
- Anthropic Claude support with extended thinking
- API keys via config or environment variables

#### Conversations
- Persistent file-based conversations (save, load, branch, rename)
- Conversation tagging with `/tag` and `/untag`
- Export to Markdown, JSON, and HTML formats
- Export all conversations with `/export-all`
- Import from Markdown and JSON exports
- HTML index page generation for exported conversations

#### Commands
- `/new`, `/save`, `/load`, `/branch`, `/rename`, `/clear`
- `/model`, `/temp`, `/stream`, `/timeout`
- `/prompt` for per-conversation system prompts
- `/file` for registering files, `@path` syntax for inline expansion
- `/send` to queue messages
- `/run` for executing script files
- `/export`, `/export-all`, `/import`
- `/reason` and `/thinking` for reasoning model controls
- `/assert`, `/assert-not` for batch testing
- `/profile`, `/chats`, `/files`, `/tags`, `/help`
- `/undo`, `/retry`, `/log-level`, `/echo`

#### Batch Mode
- Script execution via `--run` or stdin
- `--continue-on-error` flag for test scripts
- Assertions for automated testing

#### Packaging
- PyPI distribution (`pip install scriptchat`)
- Console entry points: `scriptchat` and `sc`
- `--version` flag
