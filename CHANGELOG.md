# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- This CHANGELOG file
- A CONTRIBUTING.md guide

### Fixed
- `/rename` now preserves timestamp prefix in conversation directory names
- `/rename` now updates `parent_id` references in child conversations (branches)
- `/save` now adds numeric suffix instead of overwriting when names collide (same name within same minute)

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
