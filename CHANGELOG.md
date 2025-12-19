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

### Added
- Basic markdown rendering in UI: headers (`#`-`######`), `**bold**`, and `` `code` `` are now styled in assistant and system messages
- `/models` command to list all configured models by provider with aliases, context, and reasoning info
- Model aliases: optional `alias` field in model config for shorter `/model` references
  (e.g., `alias = "dsv3"` lets you use `/model dsv3` instead of `/model fireworks/accounts/fireworks/models/deepseek-v3`)
- `/unfile <key>` command to unregister files added with `/file`
- Context usage warning colors in status bar: yellow at 75%, red at 90%
- Token estimation for `/file` command shows chars, tokens, and context percentage
- `tiktoken` dependency for accurate OpenAI token counting
- Optional `deepseek` extra for DeepSeek tokenizer (`pip install scriptchat[deepseek]`)
- `auth_format` provider config option for non-Bearer auth (e.g., Baseten uses `api-key`)
- Baseten provider example in config.toml.example
- `max_tokens` model config option for controlling output token limit (important for thinking models)
- Thinking/reasoning model support for DeepSeek, Anthropic Claude, and Fireworks models:
  - Thinking content displayed in UI with `<thinking>` tags (gray, streams in real-time)
  - Thinking content saved to `NNNN_llm_thinking.txt` files alongside responses
  - Anthropic extended thinking via `thinking_delta` events
  - DeepSeek R1 `<think>` tag extraction
  - `reasoning_content` field capture for compatible providers
- `include_thinking_in_history` config option to include thinking in messages sent to API
- Nested file reference expansion: `@path` references inside registered files are also expanded (one level deep)
- `reasoning_effort` support for Fireworks models (e.g., Kimi K2 Thinking with `/reason low|medium|high`)
- `skip_prompt_cache_param` model config option for models that don't support `prompt_cache_max_len` parameter

### Changed
- Input area now supports multiline editing with UP/DOWN/LEFT/RIGHT navigation
- LEFT/RIGHT arrows cross line boundaries for seamless multiline editing
- Auto-enable streaming when `max_tokens > 4096` (required by some providers like Fireworks)
- Alt+Enter or Ctrl+J inserts a newline without sending the message
- Input area height capped at 60% of terminal height with scrolling support
- Multiline mode (`"""`) now keeps text visible and editable in input area

### Fixed
- `/model` no longer shows duplicate provider info when switching via alias or cross-provider
- `/model` error message now suggests using `/model` without args or `provider/model` format instead of listing current provider's models
- Ignore hidden directories inside conversations (e.g. .git if convos are versioned)
- Strip leaked stop tokens (`<|im_end|>`, `<|endoftext|>`, etc.) from model responses
- Avoid duplicate `/v1` in API URLs when api_url already includes version
- `/profile` now shows context from built-in model defaults when not explicitly configured
- Handle empty `choices` array in streaming responses (prevents IndexError with some providers)
- Error messages now include provider and model context for easier debugging
- Batch mode now displays thinking content with `<thinking>` tags
- Variable expansion (`${name}`) now works in interactive UI mode, not just batch/script mode

## [0.3.0] - 2025-12-12

### Fixed
- `/retry` and saving works now, removing last response and without user message duplicate
- `/undo` and saving deletes stale message files, saved state matches in-memory state
- `/log-level` now working as intended (was only updating root logger)
- UI status messages no longer sent to LLM (was using `role='system'`, now uses `role='status'`)
- ESC+ESC to cancel inference now properly discards the response instead of showing it
- File reference inside triple backticks now expands

### Added
- Context usage percentage in status bar for all providers (OpenAI, Anthropic, Ollama)
- Built-in context window limits for well-known models (no config required)
- `scripts/update_model_defaults.py` to refresh model limits from upstream
- New openai compatible provider config entry for Fireworks AI
- DEBUG-level logging statements with responses metadata -- content filtered out
- Configurable `prompt_cache` setting for privacy for fireworks provider
- `/import-chatgpt <path> [--dry-run]` to import conversations from a ChatGPT export
  ZIP file. Use `--dry-run` to preview without saving.
- `/archive` and `/unarchive` commands to organize conversations. Supports index,
  name, range (`1-5`), and `--tag key=value` filter. Archived conversations are
  stored in `.archive/` subdirectory.
- `/chats --archived` and `/chats --all` flags to list archived or all conversations
- `/load --archived` flag to load conversations from the archive
- Interactive selection menu for `/model`, `/load`, `/reason`, and `/log-level`
  commands. Navigate with arrow keys or j/k, select with Enter/Tab, cancel with Escape.
- Typing ESC in the input area clears message or command typed so far
- `--init` flag for interactive first-run configuration setup. Supports Ollama,
  OpenAI, Anthropic, and DeepSeek providers with guided API key/model setup.

### Changed
- Model config `contexts` (list) simplified to `context` (single int) in config.toml
- `default_provider` + `default_model` merged into single `default_model` setting
  using `provider/model` format (e.g., `default_model = "ollama/llama3.2"`)
- `/model`, `/load`, `/reason`, `/log-level` no longer accept index arguments;
  use the interactive menu or pass a name directly (e.g., `/model ollama/llama3`)
- Set store=false in openai Responses API to prevent server-side storing of conversations
- Ollama: no longer spawns/manages server process. Uses `num_ctx` per-request instead
  of `OLLAMA_CONTEXT_LENGTH` environment variable. Shows warning on startup if Ollama
  is not running.

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
