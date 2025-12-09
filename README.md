# ScriptChat

A terminal-based, scriptable chat client for interacting with local LLMs (via Ollama)
and remote LLMs via OpenAI-compatible providers (OpenAI, DeepSeek...) and Anthropic Claude.

## Why ScriptChat?

ScriptChat fills the gap between writing code against LLM APIs and using GUI chat interfaces. APIs give you control but high friction; GUIs are convenient but hard to automate. ScriptChat gives you both: an interactive TUI for exploration, and scriptable automation for testing and iteration.

- **Scriptable**: Write `.sc` scripts with variables and assertions. Pipe into shell workflows. Run prompt regression tests in CI.
- **Branchable**: Save a conversation, branch it, try different approaches, compare results.
- **Multi-provider**: Same workflow across Ollama, OpenAI, Anthropic. Switch models mid-conversation.
- **File-based**: Conversations are directories of text files. Inspect, edit, or version control them.

For developers and power users who want their LLM interactions to be reproducible, scriptable, and under their control.

## Features

- Full-screen terminal UI with conversation history, status bar, and input pane
- Extended thinking support for reasoning models (`/reason`, `/thinking`)
- File references: register files and include them in prompts (`/file`, `@path`)
- Export conversations to Markdown, JSON, or HTML
- Multi-line message support
- Token usage tracking and temperature control
- System prompts per conversation

## Requirements

- Python 3.10+
- Optional: [Ollama](https://ollama.com) installed and accessible in your PATH

## Installation

### Via pipx (recommended)

```bash
pipx install scriptchat
```

This installs the `scriptchat` command (and the shorter `sc` alias) globally.

### Via pip

```bash
pip install scriptchat
```

### From source

1. Clone this repository:
   ```bash
   git clone https://github.com/scriptchat/scriptchat.git
   cd scriptchat
   ```

2. Install in development mode:
   ```bash
   pip install -e ".[dev]"
   ```

### Configuration

After installation, create your configuration file:

```bash
mkdir -p ~/.scriptchat
# Copy the example config (if installing from source)
cp config.toml.example ~/.scriptchat/config.toml
# Or create a new one based on the example in the repository
```

Edit `~/.scriptchat/config.toml` to configure your models and preferences. See `config.toml.example` for a complete example.

Key configuration options:

- `default_provider`: Provider id to use on startup (see `[[providers]]`)
- `default_model`: Model to use on startup (for the default provider)
- `default_temperature`: Temperature for new conversations (0.0-2.0)
- `system_prompt`: System prompt for all conversations (override per conversation with `/prompt`)
- `conversations_dir`: Where to store conversations (set in `[general]`)
- `exports_dir`: Where to write exports (defaults to current working directory if not set)
- `enable_streaming`: Enable token streaming (default: false)
- `[[providers]]`: List of model providers. Each has an `id`, `type` (`ollama`, `openai-compatible`, or `anthropic`), `api_url`, optional `api_key`, `models` (comma-separated or list of tables), optional `default_model`, and optional `streaming`/`headers`. A model entry can include `contexts` (for Ollama), `reasoning_levels` (for `/reason` on reasoning-capable models), and `reasoning_default` to pick the level applied when you select that model.

**API Keys:** If `api_key` is not set in config, ScriptChat will look for `{PROVIDER_ID}_API_KEY` environment variable (e.g., `OPENAI_API_KEY`, `DEEPSEEK_API_KEY`, `ANTHROPIC_API_KEY`).

Example providers:
```toml
[general]
default_provider = "ollama"
default_model = "llama3"

[[providers]]
id = "ollama"
type = "ollama"
api_url = "http://localhost:11434/api"
models = "llama3,phi3"

[[providers]]
id = "openai"
type = "openai-compatible"
api_url = "https://api.openai.com"
api_key = "sk-..."
models = [
  { name = "gpt-5-mini", reasoning_levels = ["minimal", "medium", "high"], reasoning_default = "medium" },
  { name = "gpt-5.1-mini", reasoning_levels = ["none", "low", "medium", "high"], reasoning_default = "none" }
]

[[providers]]
id = "deepseek"
type = "openai-compatible"
api_url = "https://api.deepseek.com"
api_key = "sk-..."
models = "deepseek-chat,deepseek-coder"

[[providers]]
id = "anthropic"
type = "anthropic"
api_url = "https://api.anthropic.com"
api_key = "sk-ant-..."
models = [
  { name = "claude-sonnet-4-20250514", reasoning_levels = ["low", "medium", "high", "max"] },
  { name = "claude-3-5-sonnet-20241022" }
]
```

## Usage

Run ScriptChat:

```bash
scriptchat
# or use the shorter alias:
sc
```

In batch mode you can allow assertions to log but keep running using `--continue-on-error` (exit code will still be 1 if any assertion fails):

```bash
scriptchat --run tests/demo.txt --continue-on-error
```

You can also pipe a script via stdin (no `--run` needed):

```bash
cat tests/demo.txt | scriptchat
```

Check version:

```bash
scriptchat --version
```

### Commands

All commands start with `/`:

**Conversation**
- `/new` - Start a new conversation
- `/save` - Save the current conversation
- `/load [index|name]` - Load a saved conversation by index or name
- `/branch` - Create a branch (copy) of the current conversation
- `/rename` - Rename a saved conversation (renames its directory)
- `/chats` - List saved conversations
- `/clear [index]` - Clear and delete the current conversation, or delete a saved conversation by index (requires confirmation)

**Export/Import**
- `/export [format]` - Export the current conversation (formats: `md`, `json`, `html`; prompts if omitted). `json` includes full metadata; `md`/`html` are minimal, human-friendly transcripts.
- `/export-all [format]` - Export all saved conversations in the given format.
- `/import <path>` - Import a conversation exported as `md` or `json` into the conversations folder

**Model & Settings**
- `/model` - Switch to a different model (for the current provider)
- `/temp` - Change the temperature setting
- `/reason [level]` - Show or set the reasoning level (`low`, `medium`, `high`, `max`). For Anthropic Claude, these map to thinking budgets (4K, 16K, 32K, 55K tokens). `/reason` alone lists available levels.
- `/thinking [tokens]` - Set exact thinking budget in tokens for Anthropic Claude (1024-128000). Use `/thinking off` to disable. Overrides `/reason` presets.
- `/timeout <seconds|0|off>` - Set the request timeout in seconds, or disable with `0` or `off`
- `/stream [on|off]` - Toggle or set streaming of assistant responses
- `/prompt [text|clear]` - Set or clear the system prompt for this conversation (prompts if omitted)

**Files**
- `/file [--force] <path>` - Register a file for use as `@path` in messages (content is expanded when sending, message stores `@path`). Use `--force` for large files above `file_confirm_threshold_bytes`.
- `/files [--long]` - List registered files (with sizes and hashes when using `--long`)

**Tags**
- `/tag key=value` - Apply metadata tags to the conversation (shown in `/chats` and `/load`)
- `/untag <key>` - Remove a metadata tag from the conversation
- `/tags` - List tags on the current conversation

**Messaging**
- `/send <message>` - Queue a message (sends immediately if the model is idle)
- `/history [n|all]` - Show recent user messages in current conversation (persists if saved/loaded; default: last 10)
- `/note <text>` - Add a note to the conversation (saved and visible, but not sent to model)
- `/undo [n]` - Remove the last user/assistant exchange(s) from the conversation. Without n, it removes 1.
- `/retry` - Drop the last assistant message and resend the previous user message

**Testing & Debug**
- `/assert <pattern>` - Assert the last assistant response contains the given text/regex (exits with error in batch mode). `/assert` checks only the last assistant message; it's case-insensitive and treats the pattern as a regex (falls back to substring if the regex is invalid).
- `/assert-not <pattern>` - Assert the last assistant response does NOT contain the text/regex (same matching rules as `/assert`).
- `/echo <text>` - Print a message to the console without sending to the model
- `/log-level <debug|info|warn|error|critical>` - Adjust runtime logging verbosity without restarting
- `/profile [--full]` - Show current provider/model/temp, tokens, streaming/timeout, and registered files. Use `--full` to show complete system prompt

**Scripting**
- `/run <path>` - Execute a script file (one command/message per line; lines starting with `#` are comments)
- `/sleep <seconds>` - Pause execution for the specified duration (scripts/batch mode only)
- `/set <name>=<value>` - Define a script variable for use with `${name}` syntax
- `/vars` - List all defined variables

**System**
- `/help [command|keyword]` - Show help for all commands, a specific command, or search by keyword.
- `/keys` - Show keyboard shortcuts
- `/exit` - Exit ScriptChat

### Multi-line Messages

To enter a multi-line message:

1. Type `"""` and press Enter
2. Enter your message across multiple lines
3. Type `"""` on a new line to send

This syntax also works in script files (`/run` or `--run`):

```bash
"""
Analyze this code for:
- Security issues
- Performance problems
"""
```

### Script Variables

Use `/set` to define variables that can be referenced with `${name}` syntax:

```bash
/set model=llama3
/model ${model}
/set greeting=Hello, how are you?
${greeting}
```

Variables are expanded in both commands and messages. Variable names must start with a letter or underscore and contain only letters, numbers, and underscores.

**Environment variable fallback:** If a variable isn't defined via `/set`, ScriptChat checks environment variables. This enables parameterized scripts:

```bash
# Run with different configurations
LANGUAGE=Python TOPIC="error handling" scriptchat --run test.sc
LANGUAGE=Rust TOPIC="memory safety" scriptchat --run test.sc
```

Script variables (`/set`) take precedence over environment variables. Unknown variables are left unexpanded.

**Security:** Sensitive environment variables matching patterns like `*_KEY`, `*_SECRET`, `*_TOKEN`, `*_PASSWORD` are blocked from expansion by default. Configure in `config.toml`:

```toml
[general]
# Disable env var expansion entirely
env_expand_from_environment = false

# Or override the default blocklist with your own patterns
env_var_blocklist = ["MY_PRIVATE_*", "INTERNAL_*"]

# Use empty list to allow all env vars (no blocklist)
env_var_blocklist = []
```

### Keyboard Shortcuts

**Navigation**
- `Ctrl+Up` - Focus conversation pane for scrolling
- `Ctrl+Home` - Jump to start of conversation
- `Ctrl+End` - Jump to end of conversation
- `Escape` - Return focus to input pane
- `Tab` - Return to input (when in conversation pane)

**In conversation pane**
- `Up/Down` - Scroll line by line

**In input pane**
- `Up/Down` - Navigate command history
- `Tab` - Command completion
- `Shift+Tab` - Reverse completion cycling

**General**
- `Ctrl+C` or `Ctrl+D` - Exit ScriptChat
- `Escape` twice (within 2s) - Cancel ongoing inference

### File References

`/file [--force] <path>` registers a file for use in messages. Include `@path` in any user message to inline the file contents when sending (the stored message keeps `@path` for readability). Examples:

- Register: `/file docs/plan.md`
- Send with inline file: `Summarize @docs/plan.md and list action items.` (you can also use `@{docs/plan.md}` or `@plan.md` if unique)
- If an `@path` isn’t registered, the send will error and nothing is sent.

You can register multiple files and mix references in one message. `/profile` lists full paths of registered files.

### Conversation Storage

Conversations are stored in `~/.scriptchat/conversations/` (or `conversations_dir` in config) with the following structure:

- Each conversation is a directory: `YYYYMMDDHHMM_modelname_savename/`
- Messages are stored as individual files: `0001_user.txt`, `0002_llm.txt`, etc.
- Metadata is stored in `meta.json`

You can manually edit message files or delete them as needed.

Exports (`/export`) go to the current working directory by default, or to `exports_dir` if configured.

## Example Workflow

1. Start ScriptChat: `python -m scriptchat`
2. Chat with the default model
3. Save your conversation: `/save` then enter a name
4. Switch models: `/model` then select a model
5. Continue chatting with the new model
6. Rename or branch to organize: `/rename new-name` or `/branch`
7. Load a previous conversation: `/load`
8. Exit when done: `/exit` or Ctrl+C

See the [`examples/`](examples/) folder for runnable scripts demonstrating interactive workflows and batch testing.

## Status Bar

The status bar shows:
- Provider and model (e.g., `ollama/llama3.2`) with optional reasoning level in parentheses
- Token usage (input/output), with optional context usage percentage
- Conversation ID (or `<unsaved>` for new conversations)
- Thinking indicator when the model is processing

Example: `ollama/llama3.2 (high) | 1234 in / 567 out | 202511180945_llama32_my-chat`

## Troubleshooting

**"Ollama executable not found"**: Ensure Ollama is installed and in your PATH.

**"Configuration file not found"**: Create `~/.scriptchat/config.toml` from the example.

**"An Ollama instance is already running"**: ScriptChat detects if Ollama is already running on the configured port. You can:
- Use the existing instance (recommended for most cases)
- Start a new ScriptChat-managed instance on an alternative port
- Exit and stop the existing instance first

**Connection errors**: If using an external Ollama instance, check that it's still running. If ScriptChat manages the instance, check that Ollama is installed and the port is available.

**Model not found**: Make sure the model is pulled in Ollama (`ollama pull modelname`) and configured in `config.toml`.

## License

Apache License 2.0 (see `LICENSE`). Attribution details are in `NOTICE`.
