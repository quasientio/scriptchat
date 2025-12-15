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

### From source

1. Clone this repository:
   ```bash
   git clone https://github.com/quasientio/scriptchat.git
   cd scriptchat
   ```

2. Install in development mode:
   ```bash
   pip install -e ".[dev]"
   ```

## Quickstart

**Step 1**: Run interactive setup (one-time)
```bash
scriptchat --init
```
 
**Step 2**: Test it immediately with a real example
```bash
# Download and run a working script
curl -s https://raw.githubusercontent.com/quasientio/scriptchat/main/examples/quickstart.sc | sc
```
 
**Step 3**: Start chatting
```bash
sc
```
 
**Step 4**: Explore more examples
[See the Examples Gallery →](#examples-gallery)

### Configuration

Edit `~/.scriptchat/config.toml` directly. See [`config.toml.example`](config.toml.example) for the full specification.

Key options:
- `default_model` - Model on startup in `provider/model` format (e.g., `ollama/llama3.2`)
- `default_temperature` - Temperature for new conversations (0.0-2.0)
- `system_prompt` - Default system prompt (override with `/prompt`)
- `[[providers]]` - Provider configs with `id`, `type`, `api_url`, `models`

**API Keys:** Set `api_key` in config or use environment variables (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc.).

Minimal example:
```toml
[general]
default_model = "ollama/llama3"

[[providers]]
id = "ollama"
type = "ollama"
api_url = "http://localhost:11434/api"
models = "llama3,phi3"
```

## Usage

### Batch Mode

Run a script file:

```bash
scriptchat --run script.sc
```

Use `--continue-on-error` to run all assertions even if some fail (exit code is still 1 if any fail):

```bash
scriptchat --run tests/prompt-tests.sc --continue-on-error
```

### CLI Flags

| Flag | Description |
|------|-------------|
| `--init` | Interactive configuration setup |
| `--run FILE` | Run a script file in batch mode |
| `--continue-on-error` | Don't stop on assertion failures |
| `--version` | Show version |

### Exit Codes

| Code | Meaning |
|------|---------|
| `0` | Success (all assertions passed) |
| `1` | Failure (assertion failed, config error, or runtime error) |

### Commands

All commands start with `/`:

**Conversation**
- `/new` - Start a new conversation
- `/save` - Save the current conversation
- `/load [--archived] [name]` - Load a saved conversation. Use `--archived` to load from archive. Without args, shows interactive selection menu.
- `/branch` - Create a branch (copy) of the current conversation
- `/rename` - Rename a saved conversation (renames its directory)
- `/chats [--archived|--all]` - List saved conversations. Use `--archived` for archived only, `--all` for both.
- `/archive [index|name|range] [--tag key=value]` - Archive conversations by index (`3`), name, range (`1-5`), or tag filter
- `/unarchive [index|name|range] [--tag key=value]` - Restore archived conversations (same syntax as `/archive`)
- `/clear [index]` - Clear and delete the current conversation, or delete a saved conversation by index (requires confirmation)

**Export/Import**
- `/export [format]` - Export the current conversation (formats: `md`, `json`, `html`; prompts if omitted). `json` includes full metadata; `md`/`html` are minimal, human-friendly transcripts.
- `/export-all [format]` - Export all saved conversations in the given format.
- `/import <path>` - Import a conversation exported as `md` or `json` into the conversations folder
- `/import-chatgpt <path> [--dry-run]` - Import conversations from a ChatGPT export ZIP file. Use `--dry-run` to preview without saving.

**Model & Settings**
- `/model [provider/name]` - Switch model. Without args, shows interactive selection menu. With args, switches directly (e.g., `/model ollama/llama3`).
- `/temp` - Change the temperature setting
- `/reason [level]` - Set reasoning level (`low`, `medium`, `high`, `max`). Without args, shows interactive selection menu. For Anthropic Claude, these map to thinking budgets (4K, 16K, 32K, 55K tokens).
- `/thinking [tokens]` - Set exact thinking budget in tokens for Anthropic Claude (1024-128000). Use `/thinking off` to disable. Overrides `/reason` presets.
- `/timeout <seconds|0|off>` - Set the request timeout in seconds, or disable with `0` or `off`
- `/stream [on|off]` - Toggle or set streaming of assistant responses
- `/prompt [text|clear]` - Set or clear the system prompt for this conversation (prompts if omitted)

**Files**
- `/file [--force] <path>` - Register a file for use as `@path` in messages (content is expanded when sending, message stores `@path`). Use `--force` for large files above `file_confirm_threshold_bytes`.
- `/unfile <key>` - Unregister a file (removes both the path and basename aliases)
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
- `/assert <pattern>`, `/assert-not <pattern>` - Assert the last response contains (or doesn't contain) a text/regex pattern. Case-insensitive. Exits with error in batch mode if assertion fails.
- `/echo <text>` - Print a message without sending to model
- `/log-level <level>` - Adjust logging verbosity (debug/info/warn/error/critical)
- `/profile [--full]` - Show current settings and registered files

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

| Key | Action |
|-----|--------|
| `Ctrl+Up` | Focus conversation pane |
| `Ctrl+Home/End` | Jump to start/end of conversation |
| `Up/Down` | Scroll (conversation) or history (input) |
| `Tab` | Command completion or return to input |
| `Escape` | Clear input or return to input pane |
| `Escape` ×2 | Cancel ongoing inference |
| `Ctrl+C/D` | Exit |

Use `/keys` for the full list.

### File References

`/file [--force] <path>` registers a file for use in messages. Include `@path` in any user message to inline the file contents when sending (the stored message keeps `@path` for readability). Examples:

- Register: `/file docs/plan.md`
- Send with inline file: `Summarize @docs/plan.md and list action items.` (you can also use `@{docs/plan.md}` or `@plan.md` if unique)
- If an `@path` isn’t registered, the send will error and nothing is sent.

You can register multiple files and mix references in one message. `/profile` lists full paths of registered files.

### Token Estimation

When registering files with `/file`, ScriptChat shows the token count and context percentage:

```
Registered @README.md (14234 chars, 3548 tokens / 2.7% ctx)
```

Token counting accuracy varies by provider:

| Model/Provider | Method | Accuracy |
|----------------|--------|----------|
| DeepSeek (any provider) | transformers | exact |
| OpenAI (gpt-3/gpt-4/o1/o3) | tiktoken | exact |
| Anthropic | tiktoken cl100k_base | ~approximate |
| Ollama | tiktoken cl100k_base | ~approximate |
| Other openai-compatible | tiktoken cl100k_base | ~approximate |

Approximate counts are prefixed with `~`. For exact DeepSeek tokenization, install the optional dependency:

```bash
pip install scriptchat[deepseek]   # or: pipx inject scriptchat transformers
```

### Conversation Storage

Conversations are stored in `~/.scriptchat/conversations/` (or `conversations_dir` in config) with the following structure:

- Each conversation is a directory: `YYYYMMDDHHMM_modelname_savename/`
- Messages are stored as individual files: `0001_user.txt`, `0002_llm.txt`, etc.
- Metadata is stored in `meta.json`

You can manually edit message files or delete them as needed.

Exports (`/export`) go to the current working directory by default, or to `exports_dir` if configured.

## Example Workflow

1. Start ScriptChat: `scriptchat` or `sc`
2. Chat with the default model
3. Save your conversation: `/save` then enter a name
4. Switch models: `/model` then select a model
5. Continue chatting with the new model
6. Rename or branch to organize: `/rename new-name` or `/branch`
7. Load a previous conversation: `/load`
8. Exit when done: `/exit` or Ctrl+C

## Examples Gallery

<details>
<summary><b>Click to expand example scripts</b></summary>

### Quickstart (batch)
```bash
/model ollama/llama3.2
What is 2+2?
/assert 4
```
*[View quickstart.sc →](examples/quickstart.sc)*

### CI Prompt Testing (batch)
```bash
/temp 0.1
/prompt You are a math tutor. Give clear, concise explanations.
Explain the Pythagorean theorem in one sentence.
/assert theorem|triangle|hypotenuse
/assert-not calculus|derivative
```
*[View ci-prompt-testing.sc →](examples/ci-prompt-testing.sc)*

### Security Audit (batch)
```bash
# FILE=app.py scriptchat --run examples/security-audit.sc
/file ${FILE}
"""
Review @${FILE} for security vulnerabilities:
- Hardcoded secrets, SQL injection, XSS, command injection...
"""
/assert-not admin123|sk-1234567890
```
*[View security-audit.sc →](examples/security-audit.sc)*

### Prompt Engineering (interactive)
```bash
What are the key principles of REST API design?
/save rest-api-baseline
/branch rest-api-detailed
/prompt You are an expert API architect. Be concise and technical.
/retry
```
*[View prompt-engineering.sc →](examples/prompt-engineering.sc)*

### Code Review (interactive)
```bash
/file examples/files/src/api/routes.py
/file examples/files/tests/test_routes.py
"""
Review @routes.py for security vulnerabilities, focusing on:
- Input validation, authentication, SQL injection...
"""
```
*[View code-review.sc →](examples/code-review.sc)*

### Research with Reasoning (interactive)
```bash
/model anthropic/claude-sonnet-4-20250514
/reason high
/timeout off
"""
Analyze the trade-offs between microservices and monolithic architecture...
"""
```
*[View research-reasoning.sc →](examples/research-reasoning.sc)*

</details>

See the [`examples/`](examples/) folder for full scripts and documentation.

## Status Bar

The status bar shows:
- Provider and model (e.g., `ollama/llama3.2`) with optional reasoning level in parentheses
- Token usage (input/output), with optional context usage percentage
- Conversation ID (or `<unsaved>` for new conversations)
- Thinking indicator when the model is processing

Example: `ollama/llama3.2 (high) | 1234 in / 567 out | 1801/8192 (22.0%) | 202511180945_llama32_my-chat`

## Troubleshooting

**"Configuration file not found"**: Create `~/.scriptchat/config.toml` from the example, or run `scriptchat --init`.

**"Ollama is not running"**: Start Ollama with `ollama serve` before using ScriptChat with an Ollama provider.

**Connection errors**: Check that Ollama is running (`ollama serve`) and accessible at the configured URL.

**Model not found**: Make sure the model is pulled in Ollama (`ollama pull modelname`) and configured in `config.toml`.

## License

Apache License 2.0 (see `LICENSE`). Attribution details are in `NOTICE`.
