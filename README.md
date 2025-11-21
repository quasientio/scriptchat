# lite-chat

A terminal-based chat client for interacting with local LLMs via Ollama.

## Features

- Full-screen terminal UI with conversation history, status bar, and input pane
- Multiple pre-configured models with easy switching
- Persistent, file-based conversations (save, load, and branch)
- Multi-line message support
- Send file contents into the chat
- Token usage tracking
- Temperature adjustment
- System prompts

## Requirements

- Python 3.10+
- [Ollama](https://ollama.ai) installed and accessible in your PATH

## Installation

1. Clone this repository:
   ```bash
   cd /path/to/lite-chat
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create configuration directory and file:
   ```bash
   mkdir -p ~/.lite-chat
   cp config.toml.example ~/.lite-chat/config.toml
   ```

5. Edit `~/.lite-chat/config.toml` to configure your models and preferences.

## Configuration

Edit `~/.lite-chat/config.toml` to configure lite-chat. See `config.toml.example` for a complete example.

Key configuration options:

- `api_url`: Ollama API endpoint (default: http://localhost:11434/api)
- `default_model`: Model to use on startup
- `default_temperature`: Temperature for new conversations (0.0-2.0)
- `system_prompt`: System prompt for all conversations (override per conversation with `/prompt`)
- `conversations_dir`: Where to store conversations (set in `[general]`)
- `exports_dir`: Where to write exports (defaults to current working directory if not set)
- `enable_streaming`: Enable token streaming (default: false)
- `[[models]]`: List of available models with context lengths (comma-separated in `contexts`; the first value is used to start the server)

## Usage

Run lite-chat:

```bash
python -m litechat
```

### Commands

All commands start with `/`:

- `/new` - Start a new conversation
- `/save` - Save the current conversation
- `/load` - Load a saved conversation
- `/branch` - Create a branch (copy) of the current conversation
- `/rename` - Rename a saved conversation (renames its directory)
- `/chats` - List saved conversations
- `/send <message>` - Queue a message (sends immediately if the model is idle)
- `/export [format]` - Export the current conversation (formats: `md`; prompts if omitted)
- `/stream [on|off]` - Toggle or set streaming of assistant responses
- `/prompt [text|clear]` - Set or clear the system prompt for this conversation (prompts if omitted)
- `/run <path>` - Execute a script file (one command/message per line; lines starting with `#` are comments)
- `/model` - Switch to a different model
- `/temp` - Change the temperature setting
- `/file <path>` - Send the contents of a text file as your message
- `/clear [index]` - Clear and delete the current conversation, or delete a saved conversation by index (requires confirmation)
- `/exit` - Exit lite-chat

### Multi-line Messages

To enter a multi-line message:

1. Type `"""` and press Enter
2. Enter your message across multiple lines
3. Type `"""` on a new line to send

### Conversation Storage

Conversations are stored in `~/.lite-chat/conversations/` (or your configured directory) with the following structure:

- Each conversation is a directory: `YYYYMMDDHHMM_modelname_savename/`
- Messages are stored as individual files: `0001_user.txt`, `0002_llm.txt`, etc.
- Metadata is stored in `meta.json`

You can manually edit message files or delete them as needed.

## Example Workflow

1. Start lite-chat: `python -m litechat`
2. Chat with the default model
3. Save your conversation: `/save` then enter a name
4. Switch models: `/model` then select a model
5. Continue chatting with the new model
6. Rename or branch to organize: `/rename new-name` or `/branch`
7. Load a previous conversation: `/load`
8. Exit when done: `/exit` or Ctrl+C

## Status Bar

The status bar shows:
- Current model name
- Token usage (input/output)
- Conversation ID (or `<unsaved>` for new conversations)

Example: `model: llama3.2 | tokens: 1234 in / 567 out | convo: 202511180945_llama32_my-chat`

## Troubleshooting

**"ollama executable not found"**: Ensure Ollama is installed and in your PATH.

**"Configuration file not found"**: Create `~/.lite-chat/config.toml` from the example.

**Connection errors**: Ensure Ollama is running. lite-chat will start its own Ollama server instance with the correct context length.

**Model not found**: Make sure the model is pulled in Ollama (`ollama pull modelname`) and configured in `config.toml`.

## License

MIT
