# Contributing to ScriptChat

Thank you for your interest in contributing to ScriptChat!

## Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/scriptchat/scriptchat.git
   cd scriptchat
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -e ".[dev]"
   ```

3. Set up your configuration:
   ```bash
   mkdir -p ~/.scriptchat
   cp config.toml.example ~/.scriptchat/config.toml
   # Edit the config file with your provider settings
   ```

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=scriptchat

# Run a specific test file
pytest tests/test_commands.py

# Run a specific test
pytest tests/test_commands.py::TestCommandHandling::test_new_command
```

## Code Style

- Follow PEP 8 guidelines
- Use type hints for function parameters and return values
- Add docstrings to public functions and classes
- Keep lines under 100 characters when practical

## Submitting Changes

1. Fork the repository and create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and ensure tests pass:
   ```bash
   pytest
   ```

3. Commit with a clear message describing the change:
   ```bash
   git commit -m "Add feature: description of what it does"
   ```

4. Push to your fork and open a pull request against `main`.

## Adding New Commands

Commands are defined in `scriptchat/core/commands.py`:

1. Add an entry to `COMMAND_REGISTRY` with category, usage, description, and examples
2. Add handling in `handle_command()` function
3. If the command needs UI interaction, set `needs_ui_interaction=True` and add a handler in `scriptchat/ui/command_handlers.py`
4. Add tests in `tests/test_commands.py`

## Adding New Providers

Provider clients live in `scriptchat/core/`:

1. Create a new client file (e.g., `new_provider_client.py`)
2. Implement a `chat()` method matching the signature in existing clients
3. Register the provider type in the initialization logic
4. Update `config.py` if new configuration options are needed
5. Add tests for the new provider

## Reporting Issues

When reporting bugs, please include:
- Python version (`python --version`)
- ScriptChat version (`scriptchat --version`)
- Operating system
- Steps to reproduce the issue
- Expected vs actual behavior
- Relevant configuration (without API keys)

## Questions?

Open an issue with the "question" label if you need help or clarification.
