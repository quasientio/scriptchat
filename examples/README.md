# ScriptChat Examples

This folder contains example scripts demonstrating ScriptChat workflows.

## Examples

### Interactive Mode

These examples are designed for interactive TUI sessions. Start ScriptChat and paste commands as you go.

| File | Description | Key Commands |
|------|-------------|--------------|
| [prompt-engineering.sc](prompt-engineering.sc) | Iterative prompt refinement with branching | /save, /branch, /retry, /note, /history |
| [code-review.sc](code-review.sc) | PR review workflow with file references | /file, /files, @references, /note |
| [research-reasoning.sc](research-reasoning.sc) | Deep analysis with extended thinking | /reason, /thinking, /timeout off |

### Batch Mode

These examples are designed to run non-interactively via `--run` or stdin.

| File | Description | Key Commands |
|------|-------------|--------------|
| [quickstart.sc](quickstart.sc) | Simple test to verify ScriptChat is working | /model, /assert, /echo |
| [ci-prompt-testing.sc](ci-prompt-testing.sc) | Automated prompt regression testing | /assert, /assert-not, /temp |
| [batch-testing.sc](batch-testing.sc) | Parameterized testing with model comparison | ${VAR}, /model, /assert, /save |
| [security-audit.sc](security-audit.sc) | Security vulnerability scanner for code files | /file, ${VAR}, /assert-not |
| [run-batch-testing.sh](run-batch-testing.sh) | Bash wrapper to run tests across models and languages | - |

## Running Examples

### Interactive Mode

```bash
scriptchat
# Then paste commands from the .sc file as you work through the example
```

### Batch Mode

```bash
# Run a script
scriptchat --run examples/ci-prompt-testing.sc

# Run all tests, continue even if some fail
scriptchat --run examples/ci-prompt-testing.sc --continue-on-error

# Pipe input
cat examples/ci-prompt-testing.sc | scriptchat
```

### Exit Codes (Batch Mode)

- `0` - All assertions passed
- `1` - One or more assertions failed

## Customizing Examples

### Using Variables

Variables use `${name}` syntax and check script variables first, then environment variables:

```bash
# Set via /set command
/set language=JavaScript
Write a ${language} example.

# Or pass via environment
LANGUAGE=Python scriptchat --run script.sc
```

See `batch-testing.sc` and `run-batch-testing.sh` for a complete example of running tests across multiple models and configurations.

**Note:** Sensitive env vars (`*_KEY`, `*_SECRET`, `*_TOKEN`, etc.) are blocked by default. See main README for config options.

### Multi-line Messages

Use triple quotes for multi-line messages in scripts:

```bash
"""
Analyze this code for:
- Security issues
- Performance problems
- Code style violations
"""
```

Lines inside the block preserve their formatting and indentation.

### File References

Register files with `/file` and reference with `@`:

```bash
/file src/main.py
Review @main.py for issues.
```

### Example Files

The `files/` subdirectory contains sample files for use with examples:

```
examples/files/
├── app.py                 # Flask app with intentional security vulnerabilities
├── src/api/routes.py      # Flask routes with intentional issues
└── tests/test_routes.py   # Incomplete test coverage
```

- `app.py` is used by `security-audit.sc` to demonstrate vulnerability scanning
- `src/` and `tests/` are used by `code-review.sc` for file-based code review

## Tips

- Use `/temp 0.1` in tests for more consistent outputs
- Add `/note` to document findings during interactive sessions
- Use `/history` to review what prompts you've tried
- Export with `/export json` for programmatic analysis
- Use `--continue-on-error` in CI to see all test results
