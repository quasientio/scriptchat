#!/usr/bin/env python3
"""Update model_defaults.py with latest context limits from upstream.

Fetches data from https://github.com/taylorwilsdon/llm-context-limits
and regenerates scriptchat/core/model_defaults.py.

Usage:
    python scripts/update_model_defaults.py
"""

import re
import urllib.request
from pathlib import Path

UPSTREAM_URL = "https://raw.githubusercontent.com/taylorwilsdon/llm-context-limits/main/README.md"
OUTPUT_FILE = Path(__file__).parent.parent / "scriptchat" / "core" / "model_defaults.py"


def parse_context_size(text: str) -> int | None:
    """Parse context size from text like '128k tokens' or '1,048k'."""
    text = text.lower().replace(",", "").replace(" ", "")

    match = re.search(r"([\d.]+)k", text)
    if match:
        return int(float(match.group(1)) * 1000)

    match = re.search(r"(\d+)", text)
    if match:
        return int(match.group(1))

    return None


def clean_model_name(name: str) -> str | None:
    """Clean model name, removing markdown formatting and filtering invalid entries."""
    # Remove markdown bold markers
    name = name.replace("**", "").replace("`", "")

    # Skip entries that look like API endpoints or headers
    if name.startswith("/") or name.startswith("endpoint"):
        return None
    if "token" in name.lower() and ("input" in name.lower() or "output" in name.lower()):
        return None

    # Normalize: lowercase, spaces to hyphens
    name = name.lower().strip().replace(" ", "-")

    # Remove parenthetical suffixes like "(deepseek-v3)" but keep them for matching
    # Actually, let's keep simple names - extract just the model name before parens
    if "(" in name:
        # e.g. "deepseek-chat-(deepseek-v3)" -> "deepseek-chat"
        name = name.split("(")[0].strip("-")

    # Skip if name is too short or doesn't look like a model name
    if len(name) < 2 or name in ("model", "model-name", "name"):
        return None

    return name


def parse_markdown_table(lines: list[str]) -> list[tuple[str, int]]:
    """Parse a markdown table and extract model -> context mappings."""
    results = []

    for line in lines:
        if not line.strip().startswith("|"):
            continue

        # Skip header separator lines
        if "---" in line:
            continue

        cells = [c.strip() for c in line.split("|")]
        cells = [c for c in cells if c]  # Remove empty cells

        if len(cells) < 2:
            continue

        model_name = cells[0].strip()
        context_text = cells[1].strip()

        # Skip header rows
        if model_name.lower().replace("**", "") in ("model", "model name"):
            continue

        # Clean and validate model name
        cleaned_name = clean_model_name(model_name)
        if not cleaned_name:
            continue

        context_size = parse_context_size(context_text)
        if context_size and context_size > 100:  # Filter out tiny values that aren't context sizes
            results.append((cleaned_name, context_size))

    return results


def fetch_and_parse() -> dict[str, int]:
    """Fetch upstream README and parse all model context limits."""
    print(f"Fetching {UPSTREAM_URL}...")

    with urllib.request.urlopen(UPSTREAM_URL, timeout=30) as response:
        content = response.read().decode("utf-8")

    lines = content.split("\n")
    all_models = {}

    # Parse tables from the markdown
    results = parse_markdown_table(lines)

    for model_name, context_size in results:
        if model_name not in all_models:
            all_models[model_name] = context_size

    return all_models


def generate_module(models: dict[str, int]) -> str:
    """Generate the model_defaults.py module content."""

    # Group models by provider (best effort based on naming)
    providers = {
        "OpenAI": [],
        "Anthropic": [],
        "DeepSeek": [],
        "Mistral": [],
        "Gemini": [],
        "Qwen": [],
        "Llama": [],
        "Other": [],
    }

    for model, context in sorted(models.items()):
        model_lower = model.lower()
        if model_lower.startswith(("gpt-", "o1", "o3", "o4")):
            providers["OpenAI"].append((model, context))
        elif model_lower.startswith("claude"):
            providers["Anthropic"].append((model, context))
        elif model_lower.startswith("deepseek"):
            providers["DeepSeek"].append((model, context))
        elif model_lower.startswith("mistral"):
            providers["Mistral"].append((model, context))
        elif model_lower.startswith("gemini") or model_lower.startswith("gemma"):
            providers["Gemini"].append((model, context))
        elif model_lower.startswith("qwen") or model_lower.startswith("qwq"):
            providers["Qwen"].append((model, context))
        elif model_lower.startswith("llama"):
            providers["Llama"].append((model, context))
        else:
            providers["Other"].append((model, context))

    lines = [
        '"""Default context window limits for well-known LLM models.',
        "",
        "This module provides built-in context limits for common models so that",
        "context usage % can be displayed in the status bar without requiring",
        "explicit configuration.",
        "",
        "Data sourced from: https://github.com/taylorwilsdon/llm-context-limits",
        "",
        "To update these defaults, run:",
        "    python scripts/update_model_defaults.py",
        "",
        "Precedence: User config (contexts in config.toml) > these defaults",
        '"""',
        "",
        "# Model name -> context window size in tokens",
        "# Use lowercase keys for case-insensitive matching",
        "MODEL_CONTEXT_LIMITS: dict[str, int] = {",
    ]

    for provider, model_list in providers.items():
        if not model_list:
            continue
        lines.append(f"    # {provider} models")
        for model, context in sorted(model_list):
            lines.append(f'    "{model}": {context},')
        lines.append("")

    # Remove trailing empty line inside dict
    if lines[-1] == "":
        lines.pop()

    lines.append("}")
    lines.append("")
    lines.append("")
    lines.append("def get_default_context_limit(model_name: str) -> int | None:")
    lines.append('    """Get the default context limit for a model.')
    lines.append("")
    lines.append("    Uses fuzzy matching: first tries exact match, then prefix match.")
    lines.append("    For example, 'gpt-4o-2024-08-06' will match 'gpt-4o'.")
    lines.append("")
    lines.append("    Args:")
    lines.append("        model_name: The model name to look up")
    lines.append("")
    lines.append("    Returns:")
    lines.append("        Context limit in tokens, or None if unknown")
    lines.append('    """')
    lines.append("    name_lower = model_name.lower()")
    lines.append("")
    lines.append("    # Try exact match first")
    lines.append("    if name_lower in MODEL_CONTEXT_LIMITS:")
    lines.append("        return MODEL_CONTEXT_LIMITS[name_lower]")
    lines.append("")
    lines.append("    # Try prefix match (longest match wins)")
    lines.append("    best_match = None")
    lines.append("    best_length = 0")
    lines.append("")
    lines.append("    for known_model, limit in MODEL_CONTEXT_LIMITS.items():")
    lines.append("        if name_lower.startswith(known_model) and len(known_model) > best_length:")
    lines.append("            best_match = limit")
    lines.append("            best_length = len(known_model)")
    lines.append("")
    lines.append("    return best_match")
    lines.append("")

    return "\n".join(lines)


def main():
    """Main entry point."""
    try:
        upstream_models = fetch_and_parse()
        print(f"Parsed {len(upstream_models)} models from upstream")

        if not upstream_models:
            print("Warning: No models parsed from upstream. Keeping existing file.")
            return

        # Read existing file to compare
        existing_models = {}
        if OUTPUT_FILE.exists():
            import importlib.util
            spec = importlib.util.spec_from_file_location("model_defaults", OUTPUT_FILE)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            existing_models = getattr(module, "MODEL_CONTEXT_LIMITS", {})

        # Generate new content
        content = generate_module(upstream_models)

        # Write output
        OUTPUT_FILE.write_text(content)
        print(f"Updated {OUTPUT_FILE}")

        # Show diff summary
        added = set(upstream_models.keys()) - set(existing_models.keys())
        removed = set(existing_models.keys()) - set(upstream_models.keys())
        changed = {
            k for k in set(upstream_models.keys()) & set(existing_models.keys())
            if upstream_models[k] != existing_models[k]
        }

        if added:
            print(f"  Added: {', '.join(sorted(added))}")
        if removed:
            print(f"  Removed: {', '.join(sorted(removed))}")
        if changed:
            print(f"  Changed: {', '.join(sorted(changed))}")
        if not (added or removed or changed):
            print("  No changes detected")

    except urllib.error.URLError as e:
        print(f"Error fetching upstream: {e}")
        raise SystemExit(1)
    except Exception as e:
        print(f"Error: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
