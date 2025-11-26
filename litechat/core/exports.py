# Copyright 2024 lite-chat contributors
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

"""Conversation import/export helpers."""

import html
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from .conversations import (
    Conversation,
    Message,
    _slugify_model_name,
    _slugify_save_name,
    save_conversation,
)


def export_conversation_md(convo: Conversation, export_dir: Path, filename: Optional[str] = None) -> Path:
    """Export a conversation to a Markdown file."""
    export_dir.mkdir(parents=True, exist_ok=True)

    if not filename:
        if convo.id:
            filename = f"{convo.id}.md"
        else:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            model_slug = _slugify_model_name(convo.model_name)
            filename = f"chat_{timestamp}_{model_slug}.md"

    export_path = export_dir / filename

    lines = [
        f"# Conversation {convo.id or 'unsaved'}",
        f"- Model: {convo.model_name}",
        f"- Temperature: {convo.temperature:.2f}",
        f"- Exported: {datetime.now().isoformat()}",
        ""
    ]

    for msg in convo.messages:
        # Skip system messages (status prompts, system prompt snapshots, etc.)
        if msg.role == 'system':
            continue
        heading = msg.role.capitalize()
        lines.append(f"## {heading}")
        lines.append("")
        lines.append(msg.content)
        lines.append("")

    export_path.write_text('\n'.join(lines), encoding='utf-8')
    return export_path


def export_conversation_json(convo: Conversation, export_dir: Path, filename: Optional[str] = None) -> Path:
    """Export a conversation to a JSON file."""
    export_dir.mkdir(parents=True, exist_ok=True)

    if not filename:
        if convo.id:
            filename = f"{convo.id}.json"
        else:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            model_slug = _slugify_model_name(convo.model_name)
            filename = f"chat_{timestamp}_{model_slug}.json"

    export_path = export_dir / filename

    payload = {
        "id": convo.id,
        "provider_id": convo.provider_id,
        "model": convo.model_name,
        "temperature": convo.temperature,
        "system_prompt": convo.system_prompt,
        "tokens_in": convo.tokens_in,
        "tokens_out": convo.tokens_out,
        "context_length_configured": convo.context_length_configured,
        "context_length_used": convo.context_length_used,
        "exported_at": datetime.now().isoformat(),
        "messages": [
            {"role": msg.role, "content": msg.content}
            for msg in convo.messages
        ],
    }

    export_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    return export_path


def export_conversation_html(convo: Conversation, export_dir: Path, filename: Optional[str] = None) -> Path:
    """Export a conversation to an HTML file."""
    export_dir.mkdir(parents=True, exist_ok=True)

    if not filename:
        if convo.id:
            filename = f"{convo.id}.html"
        else:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            model_slug = _slugify_model_name(convo.model_name)
            filename = f"chat_{timestamp}_{model_slug}.html"

    export_path = export_dir / filename

    def esc(text: str) -> str:
        return html.escape(text, quote=False)

    def inline(text: str) -> str:
        """Render inline Markdown (links, bold, italic, code) with escaping."""
        text = html.escape(text, quote=False)
        text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
        text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text, flags=re.S)
        text = re.sub(r"\*(.+?)\*", r"<em>\1</em>", text, flags=re.S)
        text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', text)
        return text

    def render_content(md_text: str) -> str:
        """Render a small subset of Markdown to HTML safely (headings, code, tables, lists, links, emphasis)."""
        lines = md_text.splitlines()
        out: list[str] = []
        i = 0
        in_code = False
        code_lines: list[str] = []

        def flush_paragraph(paragraph_lines: list[str]):
            if not paragraph_lines:
                return
            out.append("<p>" + "<br>".join(paragraph_lines) + "</p>")

        def flush_list(items: list[str]):
            if not items:
                return
            out.append("<ul>" + "".join(f"<li>{item}</li>" for item in items) + "</ul>")

        paragraph_buf: list[str] = []
        list_buf: list[str] = []

        while i < len(lines):
            line = lines[i]

            # Code fences
            if line.strip().startswith("```"):
                if in_code:
                    out.append("<pre><code>" + "\n".join(html.escape(l, quote=False) for l in code_lines) + "</code></pre>")
                    code_lines = []
                    in_code = False
                else:
                    flush_paragraph(paragraph_buf)
                    flush_list(list_buf)
                    list_buf = []
                    paragraph_buf = []
                    in_code = True
                i += 1
                continue

            if in_code:
                code_lines.append(line)
                i += 1
                continue

            # Headings
            heading_match = re.match(r"^(#{1,6})\s+(.*)$", line)
            if heading_match:
                flush_paragraph(paragraph_buf)
                flush_list(list_buf)
                list_buf = []
                paragraph_buf = []
                level = min(len(heading_match.group(1)), 6)
                out.append(f"<h{level}>" + inline(heading_match.group(2)) + f"</h{level}>")
                i += 1
                continue

            # Tables (simple pipe tables)
            if "|" in line and i + 1 < len(lines):
                sep_line = lines[i + 1]
                if re.match(r"^\s*\|?\s*[:\-| ]+\s*\|?\s*$", sep_line):
                    flush_paragraph(paragraph_buf)
                    flush_list(list_buf)
                    list_buf = []
                    paragraph_buf = []
                    headers = [inline(h.strip()) for h in line.strip().strip("|").split("|")]
                    i += 2
                    rows = []
                    while i < len(lines) and "|" in lines[i] and not lines[i].strip().startswith("#"):
                        row_cells = [inline(c.strip()) for c in lines[i].strip().strip("|").split("|")]
                        rows.append(row_cells)
                        i += 1
                    table_html = ["<table><thead><tr>"]
                    for h in headers:
                        table_html.append(f"<th>{h}</th>")
                    table_html.append("</tr></thead><tbody>")
                    for row in rows:
                        table_html.append("<tr>")
                        for cell in row:
                            table_html.append(f"<td>{cell}</td>")
                        table_html.append("</tr>")
                    table_html.append("</tbody></table>")
                    out.append("".join(table_html))
                    continue

            # Unordered list items
            list_match = re.match(r"^\s*[\*\-+]\s+(.*)$", line)
            if list_match:
                flush_paragraph(paragraph_buf)
                paragraph_buf = []
                list_buf.append(inline(list_match.group(1)))
                i += 1
                while i < len(lines):
                    lm = re.match(r"^\s*[\*\-+]\s+(.*)$", lines[i])
                    if not lm:
                        break
                    list_buf.append(inline(lm.group(1)))
                    i += 1
                flush_list(list_buf)
                list_buf = []
                continue

            # Blank line -> paragraph break
            if not line.strip():
                flush_paragraph(paragraph_buf)
                flush_list(list_buf)
                list_buf = []
                paragraph_buf = []
                i += 1
                continue

            # Default: accumulate as part of paragraph
            paragraph_buf.append(inline(line))
            i += 1

        flush_paragraph(paragraph_buf)
        flush_list(list_buf)
        if in_code:
            out.append("<pre><code>" + "\n".join(html.escape(l, quote=False) for l in code_lines) + "</code></pre>")

        return "".join(out)

    meta_items = [
        f"<li><strong>Model:</strong> {esc(convo.model_name)}</li>",
        f"<li><strong>Temperature:</strong> {convo.temperature:.2f}</li>",
        f"<li><strong>Exported:</strong> {datetime.now().isoformat()}</li>",
    ]
    if convo.system_prompt:
        meta_items.append(f"<li><strong>System prompt:</strong> {esc(convo.system_prompt)}</li>")

    message_blocks = []
    for msg in convo.messages:
        if msg.role == 'system':
            continue
        role = esc(msg.role.capitalize())
        content = render_content(msg.content)
        message_blocks.append(
            f"<div class=\"message role-{msg.role}\"><div class=\"role\">{role}</div><div class=\"content\">{content}</div></div>"
        )

    html_body = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8" />
  <title>Conversation {esc(convo.id or 'unsaved')}</title>
  <style>
    body {{ font-family: sans-serif; margin: 1.5rem; max-width: 900px; }}
    h1 {{ margin-top: 0; }}
    ul.meta {{ list-style: none; padding: 0; }}
    ul.meta li {{ margin: 0.25rem 0; color: #444; }}
    .message {{ border: 1px solid #ddd; border-radius: 6px; padding: 0.75rem; margin: 0.5rem 0; }}
    .role {{ font-weight: bold; margin-bottom: 0.5rem; color: #222; }}
    .role-user {{ color: #0a66c2; }}
    .role-assistant {{ color: #0a7a0a; }}
    .content {{ white-space: pre-wrap; }}
    table {{ border-collapse: collapse; margin: 0.5rem 0; width: 100%; }}
    th, td {{ border: 1px solid #ddd; padding: 6px 8px; text-align: left; }}
    th {{ background: #f5f5f5; }}
  </style>
</head>
<body>
  <h1>Conversation {esc(convo.id or 'unsaved')}</h1>
  <ul class="meta">
    {''.join(meta_items)}
  </ul>
  <div class="messages">
    {''.join(message_blocks)}
  </div>
</body>
</html>
"""
    export_path.write_text(html_body, encoding='utf-8')
    return export_path


def _dedupe_dir_name(root: Path, desired: str) -> str:
    """Ensure a directory name is unique within root by appending suffixes if needed."""
    candidate = desired
    counter = 1
    while (root / candidate).exists():
        candidate = f"{desired}-import{counter if counter > 1 else ''}"
        counter += 1
    return candidate


def import_conversation_from_file(path: Path, conversations_root: Path) -> Conversation:
    """Import a conversation from a JSON or Markdown export."""
    path = path.expanduser()
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()
    if suffix == '.json':
        convo, desired_id = _conversation_from_json_export(path)
    elif suffix == '.md':
        convo, desired_id = _conversation_from_md_export(path)
    else:
        raise ValueError("Unsupported import format. Expected .json or .md")

    # Decide on directory/id
    if desired_id:
        dir_name = _dedupe_dir_name(conversations_root, desired_id)
        convo.id = dir_name
        # Use existing id if no collision
        if dir_name != desired_id:
            save_name = _slugify_save_name(dir_name)
            convo.id = None
            convo = save_conversation(conversations_root, convo, save_name=save_name, system_prompt=convo.system_prompt)
            return convo
    else:
        save_name = _slugify_save_name(path.stem or "imported")
        convo = save_conversation(conversations_root, convo, save_name=save_name, system_prompt=convo.system_prompt)
        return convo

    # Save using the chosen id
    conversations_root.mkdir(parents=True, exist_ok=True)
    conv_dir = conversations_root / convo.id
    conv_dir.mkdir(parents=True, exist_ok=True)
    convo = save_conversation(conversations_root, convo, system_prompt=convo.system_prompt)
    return convo


def _conversation_from_json_export(path: Path) -> tuple[Conversation, Optional[str]]:
    """Build Conversation from JSON export and return desired id."""
    data = json.loads(path.read_text(encoding='utf-8'))
    system_prompt = data.get("system_prompt")
    messages = []
    if system_prompt:
        messages.append(Message(role='system', content=system_prompt))
    for msg in data.get("messages", []):
        role = msg.get("role")
        content = msg.get("content", "")
        if role in ("user", "assistant"):
            messages.append(Message(role=role, content=content))
        elif role == "system":
            messages.append(Message(role=role, content=content))

    convo = Conversation(
        id=None,
        provider_id=data.get("provider_id", "ollama"),
        model_name=data.get("model", "unknown"),
        temperature=data.get("temperature", 0.7),
        system_prompt=system_prompt,
        messages=messages,
        tokens_in=data.get("tokens_in", 0),
        tokens_out=data.get("tokens_out", 0),
        context_length_configured=data.get("context_length_configured"),
        context_length_used=data.get("context_length_used"),
    )
    desired_id = data.get("id")
    if desired_id == "unsaved":
        desired_id = None
    return convo, desired_id


def _conversation_from_md_export(path: Path) -> tuple[Conversation, Optional[str]]:
    """Build Conversation from Markdown export and return desired id."""
    text = path.read_text(encoding='utf-8')
    lines = text.splitlines()
    desired_id = None
    model = "unknown"
    temperature = 0.7

    # Parse header and metadata bullets
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("# Conversation"):
            parts = stripped.split(" ", 2)
            if len(parts) >= 3:
                candidate = parts[2].strip()
                if candidate.lower() != "unsaved":
                    desired_id = candidate
        elif stripped.startswith("- Model:"):
            model = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("- Temperature:"):
            try:
                temperature = float(stripped.split(":", 1)[1].strip())
            except Exception:
                pass

    messages: list[Message] = []
    current_role: Optional[str] = None
    current_lines: list[str] = []

    def flush():
        if current_role is None:
            return
        content = "\n".join(current_lines).strip()
        messages.append(Message(role=current_role, content=content))

    heading_re = re.compile(r"^##\s+(user|assistant|system)", flags=re.IGNORECASE)

    for line in lines:
        match = heading_re.match(line.strip())
        if match:
            flush()
            current_role = match.group(1).lower()
            current_lines = []
            continue
        if current_role is not None:
            current_lines.append(line)
    flush()

    convo = Conversation(
        id=None,
        provider_id="ollama",
        model_name=model,
        temperature=temperature,
        system_prompt=None,
        messages=messages,
        tokens_in=0,
        tokens_out=0,
    )
    return convo, desired_id
