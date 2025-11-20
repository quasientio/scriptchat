"""Conversation management for lite-chat."""

import json
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class Message:
    """A single message in a conversation."""
    role: str  # "system", "user", or "assistant"
    content: str


@dataclass
class Conversation:
    """A conversation with an LLM."""
    id: Optional[str]  # Directory name, or None if unsaved
    model_name: str
    temperature: float
    messages: list[Message] = field(default_factory=list)
    tokens_in: int = 0
    tokens_out: int = 0
    context_length_configured: Optional[int] = None  # Max context length model is running with
    context_length_used: Optional[int] = None  # Current context length used (last tokens_in)


@dataclass
class ConversationSummary:
    """Summary information about a saved conversation."""
    dir_name: str
    created_at: str
    model_name: str
    display_name: str


def _slugify_model_name(model_name: str) -> str:
    """Convert model name to lowercase alphanumeric only.

    Args:
        model_name: Original model name

    Returns:
        Slugified version (e.g., "llama3.1:8b-instruct" -> "llama318binstruct")
    """
    return re.sub(r'[^a-z0-9]', '', model_name.lower())


def _slugify_save_name(save_name: str) -> str:
    """Clean up save name for use in directory names.

    Args:
        save_name: User-provided save name

    Returns:
        Cleaned name with spaces->hyphens, only alphanumeric/hyphens/underscores
    """
    # Trim whitespace
    cleaned = save_name.strip()
    # Replace spaces with hyphens
    cleaned = cleaned.replace(' ', '-')
    # Remove non-alphanumeric, non-hyphen, non-underscore characters
    cleaned = re.sub(r'[^a-zA-Z0-9\-_]', '', cleaned)
    # Default to "untitled" if empty
    if not cleaned:
        cleaned = 'untitled'
    return cleaned


def _create_dir_name(model_name: str, save_name: str, timestamp: Optional[datetime] = None) -> str:
    """Create directory name in format: yyyymmddhhmm_modelname_savename.

    Args:
        model_name: Model name to slugify
        save_name: Save name to slugify
        timestamp: Optional timestamp (defaults to now)

    Returns:
        Directory name string
    """
    if timestamp is None:
        timestamp = datetime.now()

    time_prefix = timestamp.strftime("%Y%m%d%H%M")
    model_slug = _slugify_model_name(model_name)
    save_slug = _slugify_save_name(save_name)

    return f"{time_prefix}_{model_slug}_{save_slug}"


def list_conversations(root: Path) -> list[ConversationSummary]:
    """List all saved conversations in the conversations directory.

    Args:
        root: Conversations root directory

    Returns:
        List of ConversationSummary objects, sorted newest-first
    """
    if not root.exists():
        return []

    summaries = []

    for conv_dir in root.iterdir():
        if not conv_dir.is_dir():
            continue

        # Try to load meta.json
        meta_path = conv_dir / "meta.json"
        if meta_path.exists():
            try:
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                created_at = meta.get('created_at', '')
                model_name = meta.get('model', 'unknown')
            except (json.JSONDecodeError, IOError):
                created_at = ''
                model_name = 'unknown'
        else:
            created_at = ''
            model_name = 'unknown'

        # Parse directory name for display
        dir_name = conv_dir.name
        # Extract savename from directory name (after second underscore)
        parts = dir_name.split('_', 2)
        if len(parts) >= 3:
            display_name = parts[2]
        else:
            display_name = dir_name

        summaries.append(ConversationSummary(
            dir_name=dir_name,
            created_at=created_at,
            model_name=model_name,
            display_name=display_name
        ))

    # Sort by directory name (which starts with timestamp) in reverse
    summaries.sort(key=lambda s: s.dir_name, reverse=True)

    return summaries


def load_conversation(root: Path, dir_name: str) -> Conversation:
    """Load a conversation from disk.

    Args:
        root: Conversations root directory
        dir_name: Directory name of the conversation

    Returns:
        Loaded Conversation object

    Raises:
        FileNotFoundError: If conversation directory doesn't exist
    """
    conv_dir = root / dir_name

    if not conv_dir.exists():
        raise FileNotFoundError(f"Conversation directory not found: {conv_dir}")

    # Load meta.json if present
    meta_path = conv_dir / "meta.json"
    if meta_path.exists():
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        model_name = meta.get('model', 'unknown')
        temperature = meta.get('temperature', 0.7)
        context_length_configured = meta.get('context_length_configured', None)
        context_length_used = meta.get('context_length_used', None)
    else:
        # Try to infer from directory name
        parts = dir_name.split('_', 2)
        if len(parts) >= 2:
            model_name = parts[1]
        else:
            model_name = 'unknown'
        temperature = 0.7
        context_length_configured = None
        context_length_used = None

    # Load message files
    messages = []
    message_files = []

    # Find all message files matching pattern NNNN_(user|llm).txt
    for file_path in conv_dir.iterdir():
        if file_path.is_file() and re.match(r'^\d+_(user|llm)\.txt$', file_path.name):
            message_files.append(file_path)

    # Sort lexicographically
    message_files.sort(key=lambda p: p.name)

    # Load each message
    for file_path in message_files:
        filename = file_path.name
        # Extract role from filename
        if filename.endswith('_user.txt'):
            role = 'user'
        elif filename.endswith('_llm.txt'):
            role = 'assistant'
        else:
            continue  # Skip unknown roles

        with open(file_path, 'r') as f:
            content = f.read()

        messages.append(Message(role=role, content=content))

    return Conversation(
        id=dir_name,
        model_name=model_name,
        temperature=temperature,
        messages=messages,
        tokens_in=0,
        tokens_out=0,
        context_length_configured=context_length_configured,
        context_length_used=context_length_used
    )


def save_conversation(
    root: Path,
    convo: Conversation,
    save_name: Optional[str] = None,
    system_prompt: Optional[str] = None
) -> Conversation:
    """Save a conversation to disk.

    Args:
        root: Conversations root directory
        convo: Conversation to save
        save_name: Save name for new conversations (required if convo.id is None)
        system_prompt: Optional system prompt to save in metadata

    Returns:
        Updated Conversation object with id set

    Raises:
        ValueError: If save_name is required but not provided
    """
    # Determine directory name
    if convo.id is None:
        # New conversation - need save_name
        if save_name is None:
            raise ValueError("save_name is required for new conversations")

        dir_name = _create_dir_name(convo.model_name, save_name)
        conv_dir = root / dir_name
        conv_dir.mkdir(parents=True, exist_ok=True)
        convo.id = dir_name
    else:
        # Existing conversation - update
        conv_dir = root / convo.id
        if not conv_dir.exists():
            conv_dir.mkdir(parents=True, exist_ok=True)

    # Write meta.json
    meta = {
        'model': convo.model_name,
        'temperature': convo.temperature,
        'created_at': datetime.now().isoformat()
    }
    if system_prompt:
        meta['system_prompt_snapshot'] = system_prompt
    if convo.context_length_configured is not None:
        meta['context_length_configured'] = convo.context_length_configured
    if convo.context_length_used is not None:
        meta['context_length_used'] = convo.context_length_used

    meta_path = conv_dir / 'meta.json'
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    # Write message files
    # Simple approach: renumber all messages from 1..N
    msg_num = 1
    for message in convo.messages:
        # Skip system messages (not saved to files in minimal implementation)
        if message.role == 'system':
            continue

        # Determine suffix based on role
        if message.role == 'user':
            suffix = 'user.txt'
        elif message.role == 'assistant':
            suffix = 'llm.txt'
        else:
            continue  # Skip unknown roles

        filename = f"{msg_num:04d}_{suffix}"
        file_path = conv_dir / filename

        with open(file_path, 'w') as f:
            f.write(message.content)

        msg_num += 1

    return convo


def branch_conversation(
    root: Path,
    convo: Conversation,
    new_save_name: Optional[str] = None,
    system_prompt: Optional[str] = None
) -> Conversation:
    """Create a branch (copy) of a conversation.

    Args:
        root: Conversations root directory
        convo: Conversation to branch
        new_save_name: Optional new save name (defaults to original + "-branch")
        system_prompt: Optional system prompt for metadata

    Returns:
        New Conversation object with new id
    """
    # Ensure original is saved first
    if convo.id is None:
        if new_save_name is None:
            new_save_name = 'untitled'
        convo = save_conversation(root, convo, save_name=new_save_name, system_prompt=system_prompt)

    # Determine new save name
    if new_save_name is None:
        # Extract original save name and append "-branch"
        parts = convo.id.split('_', 2)
        if len(parts) >= 3:
            original_save_name = parts[2]
        else:
            original_save_name = 'untitled'
        new_save_name = f"{original_save_name}-branch"

    # Create new directory name with new timestamp
    new_dir_name = _create_dir_name(convo.model_name, new_save_name)
    new_conv_dir = root / new_dir_name
    old_conv_dir = root / convo.id

    # Copy entire directory
    shutil.copytree(old_conv_dir, new_conv_dir)

    # Update meta.json with new timestamp
    meta_path = new_conv_dir / 'meta.json'
    if meta_path.exists():
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        meta['created_at'] = datetime.now().isoformat()
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)

    # Create new conversation object
    new_convo = Conversation(
        id=new_dir_name,
        model_name=convo.model_name,
        temperature=convo.temperature,
        messages=convo.messages.copy(),
        tokens_in=convo.tokens_in,
        tokens_out=convo.tokens_out
    )

    return new_convo


def delete_conversation(root: Path, dir_name: str) -> None:
    """Delete a conversation directory.

    Args:
        root: Conversations root directory
        dir_name: Directory name to delete
    """
    conv_dir = root / dir_name
    if conv_dir.exists():
        shutil.rmtree(conv_dir)


def rename_conversation(root: Path, convo: Conversation, new_save_name: str) -> Conversation:
    """Rename a saved conversation by renaming its directory.

    Args:
        root: Conversations root directory
        convo: Conversation to rename (must already be saved)
        new_save_name: New save name to use in directory name

    Returns:
        Updated Conversation with new id

    Raises:
        ValueError: If conversation is unsaved
        FileExistsError: If the target directory already exists
    """
    if convo.id is None:
        raise ValueError("Conversation must be saved before it can be renamed")

    old_dir = root / convo.id
    if not old_dir.exists():
        raise FileNotFoundError(f"Conversation directory not found: {old_dir}")

    save_slug = _slugify_save_name(new_save_name)

    # Preserve original timestamp and model slug when possible
    parts = convo.id.split('_', 2)
    if len(parts) >= 3:
        new_dir_name = f"{parts[0]}_{parts[1]}_{save_slug}"
    else:
        # Fallback to new timestamp if existing id isn't in expected format
        new_dir_name = _create_dir_name(convo.model_name, save_slug)

    new_dir = root / new_dir_name

    if new_dir.exists():
        raise FileExistsError(f"Cannot rename: target already exists ({new_dir_name})")

    old_dir.rename(new_dir)
    convo.id = new_dir_name
    return convo


def export_conversation_md(convo: Conversation, export_dir: Path, filename: Optional[str] = None) -> Path:
    """Export a conversation to a Markdown file.

    Args:
        convo: Conversation to export
        export_dir: Directory to write the export file
        filename: Optional filename (defaults to conversation id or a generated name)

    Returns:
        Path to the written Markdown file
    """
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
