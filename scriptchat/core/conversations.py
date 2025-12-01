# Copyright 2024 ScriptChat contributors
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

"""Conversation management for ScriptChat."""

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
    provider_id: str
    model_name: str
    temperature: float
    system_prompt: Optional[str] = None
    messages: list[Message] = field(default_factory=list)
    tokens_in: int = 0
    tokens_out: int = 0
    reasoning_level: Optional[str] = None
    thinking_budget: Optional[int] = None  # Explicit thinking budget (tokens) for Anthropic
    context_length_configured: Optional[int] = None  # Max context length model is running with
    context_length_used: Optional[int] = None  # Current context length used (last tokens_in)
    tags: dict = field(default_factory=dict)
    file_references: dict[str, dict] = field(default_factory=dict)  # key -> {"path": str, "sha256": str}
    parent_id: Optional[str] = None  # ID of parent conversation if this is a branch
    branched_at: Optional[str] = None  # ISO timestamp when branch was created


@dataclass
class ConversationSummary:
    """Summary information about a saved conversation."""
    dir_name: str
    created_at: str
    model_name: str
    display_name: str
    last_modified: str = ""
    tags: dict = field(default_factory=dict)
    parent_id: Optional[str] = None


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
                tags = meta.get('tags', {}) or {}
                last_modified = meta.get('last_modified', '')
                parent_id = meta.get('parent_id')
            except (json.JSONDecodeError, IOError):
                created_at = ''
                model_name = 'unknown'
                tags = {}
                last_modified = ''
                parent_id = None
        else:
            created_at = ''
            model_name = 'unknown'
            tags = {}
            last_modified = ''
            parent_id = None

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
            display_name=display_name,
            last_modified=last_modified,
            tags=tags,
            parent_id=parent_id
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
        provider_id = meta.get('provider_id', 'ollama')
        temperature = meta.get('temperature', 0.7)
        context_length_configured = meta.get('context_length_configured', None)
        context_length_used = meta.get('context_length_used', None)
        system_prompt = meta.get('system_prompt') or meta.get('system_prompt_snapshot')
        tags = meta.get('tags', {}) or {}
        last_modified = meta.get('last_modified')
        reasoning_level = meta.get('reasoning_level')
        file_references = meta.get('file_references', {}) or {}
        parent_id = meta.get('parent_id')
        branched_at = meta.get('branched_at')
    else:
        # Try to infer from directory name
        parts = dir_name.split('_', 2)
        if len(parts) >= 2:
            model_name = parts[1]
        else:
            model_name = 'unknown'
        temperature = 0.7
        provider_id = 'ollama'
        context_length_configured = None
        context_length_used = None
        system_prompt = None
        tags = {}
        last_modified = None
        reasoning_level = None
        file_references = {}
        parent_id = None
        branched_at = None

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

    # Prepend system prompt if present in metadata
    if system_prompt:
        messages.insert(0, Message(role='system', content=system_prompt))

    convo = Conversation(
        id=dir_name,
        provider_id=provider_id,
        model_name=model_name,
        temperature=temperature,
        system_prompt=system_prompt,
        messages=messages,
        tokens_in=0,
        tokens_out=0,
        reasoning_level=reasoning_level,
        context_length_configured=context_length_configured,
        context_length_used=context_length_used,
        tags=tags,
        file_references=file_references,
        parent_id=parent_id,
        branched_at=branched_at
    )

    # Rehydrate file registry strictly from meta file references
    file_registry: dict = {}
    for key, ref_value in file_references.items():
        # Handle both old format (string path) and new format (dict with path and sha256)
        if isinstance(ref_value, str):
            path_str = ref_value
        else:
            path_str = ref_value.get("path", "")
        p = Path(path_str).expanduser()
        if p.exists() and p.is_file():
            try:
                content = p.read_text(encoding='utf-8')
                full_path = str(p.resolve())
                file_registry[key] = {"content": content, "full_path": full_path}
                basename = Path(full_path).name
                if basename not in file_registry:
                    file_registry[basename] = {"content": content, "full_path": full_path}
            except Exception:
                file_registry[key] = {"content": "", "full_path": str(p), "missing": True}
        else:
            file_registry[key] = {"content": "", "full_path": str(p), "missing": True}

    convo.file_registry = file_registry  # type: ignore[attr-defined]
    return convo


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
    created_at_val = None
    meta_path = conv_dir / 'meta.json'
    if meta_path.exists():
        try:
            existing_meta = json.loads(meta_path.read_text())
            created_at_val = existing_meta.get('created_at')
        except Exception:
            created_at_val = None
    now = datetime.now()
    meta = {
        'model': convo.model_name,
        'provider_id': convo.provider_id,
        'temperature': convo.temperature,
        'created_at': created_at_val or now.isoformat(),
        'last_modified': now.isoformat()
    }
    if convo.reasoning_level:
        meta['reasoning_level'] = convo.reasoning_level
    if convo.file_references:
        meta['file_references'] = convo.file_references
    prompt_snapshot = system_prompt or convo.system_prompt
    if prompt_snapshot:
        meta['system_prompt'] = prompt_snapshot
    if convo.context_length_configured is not None:
        meta['context_length_configured'] = convo.context_length_configured
    if convo.context_length_used is not None:
        meta['context_length_used'] = convo.context_length_used
    if convo.tags:
        meta['tags'] = convo.tags
    if convo.parent_id:
        meta['parent_id'] = convo.parent_id
    if convo.branched_at:
        meta['branched_at'] = convo.branched_at

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

    # Update meta.json with new timestamp and parent info
    meta_path = new_conv_dir / 'meta.json'
    now = datetime.now()
    parent_id = convo.id
    branched_at = now.isoformat()
    if meta_path.exists():
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        meta['created_at'] = now.isoformat()
        meta['last_modified'] = now.isoformat()
        meta['parent_id'] = parent_id
        meta['branched_at'] = branched_at
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)

    # Create new conversation object
    new_convo = Conversation(
        id=new_dir_name,
        provider_id=convo.provider_id,
        model_name=convo.model_name,
        temperature=convo.temperature,
        messages=convo.messages.copy(),
        system_prompt=convo.system_prompt,
        tokens_in=convo.tokens_in,
        tokens_out=convo.tokens_out,
        reasoning_level=convo.reasoning_level,
        tags=convo.tags.copy(),
        file_references=convo.file_references.copy() if convo.file_references else {},
        parent_id=parent_id,
        branched_at=branched_at
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

    Also updates parent_id references in any child conversations (branches).

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

    old_id = convo.id
    old_dir = root / old_id
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

    # Update parent_id in any child conversations that reference the old id
    _update_children_parent_id(root, old_id, new_dir_name)

    return convo


def _update_children_parent_id(root: Path, old_parent_id: str, new_parent_id: str) -> None:
    """Update parent_id references in child conversations after a rename.

    Args:
        root: Conversations root directory
        old_parent_id: The old parent conversation id
        new_parent_id: The new parent conversation id
    """
    for conv_dir in root.iterdir():
        if not conv_dir.is_dir():
            continue
        meta_path = conv_dir / 'meta.json'
        if not meta_path.exists():
            continue
        try:
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            if meta.get('parent_id') == old_parent_id:
                meta['parent_id'] = new_parent_id
                with open(meta_path, 'w') as f:
                    json.dump(meta, f, indent=2)
        except Exception:
            # Skip any conversations we can't read/write
            continue
