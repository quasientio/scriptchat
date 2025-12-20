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

"""UI event system for testing and debugging."""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class UIEventType(Enum):
    """Types of UI events for logging and testing."""
    # Lifecycle
    APP_STARTED = auto()
    APP_STOPPED = auto()

    # Display updates
    CONVERSATION_UPDATED = auto()
    STATUS_BAR_UPDATED = auto()

    # Input state
    INPUT_CHANGED = auto()
    INPUT_CLEARED = auto()
    INPUT_SUBMITTED = auto()

    # Menus and prompts
    SELECTION_MENU_SHOWN = auto()
    SELECTION_MENU_HIDDEN = auto()
    SELECTION_MENU_NAVIGATED = auto()
    SELECTION_MENU_SELECTED = auto()
    PROMPT_SHOWN = auto()
    PROMPT_ANSWERED = auto()

    # LLM interaction
    THINKING_STARTED = auto()
    THINKING_STOPPED = auto()
    MESSAGE_QUEUED = auto()
    MESSAGE_SENT = auto()
    RESPONSE_CHUNK = auto()
    RESPONSE_COMPLETE = auto()
    INFERENCE_CANCELLED = auto()

    # Commands
    COMMAND_RECEIVED = auto()
    COMMAND_EXECUTED = auto()
    SYSTEM_MESSAGE_ADDED = auto()

    # Focus
    FOCUS_CHANGED = auto()

    # History
    HISTORY_NAVIGATED = auto()


@dataclass
class UIEvent:
    """A single UI event with timestamp and data."""
    type: UIEventType
    timestamp: str
    data: dict = field(default_factory=dict)

    def to_log_line(self) -> str:
        """Format as a parseable log line."""
        data_json = json.dumps(self.data, default=str)
        return f"UI_EVENT|{self.timestamp}|{self.type.name}|{data_json}"

    @classmethod
    def from_log_line(cls, line: str) -> Optional['UIEvent']:
        """Parse from log line format."""
        if not line.startswith("UI_EVENT|"):
            return None
        try:
            parts = line.split("|", 3)
            if len(parts) != 4:
                return None
            _, timestamp, event_name, data_json = parts
            return cls(
                type=UIEventType[event_name],
                timestamp=timestamp,
                data=json.loads(data_json)
            )
        except (KeyError, json.JSONDecodeError):
            return None


class UIEventEmitter:
    """Emits and tracks UI events for testing and debugging."""

    def __init__(self, log_events: bool = False):
        self._listeners: list[Callable[[UIEvent], None]] = []
        self._event_log: list[UIEvent] = []
        self._log_events = log_events
        self._max_log_size = 1000  # Prevent unbounded growth

    def add_listener(self, callback: Callable[[UIEvent], None]):
        """Add an event listener."""
        self._listeners.append(callback)

    def remove_listener(self, callback: Callable[[UIEvent], None]):
        """Remove an event listener."""
        if callback in self._listeners:
            self._listeners.remove(callback)

    def emit(self, event_type: UIEventType, **data):
        """Emit a UI event."""
        event = UIEvent(
            type=event_type,
            timestamp=datetime.now().isoformat(),
            data=data
        )

        # Store in log (with size limit)
        self._event_log.append(event)
        if len(self._event_log) > self._max_log_size:
            self._event_log = self._event_log[-self._max_log_size:]

        # Log if enabled
        if self._log_events:
            logger.info(event.to_log_line())

        # Notify listeners
        for listener in self._listeners:
            try:
                listener(event)
            except Exception as e:
                logger.debug(f"Event listener error: {e}")

    def get_events(
        self,
        event_type: Optional[UIEventType] = None,
        since: Optional[str] = None
    ) -> list[UIEvent]:
        """Get events, optionally filtered by type or timestamp."""
        events = self._event_log
        if event_type is not None:
            events = [e for e in events if e.type == event_type]
        if since is not None:
            events = [e for e in events if e.timestamp > since]
        return list(events)

    def get_last_event(self, event_type: Optional[UIEventType] = None) -> Optional[UIEvent]:
        """Get the most recent event of a type."""
        events = self.get_events(event_type)
        return events[-1] if events else None

    def clear(self):
        """Clear the event log."""
        self._event_log.clear()

    def wait_for_event(
        self,
        event_type: UIEventType,
        timeout: float = 5.0,
        predicate: Optional[Callable[[UIEvent], bool]] = None
    ) -> Optional[UIEvent]:
        """Wait for an event (for use in tests)."""
        import threading

        result = [None]
        found = threading.Event()

        def listener(event: UIEvent):
            if event.type == event_type:
                if predicate is None or predicate(event):
                    result[0] = event
                    found.set()

        self.add_listener(listener)
        try:
            found.wait(timeout)
            return result[0]
        finally:
            self.remove_listener(listener)


@dataclass
class UIState:
    """Complete snapshot of UI state for testing."""
    # Conversation pane
    conversation_text: str
    conversation_message_count: int

    # Status bar
    status_bar_text: str
    provider_id: str
    model_name: str
    tokens_in: int
    tokens_out: int
    conversation_id: Optional[str]

    # Input pane
    input_text: str
    input_cursor_position: int

    # Modes
    thinking: bool
    multiline_mode: bool
    streaming: bool

    # Selection menu
    selection_menu_visible: bool

    # Prompts
    prompt_message: str
    has_pending_callback: bool
    expecting_single_key: bool

    # History
    history_index: Optional[int]
    history_length: int

    # Script state
    running_script: bool
    script_queue_length: int
    message_queue_length: int

    # Fields with default values must come last
    selection_menu_items: list = field(default_factory=list)
    selection_menu_index: int = 0


def parse_ui_events_from_log(log_content: str) -> list[UIEvent]:
    """Parse UI events from log file content."""
    events = []
    for line in log_content.splitlines():
        if "UI_EVENT|" in line:
            # Extract the UI_EVENT part from the log line
            start = line.find("UI_EVENT|")
            if start >= 0:
                event = UIEvent.from_log_line(line[start:])
                if event:
                    events.append(event)
    return events
