"""Interactive selection menu widget for ScriptChat."""

from typing import TYPE_CHECKING, Any, Callable, Optional

from prompt_toolkit.filters import Condition
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import ConditionalContainer, Window
from prompt_toolkit.layout.controls import FormattedTextControl

if TYPE_CHECKING:
    from .app import ScriptChatUI


class SelectionMenu:
    """Arrow-key navigable selection menu overlay.

    Provides a Claude Code-style selection interface that appears below
    the input area. Users can navigate with arrow keys and select with
    Enter/Tab.
    """

    def __init__(self, app: 'ScriptChatUI', max_visible: int = 10):
        """Initialize the selection menu.

        Args:
            app: Parent ScriptChatUI instance
            max_visible: Maximum number of items to display at once
        """
        self.app = app
        self.max_visible = max_visible
        self.items: list[tuple[Any, str]] = []
        self.selected_index: int = 0
        self.viewport_start: int = 0
        self._visible: bool = False
        self._on_select: Optional[Callable[[Any], None]] = None
        self._on_cancel: Optional[Callable[[], None]] = None

        # Create the display control
        self.control = FormattedTextControl(text=self._get_menu_text)
        self.window = Window(
            content=self.control,
            height=self._get_height,
            style='class:selection-menu',
            dont_extend_height=True
        )

    @property
    def is_visible(self) -> bool:
        """Whether the menu is currently visible."""
        return self._visible

    def _get_height(self) -> int:
        """Calculate dynamic height based on items."""
        if not self.items:
            return 0
        # Items + border lines + hint line + scroll indicators
        base_height = min(len(self.items), self.max_visible) + 3
        if self.viewport_start > 0:
            base_height += 1  # "more above" line
        if self.viewport_start + self.max_visible < len(self.items):
            base_height += 1  # "more below" line
        return base_height

    def show(
        self,
        items: list[tuple[Any, str]],
        on_select: Callable[[Any], None],
        on_cancel: Optional[Callable[[], None]] = None
    ):
        """Display the selection menu with given items.

        Args:
            items: List of (value, display_label) tuples
            on_select: Callback when item is selected, receives the value
            on_cancel: Optional callback when selection is cancelled
        """
        self.items = items
        self.selected_index = 0
        self.viewport_start = 0
        self._on_select = on_select
        self._on_cancel = on_cancel
        self._visible = True
        self.app.app.invalidate()

    def hide(self):
        """Hide the selection menu and restore input focus."""
        self._visible = False
        self.app.app.invalidate()

    def move_up(self):
        """Move selection up one item."""
        if self.selected_index > 0:
            self.selected_index -= 1
            # Scroll viewport if needed
            if self.selected_index < self.viewport_start:
                self.viewport_start = self.selected_index
            self.app.app.invalidate()

    def move_down(self):
        """Move selection down one item."""
        if self.selected_index < len(self.items) - 1:
            self.selected_index += 1
            # Scroll viewport if needed
            if self.selected_index >= self.viewport_start + self.max_visible:
                self.viewport_start = self.selected_index - self.max_visible + 1
            self.app.app.invalidate()

    def select_current(self):
        """Confirm current selection and invoke callback."""
        if self.items and self._on_select:
            value, _ = self.items[self.selected_index]
            callback = self._on_select
            self.hide()
            callback(value)

    def cancel(self):
        """Cancel selection and invoke cancel callback."""
        callback = self._on_cancel
        self.hide()
        if callback:
            callback()

    def _get_menu_text(self):
        """Generate formatted text for menu display."""
        if not self.items:
            return []

        lines = []
        menu_width = 80

        # Top border
        lines.append(('class:menu-border', '\u250c' + '\u2500' * menu_width + '\u2510\n'))

        # Calculate visible range
        visible_end = min(self.viewport_start + self.max_visible, len(self.items))

        # Scroll up indicator
        if self.viewport_start > 0:
            indicator = ' \u25b2 more above...'
            padded = indicator.ljust(menu_width)
            lines.append(('class:menu-scroll', f'\u2502{padded}\u2502\n'))

        # Visible items
        for i in range(self.viewport_start, visible_end):
            value, label = self.items[i]
            prefix = '\u25b6 ' if i == self.selected_index else '  '
            style = 'class:menu-selected' if i == self.selected_index else 'class:menu-item'
            # Format: [index] label
            display = f'{prefix}[{i}] {label}'
            # Truncate and pad to fixed width
            if len(display) > menu_width:
                display = display[:menu_width - 1] + '\u2026'
            padded = display.ljust(menu_width)
            lines.append((style, f'\u2502{padded}\u2502\n'))

        # Scroll down indicator
        if visible_end < len(self.items):
            indicator = ' \u25bc more below...'
            padded = indicator.ljust(menu_width)
            lines.append(('class:menu-scroll', f'\u2502{padded}\u2502\n'))

        # Bottom border
        lines.append(('class:menu-border', '\u2514' + '\u2500' * menu_width + '\u2518\n'))

        # Hint line
        lines.append(('class:menu-hint', 'j/k or Up/Down: Navigate | Enter/Tab: Select | Esc: Cancel'))

        return lines

    def get_container(self) -> ConditionalContainer:
        """Return a ConditionalContainer wrapping the menu window."""
        return ConditionalContainer(
            content=self.window,
            filter=Condition(lambda: self._visible)
        )

    def get_key_bindings(self) -> KeyBindings:
        """Return key bindings for the menu."""
        kb = KeyBindings()

        menu_visible = Condition(lambda: self._visible)

        @kb.add('up', filter=menu_visible)
        @kb.add('k', filter=menu_visible)
        def handle_up(event):
            self.move_up()

        @kb.add('down', filter=menu_visible)
        @kb.add('j', filter=menu_visible)
        def handle_down(event):
            self.move_down()

        @kb.add('enter', filter=menu_visible)
        def handle_enter(event):
            self.select_current()

        @kb.add('tab', filter=menu_visible)
        def handle_tab(event):
            self.select_current()

        @kb.add('escape', filter=menu_visible)
        def handle_escape(event):
            self.cancel()

        return kb
