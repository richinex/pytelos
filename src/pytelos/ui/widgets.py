"""Custom Textual widgets for the TUI.

Hides widget implementation details:
- Input history management
- Metrics display formatting
- Log rendering and scrolling
- Chat message rendering
- Image display in execution log
"""

from pathlib import Path

from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.events import Click
from textual.message import Message
from textual.widgets import Button, Input, Markdown, RichLog, Static, TextArea

from .config import LogLevel
from .formatting import clean_latex
from .models import ChatMessage

# IMPORTANT: Import textual_image FIRST to trigger terminal detection
# This must happen before the Textual app starts
try:
    # Import renderable module to trigger terminal graphics detection
    import textual_image.renderable  # noqa: F401 - triggers terminal detection

    # Import the auto-detecting Image widget (uses Sixel/TGP/Halfcell)
    from textual_image.widget import Image as TextualImageWidget
    TEXTUAL_IMAGE_AVAILABLE = True
except ImportError:
    TEXTUAL_IMAGE_AVAILABLE = False
    TextualImageWidget = None


class ClickableMessage(Vertical):
    """A chat message container that copies content when clicked.

    Clicking anywhere on the message copies the raw content to the system clipboard.
    Uses pbcopy on macOS, xclip on Linux, or fallback to Textual's OSC 52.
    """

    def __init__(self, content: str, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._content = content

    def on_click(self, event: Click) -> None:
        """Copy message content to system clipboard when clicked."""
        event.stop()
        try:
            import subprocess
            import sys

            if sys.platform == "darwin":
                # macOS - use pbcopy
                process = subprocess.Popen(
                    ["pbcopy"],
                    stdin=subprocess.PIPE,
                    env={"LANG": "en_US.UTF-8"}
                )
                process.communicate(self._content.encode("utf-8"))
                self.app.notify("Copied to clipboard", timeout=2)
            elif sys.platform.startswith("linux"):
                # Linux - try xclip
                process = subprocess.Popen(
                    ["xclip", "-selection", "clipboard"],
                    stdin=subprocess.PIPE
                )
                process.communicate(self._content.encode("utf-8"))
                self.app.notify("Copied to clipboard", timeout=2)
            else:
                # Fallback to Textual's OSC 52
                self.app.copy_to_clipboard(self._content)
                self.app.notify("Copied (terminal)", timeout=2)
        except Exception:
            # Final fallback
            self.app.copy_to_clipboard(self._content)
            self.app.notify("Copied (terminal)", timeout=2)


class HistoryInput(Input):
    """Input widget with command history support.

    Use Up/Down arrow keys to navigate through history.
    Multi-line pastes are converted to single line (newlines become spaces).
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._history: list[str] = []
        self._history_index: int = -1
        self._current_input: str = ""

    def _on_paste(self, event) -> None:
        """Handle paste events - convert newlines to spaces for single-line input."""
        from textual.events import Paste

        if isinstance(event, Paste) and event.text:
            clean_text = " ".join(event.text.split())
            self.insert_text_at_cursor(clean_text)
            event.prevent_default()
            event.stop()

    def _on_key(self, event) -> None:
        """Handle key events for history navigation."""
        if event.key == "up":
            if self._history:
                if self._history_index == -1:
                    self._current_input = self.value
                    self._history_index = len(self._history) - 1
                elif self._history_index > 0:
                    self._history_index -= 1
                self.value = self._history[self._history_index]
                self.cursor_position = len(self.value)
            event.prevent_default()
            event.stop()
        elif event.key == "down":
            if self._history_index != -1:
                if self._history_index < len(self._history) - 1:
                    self._history_index += 1
                    self.value = self._history[self._history_index]
                else:
                    self._history_index = -1
                    self.value = self._current_input
                self.cursor_position = len(self.value)
            event.prevent_default()
            event.stop()

    def add_to_history(self, command: str) -> None:
        """Add a command to history."""
        if command and (not self._history or self._history[-1] != command):
            self._history.append(command)
        self._history_index = -1
        self._current_input = ""


class ChatInputBar(Horizontal):
    """Chat input bar with TextArea and Send button."""

    class Submitted(Message):
        """Message sent when user submits input."""

        def __init__(self, value: str) -> None:
            super().__init__()
            self.value = value

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._history: list[str] = []
        self._history_index: int = -1

    def compose(self):
        text_area = TextArea(id="chat-input", show_line_numbers=False)
        text_area.cursor_blink = False
        yield text_area
        yield Button("Send", id="send-btn", variant="success").with_tooltip(
            "Submit message (Ctrl+J)"
        )

    def on_mount(self) -> None:
        text_area = self.query_one("#chat-input", TextArea)
        text_area.focus()
        # Disable cursor line highlighting to remove visual artifacts
        text_area.highlight_cursor_line = False

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "send-btn":
            self._submit()

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        """Handle text area changes."""
        pass  # Submit handled via key binding (ctrl+j) or Send button

    def on_key(self, event) -> None:
        """Handle keyboard shortcuts.

        Note: ctrl+enter cannot work in terminals (terminal doesn't pass
        ctrl/shift modifiers with Enter). Use ctrl+j as the submit shortcut.
        """
        if event.key == "ctrl+j":
            self._submit()
            event.prevent_default()
            event.stop()
        elif event.key == "up" and self._is_cursor_at_start():
            self._navigate_history(-1)
            event.prevent_default()
            event.stop()
        elif event.key == "down" and self._is_cursor_at_end():
            self._navigate_history(1)
            event.prevent_default()
            event.stop()

    def _is_cursor_at_start(self) -> bool:
        text_area = self.query_one("#chat-input", TextArea)
        return text_area.cursor_location == (0, 0)

    def _is_cursor_at_end(self) -> bool:
        text_area = self.query_one("#chat-input", TextArea)
        lines = text_area.text.split("\n")
        last_row = len(lines) - 1
        last_col = len(lines[-1]) if lines else 0
        return text_area.cursor_location == (last_row, last_col)

    def _navigate_history(self, direction: int) -> None:
        if not self._history:
            return
        text_area = self.query_one("#chat-input", TextArea)
        if direction < 0:  # Up
            if self._history_index == -1:
                self._history_index = len(self._history) - 1
            elif self._history_index > 0:
                self._history_index -= 1
        else:  # Down
            if self._history_index < len(self._history) - 1:
                self._history_index += 1
            else:
                self._history_index = -1
                text_area.text = ""
                return
        text_area.text = self._history[self._history_index]

    def _submit(self) -> None:
        text_area = self.query_one("#chat-input", TextArea)
        value = text_area.text.strip()
        if value:
            if not self._history or self._history[-1] != value:
                self._history.append(value)
            self._history_index = -1
            text_area.text = ""
            self.post_message(self.Submitted(value))

    def focus_input(self) -> None:
        """Focus the text input."""
        self.query_one("#chat-input", TextArea).focus()


class MetricsPanel(Static):
    """Panel showing execution metrics with visual indicators.

    Tracks main model tokens and execution statistics.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._iterations = 0
        self._code_blocks = 0
        self._time = 0.0
        self._input_tokens = 0
        self._output_tokens = 0
        # Nested/sub-LM tokens (from LM_QUERY calls in executed code)
        self._nested_input_tokens = 0
        self._nested_output_tokens = 0
        self._nested_calls = 0

    def on_mount(self) -> None:
        self._update_display()

    def update_metrics(
        self,
        iterations: int = 0,
        code_blocks: int = 0,
        time: float = 0.0,
        input_tokens: int = 0,
        output_tokens: int = 0,
        nested_input_tokens: int = 0,
        nested_output_tokens: int = 0,
        nested_calls: int = 0,
    ) -> None:
        """Update the metrics display.

        Args:
            iterations: Number of reasoning steps
            code_blocks: Number of code blocks executed
            time: Execution time in seconds
            input_tokens: Main model input tokens
            output_tokens: Main model output tokens
            nested_input_tokens: Sub-LM input tokens (from LM_QUERY)
            nested_output_tokens: Sub-LM output tokens (from LM_QUERY)
            nested_calls: Number of sub-LM calls
        """
        self._iterations = iterations
        self._code_blocks = code_blocks
        self._time = time
        self._input_tokens = input_tokens
        self._output_tokens = output_tokens
        self._nested_input_tokens = nested_input_tokens
        self._nested_output_tokens = nested_output_tokens
        self._nested_calls = nested_calls
        self._update_display()

    def _update_display(self) -> None:
        # Main model tokens
        main_tokens = self._input_tokens + self._output_tokens
        # Nested/sub-LM tokens
        nested_tokens = self._nested_input_tokens + self._nested_output_tokens

        # Build display with main tokens always shown
        parts = [
            f"[bold cyan]Iter:[/] {self._iterations}",
            f"[bold green]Code:[/] {self._code_blocks}",
            f"[bold yellow]Time:[/] {self._time:.2f}s",
            f"[bold magenta]Main:[/] {main_tokens:,} "
            f"[dim]({self._input_tokens:,}/{self._output_tokens:,})[/]",
        ]

        # Only show nested tokens if there were any sub-LM calls
        if self._nested_calls > 0:
            parts.append(
                f"[bold blue]Sub-LM:[/] {nested_tokens:,} "
                f"[dim]({self._nested_calls} calls)[/]"
            )

        self.update("  ".join(parts))

    def get_plain_text(self) -> str:
        """Get metrics as plain text for clipboard."""
        main_tokens = self._input_tokens + self._output_tokens
        nested_tokens = self._nested_input_tokens + self._nested_output_tokens
        text = (
            f"Iterations: {self._iterations}  "
            f"Code Blocks: {self._code_blocks}  "
            f"Time: {self._time:.2f}s  "
            f"Main Tokens: {main_tokens} ({self._input_tokens}/{self._output_tokens})"
        )
        if self._nested_calls > 0:
            text += f"  Sub-LM Tokens: {nested_tokens} ({self._nested_calls} calls)"
        return text


class ExecutionLog(RichLog):
    """RichLog-based execution log with markup support."""

    BORDER_TITLE = "Execution"
    BORDER_SUBTITLE = "Code output"
    ALLOW_MAXIMIZE = True

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(
            *args,
            markup=True,
            highlight=False,
            auto_scroll=True,
            wrap=False,  # Disable wrap to enable horizontal scroll
            **kwargs
        )
        self._is_executing = False

    def write(self, content, markup: bool = True) -> None:
        """Write content to the log.

        Args:
            content: String or Rich renderable to write
            markup: If True, parse Rich markup in strings. Set False for raw output.
        """
        if isinstance(content, str) and not markup:
            # For plain text, escape any markup-like content
            from rich.text import Text
            super().write(Text(content))
        else:
            super().write(content)

    def start_execution(self) -> None:
        """Mark execution as started."""
        self._is_executing = True
        self.border_subtitle = "Running..."
        self.add_class("executing")

    def end_execution(self, success: bool = True) -> None:
        """Mark execution as complete."""
        self._is_executing = False
        self.border_subtitle = "Complete" if success else "Error"
        self.remove_class("executing")

    def clear(self) -> None:
        """Clear log and reset subtitle."""
        super().clear()
        self.border_subtitle = "Code output"

    def get_plain_text(self) -> str:
        """Get plain text content of the log for copying."""
        lines_text = []
        for line in self.lines:
            # Strip objects have a .text property or iterate to get segments
            if hasattr(line, 'text'):
                lines_text.append(line.text)
            elif hasattr(line, '__iter__'):
                text = "".join(seg.text for seg in line if hasattr(seg, 'text'))
                lines_text.append(text)
        return "\n".join(lines_text)

    def on_click(self, event: Click) -> None:
        """Copy log content to clipboard when clicked."""
        event.stop()
        text = self.get_plain_text()
        if not text.strip():
            self.app.notify("Log is empty", timeout=2)
            return
        try:
            import pyperclip
            pyperclip.copy(text)
            self.app.notify("Execution log copied", timeout=2)
        except Exception:
            self.app.copy_to_clipboard(text)
            self.app.notify("Execution log copied (terminal)", timeout=2)


class DebugPanel(RichLog):
    """Log panel for real-time execution tracing with level filtering.

    Shows timestamped log messages from all components.
    Supports standard log levels: DEBUG < INFO < WARNING < ERROR.
    Hidden by default, shown with --log-level flag or toggled with Ctrl+D.
    """

    BORDER_TITLE = "Log"
    BORDER_SUBTITLE = "Trace log"

    def __init__(self, *args, log_level: int = LogLevel.DEBUG, **kwargs) -> None:
        super().__init__(
            *args,
            markup=True,
            highlight=False,
            auto_scroll=True,
            wrap=False,
            **kwargs
        )
        self._log_level = log_level

    @property
    def log_level(self) -> int:
        """Current log level threshold."""
        return self._log_level

    @log_level.setter
    def log_level(self, level: int) -> None:
        """Set log level threshold."""
        self._log_level = level
        self._update_subtitle()

    def _update_subtitle(self) -> None:
        """Update subtitle to show current log level."""
        if self.display:
            level_name = LogLevel.name(self._log_level)
            self.border_subtitle = f"Level: {level_name}"
        else:
            self.border_subtitle = "Hidden"

    def on_mount(self) -> None:
        """Hide by default."""
        self.display = False

    def log(
        self,
        component: str,
        message: str,
        level: int = LogLevel.DEBUG
    ) -> None:
        """Add a log entry if it meets the current level threshold.

        Args:
            component: Component name (TUI, CORE, REPL, LLM, etc.)
            message: Log message
            level: Log level (LogLevel.DEBUG, INFO, WARNING, ERROR)
        """
        # Filter by level - only show if message level >= threshold
        if level < self._log_level:
            return

        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")

        # Color-code by level
        level_colors = {
            LogLevel.DEBUG: "dim white",
            LogLevel.INFO: "cyan",
            LogLevel.WARNING: "yellow",
            LogLevel.ERROR: "red",
        }
        level_color = level_colors.get(level, "white")
        level_name = LogLevel.name(level)

        # Color-code by component
        component_colors = {
            "TUI": "cyan",
            "CORE": "green",
            "REPL": "yellow",
            "LLM": "magenta",
            "DOCKER": "blue",
            "Planner": "bright_blue",
            "Memory": "bright_green",
            "Step": "bright_yellow",
            "Tool": "bright_cyan",
            "Search": "bright_magenta",
            "Embed": "bright_white",
        }
        comp_color = component_colors.get(component, "white")

        self.write(
            f"[dim]{timestamp}[/] "
            f"[{level_color}]{level_name:<5}[/] "
            f"[{comp_color}][{component}][/] {message}"
        )

    def debug(self, component: str, message: str) -> None:
        """Log a DEBUG level message."""
        self.log(component, message, LogLevel.DEBUG)

    def info(self, component: str, message: str) -> None:
        """Log an INFO level message."""
        self.log(component, message, LogLevel.INFO)

    def warning(self, component: str, message: str) -> None:
        """Log a WARNING level message."""
        self.log(component, message, LogLevel.WARNING)

    def error(self, component: str, message: str) -> None:
        """Log an ERROR level message."""
        self.log(component, message, LogLevel.ERROR)

    def show(self) -> None:
        """Show the log panel."""
        self.display = True
        self._update_subtitle()

    def hide(self) -> None:
        """Hide the log panel."""
        self.display = False
        self.border_subtitle = "Hidden"

    def toggle(self) -> bool:
        """Toggle visibility. Returns new state."""
        if self.display:
            self.hide()
            return False
        else:
            self.show()
            return True

    def get_plain_text(self) -> str:
        """Get plain text content of the log for copying."""
        lines_text = []
        for line in self.lines:
            # Strip objects have a .text property or iterate to get segments
            if hasattr(line, 'text'):
                lines_text.append(line.text)
            elif hasattr(line, '__iter__'):
                text = "".join(seg.text for seg in line if hasattr(seg, 'text'))
                lines_text.append(text)
        return "\n".join(lines_text)

    def on_click(self, event: Click) -> None:
        """Copy log content to clipboard when clicked."""
        event.stop()
        text = self.get_plain_text()
        if not text.strip():
            self.app.notify("Debug log is empty", timeout=2)
            return
        try:
            import pyperclip
            pyperclip.copy(text)
            self.app.notify("Debug log copied", timeout=2)
        except Exception:
            self.app.copy_to_clipboard(text)
            self.app.notify("Debug log copied (terminal)", timeout=2)


class PlotPanel(Static):
    """Panel for displaying plots using high-quality terminal graphics.

    Uses textual_image.widget.Image which auto-detects the best rendering
    method: Sixel (iTerm2, xterm), TGP (Kitty), or halfcell fallback.

    Hidden by default, only shown when a plot is displayed.
    Supports multiple images with left/right navigation.
    """

    BORDER_TITLE = "Plot"
    BORDER_SUBTITLE = ""
    can_focus = True  # Allow focus for key navigation

    BINDINGS = [
        ("left", "prev_image", "Prev"),
        ("right", "next_image", "Next"),
    ]

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._image_widget: TextualImageWidget | None = None
        self._images: list[str] = []  # All image paths
        self._current_index: int = 0

    def on_mount(self) -> None:
        """Hide panel by default until a plot is shown."""
        self.display = False

    def compose(self):
        """Compose the widget - starts empty."""
        if TEXTUAL_IMAGE_AVAILABLE and TextualImageWidget is not None:
            # Create image widget with no initial image
            self._image_widget = TextualImageWidget(None, id="plot-image")
            yield self._image_widget
        else:
            yield Static("[dim]Image display not available[/]", id="plot-fallback")

    def _update_subtitle(self) -> None:
        """Update subtitle with current image name and index."""
        if not self._images:
            self.border_subtitle = ""
            return

        path = Path(self._images[self._current_index])
        if len(self._images) > 1:
            self.border_subtitle = f"{path.name} ({self._current_index + 1}/{len(self._images)}) [</>]"
        else:
            self.border_subtitle = path.name

    def _display_current(self) -> bool:
        """Display the current image."""
        if not self._images or self._image_widget is None:
            return False

        path = self._images[self._current_index]
        try:
            self._image_widget.image = path
            self._update_subtitle()
            self.display = True
            return True
        except Exception as e:
            self.border_subtitle = f"Error: {e}"
            return False

    def show_image(self, image_path: str | Path) -> bool:
        """Add and display an image in the panel.

        Args:
            image_path: Path to the image file

        Returns:
            True if image was displayed, False otherwise
        """
        path = Path(image_path)
        if not path.exists():
            self.border_subtitle = "File not found"
            return False

        path_str = str(path)

        # Add to list if not already present
        if path_str not in self._images:
            self._images.append(path_str)

        # Show the newly added image (jump to it)
        self._current_index = self._images.index(path_str)
        return self._display_current()

    def action_prev_image(self) -> None:
        """Show the previous image."""
        if len(self._images) > 1:
            self._current_index = (self._current_index - 1) % len(self._images)
            self._display_current()

    def action_next_image(self) -> None:
        """Show the next image."""
        if len(self._images) > 1:
            self._current_index = (self._current_index + 1) % len(self._images)
            self._display_current()

    def clear_image(self) -> None:
        """Clear all images and hide the panel."""
        if self._image_widget is not None:
            self._image_widget.image = None
        self._images.clear()
        self._current_index = 0
        self.border_subtitle = ""
        self.display = False


class ChatHistoryWidget(VerticalScroll):
    """Scrollable chat history with text selection support."""

    BORDER_TITLE = "Chat"
    BORDER_SUBTITLE = "Conversation history"
    ALLOW_MAXIMIZE = True
    ALLOW_SELECT = True

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._messages: list[ChatMessage] = []
        self._message_count = 0
        self._user_has_interacted = False

    def add_message(self, role: str, content: str) -> None:
        """Add a message to the chat history."""
        msg = ChatMessage(role=role, content=content)
        self._messages.append(msg)
        self._message_count += 1
        self._render_message(msg)
        # Update subtitle with message count
        self.border_subtitle = f"{self._message_count} messages"

        # After first user message, enable auto-scroll to bottom
        if role == "user" and not self._user_has_interacted:
            self._user_has_interacted = True

        # Scroll to bottom for new messages (after user has started chatting)
        if self._user_has_interacted:
            self.scroll_end(animate=False)

    def get_last_response(self) -> str | None:
        """Get the last assistant response."""
        for msg in reversed(self._messages):
            if msg.role == "assistant":
                return msg.content
        return None

    def clear_history(self) -> None:
        """Clear the chat history."""
        self._messages.clear()
        self._message_count = 0
        self.remove_children()
        self.border_subtitle = "Conversation history"

    def _render_message(self, msg: ChatMessage) -> None:
        """Render a single message to the display."""
        if msg.role == "user":
            prefix = "You"
            border_class = "user-message"
            icon = ">"
        else:
            prefix = "Assistant"
            border_class = "assistant-message"
            icon = "<"

        timestamp = msg.timestamp.strftime("%H:%M:%S")
        # Visual header with icon
        header_text = f"{icon} {prefix} [{timestamp}]"

        # Clean LaTeX from content
        cleaned_content = clean_latex(msg.content)

        # Create a clickable container - clicking copies content
        container = ClickableMessage(content=msg.content, classes=f"chat-message {border_class}")
        container.compose_add_child(Static(header_text, classes="message-header"))

        if msg.role == "assistant":
            # Check if content looks like code (multi-line with code patterns)
            content_stripped = cleaned_content.strip()
            lines = content_stripped.split("\n")
            code_indicators = ["def ", "class ", "import ", "from ", "async def ",
                               "if __name__", "return ", "yield ", "for ", "while "]
            is_pure_code = (
                len(lines) > 1 and
                any(content_stripped.startswith(ind) for ind in code_indicators) and
                "```" not in content_stripped
            )

            if is_pure_code:
                # Code output - wrap in markdown code fences for syntax highlighting
                formatted_content = f"```python\n{content_stripped}\n```"
                md_widget = Markdown(formatted_content, classes="message-content")
                container.compose_add_child(md_widget)
            elif "\n" in content_stripped and "```" not in content_stripped:
                # Multi-line text without code fences - use Static to preserve formatting
                container.compose_add_child(Static(cleaned_content, classes="message-content"))
            else:
                # Regular text or already has markdown - use Markdown widget
                md_widget = Markdown(cleaned_content, classes="message-content")
                container.compose_add_child(md_widget)
        else:
            # Plain text for user messages
            container.compose_add_child(Static(cleaned_content, classes="message-content"))

        self.mount(container)
