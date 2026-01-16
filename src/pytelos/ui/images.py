"""Image display module for the TUI.

Hidden design decisions:
- Image rendering approach (textual-image for proper Sixel/TGP support)
- Image sizing and scaling logic
- Path handling for temporary plot files
"""

from pathlib import Path
from typing import Any

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Static

# Try to import textual-image for rendering
# Use HalfcellImage renderable directly for better control
try:
    from textual_image.renderable import HalfcellImage
    TEXTUAL_IMAGE_AVAILABLE = True
except ImportError:
    TEXTUAL_IMAGE_AVAILABLE = False
    HalfcellImage = None


class ImagePanel(Widget):
    """Collapsible panel for displaying plots with proper Sixel support.

    Uses textual-image's AutoImage widget which automatically chooses
    the best rendering method (Sixel, TGP, or halfcells) for the terminal.
    """

    DEFAULT_CSS = """
    ImagePanel {
        height: auto;
        width: 100%;
        min-height: 15;
        max-height: 50;
        border: solid $primary;
        background: $surface;
        display: none;
    }

    ImagePanel.has-image {
        display: block;
    }

    ImagePanel .image-header {
        height: 1;
        padding: 0 1;
        background: $primary-darken-2;
    }

    ImagePanel .image-path {
        color: $text-muted;
        text-style: italic;
        padding: 0 1;
    }

    ImagePanel .image-container {
        height: auto;
        min-height: 10;
        max-height: 45;
        padding: 0;
    }

    ImagePanel RichLog {
        height: auto;
        min-height: 10;
        max-height: 40;
        border: none;
        background: transparent;
    }

    ImagePanel .image-placeholder {
        color: $text-muted;
        text-align: center;
        padding: 1;
    }
    """

    class PlotDisplayed(Message):
        """Message sent when a plot is displayed."""

        def __init__(self, path: str) -> None:
            super().__init__()
            self.path = path

    def __init__(
        self,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        """Initialize the image panel."""
        super().__init__(name=name, id=id, classes=classes)
        self._image_path: Path | None = None
        self._current_image_widget: Any = None

    def compose(self) -> ComposeResult:
        """Compose the widget content."""
        yield Static("Plot Output", classes="image-header")
        yield Vertical(id="image-container", classes="image-container")
        yield Static("", id="image-path-label", classes="image-path")

    def display_image(self, image_path: str | Path) -> bool:
        """Display an image in the panel.

        Args:
            image_path: Path to the image file

        Returns:
            True if image was displayed successfully
        """
        path = Path(image_path)
        if not path.exists():
            return False

        self._image_path = path
        container = self.query_one("#image-container", Vertical)
        path_label = self.query_one("#image-path-label", Static)

        # Remove any existing image widget
        container.remove_children()

        if TEXTUAL_IMAGE_AVAILABLE and HalfcellImage is not None:
            try:
                from textual.widgets import RichLog
                # Use HalfcellImage renderable with larger width for better quality
                # Width of 200 gives ~200 horizontal pixels, ~100 vertical pixels
                img_renderable = HalfcellImage(str(path), width=200)
                # Use a RichLog to display the Rich renderable
                img_log = RichLog(markup=False, highlight=False, wrap=False)
                img_log.styles.height = "auto"
                img_log.styles.min_height = 10
                img_log.styles.max_height = 30
                img_log.styles.border = "none"
                img_log.styles.padding = (0, 0)
                self._current_image_widget = img_log
                container.mount(img_log)
                img_log.write(img_renderable)
                path_label.update(f"Saved: {path}")
                self.add_class("has-image")
                self.post_message(self.PlotDisplayed(str(path)))
                return True
            except Exception as e:
                # Fallback to showing path only
                container.mount(Static(f"[Image error: {e}]", classes="image-placeholder"))
                path_label.update(f"File: {path}")
                self.add_class("has-image")
                return False
        else:
            # No textual-image available
            container.mount(Static("[textual-image not installed]", classes="image-placeholder"))
            path_label.update(f"File: {path}")
            self.add_class("has-image")
            return False

    def clear(self) -> None:
        """Clear the image display."""
        self._image_path = None
        self._current_image_widget = None
        container = self.query_one("#image-container", Vertical)
        container.remove_children()
        path_label = self.query_one("#image-path-label", Static)
        path_label.update("")
        self.remove_class("has-image")

    @property
    def image_path(self) -> Path | None:
        """Get the current image path."""
        return self._image_path


class PlotResult:
    """Represents a plot result from REPL execution.

    This is returned by PLOT_FIGURE() to signal the TUI
    that an image should be displayed.
    """

    def __init__(self, path: str, plot_type: str = "matplotlib") -> None:
        """Initialize plot result.

        Args:
            path: Path to the saved plot image
            plot_type: Type of plot (matplotlib, plotext, etc.)
        """
        self.path = path
        self.plot_type = plot_type

    def __repr__(self) -> str:
        return f"PlotResult(path='{self.path}', type='{self.plot_type}')"

    def to_dict(self) -> dict[str, str]:
        """Convert to dictionary for serialization."""
        return {"type": "plot", "path": self.path, "plot_type": self.plot_type}


def is_plot_result(value: Any) -> bool:
    """Check if a value is a plot result.

    Args:
        value: Value to check

    Returns:
        True if value represents a plot result
    """
    if isinstance(value, PlotResult):
        return True
    if isinstance(value, dict):
        return value.get("type") == "plot" and "path" in value
    return False


def get_plot_path(value: Any) -> str | None:
    """Extract plot path from a plot result.

    Args:
        value: Plot result (PlotResult or dict)

    Returns:
        Path to the plot image, or None
    """
    if isinstance(value, PlotResult):
        return value.path
    if isinstance(value, dict) and value.get("type") == "plot":
        return value.get("path")
    return None
