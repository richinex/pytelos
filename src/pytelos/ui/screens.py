"""Modal screens for the TUI.

This module hides the design decisions about:
- Confirmation dialog appearance (CSS, layout)
- Button styling and variants
- Keyboard shortcuts for dialogs
- How user confirmations are presented

To change how confirmations look, modify only this file.
"""

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Button, Static

from .input_handler import InputRequest


class InputRequestMessage(Message):
    """Message posted from background thread when user input is needed."""

    def __init__(self, request: InputRequest) -> None:
        super().__init__()
        self.request = request


class ConfirmationScreen(ModalScreen[str]):
    """Modal confirmation dialog for user input requests.

    Features a polished design with:
    - Semi-transparent backdrop
    - Rounded dialog with accent border
    - Clear title and prompt sections
    - Semantic button variants
    """

    CSS = """
    ConfirmationScreen {
        align: center middle;
        background: $background 70%;
    }

    #confirmation-dialog {
        width: 60;
        height: auto;
        max-height: 20;
        border: tall $accent;
        background: $surface;
        padding: 1 2;
    }

    #confirmation-title {
        width: 100%;
        height: auto;
        text-align: center;
        text-style: bold;
        color: $accent;
        padding: 0 0 1 0;
        border-bottom: solid $border;
        margin-bottom: 1;
    }

    #confirmation-prompt {
        width: 100%;
        height: auto;
        text-align: center;
        padding: 1 2;
        background: $panel;
        border: round $border;
        color: $foreground;
        margin-bottom: 1;
    }

    #confirmation-buttons {
        width: 100%;
        height: 3;
        align: center middle;
        margin-top: 1;
    }

    #confirmation-buttons Button {
        margin: 0 1;
        min-width: 10;
    }
    """

    BINDINGS = [
        Binding("y", "confirm_yes", "Yes", show=False),
        Binding("n", "confirm_no", "No", show=False),
        Binding("escape", "confirm_no", "Cancel", show=False),
    ]

    def __init__(self, prompt: str, options: list[str]) -> None:
        super().__init__()
        self._prompt = prompt
        self._options = options

    def compose(self) -> ComposeResult:
        with Vertical(id="confirmation-dialog"):
            yield Static("Confirmation Required", id="confirmation-title")
            yield Static(self._prompt, id="confirmation-prompt")
            with Horizontal(id="confirmation-buttons"):
                for option in self._options:
                    if option.lower() == "yes":
                        variant = "success"
                    elif option.lower() == "no":
                        variant = "error"
                    else:
                        variant = "primary"
                    yield Button(
                        option.capitalize(),
                        id=f"btn-{option}",
                        variant=variant
                    )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press - return the selected option."""
        button_id = event.button.id or ""
        if button_id.startswith("btn-"):
            response = button_id[4:]  # Remove 'btn-' prefix
            self.dismiss(response)

    def action_confirm_yes(self) -> None:
        """Keyboard shortcut for Yes."""
        if "yes" in self._options:
            self.dismiss("yes")

    def action_confirm_no(self) -> None:
        """Keyboard shortcut for No."""
        if "no" in self._options:
            self.dismiss("no")
