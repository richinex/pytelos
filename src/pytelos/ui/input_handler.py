"""Thread-safe input handler for Docker REPL user confirmations.

This module provides a thread-safe mechanism for the Docker proxy server
to request user input from the Textual TUI.

Design:
- Proxy server runs in a background thread
- Textual runs on the main asyncio event loop
- Uses post_message (thread-safe) to notify TUI of input requests
"""

import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from textual.app import App


@dataclass
class InputRequest:
    """A pending input request from the Docker container."""

    prompt: str
    options: list[str]
    timeout: int


class ThreadSafeInputHandler:
    """Thread-safe input handler for proxy server to TUI communication.

    Uses Textual's thread-safe post_message to notify the TUI of input requests,
    instead of polling with timers.

    Example:
        handler = ThreadSafeInputHandler()
        handler.set_app(app)  # Set the Textual app reference
        # Pass to proxy server
        proxy = LLMProxyServer(llm, input_handler=handler)

        # The handler will post messages to the app when input is needed
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._response_event = threading.Event()
        self._response: str | None = None
        self._app: Any | None = None  # Reference to Textual app

    def set_app(self, app: "App") -> None:
        """Set the Textual app reference for posting messages."""
        self._app = app

    def __call__(
        self, prompt: str, options: list[str], timeout: int
    ) -> str:
        """Handle input request from proxy server (called from proxy thread).

        Posts a message to the TUI and blocks until response or timeout.

        Args:
            prompt: Message to show user
            options: Available choices
            timeout: Max seconds to wait

        Returns:
            User's selection

        Raises:
            TimeoutError: If user doesn't respond in time
        """
        if not self._app:
            return "no"  # No app connected, default to no

        with self._lock:
            # Clear any previous state
            self._response_event.clear()
            self._response = None

        # Post message to TUI (thread-safe!)
        # Import here to avoid circular imports
        from .app import InputRequestMessage
        request = InputRequest(prompt, options, timeout)
        self._app.post_message(InputRequestMessage(request))

        # Wait for TUI to respond
        if not self._response_event.wait(timeout=timeout):
            raise TimeoutError(f"User input timed out after {timeout}s")

        # Get response
        with self._lock:
            response = self._response
            self._response = None

        return response or "no"

    def submit_response(self, response: str) -> None:
        """Submit user response (called from TUI thread).

        Args:
            response: User's selection (e.g., "yes" or "no")
        """
        with self._lock:
            self._response = response
            self._response_event.set()
