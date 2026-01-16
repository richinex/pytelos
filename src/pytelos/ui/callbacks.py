"""Callback interface for ReasoningAgent integration.

Hides the details of how the TUI receives updates from the agent.
Uses thread-safe methods to update UI from worker threads.
"""

import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from textual.app import App

    from .widgets import ExecutionLog, MetricsPanel, PlotPanel


class TUICallback:
    """Callback handler for ReasoningAgent execution updates.

    Uses call_from_thread for thread-safe UI updates from workers.
    """

    def __init__(
        self,
        log: "ExecutionLog",
        plot_panel: "PlotPanel | None" = None,
        metrics_panel: "MetricsPanel | None" = None,
        app: "App | None" = None
    ) -> None:
        self.log = log
        self.plot_panel = plot_panel
        self.metrics_panel = metrics_panel
        self.app = app
        self._step_count = 0
        self._code_block_count = 0
        self._start_time: float | None = None
        # Streaming buffer for real-time LLM response
        self._stream_buffer: list[str] = []
        self._stream_chars_since_update = 0

    def _call_thread_safe(self, func: Any, *args: Any, **kwargs: Any) -> None:
        """Call a function in a thread-safe manner for UI updates."""
        import threading
        if self.app is not None and self.app._thread_id != threading.get_ident():
            self.app.call_from_thread(func, *args, **kwargs)
        else:
            func(*args, **kwargs)

    def _write_markup(self, content: str) -> None:
        """Write content with Rich markup enabled."""
        self._call_thread_safe(self.log.write, content, markup=True)

    def _write_plain(self, content: str) -> None:
        """Write plain text without markup parsing."""
        self._call_thread_safe(self.log.write, content, markup=False)

    def _update_metrics(self) -> None:
        """Update the metrics panel with current execution progress.

        Called during execution to show real-time step and code block counts.
        """
        if self.metrics_panel is None:
            return

        elapsed = 0.0
        if self._start_time is not None:
            elapsed = time.time() - self._start_time

        self._call_thread_safe(
            self.metrics_panel.update_metrics,
            iterations=self._step_count,
            code_blocks=self._code_block_count,
            time=elapsed,
        )

    def start_execution(self) -> None:
        """Called when execution starts to initialize timing."""
        self._start_time = time.time()
        self._step_count = 0
        self._code_block_count = 0

    def print_step_progress(self, step: int, max_steps: int) -> None:
        """Print step progress and update metrics panel."""
        self._step_count = step
        # Update metrics panel in real-time
        self._update_metrics()
        self._write_markup(
            f"[bold cyan]{'='*15}[/] [bold yellow]Step {step}/{max_steps}[/] [bold cyan]{'='*15}[/]"
        )

    def print_tool_call(self, tool_name: str, arguments: dict) -> None:
        """Print tool call information."""
        self._write_markup(f"[bold green]Tool:[/] {tool_name}")
        import json
        args_str = json.dumps(arguments, indent=2)
        self._write_plain(args_str)

    def print_tool_result(self, result: str) -> None:
        """Print tool result."""
        self._write_markup("[bold cyan]Result:[/]")
        if len(result) > 2000:
            result = result[:2000] + "\n... (output truncated)"
        self._write_plain(result)

    def print_final_answer(self, answer: str) -> None:
        """Print final answer."""
        self._write_markup("[bold magenta]Final Answer:[/]")
        self._write_plain(str(answer))

    def set_error(self, error_message: str) -> None:
        """Print error message."""
        self._write_markup("[bold red]Error:[/]")
        self._write_plain(error_message)

    def handle_stream_chunk(self, chunk: str) -> None:
        """Handle a streaming chunk from the LLM.

        Called for each token during streaming. Special signals:
        - __START__: Beginning of streaming
        - __END__: End of streaming
        - Other strings: Actual token content

        Uses buffering to update display every ~50 characters for smooth streaming.
        """
        if chunk == "__START__":
            # Start of streaming - reset buffer and show streaming indicator
            self._stream_buffer = []
            self._stream_chars_since_update = 0
            self._write_markup("[cyan]LLM Response (streaming):[/]")
            return

        if chunk == "__END__":
            # End of streaming - flush any remaining buffer
            if self._stream_buffer:
                remaining = "".join(self._stream_buffer)
                self._write_plain(remaining)
                self._stream_buffer = []
            self._write_plain("\n")
            return

        # Regular chunk - add to buffer
        self._stream_buffer.append(chunk)
        self._stream_chars_since_update += len(chunk)

        # Update display every ~50 characters for smooth streaming
        if self._stream_chars_since_update >= 50:
            buffered = "".join(self._stream_buffer)
            self._write_plain(buffered)
            self._stream_buffer = []
            self._stream_chars_since_update = 0
