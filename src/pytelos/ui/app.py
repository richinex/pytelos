"""Main Textual TUI application.

Orchestrates the UI components and handles user interaction with ReasoningAgent.
"""

import asyncio
import contextlib
from pathlib import Path
from typing import Any

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.widgets import Footer, Header

from .callbacks import TUICallback
from .formatting import format_final_answer
from .styles import APP_CSS
from .themes import CATPPUCCIN_MOCHA
from .widgets import (
    ChatHistoryWidget,
    ChatInputBar,
    DebugPanel,
    ExecutionLog,
    LogLevel,
    MetricsPanel,
    PlotPanel,
)


class ReasoningTextualApp(App):
    """Textual TUI for ReasoningAgent Chat."""

    CSS = APP_CSS
    TITLE = "Pytelos"

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit"),
        Binding("ctrl+l", "clear_log", "Clear Log"),
        Binding("ctrl+k", "clear_chat", "Clear Chat"),
        Binding("ctrl+o", "load_from_file", "Load File"),
        Binding("ctrl+y", "copy_metrics", "Copy Metrics"),
        Binding("ctrl+r", "copy_last_response", "Copy Response"),
        Binding("escape", "cancel_execution", "Cancel"),
        Binding("ctrl+b", "toggle_maximize_chat", "Max Chat"),
        Binding("ctrl+e", "toggle_maximize_log", "Max Log"),
        Binding("ctrl+d", "toggle_debug", "Debug"),
    ]

    def __init__(
        self,
        llm: Any,
        storage: Any,
        embedder: Any,
        max_steps: int = 10,
        log_level: str | None = None,
        memory_backend: str = "memory",
        memory_path: str | None = None,
        durable: bool = False,
    ) -> None:
        super().__init__()
        self._llm = llm
        self._storage = storage
        self._embedder = embedder
        self._max_steps = max_steps
        self._log_level = log_level
        self._memory_backend = memory_backend
        self._memory_path = memory_path
        self._durable = durable  # Use PyergonReasoningAgent for durable execution
        self._current_worker = None
        self._model_name = self._get_model_name()
        self._persistent_agent: Any | None = None
        self._conversation_memory: Any | None = None

    def _get_model_name(self) -> str:
        if hasattr(self._llm, "_model"):
            return self._llm._model
        elif hasattr(self._llm, "model"):
            return self._llm.model
        return "unknown"

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)

        # Chat history (left panel)
        yield ChatHistoryWidget(id="chat-history")

        # Right panel: execution log + plot panel + debug panel
        with Vertical(id="right-panel"):
            yield ExecutionLog(id="execution-log")
            yield PlotPanel(id="plot-panel")
            yield DebugPanel(id="debug-panel")

        # Bottom bar with metrics and input
        with Vertical(id="bottom-bar"):
            yield MetricsPanel(id="metrics")
            yield ChatInputBar(id="chat-input-bar")

        yield Footer()

    def on_mount(self) -> None:
        """Called when app is mounted."""
        # Register and set Catppuccin theme
        self.register_theme(CATPPUCCIN_MOCHA)
        self.theme = "catppuccin-mocha"

        # Configure log panel if --log-level was passed
        if self._log_level is not None:
            log_panel = self.query_one("#debug-panel", DebugPanel)
            log_panel.log_level = LogLevel.from_string(self._log_level)
            log_panel.show()
            log_panel.info("TUI", f"Log panel enabled with level: {self._log_level.upper()}")

        # Create conversation memory backend
        from ..memory import create_conversation_memory
        memory_config = {}
        if self._memory_path:
            memory_config["path"] = self._memory_path
        self._conversation_memory = create_conversation_memory(
            self._memory_backend, **memory_config
        )

        # Set subtitle with model, mode, and memory info
        exec_mode = "durable" if self._durable else "standard"
        self.sub_title = f"{self._model_name} | {exec_mode} | max {self._max_steps} steps | {self._memory_backend}"

        chat = self.query_one("#chat-history", ChatHistoryWidget)
        memory_description = "persistent (SQLite)" if self._memory_backend == "sqlite" else "session-only"

        welcome_lines = [
            "Welcome to Pytelos!",
            "",
            "I can help you search and analyze your indexed codebase.",
            f"Memory: {memory_description} - I remember previous tasks.",
            "",
            "Click on any message to copy it to clipboard.",
            "",
            "Ask me anything!",
        ]

        chat.add_message("assistant", "\n".join(welcome_lines))
        self.query_one("#chat-input-bar", ChatInputBar).focus_input()

    def on_unmount(self) -> None:
        """Clean up resources when app exits."""
        if self._persistent_agent is not None:
            with contextlib.suppress(BaseException):
                asyncio.create_task(self._persistent_agent.close())
            self._persistent_agent = None

        # Disconnect memory backend
        if self._conversation_memory is not None:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self._conversation_memory.disconnect())
                else:
                    loop.run_until_complete(self._conversation_memory.disconnect())
            except BaseException:
                pass
            self._conversation_memory = None

    def on_chat_input_bar_submitted(self, event: ChatInputBar.Submitted) -> None:
        """Handle user input submission."""
        user_input = event.value
        if not user_input:
            return

        # Add to chat history
        chat = self.query_one("#chat-history", ChatHistoryWidget)
        chat.add_message("user", user_input)

        # Clear execution log and show running state
        log = self.query_one("#execution-log", ExecutionLog)
        log.clear()
        log.start_execution()

        # Hide plot panel
        plot_panel = self.query_one("#plot-panel", PlotPanel)
        plot_panel.clear_image()

        # Run agent in background
        self._run_agent(user_input)

    @work(exclusive=True)
    async def _run_agent(self, user_input: str) -> None:
        """Run the ReasoningAgent as a background async worker.

        Uses PyergonReasoningAgent if durable mode is enabled.
        """
        from ..agent import Task
        from ..reasoning_agent import (
            AnalyzeCodeTool,
            ReadFileTool,
            ReasoningAgent,
            SearchCodebaseTool,
        )

        log = self.query_one("#execution-log", ExecutionLog)
        chat = self.query_one("#chat-history", ChatHistoryWidget)
        metrics = self.query_one("#metrics", MetricsPanel)
        log_panel = self.query_one("#debug-panel", DebugPanel)

        def _log_info(msg: str) -> None:
            log_panel.info("TUI", msg)

        def _log_error(msg: str) -> None:
            log_panel.error("TUI", msg)

        _log_info(f"Starting: '{user_input[:50]}...'")

        try:
            # Connect memory backend and get conversation context
            conversation_context = None
            if self._conversation_memory is not None:
                await self._conversation_memory.connect()
                _log_info("Memory backend connected")

                # Get conversation history for context injection
                state = await self._conversation_memory.get_state()
                if state and state.history:
                    conversation_context = state.to_context_string()
                    _log_info(f"Loaded {len(state.history)} previous task(s) from memory")

            # Create agent if not exists
            if self._persistent_agent is None:
                search_tool = SearchCodebaseTool(self._storage, self._embedder)
                read_tool = ReadFileTool()
                analyze_tool = AnalyzeCodeTool()

                if self._durable:
                    from ..reasoning_agent import PyergonReasoningAgent
                    self._persistent_agent = PyergonReasoningAgent(
                        llm=self._llm,
                        tools=[search_tool, read_tool, analyze_tool]
                    )
                    _log_info("PyergonReasoningAgent created (durable mode)")
                else:
                    self._persistent_agent = ReasoningAgent(
                        llm=self._llm,
                        tools=[search_tool, read_tool, analyze_tool]
                    )
                    _log_info("ReasoningAgent created")

            agent = self._persistent_agent

            # Set up TUI callback with metrics panel for real-time updates
            plot_panel = self.query_one("#plot-panel", PlotPanel)
            metrics_panel = self.query_one("#metrics", MetricsPanel)
            callback = TUICallback(log, plot_panel, metrics_panel, app=self)
            callback.start_execution()

            # Set up streaming callback for real-time LLM token display
            if hasattr(agent, 'set_stream_callback'):
                agent.set_stream_callback(callback.handle_stream_chunk)

            # Set up debug callback for detailed logging
            if hasattr(agent, 'set_debug_callback'):
                def debug_callback(level: str, component: str, message: str) -> None:
                    """Route debug messages to the log panel."""
                    if level == "debug":
                        log_panel.debug(component, message)
                    elif level == "info":
                        log_panel.info(component, message)
                    elif level == "warning":
                        log_panel.warning(component, message)
                    elif level == "error":
                        log_panel.error(component, message)
                agent.set_debug_callback(debug_callback)

            # Execute task with conversation context
            task = Task(instruction=user_input)
            handler = agent.run(
                task,
                max_steps=self._max_steps,
                conversation_context=conversation_context
            )

            # Monitor progress
            last_step_count = 0
            while not handler.done():
                await asyncio.sleep(0.3)
                current_step = handler.step_counter
                if current_step > last_step_count:
                    callback.print_step_progress(current_step, self._max_steps)
                    last_step_count = current_step

            result = await handler

            # Update metrics
            metadata = result.metadata or {}
            metrics.update_metrics(
                iterations=metadata.get("steps", 0),
                code_blocks=0,
                time=metadata.get("processing_time_seconds", 0.0),
                input_tokens=metadata.get("total_input_tokens", 0),
                output_tokens=metadata.get("total_output_tokens", 0),
            )

            # Add final response to chat
            final_answer = result.content if result.content else "Task completed."
            formatted_answer = format_final_answer(final_answer)
            chat.add_message("assistant", formatted_answer)

            # Save to conversation memory
            if self._conversation_memory is not None:
                try:
                    await self._conversation_memory.add_task_record(
                        task=user_input,
                        answer=final_answer,
                    )
                    _log_info("Task saved to memory")
                except Exception as e:
                    _log_error(f"Failed to save to memory: {e}")

            log.write("\n[bold green]Execution complete.[/bold green]")
            log.end_execution(success=True)
            self.notify("Task completed", severity="information", timeout=3)

        except asyncio.CancelledError:
            log.write("\n[yellow]Execution cancelled.[/yellow]")
            log.end_execution(success=False)
            self.notify("Cancelled", severity="warning", timeout=2)
        except Exception as e:
            _log_error(f"Exception: {e}")
            log.write(f"\n[bold red]Error: {e}[/bold red]")
            log.end_execution(success=False)
            chat.add_message("assistant", f"Error: {str(e)}")
            self.notify(f"Error: {str(e)[:50]}", severity="error", timeout=5)

    def action_clear_log(self) -> None:
        """Clear the execution log."""
        log = self.query_one("#execution-log", ExecutionLog)
        log.clear()
        self.notify("Log cleared", timeout=2)

    def action_clear_chat(self) -> None:
        """Clear the chat history."""
        chat = self.query_one("#chat-history", ChatHistoryWidget)
        chat.clear_history()
        self.notify("Chat cleared", timeout=2)

    def action_toggle_debug(self) -> None:
        """Toggle the log panel visibility."""
        log_panel = self.query_one("#debug-panel", DebugPanel)
        is_visible = log_panel.toggle()
        self.notify(f"Log panel {'shown' if is_visible else 'hidden'}", timeout=2)

    def action_toggle_maximize_chat(self) -> None:
        """Toggle maximize for chat panel."""
        chat = self.query_one("#chat-history", ChatHistoryWidget)
        log = self.query_one("#execution-log", ExecutionLog)
        if chat.has_class("-maximized"):
            chat.remove_class("-maximized")
            log.display = True
        else:
            chat.add_class("-maximized")
            log.display = False

    def action_toggle_maximize_log(self) -> None:
        """Toggle maximize for execution log panel."""
        chat = self.query_one("#chat-history", ChatHistoryWidget)
        log = self.query_one("#execution-log", ExecutionLog)
        if log.has_class("-maximized"):
            log.remove_class("-maximized")
            chat.display = True
        else:
            log.add_class("-maximized")
            chat.display = False

    def action_cancel_execution(self) -> None:
        """Cancel current execution."""
        if self._current_worker and self._current_worker.is_running:
            self._current_worker.cancel()
            log = self.query_one("#execution-log", ExecutionLog)
            log.write("\n[yellow]Execution cancelled.[/yellow]")

    def action_load_from_file(self) -> None:
        """Load prompt from a file."""
        from textual.widgets import TextArea

        prompt_file = Path("prompt.txt")
        if prompt_file.exists():
            try:
                content = prompt_file.read_text().strip()
                if content:
                    text_area = self.query_one("#chat-input", TextArea)
                    text_area.text = content
                    self.notify("File loaded")
            except Exception:
                pass

    def action_copy_metrics(self) -> None:
        """Copy metrics to clipboard."""
        metrics = self.query_one("#metrics", MetricsPanel)
        text = metrics.get_plain_text()
        self.copy_to_clipboard(text)
        self.notify("Metrics copied")

    def action_copy_last_response(self) -> None:
        """Copy last assistant response to clipboard."""
        chat = self.query_one("#chat-history", ChatHistoryWidget)
        response = chat.get_last_response()
        if response:
            self.copy_to_clipboard(response)
            self.notify("Response copied")
        else:
            self.notify("No response to copy", severity="warning")


async def run_textual_tui(
    llm: Any,
    storage: Any,
    embedder: Any,
    max_steps: int = 10,
    log_level: str | None = None,
    memory_backend: str = "memory",
    memory_path: str | None = None,
    durable: bool = False,
) -> None:
    """Run the Textual TUI.

    Args:
        llm: LLM provider instance
        storage: Storage backend for search
        embedder: Embedding provider for search
        max_steps: Maximum reasoning steps
        log_level: Log level for panel (debug/info/warning/error), None to hide
        memory_backend: Conversation memory backend ("memory" or "sqlite")
        memory_path: Path for SQLite memory database
        durable: Use PyergonReasoningAgent for durable execution
    """
    app = ReasoningTextualApp(
        llm=llm,
        storage=storage,
        embedder=embedder,
        max_steps=max_steps,
        log_level=log_level,
        memory_backend=memory_backend,
        memory_path=memory_path,
        durable=durable,
    )
    try:
        await app.run_async()
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        if app._persistent_agent is not None:
            with contextlib.suppress(BaseException):
                await app._persistent_agent.close()
            app._persistent_agent = None
