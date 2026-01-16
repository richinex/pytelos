"""Reasoning agent with multi-step processing and tool calling.

Implements the full agent.txt patterns:
- Multi-step processing loop
- get_next_step() and run_step() methods
- ReAct reasoning (thought/action/observation)
- Dynamic tool calling
"""

import asyncio
import json
import time
from typing import Any

from ..agent.data_structures import UsageSummary
from ..llm import LLMProvider
from ..llm.models import ChatMessage
from .data_structures import (
    NextStepDecision,
    Task,
    TaskResult,
    TaskStep,
    TaskStepResult,
    ToolCall,
    ToolCallResult,
)
from .tools import BaseTool


class ReasoningAgent:
    """A reasoning agent that can use tools and multi-step processing.

    Following agent.txt LLMAgent implementation.

    Hidden design decisions:
    - Processing loop structure
    - Tool calling mechanism
    - Rollout format and tracking
    - Step decision logic
    """

    def __init__(
        self,
        llm: LLMProvider,
        tools: list[BaseTool] | None = None,
        system_prompt: str | None = None,
        stream_callback: Any | None = None
    ):
        """Initialize the reasoning agent.

        Args:
            llm: LLM provider for generation
            tools: List of available tools
            system_prompt: Optional custom system prompt (or loaded from prompts/reasoning_agent.txt)
            stream_callback: Optional callback for streaming tokens (callable(str))
        """
        self._llm = llm
        self._tools = tools or []
        self._tools_registry = {t.name: t for t in self._tools}
        self._usage = UsageSummary()
        self._stream_callback = stream_callback
        self._debug_callback: Any | None = None

        # Build tools description
        tools_desc = "\n".join([
            f"- {t.name}: {t.description}"
            for t in self._tools
        ])

        # Load prompt from file if not provided
        if system_prompt is None:
            from ..prompts import get_system_prompt
            system_prompt = get_system_prompt()

        self._system_prompt = system_prompt.format(tools_description=tools_desc or "None")

    def add_tool(self, tool: BaseTool) -> "ReasoningAgent":
        """Add a tool to the agent.

        Args:
            tool: The tool to add

        Returns:
            Self for method chaining
        """
        if tool.name in self._tools_registry:
            raise ValueError(f"Tool {tool.name} already registered")

        self._tools.append(tool)
        self._tools_registry[tool.name] = tool

        return self

    @property
    def tools(self) -> list[BaseTool]:
        """Get the list of tools."""
        return list(self._tools_registry.values())

    def run(
        self,
        task: Task,
        max_steps: int | None = 10,
        conversation_context: str | None = None
    ) -> "ReasoningAgent.TaskHandler":
        """Execute a task with multi-step processing.

        Following agent.txt run() pattern (line 742-781).

        Args:
            task: The task to execute
            max_steps: Maximum number of steps to execute
            conversation_context: Previous conversation history for context

        Returns:
            TaskHandler that can be awaited
        """
        handler = self.TaskHandler(
            agent=self,
            task=task,
            max_steps=max_steps,
            conversation_context=conversation_context
        )

        async def _process_loop() -> None:
            """The processing loop for task execution."""
            await handler._execute_processing_loop()

        handler.background_task = asyncio.create_task(_process_loop())

        return handler

    def set_stream_callback(self, callback: Any) -> None:
        """Set the stream callback for real-time token display.

        Args:
            callback: Callable that receives string chunks
        """
        self._stream_callback = callback

    def set_debug_callback(self, callback: Any) -> None:
        """Set the debug callback for detailed execution logging.

        Args:
            callback: Callable(level: str, component: str, message: str)
                      level: 'debug', 'info', 'warning', 'error'
                      component: Source component name
                      message: Log message
        """
        self._debug_callback = callback

        # Propagate callback to all tools
        for tool in self._tools:
            tool.set_debug_callback(callback)

    def _debug(self, level: str, component: str, message: str) -> None:
        """Send a debug log message if callback is set.

        Args:
            level: Log level (debug, info, warning, error)
            component: Component name for the log
            message: Log message
        """
        if self._debug_callback:
            self._debug_callback(level, component, message)

    async def _chat_completion_streaming(
        self,
        messages: list[ChatMessage]
    ) -> tuple[str, dict | None]:
        """Execute chat completion with streaming, calling the callback for each chunk.

        Args:
            messages: List of chat messages

        Returns:
            Tuple of (complete response content, usage dict or None)
        """
        if self._stream_callback:
            self._stream_callback("__START__")

        full_content = []
        stream_response = await self._llm.chat_completion_stream(messages)
        async for chunk in stream_response:
            full_content.append(chunk)
            if self._stream_callback:
                self._stream_callback(chunk)

        if self._stream_callback:
            self._stream_callback("__END__")

        # Get usage from streaming response if available
        usage = None
        if hasattr(stream_response, "usage") and stream_response.usage:
            usage = stream_response.usage

        return "".join(full_content), usage

    async def close(self) -> None:
        """Close resources."""
        await self._llm.close()

    class TaskHandler(asyncio.Future):
        """Handler for processing tasks with multi-step execution.

        Following agent.txt TaskHandler (lines 440-725).
        """

        def __init__(
            self,
            agent: "ReasoningAgent",
            task: Task,
            max_steps: int | None = 10,
            conversation_context: str | None = None,
            *args: Any,
            **kwargs: Any
        ):
            """Initialize TaskHandler.

            Args:
                agent: The ReasoningAgent instance
                task: The task to execute
                max_steps: Maximum number of steps
                conversation_context: Previous conversation history for context
                *args: Additional positional arguments
                **kwargs: Additional keyword arguments
            """
            super().__init__(*args, **kwargs)
            self.agent = agent
            self.task = task
            self.max_steps = max_steps
            self.conversation_context = conversation_context
            self.execution_trace = ""
            self.step_counter = 0
            self._background_task: asyncio.Task | None = None
            self._start_time = time.time()

        async def _execute_processing_loop(self) -> None:
            """Execute the multi-step processing loop.

            Following agent.txt _process_loop (line 753-776).
            """
            step_result = None

            while not self.done():
                try:
                    # Check max steps
                    if self.max_steps and self.step_counter >= self.max_steps:
                        self.set_result(TaskResult(
                            task_id=self.task.id_,
                            content=f"Maximum steps ({self.max_steps}) reached. Unable to complete the task within the step limit.",
                            execution_trace=self.execution_trace,
                            metadata={
                                "steps": self.step_counter,
                                "processing_time_seconds": time.time() - self._start_time,
                                "total_input_tokens": self.agent._usage.total_input_tokens,
                                "total_output_tokens": self.agent._usage.total_output_tokens,
                                "total_llm_calls": self.agent._usage.total_calls,
                                "status": "max_steps_reached"
                            }
                        ))
                        return

                    # Planning: Get next step
                    next_step = await self.get_next_step(step_result)

                    if isinstance(next_step, TaskStep):
                        # Execute the step
                        step_result = await self.run_step(next_step)
                    elif isinstance(next_step, TaskResult):
                        # Task complete
                        self.set_result(next_step)
                    else:
                        raise ValueError(f"Unexpected next_step type: {type(next_step)}")

                except Exception as e:
                    self.set_exception(e)

        async def get_next_step(
            self,
            previous_step_result: TaskStepResult | None
        ) -> TaskStep | TaskResult:
            """Decide the next step based on previous result.

            Following agent.txt get_next_step (lines 485-523).

            Args:
                previous_step_result: Result of the previous step

            Returns:
                Either a new TaskStep or final TaskResult
            """
            # Helper to truncate text for debug logging
            def _truncate(text: str, max_len: int = 100) -> str:
                return text[:max_len] + "..." if len(text) > max_len else text

            # Build conversation history section if available
            history_section = ""
            if self.conversation_context:
                history_section = f"""
**Conversation History (you remember previous tasks):**
{self.conversation_context}

"""
                self.agent._debug("debug", "Memory", f"Injecting conversation context ({len(self.conversation_context)} chars)")

            # Build prompt for next step decision
            if not previous_step_result:
                # First decision: do we need any steps?
                prompt = f"""You are an assistant with access to an indexed database of code, PDFs, and documentation. You have memory and can recall previous conversations.
{history_section}
**Current Task:** {self.task.instruction}

**Available tools:**
- search_codebase: Search ALL indexed content (code files, PDFs, documents, documentation)
- read_file: Read specific files
- analyze_code: Analyze code structure

**CRITICAL: The database contains:**
- Code files (Python, JavaScript, etc.)
- PDF documents with embedded text
- Documentation and text files
- ALL content is semantically searchable using embeddings

**When to use tools:**
Use search_codebase if the task asks about:
- Code, functions, classes, or implementations
- Documentation, PDFs, or written content
- Concepts, topics, or information that might be in indexed files
- ANYTHING that could be stored in files (code OR documents)

Do NOT use tools if the task is:
- Pure computation (e.g., "calculate 2+2", "reverse string")
- General knowledge that doesn't require searching
- Questions about previous conversation (use your memory)

**Decision:**
- If you need to find information from indexed content (including PDFs), respond with:
  {{"kind": "next_step", "content": "Search for [relevant query]"}}

- If the task is pure computation, general knowledge, or answerable from conversation history, respond with:
  {{"kind": "final_result", "content": "your answer"}}

**Important:**
- If someone asks about "a PDF", "documentation", or any content, ALWAYS search first
- PDFs and documents ARE indexed and searchable - use search_codebase to find them
- If someone asks about previous questions or conversation, use your memory to answer
"""
            else:
                # Subsequent decision after a step
                prompt = f"""You are working on a multi-step task. Review your progress and decide whether to continue or provide the final answer.

**Task:** {self.task.instruction}

**Steps completed so far:** {self.step_counter}

**Latest result:**
{previous_step_result.content}

**Decision:**
- If you have enough information to answer the task, respond with:
  {{"kind": "final_result", "content": "your comprehensive final answer based on what you've learned"}}

- If you need more information or actions, respond with:
  {{"kind": "next_step", "content": "specific action needed"}}

**Important:**
- Aim to complete the task in as few steps as possible
- If you've gathered sufficient information, provide the final answer NOW
- Don't take unnecessary additional steps
"""

            try:
                # Get decision from LLM
                step_type = "initial" if not previous_step_result else "follow-up"
                self.agent._debug("debug", "Planner", f"Building {step_type} decision prompt ({len(prompt)} chars)")
                self.agent._debug("debug", "Planner", f"Prompt preview: {_truncate(prompt, 150)}")

                messages = [ChatMessage(role="user", content=prompt)]
                self.agent._debug("info", "LLM", f"Calling LLM for {step_type} decision...")
                response = await self.agent._llm.chat_completion(messages)
                self.agent._debug("info", "LLM", f"Response received ({len(response.content)} chars)")
                self.agent._debug("debug", "LLM", f"Response preview: {_truncate(response.content, 200)}")

                # Track usage
                if hasattr(response, "usage") and response.usage:
                    usage_dict = response.usage
                    if isinstance(usage_dict, dict):
                        self.agent._usage.add_usage(
                            model=response.model,
                            input_tokens=usage_dict.get("prompt_tokens", 0),
                            output_tokens=usage_dict.get("completion_tokens", 0)
                        )

                # Parse decision
                content = response.content.strip()

                # Try to extract JSON - look for first complete JSON object
                decision = None
                if "{" in content and "}" in content:
                    # Find all potential JSON objects
                    import re
                    # Look for JSON objects
                    json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
                    matches = re.findall(json_pattern, content)

                    for match in matches:
                        try:
                            decision_data = json.loads(match)
                            if "kind" in decision_data and "content" in decision_data:
                                decision = NextStepDecision(**decision_data)
                                break
                        except json.JSONDecodeError:
                            continue

                if not decision:
                    # Fallback: assume it's the final result
                    decision = NextStepDecision(
                        kind="final_result",
                        content=content
                    )
                    self.agent._debug("debug", "Planner", "No JSON found, treating as final_result")

                self.agent._debug("info", "Planner", f"Decision: {decision.kind}")
                if decision.kind == "final_result":
                    return TaskResult(
                        task_id=self.task.id_,
                        content=decision.content,
                        execution_trace=self.execution_trace,
                        metadata={
                            "steps": self.step_counter,
                            "processing_time_seconds": time.time() - self._start_time,
                            "total_input_tokens": self.agent._usage.total_input_tokens,
                            "total_output_tokens": self.agent._usage.total_output_tokens,
                            "total_llm_calls": self.agent._usage.total_calls
                        }
                    )
                else:
                    return TaskStep(
                        task_id=self.task.id_,
                        instruction=decision.content
                    )

            except Exception as e:
                raise RuntimeError(f"Failed to get next step: {str(e)}") from e

        async def run_step(self, step: TaskStep) -> TaskStepResult:
            """Execute a single step.

            Following agent.txt run_step (lines 547-688).

            Args:
                step: The step to execute

            Returns:
                TaskStepResult with the execution result
            """
            self.step_counter += 1

            # Helper to truncate for logging
            def _truncate(text: str, max_len: int = 100) -> str:
                return text[:max_len] + "..." if len(text) > max_len else text

            self.agent._debug("info", "Step", f"=== Step {self.step_counter} Start ===")
            self.agent._debug("debug", "Step", f"Instruction: {_truncate(step.instruction, 80)}")

            # Add step start to execution_trace
            self._add_to_execution_trace(f"\n=== Step {self.step_counter} Start ===\n")
            self._add_to_execution_trace(f"Instruction: {step.instruction}\n")

            # Build system message
            system_message = ChatMessage(
                role="system",
                content=self.agent._system_prompt
            )
            self.agent._debug("debug", "Step", f"System prompt: {len(self.agent._system_prompt)} chars")

            # Build user message
            user_message_content = f"""Current instruction: {step.instruction}

Please think through this step and either:
1. Use a tool if needed (respond with JSON tool call)
2. Provide reasoning/observations about what you've learned

Previous context:
{self.execution_trace if self.execution_trace else "This is the first step."}
"""

            user_message = ChatMessage(role="user", content=user_message_content)
            self.agent._debug("debug", "Step", f"User message: {len(user_message_content)} chars")

            # Get LLM response (streaming if callback available)
            messages = [system_message, user_message]

            if self.agent._stream_callback:
                # Use streaming for real-time token display
                self.agent._debug("info", "LLM", "Calling LLM (streaming)...")
                response_content, usage = await self.agent._chat_completion_streaming(messages)

                # Track usage for streaming
                if usage:
                    self.agent._usage.add_usage(
                        model=getattr(self.agent._llm, "model", "unknown"),
                        input_tokens=usage.get("prompt_tokens", 0),
                        output_tokens=usage.get("completion_tokens", 0)
                    )
            else:
                # Non-streaming completion
                self.agent._debug("info", "LLM", "Calling LLM...")
                response = await self.agent._llm.chat_completion(messages)
                response_content = response.content

                # Track usage for non-streaming
                if hasattr(response, "usage") and response.usage:
                    usage_dict = response.usage
                    if isinstance(usage_dict, dict):
                        self.agent._usage.add_usage(
                            model=response.model,
                            input_tokens=usage_dict.get("prompt_tokens", 0),
                            output_tokens=usage_dict.get("completion_tokens", 0)
                        )

            self.agent._debug("info", "LLM", f"Response received ({len(response_content)} chars)")
            self.agent._debug("debug", "LLM", f"Response preview: {_truncate(response_content, 150)}")
            self._add_to_execution_trace(f"\nAssistant: {response_content}\n")

            # Check if response contains tool call
            tool_call = self._extract_tool_call(response_content)

            if tool_call:
                # Execute tool
                self.agent._debug("info", "Tool", f"Tool call detected: {tool_call.tool_name}")
                self.agent._debug("debug", "Tool", f"Arguments: {json.dumps(tool_call.arguments)[:100]}")
                self._add_to_execution_trace(f"\nTool Call: {tool_call.tool_name}\n")
                self._add_to_execution_trace(f"Arguments: {json.dumps(tool_call.arguments, indent=2)}\n")

                self.agent._debug("info", "Tool", f"Executing {tool_call.tool_name}...")
                tool_result = await self._execute_tool(tool_call)
                self.agent._debug("info", "Tool", f"Tool result: {len(tool_result.content)} chars")
                self.agent._debug("debug", "Tool", f"Result preview: {_truncate(tool_result.content, 150)}")

                self._add_to_execution_trace(f"\nTool Result:\n{tool_result.content}\n")

                # Build step result
                final_content = f"Used tool '{tool_call.tool_name}'. Result:\n{tool_result.content}"
            else:
                # No tool call, use response as-is
                self.agent._debug("debug", "Step", "No tool call detected, using response directly")
                final_content = response_content

            self.agent._debug("info", "Step", f"=== Step {self.step_counter} End ===")
            self._add_to_execution_trace(f"\n=== Step {self.step_counter} End ===\n")

            return TaskStepResult(
                task_step_id=step.id_,
                content=final_content
            )

        def _extract_tool_call(self, content: str) -> ToolCall | None:
            """Extract tool call from LLM response.

            Args:
                content: LLM response content

            Returns:
                ToolCall if found, None otherwise
            """
            try:
                # Look for JSON in the response
                if "{" not in content or "}" not in content:
                    return None

                start = content.index("{")
                end = content.rindex("}") + 1
                json_str = content[start:end]

                data = json.loads(json_str)

                # Check if it's a tool call
                if "tool_name" in data and "arguments" in data:
                    return ToolCall(
                        tool_name=data["tool_name"],
                        arguments=data["arguments"]
                    )

                return None

            except (json.JSONDecodeError, ValueError):
                return None

        async def _execute_tool(self, tool_call: ToolCall) -> ToolCallResult:
            """Execute a tool call.

            Args:
                tool_call: The tool call to execute

            Returns:
                ToolCallResult with the execution result
            """
            tool = self.agent._tools_registry.get(tool_call.tool_name)

            if not tool:
                return ToolCallResult(
                    tool_call_id=tool_call.id_,
                    content=f"Error: Tool '{tool_call.tool_name}' not found",
                    error=True
                )

            try:
                return await tool.execute(tool_call)
            except Exception as e:
                return ToolCallResult(
                    tool_call_id=tool_call.id_,
                    content=f"Error executing tool: {str(e)}",
                    error=True
                )

        def _add_to_execution_trace(self, text: str) -> None:
            """Add text to the execution_trace.

            Args:
                text: Text to add
            """
            self.execution_trace += text

        @property
        def background_task(self) -> asyncio.Task:
            """Get the background task."""
            if not self._background_task:
                raise RuntimeError("No background task running")
            return self._background_task

        @background_task.setter
        def background_task(self, task: asyncio.Task) -> None:
            """Set the background task."""
            if self._background_task is not None:
                raise RuntimeError("Background task already set")
            self._background_task = task
