"""Pyergon flow definitions for durable multi-step reasoning.

This module defines the Pyergon flows for distributed reasoning agent execution,
following the same pattern as pyergon_indexing.py.
"""

import json
from dataclasses import dataclass
from typing import Any

from pyergon import flow, flow_type, step
from pyergon.core import RetryPolicy

from ..llm import create_llm_provider
from ..llm.models import ChatMessage
from .connection_pool import ConnectionPool
from .data_structures import ToolCall


class ReasoningError(Exception):
    """Base class for reasoning errors."""

    def is_retryable(self) -> bool:
        """Override in subclasses to control retry behavior."""
        return False


class LLMRateLimitError(ReasoningError):
    """LLM API rate limit exceeded (retryable)."""

    def __init__(self, message: str):
        super().__init__(f"Rate limit exceeded: {message}")

    def is_retryable(self) -> bool:
        return True


class LLMNetworkError(ReasoningError):
    """Network or connection error (retryable)."""

    def __init__(self, message: str):
        super().__init__(f"Network error: {message}")

    def is_retryable(self) -> bool:
        return True


class InvalidPromptError(ReasoningError):
    """Invalid prompt or configuration (non-retryable)."""

    def __init__(self, message: str):
        super().__init__(f"Invalid prompt: {message}")

    def is_retryable(self) -> bool:
        return False


class ToolExecutionError(ReasoningError):
    """Tool execution failed (non-retryable unless specific tool error)."""

    def __init__(self, message: str, tool_name: str = None):
        msg = f"Tool execution error: {message}"
        if tool_name:
            msg += f" (tool: {tool_name})"
        super().__init__(msg)
        self.tool_name = tool_name

    def is_retryable(self) -> bool:
        return False


@dataclass
@flow_type(invokable=str)
class ReasoningStepFlow:
    """Durable flow for executing a single reasoning step.

    This flow represents one iteration of the reasoning loop:
    1. Get LLM response for the current instruction
    2. Extract tool call if present
    3. Execute tool (with retry on transient failures)
    4. Return step result

    Each step can be retried independently on failure.
    """
    instruction: str
    execution_trace: str
    step_number: int

    # Provider configurations (injected by scheduler)
    llm_config: dict[str, Any]
    tools_config: dict[str, Any]  # Tool registry serialized
    system_prompt: str

    @step(retry_policy=RetryPolicy.STANDARD)
    async def get_llm_response(self) -> dict:
        """Get LLM response for current instruction.

        Uses RetryPolicy.STANDARD to handle transient LLM API failures.

        Returns:
            {"content": str, "has_tool_call": bool, "tool_call": dict | None}

        Raises:
            LLMRateLimitError: When API rate limit is exceeded (retryable)
            LLMNetworkError: On network/connection issues (retryable)
            InvalidPromptError: On invalid prompt (non-retryable)
        """
        # Recreate LLM from config
        llm = create_llm_provider(
            self.llm_config["provider"],
            **self.llm_config.get("kwargs", {})
        )

        # Build system message
        system_message = ChatMessage(
            role="system",
            content=self.system_prompt
        )

        # Build user message
        user_message_content = f"""Current instruction: {self.instruction}

Please think through this step and either:
1. Use a tool if needed (respond with JSON tool call)
2. Provide reasoning/observations about what you've learned

Previous context:
{self.execution_trace if self.execution_trace else "This is the first step."}
"""

        user_message = ChatMessage(role="user", content=user_message_content)

        try:
            # Get LLM response
            messages = [system_message, user_message]
            response = await llm.chat_completion(messages)

            # Check for tool call
            tool_call = self._extract_tool_call(response.content)

            return {
                "content": response.content,
                "has_tool_call": tool_call is not None,
                "tool_call": {
                    "tool_name": tool_call.tool_name,
                    "arguments": tool_call.arguments
                } if tool_call else None
            }

        except Exception as e:
            error_msg = str(e).lower()

            # Classify error
            if "rate" in error_msg and "limit" in error_msg or "429" in str(e):
                raise LLMRateLimitError(str(e))
            elif "network" in error_msg or "connection" in error_msg or "timeout" in error_msg:
                raise LLMNetworkError(str(e))
            elif "invalid" in error_msg or "400" in str(e):
                raise InvalidPromptError(str(e))
            else:
                raise

        finally:
            await llm.close()

    def _extract_tool_call(self, content: str) -> ToolCall | None:
        """Extract tool call from LLM response."""
        try:
            if "{" not in content or "}" not in content:
                return None

            start = content.index("{")
            end = content.rindex("}") + 1
            json_str = content[start:end]

            data = json.loads(json_str)

            if "tool_name" in data and "arguments" in data:
                return ToolCall(
                    tool_name=data["tool_name"],
                    arguments=data["arguments"]
                )

            return None

        except (json.JSONDecodeError, ValueError):
            return None

    @step(retry_policy=RetryPolicy.STANDARD)
    async def execute_tool_call(self, tool_call_dict: dict) -> dict:
        """Execute a tool call.

        Uses RetryPolicy.STANDARD for transient tool failures.

        Args:
            tool_call_dict: Serialized tool call

        Returns:
            {"content": str, "error": bool}

        Raises:
            ToolExecutionError: On tool execution failure
        """
        tool_name = tool_call_dict["tool_name"]

        # Import tools dynamically to avoid circular imports
        from .tools import AnalyzeCodeTool, ReadFileTool, SearchCodebaseTool

        # Recreate tool from config
        tool_classes = {
            "search_codebase": SearchCodebaseTool,
            "read_file": ReadFileTool,
            "analyze_code": AnalyzeCodeTool
        }

        if tool_name not in tool_classes:
            raise ToolExecutionError(f"Unknown tool: {tool_name}", tool_name=tool_name)

        # Instantiate tool with config
        tool_class = tool_classes[tool_name]
        tool_config = self.tools_config.get(tool_name, {})

        # Handle tool-specific instantiation
        if tool_name == "search_codebase":
            # Recreate storage and embedder from config
            from ..embedding import create_embedding_provider
            from ..storage import create_storage_backend

            search_config = tool_config.get("search", {})
            storage_config = search_config.get("storage", {})
            embedder_config = search_config.get("embedder", {})

            storage = create_storage_backend(
                storage_config.get("backend", "postgres"),
                **storage_config.get("kwargs", {})
            )
            await storage.connect()

            embedder = create_embedding_provider(
                embedder_config.get("provider", "openai"),
                **embedder_config.get("kwargs", {})
            )

            tool = SearchCodebaseTool(storage, embedder)
        else:
            # Other tools don't need special config
            tool = tool_class()

        # Execute tool
        tool_call = ToolCall(
            tool_name=tool_name,
            arguments=tool_call_dict["arguments"]
        )

        try:
            result = await tool.execute(tool_call)
            return {
                "content": result.content,
                "error": result.error
            }
        except Exception as e:
            # Check if it's a retryable error
            error_msg = str(e).lower()
            if "rate" in error_msg or "network" in error_msg or "timeout" in error_msg:
                raise LLMNetworkError(str(e))  # Retryable
            else:
                raise ToolExecutionError(str(e), tool_name=tool_name)  # Non-retryable
        finally:
            # Cleanup tool resources
            if tool_name == "search_codebase":
                await storage.disconnect()
                await embedder.close()

    @flow(retry_policy=RetryPolicy.STANDARD)
    async def execute(self) -> dict:
        """Main flow: get LLM response -> execute tool if needed -> return result.

        Returns:
            {
                "step_number": int,
                "content": str,
                "used_tool": str | None,
                "tool_result": str | None
            }
        """
        # Get LLM response
        llm_response = await self.get_llm_response()

        if llm_response["has_tool_call"]:
            # Execute tool
            tool_result = await self.execute_tool_call(llm_response["tool_call"])

            return {
                "step_number": self.step_number,
                "content": llm_response["content"],
                "used_tool": llm_response["tool_call"]["tool_name"],
                "tool_result": tool_result["content"],
                "error": tool_result["error"]
            }
        else:
            # No tool, just reasoning
            return {
                "step_number": self.step_number,
                "content": llm_response["content"],
                "used_tool": None,
                "tool_result": None,
                "error": False
            }


@dataclass
@flow_type(invokable=str)
class DecisionFlow:
    """Flow for deciding next action."""
    task_instruction: str
    execution_trace: str
    step_count: int
    llm_config: dict[str, Any]
    system_prompt: str

    @flow
    async def decide(self) -> dict:
        """Decide if we need another step or have final answer."""
        # Get LLM from connection pool (reused across invocations)
        llm = await ConnectionPool.get_llm(self.llm_config)

        prompt = f"""You are an assistant with access to an indexed database of code, PDFs, and documentation. Review your progress and decide your next action.

**Task:** {self.task_instruction}

**Steps completed so far:** {self.step_count}

**Latest result:**
{self.execution_trace if self.execution_trace else "This is the first step. You have access to tools: search_codebase, read_file, analyze_code."}

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

**Decision:**
- If you need to find information from indexed content (including PDFs), respond with:
  {{"kind": "next_step", "content": "Search for [relevant query]"}}

- If you have enough information to answer the task, respond with:
  {{"kind": "final_result", "content": "your comprehensive final answer"}}

- If you need more information or actions:
  {{"kind": "next_step", "content": "specific action needed"}}

**Important:**
- If someone asks about "a PDF", "documentation", or any content, ALWAYS search first
- PDFs and documents ARE indexed and searchable - use search_codebase to find them
- Use the full information from search results when you do search
- Aim to complete the task in as few steps as possible
"""

        try:
            messages = [ChatMessage(role="user", content=prompt)]
            response = await llm.chat_completion(messages)

            # Extract usage
            usage = {}
            if hasattr(response, "usage") and response.usage:
                usage_dict = response.usage
                if isinstance(usage_dict, dict):
                    usage = {
                        "input_tokens": usage_dict.get("prompt_tokens", 0),
                        "output_tokens": usage_dict.get("completion_tokens", 0)
                    }

            # Parse decision
            content = response.content.strip()

            # Try to extract JSON
            decision = None
            if "{" in content and "}" in content:
                import re
                json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
                matches = re.findall(json_pattern, content)

                for match in matches:
                    try:
                        decision_data = json.loads(match)
                        if "kind" in decision_data and "content" in decision_data:
                            decision = decision_data
                            break
                    except json.JSONDecodeError:
                        continue

            if not decision:
                # Fallback: assume it's the final result
                decision = {
                    "kind": "final_result",
                    "content": content
                }

            # Add usage to decision
            decision["usage"] = usage

            return decision

        except Exception:
            # Don't catch - let Pyergon handle retries
            raise


@dataclass
@flow_type(invokable=str)
class ExecutionFlow:
    """Flow for executing a reasoning step."""
    instruction: str
    execution_trace: str
    llm_config: dict[str, Any]
    tools_config: dict[str, Any]
    system_prompt: str

    @flow
    async def execute(self) -> dict:
        """Execute a single reasoning step with LLM + optional tool use."""
        # Get LLM from connection pool (reused across invocations)
        llm = await ConnectionPool.get_llm(self.llm_config)

        # Build messages
        system_message = ChatMessage(role="system", content=self.system_prompt)
        user_message_content = f"""Current instruction: {self.instruction}

Please think through this step and either:
1. Use a tool if needed (respond with JSON tool call)
2. Provide reasoning/observations about what you've learned

Previous context:
{self.execution_trace if self.execution_trace else "This is the first step."}
"""
        user_message = ChatMessage(role="user", content=user_message_content)

        try:
            # Get LLM response
            messages = [system_message, user_message]
            response = await llm.chat_completion(messages)

            # Extract usage
            usage = {}
            if hasattr(response, "usage") and response.usage:
                usage_dict = response.usage
                if isinstance(usage_dict, dict):
                    usage = {
                        "input_tokens": usage_dict.get("prompt_tokens", 0),
                        "output_tokens": usage_dict.get("completion_tokens", 0)
                    }

            # Check for tool call
            tool_call = self._extract_tool_call(response.content)

            if tool_call:
                # Execute tool
                tool_result = await self._execute_tool(tool_call)
                return {
                    "content": response.content,
                    "used_tool": tool_call.tool_name,
                    "tool_result": tool_result["content"],
                    "error": tool_result["error"],
                    "usage": usage
                }
            else:
                return {
                    "content": response.content,
                    "used_tool": None,
                    "tool_result": None,
                    "error": False,
                    "usage": usage
                }
        except Exception:
            # Don't catch - let Pyergon handle retries
            raise

    async def _execute_tool(self, tool_call: ToolCall) -> dict:
        """Execute a tool call (helper method)."""
        tool_name = tool_call.tool_name

        from .tools import AnalyzeCodeTool, ReadFileTool, SearchCodebaseTool

        tool_classes = {
            "search_codebase": SearchCodebaseTool,
            "read_file": ReadFileTool,
            "analyze_code": AnalyzeCodeTool
        }

        if tool_name not in tool_classes:
            return {"content": f"Unknown tool: {tool_name}", "error": True}

        tool_config = self.tools_config.get(tool_name, {})

        # Handle tool-specific instantiation
        if tool_name == "search_codebase":
            search_config = tool_config.get("search", {})
            storage_config = search_config.get("storage", {})
            embedder_config = search_config.get("embedder", {})

            # Get from connection pool (reused across invocations)
            storage = await ConnectionPool.get_storage(storage_config)
            embedder = await ConnectionPool.get_embedder(embedder_config)

            tool = SearchCodebaseTool(storage, embedder)
        else:
            tool_class = tool_classes[tool_name]
            tool = tool_class()

        result = await tool.execute(tool_call)
        return {"content": result.content, "error": result.error}

    def _extract_tool_call(self, content: str) -> ToolCall | None:
        """Extract tool call from LLM response."""
        try:
            if "{" not in content or "}" not in content:
                return None

            start = content.index("{")
            end = content.rindex("}") + 1
            json_str = content[start:end]

            data = json.loads(json_str)

            if "tool_name" in data and "arguments" in data:
                return ToolCall(
                    tool_name=data["tool_name"],
                    arguments=data["arguments"]
                )

            return None

        except (json.JSONDecodeError, ValueError):
            return None


@dataclass
@flow_type(invokable=str)
class StepFlow:
    """Combined decision and execution flow.

    Combines DecisionFlow and ExecutionFlow into single LLM call to:
    - Reduce token usage by ~50%
    - Eliminate double LLM latency
    - Match sequential agent efficiency
    """
    task_instruction: str
    execution_trace: str
    step_count: int
    max_steps: int

    llm_config: dict[str, Any]
    tools_config: dict[str, Any]
    system_prompt: str

    @flow(retry_policy=RetryPolicy.STANDARD)
    async def execute(self) -> dict:
        """Execute combined decision + action in single LLM call.

        Returns:
            {
                "kind": "final_result" | "next_step",
                "content": str,
                "used_tool": str | None,
                "tool_result": str | None,
                "usage": {"input_tokens": int, "output_tokens": int}
            }
        """
        llm = await ConnectionPool.get_llm(self.llm_config)

        # Build unified prompt that decides AND executes
        if self.step_count == 0:
            prompt = f"""You are an assistant with access to an indexed database of code, PDFs, and documentation.

**Task:** {self.task_instruction}

**First: Classify the task type**

1. **Code/Documentation Questions** - Requires searching indexed content
   Examples: "What does function X do?", "Where is the cleanup task?", "How does authentication work?"
   Action: Use search_codebase tool

2. **Pure Computation/Math** - Can be answered directly without searching
   Examples: "Calculate factorial of 10", "What is 2+2?", "Reverse a string"
   Action: Provide final answer immediately

3. **General Knowledge** - Does not require codebase
   Examples: "What is a binary tree?", "Explain REST API"
   Action: Provide final answer immediately

**Available tools:**
- search_codebase: Search indexed code files, PDFs, and documentation
- read_file: Read specific files
- analyze_code: Analyze code structure

**CRITICAL DECISION LOGIC:**

Does the task ask about:
- Specific code/files in this project? → Use search_codebase
- Documentation/PDFs in this project? → Use search_codebase
- "Where is", "What does [code] do", "How is [feature] implemented"? → Use search_codebase

Does the task ask for:
- Mathematical calculation? → Answer directly (NO TOOLS)
- String manipulation? → Answer directly (NO TOOLS)
- General programming concepts? → Answer directly (NO TOOLS)

**Response formats:**

For code/documentation questions (USE TOOL):
{{"tool_name": "search_codebase", "arguments": {{"query": "relevant search query", "limit": 5}}}}

For computation/general knowledge (DIRECT ANSWER):
{{"kind": "final_result", "content": "your answer with calculation/explanation"}}
"""
        else:
            # Subsequent steps: review progress and decide
            prompt = f"""You are working on a multi-step task. Review your progress and decide the next action.

**Task:** {self.task_instruction}

**Steps completed:** {self.step_count} / {self.max_steps}

**Progress so far:**
{self.execution_trace[-2000:] if len(self.execution_trace) > 2000 else self.execution_trace}

**CRITICAL INSTRUCTIONS:**
- Review your progress carefully
- If you have enough information from previous steps: PROVIDE FINAL ANSWER NOW
- Only use tools if you need NEW information
- Avoid repeating the same searches

**Response formats:**

For tool use (ONLY if you need NEW information):
{{"tool_name": "tool_name_here", "arguments": {{"arg1": "value1"}}}}

For final answer (when you have enough information):
{{"kind": "final_result", "content": "your comprehensive final answer synthesizing all information gathered"}}

**Remember:**
- You've already completed {self.step_count} steps
- Use the information you already have in the progress above
- Provide your final answer if you can answer the task
"""

        try:
            # Single LLM call (not two like DecisionFlow + ExecutionFlow)
            system_message = ChatMessage(role="system", content=self.system_prompt)
            user_message = ChatMessage(role="user", content=prompt)
            messages = [system_message, user_message]

            response = await llm.chat_completion(messages)

            # Extract usage
            usage = {}
            if hasattr(response, "usage") and response.usage:
                usage_dict = response.usage
                if isinstance(usage_dict, dict):
                    usage = {
                        "input_tokens": usage_dict.get("prompt_tokens", 0),
                        "output_tokens": usage_dict.get("completion_tokens", 0)
                    }

            content = response.content.strip()

            # Check if it's a final result or tool call
            if "{" in content and "}" in content:
                try:
                    start = content.index("{")
                    end = content.rindex("}") + 1
                    json_str = content[start:end]
                    data = json.loads(json_str)

                    # Check if it's a final result
                    if data.get("kind") == "final_result":
                        return {
                            "kind": "final_result",
                            "content": data["content"],
                            "used_tool": None,
                            "tool_result": None,
                            "usage": usage
                        }

                    # Check if it's a tool call
                    if "tool_name" in data and "arguments" in data:
                        tool_call = ToolCall(
                            tool_name=data["tool_name"],
                            arguments=data["arguments"]
                        )

                        # Execute tool
                        tool_result = await self._execute_tool(tool_call)

                        return {
                            "kind": "next_step",
                            "content": content,
                            "used_tool": tool_call.tool_name,
                            "tool_result": tool_result["content"],
                            "error": tool_result["error"],
                            "usage": usage
                        }

                except (json.JSONDecodeError, ValueError, KeyError):
                    pass

            # No tool call, treat as final answer
            return {
                "kind": "final_result",
                "content": content,
                "used_tool": None,
                "tool_result": None,
                "usage": usage
            }

        except Exception as e:
            error_msg = str(e).lower()
            if "rate" in error_msg and "limit" in error_msg or "429" in str(e):
                raise LLMRateLimitError(str(e))
            elif "network" in error_msg or "connection" in error_msg or "timeout" in error_msg:
                raise LLMNetworkError(str(e))
            else:
                raise

    async def _execute_tool(self, tool_call: ToolCall) -> dict:
        """Execute a tool call using pooled connections."""
        tool_name = tool_call.tool_name

        from .tools import AnalyzeCodeTool, ReadFileTool, SearchCodebaseTool

        tool_classes = {
            "search_codebase": SearchCodebaseTool,
            "read_file": ReadFileTool,
            "analyze_code": AnalyzeCodeTool
        }

        if tool_name not in tool_classes:
            return {"content": f"Unknown tool: {tool_name}", "error": True}

        tool_config = self.tools_config.get(tool_name, {})

        if tool_name == "search_codebase":
            search_config = tool_config.get("search", {})
            storage_config = search_config.get("storage", {})
            embedder_config = search_config.get("embedder", {})

            storage = await ConnectionPool.get_storage(storage_config)
            embedder = await ConnectionPool.get_embedder(embedder_config)

            tool = SearchCodebaseTool(storage, embedder)
        else:
            tool_class = tool_classes[tool_name]
            tool = tool_class()

        try:
            result = await tool.execute(tool_call)
            return {"content": result.content, "error": result.error}
        except Exception as e:
            error_msg = str(e).lower()
            if "rate" in error_msg or "network" in error_msg or "timeout" in error_msg:
                raise LLMNetworkError(str(e))
            else:
                return {"content": f"Tool error: {str(e)}", "error": True}


@dataclass
@flow_type(invokable=str)
class ReasoningTaskFlow:
    """Durable flow for complete multi-step reasoning task.

    This flow orchestrates the entire reasoning loop:
    1. Execute step
    2. Update execution_trace
    3. Decide: continue or done?
    4. Repeat until done or max steps reached

    The execution_trace (execution trace) is persisted at each step.

    NOTE: Mutable state (execution_trace, step_count) is stored as instance variables,
    not passed as step parameters, to avoid Pyergon non-determinism errors.
    """
    task_instruction: str
    llm_config: dict[str, Any]
    tools_config: dict[str, Any]
    system_prompt: str
    max_steps: int = 10

    def __post_init__(self):
        """Initialize mutable state."""
        self.execution_trace = ""
        self.step_count = 0

    @flow
    async def solve(self) -> dict:
        """Main reasoning flow using combined StepFlow.

        Uses StepFlow which combines decision + execution in single LLM call,
        eliminating double LLM overhead and reducing token usage by ~50%.

        Returns:
            {
                "result": str,
                "steps": int,
                "execution_trace": str,
                "status": "completed" | "max_steps_reached",
                "total_input_tokens": int,
                "total_output_tokens": int
            }
        """
        total_input_tokens = 0
        total_output_tokens = 0

        while self.step_count < self.max_steps:
            # Execute combined step (decision + execution in one call)
            step_pending = self.invoke(StepFlow(
                task_instruction=self.task_instruction,
                execution_trace=self.execution_trace,
                step_count=self.step_count,
                max_steps=self.max_steps,
                llm_config=self.llm_config,
                tools_config=self.tools_config,
                system_prompt=self.system_prompt
            ))
            step_result = await step_pending.result()

            # Accumulate usage
            if "usage" in step_result:
                total_input_tokens += step_result["usage"].get("input_tokens", 0)
                total_output_tokens += step_result["usage"].get("output_tokens", 0)

            # Check if final result
            if step_result["kind"] == "final_result":
                return {
                    "result": step_result["content"],
                    "steps": self.step_count,
                    "execution_trace": self.execution_trace,
                    "status": "completed",
                    "total_input_tokens": total_input_tokens,
                    "total_output_tokens": total_output_tokens
                }

            # Update execution_trace
            step_summary = f"\n=== Step {self.step_count + 1} ===\n"
            step_summary += f"Response: {step_result['content']}\n"

            if step_result.get('used_tool'):
                step_summary += f"Tool: {step_result['used_tool']}\n"
                step_summary += f"Result: {step_result['tool_result']}\n"

            self.execution_trace += step_summary
            self.step_count += 1

        # Max steps reached
        return {
            "result": f"Maximum steps reached ({self.step_count} steps completed). Unable to provide final answer within step limit.",
            "steps": self.step_count,
            "execution_trace": self.execution_trace,
            "status": "max_steps_reached",
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens
        }
