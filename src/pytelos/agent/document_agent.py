"""Document agent for question answering over indexed documents."""

import asyncio
import time
from collections.abc import AsyncIterator
from typing import Any

from ..embedding import EmbeddingProvider
from ..llm import LLMProvider
from ..search import SearchMode
from ..storage import StorageBackend
from .data_structures import RetrievedChunk, Task, TaskResult
from .tools import DocumentRetrievalTool


class DocumentAgent:
    """Agent for answering questions over indexed documents.

    Hidden design decisions:
    - Prompt template structure
    - Context formatting strategy
    - Source citation format
    - Streaming vs synchronous response handling
    """

    def __init__(
        self,
        storage: StorageBackend,
        embedder: EmbeddingProvider,
        llm: LLMProvider,
        system_prompt: str | None = None,
        retrieval_limit: int = 5,
        search_mode: SearchMode = SearchMode.HYBRID
    ):
        """Initialize the document agent.

        Args:
            storage: Storage backend for document retrieval
            embedder: Embedding provider for vector search
            llm: LLM provider for response generation
            system_prompt: Optional custom system prompt (or loaded from prompts/document_agent.txt)
            retrieval_limit: Maximum number of documents to retrieve
            search_mode: Search mode (vector, keyword, hybrid)
        """
        self._storage = storage
        self._embedder = embedder
        self._llm = llm
        self._retrieval_limit = retrieval_limit
        self._search_mode = search_mode

        # Load prompt from file if not provided
        if system_prompt is None:
            from ..prompts import get_system_prompt
            system_prompt = get_system_prompt()

        self._system_prompt = system_prompt
        self._retrieval_tool = DocumentRetrievalTool(storage, embedder)

    def run(
        self,
        task: Task,
        stream: bool = False
    ) -> "DocumentAgent.TaskHandler":
        """Execute a task using the TaskHandler pattern.

        Inspired by agent.txt - returns an awaitable TaskHandler.

        Args:
            task: The task to execute
            stream: Whether to stream the response

        Returns:
            TaskHandler that can be awaited for the result
        """
        handler = self.TaskHandler(
            agent=self,
            task=task,
            stream=stream
        )

        async def _execute() -> None:
            """Execute the task in the background."""
            await handler.execute()

        handler.background_task = asyncio.create_task(_execute())

        return handler

    async def run_sync(
        self,
        task: Task,
        stream: bool = False
    ) -> TaskResult | AsyncIterator[str]:
        """Execute a task and await the result synchronously.

        Args:
            task: The task to execute
            stream: Whether to stream the response

        Returns:
            TaskResult if not streaming, AsyncIterator[str] if streaming
        """
        start_time = time.time()

        # Retrieve relevant documents
        chunks, context = await self._retrieval_tool.retrieve_with_context(
            query=task.instruction,
            mode=self._search_mode,
            limit=self._retrieval_limit,
            filters=task.context.get("filters")
        )

        # Build prompt
        prompt = self._build_prompt(task.instruction, context)

        if stream:
            return self._stream_response(task, chunks, prompt, start_time)
        else:
            return await self._generate_response(task, chunks, prompt, start_time)

    async def _generate_response(
        self,
        task: Task,
        chunks: list[RetrievedChunk],
        prompt: str,
        start_time: float
    ) -> TaskResult:
        """Generate a complete response.

        Args:
            task: The task being executed
            chunks: Retrieved document chunks
            prompt: The full prompt for the LLM
            start_time: Start time for measuring processing time

        Returns:
            TaskResult with the complete response
        """
        from ..llm.models import ChatMessage

        # Generate response using LLM
        messages = [ChatMessage(role="user", content=prompt)]
        llm_response = await self._llm.chat_completion(messages)
        response = llm_response.content

        processing_time = time.time() - start_time

        # Format sources
        sources = [
            {
                "source": chunk.source,
                "lines": chunk.lines,
                "score": chunk.score,
                "metadata": chunk.metadata
            }
            for chunk in chunks
        ]

        return TaskResult(
            task_id=task.id_,
            content=response,
            sources=sources,
            metadata={
                "processing_time_seconds": processing_time,
                "num_sources": len(sources),
                "search_mode": self._search_mode.value
            }
        )

    async def _stream_response(
        self,
        task: Task,
        chunks: list[RetrievedChunk],
        prompt: str,
        start_time: float
    ) -> AsyncIterator[str]:
        """Stream the response token by token.

        Args:
            task: The task being executed
            chunks: Retrieved document chunks
            prompt: The full prompt for the LLM
            start_time: Start time for measuring processing time

        Yields:
            Response tokens as they are generated
        """
        from ..llm.models import ChatMessage

        messages = [ChatMessage(role="user", content=prompt)]
        stream_response = await self._llm.chat_completion_stream(messages)
        async for token in stream_response:
            yield token

    def _build_prompt(self, question: str, context: str) -> str:
        """Build the complete prompt for the LLM.

        Args:
            question: User's question
            context: Formatted document context

        Returns:
            Complete prompt string
        """
        return f"""{self._system_prompt}

Documents:
{context}

Question: {question}

Answer:"""

    async def close(self) -> None:
        """Clean up resources."""
        await self._llm.close()

    class TaskHandler(asyncio.Future):
        """Handler for processing document QA tasks.

        Inspired by agent.txt TaskHandler pattern.
        Provides async task execution with execution_trace tracking.
        """

        def __init__(
            self,
            agent: "DocumentAgent",
            task: Task,
            stream: bool = False,
            *args: Any,
            **kwargs: Any
        ):
            """Initialize TaskHandler.

            Args:
                agent: The DocumentAgent instance
                task: The task to execute
                stream: Whether to stream the response
                *args: Additional positional arguments
                **kwargs: Additional keyword arguments
            """
            super().__init__(*args, **kwargs)
            self.agent = agent
            self.task = task
            self.stream = stream
            self.execution_trace: list[dict[str, Any]] = []
            self._background_task: asyncio.Task | None = None
            self._start_time = time.time()

        async def execute(self) -> TaskResult:
            """Execute the task and track the execution.

            Returns:
                TaskResult with the response and metadata
            """
            try:
                # Step 1: Retrieval
                self._log_step("retrieval_start", {"query": self.task.instruction})

                chunks, context = await self.agent._retrieval_tool.retrieve_with_context(
                    query=self.task.instruction,
                    mode=self.agent._search_mode,
                    limit=self.agent._retrieval_limit,
                    filters=self.task.context.get("filters")
                )

                self._log_step(
                    "retrieval_complete",
                    {
                        "num_chunks": len(chunks),
                        "sources": [c.source for c in chunks]
                    }
                )

                # Step 2: Prompt construction
                prompt = self.agent._build_prompt(self.task.instruction, context)
                self._log_step(
                    "prompt_built",
                    {"prompt_length": len(prompt)}
                )

                # Step 3: LLM generation
                self._log_step("generation_start", {})

                from ..llm.models import ChatMessage

                messages = [ChatMessage(role="user", content=prompt)]
                llm_response = await self.agent._llm.chat_completion(messages)
                response = llm_response.content

                self._log_step(
                    "generation_complete",
                    {"response_length": len(response)}
                )

                # Step 4: Build result
                processing_time = time.time() - self._start_time

                sources = [
                    {
                        "source": chunk.source,
                        "lines": chunk.lines,
                        "score": chunk.score,
                        "metadata": chunk.metadata
                    }
                    for chunk in chunks
                ]

                result = TaskResult(
                    task_id=self.task.id_,
                    content=response,
                    sources=sources,
                    metadata={
                        "processing_time_seconds": processing_time,
                        "num_sources": len(sources),
                        "search_mode": self.agent._search_mode.value,
                        "execution_trace": self.execution_trace
                    }
                )

                self.set_result(result)
                return result

            except Exception as e:
                self._log_step("error", {"error": str(e)})
                self.set_exception(e)
                raise

        def _log_step(self, step_name: str, data: dict[str, Any]) -> None:
            """Log an execution step to the execution_trace.

            Args:
                step_name: Name of the step
                data: Step data
            """
            self.execution_trace.append({
                "step": step_name,
                "timestamp": time.time() - self._start_time,
                "data": data
            })

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
