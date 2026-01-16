"""SQLite conversation memory backend.

Provides persistent conversation storage using SQLite database.
Uses aiosqlite for async access.
"""

import json
from datetime import datetime
from pathlib import Path
from uuid import uuid4

try:
    import aiosqlite
    AIOSQLITE_AVAILABLE = True
except ImportError:
    AIOSQLITE_AVAILABLE = False
    aiosqlite = None

from .base import ConversationMemory
from .models import ConversationState, TaskRecord


class SQLiteConversationMemory(ConversationMemory):
    """SQLite-backed conversation memory.

    Stores conversation state in a SQLite database file.
    Supports persistent storage across sessions.
    """

    def __init__(
        self,
        path: str | Path = "./conversation_memory.db",
        default_session_id: str | None = None
    ):
        if not AIOSQLITE_AVAILABLE:
            raise ImportError(
                "SQLite memory backend requires aiosqlite. "
                "Install with: pip install aiosqlite"
            )

        self._db_path = Path(path)
        self._default_session_id = default_session_id or str(uuid4())
        self._connection: aiosqlite.Connection | None = None

    async def connect(self) -> None:
        """Initialize database connection and schema."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = await aiosqlite.connect(self._db_path)
        await self._connection.execute("PRAGMA foreign_keys = ON")
        await self._create_schema()
        await self._ensure_session(self._default_session_id)

    async def _create_schema(self) -> None:
        """Create database tables."""
        await self._connection.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)

        await self._connection.execute("""
            CREATE TABLE IF NOT EXISTS task_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                task_id INTEGER NOT NULL,
                task TEXT NOT NULL,
                answer TEXT NOT NULL,
                files_read TEXT DEFAULT '[]',
                timestamp TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
            )
        """)

        await self._connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_task_records_session
            ON task_records(session_id, task_id)
        """)

        await self._connection.commit()

    async def _ensure_session(self, session_id: str) -> None:
        """Ensure a session exists."""
        now = datetime.utcnow().isoformat()
        await self._connection.execute("""
            INSERT OR IGNORE INTO sessions (session_id, created_at, updated_at)
            VALUES (?, ?, ?)
        """, (session_id, now, now))
        await self._connection.commit()

    async def disconnect(self) -> None:
        """Close database connection."""
        if self._connection:
            await self._connection.close()
            self._connection = None

    async def get_state(self, session_id: str | None = None) -> ConversationState:
        """Retrieve conversation state."""
        sid = session_id or self._default_session_id

        async with self._connection.execute(
            "SELECT session_id, created_at, updated_at FROM sessions WHERE session_id = ?",
            (sid,)
        ) as cursor:
            row = await cursor.fetchone()

        if row is None:
            await self._ensure_session(sid)
            return ConversationState(session_id=sid)

        session_id_db, created_at, updated_at = row

        async with self._connection.execute(
            """
            SELECT task_id, task, answer, files_read, timestamp
            FROM task_records
            WHERE session_id = ?
            ORDER BY task_id ASC
            """,
            (sid,)
        ) as cursor:
            rows = await cursor.fetchall()

        history = []
        all_files_read = []

        for row in rows:
            task_id, task, answer, files_read_json, ts = row
            files_read = json.loads(files_read_json)

            record = TaskRecord(
                task_id=task_id,
                task=task,
                answer=answer,
                files_read=files_read,
                timestamp=datetime.fromisoformat(ts)
            )
            history.append(record)
            all_files_read.extend(files_read)

        state = ConversationState(
            session_id=sid,
            history=history,
            files_read=list(set(all_files_read)),
            created_at=datetime.fromisoformat(created_at),
            updated_at=datetime.fromisoformat(updated_at),
        )

        if history:
            state.last_task = history[-1].task[:200]
            state.last_answer = history[-1].answer[:1000]

        return state

    async def save_state(self, state: ConversationState) -> None:
        """Persist conversation state."""
        now = datetime.utcnow().isoformat()

        await self._connection.execute("""
            INSERT INTO sessions (session_id, created_at, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET updated_at = excluded.updated_at
        """, (state.session_id, state.created_at.isoformat(), now))

        await self._connection.execute(
            "DELETE FROM task_records WHERE session_id = ?",
            (state.session_id,)
        )

        for record in state.history:
            await self._connection.execute("""
                INSERT INTO task_records
                (session_id, task_id, task, answer, files_read, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                state.session_id,
                record.task_id,
                record.task,
                record.answer,
                json.dumps(record.files_read),
                record.timestamp.isoformat()
            ))

        await self._connection.commit()

    async def add_task_record(
        self,
        task: str,
        answer: str,
        files_read: list[str] | None = None,
        session_id: str | None = None
    ) -> TaskRecord:
        """Add a task record."""
        sid = session_id or self._default_session_id
        await self._ensure_session(sid)

        async with self._connection.execute(
            "SELECT COALESCE(MAX(task_id), 0) + 1 FROM task_records WHERE session_id = ?",
            (sid,)
        ) as cursor:
            row = await cursor.fetchone()
            task_id = row[0]

        record = TaskRecord(
            task_id=task_id,
            task=task,
            answer=answer,
            files_read=files_read or [],
        )

        await self._connection.execute("""
            INSERT INTO task_records
            (session_id, task_id, task, answer, files_read, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            sid,
            record.task_id,
            record.task,
            record.answer,
            json.dumps(record.files_read),
            record.timestamp.isoformat()
        ))

        await self._connection.execute(
            "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
            (datetime.utcnow().isoformat(), sid)
        )

        await self._connection.commit()
        return record

    async def get_recent_tasks(
        self,
        limit: int = 5,
        session_id: str | None = None
    ) -> list[TaskRecord]:
        """Get recent tasks."""
        sid = session_id or self._default_session_id

        async with self._connection.execute(
            """
            SELECT task_id, task, answer, files_read, timestamp
            FROM task_records
            WHERE session_id = ?
            ORDER BY task_id DESC
            LIMIT ?
            """,
            (sid, limit)
        ) as cursor:
            rows = await cursor.fetchall()

        records = []
        for row in rows:
            task_id, task, answer, files_read_json, ts = row
            records.append(TaskRecord(
                task_id=task_id,
                task=task,
                answer=answer,
                files_read=json.loads(files_read_json),
                timestamp=datetime.fromisoformat(ts)
            ))

        return records

    async def clear_history(self, session_id: str | None = None) -> None:
        """Clear history."""
        sid = session_id or self._default_session_id
        await self._connection.execute(
            "DELETE FROM task_records WHERE session_id = ?",
            (sid,)
        )
        await self._connection.commit()

    @property
    def backend_type(self) -> str:
        return "sqlite"

    @property
    def default_session_id(self) -> str:
        return self._default_session_id

    @property
    def db_path(self) -> Path:
        return self._db_path
