import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import aiosqlite

from pya_durable.models import (
    ConversationRecord,
    DurableDocRecord,
    SubmissionRecord,
    SubmissionStatus,
    TaskRecord,
    TaskStatus,
    ToolInvocationRecord,
    ToolInvocationStatus,
    TranscriptEntryKind,
    TranscriptEntryRecord,
    utc_now_iso,
)

logger = logging.getLogger("pya_durable.storage")

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS conversations (
    conversation_id TEXT PRIMARY KEY,
    parent_conversation_id TEXT,
    fork_entry_id TEXT,
    agent_config_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(parent_conversation_id) REFERENCES conversations(conversation_id)
);

CREATE TABLE IF NOT EXISTS transcript_entries (
    entry_id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    parent_entry_id TEXT,
    kind TEXT NOT NULL,
    message_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY(conversation_id) REFERENCES conversations(conversation_id),
    FOREIGN KEY(parent_entry_id) REFERENCES transcript_entries(entry_id)
);

CREATE INDEX IF NOT EXISTS idx_transcript_parent ON transcript_entries(parent_entry_id);
CREATE INDEX IF NOT EXISTS idx_transcript_conv ON transcript_entries(conversation_id, created_at);

CREATE TABLE IF NOT EXISTS tasks (
    task_id TEXT PRIMARY KEY,
    task_type TEXT NOT NULL,
    version INTEGER NOT NULL DEFAULT 1,
    status TEXT NOT NULL,
    owner_kind TEXT NOT NULL,
    owner_id TEXT NOT NULL,
    phase TEXT NOT NULL,
    input_json TEXT NOT NULL,
    state_json TEXT NOT NULL,
    checkpoint_json TEXT,
    policy TEXT NOT NULL DEFAULT 'failFast',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_tasks_owner ON tasks(owner_kind, owner_id);
CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);

CREATE TABLE IF NOT EXISTS tool_invocations (
    invocation_id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    args_json TEXT NOT NULL,
    replay_policy TEXT NOT NULL,
    status TEXT NOT NULL,
    result_json TEXT,
    partial_output TEXT,
    error_message TEXT,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    FOREIGN KEY(task_id) REFERENCES tasks(task_id),
    FOREIGN KEY(conversation_id) REFERENCES conversations(conversation_id)
);

CREATE INDEX IF NOT EXISTS idx_tool_invocations_task ON tool_invocations(task_id);
CREATE INDEX IF NOT EXISTS idx_tool_invocations_conv ON tool_invocations(conversation_id);

CREATE TABLE IF NOT EXISTS submissions (
    request_id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    prompt_text TEXT NOT NULL,
    task_id TEXT NOT NULL,
    status TEXT NOT NULL,
    result_json TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY(conversation_id) REFERENCES conversations(conversation_id),
    FOREIGN KEY(task_id) REFERENCES tasks(task_id)
);

CREATE INDEX IF NOT EXISTS idx_submissions_conv ON submissions(conversation_id);

CREATE TABLE IF NOT EXISTS durable_docs (
    doc_id TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    version INTEGER NOT NULL DEFAULT 1,
    doc_type TEXT NOT NULL,
    data_json TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY(doc_id, conversation_id),
    FOREIGN KEY(conversation_id) REFERENCES conversations(conversation_id)
);
"""


class AioSqliteStorage:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        self._conn: aiosqlite.Connection | None = None
        self._lock = asyncio.Lock()
        self._initialized = False

    async def get_connection(self) -> aiosqlite.Connection:
        if self._conn is None:
            if self.db_path != ":memory:":
                parent = os.path.dirname(os.path.abspath(self.db_path))
                if parent:
                    os.makedirs(parent, exist_ok=True)
            self._conn = await aiosqlite.connect(self.db_path, isolation_level=None)
            await self._conn.execute("PRAGMA journal_mode = WAL")
            await self._conn.execute("PRAGMA synchronous = NORMAL")
            await self._conn.execute("PRAGMA foreign_keys = ON")
            await self._conn.execute("PRAGMA busy_timeout = 5000")
        return self._conn

    async def create_reader_connection(self) -> aiosqlite.Connection:
        conn = await aiosqlite.connect(self.db_path, isolation_level=None)
        await conn.execute("PRAGMA journal_mode = WAL")
        await conn.execute("PRAGMA synchronous = NORMAL")
        await conn.execute("PRAGMA foreign_keys = ON")
        await conn.execute("PRAGMA busy_timeout = 5000")
        return conn

    async def init_db(self) -> None:
        async with self._lock:
            conn = await self.get_connection()
            await conn.executescript(SCHEMA_SQL)
            self._initialized = True
            logger.info("Storage initialized at %s with WAL mode active", self.db_path)

    async def close(self) -> None:
        async with self._lock:
            if self._conn is not None:
                await self._conn.close()
                self._conn = None
                self._initialized = False

    async def __aenter__(self) -> "AioSqliteStorage":
        await self.init_db()
        return self

    async def __aexit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        await self.close()

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[aiosqlite.Connection]:
        async with self._lock:
            conn = await self.get_connection()
            if not self._initialized:
                await conn.executescript(SCHEMA_SQL)
                self._initialized = True
                logger.info("Storage initialized at %s with WAL mode active", self.db_path)
            await conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
                await conn.execute("COMMIT")
            except Exception:
                await conn.execute("ROLLBACK")
                raise

    async def insert_conversation(
        self,
        record: ConversationRecord,
        conn: aiosqlite.Connection | None = None,
    ) -> None:
        target = conn or await self.get_connection()
        await target.execute(
            """
            INSERT INTO conversations (conversation_id, parent_conversation_id, fork_entry_id, agent_config_json, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                record.conversation_id,
                record.parent_conversation_id,
                record.fork_entry_id,
                record.agent_config_json,
                record.created_at,
                record.updated_at,
            ),
        )

    async def get_conversation(
        self,
        conversation_id: str,
        conn: aiosqlite.Connection | None = None,
    ) -> ConversationRecord | None:
        target = conn or await self.get_connection()
        async with target.execute(
            """
            SELECT conversation_id, parent_conversation_id, fork_entry_id, agent_config_json, created_at, updated_at
            FROM conversations
            WHERE conversation_id = ?
            """,
            (conversation_id,),
        ) as cursor:
            row = await cursor.fetchone()
            if row is None:
                return None
            return ConversationRecord(
                conversation_id=row[0],
                parent_conversation_id=row[1],
                fork_entry_id=row[2],
                agent_config_json=row[3],
                created_at=row[4],
                updated_at=row[5],
            )

    async def insert_transcript_entry(
        self,
        entry: TranscriptEntryRecord,
        conn: aiosqlite.Connection | None = None,
    ) -> None:
        target = conn or await self.get_connection()
        await target.execute(
            """
            INSERT INTO transcript_entries (entry_id, conversation_id, parent_entry_id, kind, message_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                entry.entry_id,
                entry.conversation_id,
                entry.parent_entry_id,
                entry.kind.value if isinstance(entry.kind, TranscriptEntryKind) else str(entry.kind),
                entry.message_json,
                entry.created_at,
            ),
        )

    async def get_transcript_entry(
        self,
        entry_id: str,
        conn: aiosqlite.Connection | None = None,
    ) -> TranscriptEntryRecord | None:
        target = conn or await self.get_connection()
        async with target.execute(
            """
            SELECT entry_id, conversation_id, parent_entry_id, kind, message_json, created_at
            FROM transcript_entries
            WHERE entry_id = ?
            """,
            (entry_id,),
        ) as cursor:
            row = await cursor.fetchone()
            if row is None:
                return None
            return TranscriptEntryRecord(
                entry_id=row[0],
                conversation_id=row[1],
                parent_entry_id=row[2],
                kind=TranscriptEntryKind(row[3]),
                message_json=row[4],
                created_at=row[5],
            )

    async def get_conversation_transcript(
        self,
        conversation_id: str,
        conn: aiosqlite.Connection | None = None,
    ) -> list[TranscriptEntryRecord]:
        target = conn or await self.get_connection()
        async with target.execute(
            """
            SELECT entry_id, conversation_id, parent_entry_id, kind, message_json, created_at
            FROM transcript_entries
            WHERE conversation_id = ?
            ORDER BY created_at ASC
            """,
            (conversation_id,),
        ) as cursor:
            rows = await cursor.fetchall()
            return [
                TranscriptEntryRecord(
                    entry_id=row[0],
                    conversation_id=row[1],
                    parent_entry_id=row[2],
                    kind=TranscriptEntryKind(row[3]),
                    message_json=row[4],
                    created_at=row[5],
                )
                for row in rows
            ]

    async def get_ancestor_entries(
        self,
        leaf_entry_id: str,
        conn: aiosqlite.Connection | None = None,
    ) -> list[TranscriptEntryRecord]:
        target = conn or await self.get_connection()
        query = """
        WITH RECURSIVE ancestors(entry_id, conversation_id, parent_entry_id, kind, message_json, created_at, depth) AS (
            SELECT entry_id, conversation_id, parent_entry_id, kind, message_json, created_at, 0
            FROM transcript_entries
            WHERE entry_id = ?
            UNION ALL
            SELECT t.entry_id, t.conversation_id, t.parent_entry_id, t.kind, t.message_json, t.created_at, a.depth + 1
            FROM transcript_entries t
            JOIN ancestors a ON t.entry_id = a.parent_entry_id
        )
        SELECT entry_id, conversation_id, parent_entry_id, kind, message_json, created_at
        FROM ancestors
        ORDER BY depth DESC;
        """
        async with target.execute(query, (leaf_entry_id,)) as cursor:
            rows = await cursor.fetchall()
            return [
                TranscriptEntryRecord(
                    entry_id=row[0],
                    conversation_id=row[1],
                    parent_entry_id=row[2],
                    kind=TranscriptEntryKind(row[3]),
                    message_json=row[4],
                    created_at=row[5],
                )
                for row in rows
            ]

    async def insert_task(
        self,
        task: TaskRecord,
        conn: aiosqlite.Connection | None = None,
    ) -> None:
        target = conn or await self.get_connection()
        await target.execute(
            """
            INSERT INTO tasks (
                task_id, task_type, version, status, owner_kind, owner_id,
                phase, input_json, state_json, checkpoint_json, policy, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                task.task_id,
                task.task_type,
                task.version,
                task.status.value if isinstance(task.status, TaskStatus) else str(task.status),
                task.owner_kind,
                task.owner_id,
                task.phase,
                task.input_json,
                task.state_json,
                task.checkpoint_json,
                task.policy,
                task.created_at,
                task.updated_at,
            ),
        )

    async def update_task_state(
        self,
        task_id: str,
        status: TaskStatus,
        phase: str,
        state_json: str,
        checkpoint_json: str | None = None,
        conn: aiosqlite.Connection | None = None,
    ) -> None:
        target = conn or await self.get_connection()
        updated_at = utc_now_iso()
        await target.execute(
            """
            UPDATE tasks
            SET status = ?, phase = ?, state_json = ?, checkpoint_json = ?, updated_at = ?
            WHERE task_id = ?
            """,
            (
                status.value if isinstance(status, TaskStatus) else str(status),
                phase,
                state_json,
                checkpoint_json,
                updated_at,
                task_id,
            ),
        )

    async def get_task(
        self,
        task_id: str,
        conn: aiosqlite.Connection | None = None,
    ) -> TaskRecord | None:
        target = conn or await self.get_connection()
        async with target.execute(
            """
            SELECT task_id, task_type, version, status, owner_kind, owner_id,
                   phase, input_json, state_json, checkpoint_json, policy, created_at, updated_at
            FROM tasks
            WHERE task_id = ?
            """,
            (task_id,),
        ) as cursor:
            row = await cursor.fetchone()
            if row is None:
                return None
            return TaskRecord(
                task_id=row[0],
                task_type=row[1],
                version=row[2],
                status=TaskStatus(row[3]),
                owner_kind=row[4],
                owner_id=row[5],
                phase=row[6],
                input_json=row[7],
                state_json=row[8],
                checkpoint_json=row[9],
                policy=row[10],
                created_at=row[11],
                updated_at=row[12],
            )

    async def record_tool_started(
        self,
        invocation: ToolInvocationRecord,
        conn: aiosqlite.Connection | None = None,
    ) -> None:
        target = conn or await self.get_connection()
        await target.execute(
            """
            INSERT INTO tool_invocations (
                invocation_id, task_id, conversation_id, tool_name, args_json,
                replay_policy, status, result_json, partial_output, error_message, started_at, completed_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                invocation.invocation_id,
                invocation.task_id,
                invocation.conversation_id,
                invocation.tool_name,
                invocation.args_json,
                invocation.replay_policy.value,
                invocation.status.value,
                invocation.result_json,
                invocation.partial_output,
                invocation.error_message,
                invocation.started_at,
                invocation.completed_at,
            ),
        )

    async def record_tool_completed(
        self,
        invocation_id: str,
        result_json: str,
        conn: aiosqlite.Connection | None = None,
    ) -> None:
        target = conn or await self.get_connection()
        completed_at = utc_now_iso()
        await target.execute(
            """
            UPDATE tool_invocations
            SET status = ?, result_json = ?, completed_at = ?
            WHERE invocation_id = ?
            """,
            (ToolInvocationStatus.COMPLETED.value, result_json, completed_at, invocation_id),
        )

    async def record_tool_failed(
        self,
        invocation_id: str,
        error_message: str,
        conn: aiosqlite.Connection | None = None,
    ) -> None:
        target = conn or await self.get_connection()
        completed_at = utc_now_iso()
        await target.execute(
            """
            UPDATE tool_invocations
            SET status = ?, error_message = ?, completed_at = ?
            WHERE invocation_id = ?
            """,
            (ToolInvocationStatus.FAILED.value, error_message, completed_at, invocation_id),
        )

    async def record_tool_interrupted(
        self,
        invocation_id: str,
        partial_output: str | None = None,
        conn: aiosqlite.Connection | None = None,
    ) -> None:
        target = conn or await self.get_connection()
        completed_at = utc_now_iso()
        await target.execute(
            """
            UPDATE tool_invocations
            SET status = ?, partial_output = ?, completed_at = ?
            WHERE invocation_id = ?
            """,
            (ToolInvocationStatus.INTERRUPTED.value, partial_output, completed_at, invocation_id),
        )

    async def get_tool_invocation(
        self,
        invocation_id: str,
        conn: aiosqlite.Connection | None = None,
    ) -> ToolInvocationRecord | None:
        target = conn or await self.get_connection()
        async with target.execute(
            """
            SELECT invocation_id, task_id, conversation_id, tool_name, args_json,
                   replay_policy, status, result_json, partial_output, error_message, started_at, completed_at
            FROM tool_invocations
            WHERE invocation_id = ?
            """,
            (invocation_id,),
        ) as cursor:
            row = await cursor.fetchone()
            if row is None:
                return None
            return ToolInvocationRecord(
                invocation_id=row[0],
                task_id=row[1],
                conversation_id=row[2],
                tool_name=row[3],
                args_json=row[4],
                replay_policy=row[5],
                status=ToolInvocationStatus(row[6]),
                result_json=row[7],
                partial_output=row[8],
                error_message=row[9],
                started_at=row[10],
                completed_at=row[11],
            )

    async def insert_submission(
        self,
        submission: SubmissionRecord,
        conn: aiosqlite.Connection | None = None,
    ) -> None:
        target = conn or await self.get_connection()
        await target.execute(
            """
            INSERT INTO submissions (request_id, conversation_id, prompt_text, task_id, status, result_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                submission.request_id,
                submission.conversation_id,
                submission.prompt_text,
                submission.task_id,
                submission.status.value,
                submission.result_json,
                submission.created_at,
            ),
        )

    async def get_submission(
        self,
        request_id: str,
        conn: aiosqlite.Connection | None = None,
    ) -> SubmissionRecord | None:
        target = conn or await self.get_connection()
        async with target.execute(
            """
            SELECT request_id, conversation_id, prompt_text, task_id, status, result_json, created_at
            FROM submissions
            WHERE request_id = ?
            """,
            (request_id,),
        ) as cursor:
            row = await cursor.fetchone()
            if row is None:
                return None
            return SubmissionRecord(
                request_id=row[0],
                conversation_id=row[1],
                prompt_text=row[2],
                task_id=row[3],
                status=SubmissionStatus(row[4]),
                result_json=row[5],
                created_at=row[6],
            )

    async def upsert_durable_doc(
        self,
        doc: DurableDocRecord,
        conn: aiosqlite.Connection | None = None,
    ) -> None:
        target = conn or await self.get_connection()
        await target.execute(
            """
            INSERT INTO durable_docs (doc_id, conversation_id, version, doc_type, data_json, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(doc_id, conversation_id) DO UPDATE SET
                version = excluded.version,
                doc_type = excluded.doc_type,
                data_json = excluded.data_json,
                updated_at = excluded.updated_at
            """,
            (
                doc.doc_id,
                doc.conversation_id,
                doc.version,
                doc.doc_type,
                doc.data_json,
                doc.updated_at,
            ),
        )

    async def get_durable_doc(
        self,
        doc_id: str,
        conversation_id: str,
        conn: aiosqlite.Connection | None = None,
    ) -> DurableDocRecord | None:
        target = conn or await self.get_connection()
        async with target.execute(
            """
            SELECT doc_id, conversation_id, version, doc_type, data_json, updated_at
            FROM durable_docs
            WHERE doc_id = ? AND conversation_id = ?
            """,
            (doc_id, conversation_id),
        ) as cursor:
            row = await cursor.fetchone()
            if row is None:
                return None
            return DurableDocRecord(
                doc_id=row[0],
                conversation_id=row[1],
                version=row[2],
                doc_type=row[3],
                data_json=row[4],
                updated_at=row[5],
            )

    async def checkpoint(self) -> tuple[int, int, int]:
        conn = await self.get_connection()
        async with conn.execute("PRAGMA wal_checkpoint(TRUNCATE)") as cursor:
            row = await cursor.fetchone()
            if row is None:
                return (0, 0, 0)
            return (int(row[0]), int(row[1]), int(row[2]))

    async def vacuum(self) -> None:
        conn = await self.get_connection()
        await conn.execute("VACUUM")

    async def check_integrity(self) -> bool:
        conn = await self.get_connection()
        async with conn.execute("PRAGMA integrity_check") as cursor:
            row = await cursor.fetchone()
            return bool(row and row[0] == "ok")
