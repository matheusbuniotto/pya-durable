import asyncio
import logging
import sqlite3
import tempfile
from pathlib import Path

import aiosqlite
import pytest

from pya_durable.models import (
    ConversationRecord,
    DurableDocRecord,
    ReplayPolicy,
    SubmissionRecord,
    SubmissionStatus,
    TaskRecord,
    TaskStatus,
    ToolInvocationRecord,
    ToolInvocationStatus,
    TranscriptEntryKind,
    TranscriptEntryRecord,
)
from pya_durable.storage import AioSqliteStorage


@pytest.fixture
def db_path(tmp_path: Path) -> str:
    return str(tmp_path / "test_pya.db")


@pytest.mark.asyncio
async def test_init_db_creates_tables_and_wal(db_path: str, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    storage = AioSqliteStorage(db_path)
    try:
        await storage.init_db()

        conn = await storage.get_connection()
        async with conn.execute("PRAGMA journal_mode") as cursor:
            row = await cursor.fetchone()
            assert row is not None
            assert row[0].lower() == "wal"

        async with conn.execute("PRAGMA foreign_keys") as cursor:
            row = await cursor.fetchone()
            assert row is not None
            assert row[0] == 1

        async with conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ) as cursor:
            tables = [r[0] for r in await cursor.fetchall()]

        expected_tables = {
            "conversations",
            "durable_docs",
            "submissions",
            "tasks",
            "tool_invocations",
            "transcript_entries",
        }
        assert expected_tables.issubset(set(tables))
        assert f"Storage initialized at {db_path} with WAL mode active" in caplog.text
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_conversation_crud(db_path: str) -> None:
    async with AioSqliteStorage(db_path) as storage:
        conv = ConversationRecord(
            conversation_id="conv-1",
            parent_conversation_id=None,
            fork_entry_id=None,
            agent_config_json='{"model": "test"}',
        )
        await storage.insert_conversation(conv)
        fetched = await storage.get_conversation("conv-1")
        assert fetched is not None
        assert fetched.conversation_id == "conv-1"
        assert fetched.agent_config_json == '{"model": "test"}'

        missing = await storage.get_conversation("nonexistent")
        assert missing is None


@pytest.mark.asyncio
async def test_transcript_linkage_and_ancestor_cte(db_path: str) -> None:
    async with AioSqliteStorage(db_path) as storage:
        conv = ConversationRecord(
            conversation_id="conv-tree",
            agent_config_json="{}",
        )
        await storage.insert_conversation(conv)

        e1 = TranscriptEntryRecord(
            entry_id="entry-root",
            conversation_id="conv-tree",
            parent_entry_id=None,
            kind=TranscriptEntryKind.USER,
            message_json='{"content": "hello"}',
        )
        e2 = TranscriptEntryRecord(
            entry_id="entry-child-1",
            conversation_id="conv-tree",
            parent_entry_id="entry-root",
            kind=TranscriptEntryKind.MODEL_RESPONSE,
            message_json='{"content": "world"}',
        )
        e3 = TranscriptEntryRecord(
            entry_id="entry-leaf",
            conversation_id="conv-tree",
            parent_entry_id="entry-child-1",
            kind=TranscriptEntryKind.USER,
            message_json='{"content": "next"}',
        )

        await storage.insert_transcript_entry(e1)
        await storage.insert_transcript_entry(e2)
        await storage.insert_transcript_entry(e3)

        transcript = await storage.get_conversation_transcript("conv-tree")
        assert len(transcript) == 3

        ancestors = await storage.get_ancestor_entries("entry-leaf")
        assert len(ancestors) == 3
        assert [a.entry_id for a in ancestors] == ["entry-root", "entry-child-1", "entry-leaf"]


@pytest.mark.asyncio
async def test_foreign_key_violation_raises(db_path: str) -> None:
    async with AioSqliteStorage(db_path) as storage:
        bad_entry = TranscriptEntryRecord(
            entry_id="orphan",
            conversation_id="nonexistent-conv",
            parent_entry_id=None,
            kind=TranscriptEntryKind.USER,
            message_json="{}",
        )
        with pytest.raises((sqlite3.IntegrityError, aiosqlite.IntegrityError)):
            await storage.insert_transcript_entry(bad_entry)


@pytest.mark.asyncio
async def test_task_crud_and_status_update(db_path: str) -> None:
    async with AioSqliteStorage(db_path) as storage:
        task = TaskRecord(
            task_id="task-100",
            task_type="code-generation",
            version=1,
            status=TaskStatus.PENDING,
            owner_kind="root",
            owner_id="system",
            phase="init",
            input_json='{"prompt": "write code"}',
            state_json='{"step": 0}',
            checkpoint_json=None,
            policy="failFast",
        )
        await storage.insert_task(task)

        stored = await storage.get_task("task-100")
        assert stored is not None
        assert stored.status == TaskStatus.PENDING

        await storage.update_task_state(
            task_id="task-100",
            status=TaskStatus.RUNNING,
            phase="execute",
            state_json='{"step": 1}',
            checkpoint_json='{"saved": true}',
        )

        updated = await storage.get_task("task-100")
        assert updated is not None
        assert updated.status == TaskStatus.RUNNING
        assert updated.phase == "execute"
        assert updated.checkpoint_json == '{"saved": true}'


@pytest.mark.asyncio
async def test_tool_invocation_transitions(db_path: str) -> None:
    async with AioSqliteStorage(db_path) as storage:
        conv = ConversationRecord(conversation_id="conv-t", agent_config_json="{}")
        await storage.insert_conversation(conv)

        task = TaskRecord(
            task_id="task-t",
            task_type="test",
            status=TaskStatus.RUNNING,
            owner_kind="conversation",
            owner_id="conv-t",
            phase="active",
            input_json="{}",
            state_json="{}",
        )
        await storage.insert_task(task)

        inv = ToolInvocationRecord(
            invocation_id="inv-1",
            task_id="task-t",
            conversation_id="conv-t",
            tool_name="bash",
            args_json='{"cmd": "echo 1"}',
            replay_policy=ReplayPolicy.SAFE,
            status=ToolInvocationStatus.STARTED,
        )
        await storage.record_tool_started(inv)

        active = await storage.get_tool_invocation("inv-1")
        assert active is not None
        assert active.status == ToolInvocationStatus.STARTED

        await storage.record_tool_completed("inv-1", '{"exit": 0}')
        completed = await storage.get_tool_invocation("inv-1")
        assert completed is not None
        assert completed.status == ToolInvocationStatus.COMPLETED
        assert completed.result_json == '{"exit": 0}'
        assert completed.completed_at is not None


@pytest.mark.asyncio
async def test_submission_deduplication(db_path: str) -> None:
    async with AioSqliteStorage(db_path) as storage:
        conv = ConversationRecord(conversation_id="conv-sub", agent_config_json="{}")
        await storage.insert_conversation(conv)

        task = TaskRecord(
            task_id="task-sub",
            task_type="test",
            status=TaskStatus.PENDING,
            owner_kind="conversation",
            owner_id="conv-sub",
            phase="init",
            input_json="{}",
            state_json="{}",
        )
        await storage.insert_task(task)

        sub1 = SubmissionRecord(
            request_id="req-dup-1",
            conversation_id="conv-sub",
            prompt_text="do something",
            task_id="task-sub",
            status=SubmissionStatus.ACTIVE,
        )
        await storage.insert_submission(sub1)

        sub2 = SubmissionRecord(
            request_id="req-dup-1",
            conversation_id="conv-sub",
            prompt_text="do something else",
            task_id="task-sub",
            status=SubmissionStatus.ACTIVE,
        )
        with pytest.raises((sqlite3.IntegrityError, aiosqlite.IntegrityError)):
            await storage.insert_submission(sub2)


@pytest.mark.asyncio
async def test_durable_doc_upsert(db_path: str) -> None:
    async with AioSqliteStorage(db_path) as storage:
        conv = ConversationRecord(conversation_id="conv-doc", agent_config_json="{}")
        await storage.insert_conversation(conv)

        doc_v1 = DurableDocRecord(
            doc_id="notes",
            conversation_id="conv-doc",
            version=1,
            doc_type="markdown",
            data_json='{"text": "initial notes"}',
        )
        await storage.upsert_durable_doc(doc_v1)

        fetched_v1 = await storage.get_durable_doc("notes", "conv-doc")
        assert fetched_v1 is not None
        assert fetched_v1.version == 1

        doc_v2 = DurableDocRecord(
            doc_id="notes",
            conversation_id="conv-doc",
            version=2,
            doc_type="markdown",
            data_json='{"text": "updated notes"}',
        )
        await storage.upsert_durable_doc(doc_v2)

        fetched_v2 = await storage.get_durable_doc("notes", "conv-doc")
        assert fetched_v2 is not None
        assert fetched_v2.version == 2
        assert fetched_v2.data_json == '{"text": "updated notes"}'


@pytest.mark.asyncio
async def test_atomic_transaction_rollback(db_path: str) -> None:
    async with AioSqliteStorage(db_path) as storage:
        with pytest.raises(RuntimeError):
            async with storage.transaction():
                conv = ConversationRecord(conversation_id="conv-roll", agent_config_json="{}")
                await storage.insert_conversation(conv)
                raise RuntimeError("simulated error during transaction")

        persisted = await storage.get_conversation("conv-roll")
        assert persisted is None


@pytest.mark.asyncio
async def test_concurrent_read_during_write(db_path: str) -> None:
    storage = AioSqliteStorage(db_path)
    await storage.init_db()

    conv = ConversationRecord(conversation_id="conv-initial", agent_config_json="{}")
    await storage.insert_conversation(conv)

    read_successful = False

    async with storage.transaction():
        uncommitted_conv = ConversationRecord(
            conversation_id="conv-uncommitted", agent_config_json="{}"
        )
        await storage.insert_conversation(uncommitted_conv)

        reader = await storage.create_reader_connection()
        try:
            async with reader.execute("SELECT count(*) FROM conversations") as cursor:
                row = await cursor.fetchone()
                assert row is not None
                assert row[0] == 1
                read_successful = True
        finally:
            await reader.close()

    assert read_successful
    await storage.close()


@pytest.mark.asyncio
async def test_checkpoint_vacuum_and_integrity(db_path: str) -> None:
    async with AioSqliteStorage(db_path) as storage:
        conv = ConversationRecord(conversation_id="conv-maint", agent_config_json="{}")
        await storage.insert_conversation(conv)

        ckpt = await storage.checkpoint()
        assert ckpt[0] == 0

        await storage.vacuum()
        ok = await storage.check_integrity()
        assert ok is True
