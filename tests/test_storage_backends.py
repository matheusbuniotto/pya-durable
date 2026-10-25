import json
from pathlib import Path
import tempfile
from typing import Any

import pytest
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from pya_durable.docs import DocumentManager, define_doc
from pya_durable.harness import DurableHarness
from pya_durable.jsonl_storage import JsonlStorage
from pya_durable.memo import MemoManager
from pya_durable.memory_storage import MemoryStorage
from pya_durable.models import (
    ConversationRecord,
    DurableDocRecord,
    TaskRecord,
    TaskStatus,
    ToolInvocationStatus,
    TranscriptEntryKind,
    TranscriptEntryRecord,
    utc_now_iso,
)


class SampleDoc(BaseModel):
    title: str = "Initial"
    counter: int = 0


sample_doc_def = define_doc("sample", SampleDoc, version=1)


@pytest.mark.asyncio
async def test_memory_storage_conversation_and_transcript_crud() -> None:
    storage = MemoryStorage()
    await storage.init_db()

    conv = ConversationRecord(
        conversation_id="conv-mem-1",
        agent_config_json='{"model": "test"}',
    )
    await storage.insert_conversation(conv)

    fetched = await storage.get_conversation("conv-mem-1")
    assert fetched is not None
    assert fetched.conversation_id == "conv-mem-1"

    entry1 = TranscriptEntryRecord(
        entry_id="e1",
        conversation_id="conv-mem-1",
        parent_entry_id=None,
        kind=TranscriptEntryKind.USER,
        message_json='{"content": "hello"}',
    )
    entry2 = TranscriptEntryRecord(
        entry_id="e2",
        conversation_id="conv-mem-1",
        parent_entry_id="e1",
        kind=TranscriptEntryKind.MODEL_RESPONSE,
        message_json='{"content": "world"}',
    )
    await storage.insert_transcript_entry(entry1)
    await storage.insert_transcript_entry(entry2)

    t1 = await storage.get_transcript_entry("e1")
    assert t1 is not None
    assert t1.entry_id == "e1"

    history = await storage.get_conversation_transcript("conv-mem-1")
    assert len(history) == 2
    assert history[0].entry_id == "e1"
    assert history[1].entry_id == "e2"

    ancestors = await storage.get_ancestor_entries("e2")
    assert [a.entry_id for a in ancestors] == ["e1", "e2"]

    await storage.close()


@pytest.mark.asyncio
async def test_memory_storage_tasks_and_status_filtering() -> None:
    storage = MemoryStorage()
    await storage.init_db()

    task1 = TaskRecord(
        task_id="t1",
        task_type="turn",
        status=TaskStatus.RUNNING,
        owner_kind="conversation",
        owner_id="c1",
        phase="phase1",
        input_json="{}",
        state_json="{}",
    )
    task2 = TaskRecord(
        task_id="t2",
        task_type="turn",
        status=TaskStatus.COMPLETED,
        owner_kind="conversation",
        owner_id="c1",
        phase="phase2",
        input_json="{}",
        state_json="{}",
    )
    await storage.insert_task(task1)
    await storage.insert_task(task2)

    running = await storage.get_running_tasks()
    assert len(running) == 1
    assert running[0].task_id == "t1"

    await storage.update_task_state("t1", phase="phase_done", state_json='{"done": true}', status=TaskStatus.COMPLETED)
    t1_updated = await storage.get_task("t1")
    assert t1_updated is not None
    assert t1_updated.status == TaskStatus.COMPLETED
    assert t1_updated.phase == "phase_done"

    await storage.close()


@pytest.mark.asyncio
async def test_memory_storage_tool_invocations_and_reconciliation() -> None:
    storage = MemoryStorage()
    await storage.init_db()

    await storage.record_tool_started(
        invocation_id="inv-safe",
        task_id="t1",
        conversation_id="c1",
        tool_name="read_file",
        args_json='{"path": "a.txt"}',
        replay_policy="safe",
    )
    await storage.record_tool_started(
        invocation_id="inv-unsafe",
        task_id="t1",
        conversation_id="c1",
        tool_name="write_file",
        args_json='{"path": "a.txt"}',
        replay_policy="unsafe",
    )

    safe_count, unsafe_count = await storage.reconcile_started_tool_invocations()
    assert safe_count == 1
    assert unsafe_count == 1

    inv_unsafe = await storage.get_tool_invocation("inv-unsafe")
    assert inv_unsafe is not None
    assert inv_unsafe.status == ToolInvocationStatus.INTERRUPTED

    inv_safe = await storage.get_tool_invocation("inv-safe")
    assert inv_safe is not None
    assert inv_safe.status == ToolInvocationStatus.STARTED

    await storage.record_tool_completed("inv-safe", '{"ok": true}')
    inv_safe_done = await storage.get_tool_invocation("inv-safe")
    assert inv_safe_done is not None
    assert inv_safe_done.status == ToolInvocationStatus.COMPLETED

    await storage.close()


@pytest.mark.asyncio
async def test_memory_storage_memos_first_write_wins() -> None:
    storage = MemoryStorage()
    await storage.init_db()

    val1 = await MemoManager.memo(storage, "conv-mem", "decision:cache", "result-A")
    assert val1 == "result-A"

    val2 = await MemoManager.memo(storage, "conv-mem", "decision:cache", "result-B")
    assert val2 == "result-A"

    memo = await storage.get_memo("conv-mem", "decision:cache")
    assert memo is not None
    assert json.loads(memo.value_json) == "result-A"

    deleted = await storage.delete_memo("conv-mem", "decision:cache")
    assert deleted is True

    memo_after = await storage.get_memo("conv-mem", "decision:cache")
    assert memo_after is None

    await storage.close()


@pytest.mark.asyncio
async def test_jsonl_storage_persistence_and_file_content() -> None:
    with tempfile.TemporaryDirectory() as td:
        jsonl_path = Path(td) / "audit.jsonl"
        storage = JsonlStorage(jsonl_path)
        await storage.init_db()

        conv = ConversationRecord(
            conversation_id="c-jsonl-1",
            agent_config_json='{"model": "test"}',
        )
        await storage.insert_conversation(conv)

        entry = TranscriptEntryRecord(
            entry_id="e-j1",
            conversation_id="c-jsonl-1",
            parent_entry_id=None,
            kind=TranscriptEntryKind.USER,
            message_json='{"content": "jsonl test"}',
        )
        await storage.insert_transcript_entry(entry)

        await storage.insert_memo("c-jsonl-1", "mkey", '"mval"')

        doc = DurableDocRecord(
            doc_id="d1",
            conversation_id="c-jsonl-1",
            version=1,
            doc_type="sample",
            data_json='{"title": "doc1"}',
            updated_at=utc_now_iso(),
        )
        await storage.upsert_durable_doc(doc)

        assert jsonl_path.exists()
        lines = jsonl_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 4

        entities = [json.loads(line)["entity"] for line in lines]
        assert entities == ["conversation", "transcript_entry", "memo", "doc"]

        await storage.close()


@pytest.mark.asyncio
async def test_jsonl_storage_crash_recovery_across_reopen() -> None:
    with tempfile.TemporaryDirectory() as td:
        jsonl_path = Path(td) / "crash.jsonl"
        storage1 = JsonlStorage(jsonl_path)
        await storage1.init_db()

        conv = ConversationRecord(
            conversation_id="conv-recover",
            agent_config_json='{"mode": "crash"}',
        )
        await storage1.insert_conversation(conv)

        entry1 = TranscriptEntryRecord(
            entry_id="e1",
            conversation_id="conv-recover",
            parent_entry_id=None,
            kind=TranscriptEntryKind.USER,
            message_json='{"text": "step 1"}',
        )
        entry2 = TranscriptEntryRecord(
            entry_id="e2",
            conversation_id="conv-recover",
            parent_entry_id="e1",
            kind=TranscriptEntryKind.MODEL_RESPONSE,
            message_json='{"text": "step 2"}',
        )
        await storage1.insert_transcript_entry(entry1)
        await storage1.insert_transcript_entry(entry2)

        task = TaskRecord(
            task_id="t-crash",
            task_type="turn",
            status=TaskStatus.RUNNING,
            owner_kind="conversation",
            owner_id="conv-recover",
            phase="exec",
            input_json='{"p": 1}',
            state_json='{"s": 1}',
        )
        await storage1.insert_task(task)
        await storage1.insert_memo("conv-recover", "cached_decision", "123")
        await storage1.close()

        # Simulate host reboot by instantiating new JsonlStorage over same file
        storage2 = JsonlStorage(jsonl_path)
        await storage2.init_db()

        c_recovered = await storage2.get_conversation("conv-recover")
        assert c_recovered is not None
        assert c_recovered.conversation_id == "conv-recover"

        transcript = await storage2.get_conversation_transcript("conv-recover")
        assert len(transcript) == 2
        assert transcript[0].entry_id == "e1"
        assert transcript[1].entry_id == "e2"

        ancestors = await storage2.get_ancestor_entries("e2")
        assert [a.entry_id for a in ancestors] == ["e1", "e2"]

        task_recovered = await storage2.get_task("t-crash")
        assert task_recovered is not None
        assert task_recovered.task_id == "t-crash"
        assert task_recovered.status == TaskStatus.RUNNING

        memo_recovered = await storage2.get_memo("conv-recover", "cached_decision")
        assert memo_recovered is not None
        assert memo_recovered.value_json == "123"

        await storage2.close()


@pytest.mark.asyncio
async def test_durable_harness_end_to_end_with_memory_storage() -> None:
    storage = MemoryStorage()
    agent = Agent[Any, str](TestModel(custom_output_text="Memory storage turn complete"))
    harness = DurableHarness(agent=agent, storage=storage, conversation_id="conv-mem-harness")

    result = await harness.submit("Run test prompt with MemoryStorage")
    assert result is not None
    assert result.output == "Memory storage turn complete"

    history = await harness.get_history()
    assert len(history) >= 2

    # Verify reactive view state snapshot
    vs = await harness.view_state()
    assert vs.value.conversation_id == "conv-mem-harness"
    assert vs.value.status == "idle"

    # Verify durable document operation
    await harness.set_doc(sample_doc_def, SampleDoc(title="Memory Plan", counter=42))
    doc_state = await harness.document_state(sample_doc_def)
    assert doc_state.value.title == "Memory Plan"
    assert doc_state.value.counter == 42

    await storage.close()


@pytest.mark.asyncio
async def test_durable_harness_end_to_end_with_jsonl_storage() -> None:
    with tempfile.TemporaryDirectory() as td:
        jsonl_path = Path(td) / "harness_audit.jsonl"
        storage = JsonlStorage(jsonl_path)
        agent = Agent[Any, str](TestModel(custom_output_text="Jsonl storage turn complete"))
        harness = DurableHarness(agent=agent, storage=storage, conversation_id="conv-jsonl-harness")

        result = await harness.submit("Run test prompt with JsonlStorage")
        assert result is not None
        assert result.output == "Jsonl storage turn complete"

        history = await harness.get_history()
        assert len(history) >= 2

        # Verify documents
        await harness.set_doc(sample_doc_def, SampleDoc(title="Jsonl Plan", counter=99))
        doc_state = await harness.document_state(sample_doc_def)
        assert doc_state.value.title == "Jsonl Plan"
        assert doc_state.value.counter == 99

        await storage.close()

        # Reopen from disk and verify transcript and doc are preserved
        storage_reopened = JsonlStorage(jsonl_path)
        await storage_reopened.init_db()

        harness_reopened = DurableHarness(
            agent=agent,
            storage=storage_reopened,
            conversation_id="conv-jsonl-harness",
        )
        recovered_history = await harness_reopened.get_history()
        assert len(recovered_history) == len(history)

        recovered_doc = await DocumentManager.get_doc(
            storage_reopened,
            sample_doc_def,
            "conv-jsonl-harness",
        )
        assert recovered_doc.title == "Jsonl Plan"
        assert recovered_doc.counter == 99

        await storage_reopened.close()


@pytest.mark.asyncio
async def test_conversation_fork_with_memory_storage() -> None:
    storage = MemoryStorage()
    agent = Agent[Any, str](TestModel(custom_output_text="Branch turn answer"))
    harness = DurableHarness(agent=agent, storage=storage, conversation_id="conv-root-mem")

    turn = await harness.submit("Root message")
    assert turn is not None

    child = await harness.fork(
        fork_entry_id=turn.leaf_entry_id,
        new_conversation_id="conv-child-mem",
    )
    assert isinstance(child, DurableHarness)
    assert child.conversation_id == "conv-child-mem"
    assert child.parent_conversation_id == "conv-root-mem"

    child_history = await child.get_history()
    assert len(child_history) >= 2

    # Submit child prompt
    child_turn = await child.submit("Child branch inquiry")
    assert child_turn is not None
    assert child_turn.output == "Branch turn answer"

    await storage.close()


@pytest.mark.asyncio
async def test_conversation_fork_with_jsonl_storage() -> None:
    with tempfile.TemporaryDirectory() as td:
        jsonl_path = Path(td) / "fork_audit.jsonl"
        storage = JsonlStorage(jsonl_path)
        agent = Agent[Any, str](TestModel(custom_output_text="Fork response"))
        harness = DurableHarness(agent=agent, storage=storage, conversation_id="conv-root-jsonl")

        turn = await harness.submit("Root message in jsonl")
        assert turn is not None

        child = await harness.fork(
            fork_entry_id=turn.leaf_entry_id,
            new_conversation_id="conv-child-jsonl",
        )
        assert child.conversation_id == "conv-child-jsonl"

        child_turn = await child.submit("Child turn in jsonl")
        assert child_turn is not None
        assert child_turn.output == "Fork response"

        await storage.close()

        # Reopen and check fork lineage
        storage_reopened = JsonlStorage(jsonl_path)
        await storage_reopened.init_db()

        conv_child = await storage_reopened.get_conversation("conv-child-jsonl")
        assert conv_child is not None
        assert conv_child.parent_conversation_id == "conv-root-jsonl"

        await storage_reopened.close()
