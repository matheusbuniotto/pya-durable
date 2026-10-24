import asyncio
import json
import logging
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.test import TestModel

from pya_durable.compaction import (
    BackgroundCompactor,
    create_handoff_tool,
    create_search_history_tool,
)
from pya_durable.docs import (
    DocDefinition,
    DocumentManager,
    define_doc,
)
from pya_durable.harness import DurableHarness
from pya_durable.models import (
    ConversationRecord,
    DurableDocRecord,
    TranscriptEntryKind,
    TranscriptEntryRecord,
)
from pya_durable.storage import AioSqliteStorage
from pya_durable.streaming import StreamEvent, stream_hub


class PlanV1(BaseModel):
    title: str
    items: list[str] = []


class PlanV2(BaseModel):
    title: str
    items: list[str] = []
    status: str = "in_progress"


@pytest.mark.asyncio
async def test_document_persistence_and_version_migration(tmp_path: Path) -> None:
    db_file = tmp_path / "test_doc_mig.db"
    async with AioSqliteStorage(db_file) as storage:
        doc_def_v1 = define_doc("project_plan", PlanV1, version=1, initial=lambda: PlanV1(title="Sprint 1"))
        doc_v1 = await DocumentManager.get_doc(storage, doc_def_v1, "conv-1", doc_id="plan-1")
        assert doc_v1.title == "Sprint 1"

        doc_v1.items.append("Task 1")
        await DocumentManager.set_doc(storage, doc_def_v1, "conv-1", doc_v1, doc_id="plan-1")

        def migrate_v1_to_v2(v: int, d: dict[str, Any]) -> dict[str, Any]:
            if v == 1:
                d["status"] = "migrated"
            return d

        doc_def_v2 = define_doc(
            "project_plan",
            PlanV2,
            version=2,
            initial=lambda: PlanV2(title="Default"),
            migrate=migrate_v1_to_v2,
        )
        doc_v2 = await DocumentManager.get_doc(storage, doc_def_v2, "conv-1", doc_id="plan-1")
        assert doc_v2.title == "Sprint 1"
        assert doc_v2.items == ["Task 1"]
        assert doc_v2.status == "migrated"

        raw = await storage.get_durable_doc("plan-1", "conv-1")
        assert raw is not None
        assert raw.version == 2


@pytest.mark.asyncio
async def test_document_rollback_on_tool_failure(tmp_path: Path) -> None:
    db_file = tmp_path / "test_doc_rollback.db"
    async with AioSqliteStorage(db_file) as storage:
        doc_def = define_doc("counter", PlanV1, version=1, initial=lambda: PlanV1(title="Baseline"))
        await DocumentManager.set_doc(storage, doc_def, "conv-rb", PlanV1(title="Original"), doc_id="cnt")

        with pytest.raises(RuntimeError):
            async with storage.transaction() as conn:
                await DocumentManager.set_doc(
                    storage,
                    doc_def,
                    "conv-rb",
                    PlanV1(title="Mutated"),
                    doc_id="cnt",
                    conn=conn,
                )
                raise RuntimeError("Simulated tool crash mid-execution")

        retained = await DocumentManager.get_doc(storage, doc_def, "conv-rb", doc_id="cnt")
        assert retained.title == "Original"


@pytest.mark.asyncio
async def test_auto_compaction_and_summary_checkpoint(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    db_file = tmp_path / "test_compaction.db"
    async with AioSqliteStorage(db_file) as storage:
        agent = Agent(TestModel())
        harness = DurableHarness(
            agent,
            storage,
            conversation_id="conv-compact",
            compaction_threshold=5,
            auto_compact=True,
        )

        res = await harness.run("Execute turn 1")
        assert res is not None
        cutoff = await BackgroundCompactor.compact_conversation(
            storage,
            "conv-compact",
            res.leaf_entry_id,
            reserve_recent_turns=1,
        )
        assert cutoff != ""
        assert "Compacted turns up to entry id" in caplog.text

        summary_doc = await storage.get_durable_doc("compaction_summary", "conv-compact")
        assert summary_doc is not None
        data = json.loads(summary_doc.data_json)
        assert "cutoff_entry_id" in data


@pytest.mark.asyncio
async def test_compaction_boundary_safety(tmp_path: Path) -> None:
    db_file = tmp_path / "test_boundary.db"
    async with AioSqliteStorage(db_file) as storage:
        await storage.insert_conversation(
            ConversationRecord(conversation_id="conv-bound", agent_config_json="{}")
        )
        u1 = TranscriptEntryRecord(
            entry_id="e-u1",
            conversation_id="conv-bound",
            kind=TranscriptEntryKind.USER,
            message_json=json.dumps({"prompt": "p1"}),
        )
        r1 = TranscriptEntryRecord(
            entry_id="e-r1",
            conversation_id="conv-bound",
            parent_entry_id="e-u1",
            kind=TranscriptEntryKind.MODEL_RESPONSE,
            message_json=json.dumps({"text": "done"}),
        )
        u2 = TranscriptEntryRecord(
            entry_id="e-u2",
            conversation_id="conv-bound",
            parent_entry_id="e-r1",
            kind=TranscriptEntryKind.USER,
            message_json=json.dumps({"prompt": "p2"}),
        )
        tc2 = TranscriptEntryRecord(
            entry_id="e-tc2",
            conversation_id="conv-bound",
            parent_entry_id="e-u2",
            kind=TranscriptEntryKind.TOOL_CALL,
            message_json=json.dumps({"tool": "write_file"}),
        )
        for e in [u1, r1, u2, tc2]:
            await storage.insert_transcript_entry(e)

        cutoff = await BackgroundCompactor.compact_conversation(
            storage,
            "conv-bound",
            "e-tc2",
            reserve_recent_turns=1,
        )
        assert cutoff != "e-tc2"


@pytest.mark.asyncio
async def test_context_handoff_tool(tmp_path: Path) -> None:
    db_file = tmp_path / "test_handoff.db"
    async with AioSqliteStorage(db_file) as storage:
        agent = Agent(TestModel())
        harness = DurableHarness(agent, storage, conversation_id="conv-handoff")
        handoff_tool = create_handoff_tool(storage, "conv-handoff", harness=harness)
        agent.tool_plain(handoff_tool)

        await harness.run("first old turn")

        res_tool = await handoff_tool("Switching to feature optimization")
        assert "Handoff saved" in res_tool

        note_doc = await storage.get_durable_doc("handoff_note", "conv-handoff")
        assert note_doc is not None
        assert "Switching to feature optimization" in note_doc.data_json

        history = await harness.get_history()
        assert len(history) == 0


@pytest.mark.asyncio
async def test_search_history_tool(tmp_path: Path) -> None:
    db_file = tmp_path / "test_search.db"
    async with AioSqliteStorage(db_file) as storage:
        agent = Agent(TestModel())
        harness = DurableHarness(agent, storage, conversation_id="conv-search")
        await harness.run("Secret token is XYZ-9988-ALPHA")

        search_tool = create_search_history_tool(storage, "conv-search")
        found = await search_tool("XYZ-9988")
        assert "XYZ-9988-ALPHA" in found

        not_found = await search_tool("nonexistent-needle")
        assert "No archived messages matched query" in not_found


@pytest.mark.asyncio
async def test_realtime_state_streaming_watch(tmp_path: Path) -> None:
    db_file = tmp_path / "test_stream.db"
    async with AioSqliteStorage(db_file) as storage:
        agent = Agent(TestModel())
        harness = DurableHarness(agent, storage, conversation_id="conv-stream")

        events_received: list[StreamEvent] = []

        async def watch_collector() -> None:
            async for ev in harness.watch():
                events_received.append(ev)
                if len(events_received) >= 4:
                    break

        collector_task = asyncio.create_task(watch_collector())
        await asyncio.sleep(0.01)

        await harness.run("trigger stream nodes")
        await asyncio.wait_for(collector_task, timeout=2.0)

        assert len(events_received) >= 4
        node_names = [e.data.get("node_name") for e in events_received]
        assert "UserPromptNode" in node_names


@pytest.mark.asyncio
async def test_multiple_subscribers_receive_identical_events(tmp_path: Path) -> None:
    db_file = tmp_path / "test_multi_stream.db"
    async with AioSqliteStorage(db_file) as storage:
        agent = Agent(TestModel())
        harness = DurableHarness(agent, storage, conversation_id="conv-multi-stream")

        sub1_events: list[str] = []
        sub2_events: list[str] = []

        async def sub1() -> None:
            async for ev in harness.watch():
                sub1_events.append(ev.data.get("node_name", ""))
                if len(sub1_events) >= 3:
                    break

        async def sub2() -> None:
            async for ev in harness.watch():
                sub2_events.append(ev.data.get("node_name", ""))
                if len(sub2_events) >= 3:
                    break

        t1 = asyncio.create_task(sub1())
        t2 = asyncio.create_task(sub2())
        await asyncio.sleep(0.01)

        await harness.run("broadcast turn")
        await asyncio.wait_for(asyncio.gather(t1, t2), timeout=2.0)

        assert len(sub1_events) >= 3
        assert sub1_events == sub2_events


@pytest.mark.asyncio
async def test_streaming_client_disconnect_without_worker_crash(tmp_path: Path) -> None:
    db_file = tmp_path / "test_disconnect.db"
    async with AioSqliteStorage(db_file) as storage:
        agent = Agent(TestModel())
        harness = DurableHarness(agent, storage, conversation_id="conv-disconnect")

        async def flaky_subscriber() -> None:
            async for ev in harness.watch():
                raise ConnectionResetError("Client dropped connection abruptly")

        t = asyncio.create_task(flaky_subscriber())
        await asyncio.sleep(0.01)

        res = await harness.run("run through disconnect")
        assert res is not None

        with pytest.raises(ConnectionResetError):
            await t


@pytest.mark.asyncio
async def test_compaction_perf_under_thirty_ms(tmp_path: Path) -> None:
    db_file = tmp_path / "test_perf.db"
    async with AioSqliteStorage(db_file) as storage:
        conv_id = "conv-perf-compaction"
        await storage.insert_conversation(
            ConversationRecord(conversation_id=conv_id, agent_config_json="{}")
        )
        prev: str | None = None
        leaf: str = ""
        for i in range(150):
            eid = f"e-{i:03d}"
            kind = TranscriptEntryKind.USER if i % 3 == 0 else TranscriptEntryKind.MODEL_RESPONSE
            await storage.insert_transcript_entry(
                TranscriptEntryRecord(
                    entry_id=eid,
                    conversation_id=conv_id,
                    parent_entry_id=prev,
                    kind=kind,
                    message_json=json.dumps({"idx": i, "text": "turn body payload"}),
                )
            )
            prev = eid
            leaf = eid

        t0 = time.perf_counter()
        cutoff = await BackgroundCompactor.compact_conversation(
            storage,
            conv_id,
            leaf,
            reserve_recent_turns=3,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000

        assert cutoff != ""
        assert elapsed_ms < 30.0
