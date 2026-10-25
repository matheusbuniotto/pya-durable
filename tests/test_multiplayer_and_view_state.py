import asyncio
from pathlib import Path
import tempfile
from typing import Any

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from pya_durable.docs import define_doc
from pya_durable.harness import DurableHarness
from pya_durable.memo import MemoManager
from pya_durable.storage import AioSqliteStorage
from pya_durable.streaming import (
    ClientPresenceEvent,
    DocUpdatedEvent,
    MultiplayerPresenceHub,
    ReactiveViewState,
    TurnCompletedEvent,
    TurnStartedEvent,
)


class PlanDoc(BaseModel):
    title: str = "Root Plan"
    done: bool = False


plan_doc_def = define_doc("plan", PlanDoc, version=1)


async def test_memo_manager_first_write_wins() -> None:
    with tempfile.TemporaryDirectory() as td:
        storage = AioSqliteStorage(Path(td) / "test.db")
        await storage.init_db()

        val1 = await MemoManager.memo(storage, "conv-1", "approval:deploy", True)
        assert val1 is True

        val2 = await MemoManager.memo(storage, "conv-1", "approval:deploy", False)
        assert val2 is True

        persisted = await storage.get_memo("conv-1", "approval:deploy")
        assert persisted is not None
        assert persisted.value_json == "true"
        await storage.close()


async def test_memo_across_reopen() -> None:
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "memo.db"
        storage = AioSqliteStorage(db_path)
        await storage.init_db()
        await MemoManager.memo(storage, "conv-1", "cache_key", {"status": "ok"})
        await storage.close()

        storage2 = AioSqliteStorage(db_path)
        await storage2.init_db()
        cached = await MemoManager.memo(storage2, "conv-1", "cache_key", {"status": "replaced"})
        assert cached == {"status": "ok"}
        await storage2.close()


async def test_view_state_initial_snapshot() -> None:
    with tempfile.TemporaryDirectory() as td:
        storage = AioSqliteStorage(Path(td) / "test.db")
        agent = Agent[Any, str](TestModel(custom_output_text="Echo response"))
        harness = DurableHarness(agent=agent, storage=storage, conversation_id="conv-snap")

        await harness.set_doc(plan_doc_def, PlanDoc(title="Feature PR-09", done=True))
        harness.connect_client("operator-alice")

        vs = await harness.view_state()
        assert vs.value.conversation_id == "conv-snap"
        assert vs.value.status == "idle"
        assert "operator-alice" in vs.value.connected_clients
        assert "plan:default" in vs.value.docs
        assert vs.value.docs["plan:default"]["title"] == "Feature PR-09"
        assert vs.value.docs["plan:default"]["done"] is True
        await storage.close()


async def test_view_state_reactive_subscription() -> None:
    with tempfile.TemporaryDirectory() as td:
        storage = AioSqliteStorage(Path(td) / "test.db")
        agent = Agent[Any, str](TestModel(custom_output_text="Turn done"))
        harness = DurableHarness(agent=agent, storage=storage, conversation_id="conv-sub")

        vs = await harness.view_state()
        received_states = []

        unsubscribe = vs.subscribe(lambda state: received_states.append(state))

        await harness.submit("Run test step")

        assert len(received_states) >= 1
        latest = received_states[-1]
        assert latest.status == "idle"
        assert len(latest.entries) > 0

        unsubscribe()
        prior_count = len(received_states)
        await harness.submit("Second turn")
        assert len(received_states) == prior_count
        await storage.close()


async def test_view_state_async_watch() -> None:
    with tempfile.TemporaryDirectory() as td:
        storage = AioSqliteStorage(Path(td) / "test.db")
        agent = Agent[Any, str](TestModel(custom_output_text="Watch test"))
        harness = DurableHarness(agent=agent, storage=storage, conversation_id="conv-watch")

        vs = await harness.view_state()

        async def watcher() -> list[str]:
            statuses = []
            async for s in vs.watch():
                statuses.append(s.status)
                if len(statuses) >= 2:
                    break
            return statuses

        task = asyncio.create_task(watcher())
        await asyncio.sleep(0.01)
        await harness.submit("Hello watcher")
        statuses = await asyncio.wait_for(task, timeout=2.0)
        assert len(statuses) >= 2
        vs.close()
        await storage.close()


async def test_high_level_agent_event_translation() -> None:
    with tempfile.TemporaryDirectory() as td:
        storage = AioSqliteStorage(Path(td) / "test.db")
        agent = Agent[Any, str](TestModel(custom_output_text="Done agent event"))
        harness = DurableHarness(agent=agent, storage=storage, conversation_id="conv-events")

        collected_events = []

        async def listen() -> None:
            async for evt in harness.watch_events():
                collected_events.append(evt)
                if isinstance(evt, TurnCompletedEvent):
                    break

        listener_task = asyncio.create_task(listen())
        await asyncio.sleep(0.02)

        await harness.submit("Start high-level event turn")
        await asyncio.wait_for(listener_task, timeout=2.0)

        types = [type(e) for e in collected_events]
        assert TurnStartedEvent in types
        assert TurnCompletedEvent in types
        await storage.close()


async def test_reactive_document_state() -> None:
    with tempfile.TemporaryDirectory() as td:
        storage = AioSqliteStorage(Path(td) / "test.db")
        agent = Agent[Any, str](TestModel(custom_output_text="doc test"))
        harness = DurableHarness(agent=agent, storage=storage, conversation_id="conv-doc")

        doc_state = await harness.document_state(plan_doc_def)
        assert doc_state.value.title == "Root Plan"
        assert doc_state.value.done is False

        updated_values = []
        doc_state.subscribe(lambda val: updated_values.append(val))

        await harness.set_doc(plan_doc_def, PlanDoc(title="Updated Plan", done=True))
        await asyncio.sleep(0.02)

        assert len(updated_values) >= 1
        assert updated_values[-1].title == "Updated Plan"
        assert updated_values[-1].done is True
        assert doc_state.value.title == "Updated Plan"
        await storage.close()


async def test_multiplayer_presence_join_and_leave() -> None:
    with tempfile.TemporaryDirectory() as td:
        storage = AioSqliteStorage(Path(td) / "test.db")
        agent = Agent[Any, str](TestModel(custom_output_text="presence test"))
        harness = DurableHarness(agent=agent, storage=storage, conversation_id="conv-pres")

        collected = []

        async def listen() -> None:
            async for evt in harness.watch_events():
                if isinstance(evt, ClientPresenceEvent):
                    collected.append(evt)
                    if len(collected) >= 2:
                        break

        listener = asyncio.create_task(listen())
        await asyncio.sleep(0.02)

        harness.connect_client("client-bob")
        assert "client-bob" in harness.get_connected_clients()

        harness.disconnect_client("client-bob")
        assert "client-bob" not in harness.get_connected_clients()

        await asyncio.wait_for(listener, timeout=2.0)
        assert collected[0].client_id == "client-bob"
        assert collected[0].action == "joined"
        assert collected[1].client_id == "client-bob"
        assert collected[1].action == "left"
        await storage.close()


async def test_multiplayer_isolated_presence_across_conversations() -> None:
    hub = MultiplayerPresenceHub()
    hub.connect("conv-1", "user-alice")
    hub.connect("conv-2", "user-bob")

    assert hub.get_connected_clients("conv-1") == ["user-alice"]
    assert hub.get_connected_clients("conv-2") == ["user-bob"]

    hub.disconnect("conv-1", "user-alice")
    assert hub.get_connected_clients("conv-1") == []
    assert hub.get_connected_clients("conv-2") == ["user-bob"]


async def test_conversation_view_state_propagation_across_fork() -> None:
    with tempfile.TemporaryDirectory() as td:
        storage = AioSqliteStorage(Path(td) / "test.db")
        agent = Agent[Any, str](TestModel(custom_output_text="Fork response"))
        harness = DurableHarness(agent=agent, storage=storage, conversation_id="conv-parent")

        turn = await harness.submit("Parent turn")
        assert turn is not None

        child = await harness.fork(
            fork_entry_id=turn.leaf_entry_id,
            new_conversation_id="conv-child",
        )

        harness.connect_client("parent-client")
        child.connect_client("child-client")

        vs_parent = await harness.view_state()
        vs_child = await child.view_state()

        assert vs_parent.value.conversation_id == "conv-parent"
        assert "parent-client" in vs_parent.value.connected_clients
        assert "child-client" not in vs_parent.value.connected_clients

        assert vs_child.value.conversation_id == "conv-child"
        assert "child-client" in vs_child.value.connected_clients
        assert "parent-client" not in vs_child.value.connected_clients
        await storage.close()


async def test_subscriber_error_isolation() -> None:
    vs = ReactiveViewState(
        initial_state=type("MockState", (), {"version": 1, "conversation_id": "c1"})()  # type: ignore[arg-type]
    )

    good_received = []

    def failing_cb(s: Any) -> None:
        raise RuntimeError("Subscriber explosion")

    def good_cb(s: Any) -> None:
        good_received.append(s)

    vs.subscribe(failing_cb)
    vs.subscribe(good_cb)

    new_state = type("MockState", (), {"version": 2, "conversation_id": "c1"})()
    vs.notify(new_state)  # type: ignore[arg-type]

    assert len(good_received) == 1
    assert good_received[0].version == 2
