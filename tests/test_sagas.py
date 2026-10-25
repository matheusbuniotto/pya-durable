import asyncio
import json
from pathlib import Path

import pytest

from pydantic_ai_durable.jsonl_storage import JsonlStorage
from pydantic_ai_durable.memory_storage import MemoryStorage
from pydantic_ai_durable.models import (
    ConversationRecord,
    SagaPolicy,
    TaskRecord,
    TaskStatus,
)
from pydantic_ai_durable.recovery import CrashRecoveryManager
from pydantic_ai_durable.replay import (
    create_durable_task,
    durable_execution_context,
    durable_tool,
    wait_for_durable_task,
)
from pydantic_ai_durable.sagas import (
    SagaContext,
    SagaCoordinator,
    SagaExecutionError,
    SagaResult,
    define_saga,
)
from pydantic_ai_durable.storage import AioSqliteStorage
from pydantic_ai_durable.streaming import stream_hub
from pydantic_ai_durable.task_tree import TaskPhaseResult, TaskRunner, define_task


@pytest.mark.asyncio
async def test_sequential_saga_execution(temp_db: str) -> None:
    async with AioSqliteStorage(temp_db) as storage:
        runner = TaskRunner(storage)
        coordinator = SagaCoordinator(storage, task_runner=runner)

        @define_saga(name="test.sequential_saga", initial_step="step_one")
        class SequentialSaga:
            async def step_one(self, ctx: SagaContext) -> SagaResult:
                ctx.state["val"] = ctx.input_data.get("base", 0) + 10
                return ctx.next_step("step_two")

            async def step_two(self, ctx: SagaContext) -> SagaResult:
                ctx.state["val"] *= 2
                return ctx.complete(result=ctx.state["val"])

        res = await coordinator.run_saga(SequentialSaga, {"base": 5})
        assert res == 30

        all_tasks = await storage.get_tasks_by_status(TaskStatus.COMPLETED)
        saga_task = next(t for t in all_tasks if t.task_type == "test.sequential_saga")
        assert saga_task.status == TaskStatus.COMPLETED
        state_data = json.loads(saga_task.state_json)
        assert state_data["executed_steps"] == ["step_one", "step_two"]
        assert state_data["output"] == 30


@pytest.mark.asyncio
async def test_parallel_fan_out_and_wait_on_children(temp_db: str) -> None:
    async with AioSqliteStorage(temp_db) as storage:
        runner = TaskRunner(storage)
        coordinator = SagaCoordinator(storage, task_runner=runner)

        worker_task = define_task(
            name="cluster.worker",
            phases={
                "start": lambda ctx: TaskPhaseResult(
                    status=TaskStatus.COMPLETED,
                    result=f"node-{ctx.input_data['node']}-ready",
                )
            },
        )
        runner.register_task(worker_task)

        @define_saga(name="cluster.provision", initial_step="spawn_nodes")
        class ClusterProvisionSaga:
            async def spawn_nodes(self, ctx: SagaContext) -> SagaResult:
                worker_ids = []
                for n in ctx.input_data.get("nodes", []):
                    child = await ctx.spawn(worker_task, {"node": n})
                    worker_ids.append(child.task_id)
                    asyncio.create_task(
                        runner.run_task(
                            worker_task,
                            {"node": n},
                            task_id=child.task_id,
                            owner_kind="task",
                            owner_id=ctx.task_id,
                        )
                    )
                return ctx.wait_on(worker_ids, next_step="aggregate", policy=SagaPolicy.FAIL_FAST)

            async def aggregate(self, ctx: SagaContext) -> SagaResult:
                outcomes = await ctx.outcomes()
                assert len(outcomes) == 3
                assert all(o.is_successful for o in outcomes.values())
                return ctx.complete(result=[o.result for o in outcomes.values()])

        res = await coordinator.run_saga(
            ClusterProvisionSaga, {"nodes": ["alpha", "beta", "gamma"]}
        )
        assert res == ["node-alpha-ready", "node-beta-ready", "node-gamma-ready"]


@pytest.mark.asyncio
async def test_fail_fast_compensation_cascade(temp_db: str) -> None:
    async with AioSqliteStorage(temp_db) as storage:
        runner = TaskRunner(storage)
        coordinator = SagaCoordinator(storage, task_runner=runner)

        compensations_run: list[str] = []

        success_worker = define_task(
            name="worker.success",
            phases={"start": lambda ctx: TaskPhaseResult(status=TaskStatus.COMPLETED, result="ok")},
        )
        failing_worker = define_task(
            name="worker.failure",
            phases={"start": lambda ctx: TaskPhaseResult(status=TaskStatus.FAILED, result="err")},
        )
        runner.register_task(success_worker)
        runner.register_task(failing_worker)

        @define_saga(name="failing.saga", initial_step="step_reserve")
        class FailingSaga:
            async def step_reserve(self, ctx: SagaContext) -> SagaResult:
                ctx.state["reserved"] = True
                c1 = await ctx.spawn(success_worker, {})
                c2 = await ctx.spawn(failing_worker, {})
                asyncio.create_task(runner.run_task(success_worker, {}, task_id=c1.task_id, owner_kind="task", owner_id=ctx.task_id))
                asyncio.create_task(runner.run_task(failing_worker, {}, task_id=c2.task_id, owner_kind="task", owner_id=ctx.task_id))
                return ctx.wait_on([c1.task_id, c2.task_id], next_step="finalize", policy=SagaPolicy.FAIL_FAST)

            async def compensate_step_reserve(self, ctx: SagaContext) -> None:
                compensations_run.append("step_reserve")

            async def finalize(self, ctx: SagaContext) -> SagaResult:
                return ctx.complete()

            async def abort(self, ctx: SagaContext) -> None:
                compensations_run.append("global_abort")

        with pytest.raises(SagaExecutionError, match="Saga failed during child execution"):
            await coordinator.run_saga(FailingSaga, {})

        assert "step_reserve" in compensations_run
        assert "global_abort" in compensations_run


@pytest.mark.asyncio
async def test_all_settled_policy_collects_mixed_outcomes(temp_db: str) -> None:
    async with AioSqliteStorage(temp_db) as storage:
        runner = TaskRunner(storage)
        coordinator = SagaCoordinator(storage, task_runner=runner)

        good_worker = define_task(
            name="good_worker",
            phases={"start": lambda ctx: TaskPhaseResult(status=TaskStatus.COMPLETED, result=10)},
        )
        bad_worker = define_task(
            name="bad_worker",
            phases={"start": lambda ctx: TaskPhaseResult(status=TaskStatus.FAILED, result=None)},
        )
        runner.register_task(good_worker)
        runner.register_task(bad_worker)

        @define_saga(name="settled.saga", initial_step="launch")
        class AllSettledSaga:
            async def launch(self, ctx: SagaContext) -> SagaResult:
                g = await ctx.spawn(good_worker, {})
                b = await ctx.spawn(bad_worker, {})
                asyncio.create_task(runner.run_task(good_worker, {}, task_id=g.task_id, owner_kind="task", owner_id=ctx.task_id))
                asyncio.create_task(runner.run_task(bad_worker, {}, task_id=b.task_id, owner_kind="task", owner_id=ctx.task_id))
                return ctx.wait_on([g.task_id, b.task_id], next_step="review", policy=SagaPolicy.ALL_SETTLED)

            async def review(self, ctx: SagaContext) -> SagaResult:
                outcomes = await ctx.outcomes()
                successful = sum(1 for o in outcomes.values() if o.is_successful)
                failed = sum(1 for o in outcomes.values() if not o.is_successful)
                return ctx.complete(result={"success": successful, "failed": failed})

        res = await coordinator.run_saga(AllSettledSaga, {})
        assert res == {"success": 1, "failed": 1}


@pytest.mark.asyncio
async def test_crash_recovery_for_waiting_saga(temp_db: str) -> None:
    async with AioSqliteStorage(temp_db) as storage:
        parent_id = "saga-waiting-reboot"
        child1_id = "child-done-1"
        child2_id = "child-done-2"

        parent_record = TaskRecord(
            task_id=parent_id,
            task_type="backup.cluster",
            version=1,
            status=TaskStatus.WAITING,
            owner_kind="root",
            owner_id="root",
            phase="waiting_tasks",
            input_json="{}",
            state_json=json.dumps({"state": {"cluster": "prod"}, "executed_steps": ["step_one"]}),
            checkpoint_json=json.dumps({
                "waiting_on": [child1_id, child2_id],
                "next_step": "finalize",
                "policy": "failFast",
                "executed_steps": ["step_one"],
            }),
        )
        c1 = TaskRecord(
            task_id=child1_id,
            task_type="node.backup",
            status=TaskStatus.COMPLETED,
            owner_kind="task",
            owner_id=parent_id,
            phase="completed",
            input_json="{}",
            state_json=json.dumps({"output": "backup-1-ok"}),
        )
        c2 = TaskRecord(
            task_id=child2_id,
            task_type="node.backup",
            status=TaskStatus.COMPLETED,
            owner_kind="task",
            owner_id=parent_id,
            phase="completed",
            input_json="{}",
            state_json=json.dumps({"output": "backup-2-ok"}),
        )

        await storage.insert_task(parent_record)
        await storage.insert_task(c1)
        await storage.insert_task(c2)

        recovery = CrashRecoveryManager(storage)
        result = await recovery.reconcile_task(parent_id)

        assert result.previous_status == TaskStatus.WAITING
        assert result.new_status == TaskStatus.RUNNING
        assert result.reconciled_phase == "resumed_tasks"

        updated_parent = await storage.get_task(parent_id)
        assert updated_parent is not None
        assert updated_parent.status == TaskStatus.RUNNING
        assert updated_parent.phase == "resumed_tasks"


@pytest.mark.asyncio
async def test_durable_tool_spawning_and_awaiting_tasks(temp_db: str) -> None:
    async with AioSqliteStorage(temp_db) as storage:
        runner = TaskRunner(storage)

        bg_task = define_task(
            name="async.worker",
            phases={"start": lambda ctx: TaskPhaseResult(status=TaskStatus.COMPLETED, result=42)},
        )
        runner.register_task(bg_task)

        @durable_tool(replay="safe")
        async def orchestrate_tool(val: int) -> int:
            child = await create_durable_task("async.worker", {"val": val})
            asyncio.create_task(runner.run_task(bg_task, {"val": val}, task_id=child.task_id))
            settled = await wait_for_durable_task(child.task_id)
            assert settled.status == TaskStatus.COMPLETED
            parsed = json.loads(settled.state_json)
            return parsed.get("output", 0)

            conv = ConversationRecord(conversation_id="conv-1", agent_config_json="{}")
            parent_task = TaskRecord(
                task_id="tool-parent-task",
                task_type="parent",
                status=TaskStatus.RUNNING,
                owner_kind="root",
                owner_id="root",
                phase="running",
                input_json="{}",
                state_json="{}",
            )
            await storage.insert_conversation(conv)
            await storage.insert_task(parent_task)

            async with durable_execution_context(storage, task_id="tool-parent-task", conversation_id="conv-1"):
                res = await orchestrate_tool(100)
                assert res == 42


@pytest.mark.asyncio
async def test_memory_storage_saga_parity() -> None:
    storage = MemoryStorage()
    await storage.init_db()
    try:
        runner = TaskRunner(storage)
        coordinator = SagaCoordinator(storage, task_runner=runner)

        @define_saga(name="mem.saga", initial_step="run")
        class MemSaga:
            async def run(self, ctx: SagaContext) -> SagaResult:
                return ctx.complete(result="mem-success")

        res = await coordinator.run_saga(MemSaga, {})
        assert res == "mem-success"

        tasks = await storage.get_tasks_by_status(TaskStatus.COMPLETED)
        assert any(t.task_type == "mem.saga" for t in tasks)
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_jsonl_storage_saga_parity(tmp_path: Path) -> None:
    log_file = str(tmp_path / "saga.jsonl")
    storage = JsonlStorage(log_file)
    await storage.init_db()
    try:
        runner = TaskRunner(storage)
        coordinator = SagaCoordinator(storage, task_runner=runner)

        @define_saga(name="jsonl.saga", initial_step="step_a")
        class JsonlSaga:
            async def step_a(self, ctx: SagaContext) -> SagaResult:
                return ctx.complete(result="jsonl-success")

        res = await coordinator.run_saga(JsonlSaga, {})
        assert res == "jsonl-success"
    finally:
        await storage.close()

    fresh_storage = JsonlStorage(log_file)
    await fresh_storage.init_db()
    try:
        tasks = await fresh_storage.get_tasks_by_status(TaskStatus.COMPLETED)
        assert any(t.task_type == "jsonl.saga" for t in tasks)
    finally:
        await fresh_storage.close()


@pytest.mark.asyncio
async def test_streaming_saga_events(temp_db: str) -> None:
    async with AioSqliteStorage(temp_db) as storage:
        runner = TaskRunner(storage)
        coordinator = SagaCoordinator(storage, task_runner=runner, stream_hub=stream_hub)

        received_events: list[str] = []

        async def listen() -> None:
            async for evt in stream_hub.subscribe_events("conv-saga-test"):
                received_events.append(evt.event_type)
                if evt.event_type == "saga_completed":
                    break

        listener_task = asyncio.create_task(listen())
        await asyncio.sleep(0.02)

        @define_saga(name="stream.saga", initial_step="phase_one")
        class StreamSaga:
            async def phase_one(self, ctx: SagaContext) -> SagaResult:
                return ctx.next_step("phase_two")

            async def phase_two(self, ctx: SagaContext) -> SagaResult:
                return ctx.complete(result="stream-done")

        await coordinator.run_saga(StreamSaga, {}, conversation_id="conv-saga-test")
        await asyncio.wait_for(listener_task, timeout=2.0)

        assert "saga_started" in received_events
        assert "saga_step_completed" in received_events
        assert "saga_completed" in received_events


@pytest.mark.asyncio
async def test_harness_and_conversation_saga_integration(temp_db: str) -> None:
    from pydantic_ai import Agent
    from pydantic_ai.models.test import TestModel

    from pydantic_ai_durable.harness import DurableHarness

    agent = Agent(TestModel())
    async with AioSqliteStorage(temp_db) as storage:
        harness = DurableHarness(agent=agent, storage=storage, conversation_id="conv-harness-saga")

        @define_saga(name="harness.saga", initial_step="step_init")
        class HarnessSaga:
            async def step_init(self, ctx: SagaContext) -> SagaResult:
                ctx.state["value"] = 55
                return ctx.complete(result=ctx.state["value"])

        res = await harness.run_saga(HarnessSaga, {})
        assert res == 55


@pytest.mark.asyncio
async def test_conversation_saga_integration(temp_db: str) -> None:
    from pydantic_ai import Agent
    from pydantic_ai.models.test import TestModel

    from pydantic_ai_durable.branching import DurableConversation

    agent = Agent(TestModel())
    async with AioSqliteStorage(temp_db) as storage:
        conv = DurableConversation(
            conversation_id="conv-branch-saga",
            storage=storage,
            agent=agent,
        )

        @define_saga(name="branch.saga", initial_step="step_start")
        class BranchSaga:
            async def step_start(self, ctx: SagaContext) -> SagaResult:
                return ctx.complete(result="branch-saga-ok")

        res = await conv.run_saga(BranchSaga, {})
        assert res == "branch-saga-ok"


@pytest.mark.asyncio
async def test_multi_step_compensation_reverse_order(temp_db: str) -> None:
    async with AioSqliteStorage(temp_db) as storage:
        runner = TaskRunner(storage)
        coordinator = SagaCoordinator(storage, task_runner=runner)

        order_log: list[str] = []

        @define_saga(name="reverse.order.saga", initial_step="step_a")
        class ReverseOrderSaga:
            async def step_a(self, ctx: SagaContext) -> SagaResult:
                return ctx.next_step("step_b")

            async def compensate_step_a(self, ctx: SagaContext) -> None:
                order_log.append("compensate_a")

            async def step_b(self, ctx: SagaContext) -> SagaResult:
                return ctx.next_step("step_c")

            async def compensate_step_b(self, ctx: SagaContext) -> None:
                order_log.append("compensate_b")

            async def step_c(self, ctx: SagaContext) -> SagaResult:
                return ctx.fail(error="Failure at step C")

            async def compensate_step_c(self, ctx: SagaContext) -> None:
                order_log.append("compensate_c")

            async def abort(self, ctx: SagaContext) -> None:
                order_log.append("global_abort")

        with pytest.raises(SagaExecutionError, match="Failure at step C"):
            await coordinator.run_saga(ReverseOrderSaga, {})

        assert order_log == ["compensate_b", "compensate_a", "global_abort"]
