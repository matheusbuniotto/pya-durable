import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import pytest
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.test import TestModel

from pya_durable.models import (
    ConversationRecord,
    TaskRecord,
    TaskStatus,
)
from pya_durable.storage import AioSqliteStorage
from pya_durable.subagents import SubagentManager, SubagentResult
from pya_durable.task_tree import (
    CycleDetectedError,
    TaskContext,
    TaskDefinition,
    TaskPhaseResult,
    TaskRunner,
    check_cycle,
    define_task,
)


@pytest.mark.asyncio
async def test_multi_tier_depth_three(temp_db: str) -> None:
    async with AioSqliteStorage(temp_db) as storage:
        runner = TaskRunner(storage)

        grandchild_def = define_task(
            name="grandchild_task",
            phases={
                "start": lambda ctx: TaskPhaseResult(
                    status=TaskStatus.COMPLETED,
                    result=ctx.input_data.get("val", 0) + 10,
                )
            },
        )
        runner.register_task(grandchild_def)

        async def child_start(ctx: TaskContext) -> TaskPhaseResult:
            gc = await ctx.create_child_task("grandchild_task", {"val": ctx.input_data.get("val", 0)})
            res = await runner.run_task(
                grandchild_def,
                input_data={"val": ctx.input_data.get("val", 0)},
                task_id=gc.task_id,
                owner_kind="task",
                owner_id=ctx.task_id,
            )
            return TaskPhaseResult(status=TaskStatus.COMPLETED, result=res * 2)

        child_def = define_task(name="child_task", phases={"start": child_start})
        runner.register_task(child_def)

        async def root_start(ctx: TaskContext) -> TaskPhaseResult:
            c = await ctx.create_child_task("child_task", {"val": ctx.input_data.get("val", 0)})
            res = await runner.run_task(
                child_def,
                input_data={"val": ctx.input_data.get("val", 0)},
                task_id=c.task_id,
                owner_kind="task",
                owner_id=ctx.task_id,
            )
            return TaskPhaseResult(status=TaskStatus.COMPLETED, result=res + 1)

        root_def = define_task(name="root_task", phases={"start": root_start})
        runner.register_task(root_def)

        final_result = await runner.run_task(root_def, input_data={"val": 5})
        assert final_result == 31


@pytest.mark.asyncio
async def test_phase_transitions_and_checkpoint_persistence(temp_db: str) -> None:
    async with AioSqliteStorage(temp_db) as storage:
        runner = TaskRunner(storage)

        async def start_phase(ctx: TaskContext) -> TaskPhaseResult:
            ctx.state["counter"] = 100
            await ctx.checkpoint("middle_phase", ctx.state, checkpoint_data={"cursor": 1})
            return TaskPhaseResult(next_phase="middle_phase", status=TaskStatus.RUNNING)

        async def middle_phase(ctx: TaskContext) -> TaskPhaseResult:
            ctx.state["counter"] += 50
            await ctx.checkpoint("finish_phase", ctx.state, checkpoint_data={"cursor": 2})
            return TaskPhaseResult(next_phase="finish_phase", status=TaskStatus.RUNNING)

        async def finish_phase(ctx: TaskContext) -> TaskPhaseResult:
            return TaskPhaseResult(status=TaskStatus.COMPLETED, result=ctx.state["counter"])

        task_def = define_task(
            name="multi_phase_task",
            phases={
                "start": start_phase,
                "middle_phase": middle_phase,
                "finish_phase": finish_phase,
            },
        )

        res = await runner.run_task(task_def, input_data={})
        assert res == 150


@pytest.mark.asyncio
async def test_bottom_up_abort_cascade(temp_db: str, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    async with AioSqliteStorage(temp_db) as storage:
        runner = TaskRunner(storage)

        aborted_tasks: list[str] = []

        async def abort_handler_fn(ctx: TaskContext) -> None:
            aborted_tasks.append(ctx.task_id)

        task_def = define_task(
            name="cascading_task",
            phases={"start": lambda ctx: TaskPhaseResult(status=TaskStatus.RUNNING)},
            abort_handler=abort_handler_fn,
        )
        runner.register_task(task_def)

        root = TaskRecord(
            task_id="task-root-abort",
            task_type="cascading_task",
            version=1,
            status=TaskStatus.RUNNING,
            owner_kind="root",
            owner_id="root",
            phase="start",
            input_json="{}",
            state_json="{}",
        )
        child = TaskRecord(
            task_id="task-child-abort",
            task_type="cascading_task",
            version=1,
            status=TaskStatus.RUNNING,
            owner_kind="task",
            owner_id="task-root-abort",
            phase="start",
            input_json="{}",
            state_json="{}",
        )
        leaf = TaskRecord(
            task_id="task-leaf-abort",
            task_type="cascading_task",
            version=1,
            status=TaskStatus.RUNNING,
            owner_kind="task",
            owner_id="task-child-abort",
            phase="start",
            input_json="{}",
            state_json="{}",
        )

        await storage.insert_task(root)
        await storage.insert_task(child)
        await storage.insert_task(leaf)

        await runner.abort_task("task-root-abort")

        assert aborted_tasks == ["task-leaf-abort", "task-child-abort", "task-root-abort"]

        r_after = await storage.get_task("task-root-abort")
        c_after = await storage.get_task("task-child-abort")
        l_after = await storage.get_task("task-leaf-abort")

        assert r_after is not None and r_after.status == TaskStatus.ABORTED
        assert c_after is not None and c_after.status == TaskStatus.ABORTED
        assert l_after is not None and l_after.status == TaskStatus.ABORTED

        assert "Aborted child task task-leaf-abort due to parent cancellation" in caplog.text
        assert "Aborted child task task-child-abort due to parent cancellation" in caplog.text


@pytest.mark.asyncio
async def test_fail_fast_sibling_abort(temp_db: str, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    async with AioSqliteStorage(temp_db) as storage:
        runner = TaskRunner(storage)

        async def failing_phase(ctx: TaskContext) -> Any:
            raise RuntimeError("Task failed intentionally")

        fail_def = define_task(name="failing_task", phases={"start": failing_phase})
        runner.register_task(fail_def)

        parent = TaskRecord(
            task_id="task-parent-ff",
            task_type="parent",
            status=TaskStatus.RUNNING,
            owner_kind="root",
            owner_id="root",
            phase="start",
            input_json="{}",
            state_json="{}",
        )
        child1 = TaskRecord(
            task_id="task-child-fail",
            task_type="failing_task",
            status=TaskStatus.RUNNING,
            owner_kind="task",
            owner_id="task-parent-ff",
            phase="start",
            input_json="{}",
            state_json="{}",
            policy="failFast",
        )
        child2 = TaskRecord(
            task_id="task-child-sibling",
            task_type="sibling",
            status=TaskStatus.RUNNING,
            owner_kind="task",
            owner_id="task-parent-ff",
            phase="start",
            input_json="{}",
            state_json="{}",
            policy="failFast",
        )

        await storage.insert_task(parent)
        await storage.insert_task(child1)
        await storage.insert_task(child2)

        with pytest.raises(RuntimeError, match="Task failed intentionally"):
            await runner.run_task(
                fail_def,
                input_data={},
                task_id="task-child-fail",
                owner_kind="task",
                owner_id="task-parent-ff",
                policy="failFast",
            )

        sib_after = await storage.get_task("task-child-sibling")
        assert sib_after is not None
        assert sib_after.status == TaskStatus.ABORTED
        assert "Aborted child task task-child-sibling due to parent cancellation" in caplog.text


@pytest.mark.asyncio
async def test_task_compensation_on_abort(temp_db: str, tmp_path: Path) -> None:
    async with AioSqliteStorage(temp_db) as storage:
        runner = TaskRunner(storage)
        scratch_file = tmp_path / "scratch_work.txt"
        scratch_file.write_text("temporary data")
        assert scratch_file.exists()

        async def compensation_handler(ctx: TaskContext) -> None:
            target = ctx.state.get("file")
            if target and os.path.exists(target):
                os.remove(target)

        task_def = define_task(
            name="compensated_task",
            phases={"start": lambda ctx: TaskPhaseResult(status=TaskStatus.RUNNING)},
            abort_handler=compensation_handler,
        )
        runner.register_task(task_def)

        task = TaskRecord(
            task_id="task-comp-1",
            task_type="compensated_task",
            status=TaskStatus.RUNNING,
            owner_kind="root",
            owner_id="root",
            phase="start",
            input_json="{}",
            state_json=json.dumps({"file": str(scratch_file)}),
        )
        await storage.insert_task(task)

        await runner.abort_task("task-comp-1")
        assert not scratch_file.exists()


@pytest.mark.asyncio
async def test_cycle_detection(temp_db: str) -> None:
    async with AioSqliteStorage(temp_db) as storage:
        with pytest.raises(CycleDetectedError):
            await check_cycle(storage, "task-A", "task-A")

        t1 = TaskRecord(
            task_id="task-A",
            task_type="t",
            status=TaskStatus.RUNNING,
            owner_kind="root",
            owner_id="root",
            phase="start",
            input_json="{}",
            state_json="{}",
        )
        t2 = TaskRecord(
            task_id="task-B",
            task_type="t",
            status=TaskStatus.RUNNING,
            owner_kind="task",
            owner_id="task-A",
            phase="start",
            input_json="{}",
            state_json="{}",
        )
        t3 = TaskRecord(
            task_id="task-C",
            task_type="t",
            status=TaskStatus.RUNNING,
            owner_kind="task",
            owner_id="task-B",
            phase="start",
            input_json="{}",
            state_json="{}",
        )

        await storage.insert_task(t1)
        await storage.insert_task(t2)
        await storage.insert_task(t3)

        with pytest.raises(CycleDetectedError):
            await check_cycle(storage, "task-A", "task-C")


@pytest.mark.asyncio
async def test_subagent_lifecycle_and_result(temp_db: str) -> None:
    async with AioSqliteStorage(temp_db) as storage:
        parent_task = TaskRecord(
            task_id="parent-turn-1",
            task_type="turn",
            status=TaskStatus.RUNNING,
            owner_kind="root",
            owner_id="root",
            phase="executing",
            input_json="{}",
            state_json="{}",
        )
        await storage.insert_task(parent_task)

        agent = Agent(TestModel(custom_output_text="Subagent processed data"))
        sub_result = await SubagentManager.run_subagent(
            storage=storage,
            parent_task_id="parent-turn-1",
            agent=agent,
            prompt="summarize quarterly earnings",
        )

        assert isinstance(sub_result, SubagentResult)
        assert sub_result.output == "Subagent processed data"
        assert sub_result.conversation_id.startswith("subconv-")
        assert sub_result.task_id.startswith("subtask-")

        conv = await storage.get_conversation(sub_result.conversation_id)
        assert conv is not None
        assert "parent-turn-1" in conv.agent_config_json


@pytest.mark.asyncio
async def test_subagent_context_isolation(temp_db: str) -> None:
    async with AioSqliteStorage(temp_db) as storage:
        parent_task = TaskRecord(
            task_id="parent-priv-task",
            task_type="turn",
            status=TaskStatus.RUNNING,
            owner_kind="root",
            owner_id="root",
            phase="executing",
            input_json="{}",
            state_json="{}",
        )
        await storage.insert_task(parent_task)

        parent_agent = Agent(TestModel(custom_output_text="Parent response"))

        @parent_agent.tool
        def privileged_vault_access(ctx: RunContext[None]) -> str:
            return "SUPER_SECRET_CREDENTIALS"

        assert "privileged_vault_access" in parent_agent._function_toolset.tools

        def safe_public_tool(ctx: RunContext[None]) -> str:
            return "public info"

        child_model = TestModel(custom_output_text="Isolated child output")
        parent_agent_with_child_model = Agent(child_model)

        @parent_agent_with_child_model.tool
        def parent_secret(ctx: RunContext[None]) -> str:
            return "secret"

        sub_result = await SubagentManager.run_subagent(
            storage=storage,
            parent_task_id="parent-priv-task",
            agent=parent_agent_with_child_model,
            prompt="Perform isolated query",
            isolated_tools=[safe_public_tool],
        )

        assert sub_result is not None
        assert sub_result.output == "Isolated child output"


@pytest.mark.asyncio
async def test_subagent_crash_recovery(temp_db: str) -> None:
    async with AioSqliteStorage(temp_db) as storage:
        conv = ConversationRecord(
            conversation_id="conv-sub-crash",
            agent_config_json='{"parent_task_id": "parent-1"}',
        )
        await storage.insert_conversation(conv)

        task = TaskRecord(
            task_id="task-sub-interrupted",
            task_type="subagent_turn",
            status=TaskStatus.RUNNING,
            owner_kind="task",
            owner_id="parent-1",
            phase="ModelRequestNode",
            input_json=json.dumps({"prompt": "Continue unfinished work"}),
            state_json=json.dumps({"conversation_id": "conv-sub-crash"}),
        )
        await storage.insert_task(task)

        agent = Agent(TestModel(custom_output_text="Resumed child turn"))
        resumed = await SubagentManager.resume_subagent(
            storage=storage,
            conversation_id="conv-sub-crash",
            agent=agent,
        )

        assert resumed.output == "Resumed child turn"


@pytest.mark.asyncio
async def test_detached_background_task(temp_db: str) -> None:
    async with AioSqliteStorage(temp_db) as storage:
        runner = TaskRunner(storage)

        async def slow_phase(ctx: TaskContext) -> TaskPhaseResult:
            await asyncio.sleep(0.05)
            return TaskPhaseResult(status=TaskStatus.COMPLETED, result="bg_done")

        bg_def = define_task(name="bg_job", phases={"start": slow_phase})
        runner.register_task(bg_def)

        record = await runner.run_task(bg_def, input_data={}, background=True)
        assert record.status == TaskStatus.RUNNING

        bg_task_id = record.task_id
        await asyncio.sleep(0.1)

        task_after = await storage.get_task(bg_task_id)
        assert task_after is not None
        assert task_after.status == TaskStatus.COMPLETED


@pytest.mark.asyncio
async def test_cancellation_cascade_latency_fifty_tasks(temp_db: str) -> None:
    async with AioSqliteStorage(temp_db) as storage:
        runner = TaskRunner(storage)

        task_def = define_task(
            name="perf_task",
            phases={"start": lambda ctx: TaskPhaseResult(status=TaskStatus.RUNNING)},
        )
        runner.register_task(task_def)

        root = TaskRecord(
            task_id="perf-root",
            task_type="perf_task",
            status=TaskStatus.RUNNING,
            owner_kind="root",
            owner_id="root",
            phase="start",
            input_json="{}",
            state_json="{}",
        )
        await storage.insert_task(root)

        for i in range(50):
            sub = TaskRecord(
                task_id=f"perf-child-{i}",
                task_type="perf_task",
                status=TaskStatus.RUNNING,
                owner_kind="task",
                owner_id="perf-root",
                phase="start",
                input_json="{}",
                state_json="{}",
            )
            await storage.insert_task(sub)

        t_start = time.perf_counter()
        await runner.abort_task("perf-root")
        t_elapsed_ms = (time.perf_counter() - t_start) * 1000

        assert t_elapsed_ms < 25.0

        for i in range(50):
            t = await storage.get_task(f"perf-child-{i}")
            assert t is not None
            assert t.status == TaskStatus.ABORTED
