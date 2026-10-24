import asyncio
import json
import pytest
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from pya_durable.harness import DurableHarness
from pya_durable.models import (
    ConversationRecord,
    ReplayPolicy,
    SubmissionRecord,
    SubmissionStatus,
    TaskRecord,
    TaskStatus,
    ToolInvocationRecord,
    ToolInvocationStatus,
)
from pya_durable.recovery import CrashRecoveryManager
from pya_durable.replay import (
    ConfigurationError,
    InvalidReplayPolicyError,
    TOOL_REGISTRY,
    durable_execution_context,
    durable_tool,
    set_tool_partial_output,
)
from pya_durable.storage import AioSqliteStorage


def test_invalid_replay_policy_raises_configuration_error() -> None:
    with pytest.raises(InvalidReplayPolicyError) as exc_info:
        @durable_tool(replay="unsupported_policy")
        def sample_tool() -> str:
            return "ok"

    assert issubclass(InvalidReplayPolicyError, ConfigurationError)
    assert "Invalid replay policy" in str(exc_info.value)


def test_durable_tool_bare_decorator_defaults_to_safe() -> None:
    @durable_tool
    def sample_safe() -> str:
        return "safe_result"

    assert getattr(sample_safe, "_is_durable_tool") is True
    assert getattr(sample_safe, "replay_policy") == ReplayPolicy.SAFE


async def test_durable_tool_execution_tracking(temp_db: str) -> None:
    @durable_tool(replay="safe")
    async def add_numbers(a: int, b: int) -> int:
        return a + b

    async with AioSqliteStorage(temp_db) as storage:
        conv = ConversationRecord(
            conversation_id="conv-tool-1",
            agent_config_json="{}",
        )
        await storage.insert_conversation(conv)
        task = TaskRecord(
            task_id="task-tool-1",
            task_type="test",
            status=TaskStatus.RUNNING,
            owner_kind="conversation",
            owner_id="conv-tool-1",
            phase="test",
            input_json="{}",
            state_json="{}",
        )
        await storage.insert_task(task)

        async with durable_execution_context(storage, "task-tool-1", "conv-tool-1"):
            result = await add_numbers(10, 20)

        assert result == 30

        invocations = await storage.get_tool_invocations_by_task("task-tool-1")
        assert len(invocations) == 1
        inv = invocations[0]
        assert inv.tool_name == "add_numbers"
        assert inv.status == ToolInvocationStatus.COMPLETED
        assert json.loads(inv.args_json) == {"a": 10, "b": 20}
        assert json.loads(inv.result_json) == 30


async def test_durable_tool_timeout(temp_db: str) -> None:
    @durable_tool(replay="safe", timeout_seconds=0.05)
    async def slow_tool() -> str:
        await asyncio.sleep(0.3)
        return "done"

    async with AioSqliteStorage(temp_db) as storage:
        await storage.insert_conversation(
            ConversationRecord(conversation_id="conv-timeout-1", agent_config_json="{}")
        )
        task = TaskRecord(
            task_id="task-timeout-1",
            task_type="test",
            status=TaskStatus.RUNNING,
            owner_kind="conversation",
            owner_id="conv-timeout-1",
            phase="test",
            input_json="{}",
            state_json="{}",
        )
        await storage.insert_task(task)

        async with durable_execution_context(storage, "task-timeout-1", "conv-timeout-1"):
            with pytest.raises(asyncio.TimeoutError):
                await slow_tool()

        invocations = await storage.get_tool_invocations_by_task("task-timeout-1")
        assert len(invocations) == 1
        inv = invocations[0]
        assert inv.status == ToolInvocationStatus.FAILED
        assert "timed out" in (inv.error_message or "")


async def test_durable_tool_failure_recording(temp_db: str) -> None:
    @durable_tool(replay="unsafe")
    async def failing_tool() -> None:
        raise ValueError("Network link failure")

    async with AioSqliteStorage(temp_db) as storage:
        await storage.insert_conversation(
            ConversationRecord(conversation_id="conv-fail-1", agent_config_json="{}")
        )
        task = TaskRecord(
            task_id="task-fail-1",
            task_type="test",
            status=TaskStatus.RUNNING,
            owner_kind="conversation",
            owner_id="conv-fail-1",
            phase="test",
            input_json="{}",
            state_json="{}",
        )
        await storage.insert_task(task)

        async with durable_execution_context(storage, "task-fail-1", "conv-fail-1"):
            with pytest.raises(ValueError, match="Network link failure"):
                await failing_tool()

        invocations = await storage.get_tool_invocations_by_task("task-fail-1")
        assert len(invocations) == 1
        inv = invocations[0]
        assert inv.status == ToolInvocationStatus.FAILED
        assert "Network link failure" in (inv.error_message or "")


async def test_partial_output_preservation(temp_db: str) -> None:
    @durable_tool(replay="unsafe")
    async def streaming_tool() -> str:
        await set_tool_partial_output("chunk-1: started processing")
        await set_tool_partial_output("chunk-2: halfway done")
        return "all chunks done"

    async with AioSqliteStorage(temp_db) as storage:
        await storage.insert_conversation(
            ConversationRecord(conversation_id="conv-partial-1", agent_config_json="{}")
        )
        task = TaskRecord(
            task_id="task-partial-1",
            task_type="test",
            status=TaskStatus.RUNNING,
            owner_kind="conversation",
            owner_id="conv-partial-1",
            phase="test",
            input_json="{}",
            state_json="{}",
        )
        await storage.insert_task(task)

        async with durable_execution_context(storage, "task-partial-1", "conv-partial-1"):
            res = await streaming_tool()
            assert res == "all chunks done"

        invocations = await storage.get_tool_invocations_by_task("task-partial-1")
        assert len(invocations) == 1
        assert invocations[0].partial_output == "chunk-2: halfway done"


async def test_reconcile_unsafe_tool_sealed(temp_db: str) -> None:
    async with AioSqliteStorage(temp_db) as storage:
        await storage.insert_conversation(
            ConversationRecord(conversation_id="conv-crash-1", agent_config_json="{}")
        )
        task = TaskRecord(
            task_id="task-unsafe-crash",
            task_type="user_turn",
            status=TaskStatus.RUNNING,
            owner_kind="conversation",
            owner_id="conv-crash-1",
            phase="executing_tool",
            input_json=json.dumps({"prompt": "Mutate balance"}),
            state_json="{}",
        )
        await storage.insert_task(task)

        inv = ToolInvocationRecord(
            invocation_id="inv-mutating-1",
            task_id="task-unsafe-crash",
            conversation_id="conv-crash-1",
            tool_name="transfer_funds",
            args_json=json.dumps({"amount": 500}),
            replay_policy=ReplayPolicy.UNSAFE,
            status=ToolInvocationStatus.STARTED,
            partial_output="Transfer initiated with ref 99182",
        )
        await storage.record_tool_started(inv)

        manager = CrashRecoveryManager(storage)
        report = await manager.reconcile_task(task)

        assert report.unsafe_tools_interrupted == 1
        assert report.reconciled_phase == "interrupted"

        inv_after = await storage.get_tool_invocation("inv-mutating-1")
        assert inv_after is not None
        assert inv_after.status == ToolInvocationStatus.INTERRUPTED
        assert inv_after.partial_output == "Transfer initiated with ref 99182"


async def test_reconcile_safe_tool_replayed(temp_db: str) -> None:
    call_count = 0

    @durable_tool(replay="safe")
    async def fetch_temperature(city: str) -> str:
        nonlocal call_count
        call_count += 1
        return f"{city} is 22C"

    async with AioSqliteStorage(temp_db) as storage:
        await storage.insert_conversation(
            ConversationRecord(conversation_id="conv-crash-2", agent_config_json="{}")
        )
        task = TaskRecord(
            task_id="task-safe-crash",
            task_type="user_turn",
            status=TaskStatus.RUNNING,
            owner_kind="conversation",
            owner_id="conv-crash-2",
            phase="executing_tool",
            input_json=json.dumps({"prompt": "Check temp"}),
            state_json="{}",
        )
        await storage.insert_task(task)

        inv = ToolInvocationRecord(
            invocation_id="inv-safe-1",
            task_id="task-safe-crash",
            conversation_id="conv-crash-2",
            tool_name="fetch_temperature",
            args_json=json.dumps({"city": "Berlin"}),
            replay_policy=ReplayPolicy.SAFE,
            status=ToolInvocationStatus.STARTED,
        )
        await storage.record_tool_started(inv)

        manager = CrashRecoveryManager(storage, tool_registry={"fetch_temperature": fetch_temperature})
        report = await manager.reconcile_task(task)

        assert report.safe_tools_replayed == 1
        assert report.unsafe_tools_interrupted == 0
        assert call_count == 1

        inv_after = await storage.get_tool_invocation("inv-safe-1")
        assert inv_after is not None
        assert inv_after.status == ToolInvocationStatus.COMPLETED
        assert json.loads(inv_after.result_json) == "Berlin is 22C"


async def test_clean_noop_recovery(temp_db: str) -> None:
    async with AioSqliteStorage(temp_db) as storage:
        manager = CrashRecoveryManager(storage)
        reports = await manager.reconcile_all()
        assert len(reports) == 0


async def test_parallel_task_recovery(temp_db: str) -> None:
    async with AioSqliteStorage(temp_db) as storage:
        await storage.insert_conversation(
            ConversationRecord(conversation_id="conv-par", agent_config_json="{}")
        )
        for i in range(5):
            t = TaskRecord(
                task_id=f"task-par-{i}",
                task_type="turn",
                status=TaskStatus.RUNNING,
                owner_kind="conversation",
                owner_id="conv-par",
                phase="in_flight",
                input_json="{}",
                state_json="{}",
            )
            await storage.insert_task(t)
            inv = ToolInvocationRecord(
                invocation_id=f"inv-par-{i}",
                task_id=f"task-par-{i}",
                conversation_id="conv-par",
                tool_name="write_db",
                args_json="{}",
                replay_policy=ReplayPolicy.UNSAFE,
                status=ToolInvocationStatus.STARTED,
            )
            await storage.record_tool_started(inv)

        manager = CrashRecoveryManager(storage)
        reports = await manager.reconcile_all()

        assert len(reports) == 5
        for r in reports:
            assert r.unsafe_tools_interrupted == 1
            assert r.reconciled_phase == "interrupted"


async def test_submission_idempotency(temp_db: str) -> None:
    async with AioSqliteStorage(temp_db) as storage:
        agent = Agent(TestModel(custom_output_text="Unique Computed Value"))
        harness = DurableHarness(agent, storage, conversation_id="conv-idem")

        req_id = "req-fixed-100"
        res1 = await harness.run("Calculate pi", request_id=req_id)
        assert res1.output == "Unique Computed Value"

        agent_fail = Agent(TestModel(custom_output_text="Different Answer"))
        harness2 = DurableHarness(agent_fail, storage, conversation_id="conv-idem")
        res2 = await harness2.run("Calculate pi", request_id=req_id)

        assert res2.output == "Unique Computed Value"


async def test_resume_crashed_task_regenerates_model(temp_db: str) -> None:
    async with AioSqliteStorage(temp_db) as storage:
        agent = Agent(TestModel(custom_output_text="Resumed Cleanly"))
        harness = DurableHarness(agent, storage, conversation_id="conv-resume-1")
        await harness._ensure_conversation()

        task = TaskRecord(
            task_id="task-killed-turn",
            task_type="user_turn",
            status=TaskStatus.RUNNING,
            owner_kind="conversation",
            owner_id="conv-resume-1",
            phase="ModelRequestNode",
            input_json=json.dumps({"prompt": "Hello after restart"}),
            state_json="{}",
        )
        await storage.insert_task(task)

        resumed_res = await harness.resume(task_id="task-killed-turn")
        assert resumed_res.output == "Resumed Cleanly"

        task_after = await storage.get_task("task-killed-turn")
        assert task_after is not None
        assert task_after.status == TaskStatus.COMPLETED
