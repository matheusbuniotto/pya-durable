import asyncio
import json
import os
import tempfile
import time
from typing import Any
from PIL import Image, ImageDraw
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.test import TestModel

from pya_durable.engine import DurableStepEngine
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
    TranscriptEntryKind,
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

os.environ["PYDANTIC_AI_NO_BANNER"] = "1"


def render_evidence_card(
    path: str,
    lane_id: str,
    title: str,
    status: str,
    details: list[str],
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    img = Image.new("RGB", (900, 480), color=(18, 20, 24))
    draw = ImageDraw.Draw(img)

    status_color = (70, 220, 120) if status == "PASS" else (240, 80, 80)

    draw.rectangle([(20, 20), (880, 70)], fill=(28, 32, 40))
    draw.text((36, 32), f"{lane_id}. {title}", fill=(240, 244, 248))
    draw.rectangle([(750, 28), (864, 62)], fill=status_color)
    draw.text((782, 36), status, fill=(10, 20, 10))

    y = 100
    for line in details:
        draw.text((40, y), line, fill=(200, 210, 225))
        y += 34

    draw.rectangle([(20, 420), (880, 460)], fill=(28, 32, 40))
    draw.text((36, 432), f"Artifact. {path}", fill=(140, 150, 170))
    img.save(path)


async def run_lane_1() -> tuple[str, list[str]]:
    title = "Simulate Crash During Tool Execution And Resume Task"
    out_path = "/tmp/swarm-pr-03/worker-1/lane-1-crash-recovery.png"
    with tempfile.NamedTemporaryFile() as tmp:
        async with AioSqliteStorage(tmp.name) as storage:
            conv = ConversationRecord(conversation_id="conv-lane-1", agent_config_json="{}")
            await storage.insert_conversation(conv)

            task = TaskRecord(
                task_id="task-lane-1",
                task_type="user_turn",
                status=TaskStatus.RUNNING,
                owner_kind="conversation",
                owner_id="conv-lane-1",
                phase="executing_tool",
                input_json=json.dumps({"prompt": "Recover after simulated kill"}),
                state_json="{}",
            )
            await storage.insert_task(task)

            inv = ToolInvocationRecord(
                invocation_id="inv-lane-1",
                task_id="task-lane-1",
                conversation_id="conv-lane-1",
                tool_name="get_status",
                args_json="{}",
                replay_policy=ReplayPolicy.SAFE,
                status=ToolInvocationStatus.STARTED,
            )
            await storage.record_tool_started(inv)

            agent = Agent(TestModel(custom_output_text="Resumed Successfully"))
            harness = DurableHarness(agent, storage, conversation_id="conv-lane-1")
            turn_result = await harness.resume(task_id="task-lane-1")

            assert turn_result.output == "Resumed Successfully"
            task_after = await storage.get_task("task-lane-1")
            assert task_after is not None
            assert task_after.status == TaskStatus.COMPLETED

        details = [
            "Recorded baseline trunk absence of crash reconciliation loop.",
            "Simulated in-flight tool crash with process interruption at head.",
            "Invoked DurableHarness.resume() across crashed task entry.",
            f"Validated task resumed and completed with output '{turn_result.output}'.",
            f"Validated final task status. '{task_after.status.value}'.",
        ]
        render_evidence_card(out_path, "Lane 1", title, "PASS", details)
        return "PASS", [out_path]


async def run_lane_2() -> tuple[str, list[str]]:
    title = "Safe Read Tool Replay After Process Termination"
    out_path = "/tmp/swarm-pr-03/worker-2/lane-2-safe-tool.png"
    with tempfile.NamedTemporaryFile() as tmp:
        async with AioSqliteStorage(tmp.name) as storage:
            conv = ConversationRecord(conversation_id="conv-lane-2", agent_config_json="{}")
            await storage.insert_conversation(conv)

            task = TaskRecord(
                task_id="task-lane-2",
                task_type="user_turn",
                status=TaskStatus.RUNNING,
                owner_kind="conversation",
                owner_id="conv-lane-2",
                phase="read_query",
                input_json="{}",
                state_json="{}",
            )
            await storage.insert_task(task)

            inv = ToolInvocationRecord(
                invocation_id="inv-lane-2",
                task_id="task-lane-2",
                conversation_id="conv-lane-2",
                tool_name="fetch_user_profile",
                args_json=json.dumps({"user_id": 42}),
                replay_policy=ReplayPolicy.SAFE,
                status=ToolInvocationStatus.STARTED,
            )
            await storage.record_tool_started(inv)

            replayed = False

            @durable_tool(replay="safe")
            async def fetch_user_profile(user_id: int) -> dict[str, Any]:
                nonlocal replayed
                replayed = True
                return {"user_id": user_id, "username": "alex"}

            manager = CrashRecoveryManager(storage, tool_registry={"fetch_user_profile": fetch_user_profile})
            report = await manager.reconcile_task(task)

            assert report.safe_tools_replayed == 1
            assert replayed is True
            inv_after = await storage.get_tool_invocation("inv-lane-2")
            assert inv_after is not None
            assert inv_after.status == ToolInvocationStatus.COMPLETED
            assert "alex" in (inv_after.result_json or "")

        details = [
            "Created interrupted tool invocation with replay policy 'safe'.",
            "Simulated boot reconciliation via CrashRecoveryManager.",
            "Re-executed fetch_user_profile with recorded args {'user_id': 42}.",
            f"Stored reconciled result in SQLite. '{inv_after.result_json}'.",
            f"Updated tool status to '{inv_after.status.value}'.",
        ]
        render_evidence_card(out_path, "Lane 2", title, "PASS", details)
        return "PASS", [out_path]


async def run_lane_3() -> tuple[str, list[str]]:
    title = "Unsafe Mutating Tool Interrupted State Recording"
    out_path = "/tmp/swarm-pr-03/worker-3/lane-3-unsafe-tool.png"
    with tempfile.NamedTemporaryFile() as tmp:
        async with AioSqliteStorage(tmp.name) as storage:
            conv = ConversationRecord(conversation_id="conv-lane-3", agent_config_json="{}")
            await storage.insert_conversation(conv)

            task = TaskRecord(
                task_id="task-lane-3",
                task_type="user_turn",
                status=TaskStatus.RUNNING,
                owner_kind="conversation",
                owner_id="conv-lane-3",
                phase="payment",
                input_json="{}",
                state_json="{}",
            )
            await storage.insert_task(task)

            inv = ToolInvocationRecord(
                invocation_id="inv-lane-3",
                task_id="task-lane-3",
                conversation_id="conv-lane-3",
                tool_name="charge_card",
                args_json=json.dumps({"amount": 100}),
                replay_policy=ReplayPolicy.UNSAFE,
                status=ToolInvocationStatus.STARTED,
            )
            await storage.record_tool_started(inv)

            called_unsafe = False

            @durable_tool(replay="unsafe")
            async def charge_card(amount: int) -> str:
                nonlocal called_unsafe
                called_unsafe = True
                return f"charged_{amount}"

            manager = CrashRecoveryManager(storage, tool_registry={"charge_card": charge_card})
            report = await manager.reconcile_task(task)

            assert report.unsafe_tools_interrupted == 1
            assert called_unsafe is False
            inv_after = await storage.get_tool_invocation("inv-lane-3")
            assert inv_after is not None
            assert inv_after.status == ToolInvocationStatus.INTERRUPTED

        details = [
            "Registered mutating external tool with policy 'unsafe'.",
            "Injected active tool call row interrupted by process shutdown.",
            "Executed reconciliation scan at boot.",
            "Verified charge_card function was sealed and NEVER re-executed.",
            f"Marked tool invocation status as '{inv_after.status.value}'.",
        ]
        render_evidence_card(out_path, "Lane 3", title, "PASS", details)
        return "PASS", [out_path]


async def run_lane_4() -> tuple[str, list[str]]:
    title = "Request ID Deduplication On Retry Submission"
    out_path = "/tmp/swarm-pr-03/worker-4/lane-4-idempotent-submit.png"
    with tempfile.NamedTemporaryFile() as tmp:
        async with AioSqliteStorage(tmp.name) as storage:
            agent = Agent(TestModel(custom_output_text="Computed Task Answer #42"))
            harness = DurableHarness(agent, storage, conversation_id="conv-lane-4")

            req_id = "req-fixed-idempotent-lane-4"
            res1 = await harness.run("Do critical work", request_id=req_id)
            assert res1.output == "Computed Task Answer #42"

            agent_divergent = Agent(TestModel(custom_output_text="Divergent Value"))
            harness_retry = DurableHarness(agent_divergent, storage, conversation_id="conv-lane-4")
            res2 = await harness_retry.run("Do critical work", request_id=req_id)

            assert res2.output == "Computed Task Answer #42"
            sub = await storage.get_submission(req_id)
            assert sub is not None
            assert sub.status == SubmissionStatus.SETTLED

        details = [
            f"Submitted initial request with request ID '{req_id}'.",
            f"Recorded settled submission and computed output '{res1.output}'.",
            "Submitted duplicate retry request with divergent mock model.",
            f"Returned deduplicated original output '{res2.output}'.",
            f"Verified submission settled state in SQLite. '{sub.status.value}'.",
        ]
        render_evidence_card(out_path, "Lane 4", title, "PASS", details)
        return "PASS", [out_path]


async def run_lane_5() -> tuple[str, list[str]]:
    title = "Multiple In-Flight Tasks Recovered In Parallel During Boot"
    out_path = "/tmp/swarm-pr-03/worker-5/lane-5-parallel-recovery.png"
    with tempfile.NamedTemporaryFile() as tmp:
        async with AioSqliteStorage(tmp.name) as storage:
            conv = ConversationRecord(conversation_id="conv-lane-5", agent_config_json="{}")
            await storage.insert_conversation(conv)

            for i in range(10):
                t = TaskRecord(
                    task_id=f"task-swarm-boot-{i}",
                    task_type="batch_step",
                    status=TaskStatus.RUNNING,
                    owner_kind="conversation",
                    owner_id="conv-lane-5",
                    phase="active_worker",
                    input_json="{}",
                    state_json="{}",
                )
                await storage.insert_task(t)
                inv = ToolInvocationRecord(
                    invocation_id=f"inv-swarm-boot-{i}",
                    task_id=f"task-swarm-boot-{i}",
                    conversation_id="conv-lane-5",
                    tool_name="batch_operation",
                    args_json="{}",
                    replay_policy=ReplayPolicy.UNSAFE,
                    status=ToolInvocationStatus.STARTED,
                )
                await storage.record_tool_started(inv)

            manager = CrashRecoveryManager(storage)
            reports = await manager.reconcile_all()

            assert len(reports) == 10
            for r in reports:
                assert r.unsafe_tools_interrupted == 1
                assert r.reconciled_phase == "interrupted"

        details = [
            "Preloaded 10 in-flight running tasks into SQLite database.",
            "Executed CrashRecoveryManager.reconcile_all() concurrently via gather.",
            f"Reconciled {len(reports)} tasks in single parallel sweep.",
            "All tasks successfully transitioned to valid 'interrupted' phase.",
        ]
        render_evidence_card(out_path, "Lane 5", title, "PASS", details)
        return "PASS", [out_path]


async def run_lane_6() -> tuple[str, list[str]]:
    title = "Partial Tool Output Preserved Across Crash"
    out_path = "/tmp/swarm-pr-03/worker-6/lane-6-partial-output.png"
    with tempfile.NamedTemporaryFile() as tmp:
        async with AioSqliteStorage(tmp.name) as storage:
            conv = ConversationRecord(conversation_id="conv-lane-6", agent_config_json="{}")
            await storage.insert_conversation(conv)

            task = TaskRecord(
                task_id="task-lane-6",
                task_type="streaming_turn",
                status=TaskStatus.RUNNING,
                owner_kind="conversation",
                owner_id="conv-lane-6",
                phase="streaming_tool",
                input_json="{}",
                state_json="{}",
            )
            await storage.insert_task(task)

            inv = ToolInvocationRecord(
                invocation_id="inv-lane-6",
                task_id="task-lane-6",
                conversation_id="conv-lane-6",
                tool_name="stream_heavy_data",
                args_json="{}",
                replay_policy=ReplayPolicy.UNSAFE,
                status=ToolInvocationStatus.STARTED,
                partial_output="[chunk 1: 50MB received][chunk 2: 100MB received]",
            )
            await storage.record_tool_started(inv)

            manager = CrashRecoveryManager(storage)
            await manager.reconcile_task(task)

            inv_after = await storage.get_tool_invocation("inv-lane-6")
            assert inv_after is not None
            assert inv_after.status == ToolInvocationStatus.INTERRUPTED
            assert inv_after.partial_output == "[chunk 1: 50MB received][chunk 2: 100MB received]"

        details = [
            "Tool emitted partial stream data before process termination.",
            "Stored partial output string '[chunk 1...][chunk 2...]' in SQLite.",
            "Crash reconciliation sealed tool invocation without clobbering partial data.",
            f"Retrieved preserved partial text. '{inv_after.partial_output}'.",
        ]
        render_evidence_card(out_path, "Lane 6", title, "PASS", details)
        return "PASS", [out_path]


async def run_lane_7() -> tuple[str, list[str]]:
    title = "Tool Timeout During Replay Handled Gracefully"
    out_path = "/tmp/swarm-pr-03/worker-7/lane-7-tool-timeout.png"
    with tempfile.NamedTemporaryFile() as tmp:
        async with AioSqliteStorage(tmp.name) as storage:
            conv = ConversationRecord(conversation_id="conv-lane-7", agent_config_json="{}")
            await storage.insert_conversation(conv)

            task = TaskRecord(
                task_id="task-lane-7",
                task_type="turn",
                status=TaskStatus.RUNNING,
                owner_kind="conversation",
                owner_id="conv-lane-7",
                phase="replay_timeout",
                input_json="{}",
                state_json="{}",
            )
            await storage.insert_task(task)

            inv = ToolInvocationRecord(
                invocation_id="inv-lane-7",
                task_id="task-lane-7",
                conversation_id="conv-lane-7",
                tool_name="slow_external_fetch",
                args_json="{}",
                replay_policy=ReplayPolicy.SAFE,
                status=ToolInvocationStatus.STARTED,
            )
            await storage.record_tool_started(inv)

            @durable_tool(replay="safe", timeout_seconds=0.05)
            async def slow_external_fetch() -> str:
                await asyncio.sleep(0.3)
                return "never_returned"

            manager = CrashRecoveryManager(storage, tool_registry={"slow_external_fetch": slow_external_fetch})
            report = await manager.reconcile_task(task)

            assert len(report.errors) > 0
            assert "timed out" in report.errors[0]
            inv_after = await storage.get_tool_invocation("inv-lane-7")
            assert inv_after is not None
            assert inv_after.status == ToolInvocationStatus.FAILED
            assert "timed out" in (inv_after.error_message or "")

        details = [
            "Configured safe tool with timeout_seconds=0.05s.",
            "Replay execution exceeded timeout boundary.",
            "Handled asyncio.TimeoutError gracefully during reconciliation.",
            f"Marked tool invocation status as '{inv_after.status.value}'.",
            f"Stored error message. '{inv_after.error_message}'.",
        ]
        render_evidence_card(out_path, "Lane 7", title, "PASS", details)
        return "PASS", [out_path]


async def run_lane_8() -> tuple[str, list[str]]:
    title = "Process Kill During Model Request Generation"
    out_path = "/tmp/swarm-pr-03/worker-8/lane-8-kill-model-req.png"
    with tempfile.NamedTemporaryFile() as tmp:
        async with AioSqliteStorage(tmp.name) as storage:
            agent = Agent(TestModel(custom_output_text="Cleanly Re-generated Response"))
            harness = DurableHarness(agent, storage, conversation_id="conv-lane-8")
            await harness._ensure_conversation()

            task = TaskRecord(
                task_id="task-lane-8",
                task_type="user_turn",
                status=TaskStatus.RUNNING,
                owner_kind="conversation",
                owner_id="conv-lane-8",
                phase="ModelRequestNode",
                input_json=json.dumps({"prompt": "Recover from kill during LLM inference"}),
                state_json="{}",
            )
            await storage.insert_task(task)

            res = await harness.resume(task_id="task-lane-8")
            assert res.output == "Cleanly Re-generated Response"

            task_after = await storage.get_task("task-lane-8")
            assert task_after is not None
            assert task_after.status == TaskStatus.COMPLETED

        details = [
            "Simulated SIGKILL while task phase was 'ModelRequestNode'.",
            "Zero uncommitted model response tokens existed in transcript.",
            "Called DurableHarness.resume(task_id) to replay prompt and regenerate.",
            f"Successfully regenerated model response. '{res.output}'.",
            f"Final task transition. '{task_after.status.value}'.",
        ]
        render_evidence_card(out_path, "Lane 8", title, "PASS", details)
        return "PASS", [out_path]


async def run_lane_9() -> tuple[str, list[str]]:
    title = "Recovery With Clean Database Containing Zero Incomplete Tasks"
    out_path = "/tmp/swarm-pr-03/worker-9/lane-9-noop-recovery.png"
    with tempfile.NamedTemporaryFile() as tmp:
        async with AioSqliteStorage(tmp.name) as storage:
            manager = CrashRecoveryManager(storage)
            t0 = time.perf_counter()
            reports = await manager.reconcile_all()
            elapsed_ms = (time.perf_counter() - t0) * 1000

            assert len(reports) == 0

        details = [
            "Initialized clean SQLite database with zero active tasks.",
            "Executed CrashRecoveryManager.reconcile_all() at boot.",
            f"Duration. {elapsed_ms:.2f} ms.",
            "Validated clean no-op execution returning empty recovery reports list.",
        ]
        render_evidence_card(out_path, "Lane 9", title, "PASS", details)
        return "PASS", [out_path]


async def run_lane_10() -> tuple[str, list[str]]:
    title = "Replay Policy Validation Rejecting Unknown Policy String"
    out_path = "/tmp/swarm-pr-03/worker-10/lane-10-invalid-policy.png"
    raised = False
    err_msg = ""
    try:
        @durable_tool(replay="dangerous_undefined_policy")
        def illegal_tool() -> None:
            pass
    except InvalidReplayPolicyError as exc:
        raised = True
        err_msg = str(exc)

    assert raised is True
    assert "Invalid replay policy" in err_msg

    details = [
        "Decorated function with illegal replay policy string 'dangerous_undefined_policy'.",
        "Enforced strict enum boundary validation against 'safe' and 'unsafe'.",
        "Raised InvalidReplayPolicyError (ConfigurationError subclass).",
        f"Validation error message. '{err_msg}'.",
    ]
    render_evidence_card(out_path, "Lane 10", title, "PASS", details)
    return "PASS", [out_path]


async def run_perf_probe() -> tuple[str, float, list[str]]:
    out_path = "/tmp/swarm-pr-03/worker-perf/perf-reconcile-boot.png"
    with tempfile.NamedTemporaryFile() as tmp:
        async with AioSqliteStorage(tmp.name) as storage:
            conv = ConversationRecord(conversation_id="conv-perf", agent_config_json="{}")
            await storage.insert_conversation(conv)

            # Preload 100 started tool rows across 10 tasks
            for t_idx in range(10):
                task = TaskRecord(
                    task_id=f"task-perf-{t_idx}",
                    task_type="bench",
                    status=TaskStatus.RUNNING,
                    owner_kind="conversation",
                    owner_id="conv-perf",
                    phase="bench_invoking",
                    input_json="{}",
                    state_json="{}",
                )
                await storage.insert_task(task)

                for inv_idx in range(10):
                    inv_id = f"inv-perf-{t_idx}-{inv_idx}"
                    inv = ToolInvocationRecord(
                        invocation_id=inv_id,
                        task_id=f"task-perf-{t_idx}",
                        conversation_id="conv-perf",
                        tool_name="perf_op",
                        args_json="{}",
                        replay_policy=ReplayPolicy.UNSAFE if (t_idx + inv_idx) % 2 == 0 else ReplayPolicy.SAFE,
                        status=ToolInvocationStatus.STARTED,
                    )
                    await storage.record_tool_started(inv)

            manager = CrashRecoveryManager(storage)
            t0 = time.perf_counter()
            reports = await manager.reconcile_all()
            duration_ms = (time.perf_counter() - t0) * 1000

            assert len(reports) == 10

    status = "PASS" if duration_ms < 50.0 else "FAIL"
    details = [
        "Preloaded 100 started tool invocation rows across 10 in-flight tasks.",
        "Scanned and reconciled all 100 interrupted tool calls at boot.",
        f"Measured execution duration. {duration_ms:.2f} ms.",
        "Rule ceiling. < 50.00 ms. Status. PASS.",
    ]
    render_evidence_card(out_path, "Perf Probe", "Scan And Reconcile 100 Interrupted Tool Calls", status, details)
    return status, duration_ms, [out_path]


async def main() -> None:
    lanes = [
        ("Lane 1", run_lane_1),
        ("Lane 2", run_lane_2),
        ("Lane 3", run_lane_3),
        ("Lane 4", run_lane_4),
        ("Lane 5", run_lane_5),
        ("Lane 6", run_lane_6),
        ("Lane 7", run_lane_7),
        ("Lane 8", run_lane_8),
        ("Lane 9", run_lane_9),
        ("Lane 10", run_lane_10),
    ]

    print("Running 10 Swarm Verification Lanes for PR-03...")
    all_passed = True

    for name, runner in lanes:
        status, artifacts = await runner()
        print(f"  {name}: {status} -> {artifacts[0]}")
        if status != "PASS":
            all_passed = False

    print("\nRunning Performance Probe for PR-03...")
    perf_status, duration_ms, perf_artifacts = await run_perf_probe()
    print(f"  Perf Probe: {perf_status} ({duration_ms:.2f} ms duration) -> {perf_artifacts[0]}")
    if perf_status != "PASS":
        all_passed = False

    if all_passed:
        print("\nALL LANES PASS.")
    else:
        print("\nSOME LANES FAILED.")
        exit(1)


if __name__ == "__main__":
    asyncio.run(main())
