import json
from pathlib import Path

import pytest
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from pydantic_ai_durable.approvals import (
    ApprovalManager,
    ApprovalRejectedError,
    ApprovalRequiredError,
    InvalidApprovalDecisionError,
)
from pydantic_ai_durable.jsonl_storage import JsonlStorage
from pydantic_ai_durable.memory_storage import MemoryStorage
from pydantic_ai_durable.recovery import CrashRecoveryManager
from pydantic_ai_durable.replay import (
    durable_execution_context,
    durable_tool,
)

from pydantic_ai_durable.extensions import (
    HookContext,
    HookDecision,
    ToolCallInfo,
    define_extension,
    global_extension_registry,
    hook,
)
from pydantic_ai_durable.harness import DurableHarness
from pydantic_ai_durable.models import (
    ApprovalDecision,
    ApprovalStatus,
    ConversationRecord,
    TaskRecord,
    TaskStatus,
    ToolInvocationStatus,
)
from pydantic_ai_durable.storage import AioSqliteStorage


@pytest.fixture(autouse=True)
def clean_registry():
    global_extension_registry.clear()
    yield
    global_extension_registry.clear()


@pytest.mark.asyncio
async def test_approval_crud_sqlite(temp_db: str) -> None:
    async with AioSqliteStorage(temp_db) as storage:
        mgr = ApprovalManager(storage)
        conv = ConversationRecord(conversation_id="c1", agent_config_json="{}")
        await storage.insert_conversation(conv)
        task = TaskRecord(
            task_id="t1",
            task_type="user_turn",
            status=TaskStatus.RUNNING,
            owner_kind="conversation",
            owner_id="c1",
            phase="running",
            input_json="{}",
            state_json="{}",
        )
        await storage.insert_task(task)

        rec = await mgr.create_challenge(
            task_id="t1",
            conversation_id="c1",
            tool_name="delete_account",
            args={"user_id": "u123"},
            prompt_message="Confirm user deletion",
            timeout_seconds=300.0,
        )

        assert rec.challenge_token.startswith("appr-")
        assert rec.status == ApprovalStatus.PENDING
        assert rec.tool_name == "delete_account"
        assert rec.prompt_message == "Confirm user deletion"
        assert rec.expires_at is not None

        # Fetch by token
        fetched = await mgr.get(rec.challenge_token)
        assert fetched is not None
        assert fetched.challenge_token == rec.challenge_token

        # Fetch by task and tool
        by_tool = await storage.get_approval_by_task_and_tool("t1", "delete_account")
        assert by_tool is not None
        assert by_tool.challenge_token == rec.challenge_token

        # List pending
        pending = await mgr.list_pending("c1")
        assert len(pending) == 1
        assert pending[0].challenge_token == rec.challenge_token

        # Resolve
        resolved = await mgr.resolve(
            rec.challenge_token,
            approved=True,
            reason="Verified by admin",
            approver="admin-1",
        )
        assert resolved.status == ApprovalStatus.APPROVED
        assert resolved.decision_json is not None
        dec = ApprovalDecision.model_validate_json(resolved.decision_json)
        assert dec.status == ApprovalStatus.APPROVED
        assert dec.approver == "admin-1"
        assert dec.reason == "Verified by admin"

        # Now pending list should be empty
        pending_after = await mgr.list_pending("c1")
        assert len(pending_after) == 0


@pytest.mark.asyncio
async def test_approval_crud_memory() -> None:
    storage = MemoryStorage()
    await storage.init_db()
    mgr = ApprovalManager(storage)

    conv = ConversationRecord(conversation_id="c-mem", agent_config_json="{}")
    await storage.insert_conversation(conv)
    task = TaskRecord(
        task_id="t-mem",
        task_type="user_turn",
        status=TaskStatus.RUNNING,
        owner_kind="conversation",
        owner_id="c-mem",
        phase="running",
        input_json="{}",
        state_json="{}",
    )
    await storage.insert_task(task)

    rec = await mgr.create_challenge(
        task_id="t-mem",
        conversation_id="c-mem",
        tool_name="reboot_cluster",
        args={"force": True},
    )
    assert rec.status == ApprovalStatus.PENDING

    # Resolve rejection
    resolved = await mgr.resolve(
        rec.challenge_token,
        approved=False,
        reason="Cluster is currently under load",
        approver="ops-lead",
    )
    assert resolved.status == ApprovalStatus.REJECTED
    dec = ApprovalDecision.model_validate_json(resolved.decision_json)
    assert dec.status == ApprovalStatus.REJECTED
    assert dec.approver == "ops-lead"


@pytest.mark.asyncio
async def test_approval_crud_jsonl(tmp_path: Path) -> None:
    jsonl_file = tmp_path / "approvals_test.jsonl"
    storage = JsonlStorage(jsonl_file)
    await storage.init_db()
    mgr = ApprovalManager(storage)

    conv = ConversationRecord(conversation_id="c-jsonl", agent_config_json="{}")
    await storage.insert_conversation(conv)
    task = TaskRecord(
        task_id="t-jsonl",
        task_type="user_turn",
        status=TaskStatus.RUNNING,
        owner_kind="conversation",
        owner_id="c-jsonl",
        phase="running",
        input_json="{}",
        state_json="{}",
    )
    await storage.insert_task(task)

    rec = await mgr.create_challenge(
        task_id="t-jsonl",
        conversation_id="c-jsonl",
        tool_name="wipe_disk",
        args={"partition": "/dev/sda1"},
    )
    assert rec.status == ApprovalStatus.PENDING

    await mgr.resolve(
        rec.challenge_token,
        approved=True,
        override_args={"partition": "/dev/sdb1"},
        approver="root",
    )

    await storage.close()

    # Reopen to verify ledger replay from JSONL file
    storage2 = JsonlStorage(jsonl_file)
    await storage2.init_db()
    mgr2 = ApprovalManager(storage2)

    persisted = await mgr2.get(rec.challenge_token)
    assert persisted is not None
    assert persisted.status == ApprovalStatus.APPROVED
    dec = ApprovalDecision.model_validate_json(persisted.decision_json)
    assert dec.approver == "root"
    assert dec.override_args_json is not None
    override = json.loads(dec.override_args_json)
    assert override == {"partition": "/dev/sdb1"}
    await storage2.close()


@pytest.mark.asyncio
async def test_approval_dynamic_expiration(temp_db: str) -> None:
    async with AioSqliteStorage(temp_db) as storage:
        mgr = ApprovalManager(storage)
        conv = ConversationRecord(conversation_id="c-exp", agent_config_json="{}")
        await storage.insert_conversation(conv)
        task = TaskRecord(
            task_id="t-exp",
            task_type="user_turn",
            status=TaskStatus.RUNNING,
            owner_kind="conversation",
            owner_id="c-exp",
            phase="running",
            input_json="{}",
            state_json="{}",
        )
        await storage.insert_task(task)

        # Create challenge already expired
        rec = await mgr.create_challenge(
            task_id="t-exp",
            conversation_id="c-exp",
            tool_name="urgent_op",
            args={},
            timeout_seconds=-5.0,  # Expired 5 seconds ago
        )
        assert rec.status == ApprovalStatus.PENDING

        # Calling get() detects expiration dynamically
        checked = await mgr.get(rec.challenge_token)
        assert checked is not None
        assert checked.status == ApprovalStatus.EXPIRED

        # Trying to resolve expired token raises InvalidApprovalDecisionError
        with pytest.raises(InvalidApprovalDecisionError):
            await mgr.resolve(rec.challenge_token, approved=True)


@pytest.mark.asyncio
async def test_cannot_resolve_already_resolved(temp_db: str) -> None:
    async with AioSqliteStorage(temp_db) as storage:
        mgr = ApprovalManager(storage)
        conv = ConversationRecord(conversation_id="c2", agent_config_json="{}")
        await storage.insert_conversation(conv)
        task = TaskRecord(
            task_id="t2",
            task_type="user_turn",
            status=TaskStatus.RUNNING,
            owner_kind="conversation",
            owner_id="c2",
            phase="running",
            input_json="{}",
            state_json="{}",
        )
        await storage.insert_task(task)

        rec = await mgr.create_challenge(
            task_id="t2",
            conversation_id="c2",
            tool_name="publish_post",
            args={"id": 1},
        )
        await mgr.resolve(rec.challenge_token, approved=True)

        with pytest.raises(InvalidApprovalDecisionError) as exc_info:
            await mgr.resolve(rec.challenge_token, approved=False)
        assert "already resolved" in str(exc_info.value)


@pytest.mark.asyncio
async def test_durable_tool_approval_gate_deferred(temp_db: str) -> None:
    @durable_tool(require_approval=True, approval_prompt="Review destructive action")
    async def drop_table(name: str) -> str:
        return f"Dropped {name}"

    async with AioSqliteStorage(temp_db) as storage:
        conv = ConversationRecord(conversation_id="c-gate", agent_config_json="{}")
        await storage.insert_conversation(conv)
        task = TaskRecord(
            task_id="t-gate",
            task_type="user_turn",
            status=TaskStatus.RUNNING,
            owner_kind="conversation",
            owner_id="c-gate",
            phase="running",
            input_json="{}",
            state_json="{}",
        )
        await storage.insert_task(task)

        # First invocation should be intercepted and raise ApprovalRequiredError
        async with durable_execution_context(storage, "t-gate", "c-gate") as ctx:
            with pytest.raises(ApprovalRequiredError) as exc_info:
                await drop_table("users")

            assert "requires human approval" in str(exc_info.value)
            token = exc_info.value.challenge_token
            assert token.startswith("appr-")
            assert ctx.last_challenge_token == token

        # Verify tool invocation record is DEFERRED
        invocations = await storage.get_tool_invocations_by_task("t-gate")
        assert len(invocations) == 1
        assert invocations[0].status == ToolInvocationStatus.DEFERRED

        # Verify task is transitioned to WAITING
        updated_task = await storage.get_task("t-gate")
        assert updated_task is not None
        assert updated_task.status == TaskStatus.WAITING
        assert updated_task.phase == "waiting_approval"


@pytest.mark.asyncio
async def test_durable_tool_approval_approved_and_resumed(temp_db: str) -> None:
    call_count = 0

    @durable_tool(require_approval=True)
    async def restart_service(name: str) -> str:
        nonlocal call_count
        call_count += 1
        return f"Service {name} restarted successfully"

    async with AioSqliteStorage(temp_db) as storage:
        conv = ConversationRecord(conversation_id="c-res", agent_config_json="{}")
        await storage.insert_conversation(conv)
        task = TaskRecord(
            task_id="t-res",
            task_type="user_turn",
            status=TaskStatus.RUNNING,
            owner_kind="conversation",
            owner_id="c-res",
            phase="running",
            input_json="{}",
            state_json="{}",
        )
        await storage.insert_task(task)

        mgr = ApprovalManager(storage)

        # 1. First run: raises ApprovalRequiredError
        challenge_token = ""
        async with durable_execution_context(storage, "t-res", "c-res"):
            with pytest.raises(ApprovalRequiredError) as exc_info:
                await restart_service("nginx")
            challenge_token = exc_info.value.challenge_token

        assert call_count == 0

        # 2. Operator resolves approval
        await mgr.resolve(challenge_token, approved=True, approver="operator-alice")

        # 3. Resumption: executes the tool successfully
        async with durable_execution_context(storage, "t-res", "c-res"):
            res = await restart_service("nginx")
            assert res == "Service nginx restarted successfully"

        assert call_count == 1
        invocations = await storage.get_tool_invocations_by_task("t-res")
        assert invocations[0].status == ToolInvocationStatus.DEFERRED
        assert invocations[-1].status == ToolInvocationStatus.COMPLETED


@pytest.mark.asyncio
async def test_durable_tool_approval_override_args(temp_db: str) -> None:
    executed_arg = ""

    @durable_tool(require_approval=True)
    async def delete_records(scope: str) -> str:
        nonlocal executed_arg
        executed_arg = scope
        return f"Deleted {scope}"

    async with AioSqliteStorage(temp_db) as storage:
        conv = ConversationRecord(conversation_id="c-ov", agent_config_json="{}")
        await storage.insert_conversation(conv)
        task = TaskRecord(
            task_id="t-ov",
            task_type="user_turn",
            status=TaskStatus.RUNNING,
            owner_kind="conversation",
            owner_id="c-ov",
            phase="running",
            input_json="{}",
            state_json="{}",
        )
        await storage.insert_task(task)

        mgr = ApprovalManager(storage)

        token = ""
        async with durable_execution_context(storage, "t-ov", "c-ov"):
            with pytest.raises(ApprovalRequiredError) as exc:
                await delete_records("all_records")
            token = exc.value.challenge_token

        # Operator overrides scope to "stale_records_only"
        await mgr.resolve(
            token,
            approved=True,
            override_args={"scope": "stale_records_only"},
            approver="lead-sec",
        )

        async with durable_execution_context(storage, "t-ov", "c-ov"):
            res = await delete_records("all_records")
            assert res == "Deleted stale_records_only"
            assert executed_arg == "stale_records_only"


@pytest.mark.asyncio
async def test_durable_tool_approval_rejection(temp_db: str) -> None:
    @durable_tool(require_approval=True)
    async def shutdown_system() -> str:
        return "Shut down"

    async with AioSqliteStorage(temp_db) as storage:
        conv = ConversationRecord(conversation_id="c-rej", agent_config_json="{}")
        await storage.insert_conversation(conv)
        task = TaskRecord(
            task_id="t-rej",
            task_type="user_turn",
            status=TaskStatus.RUNNING,
            owner_kind="conversation",
            owner_id="c-rej",
            phase="running",
            input_json="{}",
            state_json="{}",
        )
        await storage.insert_task(task)

        mgr = ApprovalManager(storage)

        token = ""
        async with durable_execution_context(storage, "t-rej", "c-rej"):
            with pytest.raises(ApprovalRequiredError) as exc:
                await shutdown_system()
            token = exc.value.challenge_token

        await mgr.resolve(token, approved=False, reason="Maintenance window closed")

        async with durable_execution_context(storage, "t-rej", "c-rej"):
            with pytest.raises(ApprovalRejectedError) as rej_exc:
                await shutdown_system()
            assert "Maintenance window closed" in str(rej_exc.value)


@pytest.mark.asyncio
async def test_extension_hook_approval_gate(temp_db: str) -> None:
    # A standard tool without require_approval
    @durable_tool
    async def execute_query(query: str) -> str:
        return f"Executed {query}"

    def firewall(info: ToolCallInfo, ctx: HookContext) -> HookDecision | None:
        if info.name == "execute_query" and "DROP" in str(info.args.get("query", "")):
            return HookDecision(
                require_approval=True,
                approval_prompt="Dangerous SQL detected: requires security signoff",
            )
        return None

    ext = define_extension("sql-firewall", hooks=[hook(name="firewall", before_tool=firewall)])
    global_extension_registry.install(ext)

    async with AioSqliteStorage(temp_db) as storage:
        conv = ConversationRecord(conversation_id="c-hook", agent_config_json="{}")
        await storage.insert_conversation(conv)
        task = TaskRecord(
            task_id="t-hook",
            task_type="user_turn",
            status=TaskStatus.RUNNING,
            owner_kind="conversation",
            owner_id="c-hook",
            phase="running",
            input_json="{}",
            state_json="{}",
        )
        await storage.insert_task(task)

        # Normal query passes without gate
        async with durable_execution_context(storage, "t-hook", "c-hook"):
            res = await execute_query("SELECT 1")
            assert res == "Executed SELECT 1"

        # Dangerous query intercepted by hook
        async with durable_execution_context(storage, "t-hook", "c-hook"):
            with pytest.raises(ApprovalRequiredError) as exc:
                await execute_query("DROP TABLE users")
            assert "requires human approval" in str(exc.value)
            assert "Dangerous SQL detected" in str(exc.value)


@pytest.mark.asyncio
async def test_crash_recovery_resumption_offline_approval(temp_db: str) -> None:
    @durable_tool(require_approval=True)
    async def deploy_release(tag: str) -> str:
        return f"Release {tag} deployed"

    agent = Agent(TestModel(custom_output_text="deployment finished"))

    async with AioSqliteStorage(temp_db) as storage:
        conv = ConversationRecord(conversation_id="c-recov", agent_config_json="{}")
        await storage.insert_conversation(conv)

        mgr = ApprovalManager(storage)
        task = TaskRecord(
            task_id="t-recov",
            task_type="user_turn",
            status=TaskStatus.RUNNING,
            owner_kind="conversation",
            owner_id="c-recov",
            phase="running",
            input_json=json.dumps({"prompt": "deploy v2"}),
            state_json=json.dumps({"phase": "running"}),
        )
        await storage.insert_task(task)

        # Trigger approval gate
        token = ""
        async with durable_execution_context(storage, "t-recov", "c-recov"):
            with pytest.raises(ApprovalRequiredError) as exc:
                await deploy_release("v2.0.0")
            token = exc.value.challenge_token

        # Crash recovery check 1: Task remains WAITING while approval is pending
        recov_mgr = CrashRecoveryManager(storage, agent=agent)
        reconciled = await recov_mgr.reconcile_task("t-recov")
        assert reconciled is not None
        assert reconciled.new_status == TaskStatus.WAITING

        # Offline operator approves the challenge token
        await mgr.resolve(token, approved=True, approver="release-manager")

        # Crash recovery check 2: System restarts, reconcile detects approval
        reconciled2 = await recov_mgr.reconcile_task("t-recov")
        assert reconciled2 is not None
        assert reconciled2.new_status == TaskStatus.RUNNING
        assert reconciled2.reconciled_phase == "resumed_approved"


@pytest.mark.asyncio
async def test_harness_view_state_and_approval_resolution(temp_db: str) -> None:
    @durable_tool(require_approval=True, approval_prompt="Operator review needed")
    async def format_drive(drive: str) -> str:
        return f"Formatted {drive}"

    agent = Agent(TestModel(custom_output_text="all done"))
    async with AioSqliteStorage(temp_db) as storage:
        harness = DurableHarness(
            agent=agent,
            storage=storage,
            conversation_id="conv-harness-appr",
        )
        await harness._ensure_conversation()

        task = TaskRecord(
            task_id="task-test",
            task_type="user_turn",
            status=TaskStatus.WAITING,
            owner_kind="conversation",
            owner_id="conv-harness-appr",
            phase="waiting_approval",
            input_json="{}",
            state_json="{}",
        )
        await storage.insert_task(task)

        # Create challenge via harness.approvals
        rec = await harness.approvals.create_challenge(
            task_id="task-test",
            conversation_id="conv-harness-appr",
            tool_name="format_drive",
            args={"drive": "C:"},
            prompt_message="Format C: drive?",
        )

        # View state should list pending approval
        vs = await harness.view_state()
        assert len(vs.value.pending_approvals) == 1
        assert vs.value.pending_approvals[0]["challenge_token"] == rec.challenge_token

        # List pending via harness convenience method
        pending_list = await harness.list_pending_approvals()
        assert len(pending_list) == 1
        assert pending_list[0].challenge_token == rec.challenge_token

        # Resolve via harness convenience method
        resolved = await harness.resolve_approval(
            challenge_token=rec.challenge_token,
            approved=True,
            reason="Confirmed by IT",
            approver="admin-dan",
        )
        assert resolved.status == ApprovalStatus.APPROVED

        # View state should now have 0 pending approvals
        vs_after = await harness.view_state()
        assert len(vs_after.value.pending_approvals) == 0
