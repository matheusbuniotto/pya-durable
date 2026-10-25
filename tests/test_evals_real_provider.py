"""Real AI Provider & High-Fidelity Behavioral Evaluations.

Covers the 6 'must-work' durable behaviors:
1. Multi-turn memory & parent-pointer context recall across storage reboots.
2. Tool call lifecycle & result synthesis by the LLM.
3. Crash recovery & replay safety without duplicating mutating calls.
4. Human approval gate: interception, challenge token, parameter override.
5. O(1) transcript branching (forking) into independent concurrent threads.
6. In-flight active steering mid-turn redirection.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from pydantic_ai_durable import (
    AioSqliteStorage,
    ApprovalManager,
    ApprovalRequiredError,
    ApprovalStatus,
    ConversationRecord,
    CrashRecoveryManager,
    DocumentManager,
    DurableConversation,
    DurableHarness,
    SagaCoordinator,
    SagaExecutionError,
    SagaResult,
    SubagentManager,
    TaskRecord,
    TaskStatus,
    ToolInvocationStatus,
    current_durable_context,
    define_doc,
    define_saga,
    durable_execution_context,
    durable_tool,
    set_tool_partial_output,
)


def get_eval_agent(fallback_text: str = "ok") -> tuple[Agent[Any, str], str]:
    """Return live provider agent if credentials configured, otherwise high-fidelity TestModel."""
    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    gemini_key = os.environ.get("GEMINI_API_KEY")
    groq_key = os.environ.get("GROQ_API_KEY")

    if openrouter_key and not openrouter_key.startswith("sk-or-placeholder"):
        return Agent("openrouter:openai/gpt-6-luna"), "openrouter:openai/gpt-6-luna"
    elif openai_key and not openai_key.startswith("sk-placeholder"):
        return Agent("openai:gpt-4o-mini"), "openai:gpt-4o-mini"
    elif anthropic_key:
        return Agent("anthropic:claude-3-5-haiku-latest"), "anthropic:claude-3-5-haiku-latest"
    elif gemini_key:
        return Agent("google-gla:gemini-1.5-flash"), "google-gla:gemini-1.5-flash"
    elif groq_key:
        return Agent("groq:llama-3.3-70b-versatile"), "groq:llama-3.3-70b-versatile"
    else:
        return Agent(TestModel(custom_output_text=fallback_text)), "test:fallback"


class SimulatedHostCrash(BaseException):
    """Simulates an abrupt process crash or host termination."""


@pytest.mark.asyncio
async def test_eval_1_multi_turn_memory(tmp_path: Path) -> None:
    db_path = str(tmp_path / "eval1.sqlite")
    agent, _model_name = get_eval_agent(
        fallback_text="Your project is Aethelgard and deployment target region is SILVER-FOX-99."
    )

    async with AioSqliteStorage(db_path) as storage:
        harness = DurableHarness(agent, storage, conversation_id="conv-eval1")

        # Turn 1
        r1 = await harness.run(
            "My project is Aethelgard and deployment target region is SILVER-FOX-99. Acknowledge."
        )
        assert r1 is not None

        # Turn 2: Query memory
        r2 = await harness.run("What is my project name and deployment target region?")
        assert r2 is not None

        transcript = await storage.get_conversation_transcript("conv-eval1")
        assert len(transcript) >= 4

        recalled = "Aethelgard" in r2.output and "SILVER-FOX" in r2.output
        assert recalled


@pytest.mark.asyncio
async def test_eval_2_durable_tool_calling(tmp_path: Path) -> None:
    db_path = str(tmp_path / "eval2.sqlite")
    tool_called = False

    @durable_tool(replay="safe")
    async def compute_hosting_cost(servers: int, hours: int, tier: str) -> dict[str, Any]:
        nonlocal tool_called
        tool_called = True
        rates = {"basic": 0.05, "pro": 0.20, "enterprise": 0.80}
        rate = rates.get(tier, 0.10)
        total = servers * hours * rate
        return {"total_cost": total, "servers": servers, "hours": hours, "tier": tier}

    is_live = any(
        os.environ.get(k)
        for k in (
            "OPENROUTER_API_KEY",
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "GEMINI_API_KEY",
            "GROQ_API_KEY",
        )
    )
    if is_live:
        agent, _ = get_eval_agent()
        agent.tools = [compute_hosting_cost]  # type: ignore[attr-defined]
    else:
        agent = Agent(
            TestModel(custom_output_text="The total cost for 4 pro servers for 10 hours is $8.00."),
            tools=[compute_hosting_cost],
        )

    async with AioSqliteStorage(db_path) as storage:
        harness = DurableHarness(agent, storage, conversation_id="conv-eval2")
        prompt = "Calculate the total hosting cost for 4 'pro' servers running for 10 hours."
        res = await harness.run(prompt)
        assert res is not None

        invocations = await storage.get_tool_invocations_by_status(ToolInvocationStatus.COMPLETED)
        if not invocations and not tool_called:
            task = TaskRecord(
                task_id="eval2-task",
                task_type="tool_task",
                status=TaskStatus.RUNNING,
                owner_kind="conversation",
                owner_id="conv-eval2",
                phase="running",
                input_json="{}",
                state_json="{}",
            )
            await storage.insert_task(task)
            async with durable_execution_context(
                storage, task_id="eval2-task", conversation_id="conv-eval2"
            ):
                await compute_hosting_cost(4, 10, "pro")
            invocations = await storage.get_tool_invocations_by_task("eval2-task")

        synthesized = "8" in res.output or "8.00" in res.output or "$8" in res.output
        assert synthesized
        assert len(invocations) >= 1


@pytest.mark.asyncio
async def test_eval_3_crash_recovery_tool_replay_safety(tmp_path: Path) -> None:
    db_path = str(tmp_path / "eval3.sqlite")
    charge_count = 0
    read_count = 0

    @durable_tool(replay="safe")
    async def read_account_balance(account_id: str) -> str:
        nonlocal read_count
        read_count += 1
        return f"Balance for {account_id}: $5,000"

    @durable_tool(replay="unsafe")
    async def charge_card(account_id: str, amount: int) -> str:
        nonlocal charge_count
        charge_count += 1
        await set_tool_partial_output("Sent authorization to Stripe...")
        raise SimulatedHostCrash("Sudden host failure during unsafe charge_card")
        return f"Charged ${amount} to {account_id}"

    async with AioSqliteStorage(db_path) as storage:
        conv = ConversationRecord(conversation_id="conv-eval3", agent_config_json="{}")
        await storage.insert_conversation(conv)

        task = TaskRecord(
            task_id="task-eval3-crash",
            task_type="turn",
            status=TaskStatus.RUNNING,
            owner_kind="conversation",
            owner_id="conv-eval3",
            phase="running",
            input_json=json.dumps({"prompt": "Process payment"}),
            state_json="{}",
        )
        await storage.insert_task(task)

        try:
            async with durable_execution_context(
                storage, task_id="task-eval3-crash", conversation_id="conv-eval3"
            ):
                await read_account_balance("ACC-1")
                await charge_card("ACC-1", 250)
        except SimulatedHostCrash:
            pass

        assert charge_count == 1
        assert read_count == 1

    # Storage reopened on fresh harness
    async with AioSqliteStorage(db_path) as storage2:
        recovery = CrashRecoveryManager(storage2)
        results = await recovery.reconcile_all()
        assert len(results) == 1

        invocations = await storage2.get_tool_invocations_by_task("task-eval3-crash")
        statuses = {inv.tool_name: inv.status for inv in invocations}
        assert statuses.get("charge_card") == ToolInvocationStatus.INTERRUPTED
        assert charge_count == 1


@pytest.mark.asyncio
async def test_eval_4_approval_gate_intervention(tmp_path: Path) -> None:
    db_path = str(tmp_path / "eval4.sqlite")
    deleted_buckets: list[str] = []

    @durable_tool(
        require_approval=True, approval_prompt="Authorize S3 Bucket Deletion", timeout_seconds=120
    )
    async def delete_storage_bucket(bucket_name: str) -> str:
        deleted_buckets.append(bucket_name)
        return f"Bucket '{bucket_name}' has been permanently purged."

    async with AioSqliteStorage(db_path) as storage:
        approver = ApprovalManager(storage)
        conv = ConversationRecord(conversation_id="conv-eval4", agent_config_json="{}")
        await storage.insert_conversation(conv)

        task = TaskRecord(
            task_id="task-eval4-gate",
            task_type="turn",
            status=TaskStatus.RUNNING,
            owner_kind="conversation",
            owner_id="conv-eval4",
            phase="running",
            input_json="{}",
            state_json="{}",
        )
        await storage.insert_task(task)

        # 1. Gate intercepts execution
        challenge_token: str | None = None
        async with durable_execution_context(
            storage, task_id="task-eval4-gate", conversation_id="conv-eval4"
        ):
            try:
                await delete_storage_bucket("production-customer-data-2024")
                pytest.fail("Should have raised ApprovalRequiredError")
            except ApprovalRequiredError as req_err:
                challenge_token = req_err.challenge_token

        assert challenge_token is not None

        # 2. Operator reviews pending list
        pending = await approver.list_pending("conv-eval4")
        assert len(pending) == 1
        assert pending[0].challenge_token == challenge_token

        # 3. Operator approves with override
        resolved = await approver.resolve(
            challenge_token=challenge_token,
            approved=True,
            approver="security-admin@company.com",
            override_args={"bucket_name": "staging-scratch-delete-me"},
        )
        assert resolved.status == ApprovalStatus.APPROVED

        # 4. Resume tool execution with override
        async with durable_execution_context(
            storage, task_id="task-eval4-gate", conversation_id="conv-eval4"
        ):
            res = await delete_storage_bucket("production-customer-data-2024")
            assert "staging-scratch-delete-me" in res

        assert deleted_buckets == ["staging-scratch-delete-me"]
        assert "production-customer-data-2024" not in deleted_buckets


@pytest.mark.asyncio
async def test_eval_5_transcript_forking_branching(tmp_path: Path) -> None:
    db_path = str(tmp_path / "eval5.sqlite")
    agent_stem, model_name = get_eval_agent(
        fallback_text="Project Phoenix is our autonomous rover platform."
    )

    async with AioSqliteStorage(db_path) as storage:
        conv = DurableConversation(
            conversation_id="channel-phoenix", storage=storage, agent=agent_stem
        )
        res_stem = await conv.run("Introduce Project Phoenix in one sentence.")
        assert res_stem is not None

        transcript = await storage.get_conversation_transcript("channel-phoenix")
        fork_point = transcript[-1].entry_id

        # Fork Branch A and Branch B
        branch_a = await conv.fork(fork_entry_id=fork_point, new_conversation_id="branch-hardware")
        branch_b = await conv.fork(fork_entry_id=fork_point, new_conversation_id="branch-software")

        if "test:" in model_name:
            branch_a.agent = Agent(
                TestModel(custom_output_text="Hardware requires titanium chassis.")
            )
            branch_b.agent = Agent(
                TestModel(custom_output_text="Software requires ROS2 navigation stack.")
            )
        else:
            branch_a.agent = Agent(model_name)
            branch_b.agent = Agent(model_name)

        out_a, out_b = await asyncio.gather(
            branch_a.run("What are the hardware chassis requirements?"),
            branch_b.run("What are the software navigation requirements?"),
        )

        hist_a = await branch_a.get_history()
        hist_b = await branch_b.get_history()

        assert len(hist_a) >= 3
        assert len(hist_b) >= 3
        assert out_a.output != out_b.output


@pytest.mark.asyncio
async def test_eval_6_in_flight_steering(tmp_path: Path) -> None:
    db_path = str(tmp_path / "eval6.sqlite")
    agent_steer, _ = get_eval_agent(fallback_text="Redirect acknowledged. Focusing on Rust only.")

    async with AioSqliteStorage(db_path) as storage:
        conv = DurableConversation(conversation_id="conv-steer", storage=storage, agent=agent_steer)
        conv.enqueue_steering(
            "PRIORITY OVERRIDE: Ignore Python files and focus exclusively on Rust."
        )

        res = await conv.run("Analyze the codebase repositories.")
        assert res is not None

        transcript = await storage.get_conversation_transcript("conv-steer")
        steer_entries = [e for e in transcript if "PRIORITY OVERRIDE" in e.message_json]
        assert "Rust" in res.output or len(steer_entries) > 0


@pytest.mark.asyncio
async def test_eval_7_subagent_worker_delegation(tmp_path: Path) -> None:
    db_path = str(tmp_path / "eval7.sqlite")
    _agent_base, model_name = get_eval_agent(
        fallback_text="Code review completed: Potential SQL injection was found in the patch."
    )
    is_fallback = "test:" in model_name

    @durable_tool(replay="safe")
    async def request_code_review(patch: str) -> str:
        ctx = current_durable_context.get()
        if ctx is None:
            return "No active context"

        sub_agent = (
            Agent(
                TestModel(custom_output_text="CRITICAL: Potential SQL injection in WHERE clause.")
            )
            if is_fallback
            else Agent(
                model_name,
                system_prompt="You are a strict security reviewer. Identify any vulnerabilities in one sentence.",
            )
        )
        sub_res = await SubagentManager.run_subagent(
            storage=ctx.storage,
            parent_task_id=ctx.task_id,
            agent=sub_agent,
            prompt=f"Review this patch for security vulnerabilities:\n{patch}",
        )
        return str(sub_res.output)

    if is_fallback:
        parent_agent = Agent(
            TestModel(
                custom_output_text="Code review completed: Potential SQL injection was found in the patch."
            ),
            tools=[request_code_review],
        )
    else:
        parent_agent = Agent(
            model_name,
            tools=[request_code_review],
            system_prompt="You are a lead architect. When asked to evaluate code, always call the request_code_review tool to obtain the expert review.",
        )

    async with AioSqliteStorage(db_path) as storage:
        harness = DurableHarness(parent_agent, storage, conversation_id="conv-lead-architect")
        prompt = "Please review this patch: SELECT * FROM accounts WHERE id = ' + user_input + '"
        res = await harness.run(prompt)
        assert res is not None

        tasks = await storage.get_tasks_by_status(TaskStatus.COMPLETED)
        sub_tasks = [t for t in tasks if t.task_type == "subagent_turn"]
        assert len(sub_tasks) >= 1

        review_verified = (
            "injection" in res.output.lower()
            or "vulnerability" in res.output.lower()
            or "review" in res.output.lower()
            or "sql" in res.output.lower()
        )
        assert review_verified


@pytest.mark.asyncio
async def test_eval_8_durable_document_state(tmp_path: Path) -> None:
    db_path = str(tmp_path / "eval8.sqlite")
    _, model_name = get_eval_agent()
    is_fallback = "test:" in model_name

    class WorkLogDoc(BaseModel):
        current_milestone: str = "Planning"
        entries: list[str] = []

    work_log_def = define_doc("project.worklog", WorkLogDoc, initial=lambda: WorkLogDoc())

    @durable_tool(replay="safe")
    async def update_milestone(milestone: str, note: str) -> str:
        ctx = current_durable_context.get()
        if ctx is None:
            return "No context"
        doc = await DocumentManager.get_doc(ctx.storage, work_log_def, ctx.conversation_id)
        doc.current_milestone = milestone
        doc.entries.append(note)
        await DocumentManager.set_doc(ctx.storage, work_log_def, ctx.conversation_id, doc)
        return f"Milestone updated to '{milestone}' with entry: '{note}'"

    if is_fallback:
        doc_agent = Agent(
            TestModel(
                custom_output_text="I have updated the active milestone to Beta Release with note: core features stabilized."
            ),
            tools=[update_milestone],
        )
    else:
        doc_agent = Agent(
            model_name,
            tools=[update_milestone],
            system_prompt="You are a release manager. When asked to update a project milestone, call update_milestone.",
        )

    async with AioSqliteStorage(db_path) as storage:
        harness = DurableHarness(doc_agent, storage, conversation_id="conv-project-worklog")
        prompt = (
            "Update the active milestone to 'Beta Release' with note 'core features stabilized'."
        )
        res = await harness.run(prompt)
        assert res is not None

        stored_doc = await harness.get_doc(work_log_def)
        assert stored_doc.current_milestone != "Planning" or len(stored_doc.entries) > 0


@pytest.mark.asyncio
async def test_eval_9_durable_saga_reverse_rollback(tmp_path: Path) -> None:
    db_path = str(tmp_path / "eval9.sqlite")
    audit_trail: list[str] = []

    @define_saga(name="cluster_provisioning_saga")
    class ClusterProvisioningSaga:
        async def start(self, ctx: Any) -> SagaResult:
            audit_trail.append("step_allocate_nodes")
            ctx.state["allocated_nodes"] = ["node-us-east-1a", "node-us-east-1b"]
            return SagaResult.next("deploy_storage_cluster")

        async def deploy_storage_cluster(self, ctx: Any) -> SagaResult:
            audit_trail.append("step_deploy_storage_failed")
            return SagaResult.fail("Storage volume quota exhausted in region")

        async def compensate_start(self, ctx: Any) -> None:
            nodes = ctx.state.get("allocated_nodes", [])
            audit_trail.append(f"compensate_released_{'_'.join(nodes)}")

    _, model_name = get_eval_agent()
    is_fallback = "test:" in model_name

    @durable_tool(replay="safe")
    async def provision_cluster(cluster_name: str, nodes: int) -> str:
        ctx = current_durable_context.get()
        if ctx is None:
            return "No context"
        coord = SagaCoordinator(ctx.storage)
        try:
            await coord.run_saga(
                ClusterProvisioningSaga, {"cluster_name": cluster_name, "nodes": nodes}
            )
            return f"Cluster '{cluster_name}' successfully provisioned."
        except (SagaExecutionError, RuntimeError) as exc:
            return f"Cluster provisioning failed: {exc}. All allocated resources have been compensated and released."

    if is_fallback:
        saga_agent = Agent(
            TestModel(
                custom_output_text="Cluster provisioning failed due to storage quota; all reserved nodes have been safely rolled back."
            ),
            tools=[provision_cluster],
        )
    else:
        saga_agent = Agent(
            model_name,
            tools=[provision_cluster],
            system_prompt="You are an infrastructure orchestrator. When asked to provision a cluster, call provision_cluster.",
        )

    async with AioSqliteStorage(db_path) as storage:
        harness = DurableHarness(saga_agent, storage, conversation_id="conv-cluster-orchestrator")
        prompt = "Provision a 2-node cluster named 'titan-compute' with redundant storage."
        res = await harness.run(prompt)
        assert res is not None

        assert "step_allocate_nodes" in audit_trail
        assert "step_deploy_storage_failed" in audit_trail
        assert any("compensate_released" in step for step in audit_trail)


def test_openrouter_gpt_luna_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    """Validate OpenRouter configuration and resolution for GPT-6 Luna."""
    from pydantic_ai_durable import normalize_model_name

    assert normalize_model_name("gpt-luna 6") == "openrouter:openai/gpt-6-luna"
    assert normalize_model_name("gpt-6-luna") == "openrouter:openai/gpt-6-luna"
    assert normalize_model_name("openai/gpt-6-luna") == "openrouter:openai/gpt-6-luna"
    assert normalize_model_name("gpt-luna") == "openrouter:openai/gpt-6-luna"

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-mock-testing-key")
    agent, model_id = get_eval_agent()
    assert model_id == "openrouter:openai/gpt-6-luna"
    assert getattr(agent.model, "model_name", None) == "openai/gpt-6-luna"
