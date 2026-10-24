import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage

from pya_durable.engine import DurableStepEngine
from pya_durable.harness import DurableHarness
from pya_durable.models import (
    ConversationRecord,
    TaskRecord,
    TaskStatus,
)
from pya_durable.recovery import CrashRecoveryManager
from pya_durable.storage import AioSqliteStorage

logger = logging.getLogger("pya_durable.subagents")


@dataclass
class SubagentResult:
    conversation_id: str
    task_id: str
    output: Any
    all_messages: list[ModelMessage] = field(default_factory=list)
    token_usage: dict[str, int] = field(default_factory=dict)


class SubagentManager:
    @staticmethod
    async def create_subagent(
        storage: AioSqliteStorage,
        parent_task_id: str,
        agent_config: dict[str, Any] | None = None,
    ) -> ConversationRecord:
        sub_conv_id = f"subconv-{uuid.uuid4().hex[:12]}"

        parent_conv_id = None
        parent_task = await storage.get_task(parent_task_id)
        if parent_task and parent_task.owner_kind == "conversation":
            parent_conv_id = parent_task.owner_id

        config = {
            "parent_task_id": parent_task_id,
            **(agent_config or {}),
        }

        record = ConversationRecord(
            conversation_id=sub_conv_id,
            parent_conversation_id=parent_conv_id,
            agent_config_json=json.dumps(config),
        )
        await storage.insert_conversation(record)
        return record

    @staticmethod
    async def run_subagent(
        storage: AioSqliteStorage,
        parent_task_id: str,
        agent: Agent[Any, Any],
        prompt: str,
        isolated_tools: list[Any] | None = None,
        request_id: str | None = None,
    ) -> SubagentResult:
        conv = await SubagentManager.create_subagent(storage, parent_task_id)

        if isolated_tools is not None:
            scoped_agent = Agent(
                model=agent.model,
                tools=isolated_tools,
                system_prompt=getattr(agent, "_system_prompts", ()),
            )
        else:
            scoped_agent = agent

        sub_task_id = f"subtask-{uuid.uuid4().hex[:12]}"
        sub_task = TaskRecord(
            task_id=sub_task_id,
            task_type="subagent_turn",
            status=TaskStatus.RUNNING,
            owner_kind="task",
            owner_id=parent_task_id,
            phase="running",
            input_json=json.dumps({"prompt": prompt}),
            state_json=json.dumps({"conversation_id": conv.conversation_id}),
            policy="failFast",
        )
        await storage.insert_task(sub_task)

        harness = DurableHarness(
            scoped_agent,
            storage,
            conversation_id=conv.conversation_id,
        )

        turn_result = await harness.run(
            prompt=prompt,
            request_id=request_id,
        )

        out_str = (
            turn_result.output.model_dump_json()
            if hasattr(turn_result.output, "model_dump_json")
            else str(turn_result.output)
        )
        await storage.update_task_state(
            task_id=sub_task_id,
            status=TaskStatus.COMPLETED,
            phase="completed",
            state_json=json.dumps({"output": out_str, "usage": turn_result.token_usage, "conversation_id": conv.conversation_id}),
            checkpoint_json=turn_result.leaf_entry_id,
        )

        return SubagentResult(
            conversation_id=conv.conversation_id,
            task_id=sub_task_id,
            output=turn_result.output,
            all_messages=turn_result.all_messages,
            token_usage=turn_result.token_usage,
        )

    @staticmethod
    async def resume_subagent(
        storage: AioSqliteStorage,
        conversation_id: str,
        agent: Agent[Any, Any],
    ) -> Any:
        conn = await storage.get_connection()
        async with conn.execute(
            """
            SELECT task_id, input_json, checkpoint_json FROM tasks
            WHERE (owner_id = ? OR state_json LIKE ?)
              AND status IN ('running', 'pending', 'waiting')
            ORDER BY created_at DESC LIMIT 1
            """,
            (conversation_id, f'%"{conversation_id}"%'),
        ) as cur:
            row = await cur.fetchone()

        if row:
            target_task_id, input_json, checkpoint_json = row[0], row[1], row[2]
            prompt = ""
            try:
                inp = json.loads(input_json) if input_json else {}
                prompt = inp.get("prompt", "")
            except Exception:
                prompt = input_json or ""

            harness = DurableHarness(agent, storage, conversation_id=conversation_id)
            history = await harness.get_history()
            engine = DurableStepEngine(storage)

            turn_result = await engine.run_turn(
                agent=agent,
                user_prompt=prompt,
                conversation_id=conversation_id,
                task_id=target_task_id,
                parent_entry_id=checkpoint_json,
                message_history=history if history else None,
            )

            out_str = (
                turn_result.output.model_dump_json()
                if hasattr(turn_result.output, "model_dump_json")
                else str(turn_result.output)
            )
            await storage.update_task_state(
                task_id=target_task_id,
                status=TaskStatus.COMPLETED,
                phase="completed",
                state_json=json.dumps({"output": out_str, "usage": turn_result.token_usage, "conversation_id": conversation_id}),
                checkpoint_json=turn_result.leaf_entry_id,
            )
            return turn_result

        manager = CrashRecoveryManager(storage, agent=agent)
        return await manager.reconcile_all()
