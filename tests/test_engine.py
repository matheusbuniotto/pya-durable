import json
import logging
import os
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.test import TestModel

from pya_durable.engine import DurableStepEngine
from pya_durable.harness import DurableHarness
from pya_durable.models import (
    ConversationRecord,
    TaskRecord,
    TaskStatus,
    TranscriptEntryKind,
)
from pya_durable.storage import AioSqliteStorage

os.environ["PYDANTIC_AI_NO_BANNER"] = "1"


@pytest.fixture
def db_path(tmp_path: Path) -> str:
    return str(tmp_path / "test_engine.db")


@pytest.mark.asyncio
async def test_iter_turn_commits_nodes(db_path: str, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    async with AioSqliteStorage(db_path) as storage:
        conv = ConversationRecord(conversation_id="conv-iter", agent_config_json="{}")
        await storage.insert_conversation(conv)

        task = TaskRecord(
            task_id="task-iter",
            task_type="test",
            status=TaskStatus.PENDING,
            owner_kind="conversation",
            owner_id="conv-iter",
            phase="pending",
            input_json="{}",
            state_json="{}",
        )
        await storage.insert_task(task)

        engine = DurableStepEngine(storage)
        agent = Agent(TestModel(custom_output_text="test response"))

        nodes = []
        async for commit in engine.iter_turn(
            agent=agent,
            user_prompt="hello engine",
            conversation_id="conv-iter",
            task_id="task-iter",
        ):
            nodes.append(commit.node_name)

        assert "UserPromptNode" in nodes
        assert "ModelRequestNode" in nodes
        assert "CallToolsNode" in nodes
        assert "End" in nodes

        transcript = await storage.get_conversation_transcript("conv-iter")
        assert len(transcript) == len(nodes)

        assert "Node UserPromptNode committed with entry id" in caplog.text


@pytest.mark.asyncio
async def test_harness_multi_turn(db_path: str) -> None:
    async with AioSqliteStorage(db_path) as storage:
        agent = Agent(TestModel(custom_output_text="response"))
        harness = DurableHarness(agent, storage, conversation_id="conv-multi")

        res1 = await harness.run("turn 1 prompt")
        assert res1.output == "response"

        history1 = await harness.get_history()
        assert len(history1) == 2

        res2 = await harness.run("turn 2 prompt")
        assert res2.output == "response"

        history2 = await harness.get_history()
        assert len(history2) == 4


@pytest.mark.asyncio
async def test_tool_call_interception_and_return(db_path: str) -> None:
    async with AioSqliteStorage(db_path) as storage:
        agent = Agent(TestModel(call_tools=["multiply"]))

        @agent.tool
        def multiply(ctx: RunContext[None], a: int, b: int) -> int:
            return a * b

        harness = DurableHarness(agent, storage, conversation_id="conv-tool")
        res = await harness.run("calculate 6 * 7")

        assert res.output is not None
        transcript = await storage.get_conversation_transcript("conv-tool")
        kinds = [t.kind for t in transcript]
        assert TranscriptEntryKind.TOOL_CALL in kinds


@pytest.mark.asyncio
async def test_structured_output_end_node(db_path: str) -> None:
    class UserProfile(BaseModel):
        name: str = "Alice"
        score: int = 95

    async with AioSqliteStorage(db_path) as storage:
        agent = Agent(TestModel(), output_type=UserProfile)
        harness = DurableHarness(agent, storage, conversation_id="conv-struct")
        res = await harness.run("fetch profile")

        assert isinstance(res.output, UserProfile)
        assert res.output.name == "Alice"
        assert res.output.score == 95


@pytest.mark.asyncio
async def test_empty_prompt_rejection(db_path: str) -> None:
    async with AioSqliteStorage(db_path) as storage:
        agent = Agent(TestModel())
        harness = DurableHarness(agent, storage, conversation_id="conv-empty")

        with pytest.raises(ValueError, match="Prompt cannot be empty"):
            await harness.run("")

        transcript = await storage.get_conversation_transcript("conv-empty")
        assert len(transcript) == 0


@pytest.mark.asyncio
async def test_token_usage_tracking(db_path: str) -> None:
    async with AioSqliteStorage(db_path) as storage:
        agent = Agent(TestModel(custom_output_text="counting tokens"))
        harness = DurableHarness(agent, storage, conversation_id="conv-usage")
        res = await harness.run("count tokens")

        assert "total_tokens" in res.token_usage
        assert res.token_usage["total_tokens"] > 0


@pytest.mark.asyncio
async def test_provider_error_marks_task_failed(db_path: str) -> None:
    class ErrorModel(TestModel):
        async def request(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("429 Rate limit exceeded")

    async with AioSqliteStorage(db_path) as storage:
        agent = Agent(ErrorModel())
        harness = DurableHarness(agent, storage, conversation_id="conv-err")

        with pytest.raises(RuntimeError, match="429 Rate limit"):
            await harness.run("fail please")

        conn = await storage.get_connection()
        async with conn.execute("SELECT status, state_json FROM tasks WHERE owner_id='conv-err'") as cur:
            row = await cur.fetchone()
            assert row is not None
            assert row[0] == TaskStatus.FAILED.value
            state = json.loads(row[1])
            assert "429 Rate limit" in state.get("error", "")


@pytest.mark.asyncio
async def test_early_generator_close_records_partial_state(db_path: str) -> None:
    async with AioSqliteStorage(db_path) as storage:
        conv = ConversationRecord(conversation_id="conv-early", agent_config_json="{}")
        await storage.insert_conversation(conv)

        task = TaskRecord(
            task_id="task-early",
            task_type="test",
            status=TaskStatus.PENDING,
            owner_kind="conversation",
            owner_id="conv-early",
            phase="pending",
            input_json="{}",
            state_json="{}",
        )
        await storage.insert_task(task)

        engine = DurableStepEngine(storage)
        agent = Agent(TestModel(custom_output_text="early break"))

        _res = await engine.run_turn(
            agent=agent,
            user_prompt="stop early",
            conversation_id="conv-early",
            task_id="task-early",
            max_steps=2,
        )

        transcript = await storage.get_conversation_transcript("conv-early")
        assert len(transcript) == 2
        updated_task = await storage.get_task("task-early")
        assert updated_task is not None
        assert updated_task.phase == "interrupted"


def test_pydantic_ai_durable_alias() -> None:
    import pya_durable
    import pydantic_ai_durable

    assert pydantic_ai_durable.DurableHarness is pya_durable.DurableHarness
    assert pydantic_ai_durable.DurableConversation is pya_durable.DurableConversation
    assert pydantic_ai_durable.define_saga is pya_durable.define_saga
    assert pydantic_ai_durable.AioSqliteStorage is pya_durable.AioSqliteStorage
    assert pydantic_ai_durable.MemoryStorage is pya_durable.MemoryStorage
    assert pydantic_ai_durable.JsonlStorage is pya_durable.JsonlStorage
    assert pydantic_ai_durable.approval_gate is pya_durable.approval_gate

