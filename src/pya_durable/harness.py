import json
import uuid
from typing import Any, Generic, Sequence, TypeVar

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter

from pya_durable.engine import DurableStepEngine, TurnResult
from pya_durable.models import (
    ConversationRecord,
    TaskRecord,
    TaskStatus,
    TranscriptEntryKind,
    utc_now_iso,
)
from pya_durable.storage import AioSqliteStorage

TOutput = TypeVar("TOutput")


class DurableHarness(Generic[TOutput]):
    def __init__(
        self,
        agent: Agent[Any, TOutput],
        storage: AioSqliteStorage,
        conversation_id: str | None = None,
    ) -> None:
        self.agent = agent
        self.storage = storage
        self.conversation_id = conversation_id or f"conv-{uuid.uuid4().hex[:12]}"
        self.engine = DurableStepEngine(storage)
        self.current_leaf_entry_id: str | None = None
        self._initialized = False

    async def _ensure_conversation(self) -> None:
        if not self._initialized:
            existing = await self.storage.get_conversation(self.conversation_id)
            if existing is None:
                record = ConversationRecord(
                    conversation_id=self.conversation_id,
                    agent_config_json=json.dumps({"name": getattr(self.agent, "name", "agent")}),
                )
                await self.storage.insert_conversation(record)
            self._initialized = True

    async def run(self, prompt: str, deps: Any = None) -> TurnResult[TOutput]:
        if prompt is None or not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        await self._ensure_conversation()

        history = await self.get_history()

        task_id = f"task-{uuid.uuid4().hex[:12]}"
        task = TaskRecord(
            task_id=task_id,
            task_type="user_turn",
            status=TaskStatus.PENDING,
            owner_kind="conversation",
            owner_id=self.conversation_id,
            phase="pending",
            input_json=json.dumps({"prompt": prompt}),
            state_json=json.dumps({"phase": "pending"}),
        )
        await self.storage.insert_task(task)

        turn_result = await self.engine.run_turn(
            agent=self.agent,
            user_prompt=prompt,
            conversation_id=self.conversation_id,
            task_id=task_id,
            parent_entry_id=self.current_leaf_entry_id,
            message_history=history if history else None,
            deps=deps,
        )

        self.current_leaf_entry_id = turn_result.leaf_entry_id
        return turn_result

    async def get_history(self) -> list[ModelMessage]:
        if not self.current_leaf_entry_id:
            entries = await self.storage.get_conversation_transcript(self.conversation_id)
        else:
            entries = await self.storage.get_ancestor_entries(self.current_leaf_entry_id)

        messages: list[ModelMessage] = []
        for e in entries:
            if e.kind in (
                TranscriptEntryKind.MODEL_REQUEST,
                TranscriptEntryKind.MODEL_RESPONSE,
                TranscriptEntryKind.TOOL_CALL,
            ):
                try:
                    parsed = ModelMessagesTypeAdapter.validate_json(e.message_json)
                    messages.extend(parsed)
                except Exception:
                    pass
        return messages
