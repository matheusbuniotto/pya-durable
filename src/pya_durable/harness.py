import json
import uuid
from typing import Any, AsyncIterator, Generic, Sequence, TypeVar

import aiosqlite
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter

from pya_durable.branching import InvalidForkEntryError
from pya_durable.compaction import BackgroundCompactor
from pya_durable.docs import DocDefinition, DocumentManager
from pya_durable.engine import DurableStepEngine, TurnResult
from pya_durable.extensions import (
    ExtensionDefinition,
    ExtensionRegistry,
    SectionContext,
    global_extension_registry,
)
from pya_durable.models import (
    ConversationRecord,
    SubmissionRecord,
    SubmissionStatus,
    TaskRecord,
    TaskStatus,
    TranscriptEntryKind,
    TranscriptEntryRecord,
    utc_now_iso,
)
from pya_durable.recovery import CrashRecoveryManager
from pya_durable.steering import global_steering_queue
from pya_durable.storage import AioSqliteStorage
from pya_durable.streaming import StreamEvent, stream_hub

TOutput = TypeVar("TOutput")
TDoc = TypeVar("TDoc")


class DurableHarness(Generic[TOutput]):
    def __init__(
        self,
        agent: Agent[Any, TOutput],
        storage: AioSqliteStorage,
        conversation_id: str | None = None,
        parent_conversation_id: str | None = None,
        fork_entry_id: str | None = None,
        compaction_threshold: int = 4000,
        auto_compact: bool = False,
        registry: ExtensionRegistry | None = None,
    ) -> None:
        self.agent = agent
        self.storage = storage
        self.conversation_id = conversation_id or f"conv-{uuid.uuid4().hex[:12]}"
        self.parent_conversation_id = parent_conversation_id
        self.fork_entry_id = fork_entry_id
        self.current_leaf_entry_id: str | None = fork_entry_id
        self.engine = DurableStepEngine(storage)
        self.compaction_threshold = compaction_threshold
        self.auto_compact = auto_compact
        self.registry = registry if registry is not None else global_extension_registry
        self._busy = False
        self._initialized = False

    def install_extension(self, ext: ExtensionDefinition) -> None:
        self.registry.install(ext)

    def uninstall_extension(self, name: str) -> None:
        self.registry.uninstall(name)

    async def _ensure_conversation(self) -> None:
        if not self._initialized:
            existing = await self.storage.get_conversation(self.conversation_id)
            if existing is None:
                record = ConversationRecord(
                    conversation_id=self.conversation_id,
                    parent_conversation_id=self.parent_conversation_id,
                    fork_entry_id=self.fork_entry_id,
                    agent_config_json=json.dumps({"name": getattr(self.agent, "name", "agent")}),
                )
                await self.storage.insert_conversation(record)
            self._initialized = True

    async def watch(self) -> AsyncIterator[StreamEvent]:
        async for event in stream_hub.subscribe(self.conversation_id):
            yield event

    async def get_doc(
        self,
        doc_def: DocDefinition[TDoc],
        doc_id: str = "default",
        conn: aiosqlite.Connection | None = None,
    ) -> TDoc:
        return await DocumentManager.get_doc(
            self.storage,
            doc_def,
            self.conversation_id,
            doc_id=doc_id,
            conn=conn,
        )

    async def set_doc(
        self,
        doc_def: DocDefinition[TDoc],
        value: TDoc,
        doc_id: str = "default",
        conn: aiosqlite.Connection | None = None,
    ) -> None:
        await DocumentManager.set_doc(
            self.storage,
            doc_def,
            self.conversation_id,
            value,
            doc_id=doc_id,
            conn=conn,
        )

    async def fork(
        self,
        fork_entry_id: str | None = None,
        new_conversation_id: str | None = None,
        agent: Agent[Any, Any] | None = None,
    ) -> "DurableHarness[Any]":
        target_fork_id = fork_entry_id or self.current_leaf_entry_id
        if target_fork_id is None:
            raise InvalidForkEntryError("Fork entry ID cannot be None")

        entry = await self.storage.get_transcript_entry(target_fork_id)
        if entry is None:
            raise InvalidForkEntryError(
                f"Fork entry '{target_fork_id}' does not exist in storage"
            )

        child_conv_id = new_conversation_id or f"conv-{uuid.uuid4().hex[:12]}"
        child_agent = agent if agent is not None else self.agent

        record = ConversationRecord(
            conversation_id=child_conv_id,
            parent_conversation_id=self.conversation_id,
            fork_entry_id=target_fork_id,
            agent_config_json=json.dumps({"name": getattr(child_agent, "name", "agent")}),
        )
        await self.storage.insert_conversation(record)

        child = DurableHarness(
            agent=child_agent,
            storage=self.storage,
            conversation_id=child_conv_id,
            parent_conversation_id=self.conversation_id,
            fork_entry_id=target_fork_id,
            compaction_threshold=self.compaction_threshold,
            auto_compact=self.auto_compact,
            registry=self.registry,
        )
        child._initialized = True
        return child

    async def run(
        self,
        prompt: str | None = None,
        deps: Any = None,
        request_id: str | None = None,
        when_busy: str | None = None,
    ) -> TurnResult[TOutput] | None:
        if self._busy:
            if when_busy == "steer" and prompt:
                global_steering_queue.enqueue(self.conversation_id, prompt)
                return None
            raise RuntimeError(f"Conversation {self.conversation_id} is busy")

        if prompt is None or not prompt.strip():
            steer = global_steering_queue.pop_next(self.conversation_id)
            if steer:
                prompt = steer
            else:
                raise ValueError("Prompt cannot be empty")

        await self._ensure_conversation()

        self._busy = True
        try:
            if request_id:
                existing = await self.storage.get_submission(request_id)
                if existing is not None and existing.status == SubmissionStatus.SETTLED:
                    output_val = existing.result_json
                    if output_val is not None:
                        try:
                            output_val = json.loads(output_val)
                        except Exception:
                            pass
                    return TurnResult(
                        output=output_val,  # type: ignore[arg-type]
                        leaf_entry_id=self.current_leaf_entry_id or "",
                        all_messages=[],
                        token_usage={},
                    )

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

            if request_id:
                submission = SubmissionRecord(
                    request_id=request_id,
                    conversation_id=self.conversation_id,
                    prompt_text=prompt,
                    task_id=task_id,
                    status=SubmissionStatus.ACTIVE,
                )
                await self.storage.insert_submission(submission)

            parent_id = self.current_leaf_entry_id or self.fork_entry_id

            turn_result = await self.engine.run_turn(
                agent=self.agent,
                user_prompt=prompt,
                conversation_id=self.conversation_id,
                task_id=task_id,
                parent_entry_id=parent_id,
                message_history=history if history else None,
                deps=deps,
            )

            self.current_leaf_entry_id = turn_result.leaf_entry_id

            if self.auto_compact and BackgroundCompactor.should_compact(
                turn_result.token_usage.get("total_tokens", 0),
                threshold=self.compaction_threshold,
            ):
                await BackgroundCompactor.compact_conversation(
                    self.storage,
                    self.conversation_id,
                    turn_result.leaf_entry_id,
                )

            if request_id:
                out_str = (
                    turn_result.output.model_dump_json()
                    if hasattr(turn_result.output, "model_dump_json")
                    else (
                        json.dumps(turn_result.output)
                        if isinstance(turn_result.output, (dict, list, int, float, bool))
                        else str(turn_result.output)
                    )
                )
                await self.storage.update_submission(
                    request_id=request_id,
                    status=SubmissionStatus.SETTLED,
                    result_json=out_str,
                )

            return turn_result
        finally:
            self._busy = False

    async def submit(
        self,
        prompt: str | None = None,
        deps: Any = None,
        request_id: str | None = None,
        when_busy: str | None = None,
    ) -> TurnResult[TOutput] | None:
        effective_prompt = prompt
        section_ctx = SectionContext(conversation_id=self.conversation_id)
        rendered_sections = await self.registry.render_prompt_sections(section_ctx)
        if rendered_sections:
            if effective_prompt:
                effective_prompt = f"{rendered_sections}\n\n{effective_prompt}"
            else:
                effective_prompt = rendered_sections
        return await self.run(
            prompt=effective_prompt,
            deps=deps,
            request_id=request_id,
            when_busy=when_busy,
        )

    async def resume(self, task_id: str | None = None) -> Any:
        manager = CrashRecoveryManager(self.storage, agent=self.agent)
        res = await manager.resume(task_id=task_id)
        if isinstance(res, TurnResult):
            self.current_leaf_entry_id = res.leaf_entry_id
        return res

    async def get_transcript(self) -> list[TranscriptEntryRecord]:
        leaf_id = self.current_leaf_entry_id or self.fork_entry_id
        if not leaf_id:
            return await self.storage.get_conversation_transcript(self.conversation_id)
        return await self.storage.get_ancestor_entries(leaf_id)

    async def get_history(self) -> list[ModelMessage]:
        entries = await self.get_transcript()
        handoff_idx = -1
        for i, e in enumerate(entries):
            if e.kind == TranscriptEntryKind.SYSTEM:
                try:
                    data = json.loads(e.message_json)
                    if data.get("type") == "handoff":
                        handoff_idx = i
                except Exception:
                    pass

        if handoff_idx != -1:
            entries = entries[handoff_idx + 1 :]

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
