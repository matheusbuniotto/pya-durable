from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    TypeVar,
)

import aiosqlite
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter

from pydantic_ai_durable.branching import DurableConversation, InvalidForkEntryError
from pydantic_ai_durable.engine import DurableStepEngine, TurnResult
from pydantic_ai_durable.extensions.compaction import BackgroundCompactor
from pydantic_ai_durable.extensions.docs import DocDefinition, DocumentManager
from pydantic_ai_durable.extensions.memo import MemoManager
from pydantic_ai_durable.extensions.registry import (
    ExtensionDefinition,
    ExtensionRegistry,
    SectionContext,
    global_extension_registry,
)
from pydantic_ai_durable.models import (
    ConversationRecord,
    SubmissionRecord,
    SubmissionStatus,
    TaskRecord,
    TaskStatus,
    TranscriptEntryKind,
    TranscriptEntryRecord,
)
from pydantic_ai_durable.providers.registry import (
    ModelsRegistry,
    global_models_registry,
    resolve_agent,
)
from pydantic_ai_durable.steering import global_steering_queue
from pydantic_ai_durable.storage.base import DurableStorage
from pydantic_ai_durable.streaming.hub import (
    AgentEvent,
    ConversationViewState,
    ReactiveDocumentState,
    ReactiveViewState,
    StreamEvent,
    multiplayer_hub,
    stream_hub,
)
from pydantic_ai_durable.tasks.approvals import ApprovalManager, ApprovalRequestRecord
from pydantic_ai_durable.tasks.recovery import CrashRecoveryManager

if TYPE_CHECKING:
    from pydantic_ai_durable.env import ExecutionEnv

TOutput = TypeVar("TOutput")
TDoc = TypeVar("TDoc")


class DurableHarness(Generic[TOutput]):
    def __init__(
        self,
        agent: Agent[Any, TOutput] | Any = None,
        storage: DurableStorage | None = None,
        conversation_id: str | None = None,
        parent_conversation_id: str | None = None,
        fork_entry_id: str | None = None,
        compaction_threshold: int = 4000,
        auto_compact: bool = False,
        registry: ExtensionRegistry | None = None,
        env: ExecutionEnv | None = None,
        env_factory: Callable[[str, Path | str | None], ExecutionEnv] | ExecutionEnv | None = None,
        models: ModelsRegistry | None = None,
    ) -> None:
        if isinstance(agent, DurableStorage) and storage is None:
            storage = agent
            agent = None

        if storage is None:
            raise ValueError("A DurableStorage instance must be provided to DurableHarness")

        self.models = models or global_models_registry
        self.agent = resolve_agent(agent, models=self.models) if agent is not None else None
        self.storage = storage
        self.conversation_id = conversation_id or f"conv-{uuid.uuid4().hex[:12]}"
        self.parent_conversation_id = parent_conversation_id
        self.fork_entry_id = fork_entry_id
        self.current_leaf_entry_id: str | None = fork_entry_id
        self.engine = DurableStepEngine(storage)
        self.compaction_threshold = compaction_threshold
        self.auto_compact = auto_compact
        self.registry = registry if registry is not None else global_extension_registry
        self._env = env
        self._env_factory = env_factory
        self._busy = False
        self._initialized = False

    @classmethod
    async def open(
        cls,
        storage: DurableStorage,
        *,
        models: ModelsRegistry | None = None,
        registry: ExtensionRegistry | None = None,
        env: ExecutionEnv | None = None,
        env_factory: Callable[[str, Path | str | None], ExecutionEnv] | ExecutionEnv | None = None,
        agent: Any = None,
        conversation_id: str | None = None,
        compaction_threshold: int = 4000,
        auto_compact: bool = False,
    ) -> DurableHarness[Any]:
        await storage.init_db()
        harness = cls(
            agent=agent,
            storage=storage,
            conversation_id=conversation_id,
            compaction_threshold=compaction_threshold,
            auto_compact=auto_compact,
            registry=registry,
            env=env,
            env_factory=env_factory,
            models=models,
        )
        await harness._ensure_conversation()
        return harness

    def get_conversation_env(
        self,
        conversation_id: str | None = None,
        cwd: Path | str | None = None,
    ) -> ExecutionEnv:
        target_id = conversation_id or self.conversation_id
        if self._env is not None:
            return self._env
        if self._env_factory is not None:
            if callable(self._env_factory):
                return self._env_factory(target_id, cwd)
            return self._env_factory
        from pydantic_ai_durable.env import LocalExecutionEnv

        return LocalExecutionEnv(cwd=cwd or Path.cwd())

    @property
    def env(self) -> ExecutionEnv:
        return self.get_conversation_env()

    @property
    def approvals(self) -> ApprovalManager:
        return ApprovalManager(self.storage)

    def enqueue_steering(self, message: str) -> None:
        global_steering_queue.enqueue(self.conversation_id, message)

    def steer(self, message: str) -> None:
        global_steering_queue.enqueue(self.conversation_id, message)

    def install_extension(self, ext: ExtensionDefinition) -> None:
        self.registry.install(ext)

    def uninstall_extension(self, name: str) -> None:
        self.registry.uninstall(name)

    async def root(
        self,
        options_or_agent: Any = None,
        *,
        agent: Any = None,
        cwd: Path | str | None = None,
        env: ExecutionEnv | None = None,
    ) -> DurableConversation:
        target_agent = agent
        target_cwd = cwd
        target_env = env

        if isinstance(options_or_agent, dict):
            if "agent" in options_or_agent and target_agent is None:
                target_agent = options_or_agent["agent"]
            if "cwd" in options_or_agent and target_cwd is None:
                target_cwd = options_or_agent["cwd"]
            if "env" in options_or_agent and target_env is None:
                target_env = options_or_agent["env"]
            if target_agent is None and (
                "model" in options_or_agent or "provider" in options_or_agent
            ):
                target_agent = options_or_agent
        elif options_or_agent is not None and target_agent is None:
            target_agent = options_or_agent

        resolved_agent = (
            resolve_agent(target_agent, models=self.models)
            if target_agent is not None
            else self.agent
        )
        conv_env = target_env or self.get_conversation_env(self.conversation_id, cwd=target_cwd)

        conv = DurableConversation(
            conversation_id=self.conversation_id,
            storage=self.storage,
            agent=resolved_agent,
            parent_conversation_id=self.parent_conversation_id,
            fork_entry_id=self.fork_entry_id,
            current_leaf_entry_id=self.current_leaf_entry_id,
            compaction_threshold=self.compaction_threshold,
            auto_compact=self.auto_compact,
            env=conv_env,
            env_factory=self._env_factory,
            models=self.models,
        )
        await conv._ensure_conversation()
        return conv

    async def conversation(
        self,
        conversation_id: str,
        *,
        agent: Any = None,
        cwd: Path | str | None = None,
        env: ExecutionEnv | None = None,
    ) -> DurableConversation:
        resolved_agent = (
            resolve_agent(agent, models=self.models) if agent is not None else self.agent
        )
        conv_env = env or self.get_conversation_env(conversation_id, cwd=cwd)

        conv = DurableConversation(
            conversation_id=conversation_id,
            storage=self.storage,
            agent=resolved_agent,
            compaction_threshold=self.compaction_threshold,
            auto_compact=self.auto_compact,
            env=conv_env,
            env_factory=self._env_factory,
            models=self.models,
        )
        await conv._ensure_conversation()
        return conv

    async def close(self) -> None:
        if hasattr(self.storage, "close"):
            await self.storage.close()

    async def _ensure_conversation(self) -> None:
        if not self._initialized:
            await self.storage.init_db()
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
        await self._ensure_conversation()
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
        await self._ensure_conversation()
        await DocumentManager.set_doc(
            self.storage,
            doc_def,
            self.conversation_id,
            value,
            doc_id=doc_id,
            conn=conn,
        )
        await self.refresh_view_states()

    async def document_state(
        self,
        doc_def: DocDefinition[TDoc],
        doc_id: str = "default",
    ) -> ReactiveDocumentState[TDoc]:
        current_val = await self.get_doc(doc_def, doc_id=doc_id)
        r_doc: ReactiveDocumentState[TDoc] = ReactiveDocumentState(current_val)
        stream_hub.register_reactive_doc(self.conversation_id, doc_def.doc_type, doc_id, r_doc)
        return r_doc

    async def memo(self, key: str, value: Any = None) -> Any:
        return await MemoManager.memo(self.storage, self.conversation_id, key, value)

    def connect_client(self, client_id: str) -> None:
        multiplayer_hub.connect(self.conversation_id, client_id)
        try:
            asyncio.create_task(self.refresh_view_states())
        except RuntimeError:
            pass

    def disconnect_client(self, client_id: str) -> None:
        multiplayer_hub.disconnect(self.conversation_id, client_id)
        try:
            asyncio.create_task(self.refresh_view_states())
        except RuntimeError:
            pass

    def get_connected_clients(self) -> list[str]:
        return multiplayer_hub.get_connected_clients(self.conversation_id)

    async def watch_events(self) -> AsyncIterator[AgentEvent]:
        async for evt in stream_hub.subscribe_events(self.conversation_id):
            yield evt

    async def view_state(self) -> ReactiveViewState:
        await self._ensure_conversation()
        transcript_entries = await self.get_transcript()
        formatted_entries = []
        for e in transcript_entries:
            try:
                msg = json.loads(e.message_json)
            except Exception:
                msg = e.message_json
            formatted_entries.append(
                {
                    "entry_id": e.entry_id,
                    "kind": e.kind.value,
                    "created_at": e.created_at,
                    "message": msg,
                }
            )

        doc_records = await self.storage.list_durable_docs(self.conversation_id)
        docs_dict: dict[str, Any] = {}
        for d in doc_records:
            try:
                docs_dict[f"{d.doc_type}:{d.doc_id}"] = json.loads(d.data_json)
            except Exception:
                docs_dict[f"{d.doc_type}:{d.doc_id}"] = d.data_json

        status = "busy" if self._busy else "idle"
        clients = multiplayer_hub.get_connected_clients(self.conversation_id)

        pending_records = await self.approvals.list_pending(self.conversation_id)
        pending_approvals = [
            {
                "challenge_token": p.challenge_token,
                "task_id": p.task_id,
                "tool_name": p.tool_name,
                "args": json.loads(p.args_json) if isinstance(p.args_json, str) else p.args_json,
                "prompt_message": p.prompt_message,
                "status": p.status.value if hasattr(p.status, "value") else str(p.status),
                "created_at": p.created_at,
                "expires_at": p.expires_at,
            }
            for p in pending_records
        ]

        snapshot = ConversationViewState(
            conversation_id=self.conversation_id,
            status=status,
            current_leaf_entry_id=self.current_leaf_entry_id,
            entries=formatted_entries,
            docs=docs_dict,
            connected_clients=clients,
            pending_approvals=pending_approvals,
            version=1,
        )
        rv = ReactiveViewState(snapshot)
        stream_hub.register_reactive_view(self.conversation_id, rv)
        return rv

    async def refresh_view_states(self) -> None:
        views = stream_hub._reactive_views.get(self.conversation_id, set())
        if not views:
            return
        transcript_entries = await self.get_transcript()
        formatted_entries = []
        for e in transcript_entries:
            try:
                msg = json.loads(e.message_json)
            except Exception:
                msg = e.message_json
            formatted_entries.append(
                {
                    "entry_id": e.entry_id,
                    "kind": e.kind.value,
                    "created_at": e.created_at,
                    "message": msg,
                }
            )

        doc_records = await self.storage.list_durable_docs(self.conversation_id)
        docs_dict: dict[str, Any] = {}
        for d in doc_records:
            try:
                docs_dict[f"{d.doc_type}:{d.doc_id}"] = json.loads(d.data_json)
            except Exception:
                docs_dict[f"{d.doc_type}:{d.doc_id}"] = d.data_json

        status = "busy" if self._busy else "idle"
        clients = multiplayer_hub.get_connected_clients(self.conversation_id)

        pending_records = await self.approvals.list_pending(self.conversation_id)
        pending_approvals = [
            {
                "challenge_token": p.challenge_token,
                "task_id": p.task_id,
                "tool_name": p.tool_name,
                "args": json.loads(p.args_json) if isinstance(p.args_json, str) else p.args_json,
                "prompt_message": p.prompt_message,
                "status": p.status.value if hasattr(p.status, "value") else str(p.status),
                "created_at": p.created_at,
                "expires_at": p.expires_at,
            }
            for p in pending_records
        ]

        for rv in list(views):
            new_state = ConversationViewState(
                conversation_id=self.conversation_id,
                status=status,
                current_leaf_entry_id=self.current_leaf_entry_id,
                entries=formatted_entries,
                docs=docs_dict,
                connected_clients=clients,
                pending_approvals=pending_approvals,
                version=rv.value.version + 1,
            )
            rv.notify(new_state)

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

        if self.agent is None:
            self.agent = self.models.create_agent()
        elif not isinstance(self.agent, Agent):
            self.agent = resolve_agent(self.agent, models=self.models)

        await self._ensure_conversation()

        self._busy = True
        try:
            stream_hub.publish(
                self.conversation_id,
                StreamEvent(
                    conversation_id=self.conversation_id,
                    event_type="turn_started",
                    data={"prompt": prompt, "request_id": request_id},
                ),
            )
            await self.refresh_view_states()

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

            if turn_result.waiting_approval:
                stream_hub.publish(
                    self.conversation_id,
                    StreamEvent(
                        conversation_id=self.conversation_id,
                        event_type="approval_requested",
                        data={
                            "challenge_token": turn_result.challenge_token,
                            "task_id": task_id,
                        },
                    ),
                )
            else:
                stream_hub.publish(
                    self.conversation_id,
                    StreamEvent(
                        conversation_id=self.conversation_id,
                        event_type="turn_completed",
                        data={
                            "output": turn_result.output,
                            "token_usage": turn_result.token_usage,
                        },
                    ),
                )
            await self.refresh_view_states()

            if (
                not turn_result.waiting_approval
                and self.auto_compact
                and BackgroundCompactor.should_compact(
                    turn_result.token_usage.get("total_tokens", 0),
                    threshold=self.compaction_threshold,
                )
            ):
                await BackgroundCompactor.compact_conversation(
                    self.storage,
                    self.conversation_id,
                    turn_result.leaf_entry_id,
                )

            if request_id and not turn_result.waiting_approval:
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
            await self.refresh_view_states()

    async def resolve_approval(
        self,
        challenge_token: str,
        approved: bool,
        reason: str = "",
        override_args: dict[str, Any] | None = None,
        approver: str = "operator",
    ) -> ApprovalRequestRecord:
        rec = await self.approvals.resolve(
            challenge_token=challenge_token,
            approved=approved,
            reason=reason,
            override_args=override_args,
            approver=approver,
        )
        await self.refresh_view_states()
        return rec

    async def list_pending_approvals(self) -> list[ApprovalRequestRecord]:
        return await self.approvals.list_pending(self.conversation_id)

    async def wait_for_approval(
        self,
        challenge_token: str,
        poll_interval: float = 0.05,
        timeout: float | None = None,
    ) -> ApprovalRequestRecord:
        return await self.approvals.wait_for_resolution(
            challenge_token=challenge_token,
            poll_interval=poll_interval,
            timeout=timeout,
        )

    async def run_saga(
        self,
        saga_def: Any,
        input_data: Any,
        task_id: str | None = None,
    ) -> Any:
        from pydantic_ai_durable.tasks.sagas import SagaCoordinator
        from pydantic_ai_durable.tasks.tree import TaskRunner

        runner = TaskRunner(self.storage)
        coordinator = SagaCoordinator(self.storage, task_runner=runner, stream_hub=stream_hub)
        return await coordinator.run_saga(
            saga_def=saga_def,
            input_data=input_data,
            task_id=task_id,
            owner_kind="conversation",
            owner_id=self.conversation_id,
            conversation_id=self.conversation_id,
        )

    async def wait_for_task(
        self,
        task_id: str,
        poll_interval: float = 0.05,
        timeout: float | None = None,
    ) -> TaskRecord:
        from pydantic_ai_durable.tasks.tree import TaskRunner

        runner = TaskRunner(self.storage)
        return await runner.wait_for_task(task_id, poll_interval=poll_interval, timeout=timeout)

    async def submit(
        self,
        prompt: str | dict[str, Any] | None = None,
        deps: Any = None,
        request_id: str | None = None,
        when_busy: str | None = None,
    ) -> TurnResult[TOutput] | None:
        effective_prompt = prompt
        effective_req_id = request_id
        if isinstance(prompt, dict):
            effective_req_id = request_id or prompt.get("requestId") or prompt.get("request_id")
            effective_prompt = prompt.get("content") or prompt.get("prompt") or prompt.get("input")

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
            request_id=effective_req_id,
            when_busy=when_busy,
        )

    run_turn = run

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

    async def fork(
        self,
        fork_entry_id: str | None = None,
        new_conversation_id: str | None = None,
        agent: Any = None,
        env: ExecutionEnv | None = None,
    ) -> DurableHarness[Any]:
        await self._ensure_conversation()

        target_fork_id = fork_entry_id or self.current_leaf_entry_id
        if target_fork_id:
            ancestor = await self.storage.get_transcript_entry(target_fork_id)
            if ancestor is None:
                raise InvalidForkEntryError(f"Fork entry {target_fork_id} not found")

        child_conv_id = new_conversation_id or f"conv-{uuid.uuid4().hex[:12]}"
        child_agent = resolve_agent(agent, models=self.models) if agent is not None else self.agent

        record = ConversationRecord(
            conversation_id=child_conv_id,
            parent_conversation_id=self.conversation_id,
            fork_entry_id=target_fork_id,
            agent_config_json=json.dumps(
                {"name": getattr(child_agent, "name", "agent") if child_agent else "agent"}
            ),
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
            env=env if env is not None else self._env,
            env_factory=self._env_factory,
            models=self.models,
        )
        child._initialized = True
        return child


Harness = DurableHarness
