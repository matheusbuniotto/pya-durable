from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from pydantic_ai_durable.models import (
    ApprovalRequestRecord,
    ApprovalStatus,
    ConversationRecord,
    DurableDocRecord,
    MemoRecord,
    ReplayPolicy,
    SubmissionRecord,
    SubmissionStatus,
    TaskRecord,
    TaskStatus,
    ToolInvocationRecord,
    ToolInvocationStatus,
    TranscriptEntryRecord,
    utc_now_iso,
)
from pydantic_ai_durable.storage import DurableStorage


class MemoryStorage(DurableStorage):
    def __init__(self) -> None:
        self._conversations: dict[str, ConversationRecord] = {}
        self._transcript_entries: dict[str, TranscriptEntryRecord] = {}
        self._transcript_by_conv: dict[str, list[str]] = {}
        self._tasks: dict[str, TaskRecord] = {}
        self._tool_invocations: dict[str, ToolInvocationRecord] = {}
        self._submissions: dict[str, SubmissionRecord] = {}
        self._docs: dict[tuple[str, str], DurableDocRecord] = {}
        self._memos: dict[tuple[str, str], MemoRecord] = {}
        self._approvals: dict[str, ApprovalRequestRecord] = {}
        self._initialized = False

    async def init_db(self) -> None:
        self._initialized = True

    async def close(self) -> None:
        self._initialized = False

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[Any]:
        if not self._initialized:
            self._initialized = True
        yield None

    async def insert_conversation(
        self,
        record: ConversationRecord,
        conn: Any = None,
    ) -> None:
        self._conversations[record.conversation_id] = record

    async def get_conversation(
        self,
        conversation_id: str,
        conn: Any = None,
    ) -> ConversationRecord | None:
        return self._conversations.get(conversation_id)

    async def insert_transcript_entry(
        self,
        record: TranscriptEntryRecord,
        conn: Any = None,
    ) -> None:
        self._transcript_entries[record.entry_id] = record
        if record.conversation_id not in self._transcript_by_conv:
            self._transcript_by_conv[record.conversation_id] = []
        if record.entry_id not in self._transcript_by_conv[record.conversation_id]:
            self._transcript_by_conv[record.conversation_id].append(record.entry_id)

    async def get_transcript_entry(
        self,
        entry_id: str,
        conn: Any = None,
    ) -> TranscriptEntryRecord | None:
        return self._transcript_entries.get(entry_id)

    async def get_conversation_transcript(
        self,
        conversation_id: str,
        conn: Any = None,
    ) -> list[TranscriptEntryRecord]:
        entry_ids = self._transcript_by_conv.get(conversation_id, [])
        return [
            self._transcript_entries[eid]
            for eid in entry_ids
            if eid in self._transcript_entries
        ]

    async def get_ancestor_entries(
        self,
        leaf_entry_id: str,
        conn: Any = None,
    ) -> list[TranscriptEntryRecord]:
        result: list[TranscriptEntryRecord] = []
        curr_id: str | None = leaf_entry_id
        visited: set[str] = set()
        while curr_id and curr_id not in visited:
            visited.add(curr_id)
            entry = self._transcript_entries.get(curr_id)
            if not entry:
                break
            result.append(entry)
            curr_id = entry.parent_entry_id
        result.reverse()
        return result

    async def insert_task(
        self,
        record: TaskRecord,
        conn: Any = None,
    ) -> None:
        self._tasks[record.task_id] = record

    async def update_task_state(
        self,
        task_id: str,
        phase: str,
        state_json: str,
        status: TaskStatus = TaskStatus.RUNNING,
        checkpoint_json: str | None = None,
        conn: Any = None,
    ) -> None:
        existing = self._tasks.get(task_id)
        if existing is not None:
            updated = TaskRecord(
                task_id=existing.task_id,
                task_type=existing.task_type,
                version=existing.version,
                status=status,
                owner_kind=existing.owner_kind,
                owner_id=existing.owner_id,
                phase=phase,
                input_json=existing.input_json,
                state_json=state_json,
                checkpoint_json=checkpoint_json if checkpoint_json is not None else existing.checkpoint_json,
                policy=existing.policy,
                created_at=existing.created_at,
                updated_at=utc_now_iso(),
            )
            self._tasks[task_id] = updated

    async def get_task(
        self,
        task_id: str,
        conn: Any = None,
    ) -> TaskRecord | None:
        return self._tasks.get(task_id)

    async def get_tasks_by_status(
        self,
        status: TaskStatus,
        conn: Any = None,
    ) -> list[TaskRecord]:
        return [t for t in self._tasks.values() if t.status == status]

    async def get_running_tasks(
        self,
        conn: Any = None,
    ) -> list[TaskRecord]:
        return await self.get_tasks_by_status(TaskStatus.RUNNING, conn=conn)

    async def record_tool_started(
        self,
        invocation_id: str,
        task_id: str,
        conversation_id: str,
        tool_name: str,
        args_json: str,
        replay_policy: str,
        conn: Any = None,
    ) -> None:
        self._tool_invocations[invocation_id] = ToolInvocationRecord(
            invocation_id=invocation_id,
            task_id=task_id,
            conversation_id=conversation_id,
            tool_name=tool_name,
            args_json=args_json,
            replay_policy=replay_policy,
            status=ToolInvocationStatus.STARTED,
            started_at=utc_now_iso(),
        )

    async def record_tool_completed(
        self,
        invocation_id: str,
        result_json: str,
        conn: Any = None,
    ) -> None:
        inv = self._tool_invocations.get(invocation_id)
        if inv is not None:
            self._tool_invocations[invocation_id] = ToolInvocationRecord(
                invocation_id=inv.invocation_id,
                task_id=inv.task_id,
                conversation_id=inv.conversation_id,
                tool_name=inv.tool_name,
                args_json=inv.args_json,
                replay_policy=inv.replay_policy,
                status=ToolInvocationStatus.COMPLETED,
                result_json=result_json,
                partial_output=inv.partial_output,
                error_message=inv.error_message,
                started_at=inv.started_at,
                completed_at=utc_now_iso(),
            )

    async def record_tool_failed(
        self,
        invocation_id: str,
        error_message: str,
        conn: Any = None,
    ) -> None:
        inv = self._tool_invocations.get(invocation_id)
        if inv is not None:
            self._tool_invocations[invocation_id] = ToolInvocationRecord(
                invocation_id=inv.invocation_id,
                task_id=inv.task_id,
                conversation_id=inv.conversation_id,
                tool_name=inv.tool_name,
                args_json=inv.args_json,
                replay_policy=inv.replay_policy,
                status=ToolInvocationStatus.FAILED,
                result_json=inv.result_json,
                partial_output=inv.partial_output,
                error_message=error_message,
                started_at=inv.started_at,
                completed_at=utc_now_iso(),
            )

    async def record_tool_interrupted(
        self,
        invocation_id: str,
        partial_output: str | None = None,
        conn: Any = None,
    ) -> None:
        inv = self._tool_invocations.get(invocation_id)
        if inv is not None:
            self._tool_invocations[invocation_id] = ToolInvocationRecord(
                invocation_id=inv.invocation_id,
                task_id=inv.task_id,
                conversation_id=inv.conversation_id,
                tool_name=inv.tool_name,
                args_json=inv.args_json,
                replay_policy=inv.replay_policy,
                status=ToolInvocationStatus.INTERRUPTED,
                result_json=inv.result_json,
                partial_output=partial_output if partial_output is not None else inv.partial_output,
                error_message=inv.error_message,
                started_at=inv.started_at,
                completed_at=utc_now_iso(),
            )

    async def get_tool_invocation(
        self,
        invocation_id: str,
        conn: Any = None,
    ) -> ToolInvocationRecord | None:
        return self._tool_invocations.get(invocation_id)

    async def update_tool_partial_output(
        self,
        invocation_id: str,
        partial_output: str,
        conn: Any = None,
    ) -> None:
        inv = self._tool_invocations.get(invocation_id)
        if inv is not None:
            self._tool_invocations[invocation_id] = ToolInvocationRecord(
                invocation_id=inv.invocation_id,
                task_id=inv.task_id,
                conversation_id=inv.conversation_id,
                tool_name=inv.tool_name,
                args_json=inv.args_json,
                replay_policy=inv.replay_policy,
                status=inv.status,
                result_json=inv.result_json,
                partial_output=partial_output,
                error_message=inv.error_message,
                started_at=inv.started_at,
                completed_at=inv.completed_at,
            )

    async def get_tool_invocations_by_task(
        self,
        task_id: str,
        conn: Any = None,
    ) -> list[ToolInvocationRecord]:
        return [inv for inv in self._tool_invocations.values() if inv.task_id == task_id]

    async def get_tool_invocations_by_status(
        self,
        status: ToolInvocationStatus,
        conn: Any = None,
    ) -> list[ToolInvocationRecord]:
        return [inv for inv in self._tool_invocations.values() if inv.status == status]

    async def reconcile_started_tool_invocations(
        self,
        conn: Any = None,
    ) -> tuple[int, int]:
        safe_count = 0
        unsafe_count = 0
        now = utc_now_iso()
        for inv_id, inv in list(self._tool_invocations.items()):
            if inv.status == ToolInvocationStatus.STARTED:
                if inv.replay_policy == ReplayPolicy.UNSAFE or inv.replay_policy == "unsafe":
                    unsafe_count += 1
                    self._tool_invocations[inv_id] = ToolInvocationRecord(
                        invocation_id=inv.invocation_id,
                        task_id=inv.task_id,
                        conversation_id=inv.conversation_id,
                        tool_name=inv.tool_name,
                        args_json=inv.args_json,
                        replay_policy=inv.replay_policy,
                        status=ToolInvocationStatus.INTERRUPTED,
                        result_json=inv.result_json,
                        partial_output=inv.partial_output,
                        error_message=inv.error_message,
                        started_at=inv.started_at,
                        completed_at=now,
                    )
                elif inv.replay_policy == ReplayPolicy.SAFE or inv.replay_policy == "safe":
                    safe_count += 1
        return (safe_count, unsafe_count)

    async def insert_submission(
        self,
        submission: SubmissionRecord,
        conn: Any = None,
    ) -> None:
        self._submissions[submission.request_id] = submission

    async def get_submission(
        self,
        request_id: str,
        conn: Any = None,
    ) -> SubmissionRecord | None:
        return self._submissions.get(request_id)

    async def update_submission(
        self,
        request_id: str,
        status: SubmissionStatus,
        result_json: str | None = None,
        conn: Any = None,
    ) -> None:
        sub = self._submissions.get(request_id)
        if sub is not None:
            self._submissions[request_id] = SubmissionRecord(
                request_id=sub.request_id,
                conversation_id=sub.conversation_id,
                prompt_text=sub.prompt_text,
                task_id=sub.task_id,
                status=status,
                result_json=result_json if result_json is not None else sub.result_json,
                created_at=sub.created_at,
            )

    async def upsert_durable_doc(
        self,
        record: DurableDocRecord,
        conn: Any = None,
    ) -> None:
        self._docs[(record.conversation_id, record.doc_id)] = record

    async def get_durable_doc(
        self,
        doc_id: str,
        conversation_id: str,
        conn: Any = None,
    ) -> DurableDocRecord | None:
        return self._docs.get((conversation_id, doc_id))

    async def list_durable_docs(
        self,
        conversation_id: str,
        conn: Any = None,
    ) -> list[DurableDocRecord]:
        return [doc for doc in self._docs.values() if doc.conversation_id == conversation_id]

    async def insert_memo(
        self,
        conversation_id: str,
        key: str,
        value_json: str,
        conn: Any = None,
    ) -> bool:
        k = (conversation_id, key)
        if k not in self._memos:
            self._memos[k] = MemoRecord(
                conversation_id=conversation_id,
                key=key,
                value_json=value_json,
                created_at=utc_now_iso(),
            )
            return True
        return False

    async def get_memo(
        self,
        conversation_id: str,
        key: str,
        conn: Any = None,
    ) -> MemoRecord | None:
        return self._memos.get((conversation_id, key))

    async def delete_memo(
        self,
        conversation_id: str,
        key: str,
        conn: Any = None,
    ) -> bool:
        return self._memos.pop((conversation_id, key), None) is not None

    async def insert_approval_request(
        self,
        record: ApprovalRequestRecord,
        conn: Any = None,
    ) -> None:
        self._approvals[record.challenge_token] = record

    async def get_approval_request(
        self,
        challenge_token: str,
        conn: Any = None,
    ) -> ApprovalRequestRecord | None:
        return self._approvals.get(challenge_token)

    async def get_approval_by_task_and_tool(
        self,
        task_id: str,
        tool_name: str,
        conn: Any = None,
    ) -> ApprovalRequestRecord | None:
        matching = [
            a for a in self._approvals.values()
            if a.task_id == task_id and a.tool_name == tool_name
        ]
        if not matching:
            return None
        matching.sort(key=lambda a: a.created_at, reverse=True)
        return matching[0]

    async def list_pending_approvals(
        self,
        conversation_id: str | None = None,
        conn: Any = None,
    ) -> list[ApprovalRequestRecord]:
        results = [
            appr for appr in self._approvals.values()
            if appr.status == ApprovalStatus.PENDING
            and (conversation_id is None or appr.conversation_id == conversation_id)
        ]
        return sorted(results, key=lambda a: a.created_at)

    async def resolve_approval_request(
        self,
        challenge_token: str,
        status: ApprovalStatus,
        decision_json: str | None = None,
        resolved_at: str | None = None,
        conn: Any = None,
    ) -> ApprovalRequestRecord | None:
        existing = self._approvals.get(challenge_token)
        if existing is None:
            return None
        ts = resolved_at or utc_now_iso()
        updated = ApprovalRequestRecord(
            challenge_token=existing.challenge_token,
            task_id=existing.task_id,
            conversation_id=existing.conversation_id,
            tool_name=existing.tool_name,
            args_json=existing.args_json,
            prompt_message=existing.prompt_message,
            status=status,
            decision_json=decision_json,
            created_at=existing.created_at,
            resolved_at=ts,
            expires_at=existing.expires_at,
        )
        self._approvals[challenge_token] = updated
        return updated
