import asyncio
import json
import os
from pathlib import Path
from typing import Any

from pydantic_ai_durable.storage.memory import MemoryStorage
from pydantic_ai_durable.models import (
    ApprovalRequestRecord,
    ApprovalStatus,
    ConversationRecord,
    DurableDocRecord,
    MemoRecord,
    SubmissionRecord,
    SubmissionStatus,
    TaskRecord,
    TaskStatus,
    ToolInvocationRecord,
    TranscriptEntryRecord,
)


def _append_jsonl_line_sync(file_path: Path, line: str) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())


class JsonlStorage(MemoryStorage):
    def __init__(self, file_path: str | Path) -> None:
        super().__init__()
        self.file_path = Path(file_path)
        self._io_lock = asyncio.Lock()

    async def _append_record(self, entity: str, data: dict[str, Any]) -> None:
        async with self._io_lock:
            payload = json.dumps({"entity": entity, "data": data})
            await asyncio.to_thread(_append_jsonl_line_sync, self.file_path, payload)

    async def init_db(self) -> None:
        async with self._io_lock:
            if self._initialized:
                return

            if self.file_path.exists():
                text = await asyncio.to_thread(self.file_path.read_text, encoding="utf-8")
                for line in text.splitlines():
                    stripped = line.strip()
                    if not stripped:
                        continue
                    try:
                        obj = json.loads(stripped)
                        entity = obj.get("entity")
                        data = obj.get("data", {})
                        if entity == "conversation":
                            rec = ConversationRecord.model_validate(data)
                            self._conversations[rec.conversation_id] = rec
                        elif entity == "transcript_entry":
                            rec_entry = TranscriptEntryRecord.model_validate(data)
                            self._transcript_entries[rec_entry.entry_id] = rec_entry
                            if rec_entry.conversation_id not in self._transcript_by_conv:
                                self._transcript_by_conv[rec_entry.conversation_id] = []
                            if rec_entry.entry_id not in self._transcript_by_conv[rec_entry.conversation_id]:
                                self._transcript_by_conv[rec_entry.conversation_id].append(rec_entry.entry_id)
                        elif entity == "task":
                            rec_task = TaskRecord.model_validate(data)
                            self._tasks[rec_task.task_id] = rec_task
                        elif entity == "tool_invocation":
                            rec_inv = ToolInvocationRecord.model_validate(data)
                            self._tool_invocations[rec_inv.invocation_id] = rec_inv
                        elif entity == "submission":
                            rec_sub = SubmissionRecord.model_validate(data)
                            self._submissions[rec_sub.request_id] = rec_sub
                        elif entity == "doc":
                            rec_doc = DurableDocRecord.model_validate(data)
                            self._docs[(rec_doc.conversation_id, rec_doc.doc_id)] = rec_doc
                        elif entity == "memo":
                            rec_memo = MemoRecord.model_validate(data)
                            k = (rec_memo.conversation_id, rec_memo.key)
                            if k not in self._memos:
                                self._memos[k] = rec_memo
                        elif entity == "memo_delete":
                            self._memos.pop((data["conversation_id"], data["key"]), None)
                        elif entity == "approval_insert":
                            rec_appr = ApprovalRequestRecord.model_validate(data)
                            self._approvals[rec_appr.challenge_token] = rec_appr
                        elif entity == "approval_resolve":
                            token = data["challenge_token"]
                            if token in self._approvals:
                                prev = self._approvals[token]
                                self._approvals[token] = ApprovalRequestRecord(
                                    challenge_token=prev.challenge_token,
                                    task_id=prev.task_id,
                                    conversation_id=prev.conversation_id,
                                    tool_name=prev.tool_name,
                                    args_json=prev.args_json,
                                    prompt_message=prev.prompt_message,
                                    status=ApprovalStatus(data["status"]),
                                    decision_json=data.get("decision_json"),
                                    created_at=prev.created_at,
                                    resolved_at=data.get("resolved_at"),
                                    expires_at=prev.expires_at,
                                )
                    except Exception:
                        pass
            self._initialized = True

    async def insert_conversation(
        self,
        record: ConversationRecord,
        conn: Any = None,
    ) -> None:
        await super().insert_conversation(record, conn=conn)
        await self._append_record("conversation", record.model_dump(mode="json"))

    async def insert_transcript_entry(
        self,
        record: TranscriptEntryRecord,
        conn: Any = None,
    ) -> None:
        await super().insert_transcript_entry(record, conn=conn)
        await self._append_record("transcript_entry", record.model_dump(mode="json"))

    async def insert_task(
        self,
        record: TaskRecord,
        conn: Any = None,
    ) -> None:
        await super().insert_task(record, conn=conn)
        await self._append_record("task", record.model_dump(mode="json"))

    async def update_task_state(
        self,
        task_id: str,
        phase: str,
        state_json: str,
        status: TaskStatus = TaskStatus.RUNNING,
        checkpoint_json: str | None = None,
        conn: Any = None,
    ) -> None:
        await super().update_task_state(
            task_id=task_id,
            phase=phase,
            state_json=state_json,
            status=status,
            checkpoint_json=checkpoint_json,
            conn=conn,
        )
        t = await self.get_task(task_id)
        if t is not None:
            await self._append_record("task", t.model_dump(mode="json"))

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
        await super().record_tool_started(
            invocation_id=invocation_id,
            task_id=task_id,
            conversation_id=conversation_id,
            tool_name=tool_name,
            args_json=args_json,
            replay_policy=replay_policy,
            conn=conn,
        )
        inv = await self.get_tool_invocation(invocation_id)
        if inv is not None:
            await self._append_record("tool_invocation", inv.model_dump(mode="json"))

    async def record_tool_completed(
        self,
        invocation_id: str,
        result_json: str,
        conn: Any = None,
    ) -> None:
        await super().record_tool_completed(invocation_id, result_json, conn=conn)
        inv = await self.get_tool_invocation(invocation_id)
        if inv is not None:
            await self._append_record("tool_invocation", inv.model_dump(mode="json"))

    async def record_tool_failed(
        self,
        invocation_id: str,
        error_message: str,
        conn: Any = None,
    ) -> None:
        await super().record_tool_failed(invocation_id, error_message, conn=conn)
        inv = await self.get_tool_invocation(invocation_id)
        if inv is not None:
            await self._append_record("tool_invocation", inv.model_dump(mode="json"))

    async def record_tool_interrupted(
        self,
        invocation_id: str,
        partial_output: str | None = None,
        conn: Any = None,
    ) -> None:
        await super().record_tool_interrupted(invocation_id, partial_output, conn=conn)
        inv = await self.get_tool_invocation(invocation_id)
        if inv is not None:
            await self._append_record("tool_invocation", inv.model_dump(mode="json"))

    async def update_tool_partial_output(
        self,
        invocation_id: str,
        partial_output: str,
        conn: Any = None,
    ) -> None:
        await super().update_tool_partial_output(invocation_id, partial_output, conn=conn)
        inv = await self.get_tool_invocation(invocation_id)
        if inv is not None:
            await self._append_record("tool_invocation", inv.model_dump(mode="json"))

    async def reconcile_started_tool_invocations(
        self,
        conn: Any = None,
    ) -> tuple[int, int]:
        res = await super().reconcile_started_tool_invocations(conn=conn)
        for inv in self._tool_invocations.values():
            if inv.status == "interrupted":
                await self._append_record("tool_invocation", inv.model_dump(mode="json"))
        return res

    async def insert_submission(
        self,
        submission: SubmissionRecord,
        conn: Any = None,
    ) -> None:
        await super().insert_submission(submission, conn=conn)
        await self._append_record("submission", submission.model_dump(mode="json"))

    async def update_submission(
        self,
        request_id: str,
        status: SubmissionStatus,
        result_json: str | None = None,
        conn: Any = None,
    ) -> None:
        await super().update_submission(request_id, status, result_json=result_json, conn=conn)
        sub = await self.get_submission(request_id)
        if sub is not None:
            await self._append_record("submission", sub.model_dump(mode="json"))

    async def upsert_durable_doc(
        self,
        record: DurableDocRecord,
        conn: Any = None,
    ) -> None:
        await super().upsert_durable_doc(record, conn=conn)
        await self._append_record("doc", record.model_dump(mode="json"))

    async def insert_memo(
        self,
        conversation_id: str,
        key: str,
        value_json: str,
        conn: Any = None,
    ) -> bool:
        inserted = await super().insert_memo(
            conversation_id=conversation_id,
            key=key,
            value_json=value_json,
            conn=conn,
        )
        if inserted:
            memo = await self.get_memo(conversation_id, key)
            if memo is not None:
                await self._append_record("memo", memo.model_dump(mode="json"))
        return inserted

    async def delete_memo(
        self,
        conversation_id: str,
        key: str,
        conn: Any = None,
    ) -> bool:
        res = await super().delete_memo(conversation_id, key, conn=conn)
        if res:
            await self._append_record(
                "memo_delete",
                {"conversation_id": conversation_id, "key": key},
            )
        return res

    async def insert_approval_request(
        self,
        record: ApprovalRequestRecord,
        conn: Any = None,
    ) -> None:
        await super().insert_approval_request(record, conn=conn)
        await self._append_record("approval_insert", record.model_dump(mode="json"))

    async def resolve_approval_request(
        self,
        challenge_token: str,
        status: ApprovalStatus,
        decision_json: str | None = None,
        resolved_at: str | None = None,
        conn: Any = None,
    ) -> ApprovalRequestRecord | None:
        res = await super().resolve_approval_request(
            challenge_token=challenge_token,
            status=status,
            decision_json=decision_json,
            resolved_at=resolved_at,
            conn=conn,
        )
        if res is not None:
            await self._append_record(
                "approval_resolve",
                {
                    "challenge_token": challenge_token,
                    "status": status.value if hasattr(status, "value") else str(status),
                    "decision_json": decision_json,
                    "resolved_at": res.resolved_at,
                },
            )
        return res

