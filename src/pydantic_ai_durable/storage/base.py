import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

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
    ToolInvocationStatus,
    TranscriptEntryRecord,
)

logger = logging.getLogger("pydantic_ai_durable.storage")

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS conversations (
    conversation_id TEXT PRIMARY KEY,
    parent_conversation_id TEXT,
    fork_entry_id TEXT,
    agent_config_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(parent_conversation_id) REFERENCES conversations(conversation_id)
);

CREATE TABLE IF NOT EXISTS transcript_entries (
    entry_id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    parent_entry_id TEXT,
    kind TEXT NOT NULL,
    message_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY(conversation_id) REFERENCES conversations(conversation_id),
    FOREIGN KEY(parent_entry_id) REFERENCES transcript_entries(entry_id)
);

CREATE INDEX IF NOT EXISTS idx_transcript_parent ON transcript_entries(parent_entry_id);
CREATE INDEX IF NOT EXISTS idx_transcript_conv ON transcript_entries(conversation_id, created_at);

CREATE TABLE IF NOT EXISTS tasks (
    task_id TEXT PRIMARY KEY,
    task_type TEXT NOT NULL,
    version INTEGER NOT NULL DEFAULT 1,
    status TEXT NOT NULL,
    owner_kind TEXT NOT NULL,
    owner_id TEXT NOT NULL,
    phase TEXT NOT NULL,
    input_json TEXT NOT NULL,
    state_json TEXT NOT NULL,
    checkpoint_json TEXT,
    policy TEXT NOT NULL DEFAULT 'failFast',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_tasks_owner ON tasks(owner_kind, owner_id);
CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);

CREATE TABLE IF NOT EXISTS tool_invocations (
    invocation_id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    args_json TEXT NOT NULL,
    replay_policy TEXT NOT NULL,
    status TEXT NOT NULL,
    result_json TEXT,
    partial_output TEXT,
    error_message TEXT,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    FOREIGN KEY(task_id) REFERENCES tasks(task_id),
    FOREIGN KEY(conversation_id) REFERENCES conversations(conversation_id)
);

CREATE INDEX IF NOT EXISTS idx_tool_invocations_task ON tool_invocations(task_id);
CREATE INDEX IF NOT EXISTS idx_tool_invocations_conv ON tool_invocations(conversation_id);

CREATE TABLE IF NOT EXISTS submissions (
    request_id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    prompt_text TEXT NOT NULL,
    task_id TEXT NOT NULL,
    status TEXT NOT NULL,
    result_json TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY(conversation_id) REFERENCES conversations(conversation_id),
    FOREIGN KEY(task_id) REFERENCES tasks(task_id)
);

CREATE INDEX IF NOT EXISTS idx_submissions_conv ON submissions(conversation_id);

CREATE TABLE IF NOT EXISTS durable_docs (
    doc_id TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    version INTEGER NOT NULL DEFAULT 1,
    doc_type TEXT NOT NULL,
    data_json TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY(doc_id, conversation_id),
    FOREIGN KEY(conversation_id) REFERENCES conversations(conversation_id)
);

CREATE TABLE IF NOT EXISTS memos (
    conversation_id TEXT NOT NULL,
    key TEXT NOT NULL,
    value_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY(conversation_id, key),
    FOREIGN KEY(conversation_id) REFERENCES conversations(conversation_id)
);

CREATE INDEX IF NOT EXISTS idx_memos_conv ON memos(conversation_id);

CREATE TABLE IF NOT EXISTS approvals (
    challenge_token TEXT PRIMARY KEY,
    task_id TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    args_json TEXT NOT NULL,
    prompt_message TEXT NOT NULL,
    status TEXT NOT NULL,
    decision_json TEXT,
    created_at TEXT NOT NULL,
    resolved_at TEXT,
    expires_at TEXT,
    FOREIGN KEY(task_id) REFERENCES tasks(task_id),
    FOREIGN KEY(conversation_id) REFERENCES conversations(conversation_id)
);

CREATE INDEX IF NOT EXISTS idx_approvals_conv ON approvals(conversation_id);
CREATE INDEX IF NOT EXISTS idx_approvals_status ON approvals(status);
"""


class DurableStorage(ABC):
    _initialized: bool = False

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    @abstractmethod
    async def init_db(self) -> None:
        pass

    @abstractmethod
    async def close(self) -> None:
        pass

    async def __aenter__(self) -> "DurableStorage":
        await self.init_db()
        return self

    async def __aexit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        await self.close()

    @abstractmethod
    def transaction(self) -> AsyncIterator[Any]:
        pass

    @abstractmethod
    async def insert_conversation(
        self,
        record: ConversationRecord,
        conn: Any = None,
    ) -> None:
        pass

    @abstractmethod
    async def get_conversation(
        self,
        conversation_id: str,
        conn: Any = None,
    ) -> ConversationRecord | None:
        pass

    @abstractmethod
    async def insert_transcript_entry(
        self,
        record: TranscriptEntryRecord,
        conn: Any = None,
    ) -> None:
        pass

    @abstractmethod
    async def get_transcript_entry(
        self,
        entry_id: str,
        conn: Any = None,
    ) -> TranscriptEntryRecord | None:
        pass

    @abstractmethod
    async def get_conversation_transcript(
        self,
        conversation_id: str,
        conn: Any = None,
    ) -> list[TranscriptEntryRecord]:
        pass

    @abstractmethod
    async def get_ancestor_entries(
        self,
        leaf_entry_id: str,
        conn: Any = None,
    ) -> list[TranscriptEntryRecord]:
        pass

    @abstractmethod
    async def insert_task(
        self,
        record: TaskRecord,
        conn: Any = None,
    ) -> None:
        pass

    @abstractmethod
    async def update_task_state(
        self,
        task_id: str,
        phase: str,
        state_json: str,
        status: TaskStatus = TaskStatus.RUNNING,
        checkpoint_json: str | None = None,
        conn: Any = None,
    ) -> None:
        pass

    @abstractmethod
    async def get_task(
        self,
        task_id: str,
        conn: Any = None,
    ) -> TaskRecord | None:
        pass

    @abstractmethod
    async def get_tasks_by_status(
        self,
        status: TaskStatus,
        conn: Any = None,
    ) -> list[TaskRecord]:
        pass

    @abstractmethod
    async def get_running_tasks(
        self,
        conn: Any = None,
    ) -> list[TaskRecord]:
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    async def record_tool_completed(
        self,
        invocation_id: str,
        result_json: str,
        conn: Any = None,
    ) -> None:
        pass

    @abstractmethod
    async def record_tool_failed(
        self,
        invocation_id: str,
        error_message: str,
        conn: Any = None,
    ) -> None:
        pass

    @abstractmethod
    async def record_tool_interrupted(
        self,
        invocation_id: str,
        partial_output: str | None = None,
        conn: Any = None,
    ) -> None:
        pass

    @abstractmethod
    async def get_tool_invocation(
        self,
        invocation_id: str,
        conn: Any = None,
    ) -> ToolInvocationRecord | None:
        pass

    @abstractmethod
    async def update_tool_partial_output(
        self,
        invocation_id: str,
        partial_output: str,
        conn: Any = None,
    ) -> None:
        pass

    @abstractmethod
    async def get_tool_invocations_by_task(
        self,
        task_id: str,
        conn: Any = None,
    ) -> list[ToolInvocationRecord]:
        pass

    @abstractmethod
    async def get_tool_invocations_by_status(
        self,
        status: ToolInvocationStatus,
        conn: Any = None,
    ) -> list[ToolInvocationRecord]:
        pass

    @abstractmethod
    async def reconcile_started_tool_invocations(
        self,
        conn: Any = None,
    ) -> tuple[int, int]:
        pass

    @abstractmethod
    async def insert_submission(
        self,
        submission: SubmissionRecord,
        conn: Any = None,
    ) -> None:
        pass

    @abstractmethod
    async def get_submission(
        self,
        request_id: str,
        conn: Any = None,
    ) -> SubmissionRecord | None:
        pass

    @abstractmethod
    async def update_submission(
        self,
        request_id: str,
        status: SubmissionStatus,
        result_json: str | None = None,
        conn: Any = None,
    ) -> None:
        pass

    @abstractmethod
    async def upsert_durable_doc(
        self,
        record: DurableDocRecord,
        conn: Any = None,
    ) -> None:
        pass

    @abstractmethod
    async def get_durable_doc(
        self,
        doc_id: str,
        conversation_id: str,
        conn: Any = None,
    ) -> DurableDocRecord | None:
        pass

    @abstractmethod
    async def list_durable_docs(
        self,
        conversation_id: str,
        conn: Any = None,
    ) -> list[DurableDocRecord]:
        pass

    @abstractmethod
    async def insert_memo(
        self,
        conversation_id: str,
        key: str,
        value_json: str,
        conn: Any = None,
    ) -> bool:
        pass

    @abstractmethod
    async def get_memo(
        self,
        conversation_id: str,
        key: str,
        conn: Any = None,
    ) -> MemoRecord | None:
        pass

    @abstractmethod
    async def delete_memo(
        self,
        conversation_id: str,
        key: str,
        conn: Any = None,
    ) -> bool:
        pass

    @abstractmethod
    async def insert_approval_request(
        self,
        record: ApprovalRequestRecord,
        conn: Any = None,
    ) -> None:
        pass

    @abstractmethod
    async def get_approval_request(
        self,
        challenge_token: str,
        conn: Any = None,
    ) -> ApprovalRequestRecord | None:
        pass

    @abstractmethod
    async def get_approval_by_task_and_tool(
        self,
        task_id: str,
        tool_name: str,
        conn: Any = None,
    ) -> ApprovalRequestRecord | None:
        pass

    @abstractmethod
    async def list_pending_approvals(
        self,
        conversation_id: str | None = None,
        conn: Any = None,
    ) -> list[ApprovalRequestRecord]:
        pass

    @abstractmethod
    async def resolve_approval_request(
        self,
        challenge_token: str,
        status: ApprovalStatus,
        decision_json: str | None = None,
        resolved_at: str | None = None,
        conn: Any = None,
    ) -> ApprovalRequestRecord | None:
        pass

    async def checkpoint(self) -> tuple[int, int, int]:
        return (0, 0, 0)

    async def vacuum(self) -> None:
        pass

    async def check_integrity(self) -> bool:
        return True
