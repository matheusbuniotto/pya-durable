from datetime import datetime, timezone
from enum import Enum
from typing import Literal, NewType
from pydantic import BaseModel, ConfigDict, Field

ConversationId = NewType("ConversationId", str)
TaskId = NewType("TaskId", str)
EntryId = NewType("EntryId", str)
InvocationId = NewType("InvocationId", str)
RequestId = NewType("RequestId", str)
DocId = NewType("DocId", str)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    WAITING = "waiting"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"


class ReplayPolicy(str, Enum):
    SAFE = "safe"
    UNSAFE = "unsafe"


class ToolInvocationStatus(str, Enum):
    STARTED = "started"
    COMPLETED = "completed"
    INTERRUPTED = "interrupted"
    FAILED = "failed"


class SubmissionStatus(str, Enum):
    ACTIVE = "active"
    SETTLED = "settled"


class TranscriptEntryKind(str, Enum):
    USER = "user"
    MODEL_REQUEST = "model_request"
    MODEL_RESPONSE = "model_response"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    SYSTEM = "system"


class ConversationRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    conversation_id: str
    parent_conversation_id: str | None = None
    fork_entry_id: str | None = None
    agent_config_json: str
    created_at: str = Field(default_factory=utc_now_iso)
    updated_at: str = Field(default_factory=utc_now_iso)


class TranscriptEntryRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    entry_id: str
    conversation_id: str
    parent_entry_id: str | None = None
    kind: TranscriptEntryKind
    message_json: str
    created_at: str = Field(default_factory=utc_now_iso)


class TaskRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    task_id: str
    task_type: str
    version: int = 1
    status: TaskStatus
    owner_kind: Literal["root", "conversation", "task"]
    owner_id: str
    phase: str
    input_json: str
    state_json: str
    checkpoint_json: str | None = None
    policy: str = "failFast"
    created_at: str = Field(default_factory=utc_now_iso)
    updated_at: str = Field(default_factory=utc_now_iso)


class ToolInvocationRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    invocation_id: str
    task_id: str
    conversation_id: str
    tool_name: str
    args_json: str
    replay_policy: ReplayPolicy
    status: ToolInvocationStatus
    result_json: str | None = None
    partial_output: str | None = None
    error_message: str | None = None
    started_at: str = Field(default_factory=utc_now_iso)
    completed_at: str | None = None


class SubmissionRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    request_id: str
    conversation_id: str
    prompt_text: str
    task_id: str
    status: SubmissionStatus
    result_json: str | None = None
    created_at: str = Field(default_factory=utc_now_iso)


class DurableDocRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    doc_id: str
    conversation_id: str
    version: int = 1
    doc_type: str
    data_json: str
    updated_at: str = Field(default_factory=utc_now_iso)
