from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal, NewType

from pydantic import BaseModel, ConfigDict, Field

ConversationId = NewType("ConversationId", str)
TaskId = NewType("TaskId", str)
EntryId = NewType("EntryId", str)
InvocationId = NewType("InvocationId", str)
RequestId = NewType("RequestId", str)
DocId = NewType("DocId", str)
ChallengeToken = NewType("ChallengeToken", str)


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
    DEFERRED = "deferred"


class ApprovalStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"


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


class MemoRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    conversation_id: str
    key: str
    value_json: str
    created_at: str = Field(default_factory=utc_now_iso)


class ApprovalDecision(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: ApprovalStatus
    approver: str = "operator"
    reason: str = ""
    override_args_json: str | None = None
    resolved_at: str = Field(default_factory=utc_now_iso)


class ApprovalRequestRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    challenge_token: str
    task_id: str
    conversation_id: str
    tool_name: str
    args_json: str
    prompt_message: str
    status: ApprovalStatus = ApprovalStatus.PENDING
    decision_json: str | None = None
    created_at: str = Field(default_factory=utc_now_iso)
    resolved_at: str | None = None
    expires_at: str | None = None


class SagaPolicy(str, Enum):
    FAIL_FAST = "failFast"
    ALL_SETTLED = "allSettled"


class SagaStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    WAITING = "waiting"
    COMPENSATING = "compensating"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"


class TaskOutcome(BaseModel):
    model_config = ConfigDict(frozen=True)

    task_id: str
    status: TaskStatus
    result: Any = None
    error: str | None = None

    @property
    def is_successful(self) -> bool:
        return self.status == TaskStatus.COMPLETED

    @property
    def is_terminal(self) -> bool:
        return self.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.ABORTED)


class SagaStepRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    step_name: str
    status: str = "completed"
    result_json: str | None = None
    executed_at: str = Field(default_factory=utc_now_iso)


def normalize_model_name(model_name: str | None) -> str | None:
    """Normalize model identifiers and shortcuts like 'gpt-luna 6'."""
    if not model_name:
        return None
    cleaned = model_name.strip().lower().replace(" ", "-")
    if cleaned in (
        "gpt-luna-6",
        "gpt-6-luna",
        "gpt-luna",
        "openai/gpt-6-luna",
        "openrouter:gpt-6-luna",
        "openrouter:openai/gpt-6-luna",
    ):
        return "openrouter:openai/gpt-6-luna"
    return model_name


