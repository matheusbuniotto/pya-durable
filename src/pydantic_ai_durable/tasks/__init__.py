"""Tasks, Durable Replay, Crash Recovery, Sagas, and Human Approval Gates."""

from pydantic_ai_durable.tasks.approvals import (
    ApprovalExpiredError,
    ApprovalManager,
    ApprovalRejectedError,
    ApprovalRequiredError,
    InvalidApprovalDecisionError,
    approval_gate,
)
from pydantic_ai_durable.tasks.recovery import CrashRecoveryManager, TaskRecoveryResult
from pydantic_ai_durable.tasks.replay import (
    ConfigurationError,
    InvalidReplayPolicyError,
    create_durable_task,
    current_durable_context,
    durable_execution_context,
    durable_tool,
    set_tool_partial_output,
    wait_for_durable_task,
)
from pydantic_ai_durable.tasks.sagas import (
    SagaContext,
    SagaCoordinator,
    SagaDefinition,
    SagaExecutionError,
    SagaResult,
    define_saga,
)
from pydantic_ai_durable.tasks.tree import (
    CycleDetectedError,
    TaskContext,
    TaskDefinition,
    TaskPhaseResult,
    TaskRunner,
    check_cycle,
    define_task,
)

__all__ = [
    "ApprovalExpiredError",
    "ApprovalManager",
    "ApprovalRejectedError",
    "ApprovalRequiredError",
    "ConfigurationError",
    "CrashRecoveryManager",
    "CycleDetectedError",
    "InvalidApprovalDecisionError",
    "InvalidReplayPolicyError",
    "SagaContext",
    "SagaCoordinator",
    "SagaDefinition",
    "SagaExecutionError",
    "SagaResult",
    "TaskContext",
    "TaskDefinition",
    "TaskPhaseResult",
    "TaskRecoveryResult",
    "TaskRunner",
    "approval_gate",
    "check_cycle",
    "create_durable_task",
    "current_durable_context",
    "define_saga",
    "define_task",
    "durable_execution_context",
    "durable_tool",
    "set_tool_partial_output",
    "wait_for_durable_task",
]
