import asyncio
import functools
import inspect
import json
import logging
import uuid
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, AsyncIterator, Callable, TypeVar

from datetime import datetime, timezone
from pydantic_ai_durable.extensions import (
    HookContext,
    HookDecision,
    ToolCallInfo,
    global_extension_registry,
)
from pydantic_ai_durable.models import (
    ApprovalDecision,
    ApprovalRequestRecord,
    ApprovalStatus,
    ReplayPolicy,
    TaskRecord,
    TaskStatus,
    ToolInvocationRecord,
    ToolInvocationStatus,
    utc_now_iso,
)
from pydantic_ai_durable.storage import DurableStorage

logger = logging.getLogger("pydantic_ai_durable.replay")

F = TypeVar("F", bound=Callable[..., Any])


class ConfigurationError(ValueError):
    pass


class InvalidReplayPolicyError(ConfigurationError):
    pass


@dataclass
class DurableContext:
    storage: DurableStorage
    task_id: str
    conversation_id: str
    last_invocation_id: str | None = None
    last_tool_name: str | None = None
    last_challenge_token: str | None = None

    async def create_task(
        self,
        task_def_or_name: Any,
        input_data: Any,
        policy: str = "failFast",
    ) -> TaskRecord:
        import uuid
        child_task_id = f"task-{uuid.uuid4().hex[:12]}"
        task_name = getattr(task_def_or_name, "name", str(task_def_or_name))
        in_json = json.dumps(input_data) if not isinstance(input_data, str) else input_data
        record = TaskRecord(
            task_id=child_task_id,
            task_type=task_name,
            version=1,
            status=TaskStatus.RUNNING,
            owner_kind="task",
            owner_id=self.task_id,
            phase="start",
            input_json=in_json,
            state_json="{}",
            policy=policy,
        )
        await self.storage.insert_task(record)
        return record

    async def wait_for_task(
        self,
        task_id: str,
        poll_interval: float = 0.05,
        timeout: float | None = None,
    ) -> TaskRecord:
        import time
        start_time = time.monotonic()
        while True:
            rec = await self.storage.get_task(task_id)
            if rec is not None and rec.status in (
                TaskStatus.COMPLETED,
                TaskStatus.FAILED,
                TaskStatus.ABORTED,
            ):
                return rec
            if timeout is not None and (time.monotonic() - start_time) > timeout:
                raise TimeoutError(f"Timed out waiting for task {task_id}")
            await asyncio.sleep(poll_interval)


async def create_durable_task(task_def_or_name: Any, input_data: Any, policy: str = "failFast") -> TaskRecord:
    ctx = current_durable_context.get()
    if ctx is None:
        raise RuntimeError("No active durable execution context")
    return await ctx.create_task(task_def_or_name, input_data, policy=policy)


async def wait_for_durable_task(task_id: str, timeout: float | None = None) -> TaskRecord:
    ctx = current_durable_context.get()
    if ctx is None:
        raise RuntimeError("No active durable execution context")
    return await ctx.wait_for_task(task_id, timeout=timeout)


current_durable_context: ContextVar[DurableContext | None] = ContextVar(
    "current_durable_context", default=None
)

TOOL_REGISTRY: dict[str, Callable[..., Any]] = {}


@asynccontextmanager
async def durable_execution_context(
    storage: DurableStorage,
    task_id: str,
    conversation_id: str,
) -> AsyncIterator[DurableContext]:
    ctx = DurableContext(
        storage=storage,
        task_id=task_id,
        conversation_id=conversation_id,
    )
    token = current_durable_context.set(ctx)
    try:
        yield ctx
    finally:
        current_durable_context.reset(token)


async def set_tool_partial_output(partial_text: str) -> None:
    ctx = current_durable_context.get()
    if ctx is not None and ctx.last_invocation_id is not None:
        await ctx.storage.update_tool_partial_output(ctx.last_invocation_id, partial_text)


def _serialize_arguments(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> str:
    try:
        sig = inspect.signature(fn)
        bound = sig.bind_partial(*args, **kwargs)
        args_dict: dict[str, Any] = {}
        for k, v in bound.arguments.items():
            if type(v).__name__ == "RunContext" or k == "ctx":
                continue
            if hasattr(v, "model_dump"):
                args_dict[k] = v.model_dump()
            elif isinstance(v, (str, int, float, bool, list, dict, type(None))):
                args_dict[k] = v
            else:
                args_dict[k] = str(v)
        return json.dumps(args_dict)
    except Exception:
        clean_kwargs = {
            k: str(v) for k, v in kwargs.items() if type(v).__name__ != "RunContext"
        }
        return json.dumps(clean_kwargs)


def _serialize_result(result: Any) -> str:
    if hasattr(result, "model_dump_json"):
        return result.model_dump_json()
    if isinstance(result, (dict, list, int, float, bool, str, type(None))):
        return json.dumps(result)
    return json.dumps({"result": str(result)})


def validate_replay_policy(replay: str | ReplayPolicy) -> ReplayPolicy:
    if isinstance(replay, ReplayPolicy):
        return replay
    if isinstance(replay, str):
        cleaned = replay.strip().lower()
        if cleaned == "safe":
            return ReplayPolicy.SAFE
        if cleaned == "unsafe":
            return ReplayPolicy.UNSAFE
    raise InvalidReplayPolicyError(
        f"Invalid replay policy '{replay}'. Allowed policies are 'safe' and 'unsafe'."
    )


def durable_tool(
    replay: str | ReplayPolicy | Callable[..., Any] = ReplayPolicy.SAFE,
    timeout_seconds: float | None = None,
    name: str | None = None,
    require_approval: bool = False,
    approval_prompt: str | None = None,
    approval_timeout_seconds: float | None = None,
) -> Any:
    if callable(replay) and not isinstance(replay, (str, ReplayPolicy)):
        actual_fn = replay
        policy = ReplayPolicy.SAFE

        return _create_tool_wrapper(
            actual_fn,
            policy=policy,
            timeout_seconds=timeout_seconds,
            name=name,
            require_approval=require_approval,
            approval_prompt=approval_prompt,
            approval_timeout_seconds=approval_timeout_seconds,
        )

    policy = validate_replay_policy(replay)

    def decorator(fn: F) -> F:
        return _create_tool_wrapper(
            fn,
            policy=policy,
            timeout_seconds=timeout_seconds,
            name=name,
            require_approval=require_approval,
            approval_prompt=approval_prompt,
            approval_timeout_seconds=approval_timeout_seconds,
        )

    return decorator


def _create_tool_wrapper(
    fn: Callable[..., Any],
    policy: ReplayPolicy,
    timeout_seconds: float | None,
    name: str | None = None,
    require_approval: bool = False,
    approval_prompt: str | None = None,
    approval_timeout_seconds: float | None = None,
) -> Any:
    tool_name = name if name is not None else fn.__name__
    is_async = inspect.iscoroutinefunction(fn)

    if is_async:
        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            ctx = current_durable_context.get()

            try:
                sig = inspect.signature(fn)
                bound = sig.bind_partial(*args, **kwargs)
                args_dict = {
                    k: v for k, v in bound.arguments.items()
                    if type(v).__name__ != "RunContext" and k != "ctx"
                }
            except Exception:
                args_dict = {
                    k: v for k, v in kwargs.items()
                    if type(v).__name__ != "RunContext" and k != "ctx"
                }

            call_info = ToolCallInfo(
                name=tool_name,
                args=args_dict,
                conversation_id=ctx.conversation_id if ctx else None,
                task_id=ctx.task_id if ctx else None,
            )
            hook_ctx = HookContext(
                conversation_id=ctx.conversation_id if ctx else None,
                task_id=ctx.task_id if ctx else None,
                storage=ctx.storage if ctx else None,
            )

            decision = await global_extension_registry.run_before_tool(call_info, hook_ctx)
            if decision is not None and decision.block:
                if ctx is not None:
                    invocation_id = f"inv-{uuid.uuid4().hex[:16]}"
                    ctx.last_invocation_id = invocation_id
                    ctx.last_tool_name = tool_name
                    args_json = _serialize_arguments(fn, *args, **kwargs)
                    record = ToolInvocationRecord(
                        invocation_id=invocation_id,
                        task_id=ctx.task_id,
                        conversation_id=ctx.conversation_id,
                        tool_name=tool_name,
                        args_json=args_json,
                        replay_policy=policy,
                        status=ToolInvocationStatus.FAILED,
                        error_message=f"Blocked: {decision.block}",
                    )
                    await ctx.storage.record_tool_started(record)
                    await ctx.storage.record_tool_failed(invocation_id, f"Blocked: {decision.block}")
                return {"error": decision.block, "blocked": True}

            if decision is not None and decision.modified_args is not None:
                if args and type(args[0]).__name__ == "RunContext" and "ctx" not in decision.modified_args:
                    exec_args = (args[0],)
                    exec_kwargs = decision.modified_args
                else:
                    exec_args = ()
                    exec_kwargs = decision.modified_args
            else:
                exec_args = args
                exec_kwargs = kwargs

            if ctx is None:
                needs_approval = (
                    require_approval
                    or getattr(fn, "_requires_approval", False)
                    or (decision is not None and getattr(decision, "require_approval", False))
                )
                if needs_approval:
                    from pydantic_ai_durable.approvals import ApprovalRequiredError
                    prompt_template = (
                        approval_prompt
                        or getattr(fn, "_approval_prompt", None)
                        or (getattr(decision, "approval_prompt", None) if decision else None)
                    )
                    raise ApprovalRequiredError(
                        challenge_token="standalone-appr",
                        tool_name=tool_name,
                        args=args_dict,
                        prompt_message=prompt_template or f"Approve execution of tool '{tool_name}'",
                    )
                if timeout_seconds is not None:
                    res = await asyncio.wait_for(fn(*exec_args, **exec_kwargs), timeout=timeout_seconds)
                else:
                    res = await fn(*exec_args, **exec_kwargs)
                return await global_extension_registry.run_after_tool(call_info, res, hook_ctx)

            needs_approval = (
                require_approval
                or getattr(fn, "_requires_approval", False)
                or (decision is not None and getattr(decision, "require_approval", False))
            )
            prompt_template = (
                approval_prompt
                or getattr(fn, "_approval_prompt", None)
                or (getattr(decision, "approval_prompt", None) if decision else None)
            )
            appr_timeout = approval_timeout_seconds or getattr(fn, "_approval_timeout", None)

            if needs_approval:
                appr = await ctx.storage.get_approval_by_task_and_tool(ctx.task_id, tool_name)
                if appr is not None:
                    if appr.status == ApprovalStatus.PENDING and appr.expires_at is not None:
                        try:
                            if datetime.now(timezone.utc) > datetime.fromisoformat(appr.expires_at):
                                dec = ApprovalDecision(
                                    status=ApprovalStatus.EXPIRED,
                                    approver="system",
                                    reason="Approval deadline expired",
                                    resolved_at=utc_now_iso(),
                                )
                                appr = await ctx.storage.resolve_approval_request(
                                    challenge_token=appr.challenge_token,
                                    status=ApprovalStatus.EXPIRED,
                                    decision_json=dec.model_dump_json(),
                                    resolved_at=dec.resolved_at,
                                )
                        except Exception:
                            pass

                    if appr.status == ApprovalStatus.APPROVED:
                        if appr.decision_json:
                            try:
                                dec = ApprovalDecision.model_validate_json(appr.decision_json)
                                if dec.override_args_json:
                                    overrides = json.loads(dec.override_args_json)
                                    combined = dict(args_dict)
                                    combined.update(overrides)
                                    if exec_args and type(exec_args[0]).__name__ == "RunContext" and "ctx" not in combined:
                                        exec_args = (exec_args[0],)
                                    else:
                                        exec_args = ()
                                    exec_kwargs = combined
                            except Exception:
                                pass
                    elif appr.status == ApprovalStatus.REJECTED:
                        reason = ""
                        if appr.decision_json:
                            try:
                                dec = ApprovalDecision.model_validate_json(appr.decision_json)
                                reason = dec.reason
                            except Exception:
                                pass
                        msg = f"Tool execution rejected by human operator. Reason: {reason or 'Denied'}"
                        invocation_id = f"inv-{uuid.uuid4().hex[:16]}"
                        ctx.last_invocation_id = invocation_id
                        ctx.last_tool_name = tool_name
                        args_json = _serialize_arguments(fn, *exec_args, **exec_kwargs)
                        record = ToolInvocationRecord(
                            invocation_id=invocation_id,
                            task_id=ctx.task_id,
                            conversation_id=ctx.conversation_id,
                            tool_name=tool_name,
                            args_json=args_json,
                            replay_policy=policy,
                            status=ToolInvocationStatus.FAILED,
                            error_message=msg,
                        )
                        await ctx.storage.record_tool_started(record)
                        await ctx.storage.record_tool_failed(invocation_id, msg)
                        from pydantic_ai_durable.approvals import ApprovalRejectedError
                        raise ApprovalRejectedError(challenge_token=appr.challenge_token, reason=reason)
                    elif appr.status == ApprovalStatus.EXPIRED:
                        msg = "Tool execution approval challenge expired."
                        invocation_id = f"inv-{uuid.uuid4().hex[:16]}"
                        ctx.last_invocation_id = invocation_id
                        ctx.last_tool_name = tool_name
                        args_json = _serialize_arguments(fn, *exec_args, **exec_kwargs)
                        record = ToolInvocationRecord(
                            invocation_id=invocation_id,
                            task_id=ctx.task_id,
                            conversation_id=ctx.conversation_id,
                            tool_name=tool_name,
                            args_json=args_json,
                            replay_policy=policy,
                            status=ToolInvocationStatus.FAILED,
                            error_message=msg,
                        )
                        await ctx.storage.record_tool_started(record)
                        await ctx.storage.record_tool_failed(invocation_id, msg)
                        from pydantic_ai_durable.approvals import ApprovalExpiredError
                        raise ApprovalExpiredError(challenge_token=appr.challenge_token)
                    else:
                        ctx.last_challenge_token = appr.challenge_token
                        from pydantic_ai_durable.approvals import ApprovalRequiredError
                        raise ApprovalRequiredError(
                            challenge_token=appr.challenge_token,
                            tool_name=tool_name,
                            args=args_dict,
                            prompt_message=appr.prompt_message,
                            expires_at=appr.expires_at,
                        )
                else:
                    from pydantic_ai_durable.approvals import ApprovalManager, ApprovalRequiredError

                    appr_mgr = ApprovalManager(ctx.storage)
                    record = await appr_mgr.create_challenge(
                        task_id=ctx.task_id,
                        conversation_id=ctx.conversation_id,
                        tool_name=tool_name,
                        args=args_dict,
                        prompt_message=prompt_template,
                        timeout_seconds=appr_timeout,
                    )
                    ctx.last_challenge_token = record.challenge_token

                    invocation_id = f"inv-{uuid.uuid4().hex[:16]}"
                    ctx.last_invocation_id = invocation_id
                    ctx.last_tool_name = tool_name
                    args_json = _serialize_arguments(fn, *exec_args, **exec_kwargs)
                    inv_record = ToolInvocationRecord(
                        invocation_id=invocation_id,
                        task_id=ctx.task_id,
                        conversation_id=ctx.conversation_id,
                        tool_name=tool_name,
                        args_json=args_json,
                        replay_policy=policy,
                        status=ToolInvocationStatus.DEFERRED,
                        partial_output=f"Approval challenge token: {record.challenge_token}",
                    )
                    await ctx.storage.record_tool_started(inv_record)

                    await ctx.storage.update_task_state(
                        task_id=ctx.task_id,
                        status=TaskStatus.WAITING,
                        phase="waiting_approval",
                        state_json=json.dumps({
                            "waiting_approval": True,
                            "challenge_token": record.challenge_token,
                            "tool_name": tool_name,
                        }),
                        checkpoint_json=record.challenge_token,
                    )

                    raise ApprovalRequiredError(
                        challenge_token=record.challenge_token,
                        tool_name=tool_name,
                        args=args_dict,
                        prompt_message=record.prompt_message,
                        expires_at=record.expires_at,
                    )

            invocation_id = f"inv-{uuid.uuid4().hex[:16]}"
            ctx.last_invocation_id = invocation_id
            ctx.last_tool_name = tool_name
            args_json = _serialize_arguments(fn, *exec_args, **exec_kwargs)

            record = ToolInvocationRecord(
                invocation_id=invocation_id,
                task_id=ctx.task_id,
                conversation_id=ctx.conversation_id,
                tool_name=tool_name,
                args_json=args_json,
                replay_policy=policy,
                status=ToolInvocationStatus.STARTED,
            )
            await ctx.storage.record_tool_started(record)

            try:
                if timeout_seconds is not None:
                    res = await asyncio.wait_for(fn(*exec_args, **exec_kwargs), timeout=timeout_seconds)
                else:
                    res = await fn(*exec_args, **exec_kwargs)

                sanitized = await global_extension_registry.run_after_tool(call_info, res, hook_ctx)
                result_json = _serialize_result(sanitized)
                await ctx.storage.record_tool_completed(invocation_id, result_json)
                return sanitized
            except asyncio.TimeoutError:
                msg = f"Tool invocation timed out after {timeout_seconds}s"
                await ctx.storage.record_tool_failed(invocation_id, msg)
                raise
            except Exception as exc:
                await ctx.storage.record_tool_failed(invocation_id, str(exc))
                raise
    else:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            ctx = current_durable_context.get()

            try:
                sig = inspect.signature(fn)
                bound = sig.bind_partial(*args, **kwargs)
                args_dict = {
                    k: v for k, v in bound.arguments.items()
                    if type(v).__name__ != "RunContext" and k != "ctx"
                }
            except Exception:
                args_dict = {
                    k: v for k, v in kwargs.items()
                    if type(v).__name__ != "RunContext" and k != "ctx"
                }

            call_info = ToolCallInfo(
                name=tool_name,
                args=args_dict,
                conversation_id=ctx.conversation_id if ctx else None,
                task_id=ctx.task_id if ctx else None,
            )
            hook_ctx = HookContext(
                conversation_id=ctx.conversation_id if ctx else None,
                task_id=ctx.task_id if ctx else None,
                storage=ctx.storage if ctx else None,
            )

            decision = global_extension_registry.run_before_tool_sync(call_info, hook_ctx)
            if decision is not None and decision.block:
                if ctx is not None:
                    invocation_id = f"inv-{uuid.uuid4().hex[:16]}"
                    ctx.last_invocation_id = invocation_id
                    ctx.last_tool_name = tool_name
                    args_json = _serialize_arguments(fn, *args, **kwargs)
                    record = ToolInvocationRecord(
                        invocation_id=invocation_id,
                        task_id=ctx.task_id,
                        conversation_id=ctx.conversation_id,
                        tool_name=tool_name,
                        args_json=args_json,
                        replay_policy=policy,
                        status=ToolInvocationStatus.FAILED,
                        error_message=f"Blocked: {decision.block}",
                    )
                    try:
                        loop = asyncio.get_running_loop()
                        loop.create_task(ctx.storage.record_tool_started(record))
                        loop.create_task(ctx.storage.record_tool_failed(invocation_id, f"Blocked: {decision.block}"))
                    except RuntimeError:
                        pass
                return {"error": decision.block, "blocked": True}

            if decision is not None and decision.modified_args is not None:
                if args and type(args[0]).__name__ == "RunContext" and "ctx" not in decision.modified_args:
                    exec_args = (args[0],)
                    exec_kwargs = decision.modified_args
                else:
                    exec_args = ()
                    exec_kwargs = decision.modified_args
            else:
                exec_args = args
                exec_kwargs = kwargs

            if ctx is None:
                res = fn(*exec_args, **exec_kwargs)
                return global_extension_registry.run_after_tool_sync(call_info, res, hook_ctx)

            invocation_id = f"inv-{uuid.uuid4().hex[:16]}"
            ctx.last_invocation_id = invocation_id
            ctx.last_tool_name = tool_name
            args_json = _serialize_arguments(fn, *exec_args, **exec_kwargs)

            record = ToolInvocationRecord(
                invocation_id=invocation_id,
                task_id=ctx.task_id,
                conversation_id=ctx.conversation_id,
                tool_name=tool_name,
                args_json=args_json,
                replay_policy=policy,
                status=ToolInvocationStatus.STARTED,
            )

            try:
                loop = asyncio.get_running_loop()
                loop.create_task(ctx.storage.record_tool_started(record))
            except RuntimeError:
                pass

            try:
                res = fn(*exec_args, **exec_kwargs)
                sanitized = global_extension_registry.run_after_tool_sync(call_info, res, hook_ctx)
                result_json = _serialize_result(sanitized)
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(ctx.storage.record_tool_completed(invocation_id, result_json))
                except RuntimeError:
                    pass
                return sanitized
            except Exception as exc:
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(ctx.storage.record_tool_failed(invocation_id, str(exc)))
                except RuntimeError:
                    pass
                raise

    wrapper.__name__ = tool_name
    wrapper.name = tool_name
    wrapper._is_durable_tool = True  # type: ignore[attr-defined]
    wrapper.replay_policy = policy  # type: ignore[attr-defined]
    wrapper.timeout_seconds = timeout_seconds  # type: ignore[attr-defined]
    wrapper.raw_func = fn  # type: ignore[attr-defined]
    wrapper._requires_approval = require_approval or getattr(fn, "_requires_approval", False)  # type: ignore[attr-defined]
    wrapper._approval_prompt = approval_prompt or getattr(fn, "_approval_prompt", None)  # type: ignore[attr-defined]
    wrapper._approval_timeout = approval_timeout_seconds or getattr(fn, "_approval_timeout", None)  # type: ignore[attr-defined]

    TOOL_REGISTRY[tool_name] = wrapper
    return wrapper
