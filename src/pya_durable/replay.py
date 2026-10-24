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

from pya_durable.models import (
    ReplayPolicy,
    ToolInvocationRecord,
    ToolInvocationStatus,
)
from pya_durable.storage import AioSqliteStorage

logger = logging.getLogger("pya_durable.replay")

F = TypeVar("F", bound=Callable[..., Any])


class ConfigurationError(ValueError):
    pass


class InvalidReplayPolicyError(ConfigurationError):
    pass


@dataclass
class DurableContext:
    storage: AioSqliteStorage
    task_id: str
    conversation_id: str
    last_invocation_id: str | None = None
    last_tool_name: str | None = None


current_durable_context: ContextVar[DurableContext | None] = ContextVar(
    "current_durable_context", default=None
)

TOOL_REGISTRY: dict[str, Callable[..., Any]] = {}


@asynccontextmanager
async def durable_execution_context(
    storage: AioSqliteStorage,
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
) -> Any:
    if callable(replay) and not isinstance(replay, (str, ReplayPolicy)):
        actual_fn = replay
        policy = ReplayPolicy.SAFE

        return _create_tool_wrapper(
            actual_fn,
            policy=policy,
            timeout_seconds=timeout_seconds,
        )

    policy = validate_replay_policy(replay)

    def decorator(fn: F) -> F:
        return _create_tool_wrapper(
            fn,
            policy=policy,
            timeout_seconds=timeout_seconds,
        )

    return decorator


def _create_tool_wrapper(
    fn: Callable[..., Any],
    policy: ReplayPolicy,
    timeout_seconds: float | None,
) -> Any:
    is_async = inspect.iscoroutinefunction(fn)

    if is_async:
        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            ctx = current_durable_context.get()
            if ctx is None:
                if timeout_seconds is not None:
                    return await asyncio.wait_for(fn(*args, **kwargs), timeout=timeout_seconds)
                return await fn(*args, **kwargs)

            invocation_id = f"inv-{uuid.uuid4().hex[:16]}"
            ctx.last_invocation_id = invocation_id
            ctx.last_tool_name = fn.__name__
            args_json = _serialize_arguments(fn, *args, **kwargs)

            record = ToolInvocationRecord(
                invocation_id=invocation_id,
                task_id=ctx.task_id,
                conversation_id=ctx.conversation_id,
                tool_name=fn.__name__,
                args_json=args_json,
                replay_policy=policy,
                status=ToolInvocationStatus.STARTED,
            )
            await ctx.storage.record_tool_started(record)

            try:
                if timeout_seconds is not None:
                    res = await asyncio.wait_for(fn(*args, **kwargs), timeout=timeout_seconds)
                else:
                    res = await fn(*args, **kwargs)

                result_json = _serialize_result(res)
                await ctx.storage.record_tool_completed(invocation_id, result_json)
                return res
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
            if ctx is None:
                return fn(*args, **kwargs)

            invocation_id = f"inv-{uuid.uuid4().hex[:16]}"
            ctx.last_invocation_id = invocation_id
            ctx.last_tool_name = fn.__name__
            args_json = _serialize_arguments(fn, *args, **kwargs)

            record = ToolInvocationRecord(
                invocation_id=invocation_id,
                task_id=ctx.task_id,
                conversation_id=ctx.conversation_id,
                tool_name=fn.__name__,
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
                res = fn(*args, **kwargs)
                result_json = _serialize_result(res)
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(ctx.storage.record_tool_completed(invocation_id, result_json))
                except RuntimeError:
                    pass
                return res
            except Exception as exc:
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(ctx.storage.record_tool_failed(invocation_id, str(exc)))
                except RuntimeError:
                    pass
                raise

    wrapper._is_durable_tool = True  # type: ignore[attr-defined]
    wrapper.replay_policy = policy  # type: ignore[attr-defined]
    wrapper.timeout_seconds = timeout_seconds  # type: ignore[attr-defined]
    wrapper.raw_func = fn  # type: ignore[attr-defined]

    TOOL_REGISTRY[fn.__name__] = wrapper
    return wrapper
