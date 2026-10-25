import asyncio
import inspect
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter

from pya_durable.engine import DurableStepEngine, TurnResult
from pya_durable.models import (
    ReplayPolicy,
    TaskRecord,
    TaskStatus,
    ToolInvocationRecord,
    ToolInvocationStatus,
    TranscriptEntryKind,
    utc_now_iso,
)
from pya_durable.replay import TOOL_REGISTRY, durable_execution_context
from pya_durable.storage import DurableStorage

logger = logging.getLogger("pya_durable.recovery")


@dataclass
class TaskRecoveryResult:
    task_id: str
    previous_status: TaskStatus
    new_status: TaskStatus
    reconciled_phase: str
    safe_tools_replayed: int = 0
    unsafe_tools_interrupted: int = 0
    errors: list[str] = field(default_factory=list)


class CrashRecoveryManager:
    def __init__(
        self,
        storage: DurableStorage,
        agent: Agent[Any, Any] | None = None,
        tool_registry: dict[str, Callable[..., Any]] | None = None,
    ) -> None:
        self.storage = storage
        self.agent = agent
        self.tool_registry = tool_registry if tool_registry is not None else TOOL_REGISTRY

    async def reconcile_task(self, task: TaskRecord) -> TaskRecoveryResult:
        invocations = await self.storage.get_tool_invocations_by_task(task.task_id)
        started_invocations = [
            inv for inv in invocations if inv.status == ToolInvocationStatus.STARTED
        ]

        safe_replayed = 0
        unsafe_interrupted = 0
        errors: list[str] = []

        for inv in started_invocations:
            if inv.replay_policy == ReplayPolicy.UNSAFE:
                await self.storage.record_tool_interrupted(
                    invocation_id=inv.invocation_id,
                    partial_output=inv.partial_output,
                )
                unsafe_interrupted += 1
            elif inv.replay_policy == ReplayPolicy.SAFE:
                tool_func = self.tool_registry.get(inv.tool_name)
                if tool_func is not None:
                    try:
                        args = json.loads(inv.args_json) if inv.args_json else {}
                        timeout = getattr(tool_func, "timeout_seconds", None)

                        async with durable_execution_context(
                            self.storage,
                            task_id=task.task_id,
                            conversation_id=task.owner_id,
                        ):
                            if inspect.iscoroutinefunction(tool_func):
                                if timeout is not None:
                                    res = await asyncio.wait_for(
                                        tool_func(**args), timeout=timeout
                                    )
                                else:
                                    res = await tool_func(**args)
                            else:
                                raw_fn = getattr(tool_func, "raw_func", tool_func)
                                if timeout is not None:
                                    res = await asyncio.wait_for(
                                        asyncio.to_thread(raw_fn, **args),
                                        timeout=timeout,
                                    )
                                else:
                                    res = raw_fn(**args)

                        if hasattr(res, "model_dump_json"):
                            res_json = res.model_dump_json()
                        elif isinstance(res, (dict, list, int, float, bool, str, type(None))):
                            res_json = json.dumps(res)
                        else:
                            res_json = json.dumps({"result": str(res)})

                        await self.storage.record_tool_completed(inv.invocation_id, res_json)
                        safe_replayed += 1
                    except asyncio.TimeoutError:
                        err_msg = f"Tool replay timed out for {inv.tool_name}"
                        await self.storage.record_tool_failed(inv.invocation_id, err_msg)
                        errors.append(err_msg)
                    except Exception as exc:
                        err_msg = f"Tool replay error: {exc}"
                        await self.storage.record_tool_failed(inv.invocation_id, str(exc))
                        errors.append(err_msg)
                else:
                    await self.storage.record_tool_interrupted(
                        invocation_id=inv.invocation_id,
                        partial_output=inv.partial_output,
                    )
                    safe_replayed += 1

        if unsafe_interrupted > 0:
            new_status = TaskStatus.RUNNING
            reconciled_phase = "interrupted"
            await self.storage.update_task_state(
                task_id=task.task_id,
                status=new_status,
                phase=reconciled_phase,
                state_json=json.dumps({
                    "interrupted": True,
                    "unsafe_interrupted": unsafe_interrupted,
                    "safe_replayed": safe_replayed,
                    "reconciled": True,
                }),
                checkpoint_json=task.checkpoint_json,
            )
        elif errors:
            new_status = TaskStatus.FAILED
            reconciled_phase = "failed"
            await self.storage.update_task_state(
                task_id=task.task_id,
                status=new_status,
                phase=reconciled_phase,
                state_json=json.dumps({
                    "errors": errors,
                    "reconciled": True,
                }),
                checkpoint_json=task.checkpoint_json,
            )
        else:
            new_status = TaskStatus.RUNNING
            reconciled_phase = "reconciled"
            await self.storage.update_task_state(
                task_id=task.task_id,
                status=new_status,
                phase=reconciled_phase,
                state_json=json.dumps({
                    "safe_replayed": safe_replayed,
                    "reconciled": True,
                }),
                checkpoint_json=task.checkpoint_json,
            )

        logger.info(
            "Reconciliation finished, resumed task id %s from phase %s",
            task.task_id,
            task.phase,
        )

        return TaskRecoveryResult(
            task_id=task.task_id,
            previous_status=task.status,
            new_status=new_status,
            reconciled_phase=reconciled_phase,
            safe_tools_replayed=safe_replayed,
            unsafe_tools_interrupted=unsafe_interrupted,
            errors=errors,
        )

    async def reconcile_all(self) -> list[TaskRecoveryResult]:
        running_tasks = await self.storage.get_running_tasks()
        if not running_tasks:
            logger.info("Clean database recovery. Zero running tasks found.")
            return []

        results = await asyncio.gather(*[self.reconcile_task(t) for t in running_tasks])
        return list(results)

    async def resume(
        self,
        task_id: str | None = None,
        agent: Agent[Any, Any] | None = None,
    ) -> Any:
        if task_id is None:
            return await self.reconcile_all()

        task = await self.storage.get_task(task_id)
        if task is None:
            raise ValueError(f"Task '{task_id}' not found")

        recovery_result = await self.reconcile_task(task)

        active_agent = agent or self.agent
        if active_agent is not None and recovery_result.unsafe_tools_interrupted == 0 and not recovery_result.errors:
            prompt = ""
            try:
                data = json.loads(task.input_json)
                prompt = data.get("prompt", "")
            except Exception:
                prompt = task.input_json

            checkpoint_entry_id = task.checkpoint_json
            history: list[ModelMessage] = []
            if checkpoint_entry_id:
                ancestors = await self.storage.get_ancestor_entries(checkpoint_entry_id)
                for e in ancestors:
                    if e.kind in (
                        TranscriptEntryKind.MODEL_REQUEST,
                        TranscriptEntryKind.MODEL_RESPONSE,
                        TranscriptEntryKind.TOOL_CALL,
                    ):
                        try:
                            parsed = ModelMessagesTypeAdapter.validate_json(e.message_json)
                            history.extend(parsed)
                        except Exception:
                            pass

            engine = DurableStepEngine(self.storage)
            turn_result = await engine.run_turn(
                agent=active_agent,
                user_prompt=prompt,
                conversation_id=task.owner_id,
                task_id=task.task_id,
                parent_entry_id=checkpoint_entry_id,
                message_history=history if history else None,
            )
            logger.info(
                "Reconciliation finished, resumed task id %s from phase %s",
                task.task_id,
                task.phase,
            )
            return turn_result

        return [recovery_result]
