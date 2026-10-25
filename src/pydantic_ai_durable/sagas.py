import asyncio
import inspect
import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic_ai_durable.models import (
    SagaPolicy,
    TaskOutcome,
    TaskRecord,
    TaskStatus,
)
from pydantic_ai_durable.storage import DurableStorage

logger = logging.getLogger("pydantic_ai_durable.sagas")


class SagaExecutionError(RuntimeError):
    """Raised when a saga fails during execution."""


@dataclass
class SagaResult:
    action: str = "next"
    next_step: str | None = None
    state: dict[str, Any] | None = None
    result: Any = None
    error: str | None = None
    waiting_on: list[str] = field(default_factory=list)
    policy: SagaPolicy = SagaPolicy.FAIL_FAST

    @classmethod
    def next(cls, step: str, state: dict[str, Any] | None = None) -> "SagaResult":
        return cls(action="next", next_step=step, state=state)

    @classmethod
    def wait(
        cls,
        child_task_ids: list[str],
        next_step: str,
        policy: str | SagaPolicy = SagaPolicy.FAIL_FAST,
        state: dict[str, Any] | None = None,
    ) -> "SagaResult":
        eff_policy = SagaPolicy(policy) if isinstance(policy, str) else policy
        return cls(
            action="wait",
            next_step=next_step,
            waiting_on=list(child_task_ids),
            policy=eff_policy,
            state=state,
        )

    @classmethod
    def complete(cls, result: Any = None, state: dict[str, Any] | None = None) -> "SagaResult":
        return cls(action="complete", result=result, state=state)

    @classmethod
    def fail(cls, error: str = "", state: dict[str, Any] | None = None) -> "SagaResult":
        return cls(action="fail", error=error, state=state)

    @classmethod
    def abort(cls, reason: str = "", state: dict[str, Any] | None = None) -> "SagaResult":
        return cls(action="abort", error=reason, state=state)


class SagaContext:
    def __init__(
        self,
        task_id: str,
        storage: DurableStorage,
        input_data: Any,
        state: dict[str, Any] | None = None,
        current_step: str = "start",
        executed_steps: list[str] | None = None,
        coordinator: Any = None,
    ) -> None:
        self.task_id = task_id
        self.storage = storage
        self.input_data = input_data
        self.state = state if state is not None else {}
        self.current_step = current_step
        self.executed_steps = list(executed_steps or [])
        self.coordinator = coordinator

    async def spawn(
        self,
        task_def_or_name: Any,
        input_data: Any,
        policy: str | SagaPolicy = SagaPolicy.FAIL_FAST,
        initial_phase: str = "start",
    ) -> TaskRecord:
        if self.coordinator is not None:
            return await self.coordinator.spawn_child(
                parent_task_id=self.task_id,
                task_def_or_name=task_def_or_name,
                input_data=input_data,
                policy=policy,
                initial_phase=initial_phase,
            )
        raise RuntimeError("SagaCoordinator not attached to SagaContext")

    def wait_on(
        self,
        child_task_ids: list[str],
        next_step: str,
        policy: str | SagaPolicy = SagaPolicy.FAIL_FAST,
    ) -> SagaResult:
        return SagaResult.wait(
            child_task_ids=child_task_ids,
            next_step=next_step,
            policy=policy,
            state=self.state,
        )

    spawn_child = spawn

    def next_step(self, step_name: str) -> SagaResult:
        return SagaResult.next(step=step_name, state=self.state)

    transition_to = next_step

    def complete(self, result: Any = None) -> SagaResult:
        return SagaResult.complete(result=result, state=self.state)

    def fail(self, error: str = "") -> SagaResult:
        return SagaResult.fail(error=error, state=self.state)

    def abort(self, reason: str = "") -> SagaResult:
        return SagaResult.abort(reason=reason, state=self.state)

    async def outcomes(
        self, child_task_ids: list[str] | None = None
    ) -> dict[str, TaskOutcome]:
        if self.coordinator is not None:
            return await self.coordinator.get_outcomes(
                self.task_id, child_task_ids=child_task_ids
            )
        return {}


@dataclass
class SagaDefinition:
    name: str
    version: int = 1
    initial_step: str = "start"
    steps: dict[str, Callable[[SagaContext], Awaitable[SagaResult]]] = field(default_factory=dict)
    compensations: dict[str, Callable[[SagaContext], Awaitable[None]]] = field(default_factory=dict)
    global_abort: Callable[[SagaContext], Awaitable[None]] | None = None


def define_saga(
    target: Any = None,
    *,
    name: str | None = None,
    version: int = 1,
    initial_step: str = "start",
    steps: dict[str, Callable[[SagaContext], Awaitable[SagaResult]]] | None = None,
    compensations: dict[str, Callable[[SagaContext], Awaitable[None]]] | None = None,
    global_abort: Callable[[SagaContext], Awaitable[None]] | None = None,
) -> Any:
    if target is None:
        def decorator(cls_or_fn: Any) -> SagaDefinition:
            return _build_saga_def(
                cls_or_fn,
                name=name,
                version=version,
                initial_step=initial_step,
                steps=steps,
                compensations=compensations,
                global_abort=global_abort,
            )
        return decorator

    return _build_saga_def(
        target,
        name=name,
        version=version,
        initial_step=initial_step,
        steps=steps,
        compensations=compensations,
        global_abort=global_abort,
    )


def _build_saga_def(
    target: Any,
    name: str | None,
    version: int,
    initial_step: str,
    steps: dict[str, Callable[[SagaContext], Awaitable[SagaResult]]] | None,
    compensations: dict[str, Callable[[SagaContext], Awaitable[None]]] | None,
    global_abort: Callable[[SagaContext], Awaitable[None]] | None,
) -> SagaDefinition:
    saga_name = name or (getattr(target, "__name__", None) or str(target))
    extracted_steps: dict[str, Callable[[SagaContext], Awaitable[SagaResult]]] = dict(steps or {})
    extracted_compensations: dict[str, Callable[[SagaContext], Awaitable[None]]] = dict(compensations or {})
    extracted_abort = global_abort

    if inspect.isclass(target):
        instance = target()
        for attr_name in dir(instance):
            if attr_name.startswith("_"):
                continue
            attr = getattr(instance, attr_name)
            if callable(attr):
                if attr_name in ("abort", "on_abort", "compensate"):
                    extracted_abort = attr
                elif attr_name.startswith("compensate_"):
                    step_key = attr_name[len("compensate_"):]
                    extracted_compensations[step_key] = attr
                elif attr_name.endswith("_compensate"):
                    step_key = attr_name[:-len("_compensate")]
                    extracted_compensations[step_key] = attr
                else:
                    extracted_steps[attr_name] = attr

    return SagaDefinition(
        name=saga_name,
        version=version,
        initial_step=initial_step,
        steps=extracted_steps,
        compensations=extracted_compensations,
        global_abort=extracted_abort,
    )


class SagaCoordinator:
    def __init__(self, storage: DurableStorage, task_runner: Any = None, stream_hub: Any = None) -> None:
        self.storage = storage
        self.task_runner = task_runner
        self.stream_hub = stream_hub
        self.registry: dict[str, SagaDefinition] = {}

    def register_saga(self, saga_def: SagaDefinition) -> None:
        self.registry[saga_def.name] = saga_def

    async def spawn_child(
        self,
        parent_task_id: str,
        task_def_or_name: Any,
        input_data: Any,
        policy: str | SagaPolicy = SagaPolicy.FAIL_FAST,
        initial_phase: str = "start",
    ) -> TaskRecord:
        eff_policy = policy.value if isinstance(policy, SagaPolicy) else policy
        if self.task_runner is not None:
            if hasattr(task_def_or_name, "name"):
                task_type = task_def_or_name.name
                self.task_runner.register_task(task_def_or_name)
            else:
                task_type = str(task_def_or_name)
            
            ctx = self.task_runner._build_context_from_record(
                TaskRecord(
                    task_id=parent_task_id,
                    task_type="saga_parent",
                    status=TaskStatus.RUNNING,
                    owner_kind="root",
                    owner_id="root",
                    phase="running",
                    input_json="{}",
                    state_json="{}",
                )
            )
            return await ctx.create_child_task(
                task_type=task_type,
                input_data=input_data,
                policy=eff_policy,
                initial_phase=initial_phase,
            )
        raise RuntimeError("TaskRunner not configured on SagaCoordinator")

    async def get_outcomes(
        self, parent_task_id: str, child_task_ids: list[str] | None = None
    ) -> dict[str, TaskOutcome]:
        if self.task_runner is not None:
            children = await self.task_runner.get_child_tasks(parent_task_id)
        else:
            all_tasks: list[TaskRecord] = []
            for st in TaskStatus:
                all_tasks.extend(await self.storage.get_tasks_by_status(st))
            children = [t for t in all_tasks if t.owner_kind == "task" and t.owner_id == parent_task_id]

        target_ids = set(child_task_ids) if child_task_ids is not None else None
        outcomes_map: dict[str, TaskOutcome] = {}
        for child in children:
            if target_ids is not None and child.task_id not in target_ids:
                continue
            res_val = None
            err_msg = None
            if child.state_json:
                try:
                    parsed = json.loads(child.state_json)
                    res_val = parsed.get("output")
                    err_msg = parsed.get("error")
                except (json.JSONDecodeError, TypeError, KeyError):
                    logger.debug("Failed to decode child state_json: %s", child.state_json)
            outcomes_map[child.task_id] = TaskOutcome(
                task_id=child.task_id,
                status=child.status,
                result=res_val,
                error=err_msg,
            )
        return outcomes_map

    async def run_saga(
        self,
        saga_def: SagaDefinition,
        input_data: Any,
        task_id: str | None = None,
        owner_kind: str = "root",
        owner_id: str = "root",
        conversation_id: str | None = None,
    ) -> Any:
        self.register_saga(saga_def)
        effective_task_id = task_id or f"saga-{input_data.get('id', '')}" if isinstance(input_data, dict) and 'id' in input_data else None
        if not effective_task_id:
            import uuid
            effective_task_id = f"saga-{uuid.uuid4().hex[:12]}"

        existing = await self.storage.get_task(effective_task_id)
        if existing is None:
            initial_record = TaskRecord(
                task_id=effective_task_id,
                task_type=saga_def.name,
                version=saga_def.version,
                status=TaskStatus.RUNNING,
                owner_kind=owner_kind,
                owner_id=owner_id,
                phase=saga_def.initial_step,
                input_json=json.dumps(input_data) if not isinstance(input_data, str) else input_data,
                state_json=json.dumps({"executed_steps": []}),
                policy=SagaPolicy.FAIL_FAST.value,
            )
            await self.storage.insert_task(initial_record)
            if self.stream_hub is not None and conversation_id:
                from pydantic_ai_durable.streaming import StreamEvent

                self.stream_hub.publish(
                    conversation_id,
                    StreamEvent(
                        conversation_id=conversation_id,
                        event_type="saga_started",
                        data={"saga_id": effective_task_id, "name": saga_def.name},
                    ),
                )

        return await self.execute_saga_loop(saga_def, effective_task_id, conversation_id)

    async def execute_saga_loop(
        self,
        saga_def: SagaDefinition,
        task_id: str,
        conversation_id: str | None = None,
    ) -> Any:
        record = await self.storage.get_task(task_id)
        if record is None:
            raise ValueError(f"Task with id {task_id} not found")

        inp = json.loads(record.input_json) if record.input_json else {}
        st = json.loads(record.state_json) if record.state_json else {}
        executed = list(st.get("executed_steps", []))
        current_step = record.phase

        ctx = SagaContext(
            task_id=task_id,
            storage=self.storage,
            input_data=inp,
            state=st.get("state", st),
            current_step=current_step,
            executed_steps=executed,
            coordinator=self,
        )

        while ctx.current_step:
            step_fn = saga_def.steps.get(ctx.current_step)
            if step_fn is None:
                raise ValueError(f"Step '{ctx.current_step}' not defined in saga '{saga_def.name}'")

            try:
                res = step_fn(ctx)
                if inspect.isawaitable(res):
                    res = await res

                if not isinstance(res, SagaResult):
                    res = SagaResult.complete(result=res, state=ctx.state)

                if res.state is not None:
                    ctx.state.update(res.state)

                if res.action == "next":
                    ctx.executed_steps.append(ctx.current_step)
                    ctx.current_step = res.next_step or "completed"
                    st_json = json.dumps({"state": ctx.state, "executed_steps": ctx.executed_steps})
                    await self.storage.update_task_state(
                        task_id=task_id,
                        status=TaskStatus.RUNNING,
                        phase=ctx.current_step,
                        state_json=st_json,
                    )
                    if self.stream_hub is not None and conversation_id:
                        from pydantic_ai_durable.streaming import StreamEvent

                        self.stream_hub.publish(
                            conversation_id,
                            StreamEvent(
                                conversation_id=conversation_id,
                                event_type="saga_step_completed",
                                data={"saga_id": task_id, "step": ctx.executed_steps[-1]},
                            ),
                        )
                    continue

                if res.action == "wait":
                    ctx.executed_steps.append(ctx.current_step)
                    checkpoint_payload = {
                        "waiting_on": res.waiting_on,
                        "next_step": res.next_step,
                        "policy": res.policy.value,
                        "executed_steps": ctx.executed_steps,
                    }
                    st_json = json.dumps({"state": ctx.state, "executed_steps": ctx.executed_steps})
                    await self.storage.update_task_state(
                        task_id=task_id,
                        status=TaskStatus.WAITING,
                        phase="waiting_tasks",
                        state_json=st_json,
                        checkpoint_json=json.dumps(checkpoint_payload),
                    )

                    wait_result = await self._await_children_and_reconcile(
                        parent_task_id=task_id,
                        child_task_ids=res.waiting_on,
                        policy=res.policy,
                    )

                    if not wait_result["success"]:
                        await self.compensate_saga(saga_def, ctx, reason=wait_result["error"])
                        st_json = json.dumps({
                            "state": ctx.state,
                            "executed_steps": ctx.executed_steps,
                            "error": wait_result["error"],
                        })
                        await self.storage.update_task_state(
                            task_id=task_id,
                            status=TaskStatus.FAILED,
                            phase="failed",
                            state_json=st_json,
                        )
                        raise SagaExecutionError(f"Saga failed during child execution: {wait_result['error']}")

                    ctx.current_step = res.next_step or "completed"
                    await self.storage.update_task_state(
                        task_id=task_id,
                        status=TaskStatus.RUNNING,
                        phase=ctx.current_step,
                        state_json=json.dumps({"state": ctx.state, "executed_steps": ctx.executed_steps}),
                    )
                    continue

                if res.action == "complete":
                    ctx.executed_steps.append(ctx.current_step)
                    st_json = json.dumps({
                        "output": res.result,
                        "state": ctx.state,
                        "executed_steps": ctx.executed_steps,
                    })
                    await self.storage.update_task_state(
                        task_id=task_id,
                        status=TaskStatus.COMPLETED,
                        phase="completed",
                        state_json=st_json,
                    )
                    if self.stream_hub is not None and conversation_id:
                        from pydantic_ai_durable.streaming import StreamEvent

                        self.stream_hub.publish(
                            conversation_id,
                            StreamEvent(
                                conversation_id=conversation_id,
                                event_type="saga_completed",
                                data={"saga_id": task_id, "result": res.result},
                            ),
                        )
                    return res.result

                if res.action in ("fail", "abort"):
                    await self.compensate_saga(saga_def, ctx, reason=res.error or "Aborted")
                    terminal_status = TaskStatus.FAILED if res.action == "fail" else TaskStatus.ABORTED
                    terminal_phase = "failed" if res.action == "fail" else "aborted"
                    st_json = json.dumps({
                        "state": ctx.state,
                        "executed_steps": ctx.executed_steps,
                        "error": res.error,
                    })
                    await self.storage.update_task_state(
                        task_id=task_id,
                        status=terminal_status,
                        phase=terminal_phase,
                        state_json=st_json,
                    )
                    if res.action == "fail":
                        raise SagaExecutionError(f"Saga failed: {res.error}")
                    return res.result

            except Exception as e:
                if isinstance(e, SagaExecutionError):
                    raise
                await self.compensate_saga(saga_def, ctx, reason=str(e))
                await self.storage.update_task_state(
                    task_id=task_id,
                    status=TaskStatus.FAILED,
                    phase="failed",
                    state_json=json.dumps({"state": ctx.state, "executed_steps": ctx.executed_steps, "error": str(e)}),
                )
                raise SagaExecutionError(f"Saga execution failed: {e}") from e

        return ctx.state.get("output")

    async def _await_children_and_reconcile(
        self,
        parent_task_id: str,
        child_task_ids: list[str],
        policy: SagaPolicy,
        poll_interval: float = 0.05,
    ) -> dict[str, Any]:
        targets = set(child_task_ids)

        while True:
            outcomes = await self.get_outcomes(parent_task_id, child_task_ids)
            terminal_count = 0
            has_failure = False
            first_error = "Child task failed"

            for tid in targets:
                outcome = outcomes.get(tid)
                if outcome and outcome.is_terminal:
                    terminal_count += 1
                    if not outcome.is_successful:
                        has_failure = True
                        if outcome.error:
                            first_error = outcome.error

            if policy == SagaPolicy.FAIL_FAST and has_failure:
                if self.task_runner is not None:
                    await self.task_runner.abort_children_of(parent_task_id)
                return {"success": False, "error": first_error}

            if terminal_count == len(targets):
                if policy == SagaPolicy.FAIL_FAST and has_failure:
                    return {"success": False, "error": first_error}
                return {"success": True, "error": None}

            await asyncio.sleep(poll_interval)

    async def compensate_saga(
        self, saga_def: SagaDefinition, ctx: SagaContext, reason: str = ""
    ) -> None:
        logger.info("Starting compensation for saga %s (%s)", ctx.task_id, reason)
        steps_to_compensate = list(reversed(ctx.executed_steps))

        for step in steps_to_compensate:
            handler = saga_def.compensations.get(step)
            if handler is not None:
                try:
                    logger.info("Running compensation for step '%s' in saga %s", step, ctx.task_id)
                    res = handler(ctx)
                    if inspect.isawaitable(res):
                        await res
                except Exception as comp_err:
                    logger.error("Error compensating step '%s': %s", step, comp_err)

        if saga_def.global_abort is not None:
            try:
                logger.info("Running global abort compensation for saga %s", ctx.task_id)
                res = saga_def.global_abort(ctx)
                if inspect.isawaitable(res):
                    await res
            except Exception as global_err:
                logger.error("Error in global abort compensation: %s", global_err)
