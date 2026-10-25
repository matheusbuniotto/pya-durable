import asyncio
import inspect
import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from pya_durable.models import (
    TaskRecord,
    TaskStatus,
)
from pya_durable.storage import DurableStorage

logger = logging.getLogger("pya_durable.task_tree")


class CycleDetectedError(ValueError):
    """Raised when an ownership cycle is detected in task parent pointers."""


async def check_cycle(
    storage: DurableStorage,
    task_id: str,
    parent_task_id: str | None,
) -> None:
    if not parent_task_id or parent_task_id == "root":
        return

    if parent_task_id == task_id:
        raise CycleDetectedError(
            f"Cycle detected: task '{task_id}' cannot be its own parent"
        )

    visited: set[str] = {task_id}
    curr_id: str | None = parent_task_id

    while curr_id:
        if curr_id in visited:
            raise CycleDetectedError(
                f"Cycle detected: ancestor chain contains cycle at task '{curr_id}'"
            )
        visited.add(curr_id)

        task = await storage.get_task(curr_id)
        if task is None:
            break
        if task.owner_kind == "task" and task.owner_id:
            curr_id = task.owner_id
        else:
            break


@dataclass
class TaskPhaseResult:
    status: TaskStatus = TaskStatus.RUNNING
    next_phase: str | None = None
    state: dict[str, Any] | None = None
    result: Any = None
    checkpoint_data: Any = None


class TaskContext:
    def __init__(
        self,
        task_id: str,
        storage: DurableStorage,
        input_data: Any,
        state: dict[str, Any] | None = None,
        phase: str = "start",
        checkpoint_json: str | None = None,
        policy: str = "failFast",
        runner: Any = None,
    ) -> None:
        self.task_id = task_id
        self.storage = storage
        self.input_data = input_data
        self.state = state if state is not None else {}
        self.phase = phase
        self.checkpoint_json = checkpoint_json
        self.policy = policy
        self.runner = runner

    async def checkpoint(
        self,
        phase: str,
        state: dict[str, Any] | Any,
        checkpoint_data: Any = None,
    ) -> None:
        self.phase = phase
        if isinstance(state, dict):
            self.state = state
        state_json = json.dumps(self.state) if not isinstance(self.state, str) else self.state

        ckpt_json = (
            json.dumps(checkpoint_data)
            if checkpoint_data is not None and not isinstance(checkpoint_data, str)
            else checkpoint_data
        )
        self.checkpoint_json = ckpt_json

        await self.storage.update_task_state(
            task_id=self.task_id,
            status=TaskStatus.RUNNING,
            phase=phase,
            state_json=state_json,
            checkpoint_json=ckpt_json,
        )

    async def create_child_task(
        self,
        task_type: str,
        input_data: Any,
        policy: str = "failFast",
        initial_phase: str = "start",
        state: Any = None,
    ) -> TaskRecord:
        child_task_id = f"task-{uuid.uuid4().hex[:12]}"
        await check_cycle(self.storage, child_task_id, self.task_id)

        in_json = json.dumps(input_data) if not isinstance(input_data, str) else input_data
        st_json = json.dumps(state or {}) if not isinstance(state, str) else (state or "{}")

        record = TaskRecord(
            task_id=child_task_id,
            task_type=task_type,
            version=1,
            status=TaskStatus.PENDING,
            owner_kind="task",
            owner_id=self.task_id,
            phase=initial_phase,
            input_json=in_json,
            state_json=st_json,
            policy=policy,
        )
        await self.storage.insert_task(record)
        return record

    async def abort_children(self) -> None:
        if self.runner is not None:
            await self.runner.abort_children_of(self.task_id)
        else:
            await TaskRunner.abort_children_for(self.storage, self.task_id)


@dataclass
class TaskDefinition:
    name: str
    version: int = 1
    initial_phase: str = "start"
    policy: str = "failFast"
    phases: dict[str, Callable[[TaskContext], Awaitable[Any]]] = field(default_factory=dict)
    abort_handler: Callable[[TaskContext], Awaitable[None]] | None = None


def define_task(
    target: Any = None,
    *,
    name: str | None = None,
    version: int = 1,
    initial_phase: str = "start",
    policy: str = "failFast",
    phases: dict[str, Callable[[TaskContext], Awaitable[Any]]] | None = None,
    abort_handler: Callable[[TaskContext], Awaitable[None]] | None = None,
) -> Any:
    if target is None:
        if name is None:
            def decorator(cls_or_fn: Any) -> TaskDefinition:
                return _build_task_def(
                    cls_or_fn,
                    name=name,
                    version=version,
                    initial_phase=initial_phase,
                    policy=policy,
                    phases=phases,
                    abort_handler=abort_handler,
                )
            return decorator

        return TaskDefinition(
            name=name,
            version=version,
            initial_phase=initial_phase,
            policy=policy,
            phases=phases or {},
            abort_handler=abort_handler,
        )

    return _build_task_def(
        target,
        name=name,
        version=version,
        initial_phase=initial_phase,
        policy=policy,
        phases=phases,
        abort_handler=abort_handler,
    )


def _build_task_def(
    target: Any,
    name: str | None,
    version: int,
    initial_phase: str,
    policy: str,
    phases: dict[str, Callable[[TaskContext], Awaitable[Any]]] | None,
    abort_handler: Callable[[TaskContext], Awaitable[None]] | None,
) -> TaskDefinition:
    task_name = name or (getattr(target, "__name__", None) or str(target))
    extracted_phases: dict[str, Callable[[TaskContext], Awaitable[Any]]] = dict(phases or {})
    extracted_abort = abort_handler

    if inspect.isclass(target):
        instance = target()
        for attr_name in dir(instance):
            if attr_name.startswith("_"):
                continue
            attr = getattr(instance, attr_name)
            if callable(attr):
                if attr_name in ("abort_handler", "on_abort", "compensation", "abort"):
                    extracted_abort = attr
                else:
                    extracted_phases[attr_name] = attr
    elif callable(target) and not extracted_phases:
        extracted_phases[initial_phase] = target

    return TaskDefinition(
        name=task_name,
        version=version,
        initial_phase=initial_phase,
        policy=policy,
        phases=extracted_phases,
        abort_handler=extracted_abort,
    )


class TaskRunner:
    def __init__(self, storage: DurableStorage) -> None:
        self.storage = storage
        self.registry: dict[str, TaskDefinition] = {}
        self.active_tasks: dict[str, asyncio.Task[Any]] = {}

    def register_task(self, task_def: TaskDefinition) -> None:
        self.registry[task_def.name] = task_def

    async def get_child_tasks(self, parent_task_id: str) -> list[TaskRecord]:
        if hasattr(self.storage, "get_connection"):
            conn = await self.storage.get_connection()
            async with conn.execute(
                """
                SELECT task_id, task_type, version, status, owner_kind, owner_id,
                       phase, input_json, state_json, checkpoint_json, policy, created_at, updated_at
                FROM tasks
                WHERE owner_kind = 'task' AND owner_id = ?
                ORDER BY created_at ASC
                """,
                (parent_task_id,),
            ) as cursor:
                rows = await cursor.fetchall()
                return [
                    TaskRecord(
                        task_id=row[0],
                        task_type=row[1],
                        version=row[2],
                        status=TaskStatus(row[3]),
                        owner_kind=row[4],
                        owner_id=row[5],
                        phase=row[6],
                        input_json=row[7],
                        state_json=row[8],
                        checkpoint_json=row[9],
                        policy=row[10],
                        created_at=row[11],
                        updated_at=row[12],
                    )
                    for row in rows
                ]
        all_tasks: list[TaskRecord] = []
        for st in TaskStatus:
            all_tasks.extend(await self.storage.get_tasks_by_status(st))
        matching = [t for t in all_tasks if t.owner_kind == "task" and t.owner_id == parent_task_id]
        matching.sort(key=lambda t: t.created_at)
        return matching

    async def abort_children_of(self, parent_task_id: str) -> None:
        children = await self.get_child_tasks(parent_task_id)
        for child in children:
            if child.status not in (TaskStatus.COMPLETED, TaskStatus.ABORTED, TaskStatus.FAILED):
                await self.abort_task(child.task_id, reason="parent cancellation")

    @classmethod
    async def abort_children_for(cls, storage: DurableStorage, parent_task_id: str) -> None:
        runner = cls(storage)
        await runner.abort_children_of(parent_task_id)

    async def abort_siblings(self, failed_task_id: str, parent_task_id: str) -> None:
        children = await self.get_child_tasks(parent_task_id)
        for sibling in children:
            if sibling.task_id != failed_task_id and sibling.status not in (
                TaskStatus.COMPLETED,
                TaskStatus.ABORTED,
                TaskStatus.FAILED,
            ):
                await self.abort_task(sibling.task_id, reason="parent cancellation")

    async def abort_task(self, task_id: str, reason: str = "parent cancellation") -> None:
        async_task = self.active_tasks.get(task_id)
        if async_task and not async_task.done():
            async_task.cancel()

        conn = await self.storage.get_connection()
        query = """
        WITH RECURSIVE descendants(task_id, owner_id, depth) AS (
            SELECT task_id, owner_id, 0
            FROM tasks
            WHERE owner_kind = 'task' AND owner_id = ?
            UNION ALL
            SELECT t.task_id, t.owner_id, d.depth + 1
            FROM tasks t
            JOIN descendants d ON t.owner_kind = 'task' AND t.owner_id = d.task_id
        )
        SELECT task_id FROM descendants ORDER BY depth DESC;
        """
        async with conn.execute(query, (task_id,)) as cursor:
            rows = await cursor.fetchall()
            descendant_ids = [r[0] for r in rows]

        for desc_id in descendant_ids:
            desc_record = await self.storage.get_task(desc_id)
            if desc_record is None or desc_record.status in (
                TaskStatus.COMPLETED,
                TaskStatus.ABORTED,
                TaskStatus.FAILED,
            ):
                continue

            desc_def = self.registry.get(desc_record.task_type)
            if desc_def and desc_def.abort_handler:
                ctx_desc = self._build_context_from_record(desc_record)
                try:
                    h_res = desc_def.abort_handler(ctx_desc)
                    if inspect.isawaitable(h_res):
                        await h_res
                except Exception as e:
                    logger.warning("Error running abort_handler for child task %s: %s", desc_id, e)

            await self.storage.update_task_state(
                task_id=desc_id,
                status=TaskStatus.ABORTED,
                phase="aborted",
                state_json=desc_record.state_json,
                checkpoint_json=desc_record.checkpoint_json,
            )
            logger.info("Aborted child task %s due to parent cancellation", desc_id)

        task_record = await self.storage.get_task(task_id)
        if task_record is None or task_record.status == TaskStatus.ABORTED:
            return

        task_def = self.registry.get(task_record.task_type)
        if task_def and task_def.abort_handler:
            ctx = self._build_context_from_record(task_record)
            try:
                h_res = task_def.abort_handler(ctx)
                if inspect.isawaitable(h_res):
                    await h_res
            except Exception as e:
                logger.warning("Error running abort_handler for task %s: %s", task_id, e)

        await self.storage.update_task_state(
            task_id=task_id,
            status=TaskStatus.ABORTED,
            phase="aborted",
            state_json=task_record.state_json,
            checkpoint_json=task_record.checkpoint_json,
        )

        if task_record.owner_kind == "task":
            logger.info("Aborted child task %s due to parent cancellation", task_id)

    def _build_context_from_record(self, record: TaskRecord) -> TaskContext:
        inp = {}
        try:
            inp = json.loads(record.input_json) if record.input_json else {}
        except Exception:
            pass
        st = {}
        try:
            st = json.loads(record.state_json) if record.state_json else {}
        except Exception:
            pass
        return TaskContext(
            task_id=record.task_id,
            storage=self.storage,
            input_data=inp,
            state=st,
            phase=record.phase,
            checkpoint_json=record.checkpoint_json,
            policy=record.policy,
            runner=self,
        )

    async def run_task(
        self,
        task_def: TaskDefinition,
        input_data: Any,
        task_id: str | None = None,
        owner_kind: str = "root",
        owner_id: str = "root",
        background: bool = False,
        policy: str | None = None,
    ) -> Any:
        self.register_task(task_def)
        effective_task_id = task_id or f"task-{uuid.uuid4().hex[:12]}"
        effective_policy = policy or task_def.policy

        if owner_kind == "task":
            await check_cycle(self.storage, effective_task_id, owner_id)

        in_json = json.dumps(input_data) if not isinstance(input_data, str) else input_data

        existing = await self.storage.get_task(effective_task_id)
        if existing is None:
            record = TaskRecord(
                task_id=effective_task_id,
                task_type=task_def.name,
                version=task_def.version,
                status=TaskStatus.RUNNING,
                owner_kind=owner_kind,
                owner_id=owner_id,
                phase=task_def.initial_phase,
                input_json=in_json,
                state_json=json.dumps({}),
                policy=effective_policy,
            )
            await self.storage.insert_task(record)
        else:
            await self.storage.update_task_state(
                task_id=effective_task_id,
                status=TaskStatus.RUNNING,
                phase=task_def.initial_phase,
                state_json=existing.state_json or json.dumps({}),
            )

        if background:
            async_task = asyncio.create_task(
                self._execute_task_loop(
                    task_def=task_def,
                    task_id=effective_task_id,
                    input_data=input_data,
                    initial_phase=task_def.initial_phase,
                    initial_state={},
                    policy=effective_policy,
                    owner_kind=owner_kind,
                    owner_id=owner_id,
                )
            )
            self.active_tasks[effective_task_id] = async_task
            return await self.storage.get_task(effective_task_id)

        return await self._execute_task_loop(
            task_def=task_def,
            task_id=effective_task_id,
            input_data=input_data,
            initial_phase=task_def.initial_phase,
            initial_state={},
            policy=effective_policy,
            owner_kind=owner_kind,
            owner_id=owner_id,
        )

    async def _execute_task_loop(
        self,
        task_def: TaskDefinition,
        task_id: str,
        input_data: Any,
        initial_phase: str,
        initial_state: dict[str, Any],
        policy: str,
        owner_kind: str,
        owner_id: str,
    ) -> Any:
        ctx = TaskContext(
            task_id=task_id,
            storage=self.storage,
            input_data=input_data,
            state=dict(initial_state),
            phase=initial_phase,
            policy=policy,
            runner=self,
        )

        try:
            while ctx.phase:
                phase_fn = task_def.phases.get(ctx.phase)
                if phase_fn is None:
                    raise ValueError(f"Phase '{ctx.phase}' not defined in task '{task_def.name}'")

                res = phase_fn(ctx)
                if inspect.isawaitable(res):
                    res = await res

                if isinstance(res, TaskPhaseResult):
                    if res.state is not None:
                        ctx.state.update(res.state)

                    if res.checkpoint_data is not None:
                        ctx.checkpoint_json = (
                            json.dumps(res.checkpoint_data)
                            if not isinstance(res.checkpoint_data, str)
                            else res.checkpoint_data
                        )

                    if res.status == TaskStatus.COMPLETED:
                        out_val = res.result
                        st_json = json.dumps({"output": out_val, "state": ctx.state})
                        await self.storage.update_task_state(
                            task_id=task_id,
                            status=TaskStatus.COMPLETED,
                            phase="completed",
                            state_json=st_json,
                            checkpoint_json=ctx.checkpoint_json,
                        )
                        return out_val

                    if res.status == TaskStatus.WAITING:
                        await self.storage.update_task_state(
                            task_id=task_id,
                            status=TaskStatus.WAITING,
                            phase=ctx.phase,
                            state_json=json.dumps(ctx.state),
                            checkpoint_json=ctx.checkpoint_json,
                        )
                        return res.result

                    if res.status == TaskStatus.RUNNING:
                        if res.next_phase:
                            ctx.phase = res.next_phase
                            await self.storage.update_task_state(
                                task_id=task_id,
                                status=TaskStatus.RUNNING,
                                phase=ctx.phase,
                                state_json=json.dumps(ctx.state),
                                checkpoint_json=ctx.checkpoint_json,
                            )
                            continue
                        else:
                            out_val = res.result
                            st_json = json.dumps({"output": out_val, "state": ctx.state})
                            await self.storage.update_task_state(
                                task_id=task_id,
                                status=TaskStatus.COMPLETED,
                                phase="completed",
                                state_json=st_json,
                                checkpoint_json=ctx.checkpoint_json,
                            )
                            return out_val

                    if res.status == TaskStatus.ABORTED:
                        await self.abort_task(task_id, reason="phase requested abort")
                        return res.result
                else:
                    out_val = res
                    st_json = json.dumps({"output": out_val, "state": ctx.state})
                    await self.storage.update_task_state(
                        task_id=task_id,
                        status=TaskStatus.COMPLETED,
                        phase="completed",
                        state_json=st_json,
                        checkpoint_json=ctx.checkpoint_json,
                    )
                    return out_val

        except asyncio.CancelledError:
            await self.abort_task(task_id, reason="asyncio task cancelled")
            raise
        except Exception as exc:
            await self.storage.update_task_state(
                task_id=task_id,
                status=TaskStatus.FAILED,
                phase="failed",
                state_json=json.dumps({"error": str(exc), "state": ctx.state}),
                checkpoint_json=ctx.checkpoint_json,
            )
            if policy == "failFast":
                if owner_kind == "task":
                    await self.abort_siblings(task_id, owner_id)
                await self.abort_children_of(task_id)
            raise

    async def resume_task(self, task_id: str, task_def: TaskDefinition) -> Any:
        self.register_task(task_def)
        task_record = await self.storage.get_task(task_id)
        if task_record is None:
            raise ValueError(f"Task '{task_id}' not found")

        if task_record.status == TaskStatus.COMPLETED:
            try:
                st = json.loads(task_record.state_json)
                return st.get("output")
            except Exception:
                return None

        if task_record.status in (TaskStatus.ABORTED, TaskStatus.FAILED):
            return None

        inp_data = {}
        try:
            inp_data = json.loads(task_record.input_json) if task_record.input_json else {}
        except Exception:
            pass

        st_data = {}
        try:
            st_data = json.loads(task_record.state_json) if task_record.state_json else {}
        except Exception:
            pass

        resume_phase = task_record.phase
        if resume_phase not in task_def.phases:
            resume_phase = task_def.initial_phase

        await self.storage.update_task_state(
            task_id=task_id,
            status=TaskStatus.RUNNING,
            phase=resume_phase,
            state_json=task_record.state_json,
            checkpoint_json=task_record.checkpoint_json,
        )

        return await self._execute_task_loop(
            task_def=task_def,
            task_id=task_id,
            input_data=inp_data,
            initial_phase=resume_phase,
            initial_state=st_data,
            policy=task_record.policy,
            owner_kind=task_record.owner_kind,
            owner_id=task_record.owner_id,
        )
