import asyncio
import json
import logging
import uuid
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ToolCallPart,
)

from pydantic_ai_durable.models import (
    TaskStatus,
    TranscriptEntryKind,
    TranscriptEntryRecord,
)
from pydantic_ai_durable.storage.base import DurableStorage
from pydantic_ai_durable.streaming.hub import StreamEvent, stream_hub
from pydantic_ai_durable.tasks.approvals import ApprovalRequiredError
from pydantic_ai_durable.tasks.replay import durable_execution_context

logger = logging.getLogger("pydantic_ai_durable.engine")

TOutput = TypeVar("TOutput")


@dataclass(frozen=True)
class StepNodeCommit:
    node_name: str
    entry_id: str
    kind: TranscriptEntryKind
    parent_entry_id: str | None
    node_data: Any


@dataclass(frozen=True)
class TurnResult(Generic[TOutput]):
    output: TOutput
    leaf_entry_id: str
    all_messages: list[ModelMessage]
    token_usage: dict[str, int]
    challenge_token: str | None = None
    waiting_approval: bool = False


class DurableStepEngine:
    def __init__(self, storage: DurableStorage) -> None:
        self.storage = storage

    async def _drain_steers(
        self,
        runner: Any,
        conversation_id: str,
        current_parent_id: str | None,
    ) -> str | None:
        from pydantic_ai_durable.steering import global_steering_queue

        steer_msgs = global_steering_queue.pop_all(conversation_id)
        for steer_msg in steer_msgs:
            logger.info(
                "Injected steer message into active turn for conversation id %s",
                conversation_id,
            )
            if hasattr(runner, "enqueue"):
                runner.enqueue(steer_msg)

            steer_id = f"entry-{uuid.uuid4().hex[:16]}"
            steer_entry = TranscriptEntryRecord(
                entry_id=steer_id,
                conversation_id=conversation_id,
                parent_entry_id=current_parent_id,
                kind=TranscriptEntryKind.USER,
                message_json=json.dumps({"prompt": steer_msg, "steer": True}),
            )
            async with self.storage.transaction():
                await self.storage.insert_transcript_entry(steer_entry)
            current_parent_id = steer_id

        return current_parent_id

    def _extract_node_payload(self, node: Any, prompt_text: str) -> tuple[TranscriptEntryKind, str]:
        node_name = type(node).__name__

        if node_name == "UserPromptNode":
            return (
                TranscriptEntryKind.USER,
                json.dumps({"prompt": getattr(node, "user_prompt", prompt_text)}),
            )

        if node_name == "ModelRequestNode":
            req = getattr(node, "request", None)
            if req is not None:
                dumped = ModelMessagesTypeAdapter.dump_json([req]).decode()
                return TranscriptEntryKind.MODEL_REQUEST, dumped
            return TranscriptEntryKind.MODEL_REQUEST, json.dumps({"type": "request"})

        if node_name == "CallToolsNode":
            resp = getattr(node, "model_response", None)
            if resp is not None:
                dumped = ModelMessagesTypeAdapter.dump_json([resp]).decode()
                parts = getattr(resp, "parts", [])
                has_tool_call = any(isinstance(p, ToolCallPart) for p in parts)
                kind = (
                    TranscriptEntryKind.TOOL_CALL
                    if has_tool_call
                    else TranscriptEntryKind.MODEL_RESPONSE
                )
                return kind, dumped
            return TranscriptEntryKind.MODEL_RESPONSE, json.dumps({"type": "response"})

        if node_name == "End":
            data = getattr(node, "data", None)
            output = getattr(data, "output", "")
            if hasattr(output, "model_dump_json"):
                payload = output.model_dump_json()
            elif isinstance(output, (dict, list, int, float, bool)):
                payload = json.dumps(output)
            else:
                payload = json.dumps({"output": str(output)})
            return TranscriptEntryKind.SYSTEM, payload

        return TranscriptEntryKind.SYSTEM, json.dumps({"node": node_name})

    async def iter_turn(
        self,
        agent: Agent[Any, TOutput],
        user_prompt: str,
        conversation_id: str,
        task_id: str,
        parent_entry_id: str | None = None,
        message_history: Sequence[ModelMessage] | None = None,
        deps: Any = None,
    ) -> AsyncIterator[StepNodeCommit]:
        if user_prompt is None or not user_prompt.strip():
            raise ValueError("Prompt cannot be empty")

        current_parent_id = parent_entry_id
        cumulative_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

        await self.storage.update_task_state(
            task_id=task_id,
            status=TaskStatus.RUNNING,
            phase="iterating_nodes",
            state_json=json.dumps({"phase": "starting", "usage": cumulative_usage}),
            checkpoint_json=current_parent_id,
        )

        try:
            async with durable_execution_context(
                self.storage,
                task_id=task_id,
                conversation_id=conversation_id,
            ):
                async with agent.iter(
                    user_prompt,
                    message_history=message_history,
                    deps=deps,
                ) as runner:
                    async for node in runner:
                        node_name = type(node).__name__
                        entry_id = f"entry-{uuid.uuid4().hex[:16]}"
                        kind, payload = self._extract_node_payload(node, user_prompt)

                        if hasattr(node, "model_response") and hasattr(
                            node.model_response, "usage"
                        ):
                            u = node.model_response.usage
                            if u is not None:
                                in_tok = getattr(u, "input_tokens", 0) or 0
                                out_tok = getattr(u, "output_tokens", 0) or 0
                                cumulative_usage["input_tokens"] += in_tok
                                cumulative_usage["output_tokens"] += out_tok
                                cumulative_usage["total_tokens"] += in_tok + out_tok

                        entry = TranscriptEntryRecord(
                            entry_id=entry_id,
                            conversation_id=conversation_id,
                            parent_entry_id=current_parent_id,
                            kind=kind,
                            message_json=payload,
                        )

                        async with self.storage.transaction():
                            await self.storage.insert_transcript_entry(entry)
                            state = {
                                "last_node": node_name,
                                "entry_id": entry_id,
                                "usage": cumulative_usage,
                            }
                            await self.storage.update_task_state(
                                task_id=task_id,
                                status=TaskStatus.RUNNING,
                                phase=node_name,
                                state_json=json.dumps(state),
                                checkpoint_json=entry_id,
                            )

                        logger.info("Node %s committed with entry id %s", node_name, entry_id)
                        stream_hub.publish(
                            conversation_id,
                            StreamEvent(
                                conversation_id=conversation_id,
                                event_type="node_commit",
                                data={
                                    "node_name": node_name,
                                    "entry_id": entry_id,
                                    "kind": kind.value if hasattr(kind, "value") else str(kind),
                                    "payload": payload,
                                },
                            ),
                        )
                        current_parent_id = entry_id
                        current_parent_id = await self._drain_steers(
                            runner, conversation_id, current_parent_id
                        )

                        yield StepNodeCommit(
                            node_name=node_name,
                            entry_id=entry_id,
                            kind=kind,
                            parent_entry_id=entry.parent_entry_id,
                            node_data=node,
                        )

                    current_task = await self.storage.get_task(task_id)
                    if current_task is not None and current_task.status == TaskStatus.WAITING:
                        pass
                    else:
                        final_output = runner.result.output
                        await self.storage.update_task_state(
                            task_id=task_id,
                            status=TaskStatus.COMPLETED,
                            phase="completed",
                            state_json=json.dumps(
                                {
                                    "output": str(final_output)
                                    if not hasattr(final_output, "model_dump_json")
                                    else final_output.model_dump_json(),
                                    "usage": cumulative_usage,
                                }
                            ),
                            checkpoint_json=current_parent_id,
                        )

        except ApprovalRequiredError:
            return
        except Exception as exc:
            await self.storage.update_task_state(
                task_id=task_id,
                status=TaskStatus.FAILED,
                phase="failed",
                state_json=json.dumps(
                    {
                        "error": str(exc),
                        "error_type": type(exc).__name__,
                        "usage": cumulative_usage,
                    }
                ),
                checkpoint_json=current_parent_id,
            )
            raise

    async def run_turn(
        self,
        agent: Agent[Any, TOutput],
        user_prompt: str,
        conversation_id: str,
        task_id: str,
        parent_entry_id: str | None = None,
        message_history: Sequence[ModelMessage] | None = None,
        deps: Any = None,
        max_steps: int | None = None,
        on_node: Any = None,
    ) -> TurnResult[TOutput]:
        if user_prompt is None or not user_prompt.strip():
            raise ValueError("Prompt cannot be empty")

        current_parent_id = parent_entry_id
        cumulative_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        last_node_name = "init"
        extracted_output: Any = None
        stopped_early = False

        await self.storage.update_task_state(
            task_id=task_id,
            status=TaskStatus.RUNNING,
            phase="iterating_nodes",
            state_json=json.dumps({"phase": "starting", "usage": cumulative_usage}),
            checkpoint_json=current_parent_id,
        )

        try:
            async with durable_execution_context(
                self.storage,
                task_id=task_id,
                conversation_id=conversation_id,
            ):
                async with agent.iter(
                    user_prompt,
                    message_history=message_history,
                    deps=deps,
                ) as runner:
                    step_count = 0
                    async for node in runner:
                        step_count += 1
                        node_name = type(node).__name__
                        last_node_name = node_name
                        entry_id = f"entry-{uuid.uuid4().hex[:16]}"
                        kind, payload = self._extract_node_payload(node, user_prompt)

                        if hasattr(node, "model_response") and hasattr(
                            node.model_response, "usage"
                        ):
                            u = node.model_response.usage
                            if u is not None:
                                in_tok = getattr(u, "input_tokens", 0) or 0
                                out_tok = getattr(u, "output_tokens", 0) or 0
                                cumulative_usage["input_tokens"] += in_tok
                                cumulative_usage["output_tokens"] += out_tok
                                cumulative_usage["total_tokens"] += in_tok + out_tok

                        entry = TranscriptEntryRecord(
                            entry_id=entry_id,
                            conversation_id=conversation_id,
                            parent_entry_id=current_parent_id,
                            kind=kind,
                            message_json=payload,
                        )

                        async with self.storage.transaction():
                            await self.storage.insert_transcript_entry(entry)
                            state = {
                                "last_node": node_name,
                                "entry_id": entry_id,
                                "usage": cumulative_usage,
                            }
                            await self.storage.update_task_state(
                                task_id=task_id,
                                status=TaskStatus.RUNNING,
                                phase=node_name,
                                state_json=json.dumps(state),
                                checkpoint_json=entry_id,
                            )

                        logger.info("Node %s committed with entry id %s", node_name, entry_id)
                        stream_hub.publish(
                            conversation_id,
                            StreamEvent(
                                conversation_id=conversation_id,
                                event_type="node_commit",
                                data={
                                    "node_name": node_name,
                                    "entry_id": entry_id,
                                    "kind": kind.value if hasattr(kind, "value") else str(kind),
                                    "payload": payload,
                                },
                            ),
                        )
                        current_parent_id = entry_id
                        current_parent_id = await self._drain_steers(
                            runner, conversation_id, current_parent_id
                        )

                        commit = StepNodeCommit(
                            node_name=node_name,
                            entry_id=entry_id,
                            kind=kind,
                            parent_entry_id=entry.parent_entry_id,
                            node_data=node,
                        )

                        if node_name == "End" and hasattr(node, "data"):
                            extracted_output = node.data.output

                        if on_node is not None:
                            res = on_node(commit)
                            if asyncio.iscoroutine(res):
                                res = await res
                            if res is False:
                                stopped_early = True
                                break

                        if max_steps is not None and step_count >= max_steps:
                            stopped_early = True
                            break

                    current_task = await self.storage.get_task(task_id)
                    is_waiting = (
                        current_task is not None and current_task.status == TaskStatus.WAITING
                    )
                    challenge_tok = None
                    if is_waiting and current_task is not None:
                        challenge_tok = current_task.checkpoint_json

                    if stopped_early:
                        await self.storage.update_task_state(
                            task_id=task_id,
                            status=TaskStatus.RUNNING,
                            phase="interrupted",
                            state_json=json.dumps(
                                {
                                    "interrupted": True,
                                    "last_node": last_node_name,
                                    "usage": cumulative_usage,
                                }
                            ),
                            checkpoint_json=current_parent_id,
                        )
                    elif not is_waiting:
                        if (
                            extracted_output is None
                            and hasattr(runner, "result")
                            and runner.result is not None
                        ):
                            extracted_output = runner.result.output

                        out_str = (
                            extracted_output.model_dump_json()
                            if hasattr(extracted_output, "model_dump_json")
                            else str(extracted_output)
                        )
                        await self.storage.update_task_state(
                            task_id=task_id,
                            status=TaskStatus.COMPLETED,
                            phase="completed",
                            state_json=json.dumps(
                                {
                                    "output": out_str,
                                    "usage": cumulative_usage,
                                }
                            ),
                            checkpoint_json=current_parent_id,
                        )

        except ApprovalRequiredError as exc:
            leaf_id = current_parent_id or ""
            ancestor_entries = await self.storage.get_ancestor_entries(leaf_id)
            messages: list[ModelMessage] = []
            for e in ancestor_entries:
                if e.kind in (
                    TranscriptEntryKind.MODEL_REQUEST,
                    TranscriptEntryKind.MODEL_RESPONSE,
                    TranscriptEntryKind.TOOL_CALL,
                ):
                    try:
                        parsed = ModelMessagesTypeAdapter.validate_json(e.message_json)
                        messages.extend(parsed)
                    except Exception:
                        pass

            return TurnResult(
                output=None,  # type: ignore[arg-type]
                leaf_entry_id=leaf_id,
                all_messages=messages,
                token_usage=cumulative_usage,
                challenge_token=exc.challenge_token,
                waiting_approval=True,
            )
        except Exception as exc:
            await self.storage.update_task_state(
                task_id=task_id,
                status=TaskStatus.FAILED,
                phase="failed",
                state_json=json.dumps(
                    {
                        "error": str(exc),
                        "error_type": type(exc).__name__,
                        "usage": cumulative_usage,
                    }
                ),
                checkpoint_json=current_parent_id,
            )
            raise

        leaf_id = current_parent_id or ""
        ancestor_entries = await self.storage.get_ancestor_entries(leaf_id)
        messages: list[ModelMessage] = []
        for e in ancestor_entries:
            if e.kind in (
                TranscriptEntryKind.MODEL_REQUEST,
                TranscriptEntryKind.MODEL_RESPONSE,
                TranscriptEntryKind.TOOL_CALL,
            ):
                try:
                    parsed = ModelMessagesTypeAdapter.validate_json(e.message_json)
                    messages.extend(parsed)
                except Exception:
                    pass

        return TurnResult(
            output=extracted_output,
            leaf_entry_id=leaf_id,
            all_messages=messages,
            token_usage=cumulative_usage,
            challenge_token=challenge_tok,
            waiting_approval=is_waiting,
        )
