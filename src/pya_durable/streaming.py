from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import inspect
from typing import Any, AsyncIterator, Awaitable, Callable, Generic, TypeVar

from pya_durable.models import utc_now_iso

TDoc = TypeVar("TDoc")


@dataclass(frozen=True)
class StreamEvent:
    conversation_id: str
    event_type: str
    data: dict[str, Any]
    timestamp: str = field(default_factory=utc_now_iso)


@dataclass(frozen=True)
class AgentEvent:
    conversation_id: str
    event_type: str
    timestamp: str = field(default_factory=utc_now_iso)


@dataclass(frozen=True)
class TurnStartedEvent(AgentEvent):
    prompt: str = ""
    request_id: str | None = None
    event_type: str = "turn_started"


@dataclass(frozen=True)
class ModelRequestStartedEvent(AgentEvent):
    messages: list[dict[str, Any]] = field(default_factory=list)
    event_type: str = "model_request_started"


@dataclass(frozen=True)
class ModelDeltaEvent(AgentEvent):
    delta_text: str = ""
    event_type: str = "model_delta"


@dataclass(frozen=True)
class ToolCallStartedEvent(AgentEvent):
    tool_call_id: str = ""
    tool_name: str = ""
    args: dict[str, Any] = field(default_factory=dict)
    event_type: str = "tool_call_started"


@dataclass(frozen=True)
class ToolCallCompletedEvent(AgentEvent):
    tool_call_id: str = ""
    tool_name: str = ""
    result: Any = None
    is_error: bool = False
    event_type: str = "tool_call_completed"


@dataclass(frozen=True)
class DocUpdatedEvent(AgentEvent):
    kind: str = ""
    doc_id: str = "default"
    version: int = 1
    event_type: str = "doc_updated"


@dataclass(frozen=True)
class SteerInjectedEvent(AgentEvent):
    steer_text: str = ""
    event_type: str = "steer_injected"


@dataclass(frozen=True)
class TurnCompletedEvent(AgentEvent):
    output: Any = None
    token_usage: dict[str, int] = field(default_factory=dict)
    event_type: str = "turn_completed"


@dataclass(frozen=True)
class ErrorEvent(AgentEvent):
    error_message: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    event_type: str = "error"


@dataclass(frozen=True)
class ClientPresenceEvent(AgentEvent):
    client_id: str = ""
    action: str = "joined"  # "joined" | "left"
    event_type: str = "client_presence"


@dataclass(frozen=True)
class ApprovalRequestedEvent(AgentEvent):
    challenge_token: str = ""
    task_id: str = ""
    tool_name: str = ""
    args: dict[str, Any] = field(default_factory=dict)
    prompt_message: str = ""
    expires_at: str | None = None
    event_type: str = "approval_requested"


@dataclass(frozen=True)
class ApprovalResolvedEvent(AgentEvent):
    challenge_token: str = ""
    task_id: str = ""
    status: str = "approved"  # "approved" | "rejected" | "expired"
    approver: str = "operator"
    reason: str = ""
    override_args: dict[str, Any] | None = None
    resolved_at: str = ""
    event_type: str = "approval_resolved"


@dataclass(frozen=True)
class SagaStartedEvent(AgentEvent):
    saga_id: str = ""
    name: str = ""
    event_type: str = "saga_started"


@dataclass(frozen=True)
class SagaStepCompletedEvent(AgentEvent):
    saga_id: str = ""
    step: str = ""
    event_type: str = "saga_step_completed"


@dataclass(frozen=True)
class SagaCompensatingEvent(AgentEvent):
    saga_id: str = ""
    step: str = ""
    reason: str = ""
    event_type: str = "saga_compensating"


@dataclass(frozen=True)
class SagaCompletedEvent(AgentEvent):
    saga_id: str = ""
    result: Any = None
    event_type: str = "saga_completed"


def translate_stream_event(event: StreamEvent) -> AgentEvent:
    data = event.data or {}
    et = event.event_type

    if et == "turn_started":
        return TurnStartedEvent(
            conversation_id=event.conversation_id,
            prompt=str(data.get("prompt", "")),
            request_id=data.get("request_id"),
            timestamp=event.timestamp,
        )
    if et == "model_delta":
        return ModelDeltaEvent(
            conversation_id=event.conversation_id,
            delta_text=str(data.get("delta_text", "")),
            timestamp=event.timestamp,
        )
    if et == "tool_started" or et == "tool_call_started":
        return ToolCallStartedEvent(
            conversation_id=event.conversation_id,
            tool_call_id=str(data.get("tool_call_id", data.get("invocation_id", ""))),
            tool_name=str(data.get("tool_name", "")),
            args=data.get("args", {}),
            timestamp=event.timestamp,
        )
    if et == "tool_completed" or et == "tool_call_completed":
        return ToolCallCompletedEvent(
            conversation_id=event.conversation_id,
            tool_call_id=str(data.get("tool_call_id", data.get("invocation_id", ""))),
            tool_name=str(data.get("tool_name", "")),
            result=data.get("result"),
            is_error=bool(data.get("is_error", False)),
            timestamp=event.timestamp,
        )
    if et == "doc_updated":
        return DocUpdatedEvent(
            conversation_id=event.conversation_id,
            kind=str(data.get("kind", data.get("doc_type", ""))),
            doc_id=str(data.get("doc_id", "default")),
            version=int(data.get("version", 1)),
            timestamp=event.timestamp,
        )
    if et == "steer":
        return SteerInjectedEvent(
            conversation_id=event.conversation_id,
            steer_text=str(data.get("steer_text", data.get("prompt", ""))),
            timestamp=event.timestamp,
        )
    if et == "turn_completed":
        return TurnCompletedEvent(
            conversation_id=event.conversation_id,
            output=data.get("output"),
            token_usage=data.get("token_usage", {}),
            timestamp=event.timestamp,
        )
    if et == "error":
        return ErrorEvent(
            conversation_id=event.conversation_id,
            error_message=str(data.get("error_message", data.get("message", "Unknown error"))),
            details=data.get("details", {}),
            timestamp=event.timestamp,
        )
    if et == "presence" or et == "client_presence":
        return ClientPresenceEvent(
            conversation_id=event.conversation_id,
            client_id=str(data.get("client_id", "")),
            action=str(data.get("action", "joined")),
            timestamp=event.timestamp,
        )
    if et == "node":
        node_name = str(data.get("node", ""))
        if node_name == "UserPromptNode":
            return TurnStartedEvent(
                conversation_id=event.conversation_id,
                prompt=str(data.get("prompt", "")),
                timestamp=event.timestamp,
            )
        if node_name == "ModelRequestNode":
            return ModelRequestStartedEvent(
                conversation_id=event.conversation_id,
                messages=data.get("messages", []),
                timestamp=event.timestamp,
            )
        if node_name == "CallToolsNode":
            return ToolCallStartedEvent(
                conversation_id=event.conversation_id,
                tool_call_id=str(data.get("tool_call_id", "")),
                tool_name=str(data.get("tool_name", "")),
                args=data.get("args", {}),
                timestamp=event.timestamp,
            )
        if node_name == "End":
            return TurnCompletedEvent(
                conversation_id=event.conversation_id,
                output=data.get("output"),
                token_usage=data.get("token_usage", {}),
                timestamp=event.timestamp,
            )

    if et == "approval_requested":
        return ApprovalRequestedEvent(
            conversation_id=event.conversation_id,
            challenge_token=str(data.get("challenge_token", "")),
            task_id=str(data.get("task_id", "")),
            tool_name=str(data.get("tool_name", "")),
            args=data.get("args", {}),
            prompt_message=str(data.get("prompt_message", "")),
            expires_at=data.get("expires_at"),
            timestamp=event.timestamp,
        )
    if et == "approval_resolved":
        return ApprovalResolvedEvent(
            conversation_id=event.conversation_id,
            challenge_token=str(data.get("challenge_token", "")),
            task_id=str(data.get("task_id", "")),
            status=str(data.get("status", "approved")),
            approver=str(data.get("approver", "operator")),
            reason=str(data.get("reason", "")),
            override_args=data.get("override_args"),
            resolved_at=str(data.get("resolved_at", "")),
            timestamp=event.timestamp,
        )
    if et == "saga_started":
        return SagaStartedEvent(
            conversation_id=event.conversation_id,
            saga_id=str(data.get("saga_id", "")),
            name=str(data.get("name", "")),
            timestamp=event.timestamp,
        )
    if et == "saga_step_completed":
        return SagaStepCompletedEvent(
            conversation_id=event.conversation_id,
            saga_id=str(data.get("saga_id", "")),
            step=str(data.get("step", "")),
            timestamp=event.timestamp,
        )
    if et == "saga_compensating":
        return SagaCompensatingEvent(
            conversation_id=event.conversation_id,
            saga_id=str(data.get("saga_id", "")),
            step=str(data.get("step", "")),
            reason=str(data.get("reason", "")),
            timestamp=event.timestamp,
        )
    if et == "saga_completed":
        return SagaCompletedEvent(
            conversation_id=event.conversation_id,
            saga_id=str(data.get("saga_id", "")),
            result=data.get("result"),
            timestamp=event.timestamp,
        )

    return AgentEvent(
        conversation_id=event.conversation_id,
        event_type=et,
        timestamp=event.timestamp,
    )


@dataclass
class ConversationViewState:
    conversation_id: str
    status: str = "idle"  # "idle" | "busy" | "settled" | "failed"
    current_leaf_entry_id: str | None = None
    entries: list[dict[str, Any]] = field(default_factory=list)
    docs: dict[str, Any] = field(default_factory=dict)
    active_task: dict[str, Any] | None = None
    connected_clients: list[str] = field(default_factory=list)
    pending_approvals: list[dict[str, Any]] = field(default_factory=list)
    active_sagas: list[dict[str, Any]] = field(default_factory=list)
    version: int = 1
    timestamp: str = field(default_factory=utc_now_iso)


class ReactiveViewState:
    def __init__(self, initial_state: ConversationViewState) -> None:
        self.value = initial_state
        self._subscribers: set[Callable[[ConversationViewState], Any]] = set()
        self._queues: set[asyncio.Queue[ConversationViewState]] = set()
        self._lock = asyncio.Lock()
        self._closed = False

    def subscribe(
        self,
        callback: Callable[[ConversationViewState], Any],
    ) -> Callable[[], None]:
        self._subscribers.add(callback)

        def unsubscribe() -> None:
            self._subscribers.discard(callback)

        return unsubscribe

    async def watch(self) -> AsyncIterator[ConversationViewState]:
        q: asyncio.Queue[ConversationViewState] = asyncio.Queue(maxsize=100)
        async with self._lock:
            self._queues.add(q)
            q.put_nowait(self.value)
        try:
            while not self._closed:
                val = await q.get()
                yield val
        finally:
            async with self._lock:
                self._queues.discard(q)

    def notify(self, new_state: ConversationViewState) -> None:
        if self._closed:
            return
        self.value = new_state
        for cb in list(self._subscribers):
            try:
                res = cb(new_state)
                if inspect.isawaitable(res):
                    asyncio.create_task(res)
            except Exception:
                pass

        for q in list(self._queues):
            try:
                q.put_nowait(new_state)
            except (asyncio.QueueFull, Exception):
                pass

    def close(self) -> None:
        self._closed = True
        self._subscribers.clear()


class ReactiveDocumentState(Generic[TDoc]):
    def __init__(self, initial_value: TDoc) -> None:
        self.value: TDoc = initial_value
        self._subscribers: set[Callable[[TDoc], Any]] = set()
        self._queues: set[asyncio.Queue[TDoc]] = set()
        self._lock = asyncio.Lock()
        self._closed = False

    def subscribe(self, callback: Callable[[TDoc], Any]) -> Callable[[], None]:
        self._subscribers.add(callback)

        def unsubscribe() -> None:
            self._subscribers.discard(callback)

        return unsubscribe

    async def watch(self) -> AsyncIterator[TDoc]:
        q: asyncio.Queue[TDoc] = asyncio.Queue(maxsize=100)
        async with self._lock:
            self._queues.add(q)
            q.put_nowait(self.value)
        try:
            while not self._closed:
                val = await q.get()
                yield val
        finally:
            async with self._lock:
                self._queues.discard(q)

    def notify(self, new_val: TDoc) -> None:
        if self._closed:
            return
        self.value = new_val
        for cb in list(self._subscribers):
            try:
                res = cb(new_val)
                if inspect.isawaitable(res):
                    asyncio.create_task(res)
            except Exception:
                pass

        for q in list(self._queues):
            try:
                q.put_nowait(new_val)
            except (asyncio.QueueFull, Exception):
                pass

    def close(self) -> None:
        self._closed = True
        self._subscribers.clear()


class MultiplayerPresenceHub:
    def __init__(self) -> None:
        self._clients: dict[str, set[str]] = {}
        self._lock = asyncio.Lock()

    def connect(self, conversation_id: str, client_id: str) -> None:
        if conversation_id not in self._clients:
            self._clients[conversation_id] = set()
        self._clients[conversation_id].add(client_id)
        stream_hub.publish(
            conversation_id,
            StreamEvent(
                conversation_id=conversation_id,
                event_type="client_presence",
                data={"client_id": client_id, "action": "joined"},
            ),
        )

    def disconnect(self, conversation_id: str, client_id: str) -> None:
        if conversation_id in self._clients:
            self._clients[conversation_id].discard(client_id)
            if not self._clients[conversation_id]:
                del self._clients[conversation_id]
        stream_hub.publish(
            conversation_id,
            StreamEvent(
                conversation_id=conversation_id,
                event_type="client_presence",
                data={"client_id": client_id, "action": "left"},
            ),
        )

    def get_connected_clients(self, conversation_id: str) -> list[str]:
        return sorted(list(self._clients.get(conversation_id, set())))

    def clear(self) -> None:
        self._clients.clear()


multiplayer_hub = MultiplayerPresenceHub()


class ConversationStreamHub:
    def __init__(self) -> None:
        self._subscribers: dict[str, set[asyncio.Queue[StreamEvent]]] = {}
        self._reactive_views: dict[str, set[ReactiveViewState]] = {}
        self._reactive_docs: dict[tuple[str, str, str], set[ReactiveDocumentState[Any]]] = {}
        self._lock = asyncio.Lock()

    async def subscribe(self, conversation_id: str) -> AsyncIterator[StreamEvent]:
        q: asyncio.Queue[StreamEvent] = asyncio.Queue(maxsize=1000)
        async with self._lock:
            if conversation_id not in self._subscribers:
                self._subscribers[conversation_id] = set()
            self._subscribers[conversation_id].add(q)

        try:
            while True:
                event = await q.get()
                yield event
        finally:
            async with self._lock:
                if conversation_id in self._subscribers:
                    self._subscribers[conversation_id].discard(q)
                    if not self._subscribers[conversation_id]:
                        del self._subscribers[conversation_id]

    async def subscribe_events(self, conversation_id: str) -> AsyncIterator[AgentEvent]:
        async for raw_event in self.subscribe(conversation_id):
            agent_evt = translate_stream_event(raw_event)
            yield agent_evt

    def register_reactive_view(
        self,
        conversation_id: str,
        reactive_view: ReactiveViewState,
    ) -> None:
        if conversation_id not in self._reactive_views:
            self._reactive_views[conversation_id] = set()
        self._reactive_views[conversation_id].add(reactive_view)

    def unregister_reactive_view(
        self,
        conversation_id: str,
        reactive_view: ReactiveViewState,
    ) -> None:
        if conversation_id in self._reactive_views:
            self._reactive_views[conversation_id].discard(reactive_view)
            if not self._reactive_views[conversation_id]:
                del self._reactive_views[conversation_id]

    def register_reactive_doc(
        self,
        conversation_id: str,
        kind: str,
        doc_id: str,
        reactive_doc: ReactiveDocumentState[Any],
    ) -> None:
        key = (conversation_id, kind, doc_id)
        if key not in self._reactive_docs:
            self._reactive_docs[key] = set()
        self._reactive_docs[key].add(reactive_doc)

    def unregister_reactive_doc(
        self,
        conversation_id: str,
        kind: str,
        doc_id: str,
        reactive_doc: ReactiveDocumentState[Any],
    ) -> None:
        key = (conversation_id, kind, doc_id)
        if key in self._reactive_docs:
            self._reactive_docs[key].discard(reactive_doc)
            if not self._reactive_docs[key]:
                del self._reactive_docs[key]

    def notify_reactive_doc(
        self,
        conversation_id: str,
        kind: str,
        doc_id: str,
        value: Any,
    ) -> None:
        key = (conversation_id, kind, doc_id)
        for r_doc in list(self._reactive_docs.get(key, set())):
            r_doc.notify(value)

    def publish(self, conversation_id: str, event: StreamEvent) -> None:
        queues = list(self._subscribers.get(conversation_id, []))
        for q in queues:
            try:
                q.put_nowait(event)
            except (asyncio.QueueFull, Exception):
                pass


stream_hub = ConversationStreamHub()
