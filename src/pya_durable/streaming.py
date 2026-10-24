import asyncio
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from pya_durable.models import utc_now_iso


@dataclass(frozen=True)
class StreamEvent:
    conversation_id: str
    event_type: str
    data: dict[str, Any]
    timestamp: str = field(default_factory=utc_now_iso)


class ConversationStreamHub:
    def __init__(self) -> None:
        self._subscribers: dict[str, set[asyncio.Queue[StreamEvent]]] = {}
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

    def publish(self, conversation_id: str, event: StreamEvent) -> None:
        queues = list(self._subscribers.get(conversation_id, []))
        for q in queues:
            try:
                q.put_nowait(event)
            except (asyncio.QueueFull, Exception):
                pass


stream_hub = ConversationStreamHub()
