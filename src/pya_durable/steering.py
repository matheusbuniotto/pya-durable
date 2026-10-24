from collections import deque


class SteeringQueue:
    def __init__(self) -> None:
        self._queues: dict[str, deque[str]] = {}

    def enqueue(self, conversation_id: str, message: str) -> None:
        if conversation_id not in self._queues:
            self._queues[conversation_id] = deque()
        self._queues[conversation_id].append(message)

    def pop_next(self, conversation_id: str) -> str | None:
        q = self._queues.get(conversation_id)
        if q and len(q) > 0:
            return q.popleft()
        return None

    def pop_all(self, conversation_id: str) -> list[str]:
        q = self._queues.get(conversation_id)
        if not q:
            return []
        items = list(q)
        q.clear()
        return items

    def has_pending(self, conversation_id: str) -> bool:
        q = self._queues.get(conversation_id)
        return bool(q and len(q) > 0)

    def clear(self, conversation_id: str) -> None:
        if conversation_id in self._queues:
            self._queues[conversation_id].clear()


global_steering_queue = SteeringQueue()
