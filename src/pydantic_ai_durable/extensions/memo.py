from __future__ import annotations

import json
from typing import Any

from pydantic_ai_durable.storage.base import DurableStorage


class MemoManager:
    @staticmethod
    async def memo(
        storage: DurableStorage,
        conversation_id: str,
        key: str,
        value: Any = None,
    ) -> Any:
        if not storage._initialized:
            await storage.init_db()
        existing = await storage.get_memo(conversation_id, key)
        if existing is not None:
            try:
                return json.loads(existing.value_json)
            except Exception:
                return existing.value_json

        if value is None:
            return None

        conv = await storage.get_conversation(conversation_id)
        if conv is None:
            from pydantic_ai_durable.models import ConversationRecord

            await storage.insert_conversation(
                ConversationRecord(
                    conversation_id=conversation_id,
                    agent_config_json="{}",
                )
            )

        val_json = json.dumps(value)
        await storage.insert_memo(conversation_id, key, val_json)

        committed = await storage.get_memo(conversation_id, key)
        if committed is not None:
            try:
                return json.loads(committed.value_json)
            except Exception:
                return committed.value_json
        return value
