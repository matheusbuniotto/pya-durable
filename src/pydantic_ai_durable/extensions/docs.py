import json
import logging
from dataclasses import dataclass
from typing import Any, Callable, Generic, TypeVar

import aiosqlite

from pydantic_ai_durable.models import ConversationRecord, DurableDocRecord, utc_now_iso
from pydantic_ai_durable.storage.base import DurableStorage

logger = logging.getLogger("pydantic_ai_durable.extensions.docs")

T = TypeVar("T")


@dataclass
class DocDefinition(Generic[T]):
    doc_type: str
    schema: type[T]
    version: int = 1
    initial: Callable[[], T] | None = None
    migrate: Callable[[int, dict[str, Any]], dict[str, Any]] | None = None


def define_doc(
    doc_type: str,
    schema: type[T],
    version: int = 1,
    initial: Callable[[], T] | None = None,
    migrate: Callable[[int, dict[str, Any]], dict[str, Any]] | None = None,
) -> DocDefinition[T]:
    effective_initial = initial
    if effective_initial is None:
        effective_initial = lambda: schema()  # type: ignore[call-arg]
    return DocDefinition(
        doc_type=doc_type,
        schema=schema,
        version=version,
        initial=effective_initial,
        migrate=migrate,
    )


class DocumentManager:
    @staticmethod
    async def get_doc(
        storage: DurableStorage,
        doc_def: DocDefinition[T],
        conversation_id: str,
        doc_id: str = "default",
        conn: Any = None,
    ) -> T:
        if not storage._initialized:
            await storage.init_db()
        record = await storage.get_durable_doc(doc_id, conversation_id, conn=conn)
        if record is None:
            if doc_def.initial is not None:
                return doc_def.initial()
            return doc_def.schema()  # type: ignore[call-arg]

        data = json.loads(record.data_json)
        current_version = record.version

        if current_version < doc_def.version and doc_def.migrate is not None:
            data = doc_def.migrate(current_version, data)
            current_version = doc_def.version
            migrated_record = DurableDocRecord(
                doc_id=doc_id,
                conversation_id=conversation_id,
                version=current_version,
                doc_type=doc_def.doc_type,
                data_json=json.dumps(data),
                updated_at=utc_now_iso(),
            )
            await storage.upsert_durable_doc(migrated_record, conn=conn)

        schema = doc_def.schema
        if hasattr(schema, "model_validate"):
            return schema.model_validate(data)  # type: ignore[no-any-return]
        if isinstance(data, dict):
            try:
                return schema(**data)  # type: ignore[no-any-return]
            except TypeError:
                pass
        return data  # type: ignore[no-any-return]

    @staticmethod
    async def set_doc(
        storage: DurableStorage,
        doc_def: DocDefinition[T],
        conversation_id: str,
        value: T,
        doc_id: str = "default",
        conn: Any = None,
    ) -> None:
        if not storage._initialized:
            await storage.init_db()
        conv = await storage.get_conversation(conversation_id, conn=conn)
        if conv is None:
            await storage.insert_conversation(
                ConversationRecord(
                    conversation_id=conversation_id,
                    agent_config_json="{}",
                ),
                conn=conn,
            )

        if hasattr(value, "model_dump_json"):
            data_json = value.model_dump_json()
        elif hasattr(value, "model_dump"):
            data_json = json.dumps(value.model_dump())
        elif isinstance(value, (dict, list, str, int, float, bool)):
            data_json = json.dumps(value)
        elif hasattr(value, "__dict__"):
            data_json = json.dumps(value.__dict__)
        else:
            data_json = json.dumps(value)

        record = DurableDocRecord(
            doc_id=doc_id,
            conversation_id=conversation_id,
            version=doc_def.version,
            doc_type=doc_def.doc_type,
            data_json=data_json,
            updated_at=utc_now_iso(),
        )
        await storage.upsert_durable_doc(record, conn=conn)

        from pydantic_ai_durable.streaming import StreamEvent, stream_hub
        stream_hub.publish(
            conversation_id,
            StreamEvent(
                conversation_id=conversation_id,
                event_type="doc_updated",
                data={
                    "doc_id": doc_id,
                    "kind": doc_def.doc_type,
                    "version": doc_def.version,
                },
            ),
        )
        stream_hub.notify_reactive_doc(conversation_id, doc_def.doc_type, doc_id, value)

