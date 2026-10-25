import inspect
import json
import logging
import uuid
from typing import Any, Awaitable, Callable

from pya_durable.models import (
    DurableDocRecord,
    TranscriptEntryKind,
    TranscriptEntryRecord,
    utc_now_iso,
)
from pya_durable.replay import durable_tool
from pya_durable.storage import DurableStorage

logger = logging.getLogger("pya_durable.compaction")


class BackgroundCompactor:
    @staticmethod
    def should_compact(token_count: int, threshold: int = 4000) -> bool:
        return token_count >= threshold

    @classmethod
    async def compact_conversation(
        cls,
        storage: DurableStorage,
        conversation_id: str,
        leaf_entry_id: str,
        summarizer: Callable[[list[TranscriptEntryRecord]], Awaitable[str] | str] | None = None,
        reserve_recent_turns: int = 2,
    ) -> str:
        ancestors = await storage.get_ancestor_entries(leaf_entry_id)
        if not ancestors:
            return leaf_entry_id

        turn_start_indices: list[int] = []
        for idx, entry in enumerate(ancestors):
            if entry.kind == TranscriptEntryKind.USER:
                turn_start_indices.append(idx)

        if len(turn_start_indices) > reserve_recent_turns:
            cutoff_limit = turn_start_indices[-reserve_recent_turns]
        else:
            cutoff_limit = max(0, len(ancestors) - reserve_recent_turns * 3)

        if cutoff_limit <= 0:
            cutoff_id = ancestors[0].entry_id
            doc_record = DurableDocRecord(
                doc_id="compaction_summary",
                conversation_id=conversation_id,
                version=1,
                doc_type="compaction",
                data_json=json.dumps({"cutoff_entry_id": cutoff_id, "summary": f"Summary up to {cutoff_id}"}),
                updated_at=utc_now_iso(),
            )
            await storage.upsert_durable_doc(doc_record)
            logger.info(
                "Compacted turns up to entry id %s into background checkpoint",
                cutoff_id,
            )
            return cutoff_id

        candidate_idx = cutoff_limit - 1
        safe_idx = candidate_idx
        while safe_idx >= 0:
            entry = ancestors[safe_idx]
            if entry.kind == TranscriptEntryKind.TOOL_CALL:
                has_subsequent_result = False
                for forward in ancestors[safe_idx + 1 : safe_idx + 3]:
                    if forward.kind in (
                        TranscriptEntryKind.TOOL_RESULT,
                        TranscriptEntryKind.MODEL_RESPONSE,
                        TranscriptEntryKind.MODEL_REQUEST,
                    ):
                        has_subsequent_result = True
                        break
                if not has_subsequent_result:
                    safe_idx -= 1
                    continue
            break

        if safe_idx < 0:
            safe_idx = 0

        cutoff_entry = ancestors[safe_idx]
        cutoff_id = cutoff_entry.entry_id
        compacted_entries = ancestors[: safe_idx + 1]

        if summarizer is not None:
            res = summarizer(compacted_entries)
            if inspect.isawaitable(res):
                summary_text = await res
            else:
                summary_text = str(res)
        else:
            summary_text = f"Compacted {len(compacted_entries)} transcript entries up to {cutoff_id}."

        doc_record = DurableDocRecord(
            doc_id="compaction_summary",
            conversation_id=conversation_id,
            version=1,
            doc_type="compaction",
            data_json=json.dumps({"cutoff_entry_id": cutoff_id, "summary": summary_text}),
            updated_at=utc_now_iso(),
        )
        await storage.upsert_durable_doc(doc_record)

        logger.info(
            "Compacted turns up to entry id %s into background checkpoint",
            cutoff_id,
        )
        return cutoff_id


def create_handoff_tool(
    storage: DurableStorage,
    conversation_id: str,
    harness: Any = None,
) -> Any:
    @durable_tool(replay="safe")
    async def handoff(note: str) -> str:
        doc_record = DurableDocRecord(
            doc_id="handoff_note",
            conversation_id=conversation_id,
            version=1,
            doc_type="handoff",
            data_json=json.dumps({"note": note}),
            updated_at=utc_now_iso(),
        )
        await storage.upsert_durable_doc(doc_record)

        parent_id = getattr(harness, "current_leaf_entry_id", None) if harness else None
        if not parent_id:
            transcript = await storage.get_conversation_transcript(conversation_id)
            if transcript:
                parent_id = transcript[-1].entry_id

        entry_id = f"entry-{uuid.uuid4().hex[:16]}"
        entry = TranscriptEntryRecord(
            entry_id=entry_id,
            conversation_id=conversation_id,
            parent_entry_id=parent_id,
            kind=TranscriptEntryKind.SYSTEM,
            message_json=json.dumps({"type": "handoff", "handoff_note": note}),
            created_at=utc_now_iso(),
        )
        await storage.insert_transcript_entry(entry)
        if harness is not None:
            harness.current_leaf_entry_id = entry_id
        return f"Handoff saved: {note}"

    return handoff


def create_search_history_tool(storage: DurableStorage, conversation_id: str) -> Any:
    @durable_tool(replay="safe")
    async def search_history(query: str) -> str:
        entries = await storage.get_conversation_transcript(conversation_id)
        results = []
        q_lower = query.lower()
        for e in entries:
            if q_lower in e.message_json.lower():
                results.append(
                    {
                        "entry_id": e.entry_id,
                        "kind": e.kind.value if hasattr(e.kind, "value") else str(e.kind),
                        "content": e.message_json,
                    }
                )
        if not results:
            return f"No archived messages matched query: {query}"
        return json.dumps(results)

    return search_history
