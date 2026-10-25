"""The overeager intern: an agent tries to DROP a production table, a human steps in.

The tool call pauses behind an approval gate. The operator doesn't just say "no":
they rewrite the arguments so the (still-running) agent harmlessly drops a backup instead.
Runs offline (no API key):  uv run python examples/approval_override.py
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from pydantic_ai_durable import (
    ApprovalManager,
    ApprovalRequiredError,
    ConversationRecord,
    SqliteStorage,
    TaskRecord,
    TaskStatus,
    durable_execution_context,
    durable_tool,
)

DATABASE = {"users", "orders", "users_backup"}


@durable_tool(require_approval=True)
async def drop_table(name: str) -> str:
    DATABASE.discard(name)
    return f"dropped table {name!r}"


async def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        storage = SqliteStorage(str(Path(tmp) / "agent.db"))
        await storage.init_db()
        await storage.insert_conversation(
            ConversationRecord(conversation_id="c1", agent_config_json="{}")
        )
        await storage.insert_task(
            TaskRecord(
                task_id="t1",
                task_type="user_turn",
                status=TaskStatus.RUNNING,
                owner_kind="conversation",
                owner_id="c1",
                phase="running",
                input_json="{}",
                state_json="{}",
            )
        )

        print("🤖 intern: 'Cleaning up... DROP TABLE users'")
        async with durable_execution_context(storage, "t1", "c1"):
            try:
                await drop_table("users")
            except ApprovalRequiredError as gate:
                token = gate.challenge_token
                print(f"⏸️  paused behind approval gate (token {token[:8]}...)")

        print("🧑‍💼 operator: 'Nice try. Drop users_backup instead.'")
        await ApprovalManager(storage).resolve(
            token, approved=True, override_args={"name": "users_backup"}, approver="operator"
        )

        async with durable_execution_context(storage, "t1", "c1"):
            print("🤖 intern retries the same call:", await drop_table("users"))

        print(f"🗄️  tables left: {sorted(DATABASE)}  (users is safe)")
        await storage.close()


if __name__ == "__main__":
    asyncio.run(main())
