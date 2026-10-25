"""Zero-copy conversation forking: explore a "what-if" without re-running history.

Runs offline (no API key):  uv run python examples/fork_conversation.py
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path

os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")

from pydantic_ai import Agent  # noqa: E402
from pydantic_ai.models.test import TestModel  # noqa: E402

from pydantic_ai_durable import Harness, SqliteStorage  # noqa: E402


async def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        agent = Agent(TestModel(custom_output_text="noted"))
        harness = await Harness.open(
            SqliteStorage(str(Path(tmp) / "agent.db")), agent=agent, conversation_id="main"
        )
        await harness.run("Plan the Q3 rollout")
        checkpoint = harness.current_leaf_entry_id

        await harness.run("Proceed with the standard rollout")

        what_if = await harness.fork(fork_entry_id=checkpoint, new_conversation_id="what-if")
        await what_if.run("Actually, what if we roll back immediately?")

        main_log = await harness.get_transcript()
        fork_log = await what_if.get_transcript()
        print(f"main    transcript entries: {len(main_log)}")
        print(f"what-if transcript entries: {len(fork_log)} (shares history up to the fork point)")
        await harness.close()


if __name__ == "__main__":
    asyncio.run(main())
