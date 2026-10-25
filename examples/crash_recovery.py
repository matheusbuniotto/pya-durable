"""Crash recovery demo: kill the process mid-run, restart, and pick up where it stopped.

Runs offline (no API key). Usage:

    uv run python examples/crash_recovery.py           # orchestrates the whole demo
    uv run python examples/crash_recovery.py --worker  # one worker run (used by the demo)
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from pydantic_ai_durable import Harness, SqliteStorage, durable_tool

STEPS = ["provision", "migrate", "deploy", "verify"]


@durable_tool(replay="unsafe")
async def run_step(name: str) -> str:
    """A side-effecting step. 'unsafe' means: never silently repeated after a crash."""
    print(f"  [tool] running step: {name}", flush=True)
    await asyncio.sleep(0.2)
    if name == "deploy" and "--crash" in sys.argv:
        print("  [tool] 💥 process killed mid-deploy (simulated kill -9)", flush=True)
        os._exit(137)
    return f"{name}: ok"


def scripted_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    done = sum(isinstance(p, ToolReturnPart) for m in messages for p in m.parts)
    if done < len(STEPS):
        return ModelResponse(parts=[ToolCallPart("run_step", {"name": STEPS[done]})])
    return ModelResponse(parts=[TextPart("All steps finished.")])


async def worker(db: str) -> None:
    agent = Agent(FunctionModel(scripted_model), tools=[run_step])
    harness = await Harness.open(SqliteStorage(db), agent=agent)
    reports = await harness.resume()
    for r in reports:
        print(
            f"  ♻️  recovered task {r.task_id}: {r.previous_status.value} -> {r.new_status.value}, "
            f"unsafe tools sealed (NOT re-run): {r.unsafe_tools_interrupted}, "
            f"safe tools replayed: {r.safe_tools_replayed}"
        )
    if not reports:
        root = await harness.root()
        result = await root.submit({"content": "Ship it", "requestId": "ship-001"})
        print(f"  ✅ agent says: {result.output}")
    await harness.close()


def main() -> None:
    if "--worker" in sys.argv:
        asyncio.run(worker(os.environ["DEMO_DB"]))
        return
    with tempfile.TemporaryDirectory() as tmp:
        env = {**os.environ, "PYDANTIC_AI_NO_BANNER": "1", "DEMO_DB": str(Path(tmp) / "agent.db")}
        me = [sys.executable, __file__, "--worker"]
        print("Run 1: starting (will crash during 'deploy')", flush=True)
        subprocess.run([*me, "--crash"], env=env, check=False)
        print("\nRun 2: restarting the same process", flush=True)
        subprocess.run(me, env=env, check=True)


if __name__ == "__main__":
    main()
