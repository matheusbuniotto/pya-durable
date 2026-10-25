import asyncio
import json
import logging
import subprocess
from pathlib import Path

import pytest
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.test import TestModel
from pydantic_ai_durable.worktrees import GitWorktreeManager

from pydantic_ai_durable.branching import (
    DurableConversation,
    InvalidForkEntryError,
    get_branch_ancestors,
)
from pydantic_ai_durable.harness import DurableHarness
from pydantic_ai_durable.models import (
    ConversationRecord,
    TranscriptEntryKind,
    TranscriptEntryRecord,
)
from pydantic_ai_durable.steering import SteeringQueue, global_steering_queue
from pydantic_ai_durable.storage import AioSqliteStorage


@pytest.mark.asyncio
async def test_recursive_cte_ancestor_query_multiple_forks(tmp_path: Path) -> None:
    db_file = tmp_path / "test_cte.db"
    async with AioSqliteStorage(db_file) as storage:
        conv_root = ConversationRecord(
            conversation_id="conv-root",
            agent_config_json="{}",
        )
        await storage.insert_conversation(conv_root)

        entry1 = TranscriptEntryRecord(
            entry_id="entry-1",
            conversation_id="conv-root",
            parent_entry_id=None,
            kind=TranscriptEntryKind.USER,
            message_json=json.dumps({"prompt": "root prompt"}),
        )
        entry2 = TranscriptEntryRecord(
            entry_id="entry-2",
            conversation_id="conv-root",
            parent_entry_id="entry-1",
            kind=TranscriptEntryKind.MODEL_RESPONSE,
            message_json=json.dumps({"text": "root response"}),
        )
        await storage.insert_transcript_entry(entry1)
        await storage.insert_transcript_entry(entry2)

        conv_child = ConversationRecord(
            conversation_id="conv-child",
            parent_conversation_id="conv-root",
            fork_entry_id="entry-2",
            agent_config_json="{}",
        )
        await storage.insert_conversation(conv_child)

        entry3 = TranscriptEntryRecord(
            entry_id="entry-3",
            conversation_id="conv-child",
            parent_entry_id="entry-2",
            kind=TranscriptEntryKind.USER,
            message_json=json.dumps({"prompt": "child prompt"}),
        )
        await storage.insert_transcript_entry(entry3)

        ancestors = await get_branch_ancestors(storage, "entry-3")
        assert len(ancestors) == 3
        assert [a.entry_id for a in ancestors] == ["entry-1", "entry-2", "entry-3"]


@pytest.mark.asyncio
async def test_fork_creation_without_copying_rows(tmp_path: Path) -> None:
    db_file = tmp_path / "test_fork.db"
    async with AioSqliteStorage(db_file) as storage:
        agent = Agent(TestModel())
        harness = DurableHarness(agent, storage, conversation_id="conv-parent")
        res = await harness.run("hello")
        leaf_id = res.leaf_entry_id

        conn = await storage.get_connection()
        async with conn.execute("SELECT count(*) FROM transcript_entries") as cursor:
            row = await cursor.fetchone()
            count_before = row[0] if row else 0

        child = await harness.fork(fork_entry_id=leaf_id, new_conversation_id="conv-child")

        async with conn.execute("SELECT count(*) FROM transcript_entries") as cursor:
            row = await cursor.fetchone()
            count_after = row[0] if row else 0

        assert count_after == count_before
        assert child.conversation_id == "conv-child"
        assert child.parent_conversation_id == "conv-parent"
        assert child.fork_entry_id == leaf_id

        transcript = await child.get_transcript()
        assert len(transcript) == count_before


@pytest.mark.asyncio
async def test_invalid_fork_entry_error(tmp_path: Path) -> None:
    db_file = tmp_path / "test_invalid_fork.db"
    async with AioSqliteStorage(db_file) as storage:
        agent = Agent(TestModel())
        harness = DurableHarness(agent, storage, conversation_id="conv-test")
        await harness.run("initial message")

        with pytest.raises(InvalidForkEntryError):
            await harness.fork(fork_entry_id="nonexistent-entry-id")

        with pytest.raises(InvalidForkEntryError):
            conv = DurableConversation("conv-empty", storage, agent)
            await conv.fork(fork_entry_id=None)


@pytest.mark.asyncio
async def test_concurrent_execution_sibling_forks(tmp_path: Path) -> None:
    db_file = tmp_path / "test_sibling_forks.db"
    async with AioSqliteStorage(db_file) as storage:
        agent = Agent(TestModel())
        harness = DurableHarness(agent, storage, conversation_id="conv-root")
        res = await harness.run("root turn")
        fork_entry = res.leaf_entry_id

        fork1 = await harness.fork(fork_entry_id=fork_entry, new_conversation_id="conv-branch-1")
        fork2 = await harness.fork(fork_entry_id=fork_entry, new_conversation_id="conv-branch-2")

        res1, res2 = await asyncio.gather(
            fork1.run("branch 1 turn"),
            fork2.run("branch 2 turn"),
        )

        assert res1 is not None and res2 is not None
        assert res1.leaf_entry_id != res2.leaf_entry_id

        t1 = await fork1.get_transcript()
        t2 = await fork2.get_transcript()

        t1_ids = {e.entry_id for e in t1}
        t2_ids = {e.entry_id for e in t2}

        assert fork_entry in t1_ids
        assert fork_entry in t2_ids
        assert res1.leaf_entry_id in t1_ids
        assert res1.leaf_entry_id not in t2_ids
        assert res2.leaf_entry_id in t2_ids
        assert res2.leaf_entry_id not in t1_ids


@pytest.mark.asyncio
async def test_nested_branch_traversal(tmp_path: Path) -> None:
    db_file = tmp_path / "test_nested.db"
    async with AioSqliteStorage(db_file) as storage:
        agent = Agent(TestModel())
        harness_a = DurableHarness(agent, storage, conversation_id="conv-a")
        res_a = await harness_a.run("prompt a")

        harness_b = await harness_a.fork(
            fork_entry_id=res_a.leaf_entry_id, new_conversation_id="conv-b"
        )
        res_b = await harness_b.run("prompt b")

        harness_c = await harness_b.fork(
            fork_entry_id=res_b.leaf_entry_id, new_conversation_id="conv-c"
        )
        res_c = await harness_c.run("prompt c")

        transcript_c = await harness_c.get_transcript()
        c_ids = [e.entry_id for e in transcript_c]

        assert res_a.leaf_entry_id in c_ids
        assert res_b.leaf_entry_id in c_ids
        assert res_c.leaf_entry_id in c_ids
        assert c_ids.index(res_a.leaf_entry_id) < c_ids.index(res_b.leaf_entry_id)
        assert c_ids.index(res_b.leaf_entry_id) < c_ids.index(res_c.leaf_entry_id)


def test_steer_queue_fifo_buffer() -> None:
    q = SteeringQueue()
    conv = "conv-fifo-test"
    assert not q.has_pending(conv)

    q.enqueue(conv, "steer 1")
    q.enqueue(conv, "steer 2")
    q.enqueue(conv, "steer 3")
    assert q.has_pending(conv)

    assert q.pop_next(conv) == "steer 1"
    all_rest = q.pop_all(conv)
    assert all_rest == ["steer 2", "steer 3"]
    assert not q.has_pending(conv)


@pytest.mark.asyncio
async def test_mid_run_steer_injection_multi_step_tool(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    db_file = tmp_path / "test_steer.db"
    async with AioSqliteStorage(db_file) as storage:
        agent = Agent(TestModel())

        steered = False

        @agent.tool
        async def slow_tool(ctx: RunContext[None]) -> str:
            nonlocal steered
            if not steered:
                global_steering_queue.enqueue("conv-steer-lane", "inject mid-flight instruction")
                steered = True
            return "tool result"

        harness = DurableHarness(agent, storage, conversation_id="conv-steer-lane")
        res = await harness.run("trigger slow tool")

        assert res is not None
        assert (
            "Injected steer message into active turn for conversation id conv-steer-lane"
            in caplog.text
        )


@pytest.mark.asyncio
async def test_late_steer_handled_as_next_input(tmp_path: Path) -> None:
    db_file = tmp_path / "test_late_steer.db"
    async with AioSqliteStorage(db_file) as storage:
        agent = Agent(TestModel())
        harness = DurableHarness(agent, storage, conversation_id="conv-late-steer")
        await harness.run("first turn")

        global_steering_queue.enqueue("conv-late-steer", "late steering message")
        res = await harness.run()
        assert res is not None
        assert not global_steering_queue.has_pending("conv-late-steer")


def test_git_worktree_creation(tmp_path: Path) -> None:
    repo_dir = tmp_path / "main_repo"
    repo_dir.mkdir()
    subprocess.run(["git", "init"], cwd=str(repo_dir), check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.local"], cwd=str(repo_dir), check=True
    )
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=str(repo_dir), check=True)

    dummy_file = repo_dir / "README.md"
    dummy_file.write_text("initial commit\n")
    subprocess.run(["git", "add", "."], cwd=str(repo_dir), check=True)
    subprocess.run(["git", "commit", "-m", "initial commit"], cwd=str(repo_dir), check=True)

    wt_path = GitWorktreeManager.create_worktree(
        repo_dir=repo_dir,
        conversation_id="conv-worktree-1",
        branch_name="branch-explore-1",
    )
    try:
        assert wt_path.exists()
        assert (wt_path / ".git").exists()
        worktrees = GitWorktreeManager.list_worktrees(repo_dir)
        paths = [Path(w.get("worktree", "")).resolve() for w in worktrees]
        assert wt_path.resolve() in paths
    finally:
        GitWorktreeManager.remove_worktree(repo_dir, wt_path)


def test_git_worktree_cleanup(tmp_path: Path) -> None:
    repo_dir = tmp_path / "main_repo"
    repo_dir.mkdir()
    subprocess.run(["git", "init"], cwd=str(repo_dir), check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.local"], cwd=str(repo_dir), check=True
    )
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=str(repo_dir), check=True)

    dummy_file = repo_dir / "README.md"
    dummy_file.write_text("initial commit\n")
    subprocess.run(["git", "add", "."], cwd=str(repo_dir), check=True)
    subprocess.run(["git", "commit", "-m", "initial commit"], cwd=str(repo_dir), check=True)

    captured_path: Path | None = None
    with GitWorktreeManager.isolated_worktree(
        repo_dir=repo_dir,
        conversation_id="conv-temp-explore",
        branch_name="branch-temp-explore",
    ) as wt_path:
        captured_path = wt_path
        assert wt_path.exists()

    assert captured_path is not None
    assert not captured_path.exists()
    check_branch = subprocess.run(
        ["git", "rev-parse", "--verify", "branch-temp-explore"],
        cwd=str(repo_dir),
        capture_output=True,
    )
    assert check_branch.returncode != 0
