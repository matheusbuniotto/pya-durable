from pathlib import Path

import pytest
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from pya_durable.env import (
    CommandTimeoutError,
    ExecutionEnvError,
    FileNotFoundInEnvError,
    IsolatedSandboxEnv,
    LocalExecutionEnv,
    MockExecutionEnv,
    PathTraversalError,
)
from pya_durable.extensions import ExtensionRegistry
from pya_durable.harness import DurableHarness
from pya_durable.models import ReplayPolicy
from pya_durable.storage import AioSqliteStorage
from pya_durable.tools import (
    create_coding_extension,
    create_edit_file_tool,
    create_exec_command_tool,
    create_list_dir_tool,
    create_read_file_tool,
    create_write_file_tool,
)


@pytest.mark.asyncio
async def test_local_execution_env_file_lifecycle(tmp_path: Path):
    env = LocalExecutionEnv(cwd=tmp_path)
    assert env.cwd == tmp_path.resolve()

    await env.write_file("test.txt", "Initial file content line 1\nLine 2")
    assert await env.file_exists("test.txt") is True

    content = await env.read_file("test.txt")
    assert content == "Initial file content line 1\nLine 2"

    slice_content = await env.read_file("test.txt", offset=8, limit=12)
    assert slice_content == "file content"

    await env.edit_file("test.txt", target="Line 2", replacement="Modified Line 2")
    updated = await env.read_file("test.txt")
    assert updated == "Initial file content line 1\nModified Line 2"

    entries = await env.list_dir(".")
    assert len(entries) == 1
    assert entries[0].name == "test.txt"
    assert entries[0].is_dir is False


@pytest.mark.asyncio
async def test_local_execution_env_path_traversal_blocked(tmp_path: Path):
    sub = tmp_path / "sandbox"
    sub.mkdir()
    env = LocalExecutionEnv(cwd=sub, allow_outside_cwd=False)

    with pytest.raises(PathTraversalError):
        await env.write_file("../outside.txt", "escape payload")

    with pytest.raises(PathTraversalError):
        await env.read_file("/etc/passwd")


@pytest.mark.asyncio
async def test_local_execution_env_allow_outside_cwd(tmp_path: Path):
    sub = tmp_path / "sandbox"
    sub.mkdir()
    env = LocalExecutionEnv(cwd=sub, allow_outside_cwd=True)

    outside = tmp_path / "permitted.txt"
    outside.write_text("outside text", encoding="utf-8")

    res = await env.read_file(str(outside))
    assert res == "outside text"


@pytest.mark.asyncio
async def test_command_execution_success_and_exit_code(tmp_path: Path):
    env = LocalExecutionEnv(cwd=tmp_path, env_vars={"MY_VAR": "PSTACK"})
    res = await env.exec_command("echo hello && echo $MY_VAR")
    assert res.exit_code == 0
    assert "hello" in res.stdout
    assert "PSTACK" in res.stdout


@pytest.mark.asyncio
async def test_command_timeout_error(tmp_path: Path):
    env = LocalExecutionEnv(cwd=tmp_path)
    with pytest.raises(CommandTimeoutError):
        await env.exec_command("sleep 1", timeout_ms=50)


@pytest.mark.asyncio
async def test_isolated_sandbox_env_creation_and_cleanup(tmp_path: Path):
    sandbox = IsolatedSandboxEnv(base_dir=tmp_path)
    sandbox_path = sandbox.cwd
    assert sandbox_path.exists()

    await sandbox.write_file("ephemeral.txt", "data")
    assert (sandbox_path / "ephemeral.txt").exists()

    await sandbox.cleanup()
    assert not sandbox_path.exists()


@pytest.mark.asyncio
async def test_mock_execution_env_virtual_filesystem():
    mock_env = MockExecutionEnv(cwd="/virtual")
    assert mock_env.cwd == Path("/virtual")

    await mock_env.write_file("virtual.txt", "virtual content")
    assert await mock_env.file_exists("virtual.txt") is True

    res = await mock_env.read_file("virtual.txt")
    assert res == "virtual content"

    await mock_env.edit_file("virtual.txt", "content", "memory")
    edited = await mock_env.read_file("virtual.txt")
    assert edited == "virtual memory"

    entries = await mock_env.list_dir(".")
    assert len(entries) == 1
    assert entries[0].name == "virtual.txt"

    cmd_res = await mock_env.exec_command("echo test-echo")
    assert cmd_res.exit_code == 0
    assert "test-echo" in cmd_res.stdout


def test_durable_tools_replay_policies():
    mock_env = MockExecutionEnv()
    read_tool = create_read_file_tool(mock_env)
    write_tool = create_write_file_tool(mock_env)
    edit_tool = create_edit_file_tool(mock_env)
    exec_tool = create_exec_command_tool(mock_env)
    list_tool = create_list_dir_tool(mock_env)

    assert read_tool.replay_policy == ReplayPolicy.SAFE
    assert list_tool.replay_policy == ReplayPolicy.SAFE
    assert write_tool.replay_policy == ReplayPolicy.UNSAFE
    assert edit_tool.replay_policy == ReplayPolicy.UNSAFE
    assert exec_tool.replay_policy == ReplayPolicy.UNSAFE


@pytest.mark.asyncio
async def test_env_error_conditions(tmp_path: Path):
    env = LocalExecutionEnv(cwd=tmp_path)
    with pytest.raises(FileNotFoundInEnvError):
        await env.read_file("nonexistent.txt")

    await env.write_file("file.txt", "abc")
    with pytest.raises(ExecutionEnvError):
        await env.write_file("file.txt", "def", overwrite=False)

    with pytest.raises(ExecutionEnvError):
        await env.edit_file("file.txt", target="xyz", replacement="123")


@pytest.mark.asyncio
async def test_create_coding_extension_registered_and_rendered(tmp_path: Path):
    env = LocalExecutionEnv(cwd=tmp_path)
    ext = create_coding_extension(env)
    registry = ExtensionRegistry()
    registry.install(ext)

    assert "coding_tools" in registry.list_extensions()
    resolved = registry.resolve_tools([])
    tool_names = [getattr(t, "name", t.__name__) for t in resolved]
    assert "read_file" in tool_names
    assert "write_file" in tool_names
    assert "edit_file" in tool_names
    assert "exec_command" in tool_names
    assert "list_dir" in tool_names


@pytest.mark.asyncio
async def test_harness_and_conversation_env_scoping_and_forking(tmp_path: Path):
    db_file = tmp_path / "env_test.db"
    async with AioSqliteStorage(db_file) as storage:
        agent = Agent(TestModel())
        sandbox1 = IsolatedSandboxEnv(base_dir=tmp_path)
        sandbox2 = IsolatedSandboxEnv(base_dir=tmp_path)

        def env_factory(conv_id: str, cwd: Path | None):
            if "fork" in conv_id:
                return sandbox2
            return sandbox1

        harness = DurableHarness(
            agent,
            storage,
            conversation_id="conv-root",
            env_factory=env_factory,
        )
        assert harness.env.cwd == sandbox1.cwd

        fork_harness = await harness.fork(
            new_conversation_id="conv-fork",
            env=sandbox2,
        )
        assert fork_harness.env.cwd == sandbox2.cwd

        await sandbox1.cleanup()
        await sandbox2.cleanup()
