from __future__ import annotations

from typing import Any

from pydantic_ai_durable.env.local import CommandTimeoutError, ExecutionEnv
from pydantic_ai_durable.extensions.registry import ExtensionDefinition, define_extension, define_section
from pydantic_ai_durable.tasks.replay import durable_tool, set_tool_partial_output


def create_read_file_tool(env: ExecutionEnv) -> Any:
    @durable_tool(name="read_file", replay="safe")
    async def read_file(path: str, offset: int = 0, limit: int | None = None) -> str:
        return await env.read_file(path, offset=offset, limit=limit)

    return read_file


def create_write_file_tool(env: ExecutionEnv) -> Any:
    @durable_tool(name="write_file", replay="unsafe")
    async def write_file(path: str, content: str, overwrite: bool = True) -> str:
        await env.write_file(path, content, overwrite=overwrite)
        return f"Successfully wrote {len(content)} characters to {path}"

    return write_file


def create_edit_file_tool(env: ExecutionEnv) -> Any:
    @durable_tool(name="edit_file", replay="unsafe")
    async def edit_file(path: str, target: str, replacement: str, count: int = 1) -> str:
        await env.edit_file(path, target, replacement, count=count)
        return f"Successfully replaced target in {path}"

    return edit_file


def create_exec_command_tool(env: ExecutionEnv) -> Any:
    @durable_tool(name="exec_command", replay="unsafe")
    async def exec_command(command: str, timeout_ms: int = 30000) -> dict[str, Any]:
        try:
            res = await env.exec_command(command, timeout_ms=timeout_ms)
            if res.stdout:
                await set_tool_partial_output(res.stdout)
            return {
                "exit_code": res.exit_code,
                "stdout": res.stdout,
                "stderr": res.stderr,
            }
        except CommandTimeoutError:
            await set_tool_partial_output(f"Command timed out: {command}")
            raise
        except Exception as exc:
            await set_tool_partial_output(f"Command failed: {exc!s}")
            raise

    return exec_command


def create_list_dir_tool(env: ExecutionEnv) -> Any:
    @durable_tool(name="list_dir", replay="safe")
    async def list_dir(path: str = ".", recursive: bool = False) -> list[dict[str, Any]]:
        entries = await env.list_dir(path=path, recursive=recursive)
        return [
            {
                "name": e.name,
                "path": e.path,
                "is_dir": e.is_dir,
                "size_bytes": e.size_bytes,
                "modified_at": e.modified_at,
            }
            for e in entries
        ]

    return list_dir


def create_coding_extension(
    env: ExecutionEnv,
    name: str = "coding_tools",
) -> ExtensionDefinition:
    tools = [
        create_read_file_tool(env),
        create_write_file_tool(env),
        create_edit_file_tool(env),
        create_exec_command_tool(env),
        create_list_dir_tool(env),
    ]

    section = define_section(
        name=f"{name}_workspace",
        fn=lambda ctx: (
            f"Available filesystem and execution environment.\n"
            f"Current working directory: {env.cwd}\n"
            f"Tools: read_file, write_file, edit_file, exec_command, list_dir."
        ),
        priority=20,
    )

    return define_extension(
        name=name,
        version="1.0.0",
        sections=[section],
        tools=tools,
    )
