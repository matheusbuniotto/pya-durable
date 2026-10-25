from __future__ import annotations

import asyncio
import os
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Self


class ExecutionEnvError(Exception):
    pass


class PathTraversalError(ExecutionEnvError):
    pass


class CommandTimeoutError(ExecutionEnvError):
    pass


class FileNotFoundInEnvError(ExecutionEnvError):
    pass


@dataclass(frozen=True)
class ExecResult:
    exit_code: int
    stdout: str
    stderr: str


@dataclass(frozen=True)
class FileInfo:
    name: str
    path: str
    is_dir: bool
    size_bytes: int
    modified_at: str


class ExecutionEnv(ABC):
    @property
    @abstractmethod
    def cwd(self) -> Path: ...

    @abstractmethod
    async def read_file(self, path: str, offset: int = 0, limit: int | None = None) -> str: ...

    @abstractmethod
    async def write_file(self, path: str, content: str, overwrite: bool = True) -> None: ...

    @abstractmethod
    async def edit_file(self, path: str, target: str, replacement: str, count: int = 1) -> None: ...

    @abstractmethod
    async def exec_command(
        self,
        command: str,
        timeout_ms: int = 30000,
        env_vars: dict[str, str] | None = None,
    ) -> ExecResult: ...

    @abstractmethod
    async def list_dir(self, path: str = ".", recursive: bool = False) -> list[FileInfo]: ...

    @abstractmethod
    async def file_exists(self, path: str) -> bool: ...

    @abstractmethod
    async def cleanup(self) -> None: ...

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        await self.cleanup()


class LocalExecutionEnv(ExecutionEnv):
    def __init__(
        self,
        cwd: Path | str,
        env_vars: dict[str, str] | None = None,
        allow_outside_cwd: bool = False,
    ) -> None:
        self._cwd = Path(cwd).resolve()
        self.env_vars = env_vars or {}
        self.allow_outside_cwd = allow_outside_cwd

    @property
    def cwd(self) -> Path:
        return self._cwd

    def _resolve_safe_path(self, path: str | Path) -> Path:
        p = Path(path)
        if p.is_absolute():
            resolved = p.resolve()
        else:
            resolved = (self._cwd / p).resolve()

        if not self.allow_outside_cwd:
            try:
                resolved.relative_to(self._cwd)
            except ValueError:
                raise PathTraversalError(
                    f"Path '{path}' escapes working directory '{self._cwd}'"
                ) from None
        return resolved

    async def read_file(self, path: str, offset: int = 0, limit: int | None = None) -> str:
        resolved = self._resolve_safe_path(path)
        if not resolved.exists() or not resolved.is_file():
            raise FileNotFoundInEnvError(f"File not found: {path}")

        loop = asyncio.get_running_loop()
        content = await loop.run_in_executor(None, resolved.read_text, "utf-8")

        if offset > 0:
            content = content[offset:]
        if limit is not None:
            content = content[:limit]
        return content

    async def write_file(self, path: str, content: str, overwrite: bool = True) -> None:
        resolved = self._resolve_safe_path(path)
        if resolved.exists() and not overwrite:
            raise ExecutionEnvError(f"File already exists: {path}")

        loop = asyncio.get_running_loop()

        def _do_write() -> None:
            resolved.parent.mkdir(parents=True, exist_ok=True)
            resolved.write_text(content, encoding="utf-8")

        await loop.run_in_executor(None, _do_write)

    async def edit_file(
        self,
        path: str,
        target: str,
        replacement: str,
        count: int = 1,
    ) -> None:
        resolved = self._resolve_safe_path(path)
        if not resolved.exists() or not resolved.is_file():
            raise FileNotFoundInEnvError(f"File not found: {path}")

        loop = asyncio.get_running_loop()
        content = await loop.run_in_executor(None, resolved.read_text, "utf-8")

        if target not in content:
            raise ExecutionEnvError(f"Target string not found in {path}")

        new_content = content.replace(target, replacement, count)
        await loop.run_in_executor(
            None,
            lambda: resolved.write_text(new_content, encoding="utf-8"),
        )

    async def exec_command(
        self,
        command: str,
        timeout_ms: int = 30000,
        env_vars: dict[str, str] | None = None,
    ) -> ExecResult:
        merged_env = os.environ.copy()
        merged_env.update(self.env_vars)
        if env_vars:
            merged_env.update(env_vars)

        proc = await asyncio.create_subprocess_shell(
            command,
            cwd=str(self._cwd),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=merged_env,
        )

        timeout_sec = timeout_ms / 1000.0
        try:
            stdout_b, stderr_b = await asyncio.wait_for(
                proc.communicate(),
                timeout=timeout_sec,
            )
            stdout_str = stdout_b.decode("utf-8", errors="replace")
            stderr_str = stderr_b.decode("utf-8", errors="replace")
            return ExecResult(
                exit_code=proc.returncode if proc.returncode is not None else 0,
                stdout=stdout_str,
                stderr=stderr_str,
            )
        except asyncio.TimeoutError:
            try:
                proc.kill()
                await proc.wait()
            except ProcessLookupError:
                pass
            raise CommandTimeoutError(
                f"Command '{command}' timed out after {timeout_ms}ms"
            ) from None

    async def list_dir(self, path: str = ".", recursive: bool = False) -> list[FileInfo]:
        resolved = self._resolve_safe_path(path)
        if not resolved.exists() or not resolved.is_dir():
            raise FileNotFoundInEnvError(f"Directory not found: {path}")

        loop = asyncio.get_running_loop()

        def _scan() -> list[FileInfo]:
            results: list[FileInfo] = []
            if recursive:
                entries = list(resolved.rglob("*"))
            else:
                entries = list(resolved.iterdir())

            for entry in entries:
                try:
                    rel_p = str(entry.relative_to(self._cwd))
                except ValueError:
                    rel_p = str(entry)
                stat = entry.stat()
                mod_iso = datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat()
                size = stat.st_size if entry.is_file() else 0
                results.append(
                    FileInfo(
                        name=entry.name,
                        path=rel_p,
                        is_dir=entry.is_dir(),
                        size_bytes=size,
                        modified_at=mod_iso,
                    )
                )
            results.sort(key=lambda f: f.path)
            return results

        return await loop.run_in_executor(None, _scan)

    async def file_exists(self, path: str) -> bool:
        resolved = self._resolve_safe_path(path)
        return resolved.exists()

    async def cleanup(self) -> None:
        pass


class IsolatedSandboxEnv(LocalExecutionEnv):
    def __init__(
        self,
        base_dir: Path | str | None = None,
        env_vars: dict[str, str] | None = None,
    ) -> None:
        self._temp_dir: tempfile.TemporaryDirectory[str] | None = tempfile.TemporaryDirectory(
            dir=base_dir
        )
        super().__init__(
            cwd=Path(self._temp_dir.name),
            env_vars=env_vars,
            allow_outside_cwd=False,
        )

    async def cleanup(self) -> None:
        if self._temp_dir is not None:
            self._temp_dir.cleanup()
            self._temp_dir = None


class MockExecutionEnv(ExecutionEnv):
    def __init__(
        self,
        cwd: Path | str = "/workspace",
        env_vars: dict[str, str] | None = None,
    ) -> None:
        self._cwd = Path(cwd)
        self.env_vars = env_vars or {}
        self._files: dict[str, str] = {}

    @property
    def cwd(self) -> Path:
        return self._cwd

    def _normalize(self, path: str) -> str:
        p = Path(path)
        if p.is_absolute():
            return str(p)
        return str(self._cwd / p)

    async def read_file(self, path: str, offset: int = 0, limit: int | None = None) -> str:
        norm = self._normalize(path)
        if norm not in self._files:
            raise FileNotFoundInEnvError(f"File not found: {path}")
        content = self._files[norm]
        if offset > 0:
            content = content[offset:]
        if limit is not None:
            content = content[:limit]
        return content

    async def write_file(self, path: str, content: str, overwrite: bool = True) -> None:
        norm = self._normalize(path)
        if norm in self._files and not overwrite:
            raise ExecutionEnvError(f"File already exists: {path}")
        self._files[norm] = content

    async def edit_file(
        self,
        path: str,
        target: str,
        replacement: str,
        count: int = 1,
    ) -> None:
        norm = self._normalize(path)
        if norm not in self._files:
            raise FileNotFoundInEnvError(f"File not found: {path}")
        content = self._files[norm]
        if target not in content:
            raise ExecutionEnvError(f"Target string not found in {path}")
        self._files[norm] = content.replace(target, replacement, count)

    async def exec_command(
        self,
        command: str,
        timeout_ms: int = 30000,
        env_vars: dict[str, str] | None = None,
    ) -> ExecResult:
        cmd = command.strip()
        if cmd.startswith("echo "):
            text = cmd[5:].strip().strip('"').strip("'")
            return ExecResult(exit_code=0, stdout=f"{text}\n", stderr="")
        if cmd == "pwd":
            return ExecResult(exit_code=0, stdout=f"{self._cwd}\n", stderr="")
        return ExecResult(exit_code=0, stdout=f"mock: {cmd}\n", stderr="")

    async def list_dir(self, path: str = ".", recursive: bool = False) -> list[FileInfo]:
        prefix = self._normalize(path).rstrip("/") + "/"
        now_iso = datetime.now(timezone.utc).isoformat()
        results: list[FileInfo] = []

        seen_paths: set[str] = set()
        for fpath, content in self._files.items():
            if fpath.startswith(prefix):
                sub = fpath[len(prefix) :]
                if not recursive and "/" in sub:
                    dir_name = sub.split("/")[0]
                    dir_path = prefix + dir_name
                    if dir_path not in seen_paths:
                        seen_paths.add(dir_path)
                        results.append(
                            FileInfo(
                                name=dir_name,
                                path=dir_path,
                                is_dir=True,
                                size_bytes=0,
                                modified_at=now_iso,
                            )
                        )
                else:
                    if fpath not in seen_paths:
                        seen_paths.add(fpath)
                        results.append(
                            FileInfo(
                                name=Path(fpath).name,
                                path=fpath,
                                is_dir=False,
                                size_bytes=len(content.encode("utf-8")),
                                modified_at=now_iso,
                            )
                        )
        results.sort(key=lambda f: f.path)
        return results

    async def file_exists(self, path: str) -> bool:
        norm = self._normalize(path)
        return norm in self._files

    async def cleanup(self) -> None:
        self._files.clear()
