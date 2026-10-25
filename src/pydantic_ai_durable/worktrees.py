import shutil
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


class GitWorktreeManager:
    def __init__(self, repo_dir: str | Path | None = None) -> None:
        self.repo_dir = Path(repo_dir) if repo_dir else None

    @classmethod
    def create_worktree(
        cls,
        repo_dir: str | Path,
        conversation_id: str,
        branch_name: str | None = None,
        base_ref: str = "HEAD",
        worktree_path: str | Path | None = None,
    ) -> Path:
        repo_dir = Path(repo_dir).resolve()
        target_branch = branch_name or f"branch-{conversation_id}"
        wt_path = Path(worktree_path).resolve() if worktree_path else (repo_dir.parent / f"wt-{conversation_id}").resolve()

        if wt_path.exists():
            shutil.rmtree(wt_path, ignore_errors=True)

        check_branch = subprocess.run(
            ["git", "rev-parse", "--verify", target_branch],
            cwd=str(repo_dir),
            capture_output=True,
            text=True,
        )
        if check_branch.returncode == 0:
            cmd = ["git", "worktree", "add", str(wt_path), target_branch]
        else:
            cmd = ["git", "worktree", "add", "-b", target_branch, str(wt_path), base_ref]

        subprocess.run(cmd, cwd=str(repo_dir), capture_output=True, text=True, check=True)
        return wt_path

    @classmethod
    def remove_worktree(
        cls,
        repo_dir: str | Path,
        worktree_path: str | Path,
        force: bool = True,
        delete_branch: bool = True,
    ) -> None:
        repo_dir = Path(repo_dir).resolve()
        wt_path = Path(worktree_path).resolve()

        branch_to_delete = None
        if delete_branch:
            try:
                wts = cls.list_worktrees(repo_dir)
                for wt in wts:
                    if Path(wt.get("worktree", "")).resolve() == wt_path:
                        branch_ref = wt.get("branch", "")
                        if branch_ref.startswith("refs/heads/"):
                            branch_to_delete = branch_ref[len("refs/heads/") :]
                        elif branch_ref:
                            branch_to_delete = branch_ref
                        break
            except Exception:
                pass

        cmd = ["git", "worktree", "remove"]
        if force:
            cmd.append("--force")
        cmd.append(str(wt_path))
        try:
            subprocess.run(cmd, cwd=str(repo_dir), capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError:
            pass

        subprocess.run(
            ["git", "worktree", "prune"],
            cwd=str(repo_dir),
            capture_output=True,
            text=True,
        )

        if wt_path.exists():
            shutil.rmtree(wt_path, ignore_errors=True)

        if delete_branch and branch_to_delete:
            try:
                subprocess.run(
                    ["git", "branch", "-D", branch_to_delete],
                    cwd=str(repo_dir),
                    capture_output=True,
                    text=True,
                    check=True,
                )
            except subprocess.CalledProcessError:
                pass

    @classmethod
    def list_worktrees(cls, repo_dir: str | Path) -> list[dict[str, str]]:
        repo_dir = Path(repo_dir).resolve()
        cmd = ["git", "worktree", "list", "--porcelain"]
        proc = subprocess.run(
            cmd, cwd=str(repo_dir), capture_output=True, text=True, check=True
        )

        worktrees: list[dict[str, str]] = []
        current: dict[str, str] = {}
        for line in proc.stdout.splitlines():
            line = line.strip()
            if not line:
                if current:
                    worktrees.append(current)
                    current = {}
                continue
            parts = line.split(" ", 1)
            if len(parts) == 2:
                current[parts[0]] = parts[1]
            elif len(parts) == 1:
                current[parts[0]] = ""
        if current:
            worktrees.append(current)

        return worktrees

    @classmethod
    @contextmanager
    def isolated_worktree(
        cls,
        repo_dir: str | Path,
        conversation_id: str,
        branch_name: str | None = None,
        base_ref: str = "HEAD",
        force: bool = True,
        delete_branch: bool = True,
    ) -> Iterator[Path]:
        wt_path = cls.create_worktree(
            repo_dir=repo_dir,
            conversation_id=conversation_id,
            branch_name=branch_name,
            base_ref=base_ref,
        )
        try:
            yield wt_path
        finally:
            cls.remove_worktree(
                repo_dir=repo_dir,
                worktree_path=wt_path,
                force=force,
                delete_branch=delete_branch,
            )
