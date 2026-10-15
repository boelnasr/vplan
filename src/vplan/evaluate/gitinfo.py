"""Git provenance for manifests (spec §14 "git SHA"), never silently null: {"head", "dirty", "source"} or None.

Sources, first that answers:
- "git": `git rev-parse HEAD` and `git status --porcelain` in the checkout.
- "files": HEAD read from the repository files. Windows git, called from the Windows interpreter, cannot follow a
  worktree's .git file whose gitdir is a WSL path (/mnt/c/...), so every manifest made in a git worktree had a null
  SHA (D53). The .git file's gitdir (translated to C:/... on Windows) and its commondir are followed, and HEAD is
  resolved through loose refs and packed-refs. dirty is None: it cannot be known without git.
- "env": VPLAN_GIT_SHA (and VPLAN_GIT_DIRTY: true/false/1/0) from the environment given, for the container, which
  has neither git nor .git (docker/run-local.sh sets them from the host checkout). Callers pass the environment only
  for the vplan checkout, so another checkout (ManipulaPy) never borrows its SHA.
"""
from __future__ import annotations

import os
import re
import subprocess
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

SHA = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")  # SHA-1 or SHA-256 object names
ENV_SHA, ENV_DIRTY = "VPLAN_GIT_SHA", "VPLAN_GIT_DIRTY"
_DIRTY_WORDS = {"true": True, "1": True, "false": False, "0": False}
# Refs git keeps per worktree (in the worktree's own gitdir); every other ref lives in the common directory.
_PER_WORKTREE = ("refs/worktree/", "refs/bisect/", "refs/rewritten/")
_WSL_DRIVE = re.compile(r"/mnt/([A-Za-z])(/.*)?")

Run = Callable[..., "subprocess.CompletedProcess[str]"]


def wsl_to_windows(path: str) -> str:
    """/mnt/c/x -> C:/x; anything else unchanged."""
    m = _WSL_DRIVE.fullmatch(path)
    return f"{m[1].upper()}:{m[2] or '/'}" if m else path


def _resolve(value: str, base: Path, windows: bool) -> Path:
    path = Path(wsl_to_windows(value) if windows else value)
    return path if path.is_absolute() else base / path


def _git_dirs(start: Path, windows: bool) -> tuple[Path, Path] | None:
    """(gitdir, common dir) of the checkout that contains start, or None."""
    for d in (start, *start.parents):
        dot = d / ".git"
        if dot.is_dir():
            gitdir = dot
            break
        if dot.is_file():
            text = dot.read_text(encoding="utf-8").strip()
            if not text.startswith("gitdir:"):
                return None
            gitdir = _resolve(text[len("gitdir:"):].strip(), d, windows)
            break
    else:
        return None
    commondir = gitdir / "commondir"
    common = _resolve(commondir.read_text(encoding="utf-8").strip(), gitdir, windows) if commondir.is_file() else gitdir
    return gitdir, common


def _read_ref(ref: str, gitdir: Path, common: Path) -> str | None:
    if ".." in ref.split("/"):
        return None  # never read outside the git directory
    loose = (gitdir if ref.startswith(_PER_WORKTREE) else common) / ref
    if loose.is_file():
        return loose.read_text(encoding="utf-8").strip()
    packed = common / "packed-refs"
    if packed.is_file():
        for line in packed.read_text(encoding="utf-8").splitlines():
            if line.startswith(("#", "^")):  # header, and the peeled object of an annotated tag
                continue
            sha, _, name = line.partition(" ")
            if name.strip() == ref:
                return sha.strip()
    return None


def _from_files(cwd: Path, windows: bool) -> dict[str, Any] | None:
    try:
        dirs = _git_dirs(Path(cwd), windows)
        if dirs is None:
            return None
        gitdir, common = dirs
        value: str | None = (gitdir / "HEAD").read_text(encoding="utf-8").strip()
        for _ in range(5):  # HEAD -> branch, and at most a few symbolic refs after it
            if value is None or SHA.fullmatch(value):
                break
            if not value.startswith("ref:"):
                return None
            value = _read_ref(value[len("ref:"):].strip(), gitdir, common)
    except (OSError, UnicodeDecodeError):
        return None
    return {"head": value, "dirty": None, "source": "files"} if value and SHA.fullmatch(value) else None


def _from_git(cwd: Path, run: Run) -> dict[str, Any] | None:
    def git(*args: str) -> str:
        return run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=10, check=True).stdout

    try:
        head = git("rev-parse", "HEAD").strip()
    except (OSError, subprocess.SubprocessError):
        return None
    if not SHA.fullmatch(head):
        return None
    try:
        dirty: bool | None = bool(git("status", "--porcelain").strip())
    except (OSError, subprocess.SubprocessError):
        dirty = None
    return {"head": head, "dirty": dirty, "source": "git"}


def _from_env(env: Mapping[str, str]) -> dict[str, Any] | None:
    head = env.get(ENV_SHA, "").strip().lower()
    if not SHA.fullmatch(head):
        return None  # unset, or e.g. the Dockerfile's "unknown" placeholder
    return {"head": head, "dirty": _DIRTY_WORDS.get(env.get(ENV_DIRTY, "").strip().lower()), "source": "env"}


def git_provenance(cwd: str | Path, *, env: Mapping[str, str] | None = None, windows: bool | None = None,
                   run: Run = subprocess.run) -> dict[str, Any] | None:
    """{"head": SHA, "dirty": bool or None, "source": "git" | "files" | "env"}, or None when no source answers.
    env is consulted last and only when given; windows defaults to this interpreter's platform."""
    cwd = Path(cwd)
    found = _from_git(cwd, run) or _from_files(cwd, os.name == "nt" if windows is None else windows)
    if found is None and env is not None:
        found = _from_env(env)
    return found
