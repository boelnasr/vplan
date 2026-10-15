"""Git provenance that is never silently null (spec §14 "git SHA", D53): the git command, then the repository files
(a worktree whose .git file names a WSL gitdir, which Windows git cannot follow), then VPLAN_GIT_SHA (the container).
Repositories here are hand-written .git trees in tmp_path; the git command is a fake unless a test says otherwise."""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

from vplan.evaluate import gitinfo

SHA_A, SHA_B, SHA_C = "a" * 40, "b" * 40, "c" * 40
HEX = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")


def no_git(cmd, **kwargs):
    raise subprocess.CalledProcessError(128, cmd, stderr="fatal: not a git repository")


def git_says(head: str, status: str | None = ""):
    def run(cmd, **kwargs):
        if cmd[1] == "rev-parse":
            return subprocess.CompletedProcess(cmd, 0, stdout=head + "\n", stderr="")
        if status is None:
            raise subprocess.TimeoutExpired(cmd, 10)
        return subprocess.CompletedProcess(cmd, 0, stdout=status, stderr="")
    return run


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def repo(root: Path, *, branch: str = "main", sha: str | None = SHA_A, packed: str | None = None) -> Path:
    """A plain checkout: root/.git with HEAD on branch, the branch as a loose ref (sha) and/or in packed-refs."""
    write(root / ".git" / "HEAD", f"ref: refs/heads/{branch}\n")
    if sha is not None:
        write(root / ".git" / "refs" / "heads" / branch, sha + "\n")
    if packed is not None:
        write(root / ".git" / "packed-refs", packed)
    return root


def wsl(path: Path) -> str:
    """The WSL spelling of a Windows path (C:\\x\\y -> /mnt/c/x/y), as git in WSL writes it into a worktree's .git."""
    drive, rest = os.path.splitdrive(str(path))
    return f"/mnt/{drive[0].lower()}{rest.replace(os.sep, '/')}"


def worktree(tmp_path: Path, *, gitdir_text: str | None = None, branch: str = "feature") -> tuple[Path, Path]:
    """main/ with a linked worktree wt/ on branch: wt/.git names main/.git/worktrees/wt, whose commondir is ../.."""
    main = repo(tmp_path / "main", sha=SHA_A)
    admin = main / ".git" / "worktrees" / "wt"
    write(admin / "HEAD", f"ref: refs/heads/{branch}\n")
    write(admin / "commondir", "../..\n")
    wt = tmp_path / "wt"
    write(wt / "src" / "x.py", "")
    write(wt / ".git", f"gitdir: {gitdir_text if gitdir_text is not None else admin}\n")
    return main, wt


# ---- path translation ------------------------------------------------------------------------------------------
@pytest.mark.parametrize("given, expected", [
    ("/mnt/c/Users/x/.git/worktrees/w3", "C:/Users/x/.git/worktrees/w3"), ("/mnt/d", "D:/"), ("/mnt/e/", "E:/"),
    ("/home/x/.git", "/home/x/.git"), ("/mnt/cc/x", "/mnt/cc/x"), ("C:/Users/x", "C:/Users/x"),
    ("../main/.git", "../main/.git")])
def test_wsl_paths_become_windows_drive_paths(given, expected):
    assert gitinfo.wsl_to_windows(given) == expected


# ---- the git command -------------------------------------------------------------------------------------------
@pytest.mark.parametrize("status, dirty", [("", False), (" M src/vplan/cli.py\n", True), (None, None)])
def test_the_git_command_comes_first(tmp_path, status, dirty):
    repo(tmp_path, sha=SHA_B)  # the files would say SHA_B; the command wins
    got = gitinfo.git_provenance(tmp_path, run=git_says(SHA_A, status), env={"VPLAN_GIT_SHA": SHA_C})
    assert got == {"head": SHA_A, "dirty": dirty, "source": "git"}


def test_git_output_that_is_not_a_sha_falls_through_to_the_files(tmp_path):
    repo(tmp_path, sha=SHA_B)
    assert gitinfo.git_provenance(tmp_path, run=git_says("HEAD"))["source"] == "files"


def test_a_missing_git_binary_falls_through(tmp_path):
    def missing(cmd, **kwargs):
        raise FileNotFoundError("git")
    repo(tmp_path, sha=SHA_B)
    assert gitinfo.git_provenance(tmp_path, run=missing) == {"head": SHA_B, "dirty": None, "source": "files"}


# ---- the repository files --------------------------------------------------------------------------------------
def test_files_resolve_a_loose_branch_ref_with_dirty_unknown(tmp_path):
    repo(tmp_path, sha=SHA_A)
    assert gitinfo.git_provenance(tmp_path, run=no_git) == {"head": SHA_A, "dirty": None, "source": "files"}


def test_files_resolve_a_packed_ref_and_skip_peeled_lines(tmp_path):
    packed = (f"# pack-refs with: peeled fully-peeled sorted\n{SHA_C} refs/heads/other\n{SHA_B} refs/heads/main\n"
              f"^{SHA_C}\n")
    repo(tmp_path, sha=None, packed=packed)
    assert gitinfo.git_provenance(tmp_path, run=no_git)["head"] == SHA_B


def test_a_loose_ref_wins_over_packed_refs(tmp_path):
    repo(tmp_path, sha=SHA_A, packed=f"{SHA_B} refs/heads/main\n")
    assert gitinfo.git_provenance(tmp_path, run=no_git)["head"] == SHA_A


def test_files_resolve_a_detached_head_from_a_subdirectory(tmp_path):
    repo(tmp_path)
    write(tmp_path / ".git" / "HEAD", SHA_C + "\n")
    write(tmp_path / "src" / "vplan" / "x.py", "")
    assert gitinfo.git_provenance(tmp_path / "src" / "vplan", run=no_git)["head"] == SHA_C


@pytest.mark.parametrize("packed", [False, True])
def test_a_worktree_whose_gitdir_is_a_wsl_path_resolves_through_its_commondir(tmp_path, packed):
    # D53: Windows git could not follow this .git file, so every manifest made in a worktree had a null SHA.
    windows = os.name == "nt"
    admin = tmp_path / "main" / ".git" / "worktrees" / "wt"
    main, wt = worktree(tmp_path, gitdir_text=wsl(admin) if windows else str(admin))
    if packed:
        write(main / ".git" / "packed-refs", f"{SHA_B} refs/heads/feature\n")
    else:
        write(main / ".git" / "refs" / "heads" / "feature", SHA_B + "\n")
    got = gitinfo.git_provenance(wt, run=no_git, windows=windows)
    assert got == {"head": SHA_B, "dirty": None, "source": "files"}


def test_a_worktree_with_a_relative_gitdir_resolves(tmp_path):
    main, wt = worktree(tmp_path, gitdir_text="../main/.git/worktrees/wt")
    write(main / ".git" / "refs" / "heads" / "feature", SHA_C + "\n")
    assert gitinfo.git_provenance(wt, run=no_git)["head"] == SHA_C


@pytest.mark.parametrize("damage", ["no_ref", "bad_dot_git", "outside", "garbage_head", "no_repo"])
def test_unreadable_files_give_no_sha_rather_than_a_wrong_one(tmp_path, damage):
    if damage == "no_repo":
        (tmp_path / "plain").mkdir()
        start = tmp_path / "plain"
    else:
        start = repo(tmp_path / "r", sha=None)
    if damage == "bad_dot_git":
        (start / ".git" / "HEAD").unlink()
        write(start / "sub" / ".git", "not a gitdir line\n")
        start = start / "sub"
    elif damage == "outside":
        write(start / ".git" / "HEAD", "ref: refs/heads/../../../../escape\n")
    elif damage == "garbage_head":
        write(start / ".git" / "HEAD", "not a sha\n")
    assert gitinfo.git_provenance(start, run=no_git) is None


# ---- the environment (the container has no git and no .git) ----------------------------------------------------
@pytest.mark.parametrize("dirty_text, dirty", [("true", True), ("1", True), ("false", False), ("0", False),
                                               ("", None), ("maybe", None)])
def test_the_environment_supplies_the_sha_last(tmp_path, dirty_text, dirty):
    env = {"VPLAN_GIT_SHA": SHA_A.upper(), "VPLAN_GIT_DIRTY": dirty_text}
    assert gitinfo.git_provenance(tmp_path, run=no_git, env=env) == {"head": SHA_A, "dirty": dirty, "source": "env"}


@pytest.mark.parametrize("value", ["unknown", "", SHA_A[:12], SHA_A + "-dirty"])
def test_a_malformed_environment_sha_is_not_a_sha(tmp_path, value):
    assert gitinfo.git_provenance(tmp_path, run=no_git, env={"VPLAN_GIT_SHA": value}) is None


def test_the_environment_is_used_only_when_passed(tmp_path, monkeypatch):
    monkeypatch.setenv("VPLAN_GIT_SHA", SHA_A)
    assert gitinfo.git_provenance(tmp_path, run=no_git) is None  # e.g. the ManipulaPy checkout never borrows it


def test_files_win_over_the_environment(tmp_path):
    repo(tmp_path, sha=SHA_B)
    assert gitinfo.git_provenance(tmp_path, run=no_git, env={"VPLAN_GIT_SHA": SHA_A})["head"] == SHA_B


# ---- this checkout ---------------------------------------------------------------------------------------------
def test_this_checkout_resolves_a_sha_and_names_its_source():
    from vplan.evaluate import pilot
    if not (pilot.ROOT / ".git").exists():
        pytest.skip("no .git here (the container build context drops it)")
    got = pilot._git()
    assert got is not None and HEX.fullmatch(got["head"]) and got["source"] in ("git", "files")
    # The files source never knows dirty; git may not either (a failed or timed-out git status), so only one way.
    assert got["dirty"] is None or got["source"] == "git"


def test_the_repo_helper_reads_the_environment_but_other_checkouts_do_not(tmp_path, monkeypatch):
    from vplan.evaluate import pilot
    monkeypatch.setenv("VPLAN_GIT_SHA", SHA_C)
    monkeypatch.setattr(gitinfo, "_from_git", lambda cwd, run: None)
    monkeypatch.setattr(gitinfo, "_from_files", lambda cwd, windows: None)
    assert pilot._git() == {"head": SHA_C, "dirty": None, "source": "env"}
    assert pilot._git(tmp_path) is None
