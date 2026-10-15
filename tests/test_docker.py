"""R8 container files, checked without Docker (docs/plan.md W3.4): the build context never carries secrets or run
output, and the image installs exactly the locked dependencies (spec §4: pinned, CPU-only Jobs container)."""
from __future__ import annotations

import json
import posixpath
import re
import shlex
from pathlib import Path

import pytest

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # Python 3.10: tomli is a locked pytest dependency there
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]
DOCKER = ROOT / "docker"
# D49: the PyPI wheel the container installs (development imports the editable checkout instead).
MANIPULAPY_WHEEL_SHA256 = "611c2aa0708cc965b2f681d7a324e97f74578a77cec6cef9bea47be3d376cb10"


# --- .dockerignore, read the way the Docker CLI reads it (moby/patternmatcher) ---------------------------------

def _glob_regex(pattern: str) -> re.Pattern[str]:
    """'*' and '?' stop at '/', '**' spans zero or more directories, '[...]' classes pass through."""
    out, i = "", 0
    while i < len(pattern):
        if pattern.startswith("**", i):
            i += 2
            if pattern.startswith("/", i):
                i += 1
            out += ".*" if i == len(pattern) else "(.*/)?"
            continue
        ch = pattern[i]
        if ch == "*":
            out += "[^/]*"
        elif ch == "?":
            out += "[^/]"
        elif ch in "[]":
            out += ch
        elif ch == "\\" and i + 1 < len(pattern):
            i += 1
            out += re.escape(pattern[i])
        else:
            out += re.escape(ch)
        i += 1
    return re.compile(out)


def dockerignore_rules(text: str) -> list[tuple[bool, re.Pattern[str]]]:
    """(is_exception, regex) per pattern line; patterns are cleaned and made context-relative."""
    rules = []
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        line = line.strip()
        if not line:
            continue
        exception = line.startswith("!")
        pattern = posixpath.normpath(line[1:].strip() if exception else line).lstrip("/")
        rules.append((exception, _glob_regex(pattern)))
    return rules


def excluded(path: str, rules: list[tuple[bool, re.Pattern[str]]]) -> bool:
    """The last rule that matches the path or one of its parent directories decides."""
    parts = path.split("/")
    candidates = ["/".join(parts[:i + 1]) for i in range(len(parts))]
    verdict = False
    for exception, regex in rules:
        if any(regex.fullmatch(c) for c in candidates):
            verdict = not exception
    return verdict


@pytest.fixture(scope="module")
def rules() -> list[tuple[bool, re.Pattern[str]]]:
    return dockerignore_rules((ROOT / ".dockerignore").read_text(encoding="utf-8"))


def test_matcher_follows_docker_semantics() -> None:  # guards the helper the context tests rely on
    def ex(path: str, text: str) -> bool:
        return excluded(path, dockerignore_rules(text))

    assert ex(".git/HEAD", "*")  # '*' matches dot files, and a matched directory excludes its contents
    assert ex("a.py", "*.py") and not ex("pkg/a.py", "*.py")  # '*' never crosses '/'
    assert ex("__pycache__/x.pyc", "**/__pycache__") and ex("a/b/__pycache__/x.pyc", "**/__pycache__")
    assert ex("docs/x.md", "/docs") and ex("runs/a/b", "runs/")  # leading '/' and trailing '/' are cleaned away
    assert not ex("src/x.py", "*\n!src/") and ex("src/x.py", "*\n!src/\nsrc/x.py")  # the last match decides
    assert not ex("docs/keep.md", "docs\n!docs/keep.md") and ex("docs/other.md", "docs\n!docs/keep.md")
    assert not ex("x.py", "# x.py\n")  # comment line


@pytest.mark.parametrize("path", [
    ".env", ".env.local", ".env.nebius", "tests/.env", "config/.env.prod", "src/vplan/.env",
    "runs", "runs/r8-local/20261014T000000Z/cohort/manifest.json",
    ".git", ".git/HEAD", ".claude/settings.local.json", ".claude/worktrees/w3-docker/.env",
    ".superpowers/sdd/plan/progress.md", ".venv/bin/python", "tests/.venv/bin/python",
    "__pycache__/x.cpython-310.pyc", "src/vplan/__pycache__/cli.cpython-310.pyc", "tests/conftest.pyc",
    ".pytest_cache/v/cache/nodeids", "src/vplan.egg-info/PKG-INFO", "docs/decisions.md", "README.md",
])
def test_dockerignore_keeps_secrets_and_local_state_out_of_the_build_context(rules, path) -> None:
    assert excluded(path, rules)


def test_dockerignore_spares_the_env_example(rules) -> None:
    assert not excluded(".env.example", rules)


def test_the_spec_the_metrics_test_reads_reaches_the_image(rules) -> None:
    # tests/test_metrics.py parses docs/spec.md §14; the rest of docs/ stays out (see the list above).
    assert not excluded("docs/spec.md", rules)
    assert "docs/spec.md docs/spec.md" in _args("COPY")


def test_package_tests_and_configs_reach_the_build_context(rules) -> None:
    files = [p for d in ("src", "tests", "scripts") for p in (ROOT / d).rglob("*.py")]
    files += list((ROOT / "config").glob("*.json"))
    assert files
    assert [p for p in files if excluded(p.relative_to(ROOT).as_posix(), rules)] == []


# --- Dockerfile -------------------------------------------------------------------------------------------------

def dockerfile_instructions() -> list[tuple[str, str]]:
    """(INSTRUCTION, arguments) with comment lines dropped and backslash continuations joined."""
    out, current = [], ""
    for raw in (DOCKER / "Dockerfile").read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("#") or (not line and not current):
            continue
        if line.endswith("\\"):
            current += line[:-1] + " "
            continue
        keyword, _, args = (current + line).partition(" ")
        out.append((keyword.upper(), args.strip()))
        current = ""
    return out


def _args(keyword: str) -> list[str]:
    return [args for k, args in dockerfile_instructions() if k == keyword]


def test_dockerfile_pins_the_base_image_by_exact_tag_and_digest() -> None:
    (base,) = _args("FROM")
    assert re.fullmatch(r"python:3\.10\.\d+-slim(-[a-z]+)?@sha256:[0-9a-f]{64}", base), base


def test_dockerfile_installs_only_the_hashed_requirements() -> None:
    (install,) = [a for a in _args("RUN") if "pip install" in a]
    tokens = shlex.split(install)
    assert {"--require-hashes", "--no-deps", "--only-binary=:all:"} <= set(tokens)
    assert tokens[tokens.index("-r") + 1] == "docker/requirements.txt"


def test_dockerfile_runs_as_a_non_root_user_with_utf8_and_a_quiet_manipulapy() -> None:
    env = dict(token.partition("=")[::2] for args in _args("ENV") for token in shlex.split(args))
    assert env["PYTHONUTF8"] == "1" and env["MANIPULAPY_QUIET"] == "1"
    user = _args("USER")[-1].split(":")[0]
    assert user not in ("", "root", "0")
    assert json.loads(_args("ENTRYPOINT")[-1]) == ["python", "-m", "vplan"]


def test_every_copied_path_exists_and_is_in_the_build_context(rules) -> None:
    sources = [token for args in _args("COPY") for token in shlex.split(args)[:-1] if not token.startswith("--")]
    assert sources
    for source in sources:
        path = posixpath.normpath(source)
        assert (ROOT / path).exists(), source
        assert not excluded(path, rules), source


# --- pins: docker/requirements.txt against uv.lock and pyproject.toml ----------------------------------------

def _normalize(name: str) -> str:  # PEP 503
    return re.sub(r"[-_.]+", "-", name).lower()


def image_requirements() -> dict[str, tuple[str, frozenset[str]]]:
    """name -> (version, sha256 set) from the uv export the image installs."""
    text = re.sub(r"\\\r?\n", " ", (DOCKER / "requirements.txt").read_text(encoding="utf-8"))
    pins = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.fullmatch(r"([A-Za-z0-9][A-Za-z0-9._-]*)==(\S+?)(?:\s*;.*?)?((?:\s+--hash=sha256:[0-9a-f]{64})+)", line)
        assert m, f"not an exact, hashed pin: {line}"
        pins[_normalize(m.group(1))] = (m.group(2), frozenset(re.findall(r"sha256:([0-9a-f]{64})", m.group(3))))
    return pins


def locked_packages() -> dict[str, tuple[str, frozenset[str]]]:
    """name -> (version, sha256 set of every sdist and wheel) for each third-party package in uv.lock."""
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    out = {}
    for pkg in lock["package"]:
        if {"virtual", "editable"} & set(pkg.get("source", {})):
            continue  # the vplan project itself
        dists = ([pkg["sdist"]] if "sdist" in pkg else []) + pkg.get("wheels", [])
        out[_normalize(pkg["name"])] = (pkg["version"], frozenset(d["hash"].split(":", 1)[1] for d in dists))
    return out


def pyproject_pins() -> dict[str, str]:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    pins = {}
    for spec in project["dependencies"] + project["optional-dependencies"]["dev"]:
        m = re.fullmatch(r"([A-Za-z0-9][A-Za-z0-9._-]*)(?:\[[^\]]*\])?==(\S+)", spec)
        assert m, f"not an exact pin (spec §4 pins every dependency): {spec}"
        pins[_normalize(m.group(1))] = m.group(2)
    return pins


def test_image_requirements_are_exactly_uv_lock() -> None:
    # After any lock change, re-export (the dev extra brings pytest, which the R8 check runs in the image):
    #   uv export --frozen --format requirements.txt --extra dev --no-emit-project -o docker/requirements.txt
    assert image_requirements() == locked_packages()


def test_image_requirements_honor_every_pyproject_pin() -> None:
    reqs = image_requirements()
    for name, version in pyproject_pins().items():
        assert reqs[name][0] == version, name


def test_image_installs_the_manipulapy_1_4_2_wheel_from_pypi() -> None:  # D49
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert "ManipulaPy[simulation]==1.4.2" in project["dependencies"]
    version, hashes = image_requirements()["manipulapy"]
    assert version == "1.4.2" and MANIPULAPY_WHEEL_SHA256 in hashes
    assert image_requirements()["pybullet"][0] == "3.2.7"  # the [simulation] extra


# --- run-local.sh -----------------------------------------------------------------------------------------------

def test_run_local_is_strict_bash_and_never_passes_credentials() -> None:
    text = (DOCKER / "run-local.sh").read_text(encoding="utf-8")
    assert text.startswith("#!/usr/bin/env bash\n") and "\nset -euo pipefail\n" in text
    assert "--env-file" not in text and "API_KEY" not in text
    mounts = re.findall(r"--(?:volume|mount)[ =](\S+)", text)  # the script spells docker flags out in full
    assert mounts and all(m.strip("\"'").endswith(":/out") for m in mounts), mounts
    assert "VPLAN_IMAGE_DIGEST=" in text


def test_run_local_passes_the_host_git_sha_to_both_containers() -> None:
    # The image has neither git nor .git, so manifests made inside it take the SHA from the environment (gitinfo).
    text = (DOCKER / "run-local.sh").read_text(encoding="utf-8")
    assert '"VPLAN_GIT_SHA=${sha}"' in text and '"VPLAN_GIT_DIRTY=${dirty}"' in text
    runs = [line for line in text.splitlines() if line.lstrip().startswith("docker run ")]
    assert len(runs) == 2 and all('${git_env[@]+"${git_env[@]}"}' in line for line in runs), runs
