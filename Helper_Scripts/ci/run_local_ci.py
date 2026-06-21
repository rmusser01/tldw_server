#!/usr/bin/env python3
"""Run the gating CI checks locally, on any of linux/macOS/Windows.

This mirrors the *blocking* lanes of ``.github/workflows/ci.yml`` so you can get
the same signal locally and skip waiting on the remote GitHub runners. It is pure
standard library and invokes tools through ``sys.executable -m ...`` so it works
identically on a Windows Server box (no bash required).

Tiers
-----
``--fast`` (default)
    compileall + ruff (changed files) + repo guards + pytest on the changed test
    files only. Sub-couple-of-minutes feedback loop for the "did I break what I
    touched" question.
``--full``
    compileall + ruff (whole backend) + guards + the whole pytest suite under
    ``-n auto`` (pytest-xdist, already a project dependency). The local stand-in
    for the remote full suite.
``--lane PATH [PATH ...]``
    compileall + guards + pytest on the given path(s) under ``-n auto``. Use this
    to run one area (e.g. a directory matching a CI shard).

Mapping to CI jobs
------------------
- compileall            -> ``syntax-check`` job (compileall over app/)
- ruff                  -> ``lint`` job (ruff check tldw_Server_API/)
- guards                -> pre-commit local hooks (http-client patch / legacy body / syntax)
- pytest                -> ``full-suite-*`` shard jobs

Postgres / Redis dependent tests honor the same env the test suite already uses:
set ``TEST_DATABASE_URL`` to point at a running Postgres, or
``TLDW_TEST_NO_DOCKER=1`` to skip the autostart. See
``Helper_Scripts/Testing-related/start_postgres_for_tests.sh`` to spin one up.

Examples
--------
    python Helper_Scripts/ci/run_local_ci.py            # fast tier
    python Helper_Scripts/ci/run_local_ci.py --full
    python Helper_Scripts/ci/run_local_ci.py --lane tldw_Server_API/tests/Security
    python Helper_Scripts/ci/run_local_ci.py --base origin/dev
    python Helper_Scripts/ci/run_local_ci.py --full --pytest-args "-k embeddings"
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

# Tool versions the remote CI pins (see the `lint` job). Local installs may
# differ; we only warn so the script stays usable without an exact match.
CI_RUFF_VERSION = "0.15.10"
CI_MYPY_VERSION = "1.20.1"

APP_DIR = "tldw_Server_API/app"
TESTS_DIR = "tldw_Server_API/tests"
RUFF_TARGET = "tldw_Server_API/"

GUARD_SCRIPTS = (
    "Helper_Scripts/checks/guard_http_client_patching.py",
    "Helper_Scripts/checks/guard_no_nonempty_legacy_complete.py",
)
SYNTAX_GUARD = "Helper_Scripts/checks/check_python_syntax.py"


@dataclass
class PhaseResult:
    name: str
    ok: bool
    seconds: float
    skipped: bool = False
    note: str = ""


@dataclass
class Context:
    repo_root: Path
    base: str | None
    changed_py: list[str] = field(default_factory=list)
    changed_tests: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _c(text: str, color: str) -> str:
    if not sys.stdout.isatty() or os.environ.get("NO_COLOR"):
        return text
    codes = {"green": "32", "red": "31", "yellow": "33", "cyan": "36", "bold": "1"}
    return f"\033[{codes.get(color, '0')}m{text}\033[0m"


def _run(cmd: Sequence[str], cwd: Path) -> int:
    print(_c("$ " + " ".join(cmd), "cyan"))
    try:
        return subprocess.run(list(cmd), cwd=str(cwd)).returncode
    except FileNotFoundError as exc:  # missing executable
        print(_c(f"  command not found: {exc}", "red"), file=sys.stderr)
        return 127


def _capture(cmd: Sequence[str], cwd: Path) -> tuple[int, str]:
    try:
        proc = subprocess.run(
            list(cmd), cwd=str(cwd), capture_output=True, text=True
        )
        return proc.returncode, proc.stdout
    except FileNotFoundError:
        return 127, ""


def _venv_python(repo_root: Path) -> Path | None:
    """Return the repo's .venv interpreter path if it exists, else None."""
    candidates = [
        repo_root / ".venv" / "bin" / "python",          # POSIX
        repo_root / ".venv" / "Scripts" / "python.exe",  # Windows
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def _maybe_reexec_into_venv(repo_root: Path) -> None:
    """Re-exec under the project venv so we never silently use system Python.

    Honors the project rule of not running against the system interpreter. A
    guard env var prevents infinite re-exec loops, and ``TLDW_CI_NO_REEXEC=1``
    opts out entirely (e.g. when the caller already manages the environment).
    """
    if os.environ.get("TLDW_CI_REEXEC") == "1" or os.environ.get("TLDW_CI_NO_REEXEC") == "1":
        return
    venv_py = _venv_python(repo_root)
    if venv_py is None:
        return
    try:
        same = os.path.samefile(str(venv_py), sys.executable)
    except OSError:
        same = Path(sys.executable).resolve() == venv_py.resolve()
    if same:
        return
    print(_c(f"Re-executing under project venv: {venv_py}", "cyan"))
    env = dict(os.environ, TLDW_CI_REEXEC="1")
    os.execve(str(venv_py), [str(venv_py), str(Path(__file__).resolve()), *sys.argv[1:]], env)


def _git_repo_root() -> Path:
    rc, out = _capture(["git", "rev-parse", "--show-toplevel"], Path.cwd())
    if rc != 0 or not out.strip():
        print(_c("Not inside a git repository.", "red"), file=sys.stderr)
        raise SystemExit(2)
    return Path(out.strip())


def _resolve_base(repo_root: Path, explicit: str | None) -> str | None:
    """Pick a base ref to diff against for 'changed files' detection."""
    candidates = []
    if explicit:
        candidates.append(explicit)
    if os.environ.get("TLDW_CI_BASE"):
        candidates.append(os.environ["TLDW_CI_BASE"])
    candidates += ["origin/dev", "origin/main", "HEAD~1"]
    for ref in candidates:
        rc, out = _capture(["git", "merge-base", "HEAD", ref], repo_root)
        if rc == 0 and out.strip():
            return out.strip()
    return None


def _changed_python(repo_root: Path, base: str | None) -> list[str]:
    files: set[str] = set()
    if base:
        rc, out = _capture(
            ["git", "diff", "--name-only", "--diff-filter=ACMRTUXB", f"{base}...HEAD", "--", "*.py"],
            repo_root,
        )
        if rc == 0:
            files.update(line for line in out.splitlines() if line.strip())
    # Always include working-tree + untracked changes so local edits count.
    rc, out = _capture(
        ["git", "diff", "--name-only", "--diff-filter=ACMRTUXB", "--", "*.py"], repo_root
    )
    if rc == 0:
        files.update(line for line in out.splitlines() if line.strip())
    rc, out = _capture(
        ["git", "ls-files", "--others", "--exclude-standard", "--", "*.py"], repo_root
    )
    if rc == 0:
        files.update(line for line in out.splitlines() if line.strip())
    # Keep only files that still exist (skip deletions).
    return sorted(f for f in files if (repo_root / f).exists())


def _is_test_file(path: str) -> bool:
    """True if ``path`` is a pytest-collectable test module under the tests dir."""
    if not path.startswith(TESTS_DIR):
        return False
    name = Path(path).name
    return name.startswith("test_") or name.endswith("_test.py")


def _py(*args: str) -> list[str]:
    return [sys.executable, *args]


def _check_tool_version(repo_root: Path, module: str, expected: str) -> None:
    rc, out = _capture(_py("-m", module, "--version"), repo_root)
    if rc != 0:
        return
    if expected not in out:
        got = out.strip().splitlines()[0] if out.strip() else "unknown"
        print(
            _c(f"  note: local {module} ({got}) != CI pin {expected}; results may differ.", "yellow")
        )


# --------------------------------------------------------------------------- #
# Phases
# --------------------------------------------------------------------------- #
def phase_compileall(ctx: Context) -> PhaseResult:
    start = time.time()
    rc = _run(_py("-m", "compileall", "-q", APP_DIR), ctx.repo_root)
    return PhaseResult("compileall (syntax-check)", rc == 0, time.time() - start)


def phase_ruff(ctx: Context, full: bool) -> PhaseResult:
    # Non-blocking, mirroring the CI `lint` job (continue-on-error against a large
    # baseline). Reported for visibility but never fails the local run.
    start = time.time()
    if shutil.which("ruff") is None:
        rc, _ = _capture(_py("-m", "ruff", "--version"), ctx.repo_root)
        if rc != 0:
            return PhaseResult("ruff (non-blocking)", True, time.time() - start, skipped=True,
                               note="ruff not installed (pip install ruff)")
    _check_tool_version(ctx.repo_root, "ruff", CI_RUFF_VERSION)
    if full:
        targets = [RUFF_TARGET]
    else:
        if not ctx.changed_py:
            return PhaseResult("ruff (non-blocking)", True, time.time() - start, skipped=True,
                               note="no changed .py files")
        targets = ctx.changed_py
    rc = _run(_py("-m", "ruff", "check", *targets), ctx.repo_root)
    note = "" if rc == 0 else "lint findings reported (informational, like CI)"
    return PhaseResult("ruff (non-blocking)", True, time.time() - start, note=note)


def phase_guards(ctx: Context) -> PhaseResult:
    start = time.time()
    ok = True
    for guard in GUARD_SCRIPTS:
        if (ctx.repo_root / guard).exists():
            rc = _run(_py(guard), ctx.repo_root)
            ok = ok and rc == 0
    # Syntax guard takes file paths; run on changed files (or app/ in full runs).
    syntax_targets = ctx.changed_py or [APP_DIR]
    if (ctx.repo_root / SYNTAX_GUARD).exists():
        rc = _run(_py(SYNTAX_GUARD, *syntax_targets), ctx.repo_root)
        ok = ok and rc == 0
    return PhaseResult("repo guards", ok, time.time() - start)


def _pytest_base_cmd(jobs: str) -> list[str]:
    cmd = _py("-m", "pytest", "-q", "--disable-warnings", "-p", "no:cacheprovider")
    if jobs and jobs != "0":
        cmd += ["-n", jobs]
    return cmd


def phase_pytest(ctx: Context, paths: list[str], jobs: str, extra: list[str]) -> PhaseResult:
    start = time.time()
    if not paths:
        return PhaseResult("pytest", True, time.time() - start, skipped=True,
                           note="no test paths selected (touch a test file or use --lane/--full)")
    cmd = _pytest_base_cmd(jobs) + extra + paths
    rc = _run(cmd, ctx.repo_root)
    return PhaseResult("pytest", rc == 0, time.time() - start)


def phase_mypy(ctx: Context) -> PhaseResult:
    start = time.time()
    _check_tool_version(ctx.repo_root, "mypy", CI_MYPY_VERSION)
    rc = _run(_py("-m", "mypy", RUFF_TARGET), ctx.repo_root)
    # mypy is non-blocking in CI (baseline backlog); report but never fail the run.
    return PhaseResult("mypy (non-blocking)", True, time.time() - start,
                       note="" if rc == 0 else "type errors reported (informational)")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the gating CI checks locally (compileall, ruff, guards, pytest).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    tier = p.add_mutually_exclusive_group()
    tier.add_argument("--fast", action="store_true",
                      help="Default. compileall + ruff(changed) + guards + pytest(changed tests).")
    tier.add_argument("--full", action="store_true",
                      help="compileall + ruff(all) + guards + whole pytest suite under -n auto.")
    tier.add_argument("--lane", nargs="+", metavar="PATH",
                      help="compileall + guards + pytest on the given path(s) under -n auto.")
    p.add_argument("--base", help="Git ref to diff against for changed-file detection.")
    p.add_argument("--jobs", default="auto",
                   help="pytest-xdist worker count ('auto', a number, or '0' to disable). Default: auto.")
    p.add_argument("--mypy", action="store_true", help="Also run mypy (non-blocking, like CI).")
    p.add_argument("--no-pytest", action="store_true", help="Skip the pytest phase.")
    p.add_argument("--pytest-args", default="",
                   help="Extra args appended to pytest, e.g. --pytest-args \"-k embeddings\".")
    p.add_argument("--list-changed", action="store_true",
                   help="Print detected changed Python files and exit.")
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    repo_root = _git_repo_root()
    _maybe_reexec_into_venv(repo_root)  # never silently use system Python
    base = _resolve_base(repo_root, args.base)
    changed = _changed_python(repo_root, base)
    changed_tests = [f for f in changed if _is_test_file(f)]
    ctx = Context(repo_root=repo_root, base=base, changed_py=changed, changed_tests=changed_tests)

    if args.list_changed:
        print(f"base: {base or '(working tree only)'}")
        for f in changed:
            print(f"  {f}")
        return 0

    full = bool(args.full)
    lane = args.lane
    extra = args.pytest_args.split() if args.pytest_args else []

    print(_c("=" * 70, "bold"))
    tier_name = "lane" if lane else ("full" if full else "fast")
    print(_c(f"Local CI — tier: {tier_name}   base: {base or '(working tree only)'}", "bold"))
    print(_c(f"changed .py: {len(changed)}   changed tests: {len(changed_tests)}", "bold"))
    print(_c("=" * 70, "bold"))

    results: list[PhaseResult] = []
    results.append(phase_compileall(ctx))
    if not lane:  # ruff covered by lint job; lane runs skip it for speed
        results.append(phase_ruff(ctx, full=full))
    results.append(phase_guards(ctx))
    if args.mypy:
        results.append(phase_mypy(ctx))

    if not args.no_pytest:
        if lane:
            pytest_paths = list(lane)
        elif full:
            pytest_paths = [TESTS_DIR]
        else:
            pytest_paths = changed_tests
        results.append(phase_pytest(ctx, pytest_paths, args.jobs, extra))

    # Summary
    print()
    print(_c("-" * 70, "bold"))
    print(_c("Summary", "bold"))
    failed = False
    for r in results:
        if r.skipped:
            status = _c("SKIP", "yellow")
        elif r.ok:
            status = _c("PASS", "green")
        else:
            status = _c("FAIL", "red")
            failed = True
        note = f"  ({r.note})" if r.note else ""
        print(f"  {status}  {r.name:<28} {r.seconds:6.1f}s{note}")
    print(_c("-" * 70, "bold"))

    if failed:
        print(_c("Local CI FAILED — fix the above before pushing.", "red"))
        return 1
    print(_c("Local CI passed.", "green"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
