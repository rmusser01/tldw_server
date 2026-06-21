"""Regression tests for the local CI runner."""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path

import pytest
from Helper_Scripts.ci import run_local_ci


def _git(repo: Path, *args: str) -> str:
    """Run a git command in ``repo`` and return stdout."""
    proc = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return proc.stdout.strip()


def _phase(name: str):
    """Return a passing phase stub for ``run_local_ci.main`` tests."""

    def stub(*_args, **_kwargs) -> run_local_ci.PhaseResult:
        return run_local_ci.PhaseResult(name, True, 0.0)

    return stub


def _has_arg_pair(cmd: list[str], left: str, right: str) -> bool:
    """Return True when adjacent arguments are present in order."""
    return any(cmd[index : index + 2] == [left, right] for index in range(len(cmd) - 1))


def test_pytest_args_preserve_quoted_expressions(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Quoted ``--pytest-args`` expressions are passed to pytest as one argument."""
    captured: dict[str, list[str]] = {}

    def fake_phase_pytest(
        _ctx: run_local_ci.Context,
        _paths: list[str],
        _jobs: str,
        extra: list[str],
    ) -> run_local_ci.PhaseResult:
        captured["extra"] = extra
        return run_local_ci.PhaseResult("pytest", True, 0.0)

    monkeypatch.setattr(run_local_ci, "_git_repo_root", lambda: tmp_path)
    monkeypatch.setattr(run_local_ci, "_maybe_reexec_into_venv", lambda _repo_root: None)
    monkeypatch.setattr(run_local_ci, "_resolve_base", lambda _repo_root, _explicit: None)
    monkeypatch.setattr(
        run_local_ci,
        "_changed_python",
        lambda _repo_root, _base: ["tldw_Server_API/tests/CI/test_example.py"],
    )
    monkeypatch.setattr(run_local_ci, "phase_compileall", _phase("compileall"))
    monkeypatch.setattr(run_local_ci, "phase_ruff", _phase("ruff"))
    monkeypatch.setattr(run_local_ci, "phase_guards", _phase("guards"))
    monkeypatch.setattr(run_local_ci, "phase_pytest", fake_phase_pytest)

    rc = run_local_ci.main(["--pytest-args", "-k 'alpha and beta'"])

    assert rc == 0
    assert captured["extra"] == ["-k", "alpha and beta"]


def test_changed_python_filters_git_output_in_python(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Changed-file detection filters Python files after collecting git output."""
    seen_commands: list[list[str]] = []

    def fake_capture(cmd: list[str], _cwd: Path) -> tuple[int, str]:
        seen_commands.append(cmd)
        if cmd[1] == "diff":
            return 0, "nested/changed.py\nREADME.md\nscripts/tool.py\n"
        if cmd[1] == "ls-files":
            return 0, "notes.txt\nuntracked/new_test.py\n"
        return 1, ""

    for rel_path in ("nested/changed.py", "scripts/tool.py", "untracked/new_test.py"):
        path = tmp_path / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("print('ok')\n")

    monkeypatch.setattr(run_local_ci, "_capture", fake_capture)

    changed = run_local_ci._changed_python(tmp_path, "base-ref")

    assert changed == ["nested/changed.py", "scripts/tool.py", "untracked/new_test.py"]
    assert all("--" not in cmd for cmd in seen_commands)


def test_changed_python_detects_nested_committed_files(tmp_path: Path) -> None:
    """Nested Python files committed after the base ref are detected."""
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "tests@example.invalid")
    _git(tmp_path, "config", "user.name", "Tests")
    (tmp_path / "README.md").write_text("base\n")
    _git(tmp_path, "add", "README.md")
    _git(tmp_path, "commit", "-q", "-m", "base")
    base = _git(tmp_path, "rev-parse", "HEAD")

    nested = tmp_path / "tldw_Server_API" / "app" / "nested" / "changed.py"
    nested.parent.mkdir(parents=True)
    nested.write_text("VALUE = 1\n")
    _git(tmp_path, "add", str(nested.relative_to(tmp_path)))
    _git(tmp_path, "commit", "-q", "-m", "changed")

    assert run_local_ci._changed_python(tmp_path, base) == [
        "tldw_Server_API/app/nested/changed.py"
    ]


def test_phase_pytest_uses_ci_like_env_and_explicit_xdist(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Pytest runs with CI-like environment defaults and explicit xdist loading."""
    captured: dict[str, object] = {}

    def fake_run(cmd: list[str], cwd: Path, env: dict[str, str] | None = None) -> int:
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        captured["env"] = env
        return 0

    for key in run_local_ci.CI_PYTEST_ENV_DEFAULTS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(run_local_ci, "_run", fake_run)

    result = run_local_ci.phase_pytest(
        run_local_ci.Context(repo_root=tmp_path, base=None),
        [run_local_ci.TESTS_DIR],
        "auto",
        [],
    )

    assert result.ok is True
    cmd = captured["cmd"]
    assert isinstance(cmd, list)
    assert _has_arg_pair(cmd, "-p", "xdist.plugin")
    assert "-n" in cmd
    env = captured["env"]
    assert isinstance(env, dict)
    assert env["PYTHONPATH"] == "."
    assert env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] == "1"
    assert env["TEST_MODE"] == "true"
    assert env["DISABLE_HEAVY_STARTUP"] == "1"
    assert env["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] == "python"


def test_phase_pytest_does_not_load_xdist_when_jobs_disabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Passing ``--jobs 0`` disables xdist arguments while keeping CI env defaults."""
    captured: dict[str, object] = {}

    def fake_run(cmd: list[str], cwd: Path, env: dict[str, str] | None = None) -> int:
        captured["cmd"] = cmd
        captured["env"] = env
        return 0

    monkeypatch.setattr(run_local_ci, "_run", fake_run)

    run_local_ci.phase_pytest(
        run_local_ci.Context(repo_root=tmp_path, base=None),
        [run_local_ci.TESTS_DIR],
        "0",
        [],
    )

    cmd = captured["cmd"]
    assert isinstance(cmd, list)
    assert "xdist.plugin" not in cmd
    assert "-n" not in cmd


def test_windows_reexec_waits_and_exits_with_child_status(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Windows venv re-exec waits for the child process and returns its status."""
    venv_python = tmp_path / ".venv" / "Scripts" / "python.exe"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("")
    captured: dict[str, object] = {}

    def fake_call(cmd: list[str], env: dict[str, str]) -> int:
        captured["cmd"] = cmd
        captured["env"] = env
        return 23

    monkeypatch.delenv("TLDW_CI_REEXEC", raising=False)
    monkeypatch.delenv("TLDW_CI_NO_REEXEC", raising=False)
    monkeypatch.setattr(run_local_ci, "_venv_python", lambda _repo_root: venv_python)
    monkeypatch.setattr(run_local_ci.os.path, "samefile", lambda _left, _right: False)
    monkeypatch.setattr(run_local_ci.sys, "platform", "win32")
    monkeypatch.setattr(run_local_ci.sys, "argv", ["run_local_ci.py"])
    monkeypatch.setattr(run_local_ci.subprocess, "call", fake_call)
    monkeypatch.setattr(
        run_local_ci.os,
        "execve",
        lambda *_args, **_kwargs: pytest.fail("Windows re-exec must not use os.execve"),
    )

    with pytest.raises(SystemExit) as exc:
        run_local_ci._maybe_reexec_into_venv(tmp_path)

    assert exc.value.code == 23
    assert captured["cmd"] == [
        str(venv_python),
        str(Path(run_local_ci.__file__).resolve()),
    ]
    env = captured["env"]
    assert isinstance(env, dict)
    assert env["TLDW_CI_REEXEC"] == "1"


def test_runner_owned_messages_use_loguru_instead_of_prints() -> None:
    """The runner does not use direct print calls for its own messages."""
    tree = ast.parse(Path(run_local_ci.__file__).read_text())
    print_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "print"
    ]

    assert print_calls == []


def test_runner_classes_and_functions_have_docstrings() -> None:
    """New runner classes and functions provide docstrings for maintainability."""
    names = [
        "PhaseResult",
        "Context",
        "_c",
        "_emit",
        "_emit_error",
        "_run",
        "_capture",
        "_venv_python",
        "_maybe_reexec_into_venv",
        "_git_repo_root",
        "_resolve_base",
        "_changed_python",
        "_is_test_file",
        "_py",
        "_check_tool_version",
        "phase_compileall",
        "phase_ruff",
        "phase_guards",
        "_pytest_base_cmd",
        "_ci_pytest_env",
        "phase_pytest",
        "phase_mypy",
        "parse_args",
        "main",
    ]

    missing = [name for name in names if not getattr(run_local_ci, name).__doc__]

    assert missing == []
