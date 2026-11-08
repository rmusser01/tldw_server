"""Qualify committed API artifacts at the existing exporter boundary."""

from __future__ import annotations

import errno
import importlib.util
import json
import os
import shutil

# Command-level tests execute trusted local programs without a shell.
import subprocess  # nosec B404
import sys
from pathlib import Path

import pytest


@pytest.fixture
def exporter(monkeypatch):
    source = Path(__file__).resolve().parents[3] / "Helper_Scripts/export_openapi_schema.py"
    spec = importlib.util.spec_from_file_location("openapi_contract_exporter", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(
        module,
        "_load_schema",
        lambda: {
            "openapi": "3.1.0",
            "paths": {"/api/v1/health": {}, "/api/v1/chat/completions": {}},
            "components": {"schemas": {}},
        },
    )
    return module


def test_exporter_generates_and_checks_both_committed_artifacts(exporter, tmp_path):
    fingerprint = tmp_path / "openapi.fingerprint.json"
    paths = tmp_path / "paths.d.ts"
    assert exporter.main(["--fingerprint", str(fingerprint), "--paths", str(paths)]) == 0
    assert exporter.main(["--check", str(fingerprint), "--check-paths", str(paths)]) == 0
    literals = [
        json.loads(line.removeprefix("  | ")) for line in paths.read_text().splitlines() if line.startswith("  | ")
    ]
    assert literals == ["/api/v1/chat/completions", "/api/v1/health"]


@pytest.mark.parametrize(
    "stale",
    [
        "export type ApiPath = never",
        'export type ApiPath = "/api/v1/health"',
        'export type ApiPath = "/api/v1/health" | "/api/v1/old"',
    ],
)
def test_matching_fingerprint_does_not_certify_stale_union(exporter, tmp_path, capsys, stale):
    fingerprint = tmp_path / "openapi.fingerprint.json"
    paths = tmp_path / "paths.d.ts"
    assert exporter.main(["--fingerprint", str(fingerprint)]) == 0
    paths.write_text(stale)
    assert exporter.main(["--check", str(fingerprint), "--check-paths", str(paths)]) == 1
    assert "committed path union" in capsys.readouterr().err


def test_missing_path_artifact_fails_closed(exporter, tmp_path):
    assert exporter.main(["--check-paths", str(tmp_path / "missing.d.ts")]) == 2


def test_path_union_escapes_literals_without_changing_route_identity(exporter, monkeypatch, tmp_path):
    route = '/api/v1/quoted"/back\\slash/line\nend'
    monkeypatch.setattr(exporter, "_load_schema", lambda: {"paths": {route: {}}})
    paths = tmp_path / "paths.d.ts"
    assert exporter.main(["--paths", str(paths)]) == 0
    literal = next(line.removeprefix("  | ") for line in paths.read_text().splitlines() if line.startswith("  | "))
    assert json.loads(literal) == route


def test_empty_contract_cannot_generate_a_path_union(exporter, monkeypatch, tmp_path):
    monkeypatch.setattr(exporter, "_load_schema", lambda: {"paths": {}})
    assert exporter.main(["--paths", str(tmp_path / "paths.d.ts")]) == 2


def test_required_drift_gate_checks_committed_path_union():
    root = Path(__file__).resolve().parents[3]
    workflow = (root / ".github/workflows/backend-required.yml").read_text()
    makefile = (root / "Makefile").read_text()
    for source in (workflow, makefile):
        assert "--check-paths apps/packages/api-types/paths.d.ts" in source


@pytest.fixture
def openapi_make_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Run real make/Python commands with controlled versions and a small exporter."""
    if os.name == "nt":
        pytest.skip("TASK-15477: OpenAPI Make recipes require a POSIX shell and executable symlinks")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    try:
        for name in ("python3.12", "custom python"):
            (bin_dir / name).symlink_to(sys.executable)
    except NotImplementedError:
        _openapi_capability_unavailable("Executable symlinks are unsupported")
    except OSError as exc:
        if exc.errno not in (errno.ENOSYS, errno.ENOTSUP, errno.EPERM):
            raise
        _openapi_capability_unavailable(f"Executable symlinks are unsupported: {exc}")
    (tmp_path / "sitecustomize.py").write_text(
        "import os, sys\nsys.version_info = (3, int(os.environ['OPENAPI_TEST_MINOR']), 0, 'final', 0)\n"
    )
    scripts = tmp_path / "Helper_Scripts"
    scripts.mkdir()
    (scripts / "export_openapi_schema.py").write_text(
        "import json, sys\nfrom pathlib import Path\n"
        "Path('export-invocation.json').write_text(json.dumps(\n"
        "    {'python': sys.executable, 'args': sys.argv[1:]}))\n"
    )
    monkeypatch.setenv("PATH", str(bin_dir) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    monkeypatch.setenv("OPENAPI_TEST_MINOR", "12")
    monkeypatch.delenv("OPENAPI_PYTHON", raising=False)
    monkeypatch.delenv("MAKEFLAGS", raising=False)
    return tmp_path


def _openapi_capability_unavailable(reason: str) -> None:
    """Keep required Linux coverage hard-failing while allowing unsupported hosts."""
    if sys.platform.startswith("linux"):
        pytest.fail(f"TASK-15477: required Linux CI capability unavailable: {reason}")
    pytest.skip(f"TASK-15477: {reason}")


def _run_openapi_make(project: Path, target: str, *assignments: str) -> subprocess.CompletedProcess[str]:
    """Execute the checkout's actual OpenAPI recipe in the isolated project."""
    make = shutil.which("make")
    if make is None:
        _openapi_capability_unavailable("GNU make is unavailable")
    shell = shutil.which("sh")
    if shell is None:
        _openapi_capability_unavailable("POSIX shell is unavailable")
    executable = str(Path(make).resolve())
    # Only the resolved local make executable and its fixed version option are run.
    version = subprocess.run(  # nosec B603
        [executable, "--version"], capture_output=True, text=True, timeout=10, check=False, shell=False
    )
    if version.returncode != 0 or "GNU Make" not in version.stdout:
        _openapi_capability_unavailable("GNU make is required for these recipes")
    makefile = Path(__file__).resolve().parents[3] / "Makefile"
    # The executable is resolved absolutely; every argument is a fixed target or test-owned path/assignment.
    return subprocess.run(  # nosec B603
        [
            executable,
            "-f",
            str(makefile),
            target,
            f"SHELL={Path(shell).resolve()}",
            f"CI_LOCAL_PYTHON={sys.executable}",
            *assignments,
        ],
        cwd=project,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
        shell=False,
    )


@pytest.mark.parametrize(
    ("platform", "expected"), [("linux", pytest.fail.Exception), ("darwin", pytest.skip.Exception)]
)
def test_make_openapi_missing_tool_preserves_linux_ci_coverage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, platform: str, expected: type[BaseException]
) -> None:
    monkeypatch.setattr(shutil, "which", lambda name: None)
    monkeypatch.setattr(sys, "platform", platform)
    with pytest.raises(expected, match="make"):
        _run_openapi_make(tmp_path, "openapi-drift-check")


@pytest.mark.parametrize("capability", ["POSIX shell", "GNU make"])
@pytest.mark.parametrize(
    ("platform", "expected"), [("linux", pytest.fail.Exception), ("darwin", pytest.skip.Exception)]
)
def test_make_openapi_unsupported_capability_preserves_linux_ci_coverage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capability: str, platform: str, expected: type[BaseException]
) -> None:
    monkeypatch.setattr(sys, "platform", platform)
    executables: dict[str, str | None] = {"make": str(tmp_path / "make"), "sh": str(tmp_path / "sh")}
    if capability == "POSIX shell":
        executables["sh"] = None
    monkeypatch.setattr(shutil, "which", executables.get)
    monkeypatch.setattr(subprocess, "run", lambda args, **kwargs: subprocess.CompletedProcess(args, 0, "BSD make", ""))
    with pytest.raises(expected, match=capability):
        _run_openapi_make(tmp_path, "openapi-drift-check")


@pytest.mark.parametrize(
    ("target", "expected_args"),
    [
        ("openapi-fingerprint", ["--fingerprint", "apps/packages/api-types/openapi.fingerprint.json"]),
        (
            "openapi-drift-check",
            [
                "--check",
                "apps/packages/api-types/openapi.fingerprint.json",
                "--check-paths",
                "apps/packages/api-types/paths.d.ts",
            ],
        ),
    ],
)
@pytest.mark.parametrize("override", [False, True], ids=["default", "explicit-path"])
def test_make_openapi_uses_canonical_python(
    openapi_make_project: Path, target: str, expected_args: list[str], override: bool
) -> None:
    project = openapi_make_project
    python = project / "bin" / ("custom python" if override else "python3.12")
    assignments = [f"OPENAPI_PYTHON={python}"] if override else []
    result = _run_openapi_make(project, target, *assignments)
    assert result.returncode == 0, result.stdout + result.stderr
    invocation = json.loads((project / "export-invocation.json").read_text())
    assert invocation == {"python": str(python), "args": expected_args}


@pytest.mark.parametrize("target", ["openapi-fingerprint", "openapi-drift-check"])
def test_make_openapi_rejects_wrong_python_before_export(
    openapi_make_project: Path, monkeypatch: pytest.MonkeyPatch, target: str
) -> None:
    monkeypatch.setenv("OPENAPI_TEST_MINOR", "14")
    result = _run_openapi_make(openapi_make_project, target, f"OPENAPI_PYTHON={sys.executable}")
    assert result.returncode != 0
    assert "Python 3.12" in result.stderr
    assert "OPENAPI_PYTHON" in result.stderr
    assert not (openapi_make_project / "export-invocation.json").exists()


@pytest.mark.parametrize("target", ["openapi-fingerprint", "openapi-drift-check"])
def test_make_openapi_missing_python_is_actionable(openapi_make_project: Path, target: str) -> None:
    result = _run_openapi_make(openapi_make_project, target, "OPENAPI_PYTHON=/missing/python3.12")
    assert result.returncode != 0
    assert "Python 3.12" in result.stderr
    assert "OPENAPI_PYTHON" in result.stderr
    assert not (openapi_make_project / "export-invocation.json").exists()
