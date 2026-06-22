#!/usr/bin/env python3
"""Internal MCP Unified release-candidate harness.

This module intentionally stays independent from ``tldw_Server_API`` and root
pytest fixtures. It builds and validates the standalone package project under
``apps/mcp-unified`` and writes local-only RC evidence.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess  # nosec B404
import sys
import tempfile
import time
import venv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility.
    tomllib = None  # type: ignore[assignment]

RESULT_STATUSES = {"passed", "failed", "skipped"}
SOURCE_PATH = "apps/mcp-unified"
LAYOUT = "src"
EVIDENCE_JSON = "mcp-unified-rc-evidence.json"
EVIDENCE_MARKDOWN = "mcp-unified-rc-summary.md"
PACKAGE_IMPORT_NAME = "mcp_unified"
OPTIONAL_EXTRAS = ("core", "fastapi", "sqlite", "federation", "gateway", "dev")

SECRET_KEY_VALUE_PATTERN = re.compile(
    r"(?i)\b(api[_-]?key|token|secret|password|bearer[_-]?token)\b"
    r"(\s*[=:]\s*)([^\s,\"'}]+)"
)
SECRET_JSON_PATTERN = re.compile(
    r"(?i)([\"']?(?:api[_-]?key|token|secret|password|bearer[_-]?token)"
    r"[\"']?\s*:\s*[\"'])([^\"']+)([\"'])"
)
AUTHORIZATION_BEARER_PATTERN = re.compile(
    r"(?i)\b(authorization\s*[:=]\s*bearer\s+)([A-Za-z0-9._~+/=-]+)"
)
BARE_BEARER_PATTERN = re.compile(r"(?i)\b(bearer\s+)([A-Za-z0-9._~+/=-]+)")


@dataclass(frozen=True)
class RcPaths:
    """Resolved paths used by the internal MCP Unified RC harness."""

    repo_root: Path
    package_project: Path
    package_src: Path
    evidence_dir: Path
    dist_dir: Path

    @classmethod
    def from_repo_root(cls, repo_root: Path) -> "RcPaths":
        """Return canonical RC paths for a repository root."""

        return cls(
            repo_root=repo_root,
            package_project=repo_root / "apps" / "mcp-unified",
            package_src=repo_root / "apps" / "mcp-unified" / "src" / "mcp_unified",
            evidence_dir=repo_root / ".artifacts" / "mcp-unified-rc",
            dist_dir=repo_root / ".artifacts" / "mcp-unified-rc" / "dist",
        )


@dataclass
class RcCommandResult:
    """Captured subprocess result with redacted output."""

    command: list[str]
    cwd: str
    returncode: int
    stdout: str
    stderr: str
    duration_ms: int

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation for evidence output."""

        return {
            "command": self.command,
            "cwd": self.cwd,
            "returncode": self.returncode,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "duration_ms": self.duration_ms,
        }


@dataclass
class RcEvidenceRecorder:
    """Collect and write MCP Unified RC evidence."""

    evidence_dir: Path
    package_name: str
    package_version: str
    package_status: str
    publishing_status: str
    commit: str
    source_path: str
    layout: str
    package_metadata: dict[str, Any] = field(default_factory=dict)
    known_limitations: list[str] = field(default_factory=list)
    results: list[dict[str, Any]] = field(default_factory=list)
    artifacts: list[dict[str, str]] = field(default_factory=list)

    def record(
        self,
        *,
        phase: str,
        name: str,
        status: str,
        duration_ms: int,
        reason: str | None = None,
        details: dict[str, Any] | None = None,
        required: bool = True,
    ) -> None:
        """Record one RC check result."""

        if status not in RESULT_STATUSES:
            raise ValueError(f"invalid RC result status: {status}")
        entry: dict[str, Any] = {
            "phase": phase,
            "name": name,
            "status": status,
            "duration_ms": duration_ms,
            "required": required,
        }
        if reason:
            entry["reason"] = reason
        if details:
            entry["details"] = details
        self.results.append(entry)

    def record_artifact(self, *, kind: str, path: Path) -> None:
        """Record an artifact filename and SHA256 hash."""

        self.artifacts.append(
            {
                "kind": kind,
                "filename": path.name,
                "sha256": sha256_file(path),
            }
        )

    def has_required_failures(self) -> bool:
        """Return whether any required result failed."""

        return any(
            result["status"] == "failed" and result.get("required", True)
            for result in self.results
        )

    def write(self) -> tuple[Path, Path]:
        """Write JSON and Markdown evidence files."""

        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "passed": sum(1 for result in self.results if result["status"] == "passed"),
            "failed": sum(1 for result in self.results if result["status"] == "failed"),
            "skipped": sum(1 for result in self.results if result["status"] == "skipped"),
        }
        payload = {
            "schema_version": "1",
            "ok": not self.has_required_failures(),
            "package": {
                "name": self.package_name,
                "version": self.package_version,
                "commit": self.commit,
                "source_path": self.source_path,
                "layout": self.layout,
                "status": self.package_status,
                "publishing_status": self.publishing_status,
                "metadata": self.package_metadata,
            },
            "artifacts": self.artifacts,
            "environment": {
                "platform": platform.platform(),
                "system": platform.system(),
                "machine": platform.machine(),
                "python": platform.python_version(),
                "python_executable": sys.executable,
                "runner": "github-actions" if os.environ.get("GITHUB_ACTIONS") else "local",
            },
            "results": self.results,
            "summary": summary,
            "known_limitations": self.known_limitations,
        }
        json_path = self.evidence_dir / EVIDENCE_JSON
        markdown_path = self.evidence_dir / EVIDENCE_MARKDOWN
        json_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        markdown_path.write_text(_render_markdown(payload), encoding="utf-8")
        return json_path, markdown_path


def redact_text(value: str) -> str:
    """Redact secret-like values while preserving normal output."""

    redacted = SECRET_KEY_VALUE_PATTERN.sub(
        lambda match: f"{match.group(1)}{match.group(2)}[redacted]",
        value,
    )
    redacted = SECRET_JSON_PATTERN.sub(
        lambda match: f"{match.group(1)}[redacted]{match.group(3)}",
        redacted,
    )
    redacted = AUTHORIZATION_BEARER_PATTERN.sub(
        lambda match: f"{match.group(1)}[redacted]",
        redacted,
    )
    redacted = BARE_BEARER_PATTERN.sub(
        lambda match: f"{match.group(1)}[redacted]",
        redacted,
    )
    return redacted


def sha256_file(path: Path) -> str:
    """Return the SHA256 hex digest for a file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_command(
    command: Sequence[str],
    *,
    cwd: Path,
    timeout: int = 180,
    env: dict[str, str] | None = None,
) -> RcCommandResult:
    """Run a subprocess with captured, redacted output."""

    started = time.perf_counter()
    run_env = {**os.environ, "PIP_DISABLE_PIP_VERSION_CHECK": "1"}
    run_env.pop("PYTHONPATH", None)
    if env:
        run_env.update(env)
    redacted_command = [redact_text(part) for part in command]
    try:
        completed = subprocess.run(  # nosec B603
            list(command),
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=run_env,
        )
        return RcCommandResult(
            command=redacted_command,
            cwd=str(cwd),
            returncode=completed.returncode,
            stdout=redact_text(completed.stdout[-6000:]),
            stderr=redact_text(completed.stderr[-6000:]),
            duration_ms=int((time.perf_counter() - started) * 1000),
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else (exc.stdout or b"").decode("utf-8", "replace")
        stderr = exc.stderr if isinstance(exc.stderr, str) else (exc.stderr or b"").decode("utf-8", "replace")
        return RcCommandResult(
            command=redacted_command,
            cwd=str(cwd),
            returncode=124,
            stdout=redact_text(stdout[-6000:]),
            stderr=redact_text((stderr[-6000:] + f"\nTimed out after {timeout}s").strip()),
            duration_ms=int((time.perf_counter() - started) * 1000),
        )


def run_build(paths: RcPaths) -> int:
    """Build wheel and sdist artifacts from the standalone package project."""

    recorder = _new_recorder(paths)
    _run_build(paths, recorder)
    return _write_and_report(recorder)


def run_artifact_gate(paths: RcPaths) -> int:
    """Run private artifact validation checks for built package artifacts."""

    recorder = _new_recorder(paths)
    _record_existing_artifacts(paths, recorder)
    _run_artifact_gate(paths, recorder)
    return _write_and_report(recorder)


def run_install_smoke(paths: RcPaths) -> int:
    """Run clean-environment install smoke checks against the built wheel."""

    recorder = _new_recorder(paths)
    _record_existing_artifacts(paths, recorder)
    _run_install_smoke(paths, recorder)
    return _write_and_report(recorder)


def run_extras_matrix(paths: RcPaths) -> int:
    """Install selected extras in isolated virtual environments."""

    recorder = _new_recorder(paths)
    _record_existing_artifacts(paths, recorder)
    _run_extras_matrix(paths, recorder)
    return _write_and_report(recorder)


def run_cli_uat(paths: RcPaths) -> int:
    """Run the first-slice installed gateway CLI UAT."""

    recorder = _new_recorder(paths)
    _record_existing_artifacts(paths, recorder)
    _run_cli_uat(paths, recorder)
    return _write_and_report(recorder)


def run_smoke_uat(paths: RcPaths) -> int:
    """Run the first-slice installed smoke CLI UAT."""

    recorder = _new_recorder(paths)
    _record_existing_artifacts(paths, recorder)
    _run_smoke_uat(paths, recorder)
    return _write_and_report(recorder)


def run_evidence(paths: RcPaths) -> int:
    """Write or refresh the evidence bundle."""

    recorder = _new_recorder(paths)
    _record_existing_artifacts(paths, recorder)
    if recorder.artifacts:
        recorder.record(
            phase="evidence",
            name="artifact_hash_snapshot",
            status="passed",
            duration_ms=0,
            details={"artifact_count": len(recorder.artifacts)},
        )
    else:
        recorder.record(
            phase="evidence",
            name="artifact_hash_snapshot",
            status="skipped",
            duration_ms=0,
            reason="no built artifacts found in .artifacts/mcp-unified-rc/dist",
            required=False,
        )
    return _write_and_report(recorder)


def run_all(paths: RcPaths) -> int:
    """Run the internal RC sequence and write combined evidence."""

    recorder = _new_recorder(paths)
    _run_build(paths, recorder)
    _run_artifact_gate(paths, recorder)
    _run_install_smoke(paths, recorder)
    _run_cli_uat(paths, recorder)
    _run_smoke_uat(paths, recorder)
    _run_extras_matrix(paths, recorder)
    return _write_and_report(recorder)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(
        prog="mcp-unified-rc",
        description="Build and validate the private MCP Unified internal RC.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root. Defaults to the current working directory.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in (
        "build",
        "artifact-gate",
        "install-smoke",
        "extras-matrix",
        "cli-uat",
        "smoke-uat",
        "evidence",
        "all",
    ):
        subparsers.add_parser(name, help=f"Run the {name} RC phase.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the MCP Unified RC harness."""

    parser = build_parser()
    args = parser.parse_args(argv)
    paths = RcPaths.from_repo_root(args.repo_root.resolve())
    if args.command == "build":
        return run_build(paths)
    if args.command == "artifact-gate":
        return run_artifact_gate(paths)
    if args.command == "install-smoke":
        return run_install_smoke(paths)
    if args.command == "extras-matrix":
        return run_extras_matrix(paths)
    if args.command == "cli-uat":
        return run_cli_uat(paths)
    if args.command == "smoke-uat":
        return run_smoke_uat(paths)
    if args.command == "evidence":
        return run_evidence(paths)
    return run_all(paths)


def _run_build(paths: RcPaths, recorder: RcEvidenceRecorder) -> None:
    started = time.perf_counter()
    if not paths.package_project.is_dir():
        recorder.record(
            phase="build",
            name="package_project_exists",
            status="failed",
            duration_ms=0,
            reason=f"missing package project: {paths.package_project}",
        )
        return

    if paths.dist_dir.exists():
        shutil.rmtree(paths.dist_dir)
    paths.dist_dir.mkdir(parents=True, exist_ok=True)

    result = run_command(
        [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--sdist",
            "--no-isolation",
            "--outdir",
            str(paths.dist_dir),
            str(paths.package_project),
        ],
        cwd=paths.repo_root,
        timeout=300,
        env={"PIP_NO_INDEX": "1"},
    )
    _record_command_result(
        recorder,
        phase="build",
        name="python_build_wheel_sdist",
        result=result,
    )
    if result.returncode != 0:
        return

    wheels, sdists = _dist_artifacts(paths)
    if len(wheels) != 1 or len(sdists) != 1:
        recorder.record(
            phase="build",
            name="artifact_count",
            status="failed",
            duration_ms=int((time.perf_counter() - started) * 1000),
            reason=f"expected one wheel and one sdist, found {len(wheels)} wheel(s), {len(sdists)} sdist(s)",
            details={
                "wheels": [path.name for path in wheels],
                "sdists": [path.name for path in sdists],
            },
        )
        return
    recorder.record_artifact(kind="wheel", path=wheels[0])
    recorder.record_artifact(kind="sdist", path=sdists[0])
    recorder.record(
        phase="build",
        name="artifact_hashes",
        status="passed",
        duration_ms=0,
        details={"artifact_count": 2},
    )


def _run_artifact_gate(paths: RcPaths, recorder: RcEvidenceRecorder) -> None:
    wheels, sdists = _dist_artifacts(paths)
    if not wheels or not sdists:
        recorder.record(
            phase="artifact_gate",
            name="built_artifacts_present",
            status="failed",
            duration_ms=0,
            reason="expected built wheel and sdist in .artifacts/mcp-unified-rc/dist; run build first",
        )
    else:
        twine_result = run_command(
            [sys.executable, "-m", "twine", "check", *[str(path) for path in [*wheels, *sdists]]],
            cwd=paths.repo_root,
            timeout=180,
        )
        _record_command_result(
            recorder,
            phase="artifact_gate",
            name="twine_check",
            result=twine_result,
        )

    pytest_result = run_command(
        [
            sys.executable,
            "-m",
            "pytest",
            "-c",
            str(paths.package_project / "pytest-artifact-gate.ini"),
            ".github/tests/test_mcp_unified_artifact_gate.py",
            "-q",
        ],
        cwd=paths.repo_root,
        timeout=300,
        env={"PYTHONPATH": str(paths.package_project / "src")},
    )
    _record_command_result(
        recorder,
        phase="artifact_gate",
        name="pytest_artifact_gate",
        result=pytest_result,
    )


def _run_install_smoke(paths: RcPaths, recorder: RcEvidenceRecorder) -> None:
    wheel = _latest_wheel(paths)
    if wheel is None:
        recorder.record(
            phase="install_smoke",
            name="wheel_present",
            status="failed",
            duration_ms=0,
            reason="no built wheel found; run build first",
        )
        return

    with tempfile.TemporaryDirectory(prefix="mcp-unified-rc-nodeps-") as temp_name:
        temp_dir = Path(temp_name)
        target_dir = temp_dir / "target"
        work_dir = temp_dir / "work"
        work_dir.mkdir()
        install_result = run_command(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-deps",
                "--target",
                str(target_dir),
                str(wheel),
            ],
            cwd=work_dir,
            timeout=180,
        )
        _record_command_result(
            recorder,
            phase="install_smoke",
            name="no_deps_target_install",
            result=install_result,
        )
        if install_result.returncode == 0:
            import_result = run_command(
                [
                    sys.executable,
                    "-S",
                    "-c",
                    (
                        "import importlib.metadata as metadata; "
                        "import sys; "
                        f"sys.path.insert(0, {str(target_dir)!r}); "
                        "import mcp_unified; "
                        "print(mcp_unified.__version__); "
                        "print(metadata.metadata('mcp-unified')['Name'])"
                    ),
                ],
                cwd=work_dir,
                timeout=60,
            )
            _record_command_result(
                recorder,
                phase="install_smoke",
                name="no_deps_import_package_info",
                result=import_result,
            )

    _run_normal_install_checks(paths, recorder, wheel, phase="install_smoke")


def _run_extras_matrix(paths: RcPaths, recorder: RcEvidenceRecorder) -> None:
    wheel = _latest_wheel(paths)
    if wheel is None:
        recorder.record(
            phase="extras_matrix",
            name="wheel_present",
            status="failed",
            duration_ms=0,
            reason="no built wheel found; run build first",
        )
        return

    for extra in OPTIONAL_EXTRAS:
        with tempfile.TemporaryDirectory(prefix=f"mcp-unified-rc-{extra}-") as temp_name:
            temp_dir = Path(temp_name)
            venv_dir = temp_dir / ".venv"
            if not _create_venv(venv_dir, recorder, phase="extras_matrix", name=f"{extra}_venv"):
                continue
            python_path = _venv_executable(venv_dir, "python")
            install_result = run_command(
                [
                    str(python_path),
                    "-m",
                    "pip",
                    "install",
                    f"{wheel}[{extra}]",
                ],
                cwd=temp_dir,
                timeout=300,
            )
            _record_command_result(
                recorder,
                phase="extras_matrix",
                name=f"{extra}_install",
                result=install_result,
            )
            if install_result.returncode != 0:
                continue
            import_result = run_command(
                [str(python_path), "-c", "import mcp_unified; print(mcp_unified.__version__)"],
                cwd=temp_dir,
                timeout=60,
            )
            _record_command_result(
                recorder,
                phase="extras_matrix",
                name=f"{extra}_import",
                result=import_result,
            )


def _run_cli_uat(paths: RcPaths, recorder: RcEvidenceRecorder) -> None:
    wheel = _latest_wheel(paths)
    if wheel is None:
        recorder.record(
            phase="cli_uat",
            name="wheel_present",
            status="failed",
            duration_ms=0,
            reason="no built wheel found; run build first",
        )
        return
    _run_normal_install_checks(
        paths,
        recorder,
        wheel,
        phase="cli_uat",
        command_checks=("gateway_package_info",),
    )
    recorder.record(
        phase="cli_uat",
        name="user_guide_wheel_mode",
        status="skipped",
        duration_ms=0,
        reason="full user-guide UAT wheel mode is scheduled for the later UAT harness task",
        required=False,
    )


def _run_smoke_uat(paths: RcPaths, recorder: RcEvidenceRecorder) -> None:
    wheel = _latest_wheel(paths)
    if wheel is None:
        recorder.record(
            phase="smoke_uat",
            name="wheel_present",
            status="failed",
            duration_ms=0,
            reason="no built wheel found; run build first",
        )
        return

    with tempfile.TemporaryDirectory(prefix="mcp-unified-rc-smoke-") as temp_name:
        temp_dir = Path(temp_name)
        venv_dir = temp_dir / ".venv"
        if not _create_venv(venv_dir, recorder, phase="smoke_uat", name="smoke_venv"):
            return
        python_path = _venv_executable(venv_dir, "python")
        install_result = run_command(
            [str(python_path), "-m", "pip", "install", str(wheel)],
            cwd=temp_dir,
            timeout=300,
        )
        _record_command_result(
            recorder,
            phase="smoke_uat",
            name="normal_install",
            result=install_result,
        )
        if install_result.returncode != 0:
            return
        report_path = paths.evidence_dir / "smoke-inprocess.json"
        smoke_result = run_command(
            [
                str(_venv_executable(venv_dir, "mcp-unified-smoke")),
                "inprocess",
                "--json-report",
                str(report_path),
            ],
            cwd=temp_dir,
            timeout=180,
        )
        _record_command_result(
            recorder,
            phase="smoke_uat",
            name="smoke_inprocess",
            result=smoke_result,
            details={"json_report": str(report_path)},
        )
    recorder.record(
        phase="smoke_uat",
        name="live_transport_smoke",
        status="skipped",
        duration_ms=0,
        reason="stdio/http/websocket installed-wheel UAT is deferred until fixture-backed wheel mode is available",
        required=False,
    )


def _run_normal_install_checks(
    paths: RcPaths,
    recorder: RcEvidenceRecorder,
    wheel: Path,
    *,
    phase: str,
    command_checks: tuple[str, ...] = (
        "import",
        "gateway_package_info",
        "smoke_help",
    ),
) -> None:
    with tempfile.TemporaryDirectory(prefix=f"mcp-unified-rc-{phase}-") as temp_name:
        temp_dir = Path(temp_name)
        venv_dir = temp_dir / ".venv"
        if not _create_venv(venv_dir, recorder, phase=phase, name="normal_venv"):
            return
        python_path = _venv_executable(venv_dir, "python")
        install_result = run_command(
            [str(python_path), "-m", "pip", "install", str(wheel)],
            cwd=temp_dir,
            timeout=300,
        )
        _record_command_result(
            recorder,
            phase=phase,
            name="normal_install",
            result=install_result,
        )
        if install_result.returncode != 0:
            return
        if "import" in command_checks:
            import_result = run_command(
                [str(python_path), "-c", "import mcp_unified; print(mcp_unified.__version__)"],
                cwd=temp_dir,
                timeout=60,
            )
            _record_command_result(
                recorder,
                phase=phase,
                name="normal_import",
                result=import_result,
            )
        if "gateway_package_info" in command_checks:
            gateway_result = run_command(
                [str(_venv_executable(venv_dir, "mcp-unified-gateway")), "package-info"],
                cwd=temp_dir,
                timeout=60,
            )
            _record_command_result(
                recorder,
                phase=phase,
                name="gateway_package_info",
                result=gateway_result,
            )
        if "smoke_help" in command_checks:
            smoke_help_result = run_command(
                [str(_venv_executable(venv_dir, "mcp-unified-smoke")), "--help"],
                cwd=temp_dir,
                timeout=60,
            )
            _record_command_result(
                recorder,
                phase=phase,
                name="smoke_help",
                result=smoke_help_result,
            )


def _new_recorder(paths: RcPaths) -> RcEvidenceRecorder:
    pyproject_metadata = _read_pyproject_project(paths.package_project / "pyproject.toml")
    package_constants = _read_package_constants(paths.package_src / "package_metadata.py")
    package_name = str(pyproject_metadata.get("name") or package_constants.get("PACKAGE_NAME") or "mcp-unified")
    return RcEvidenceRecorder(
        evidence_dir=paths.evidence_dir,
        package_name=package_name,
        package_version=str(pyproject_metadata.get("version") or "unknown"),
        package_status=str(package_constants.get("PACKAGE_STATUS") or "unknown"),
        publishing_status=str(package_constants.get("PUBLISHING_STATUS") or "unknown"),
        commit=_git_short_commit(paths.repo_root),
        source_path=SOURCE_PATH,
        layout=LAYOUT,
        package_metadata={
            **package_constants,
            "pyproject": pyproject_metadata,
        },
        known_limitations=[
            "Internal RC only: this harness does not upload or publish to TestPyPI or PyPI.",
            "Full user-guide wheel-mode UAT is deferred to the later UAT harness task.",
        ],
    )


def _record_command_result(
    recorder: RcEvidenceRecorder,
    *,
    phase: str,
    name: str,
    result: RcCommandResult,
    required: bool = True,
    details: dict[str, Any] | None = None,
) -> None:
    result_details = {"command": result.as_dict()}
    if details:
        result_details.update(details)
    recorder.record(
        phase=phase,
        name=name,
        status="passed" if result.returncode == 0 else "failed",
        duration_ms=result.duration_ms,
        reason=None if result.returncode == 0 else f"command exited with status {result.returncode}",
        details=result_details,
        required=required,
    )


def _create_venv(
    venv_dir: Path,
    recorder: RcEvidenceRecorder,
    *,
    phase: str,
    name: str,
) -> bool:
    started = time.perf_counter()
    try:
        venv.EnvBuilder(with_pip=True, clear=True).create(venv_dir)
    except Exception as exc:  # pragma: no cover - platform/environment-specific.
        recorder.record(
            phase=phase,
            name=name,
            status="failed",
            duration_ms=int((time.perf_counter() - started) * 1000),
            reason=f"failed to create venv: {redact_text(str(exc))}",
        )
        return False
    recorder.record(
        phase=phase,
        name=name,
        status="passed",
        duration_ms=int((time.perf_counter() - started) * 1000),
    )
    return True


def _record_existing_artifacts(paths: RcPaths, recorder: RcEvidenceRecorder) -> None:
    seen: set[tuple[str, str]] = set()
    for kind, path in [("wheel", path) for path in _dist_artifacts(paths)[0]]:
        key = (kind, path.name)
        if key not in seen:
            recorder.record_artifact(kind=kind, path=path)
            seen.add(key)
    for kind, path in [("sdist", path) for path in _dist_artifacts(paths)[1]]:
        key = (kind, path.name)
        if key not in seen:
            recorder.record_artifact(kind=kind, path=path)
            seen.add(key)


def _dist_artifacts(paths: RcPaths) -> tuple[list[Path], list[Path]]:
    wheels = sorted(
        [*paths.dist_dir.glob("mcp_unified-*.whl"), *paths.dist_dir.glob("mcp-unified-*.whl")]
    )
    sdists = sorted(
        [*paths.dist_dir.glob("mcp_unified-*.tar.gz"), *paths.dist_dir.glob("mcp-unified-*.tar.gz")]
    )
    return wheels, sdists


def _latest_wheel(paths: RcPaths) -> Path | None:
    wheels, _sdists = _dist_artifacts(paths)
    if not wheels:
        return None
    return max(wheels, key=lambda path: path.stat().st_mtime)


def _venv_executable(venv_dir: Path, name: str) -> Path:
    bin_dir = venv_dir / ("Scripts" if os.name == "nt" else "bin")
    suffix = ".exe" if os.name == "nt" and not name.endswith(".exe") else ""
    return bin_dir / f"{name}{suffix}"


def _read_pyproject_project(pyproject_path: Path) -> dict[str, Any]:
    if not pyproject_path.is_file():
        return {}
    if tomllib is not None:
        with pyproject_path.open("rb") as handle:
            data = tomllib.load(handle)
        project = data.get("project", {})
        return project if isinstance(project, dict) else {}

    text = pyproject_path.read_text(encoding="utf-8")
    project_text = _section_text(text, "project")
    return {
        key: value
        for key in ("name", "version", "description", "requires-python")
        if (value := _toml_string_value(project_text, key)) is not None
    }


def _read_package_constants(metadata_path: Path) -> dict[str, Any]:
    if not metadata_path.is_file():
        return {}
    tree = ast.parse(metadata_path.read_text(encoding="utf-8"))
    constants: dict[str, Any] = {}
    wanted = {
        "PACKAGE_NAME",
        "PACKAGE_IMPORT_NAME",
        "PACKAGE_STATUS",
        "PUBLISHING_STATUS",
        "LICENSE_EXPRESSION",
        "SOURCE_DISTRIBUTION",
        "DEPENDENCY_VERSION_POLICY",
    }
    for node in tree.body:
        target_name: str | None = None
        value_node: ast.AST | None = None
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target_name = node.target.id
            value_node = node.value
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            target_name = node.targets[0].id
            value_node = node.value
        if target_name in wanted and isinstance(value_node, ast.Constant):
            constants[target_name] = value_node.value
    return constants


def _section_text(text: str, section: str) -> str:
    match = re.search(rf"(?ms)^\[{re.escape(section)}\]\s*(.*?)(?=^\[|\Z)", text)
    return match.group(1) if match else ""


def _toml_string_value(section: str, key: str) -> str | None:
    match = re.search(rf"(?m)^{re.escape(key)}\s*=\s*[\"']([^\"']+)[\"']", section)
    return match.group(1) if match else None


def _git_short_commit(repo_root: Path) -> str:
    result = run_command(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=repo_root,
        timeout=30,
    )
    if result.returncode != 0:
        return "unknown"
    return result.stdout.strip() or "unknown"


def _render_markdown(payload: dict[str, Any]) -> str:
    package = payload["package"]
    lines = [
        "# MCP Unified RC Summary",
        "",
        f"OK: {payload['ok']}",
        f"Package: {package['name']} {package['version']}",
        f"Source path: {package['source_path']}",
        f"Layout: {package['layout']}",
        "",
        "## Summary",
        "",
    ]
    for key, value in payload["summary"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Artifacts", ""])
    if payload["artifacts"]:
        for artifact in payload["artifacts"]:
            lines.append(f"- {artifact['kind']}: {artifact['filename']} `{artifact['sha256']}`")
    else:
        lines.append("- none")
    lines.extend(["", "## Results", ""])
    for result in payload["results"]:
        required = "required" if result.get("required", True) else "optional"
        line = f"- {result['status']}: {result['phase']} / {result['name']} ({required})"
        if result.get("reason"):
            line += f" - {result['reason']}"
        lines.append(line)
    lines.extend(["", "## Known Limitations", ""])
    if payload["known_limitations"]:
        lines.extend(f"- {item}" for item in payload["known_limitations"])
    else:
        lines.append("- none")
    return "\n".join(lines) + "\n"


def _write_and_report(recorder: RcEvidenceRecorder) -> int:
    json_path, markdown_path = recorder.write()
    print(f"Wrote RC evidence JSON: {json_path}")
    print(f"Wrote RC evidence Markdown: {markdown_path}")
    if recorder.has_required_failures():
        print("RC status: failed")
        return 1
    print("RC status: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
