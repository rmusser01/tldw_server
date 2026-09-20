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
import tarfile
import tempfile
import time
import venv
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from email.message import Message
from email.parser import Parser
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from loguru import logger

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility.
    try:
        import tomli as tomllib
    except ModuleNotFoundError:
        tomllib = None  # type: ignore[assignment]

RESULT_STATUSES = {"passed", "failed", "skipped"}
SOURCE_PATH = "apps/mcp-unified"
LAYOUT = "src"
EVIDENCE_JSON = "mcp-unified-rc-evidence.json"
EVIDENCE_MARKDOWN = "mcp-unified-rc-summary.md"
PACKAGE_IMPORT_NAME = "mcp_unified"
OPTIONAL_EXTRAS = ("core", "fastapi", "sqlite", "federation", "gateway", "dev")
PUBLISH_ALLOW_ENV = "MCP_UNIFIED_ALLOW_PUBLISH"
PUBLISH_TARGET_REPOSITORIES = {
    "testpypi": "https://test.pypi.org/legacy/",
    "pypi": "https://upload.pypi.org/legacy/",
}
USER_GUIDE_UAT_SCRIPT = Path("Helper_Scripts") / "Testing-related" / "mcp_standalone_user_guide_uat.py"
OFFICIAL_SDK_SMOKE_SCRIPT = Path("Helper_Scripts") / "Testing-related" / "mcp_official_sdk_stdio_smoke.py"
OFFICIAL_SDK_REQUIREMENT = "mcp==2.0.0"
OFFICIAL_SDK_TIER = "Tier 1"
OFFICIAL_SDK_TAG_COMMIT = "6f69a37"
OFFICIAL_SDK_RELEASE_URL = "https://github.com/modelcontextprotocol/python-sdk/releases/tag/v2.0.0"
OFFICIAL_SDK_INDEX_URL = "https://modelcontextprotocol.io/docs/sdk"
PROTOCOL_ARTIFACT_CONSUMER_TEST = (
    Path("tldw_Server_API") / "app" / "core" / "MCP_unified" / "tests" / "test_gateway_protocol_artifact_consumer.py"
)
PROTOCOL_TEST_SUITES = tuple(
    PROTOCOL_ARTIFACT_CONSUMER_TEST.with_name(filename)
    for filename in (
        "test_gateway_protocol_contracts.py",
        "test_gateway_protocol_validation.py",
        "test_gateway_protocol_projection.py",
        "test_gateway_protocol_connection.py",
        "test_gateway_protocol_stdio.py",
    )
)
PROTOCOL_FIXTURE_ROOT = Path("tldw_Server_API") / "app" / "core" / "MCP_unified" / "tests" / "fixtures" / "mcp_protocol"
PROTOCOL_FIXTURE_COMMIT = "5f5440bb26a62e2cf3440b92da5a667efa03b267"
PROTOCOL_FIXTURE_SHA256 = {
    "2026-07-28": "ef70b61f99b6d2e5e3b46863822eab08dff6a45bedc7a08914e0e5b133f40203",
    "2025-11-25": "268a5f82ba70fd7e4b6dc4aa1e64f116f74b4d0edcb69dc046829c79dd4e97e7",
    "2025-06-18": "af845e7e5b9d27107d1690f0936022546177a1403e63ffb11470135b296a2e01",
    "2025-03-26": "e720669548c8100a4282c49e580efd6ddf7f28899ea786fc8db251dbdb356131",
    "2024-11-05": "61cea2392d4f284092d09bc84b9ac488c0d5618ac2b38a56942fc5b99fd960ce",
}
PROTOCOL_FIXTURE_PATHS = {revision: Path(revision) / "schema.json" for revision in PROTOCOL_FIXTURE_SHA256}
PROTOCOL_FIXTURE_REPOSITORY = "https://github.com/modelcontextprotocol/modelcontextprotocol"
PROTOCOL_FIXTURE_LICENSE = "Apache-2.0"
PROTOCOL_FIXTURE_URLS = {
    revision: (
        "https://raw.githubusercontent.com/modelcontextprotocol/"
        f"modelcontextprotocol/{PROTOCOL_FIXTURE_COMMIT}/schema/{revision}/schema.json"
    )
    for revision in PROTOCOL_FIXTURE_SHA256
}
PROTOCOL_FIXTURE_SUPPORT_FILES = (Path("manifest.json"), Path("NOTICE.md"))
JSONSCHEMA_REQUIREMENT_BOUNDS = frozenset({">=4.23", "<5"})
PIP_DEPENDENCY_FAILURE_MARKERS = (
    "could not find a version that satisfies the requirement",
    "no matching distribution found",
)
PIP_NETWORK_FAILURE_MARKERS = (
    "failed to establish a new connection",
    "name or service not known",
    "nodename nor servname",
    "temporary failure in name resolution",
    "connection refused",
    "connection timed out",
)
PIP_DEPENDENCY_OUTAGE_REASON = "dependency resolution unavailable in this environment"
POSIX_ABSOLUTE_PATH_PATTERN = re.compile(r"(^|[\s\"'(\[{=,:])(/(?!/)[^\r\n\"',}\]]+)")
WINDOWS_DRIVE_PATH_PATTERN = re.compile(r"(?i)(^|[\s\"'(\[{=,:])([A-Z]:[\\/][^\r\n\"',}\]]+)")
WINDOWS_UNC_PATH_PATTERN = re.compile(r"(^|[\s\"'(\[{=,:])(\\\\[^\r\n\"',}\]]+)")
RELATIVE_LOCAL_PATH_PATTERN = re.compile(
    r"(?:\.\./)+(?:Users|private|var|tmp|Volumes|home|opt|usr|workspace|runner)/"
    r"[^\s\"',}\]]+"
)
FILE_URI_PATTERN = re.compile(r"(?i)\bfile:///(?:[^\r\n\"',}\]]+)")
URI_USERINFO_PATTERN = re.compile(r"(?i)(\b[a-z][a-z0-9+.-]*://)([^/@\s]+)@")

SECRET_KEY_VALUE_PATTERN = re.compile(
    r"(?i)\b(api[_-]?key|token|secret|password|bearer[_-]?token)\b"
    r"(\s*[=:]\s*)([^\s,\"'}]+)"
)
SECRET_JSON_PATTERN = re.compile(
    r"(?i)([\"']?(?:api[_-]?key|token|secret|password|bearer[_-]?token)"
    r"[\"']?\s*:\s*[\"'])([^\"']+)([\"'])"
)
AUTHORIZATION_BEARER_PATTERN = re.compile(r"(?i)\b(authorization\s*[:=]\s*bearer\s+)([A-Za-z0-9._~+/=-]+)")
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
    def from_repo_root(cls, repo_root: Path) -> RcPaths:
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


@dataclass(frozen=True)
class PublishPlan:
    """Dry-run or executable publish command plan for built MCP Unified artifacts."""

    target: str
    repository_url: str
    dry_run: bool
    execute: bool
    artifact_filenames: list[str]
    command: list[str]

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation for RC evidence output."""

        return {
            "target": self.target,
            "repository_url": self.repository_url,
            "dry_run": self.dry_run,
            "execute": self.execute,
            "artifact_filenames": self.artifact_filenames,
            "command": self.command,
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
    repo_root: Path | None = None

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
            entry["reason"] = _sanitize_evidence_value(reason, self)
        if details:
            entry["details"] = _sanitize_evidence_value(details, self)
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

        return any(result["status"] == "failed" and result.get("required", True) for result in self.results)

    def write(self) -> tuple[Path, Path]:
        """Write JSON and Markdown evidence files."""

        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "passed": sum(1 for result in self.results if result["status"] == "passed"),
            "failed": sum(1 for result in self.results if result["status"] == "failed"),
            "skipped": sum(1 for result in self.results if result["status"] == "skipped"),
        }
        payload: dict[str, Any] = {
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
                "python_executable": _redact_evidence_text(sys.executable, self),
                "runner": "github-actions" if os.environ.get("GITHUB_ACTIONS") else "local",
            },
            "results": self.results,
            "summary": summary,
            "known_limitations": self.known_limitations,
        }
        sanitized_payload = _sanitize_evidence_value(payload, self)
        if not isinstance(sanitized_payload, dict):  # pragma: no cover - structural invariant.
            raise TypeError("sanitized RC evidence payload must remain a dictionary")
        payload = sanitized_payload
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


def _sanitize_evidence_value(value: Any, recorder: RcEvidenceRecorder) -> Any:
    """Return a JSON-safe value with local paths and secrets redacted."""

    if isinstance(value, str):
        return _redact_evidence_text(value, recorder)
    if isinstance(value, list):
        return [_sanitize_evidence_value(item, recorder) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_evidence_value(item, recorder) for item in value]
    if isinstance(value, dict):
        return {
            _redact_evidence_text(str(key), recorder): _sanitize_evidence_value(item, recorder)
            for key, item in value.items()
        }
    return value


def _redact_evidence_text(value: str, recorder: RcEvidenceRecorder) -> str:
    """Redact secrets and local absolute filesystem paths from evidence text."""

    redacted = URI_USERINFO_PATTERN.sub(r"\1[redacted]@", redact_text(value))
    replacements: list[tuple[str, str]] = []
    if recorder.repo_root is not None:
        replacements.append((str(recorder.repo_root), "<repo>"))
    replacements.append((str(recorder.evidence_dir), "<evidence>"))
    for path, marker in sorted(replacements, key=lambda item: len(item[0]), reverse=True):
        if path:
            redacted = redacted.replace(path, marker)
    redacted = FILE_URI_PATTERN.sub("file:///<redacted-path>", redacted)
    redacted = RELATIVE_LOCAL_PATH_PATTERN.sub("<redacted-path>", redacted)
    redacted = WINDOWS_UNC_PATH_PATTERN.sub(r"\1<redacted-path>", redacted)
    redacted = WINDOWS_DRIVE_PATH_PATTERN.sub(r"\1<redacted-path>", redacted)
    return POSIX_ABSOLUTE_PATH_PATTERN.sub(r"\1<redacted-path>", redacted)


def sha256_file(path: Path) -> str:
    """Return the SHA256 hex digest for a file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_protocol_fixture_files(fixture_root: Path) -> tuple[Path, ...]:
    """Validate and return the exact seven approved protocol fixture files."""

    if fixture_root.is_symlink() or not fixture_root.is_dir():
        raise ValueError("protocol fixture root must be a regular directory")
    resolved_root = fixture_root.resolve(strict=True)
    expected_files = {
        *PROTOCOL_FIXTURE_SUPPORT_FILES,
        *PROTOCOL_FIXTURE_PATHS.values(),
    }
    expected_directories = {relative.parent for relative in PROTOCOL_FIXTURE_PATHS.values()}
    actual_files: set[Path] = set()
    actual_directories: set[Path] = set()
    for member in fixture_root.rglob("*"):
        relative = member.relative_to(fixture_root)
        if member.is_symlink():
            raise ValueError(f"protocol fixture member must not be a symlink: {relative}")
        if member.is_dir():
            actual_directories.add(relative)
        elif member.is_file():
            actual_files.add(relative)
        else:
            raise ValueError(f"protocol fixture member must be a regular file: {relative}")
    if actual_files != expected_files or actual_directories != expected_directories:
        raise ValueError("protocol fixture tree members do not match the exact release allowlist")

    manifest_path = fixture_root / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("protocol fixture manifest is not readable JSON") from exc
    if not isinstance(manifest, dict) or set(manifest) != {"upstream", "fixtures"}:
        raise ValueError("protocol fixture manifest has unexpected top-level fields")
    expected_upstream = {
        "repository": PROTOCOL_FIXTURE_REPOSITORY,
        "commit": PROTOCOL_FIXTURE_COMMIT,
        "license": PROTOCOL_FIXTURE_LICENSE,
    }
    if manifest.get("upstream") != expected_upstream:
        raise ValueError("protocol fixture upstream metadata does not match the release pin")
    fixtures = manifest.get("fixtures")
    if not isinstance(fixtures, list) or len(fixtures) != len(PROTOCOL_FIXTURE_SHA256):
        raise ValueError("protocol fixture manifest must contain exactly five entries")

    seen_revisions: set[str] = set()
    seen_paths: set[str] = set()
    for item in fixtures:
        if not isinstance(item, dict) or set(item) != {"revision", "path", "url", "sha256"}:
            raise ValueError("protocol fixture entry fields do not match the release contract")
        revision = item.get("revision")
        raw_path = item.get("path")
        if not isinstance(revision, str) or not isinstance(raw_path, str):
            raise ValueError("protocol fixture revision and path must be strings")
        if revision in seen_revisions or raw_path in seen_paths:
            raise ValueError("protocol fixture revisions and paths must be unique")
        seen_revisions.add(revision)
        seen_paths.add(raw_path)
        if (
            PurePosixPath(raw_path).is_absolute()
            or PureWindowsPath(raw_path).is_absolute()
            or ".." in PurePosixPath(raw_path).parts
            or ".." in PureWindowsPath(raw_path).parts
        ):
            raise ValueError("protocol fixture path must be confined and relative")
        expected_path = PROTOCOL_FIXTURE_PATHS.get(revision)
        expected_hash = PROTOCOL_FIXTURE_SHA256.get(revision)
        expected_url = PROTOCOL_FIXTURE_URLS.get(revision)
        if expected_path is None or raw_path != expected_path.as_posix():
            raise ValueError("protocol fixture revision path does not match the release pin")
        if item.get("url") != expected_url or item.get("sha256") != expected_hash:
            raise ValueError("protocol fixture URL or SHA-256 does not match the release pin")
        candidate = fixture_root / expected_path
        try:
            resolved_candidate = candidate.resolve(strict=True)
        except OSError as exc:
            raise ValueError("protocol fixture schema file is missing") from exc
        if not resolved_candidate.is_relative_to(resolved_root):
            raise ValueError("protocol fixture schema resolves outside the fixture root")
        if candidate.is_symlink() or not candidate.is_file():
            raise ValueError("protocol fixture schema must be a regular non-symlink file")
        if sha256_file(candidate) != expected_hash:
            raise ValueError("protocol fixture schema content does not match the pinned SHA-256")
    if seen_revisions != set(PROTOCOL_FIXTURE_SHA256):
        raise ValueError("protocol fixture revisions do not match the exact release pin")

    return (*PROTOCOL_FIXTURE_SUPPORT_FILES, *PROTOCOL_FIXTURE_PATHS.values())


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
            errors="replace",
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
    except OSError as exc:
        return RcCommandResult(
            command=redacted_command,
            cwd=str(cwd),
            returncode=127,
            stdout="",
            stderr=redact_text(f"Command failed to start: {exc}"),
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


def build_publish_plan(
    paths: RcPaths,
    *,
    target: str,
    execute: bool = False,
    dry_run: bool = True,
) -> PublishPlan:
    """Build a guarded twine upload plan for existing MCP Unified artifacts."""

    repository_url = PUBLISH_TARGET_REPOSITORIES.get(target)
    if repository_url is None:
        valid_targets = ", ".join(sorted(PUBLISH_TARGET_REPOSITORIES))
        raise ValueError(f"invalid publish target {target!r}; expected one of: {valid_targets}")

    if execute and os.environ.get(PUBLISH_ALLOW_ENV) != "1":
        raise RuntimeError(f"live MCP Unified publishing requires {PUBLISH_ALLOW_ENV}=1")

    wheels, sdists = _dist_artifacts(paths)
    if not wheels or not sdists:
        raise FileNotFoundError("expected built wheel and sdist in .artifacts/mcp-unified-rc/dist; run build first")
    if len(wheels) != 1 or len(sdists) != 1:
        raise ValueError(
            "expected exactly one wheel and one sdist in .artifacts/mcp-unified-rc/dist; "
            f"found wheels={[path.name for path in wheels]}, "
            f"sdists={[path.name for path in sdists]}"
        )

    artifact_paths = [*wheels, *sdists]
    return PublishPlan(
        target=target,
        repository_url=repository_url,
        dry_run=False if execute else dry_run,
        execute=execute,
        artifact_filenames=[artifact.name for artifact in artifact_paths],
        command=[
            sys.executable,
            "-m",
            "twine",
            "upload",
            "--repository-url",
            repository_url,
            "--non-interactive",
            *[str(artifact) for artifact in artifact_paths],
        ],
    )


def run_publish_plan(
    paths: RcPaths,
    *,
    target: str,
    execute: bool,
    dry_run: bool,
) -> int:
    """Record or execute a guarded publish plan for built artifacts."""

    started = time.perf_counter()
    recorder = _new_recorder(paths)
    _record_existing_artifacts(paths, recorder)
    try:
        plan = build_publish_plan(
            paths,
            target=target,
            execute=execute,
            dry_run=dry_run,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        recorder.record(
            phase="publish_plan",
            name="twine_upload_plan",
            status="failed",
            duration_ms=int((time.perf_counter() - started) * 1000),
            reason=str(exc),
            details={"target": target, "execute": execute, "dry_run": dry_run},
        )
        return _write_and_report(recorder)

    if not plan.execute:
        recorder.record(
            phase="publish_plan",
            name="twine_upload_plan",
            status="passed",
            duration_ms=int((time.perf_counter() - started) * 1000),
            details={"plan": plan.as_dict()},
        )
        return _write_and_report(recorder)

    try:
        result = run_command(plan.command, cwd=paths.repo_root, timeout=300)
    except (RuntimeError, subprocess.SubprocessError, OSError) as exc:
        logger.exception("MCP Unified publish upload command failed unexpectedly")
        recorder.record(
            phase="publish_plan",
            name="twine_upload",
            status="failed",
            duration_ms=int((time.perf_counter() - started) * 1000),
            reason=f"unexpected upload execution error: {exc}",
            details={"plan": plan.as_dict()},
        )
    else:
        _record_command_result(
            recorder,
            phase="publish_plan",
            name="twine_upload",
            result=result,
            details={"plan": plan.as_dict()},
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


def run_portable_gate(paths: RcPaths) -> int:
    """Build artifacts and run only the installed package protocol gates."""

    recorder = _new_recorder(paths)
    _run_build(paths, recorder)
    _run_artifact_gate(paths, recorder)
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
        "portable-gate",
        "all",
    ):
        subparsers.add_parser(name, help=f"Run the {name} RC phase.")
    publish_parser = subparsers.add_parser(
        "publish-plan",
        help="Build or execute a guarded MCP Unified publish plan.",
    )
    publish_parser.add_argument(
        "--target",
        choices=tuple(PUBLISH_TARGET_REPOSITORIES),
        default="testpypi",
        help="Publish target repository. Defaults to TestPyPI.",
    )
    publish_parser.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Record the publish command without uploading artifacts. Use --no-dry-run to request guarded upload.",
    )
    publish_parser.add_argument(
        "--execute",
        action="store_true",
        help=f"Run twine upload. Requires {PUBLISH_ALLOW_ENV}=1.",
    )
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
    if args.command == "portable-gate":
        return run_portable_gate(paths)
    if args.command == "publish-plan":
        return run_publish_plan(
            paths,
            target=args.target,
            execute=args.execute or not args.dry_run,
            dry_run=args.dry_run,
        )
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

        _record_jsonschema_dependency(
            recorder,
            artifact=wheels[0],
            kind="wheel",
        )
        _record_jsonschema_dependency(
            recorder,
            artifact=sdists[0],
            kind="sdist",
        )

    _record_protocol_fixture_provenance(paths, recorder)

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

    if wheels and sdists:
        _run_installed_protocol_suites(
            paths,
            recorder,
            artifact=wheels[0],
            kind="wheel",
        )
        _run_installed_protocol_suites(
            paths,
            recorder,
            artifact=sdists[0],
            kind="sdist",
        )
        _run_installed_artifact_consumer(
            paths,
            recorder,
            wheel=wheels[0],
            sdist=sdists[0],
        )


def _distribution_metadata(artifact: Path, *, kind: str) -> Message:
    """Return RFC package metadata from a wheel or source distribution."""

    if kind == "wheel":
        with zipfile.ZipFile(artifact) as archive:
            members = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
            if len(members) != 1:
                raise ValueError("wheel must contain exactly one dist-info/METADATA")
            raw = archive.read(members[0]).decode("utf-8")
        return Parser().parsestr(raw)

    if kind != "sdist":
        raise ValueError(f"unsupported distribution kind: {kind}")
    with tarfile.open(artifact, "r:gz") as archive:
        members = [
            member
            for member in archive.getmembers()
            if member.name.endswith("/PKG-INFO") and len(Path(member.name).parts) == 2
        ]
        if len(members) != 1:
            raise ValueError("sdist must contain exactly one root PKG-INFO")
        extracted = archive.extractfile(members[0])
        if extracted is None:
            raise ValueError("sdist root PKG-INFO is not readable")
        raw = extracted.read().decode("utf-8")
    return Parser().parsestr(raw)


def _jsonschema_base_dependency(metadata: Message) -> str:
    """Return the exact unmarked jsonschema base requirement or fail closed."""

    matches: dict[frozenset[str], str] = {}
    for value in metadata.get_all("Requires-Dist") or []:
        requirement, separator, marker = value.partition(";")
        name_match = re.match(r"\s*([A-Za-z0-9_.-]+)", requirement)
        if name_match is None:
            continue
        name = name_match.group(1).lower().replace("_", "-")
        if name == "jsonschema" and (not separator or not marker.strip()):
            normalized = requirement.strip()
            specifier_text = normalized[name_match.end() :].replace(" ", "").strip("()")
            specifiers = frozenset(part for part in specifier_text.split(",") if part)
            matches.setdefault(specifiers, normalized)
    if len(matches) != 1:
        raise ValueError("metadata must declare one unmarked jsonschema base dependency")

    specifiers, requirement = next(iter(matches.items()))
    if specifiers != JSONSCHEMA_REQUIREMENT_BOUNDS:
        raise ValueError("jsonschema base dependency must be bounded to >=4.23,<5")
    return requirement


def _record_jsonschema_dependency(
    recorder: RcEvidenceRecorder,
    *,
    artifact: Path,
    kind: str,
) -> None:
    """Record the direct bounded validator dependency for one artifact."""

    started = time.perf_counter()
    try:
        requirement = _jsonschema_base_dependency(_distribution_metadata(artifact, kind=kind))
    except (OSError, UnicodeError, ValueError, zipfile.BadZipFile, tarfile.TarError) as exc:
        recorder.record(
            phase="artifact_gate",
            name=f"{kind}_jsonschema_base_dependency",
            status="failed",
            duration_ms=int((time.perf_counter() - started) * 1000),
            reason=str(exc),
        )
        return
    recorder.record(
        phase="artifact_gate",
        name=f"{kind}_jsonschema_base_dependency",
        status="passed",
        duration_ms=int((time.perf_counter() - started) * 1000),
        details={"requirement": requirement},
    )


def _record_protocol_fixture_provenance(
    paths: RcPaths,
    recorder: RcEvidenceRecorder,
) -> None:
    """Verify and record the pinned normative schema commit and five hashes."""

    started = time.perf_counter()
    fixture_root = paths.repo_root / PROTOCOL_FIXTURE_ROOT
    try:
        _validated_protocol_fixture_files(fixture_root)
    except (OSError, TypeError, ValueError) as exc:
        recorder.record(
            phase="artifact_gate",
            name="normative_fixture_provenance",
            status="failed",
            duration_ms=int((time.perf_counter() - started) * 1000),
            reason=str(exc),
        )
        return
    recorder.record(
        phase="artifact_gate",
        name="normative_fixture_provenance",
        status="passed",
        duration_ms=int((time.perf_counter() - started) * 1000),
        details={
            "commit": PROTOCOL_FIXTURE_COMMIT,
            "sha256": PROTOCOL_FIXTURE_SHA256,
        },
    )


def _run_installed_protocol_suites(
    paths: RcPaths,
    recorder: RcEvidenceRecorder,
    *,
    artifact: Path,
    kind: str,
) -> None:
    """Install one artifact cleanly and run all five protocol suites against it."""

    with tempfile.TemporaryDirectory(prefix=f"mcp-unified-rc-protocol-{kind}-") as temp_name:
        temp_dir = Path(temp_name)
        venv_dir = temp_dir / ".venv"
        if not _create_venv(
            venv_dir,
            recorder,
            phase="artifact_gate",
            name=f"{kind}_protocol_venv",
        ):
            return
        python_path = _venv_executable(venv_dir, "python")
        install_result = run_command(
            [
                str(python_path),
                "-m",
                "pip",
                "install",
                f"{artifact}[dev]",
                OFFICIAL_SDK_REQUIREMENT,
            ],
            cwd=temp_dir,
            timeout=600,
            env={"PIP_NO_CACHE_DIR": "1", "PYTHONNOUSERSITE": "1"},
        )
        _record_command_result(
            recorder,
            phase="artifact_gate",
            name=f"{kind}_protocol_install",
            result=install_result,
        )
        if install_result.returncode != 0:
            return

        test_root, installed_test_suites = _prepare_installed_protocol_test_tree(
            paths,
            temp_dir,
        )
        import_result = run_command(
            [
                str(python_path),
                "-c",
                (
                    "from pathlib import Path; import mcp_unified, sysconfig; "
                    "module_path = Path(mcp_unified.__file__).resolve(); "
                    "purelib = Path(sysconfig.get_paths()['purelib']).resolve(); "
                    "assert module_path.is_relative_to(purelib); "
                    "print('MCP_UNIFIED_INSTALLED_IMPORT_OK')"
                ),
            ],
            cwd=test_root,
            timeout=60,
            env={"PYTHONNOUSERSITE": "1"},
        )
        _record_command_result(
            recorder,
            phase="artifact_gate",
            name=f"{kind}_installed_protocol_import",
            result=import_result,
        )
        if import_result.returncode != 0:
            return
        suite_result = run_command(
            [
                str(python_path),
                "-m",
                "pytest",
                "-c",
                str(test_root / "pytest-artifact-gate.ini"),
                "--noconftest",
                *[str(path) for path in installed_test_suites],
                "-q",
            ],
            cwd=test_root,
            timeout=1_200,
            env={"PYTHONNOUSERSITE": "1"},
        )
        _record_command_result(
            recorder,
            phase="artifact_gate",
            name=f"{kind}_installed_protocol_suites",
            result=suite_result,
        )
        if suite_result.returncode != 0:
            return
        sdk_result = run_command(
            [str(python_path), str(test_root / OFFICIAL_SDK_SMOKE_SCRIPT)],
            cwd=test_root,
            timeout=60,
            env={
                "MCP_UNIFIED_FORBIDDEN_CHECKOUT": str(paths.repo_root),
                "PYTHONNOUSERSITE": "1",
            },
        )
        if sdk_result.returncode == 0 and (
            sdk_result.stdout != "MCP_UNIFIED_OFFICIAL_SDK_STDIO_OK\n" or sdk_result.stderr
        ):
            sdk_result.returncode = 1
            sdk_result.stderr = "official SDK smoke did not emit only its success marker"
        _record_command_result(
            recorder,
            phase="artifact_gate",
            name=f"{kind}_official_sdk_stdio_interop",
            result=sdk_result,
            details={
                "requirement": OFFICIAL_SDK_REQUIREMENT,
                "sdk_index": OFFICIAL_SDK_INDEX_URL,
                "tier": OFFICIAL_SDK_TIER,
                "release": OFFICIAL_SDK_RELEASE_URL,
                "tag_commit": OFFICIAL_SDK_TAG_COMMIT,
                "import_provenance": "mcp and mcp_unified resolved beneath the clean virtualenv purelib",
            },
        )


def _run_installed_artifact_consumer(
    paths: RcPaths,
    recorder: RcEvidenceRecorder,
    *,
    wheel: Path,
    sdist: Path,
) -> None:
    """Run the downstream wheel/sdist consumer only from a mirrored test tree."""

    with tempfile.TemporaryDirectory(prefix="mcp-unified-rc-consumer-") as temp_name:
        temp_dir = Path(temp_name)
        test_root, _ = _prepare_installed_protocol_test_tree(paths, temp_dir)
        local_dist = test_root / "dist"
        local_dist.mkdir()
        for artifact in (wheel, sdist):
            shutil.copy2(artifact, local_dist / artifact.name)
        consumer_result = run_command(
            [
                sys.executable,
                "-m",
                "pytest",
                "-c",
                str(test_root / "pytest-artifact-gate.ini"),
                "--noconftest",
                str(test_root / PROTOCOL_ARTIFACT_CONSUMER_TEST),
                "-q",
            ],
            cwd=test_root,
            timeout=1_200,
            env={
                "MCP_UNIFIED_TEST_DIST_DIR": str(local_dist),
                "PYTHONNOUSERSITE": "1",
            },
        )
        _record_command_result(
            recorder,
            phase="artifact_gate",
            name="installed_artifact_consumer",
            result=consumer_result,
        )


def _prepare_installed_protocol_test_tree(
    paths: RcPaths,
    temp_dir: Path,
) -> tuple[Path, tuple[Path, ...]]:
    """Copy only installed-artifact protocol test inputs outside the checkout."""

    test_root = temp_dir / "protocol-test-root"
    test_dir = test_root / "tldw_Server_API" / "app" / "core" / "MCP_unified" / "tests"
    test_dir.mkdir(parents=True)
    installed_suites: list[Path] = []
    for relative_path in PROTOCOL_TEST_SUITES:
        target = test_dir / relative_path.name
        shutil.copy2(paths.repo_root / relative_path, target)
        installed_suites.append(target)

    for relative_path in (
        PROTOCOL_ARTIFACT_CONSUMER_TEST,
        PROTOCOL_ARTIFACT_CONSUMER_TEST.with_name("mcp_unified_artifact_test_utils.py"),
        Path("Helper_Scripts") / "mcp_unified_rc.py",
        OFFICIAL_SDK_SMOKE_SCRIPT,
    ):
        target = test_root / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(paths.repo_root / relative_path, target)

    fixture_source = paths.repo_root / PROTOCOL_FIXTURE_ROOT
    fixture_target = test_dir / "fixtures" / "mcp_protocol"
    for relative_path in _validated_protocol_fixture_files(fixture_source):
        target = fixture_target / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(
            fixture_source / relative_path,
            target,
            follow_symlinks=False,
        )
    shutil.copy2(
        paths.package_project / "pytest-artifact-gate.ini",
        test_root / "pytest-artifact-gate.ini",
    )
    package_config = test_root / "apps" / "mcp-unified" / "pytest-artifact-gate.ini"
    package_config.parent.mkdir(parents=True)
    shutil.copy2(
        paths.package_project / "pytest-artifact-gate.ini",
        package_config,
    )
    workflow_target = test_root / ".github" / "workflows" / "mcp-unified-rc.yml"
    workflow_target.parent.mkdir(parents=True)
    shutil.copy2(
        paths.repo_root / ".github" / "workflows" / "mcp-unified-rc.yml",
        workflow_target,
    )
    shutil.copy2(
        paths.repo_root / ".github" / "license-first-paths.json",
        test_root / ".github" / "license-first-paths.json",
    )
    return test_root, tuple(installed_suites)


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
            _record_pip_install_result(
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
            _run_extra_tier_checks(
                recorder,
                extra=extra,
                repo_root=paths.repo_root,
                python_path=python_path,
                venv_dir=venv_dir,
                temp_dir=temp_dir,
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
    _run_user_guide_wheel_uat(
        paths,
        recorder,
        wheel,
        phase="cli_uat",
        name="user_guide_wheel_mode",
        mode="cli",
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
    _run_user_guide_wheel_uat(
        paths,
        recorder,
        wheel,
        phase="smoke_uat",
        name="user_guide_smoke_transports",
        mode="smoke",
    )


def _run_extra_tier_checks(
    recorder: RcEvidenceRecorder,
    *,
    extra: str,
    repo_root: Path,
    python_path: Path,
    venv_dir: Path,
    temp_dir: Path,
) -> None:
    """Run extra-specific installed-package checks required by the RC spec."""

    if extra == "core":
        result = run_command(
            [str(_venv_executable(venv_dir, "mcp-unified-gateway")), "package-info"],
            cwd=temp_dir,
            timeout=60,
        )
        _record_command_result(
            recorder,
            phase="extras_matrix",
            name="core_package_info",
            result=result,
        )
        return
    if extra == "gateway":
        _run_gateway_extra_checks(recorder, python_path=python_path, venv_dir=venv_dir, temp_dir=temp_dir)
        return
    if extra == "sqlite":
        result = run_command(
            [
                str(python_path),
                "-c",
                (
                    "import asyncio; "
                    "from pathlib import Path; "
                    "from mcp_unified.storage.sqlite import SQLiteMCPStore; "
                    "from mcp_unified.tool_use_reporting.sqlite import SQLiteToolUseEventStore; "
                    "profile_store = SQLiteMCPStore(Path('mcp-store.db')); "
                    "profile_store.close(); "
                    "event_store = SQLiteToolUseEventStore(Path('mcp-tool-events.db')); "
                    "asyncio.run(event_store.aclose()); "
                    "print('sqlite-ok')"
                ),
            ],
            cwd=temp_dir,
            timeout=60,
        )
        _record_command_result(
            recorder,
            phase="extras_matrix",
            name="sqlite_storage_smoke",
            result=result,
        )
        return
    if extra == "dev":
        result = run_command(
            [
                str(python_path),
                "-m",
                "pytest",
                "-c",
                str(Path("apps") / "mcp-unified" / "pytest-artifact-gate.ini"),
                ".github/tests/test_mcp_unified_artifact_gate.py",
                "-q",
            ],
            cwd=repo_root,
            timeout=300,
            env={"PYTHONPATH": str(repo_root / "apps" / "mcp-unified" / "src")},
        )
        _record_command_result(
            recorder,
            phase="extras_matrix",
            name="dev_artifact_gate_selection",
            result=result,
        )


def _run_gateway_extra_checks(
    recorder: RcEvidenceRecorder,
    *,
    python_path: Path,
    venv_dir: Path,
    temp_dir: Path,
) -> None:
    """Run installed gateway-extra import and config-validation checks."""

    import_result = run_command(
        [
            str(python_path),
            "-c",
            ("import mcp_unified.gateway.cli; import mcp_unified.gateway.config; print('gateway-ok')"),
        ],
        cwd=temp_dir,
        timeout=60,
    )
    _record_command_result(
        recorder,
        phase="extras_matrix",
        name="gateway_module_imports",
        result=import_result,
    )
    config_path = temp_dir / "gateway-extra-config.json"
    config_path.write_text(
        json.dumps(
            {
                "store": {"kind": "memory"},
                "default_preset_id": "project-researcher",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    validation_result = run_command(
        [
            str(_venv_executable(venv_dir, "mcp-unified-gateway")),
            "validate-config",
            str(config_path),
        ],
        cwd=temp_dir,
        timeout=60,
    )
    _record_command_result(
        recorder,
        phase="extras_matrix",
        name="gateway_config_validation",
        result=validation_result,
    )


def _run_user_guide_wheel_uat(
    paths: RcPaths,
    recorder: RcEvidenceRecorder,
    wheel: Path,
    *,
    phase: str,
    name: str,
    mode: str,
) -> None:
    """Run the standalone user-guide UAT harness against the built wheel."""

    report_path = paths.evidence_dir / f"{phase}-{name}.json"
    with tempfile.TemporaryDirectory(prefix=f"mcp-unified-rc-{phase}-") as temp_name:
        workspace = Path(temp_name) / "workspace"
        command = [
            sys.executable,
            str(USER_GUIDE_UAT_SCRIPT),
            "--repo-root",
            str(paths.repo_root),
            "--wheel",
            str(wheel),
            "--workspace",
            str(workspace),
            "--mode",
            mode,
            "--json-report",
            str(report_path),
        ]
        result = run_command(command, cwd=paths.repo_root, timeout=900)
    _record_user_guide_uat_result(
        recorder,
        phase=phase,
        name=name,
        result=result,
        report_path=report_path,
    )


def _record_user_guide_uat_result(
    recorder: RcEvidenceRecorder,
    *,
    phase: str,
    name: str,
    result: RcCommandResult,
    report_path: Path,
) -> None:
    """Record the user-guide UAT command and its structured report summary."""

    report = _read_json_file(report_path)
    details: dict[str, Any] = {"json_report": str(report_path)}
    if isinstance(report, dict):
        details["uat_summary"] = report.get("summary")
        details["uat_ok"] = report.get("ok")
        details["uat_failed_steps"] = [
            step.get("id")
            for step in report.get("steps", [])
            if isinstance(step, dict) and step.get("status") == "failed"
        ]

    if _uat_dependency_resolution_unavailable(result, report):
        recorder.record(
            phase=phase,
            name=name,
            status="skipped",
            duration_ms=result.duration_ms,
            reason=PIP_DEPENDENCY_OUTAGE_REASON,
            details={
                "command": _command_result_as_evidence(result, recorder),
                **details,
            },
            required=False,
        )
        return
    _record_command_result(
        recorder,
        phase=phase,
        name=name,
        result=result,
        details=details,
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
        _record_pip_install_result(
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
            "Publishing is guarded: publish-plan is dry-run by default and live upload requires MCP_UNIFIED_ALLOW_PUBLISH=1.",
            "Remote runtime UAT requires MCP_UNIFIED_GATEWAY_URL or --gateway-url and is optional in local RC runs.",
        ],
        repo_root=paths.repo_root,
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
    result_details = {"command": _command_result_as_evidence(result, recorder)}
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


def _command_result_as_evidence(
    result: RcCommandResult,
    recorder: RcEvidenceRecorder,
) -> dict[str, Any]:
    """Return a command result payload safe for persisted evidence."""

    return _sanitize_evidence_value(result.as_dict(), recorder)


def _record_pip_install_result(
    recorder: RcEvidenceRecorder,
    *,
    phase: str,
    name: str,
    result: RcCommandResult,
) -> None:
    """Record dependency-resolving pip installs, allowing local offline skips."""

    if _pip_dependency_resolution_unavailable(result):
        recorder.record(
            phase=phase,
            name=name,
            status="skipped",
            duration_ms=result.duration_ms,
            reason=PIP_DEPENDENCY_OUTAGE_REASON,
            details={"command": _command_result_as_evidence(result, recorder)},
            required=False,
        )
        return
    _record_command_result(recorder, phase=phase, name=name, result=result)


def _pip_dependency_resolution_unavailable(result: RcCommandResult) -> bool:
    """Return whether pip failed only because this local run cannot reach indexes."""

    if result.returncode == 0 or os.environ.get("GITHUB_ACTIONS"):
        return False
    output = f"{result.stdout}\n{result.stderr}".lower()
    return _dependency_resolution_unavailable_text(output)


def _uat_dependency_resolution_unavailable(
    result: RcCommandResult,
    report: Any,
) -> bool:
    """Return whether user-guide UAT failed only because indexes were unreachable."""

    if result.returncode == 0 or os.environ.get("GITHUB_ACTIONS"):
        return False
    output = f"{result.stdout}\n{result.stderr}\n{json.dumps(report, sort_keys=True, default=str)}".lower()
    return _dependency_resolution_unavailable_text(output)


def _dependency_resolution_unavailable_text(output: str) -> bool:
    """Return whether output contains dependency and network failure markers."""

    return any(marker in output for marker in PIP_DEPENDENCY_FAILURE_MARKERS) and any(
        marker in output for marker in PIP_NETWORK_FAILURE_MARKERS
    )


def _read_json_file(path: Path) -> Any:
    """Read a JSON file if present, returning ``None`` for missing or invalid files."""

    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _create_venv(
    venv_dir: Path,
    recorder: RcEvidenceRecorder,
    *,
    phase: str,
    name: str,
) -> bool:
    started = time.perf_counter()
    try:
        venv.EnvBuilder(
            with_pip=True,
            clear=True,
            symlinks=os.name != "nt",
        ).create(venv_dir)
    except (
        OSError,
        subprocess.SubprocessError,
        ValueError,
    ) as exc:  # pragma: no cover - platform/environment-specific.
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
    wheels = sorted([*paths.dist_dir.glob("mcp_unified-*.whl"), *paths.dist_dir.glob("mcp-unified-*.whl")])
    sdists = sorted([*paths.dist_dir.glob("mcp_unified-*.tar.gz"), *paths.dist_dir.glob("mcp-unified-*.tar.gz")])
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
        "PACKAGE_AUTHORS",
        "PACKAGE_MAINTAINERS",
        "PACKAGE_KEYWORDS",
        "PACKAGE_CLASSIFIERS",
        "PACKAGE_URLS",
        "LICENSE_FILES",
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
        if target_name in wanted and value_node is not None:
            try:
                constants[target_name] = ast.literal_eval(value_node)
            except (SyntaxError, ValueError, TypeError):
                if isinstance(value_node, ast.Constant):
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
    logger.info("Wrote RC evidence JSON: {}", json_path)
    logger.info("Wrote RC evidence Markdown: {}", markdown_path)
    if recorder.has_required_failures():
        failed_checks = ", ".join(
            f"{result['phase']}/{result['name']}"
            for result in recorder.results
            if result.get("required", True) and result["status"] == "failed"
        )
        logger.error("RC status: failed ({})", failed_checks)
        return 1
    logger.info("RC status: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
