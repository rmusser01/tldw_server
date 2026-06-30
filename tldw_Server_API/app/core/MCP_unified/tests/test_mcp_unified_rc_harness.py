"""Unit coverage for the internal MCP Unified release-candidate harness."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from Helper_Scripts import mcp_unified_rc

pytestmark = pytest.mark.unit

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility.
    import tomli as tomllib

REPO_ROOT = Path(__file__).resolve().parents[5]
USER_GUIDE_UAT_PATH = (
    REPO_ROOT
    / "Helper_Scripts"
    / "Testing-related"
    / "mcp_standalone_user_guide_uat.py"
)


def _load_user_guide_harness() -> ModuleType:
    """Load the standalone user-guide UAT harness from its script path."""

    spec = importlib.util.spec_from_file_location(
        "_mcp_standalone_user_guide_uat",
        USER_GUIDE_UAT_PATH,
    )
    assert spec is not None  # nosec B101
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None  # nosec B101
    spec.loader.exec_module(module)
    return module


def test_default_paths_point_to_apps_package() -> None:
    """Default RC paths should target the apps/mcp-unified package project."""

    paths = mcp_unified_rc.RcPaths.from_repo_root(Path("/repo"))

    assert paths.package_project == Path("/repo/apps/mcp-unified")  # nosec B101
    assert paths.package_src == Path("/repo/apps/mcp-unified/src/mcp_unified")  # nosec B101
    assert paths.evidence_dir == Path("/repo/.artifacts/mcp-unified-rc")  # nosec B101


def _create_publish_plan_artifacts(tmp_path: Path) -> mcp_unified_rc.RcPaths:
    """Create placeholder dist files for publish-plan unit tests."""

    paths = mcp_unified_rc.RcPaths.from_repo_root(tmp_path)
    paths.dist_dir.mkdir(parents=True)
    (paths.dist_dir / "mcp_unified-0.1.0-py3-none-any.whl").write_text(
        "wheel",
        encoding="utf-8",
    )
    (paths.dist_dir / "mcp_unified-0.1.0.tar.gz").write_text(
        "sdist",
        encoding="utf-8",
    )
    return paths


def test_rc_publish_plan_is_dry_run_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Publish plans should default to a non-uploading TestPyPI dry run."""

    monkeypatch.delenv("MCP_UNIFIED_ALLOW_PUBLISH", raising=False)
    paths = _create_publish_plan_artifacts(tmp_path)
    parser = mcp_unified_rc.build_parser()

    parsed = parser.parse_args(["publish-plan", "--target", "testpypi"])
    plan = mcp_unified_rc.build_publish_plan(
        paths,
        target=parsed.target,
        execute=parsed.execute,
    )

    assert parsed.command == "publish-plan"  # nosec B101
    assert plan.target == "testpypi"  # nosec B101
    assert plan.repository_url == "https://test.pypi.org/legacy/"  # nosec B101
    assert plan.execute is False  # nosec B101
    assert plan.dry_run is True  # nosec B101
    assert plan.artifact_filenames == [  # nosec B101
        "mcp_unified-0.1.0-py3-none-any.whl",
        "mcp_unified-0.1.0.tar.gz",
    ]
    assert plan.command[:5] == [  # nosec B101
        sys.executable,
        "-m",
        "twine",
        "upload",
        "--repository-url",
    ]
    assert "https://test.pypi.org/legacy/" in plan.command  # nosec B101
    assert "--non-interactive" in plan.command  # nosec B101
    assert not any("token" in part.lower() for part in plan.command)  # nosec B101


def test_rc_publish_plan_parser_supports_no_dry_run() -> None:
    """The publish-plan dry-run flag should be a real boolean option."""

    parser = mcp_unified_rc.build_parser()

    parsed = parser.parse_args(["publish-plan", "--target", "testpypi", "--no-dry-run"])

    assert parsed.command == "publish-plan"  # nosec B101
    assert parsed.dry_run is False  # nosec B101
    assert parsed.execute is False  # nosec B101


def test_rc_publish_plan_rejects_stale_extra_artifacts(tmp_path: Path) -> None:
    """Publish plans should not include stale dist artifacts."""

    paths = _create_publish_plan_artifacts(tmp_path)
    (paths.dist_dir / "mcp_unified-0.1.0+stale-py3-none-any.whl").write_text(
        "stale wheel",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="expected exactly one wheel and one sdist"):
        mcp_unified_rc.build_publish_plan(paths, target="testpypi")


def test_rc_publish_plan_execute_requires_opt_in_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Live publish plans should require an explicit environment opt-in."""

    paths = _create_publish_plan_artifacts(tmp_path)
    monkeypatch.delenv("MCP_UNIFIED_ALLOW_PUBLISH", raising=False)

    with pytest.raises(RuntimeError, match="MCP_UNIFIED_ALLOW_PUBLISH"):
        mcp_unified_rc.build_publish_plan(
            paths,
            target="pypi",
            execute=True,
        )

    monkeypatch.setenv("MCP_UNIFIED_ALLOW_PUBLISH", "1")
    plan = mcp_unified_rc.build_publish_plan(
        paths,
        target="pypi",
        execute=True,
    )

    assert plan.target == "pypi"  # nosec B101
    assert plan.repository_url == "https://upload.pypi.org/legacy/"  # nosec B101
    assert plan.execute is True  # nosec B101
    assert plan.dry_run is False  # nosec B101


def test_run_publish_plan_records_dry_run_evidence(tmp_path: Path) -> None:
    """run_publish_plan should directly record dry-run evidence."""

    paths = _create_publish_plan_artifacts(tmp_path)

    exit_code = mcp_unified_rc.run_publish_plan(
        paths,
        target="testpypi",
        execute=False,
        dry_run=True,
    )

    evidence = json.loads(
        (paths.evidence_dir / mcp_unified_rc.EVIDENCE_JSON).read_text(
            encoding="utf-8",
        )
    )
    result = next(
        item
        for item in evidence["results"]
        if item["phase"] == "publish_plan" and item["name"] == "twine_upload_plan"
    )
    assert exit_code == 0  # nosec B101
    assert result["status"] == "passed"  # nosec B101
    assert result["details"]["plan"]["dry_run"] is True  # nosec B101
    assert result["details"]["plan"]["execute"] is False  # nosec B101


def test_run_publish_plan_records_upload_execution_exception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unexpected upload execution errors should be recorded as evidence."""

    paths = _create_publish_plan_artifacts(tmp_path)
    monkeypatch.setenv("MCP_UNIFIED_ALLOW_PUBLISH", "1")

    def fail_run_command(
        command: list[str],
        *,
        cwd: Path,
        timeout: int = 180,
        env: dict[str, str] | None = None,
    ) -> mcp_unified_rc.RcCommandResult:
        del env, timeout
        if command[:3] == [sys.executable, "-m", "twine"]:
            raise RuntimeError("upload transport failed")
        return mcp_unified_rc.RcCommandResult(
            command=command,
            cwd=str(cwd),
            returncode=0,
            stdout="abc123\n",
            stderr="",
            duration_ms=0,
        )

    monkeypatch.setattr(mcp_unified_rc, "run_command", fail_run_command)

    exit_code = mcp_unified_rc.run_publish_plan(
        paths,
        target="testpypi",
        execute=True,
        dry_run=False,
    )

    evidence = json.loads(
        (paths.evidence_dir / mcp_unified_rc.EVIDENCE_JSON).read_text(
            encoding="utf-8",
        )
    )
    result = next(
        item
        for item in evidence["results"]
        if item["phase"] == "publish_plan" and item["name"] == "twine_upload"
    )
    assert exit_code == 1  # nosec B101
    assert result["status"] == "failed"  # nosec B101
    assert "upload transport failed" in result["reason"]  # nosec B101
    assert result["details"]["plan"]["execute"] is True  # nosec B101


def test_read_package_constants_ignores_literal_eval_type_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Metadata extraction should not crash when literal_eval raises TypeError."""

    metadata_path = tmp_path / "package_metadata.py"
    metadata_path.write_text(
        "\n".join(
            [
                "from typing import Final",
                "PACKAGE_NAME: Final = 'mcp-unified'",
                "PACKAGE_URLS: Final = {'Homepage': 'https://tldwproject.com'}",
            ]
        ),
        encoding="utf-8",
    )
    original_literal_eval = mcp_unified_rc.ast.literal_eval

    def literal_eval_with_type_error(node: object) -> object:
        if isinstance(node, mcp_unified_rc.ast.Dict):
            raise TypeError("unsupported literal")
        return original_literal_eval(node)

    monkeypatch.setattr(mcp_unified_rc.ast, "literal_eval", literal_eval_with_type_error)

    constants = mcp_unified_rc._read_package_constants(metadata_path)

    assert constants["PACKAGE_NAME"] == "mcp-unified"  # nosec B101
    assert "PACKAGE_URLS" not in constants  # nosec B101


def test_user_guide_uat_install_spec_uses_apps_project_by_default() -> None:
    """User-guide UAT install specs should default to the app package project."""

    harness = _load_user_guide_harness()
    wheel = Path("/tmp/mcp_unified-0.1.0-py3-none-any.whl")  # nosec B108
    relative_wheel = Path("dist/mcp_unified-0.1.0-py3-none-any.whl")

    assert harness.default_package_project(Path("/repo")) == Path("/repo/apps/mcp-unified")  # nosec B101
    assert harness.package_install_spec(
        repo_root=Path("/repo"),
        wheel_path=None,
        editable=False,
    ) == ["/repo/apps/mcp-unified[gateway]"]  # nosec B101
    assert harness.package_install_spec(
        repo_root=Path("/repo"),
        wheel_path=None,
        editable=True,
    ) == ["-e", "/repo/apps/mcp-unified[gateway]"]  # nosec B101
    assert harness.package_install_spec(
        repo_root=Path("/repo"),
        wheel_path=wheel,
        editable=False,
    ) == [f"{wheel.resolve()}[gateway]"]  # nosec B101
    assert harness.package_install_spec(
        repo_root=Path("/repo"),
        wheel_path=wheel,
        editable=True,
    ) == [f"{wheel.resolve()}[gateway]"]  # nosec B101
    assert harness.package_install_spec(
        repo_root=Path("/repo"),
        wheel_path=relative_wheel,
        editable=False,
    ) == [f"{relative_wheel.resolve()}[gateway]"]  # nosec B101


def test_user_guide_uat_plan_uses_package_install_args(tmp_path: Path) -> None:
    """Generated UAT plans should pass selected install args to pip."""

    harness = _load_user_guide_harness()
    install_args = [str(tmp_path / "mcp_unified-0.1.0-py3-none-any.whl")]

    plan = harness.build_uat_plan(
        repo_root=Path("/repo"),
        workspace=tmp_path,
        python_executable="python",
        gateway_executable="mcp-unified-gateway",
        smoke_executable="mcp-unified-smoke",
        gateway_url=None,
        package_install_args=install_args,
    )

    install_step = next(step for step in plan if step.step_id == "install_package_boundary")
    assert install_step.command == [  # nosec B101
        str(harness._venv_python(tmp_path / ".venv")),
        "-m",
        "pip",
        "install",
        *install_args,
    ]


def test_user_guide_uat_plan_uses_symlinked_venv_on_posix(tmp_path: Path) -> None:
    """Generated UAT plans should use symlinked virtualenvs on POSIX."""

    harness = _load_user_guide_harness()

    plan = harness.build_uat_plan(
        repo_root=Path("/repo"),
        workspace=tmp_path,
        python_executable="python",
        gateway_executable="mcp-unified-gateway",
        smoke_executable="mcp-unified-smoke",
        gateway_url=None,
        package_install_args=["/repo/apps/mcp-unified[gateway]"],
    )

    create_step = next(step for step in plan if step.step_id == "create_venv")
    expected_command = ["python", "-m", "venv"]
    if os.name != "nt":
        expected_command.append("--symlinks")
    expected_command.append(str(tmp_path / ".venv"))
    assert create_step.command == expected_command  # nosec B101


def test_user_guide_uat_plan_filters_cli_and_smoke_modes(tmp_path: Path) -> None:
    """UAT mode filters should select CLI and smoke slices independently."""

    harness = _load_user_guide_harness()

    cli_plan = harness.build_uat_plan(
        repo_root=Path("/repo"),
        workspace=tmp_path / "cli",
        python_executable="python",
        gateway_executable="mcp-unified-gateway",
        smoke_executable="mcp-unified-smoke",
        gateway_url=None,
        package_install_args=["/repo/apps/mcp-unified[gateway]"],
        mode="cli",
    )
    smoke_plan = harness.build_uat_plan(
        repo_root=Path("/repo"),
        workspace=tmp_path / "smoke",
        python_executable="python",
        gateway_executable="mcp-unified-gateway",
        smoke_executable="mcp-unified-smoke",
        gateway_url=None,
        package_install_args=["/repo/apps/mcp-unified[gateway]"],
        mode="smoke",
    )

    cli_ids = {step.step_id for step in cli_plan}
    smoke_ids = {step.step_id for step in smoke_plan}
    assert "gateway_package_info" in cli_ids  # nosec B101
    assert "smoke_stdio_subprocess" not in cli_ids  # nosec B101
    assert "smoke_websocket" not in cli_ids  # nosec B101
    assert "smoke_stdio_subprocess" in smoke_ids  # nosec B101
    assert "smoke_http" in smoke_ids  # nosec B101
    assert "smoke_websocket" in smoke_ids  # nosec B101
    assert "list_presets" not in smoke_ids  # nosec B101


def test_user_guide_uat_cli_mode_includes_tool_event_export_and_cleanup(
    tmp_path: Path,
) -> None:
    """CLI-mode UAT should include tool-event export and cleanup workflows."""

    harness = _load_user_guide_harness()

    cli_plan = harness.build_uat_plan(
        repo_root=Path("/repo"),
        workspace=tmp_path,
        python_executable="python",
        gateway_executable="mcp-unified-gateway",
        smoke_executable="mcp-unified-smoke",
        gateway_url=None,
        package_install_args=["/repo/apps/mcp-unified[gateway]"],
        mode="cli",
    )

    steps_by_id = {step.step_id: step for step in cli_plan}
    export_step = steps_by_id["tool_events_export"]
    cleanup_step = steps_by_id["tool_events_cleanup"]
    assert export_step.command is not None  # nosec B101
    assert export_step.command[:3] == ["mcp-unified-gateway", "tool-events", "export"]  # nosec B101
    assert "--format" in export_step.command  # nosec B101
    assert "jsonl" in export_step.command  # nosec B101
    assert "--since" in export_step.command  # nosec B101
    assert "7d" in export_step.command  # nosec B101
    assert "--output" in export_step.command  # nosec B101
    assert cleanup_step.command is not None  # nosec B101
    assert cleanup_step.command[:3] == ["mcp-unified-gateway", "tool-events", "cleanup"]  # nosec B101
    assert "--max-age-days" in cleanup_step.command  # nosec B101
    assert "30" in cleanup_step.command  # nosec B101
    assert "--max-events" in cleanup_step.command  # nosec B101
    assert "100000" in cleanup_step.command  # nosec B101


def test_user_guide_uat_result_payload_redacts_reason(tmp_path: Path) -> None:
    """UAT result payloads should redact sensitive reason text."""

    harness = _load_user_guide_harness()
    wheel_path = tmp_path / "dist" / "mcp_unified-0.1.0-py3-none-any.whl"
    context = harness.UatRunContext(
        repo_root=Path("/repo"),
        workspace=tmp_path,
        bootstrap_python="python",
        gateway_url="https://example.invalid",
        admin_key="secret-admin-key",
        timeout_seconds=1.0,
        package_install_args=[str(wheel_path)],
        secrets=["secret-admin-key"],
    )
    result = harness.UatStepResult(
        step_id="install_package_boundary",
        description="Install package.",
        status="failed",
        required=True,
        duration_ms=1.0,
        reason=f"TimeoutExpired: {wheel_path} Bearer secret-admin-key",
    )

    payload = harness._step_result_payload(result, context)

    assert str(wheel_path) not in payload["reason"]  # nosec B101
    assert "secret-admin-key" not in payload["reason"]  # nosec B101
    assert "<redacted-path>" in payload["reason"]  # nosec B101
    assert "<redacted-secret>" in payload["reason"]  # nosec B101


def test_user_guide_uat_result_payload_redacts_absolute_paths(tmp_path: Path) -> None:
    """UAT result payloads should redact local absolute paths."""

    harness = _load_user_guide_harness()
    context = harness.UatRunContext(
        repo_root=Path("/repo"),
        workspace=tmp_path,
        bootstrap_python="python",
        gateway_url=None,
        admin_key=None,
        timeout_seconds=1.0,
        package_install_args=["/repo/apps/mcp-unified[gateway]"],
        secrets=[],
    )
    result = harness.UatStepResult(
        step_id="install_package_boundary",
        description="Install package.",
        status="failed",
        required=True,
        duration_ms=1.0,
        command=[
            "/Users/example/.pyenv/versions/3.12/bin/python",
            "-m",
            "pip",
            "install",
            str(tmp_path / "mcp_unified-0.1.0-py3-none-any.whl"),
        ],
        stderr=(
            "WARNING: The directory '/Users/example/Library/Caches/pip' is not writable. "
            f"Temporary wheel path: {tmp_path / 'mcp.whl'}"
        ),
    )

    payload = harness._step_result_payload(result, context)
    rendered = json.dumps(payload)

    assert "/Users/" not in rendered  # nosec B101
    assert str(tmp_path) not in rendered  # nosec B101
    assert "<redacted-path>" in rendered  # nosec B101


def test_user_guide_uat_redaction_preserves_url_paths(tmp_path: Path) -> None:
    """UAT redaction should preserve URL paths while hiding local paths."""

    harness = _load_user_guide_harness()

    redacted = harness.redact_text(
        (
            "download https://files.pythonhosted.org/packages/tmp/pkg.whl "
            "and https://github.com/actions/runner/releases while "
            "reading /Users/example/Library/Caches/pip"
        ),
        secrets=[],
        sensitive_paths=[tmp_path],
    )

    assert "https://files.pythonhosted.org/packages/tmp/pkg.whl" in redacted  # nosec B101
    assert "https://github.com/actions/runner/releases" in redacted  # nosec B101
    assert "/Users/" not in redacted  # nosec B101


def test_rc_evidence_redaction_preserves_url_paths(tmp_path: Path) -> None:
    """RC evidence redaction should preserve URL paths while hiding local paths."""

    recorder = mcp_unified_rc.RcEvidenceRecorder(
        evidence_dir=tmp_path / "evidence",
        package_name="mcp-unified",
        package_version="0.1.0",
        package_status="internal-experimental",
        publishing_status="published",
        commit="abc1234",
        source_path="apps/mcp-unified",
        layout="src",
        repo_root=tmp_path / "repo",
    )

    redacted = mcp_unified_rc._redact_evidence_text(
        (
            "download https://files.pythonhosted.org/packages/home/pkg.whl "
            "and https://github.com/actions/runner/releases while "
            "reading /Users/example/Library/Caches/pip"
        ),
        recorder,
    )

    assert "https://files.pythonhosted.org/packages/home/pkg.whl" in redacted  # nosec B101
    assert "https://github.com/actions/runner/releases" in redacted  # nosec B101
    assert "/Users/" not in redacted  # nosec B101


def test_redact_text_removes_secret_like_values() -> None:
    """Secret-like token values should be removed from harness output."""

    raw = 'token=abc123\nAPI_KEY=secret-value\n{"secret": "json-value"}\nnormal=value'

    redacted = mcp_unified_rc.redact_text(raw)

    assert "secret-value" not in redacted  # nosec B101
    assert "abc123" not in redacted  # nosec B101
    assert "json-value" not in redacted  # nosec B101
    assert "normal=value" in redacted  # nosec B101


def test_run_command_clears_inherited_pythonpath(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """run_command should avoid inheriting checkout PYTHONPATH by default."""

    captured_env: dict[str, str] = {}

    def fake_run(*args: Any, **kwargs: Any) -> Any:
        captured_env.update(kwargs["env"])
        return mcp_unified_rc.subprocess.CompletedProcess(
            args=args[0],
            returncode=0,
            stdout="ok",
            stderr="",
        )

    monkeypatch.setenv("PYTHONPATH", "/repo/apps/mcp-unified/src")
    monkeypatch.setattr(mcp_unified_rc.subprocess, "run", fake_run)

    result = mcp_unified_rc.run_command(["python", "-c", "print('ok')"], cwd=tmp_path)

    assert result.returncode == 0  # nosec B101
    assert "PYTHONPATH" not in captured_env  # nosec B101


def test_run_command_reports_command_start_failure(tmp_path: Path) -> None:
    """run_command should return evidence instead of raising on OSError."""

    result = mcp_unified_rc.run_command(
        ["definitely-not-a-real-mcp-rc-command"],
        cwd=tmp_path,
    )

    assert result.returncode == 127  # nosec B101
    assert result.stdout == ""  # nosec B101
    assert "Command failed to start:" in result.stderr  # nosec B101


def test_rc_create_venv_uses_symlinks_on_posix(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """RC venv creation should use symlinks on POSIX hosts."""

    captured: dict[str, Any] = {}

    class FakeEnvBuilder:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

        def create(self, venv_dir: Path) -> None:
            captured["venv_dir"] = venv_dir

    monkeypatch.setattr(mcp_unified_rc.venv, "EnvBuilder", FakeEnvBuilder)
    recorder = mcp_unified_rc.RcEvidenceRecorder(
        evidence_dir=tmp_path,
        package_name="mcp-unified",
        package_version="0.1.0",
        package_status="internal-experimental",
        publishing_status="published",
        commit="abc1234",
        source_path="apps/mcp-unified",
        layout="src",
    )

    created = mcp_unified_rc._create_venv(
        tmp_path / ".venv",
        recorder,
        phase="install_smoke",
        name="normal_venv",
    )

    assert created is True  # nosec B101
    assert captured["with_pip"] is True  # nosec B101
    assert captured["clear"] is True  # nosec B101
    assert captured["symlinks"] is (os.name != "nt")  # nosec B101
    assert captured["venv_dir"] == tmp_path / ".venv"  # nosec B101


def test_rc_pip_dependency_outage_records_optional_skip(tmp_path: Path) -> None:
    """Local dependency-index outages should be optional skips outside CI."""

    recorder = mcp_unified_rc.RcEvidenceRecorder(
        evidence_dir=tmp_path,
        package_name="mcp-unified",
        package_version="0.1.0",
        package_status="internal-experimental",
        publishing_status="published",
        commit="abc1234",
        source_path="apps/mcp-unified",
        layout="src",
    )
    result = mcp_unified_rc.RcCommandResult(
        command=["python", "-m", "pip", "install", "mcp_unified.whl"],
        cwd=str(tmp_path),
        returncode=1,
        stdout="Processing mcp_unified.whl",
        stderr=(
            "Failed to establish a new connection: [Errno 8] nodename nor "
            "servname provided, or not known\n"
            "ERROR: Could not find a version that satisfies the requirement "
            "pydantic>=2.0.0"
        ),
        duration_ms=10,
    )

    mcp_unified_rc._record_pip_install_result(
        recorder,
        phase="install_smoke",
        name="normal_install",
        result=result,
    )

    assert recorder.results == [  # nosec B101
        {
            "phase": "install_smoke",
            "name": "normal_install",
            "status": "skipped",
            "duration_ms": 10,
            "required": False,
            "reason": "dependency resolution unavailable in this environment",
            "details": {"command": mcp_unified_rc._command_result_as_evidence(result, recorder)},
        }
    ]


def test_rc_pip_dependency_outage_is_required_failure_in_ci(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """CI dependency-index outages should remain required failures."""

    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    recorder = mcp_unified_rc.RcEvidenceRecorder(
        evidence_dir=tmp_path,
        package_name="mcp-unified",
        package_version="0.1.0",
        package_status="internal-experimental",
        publishing_status="published",
        commit="abc1234",
        source_path="apps/mcp-unified",
        layout="src",
    )
    result = mcp_unified_rc.RcCommandResult(
        command=["python", "-m", "pip", "install", "mcp_unified.whl"],
        cwd=str(tmp_path),
        returncode=1,
        stdout="Processing mcp_unified.whl",
        stderr=(
            "Failed to establish a new connection: [Errno 8] nodename nor "
            "servname provided, or not known\n"
            "ERROR: Could not find a version that satisfies the requirement "
            "pydantic>=2.0.0"
        ),
        duration_ms=10,
    )

    mcp_unified_rc._record_pip_install_result(
        recorder,
        phase="install_smoke",
        name="normal_install",
        result=result,
    )

    assert recorder.results[0]["status"] == "failed"  # nosec B101
    assert recorder.results[0]["required"] is True  # nosec B101
    assert recorder.has_required_failures() is True  # nosec B101


def test_rc_user_guide_uat_dependency_outage_records_optional_skip(tmp_path: Path) -> None:
    """User-guide UAT dependency outages should be optional skips locally."""

    report_path = tmp_path / "user-guide-uat.json"
    report_path.write_text(
        json.dumps(
            {
                "ok": False,
                "summary": {"passed": 1, "failed": 1, "skipped": 0},
                "steps": [
                    {
                        "id": "install_package_boundary",
                        "status": "failed",
                        "stderr": (
                            "Failed to establish a new connection: [Errno 8] "
                            "nodename nor servname provided, or not known\n"
                            "ERROR: Could not find a version that satisfies the "
                            "requirement pydantic>=2.0.0"
                        ),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    recorder = mcp_unified_rc.RcEvidenceRecorder(
        evidence_dir=tmp_path,
        package_name="mcp-unified",
        package_version="0.1.0",
        package_status="internal-experimental",
        publishing_status="published",
        commit="abc1234",
        source_path="apps/mcp-unified",
        layout="src",
    )
    result = mcp_unified_rc.RcCommandResult(
        command=["python", "Helper_Scripts/Testing-related/mcp_standalone_user_guide_uat.py"],
        cwd=str(tmp_path),
        returncode=1,
        stdout="",
        stderr="",
        duration_ms=10,
    )

    mcp_unified_rc._record_user_guide_uat_result(
        recorder,
        phase="cli_uat",
        name="user_guide_wheel_mode",
        result=result,
        report_path=report_path,
    )

    assert recorder.results[0]["status"] == "skipped"  # nosec B101
    assert recorder.results[0]["required"] is False  # nosec B101
    assert recorder.results[0]["reason"] == "dependency resolution unavailable in this environment"  # nosec B101
    assert recorder.has_required_failures() is False  # nosec B101


def test_rc_cli_uat_runs_user_guide_wheel_mode(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """CLI UAT should delegate to the user-guide wheel-mode harness."""

    paths = mcp_unified_rc.RcPaths.from_repo_root(tmp_path)
    paths.dist_dir.mkdir(parents=True)
    wheel = paths.dist_dir / "mcp_unified-0.1.0-py3-none-any.whl"
    wheel.write_text("placeholder", encoding="utf-8")
    commands: list[list[str]] = []
    recorder = mcp_unified_rc.RcEvidenceRecorder(
        evidence_dir=paths.evidence_dir,
        package_name="mcp-unified",
        package_version="0.1.0",
        package_status="internal-experimental",
        publishing_status="published",
        commit="abc1234",
        source_path="apps/mcp-unified",
        layout="src",
    )

    def fail_normal_install_checks(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("cli UAT must use the user-guide wheel-mode harness")

    def fake_run_command(
        command: list[str],
        *,
        cwd: Path,
        timeout: int = 180,
        env: dict[str, str] | None = None,
    ) -> mcp_unified_rc.RcCommandResult:
        commands.append(command)
        report_path = Path(command[command.index("--json-report") + 1])
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps({"ok": True, "summary": {"passed": 30, "failed": 0, "skipped": 1}, "steps": []}),
            encoding="utf-8",
        )
        return mcp_unified_rc.RcCommandResult(
            command=command,
            cwd=str(cwd),
            returncode=0,
            stdout="",
            stderr="",
            duration_ms=10,
        )

    monkeypatch.setattr(mcp_unified_rc, "_run_normal_install_checks", fail_normal_install_checks)
    monkeypatch.setattr(mcp_unified_rc, "run_command", fake_run_command)

    mcp_unified_rc._run_cli_uat(paths, recorder)

    assert len(commands) == 1  # nosec B101
    command = commands[0]
    assert "Helper_Scripts/Testing-related/mcp_standalone_user_guide_uat.py" in command  # nosec B101
    assert "--wheel" in command  # nosec B101
    assert str(wheel) in command  # nosec B101
    assert "--json-report" in command  # nosec B101
    assert recorder.results[0]["phase"] == "cli_uat"  # nosec B101
    assert recorder.results[0]["name"] == "user_guide_wheel_mode"  # nosec B101
    assert recorder.results[0]["status"] == "passed"  # nosec B101
    assert "--mode" in command  # nosec B101
    assert command[command.index("--mode") + 1] == "cli"  # nosec B101


def test_rc_smoke_uat_runs_user_guide_transport_checks(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """Smoke UAT should delegate to user-guide transport checks."""

    paths = mcp_unified_rc.RcPaths.from_repo_root(tmp_path)
    paths.dist_dir.mkdir(parents=True)
    wheel = paths.dist_dir / "mcp_unified-0.1.0-py3-none-any.whl"
    wheel.write_text("placeholder", encoding="utf-8")
    commands: list[list[str]] = []
    recorder = mcp_unified_rc.RcEvidenceRecorder(
        evidence_dir=paths.evidence_dir,
        package_name="mcp-unified",
        package_version="0.1.0",
        package_status="internal-experimental",
        publishing_status="published",
        commit="abc1234",
        source_path="apps/mcp-unified",
        layout="src",
    )

    def fake_run_command(
        command: list[str],
        *,
        cwd: Path,
        timeout: int = 180,
        env: dict[str, str] | None = None,
    ) -> mcp_unified_rc.RcCommandResult:
        commands.append(command)
        report_path = Path(command[command.index("--json-report") + 1])
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(
                {
                    "ok": True,
                    "summary": {"passed": 30, "failed": 0, "skipped": 1},
                    "steps": [
                        {"id": "smoke_stdio_subprocess", "status": "passed"},
                        {"id": "smoke_http", "status": "passed"},
                        {"id": "smoke_websocket", "status": "passed"},
                    ],
                }
            ),
            encoding="utf-8",
        )
        return mcp_unified_rc.RcCommandResult(
            command=command,
            cwd=str(cwd),
            returncode=0,
            stdout="",
            stderr="",
            duration_ms=10,
        )

    monkeypatch.setattr(mcp_unified_rc, "run_command", fake_run_command)

    mcp_unified_rc._run_smoke_uat(paths, recorder)

    assert len(commands) == 1  # nosec B101
    command = commands[0]
    assert "Helper_Scripts/Testing-related/mcp_standalone_user_guide_uat.py" in command  # nosec B101
    assert "--wheel" in command  # nosec B101
    assert str(wheel) in command  # nosec B101
    assert recorder.results[0]["phase"] == "smoke_uat"  # nosec B101
    assert recorder.results[0]["name"] == "user_guide_smoke_transports"  # nosec B101
    assert recorder.results[0]["status"] == "passed"  # nosec B101
    assert "--mode" in command  # nosec B101
    assert command[command.index("--mode") + 1] == "smoke"  # nosec B101


def test_rc_evidence_redacts_local_absolute_paths(tmp_path: Path) -> None:
    """Recorded RC evidence should not leak local absolute paths."""

    repo_root = tmp_path / "repo"
    evidence_dir = repo_root / ".artifacts" / "mcp-unified-rc"
    local_repo_path = repo_root / "apps" / "mcp-unified" / "pyproject.toml"
    local_temp_path = "/private/var/folders/example/mcp/.venv/bin/python"
    recorder = mcp_unified_rc.RcEvidenceRecorder(
        evidence_dir=evidence_dir,
        package_name="mcp-unified",
        package_version="0.1.0",
        package_status="internal-experimental",
        publishing_status="published",
        commit="abc1234",
        source_path="apps/mcp-unified",
        layout="src",
        repo_root=repo_root,
    )
    result = mcp_unified_rc.RcCommandResult(
        command=[str(local_repo_path), local_temp_path],
        cwd="/private/var/folders/example/mcp/work",
        returncode=1,
        stdout=f"reading {local_repo_path}",
        stderr=f"failed at {local_temp_path}",
        duration_ms=10,
    )

    mcp_unified_rc._record_command_result(
        recorder,
        phase="install_smoke",
        name="path_redaction",
        result=result,
    )
    json_path, _markdown_path = recorder.write()

    rendered = json_path.read_text(encoding="utf-8")
    assert str(repo_root) not in rendered  # nosec B101
    assert "/private/var/folders" not in rendered  # nosec B101
    assert "<repo>" in rendered  # nosec B101
    assert "<redacted-path>" in rendered  # nosec B101


def test_rc_extras_matrix_records_tier_specific_checks(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """Extras matrix should record checks specific to each optional tier."""

    paths = mcp_unified_rc.RcPaths.from_repo_root(tmp_path)
    paths.dist_dir.mkdir(parents=True)
    wheel = paths.dist_dir / "mcp_unified-0.1.0-py3-none-any.whl"
    wheel.write_text("placeholder", encoding="utf-8")
    recorder = mcp_unified_rc.RcEvidenceRecorder(
        evidence_dir=paths.evidence_dir,
        package_name="mcp-unified",
        package_version="0.1.0",
        package_status="internal-experimental",
        publishing_status="published",
        commit="abc1234",
        source_path="apps/mcp-unified",
        layout="src",
    )
    commands: list[list[str]] = []

    monkeypatch.setattr(mcp_unified_rc, "OPTIONAL_EXTRAS", ("core", "gateway", "sqlite", "dev"))
    monkeypatch.setattr(mcp_unified_rc, "_create_venv", lambda *_args, **_kwargs: True)

    def fake_run_command(
        command: list[str],
        *,
        cwd: Path,
        timeout: int = 180,
        env: dict[str, str] | None = None,
    ) -> mcp_unified_rc.RcCommandResult:
        commands.append(command)
        return mcp_unified_rc.RcCommandResult(
            command=command,
            cwd=str(cwd),
            returncode=0,
            stdout="ok",
            stderr="",
            duration_ms=10,
        )

    monkeypatch.setattr(mcp_unified_rc, "run_command", fake_run_command)

    mcp_unified_rc._run_extras_matrix(paths, recorder)

    names = {result["name"] for result in recorder.results}
    assert "core_package_info" in names  # nosec B101
    assert "gateway_config_validation" in names  # nosec B101
    assert "sqlite_storage_smoke" in names  # nosec B101
    assert "dev_artifact_gate_selection" in names  # nosec B101
    flattened_commands = [" ".join(command) for command in commands]
    assert any("mcp-unified-gateway package-info" in command for command in flattened_commands)  # nosec B101
    assert any("mcp-unified-gateway validate-config" in command for command in flattened_commands)  # nosec B101
    assert any("SQLiteMCPStore" in command for command in flattened_commands)  # nosec B101
    assert any(".github/tests/test_mcp_unified_artifact_gate.py" in command for command in flattened_commands)  # nosec B101


def test_mcp_unified_dev_extra_declares_artifact_gate_dependencies() -> None:
    """The dev extra should include dependencies needed by artifact gates."""

    pyproject = tomllib.loads((REPO_ROOT / "apps" / "mcp-unified" / "pyproject.toml").read_text())
    dev_dependencies = pyproject["project"]["optional-dependencies"]["dev"]
    build_dependencies = pyproject["build-system"]["requires"]
    dev_dependency_names = {
        dependency.split(";", 1)[0]
        .split("[", 1)[0]
        .split("<", 1)[0]
        .split(">", 1)[0]
        .split("=", 1)[0]
        .strip()
        .lower()
        .replace("_", "-")
        for dependency in dev_dependencies
    }
    build_dependency_names = {
        dependency.split("<", 1)[0]
        .split(">", 1)[0]
        .split("=", 1)[0]
        .strip()
        .lower()
        .replace("_", "-")
        for dependency in build_dependencies
    }

    assert "build" in dev_dependency_names  # nosec B101
    assert "tomli" in dev_dependency_names  # nosec B101
    assert build_dependency_names.issubset(dev_dependency_names)  # nosec B101
    assert any(  # nosec B101
        "python_version" in dependency
        for dependency in dev_dependencies
        if dependency.startswith("tomli")
    )


def test_result_recorder_writes_json_and_markdown(tmp_path: Path) -> None:
    """The evidence recorder should write JSON and Markdown summaries."""

    recorder = mcp_unified_rc.RcEvidenceRecorder(
        evidence_dir=tmp_path,
        package_name="mcp-unified",
        package_version="0.1.0",
        package_status="internal-experimental",
        publishing_status="published",
        commit="abc1234",
        source_path="apps/mcp-unified",
        layout="src",
    )
    recorder.record(
        phase="artifact_metadata",
        name="wheel_metadata",
        status="passed",
        duration_ms=12,
    )

    json_path, markdown_path = recorder.write()

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["ok"] is True  # nosec B101
    assert payload["package"]["source_path"] == "apps/mcp-unified"  # nosec B101
    assert payload["summary"] == {"passed": 1, "failed": 0, "skipped": 0}  # nosec B101
    assert "wheel_metadata" in markdown_path.read_text(encoding="utf-8")  # nosec B101


def test_result_recorder_marks_required_failure(tmp_path: Path) -> None:
    """Required failures should make the evidence payload not ok."""

    recorder = mcp_unified_rc.RcEvidenceRecorder(
        evidence_dir=tmp_path,
        package_name="mcp-unified",
        package_version="0.1.0",
        package_status="internal-experimental",
        publishing_status="published",
        commit="abc1234",
        source_path="apps/mcp-unified",
        layout="src",
    )
    recorder.record(
        phase="fresh_install",
        name="normal_install",
        status="failed",
        duration_ms=15,
        reason="pip failed",
    )

    json_path, _markdown_path = recorder.write()

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["ok"] is False  # nosec B101
    assert payload["summary"] == {"passed": 0, "failed": 1, "skipped": 0}  # nosec B101


def test_result_recorder_includes_artifact_hashes(tmp_path: Path) -> None:
    """Artifact records should include filenames and SHA256 hashes."""

    artifact = tmp_path / "mcp_unified-0.1.0-py3-none-any.whl"
    artifact.write_text("wheel-content", encoding="utf-8")
    recorder = mcp_unified_rc.RcEvidenceRecorder(
        evidence_dir=tmp_path / "evidence",
        package_name="mcp-unified",
        package_version="0.1.0",
        package_status="internal-experimental",
        publishing_status="published",
        commit="abc1234",
        source_path="apps/mcp-unified",
        layout="src",
    )

    recorder.record_artifact(kind="wheel", path=artifact)
    json_path, _markdown_path = recorder.write()

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["artifacts"] == [  # nosec B101
        {
            "kind": "wheel",
            "filename": "mcp_unified-0.1.0-py3-none-any.whl",
            "sha256": mcp_unified_rc.sha256_file(artifact),
        }
    ]
