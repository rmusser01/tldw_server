from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

from Helper_Scripts import mcp_unified_rc

REPO_ROOT = Path(__file__).resolve().parents[5]
USER_GUIDE_UAT_PATH = (
    REPO_ROOT
    / "Helper_Scripts"
    / "Testing-related"
    / "mcp_standalone_user_guide_uat.py"
)


def _load_user_guide_harness() -> ModuleType:
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
    paths = mcp_unified_rc.RcPaths.from_repo_root(Path("/repo"))

    assert paths.package_project == Path("/repo/apps/mcp-unified")  # nosec B101
    assert paths.package_src == Path("/repo/apps/mcp-unified/src/mcp_unified")  # nosec B101
    assert paths.evidence_dir == Path("/repo/.artifacts/mcp-unified-rc")  # nosec B101


def test_user_guide_uat_install_spec_uses_apps_project_by_default() -> None:
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


def test_user_guide_uat_result_payload_redacts_reason(tmp_path: Path) -> None:
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


def test_redact_text_removes_secret_like_values() -> None:
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


def test_rc_create_venv_uses_symlinks_on_posix(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
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
        publishing_status="not-published",
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
    recorder = mcp_unified_rc.RcEvidenceRecorder(
        evidence_dir=tmp_path,
        package_name="mcp-unified",
        package_version="0.1.0",
        package_status="internal-experimental",
        publishing_status="not-published",
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
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    recorder = mcp_unified_rc.RcEvidenceRecorder(
        evidence_dir=tmp_path,
        package_name="mcp-unified",
        package_version="0.1.0",
        package_status="internal-experimental",
        publishing_status="not-published",
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
        publishing_status="not-published",
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
        publishing_status="not-published",
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


def test_rc_smoke_uat_runs_user_guide_transport_checks(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
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
        publishing_status="not-published",
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


def test_rc_evidence_redacts_local_absolute_paths(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    evidence_dir = repo_root / ".artifacts" / "mcp-unified-rc"
    local_repo_path = repo_root / "apps" / "mcp-unified" / "pyproject.toml"
    local_temp_path = "/private/var/folders/example/mcp/.venv/bin/python"
    recorder = mcp_unified_rc.RcEvidenceRecorder(
        evidence_dir=evidence_dir,
        package_name="mcp-unified",
        package_version="0.1.0",
        package_status="internal-experimental",
        publishing_status="not-published",
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


def test_result_recorder_writes_json_and_markdown(tmp_path: Path) -> None:
    recorder = mcp_unified_rc.RcEvidenceRecorder(
        evidence_dir=tmp_path,
        package_name="mcp-unified",
        package_version="0.1.0",
        package_status="internal-experimental",
        publishing_status="not-published",
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
    recorder = mcp_unified_rc.RcEvidenceRecorder(
        evidence_dir=tmp_path,
        package_name="mcp-unified",
        package_version="0.1.0",
        package_status="internal-experimental",
        publishing_status="not-published",
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
    artifact = tmp_path / "mcp_unified-0.1.0-py3-none-any.whl"
    artifact.write_text("wheel-content", encoding="utf-8")
    recorder = mcp_unified_rc.RcEvidenceRecorder(
        evidence_dir=tmp_path / "evidence",
        package_name="mcp-unified",
        package_version="0.1.0",
        package_status="internal-experimental",
        publishing_status="not-published",
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
