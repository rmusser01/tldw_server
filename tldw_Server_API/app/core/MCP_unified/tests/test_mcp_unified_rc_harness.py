from __future__ import annotations

import importlib.util
import json
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
    ) == [str(wheel.resolve())]  # nosec B101
    assert harness.package_install_spec(
        repo_root=Path("/repo"),
        wheel_path=wheel,
        editable=True,
    ) == [str(wheel.resolve())]  # nosec B101
    assert harness.package_install_spec(
        repo_root=Path("/repo"),
        wheel_path=relative_wheel,
        editable=False,
    ) == [str(relative_wheel.resolve())]  # nosec B101


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
