from __future__ import annotations

import json
from pathlib import Path

from Helper_Scripts import mcp_unified_rc


def test_default_paths_point_to_apps_package() -> None:
    paths = mcp_unified_rc.RcPaths.from_repo_root(Path("/repo"))

    assert paths.package_project == Path("/repo/apps/mcp-unified")  # nosec B101
    assert paths.package_src == Path("/repo/apps/mcp-unified/src/mcp_unified")  # nosec B101
    assert paths.evidence_dir == Path("/repo/.artifacts/mcp-unified-rc")  # nosec B101


def test_redact_text_removes_secret_like_values() -> None:
    raw = 'token=abc123\nAPI_KEY=secret-value\n{"secret": "json-value"}\nnormal=value'

    redacted = mcp_unified_rc.redact_text(raw)

    assert "secret-value" not in redacted  # nosec B101
    assert "abc123" not in redacted  # nosec B101
    assert "json-value" not in redacted  # nosec B101
    assert "normal=value" in redacted  # nosec B101


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
