from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from tldw_Server_API.app.core.Sandbox.operator_evidence import (
    ENV_VZ_EVIDENCE_DIR,
    _dir_fd_operations_available,
    collect_operator_evidence,
)


NOW = datetime(2026, 6, 19, 12, 0, tzinfo=timezone.utc)


def _valid_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": 1,
        "created_at": "2026-06-19T11:59:00+00:00",
        "source_bundle_path": "/tmp/source",
        "run_bundle_path": "/tmp/run",
        "image_store_root": "/tmp/store",
        "smoke_run_id": "smoke-123",
        "socket_path": "/tmp/helper.sock",
        "serial_log_dir": "/tmp/serial",
        "evidence_dir": "/tmp/evidence",
        "helper_path": "/tmp/private-helper-path-must-not-leak",
        "helper_pid_file": "/tmp/helper.pid",
        "skip_build": False,
        "skip_sign": False,
        "include_failure_drills": False,
        "final_exit_code": 0,
        "phases": {
            "build": {
                "status": "ok",
                "exit_code": 0,
                "timestamp": "2026-06-19T11:59:10Z",
            },
        },
        "cleanup": {
            "status": 0,
            "helper_pid": "123",
            "helper_running_after_cleanup": False,
            "socket_present_after_cleanup": False,
        },
        "evidence_files": {},
        "log_artifacts": [
            {"path": "/tmp/serial.log", "size_bytes": 10, "sha256": "abc"}
        ],
    }
    payload.update(overrides)
    return payload


def _write_evidence(root: Path, payload: dict[str, object]) -> None:
    root.mkdir(mode=0o700, exist_ok=True)
    (root / "host-smoke-evidence.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )
    for name in (
        "source-bundle-hashes-before.txt",
        "source-bundle-hashes-after.txt",
        "run-bundle-hashes.txt",
        "runtime-paths.txt",
        "cleanup-status.txt",
    ):
        (root / name).write_text("ok\n", encoding="utf-8")


def test_collect_operator_evidence_unconfigured() -> None:
    summary = collect_operator_evidence(environ={}, now=NOW)

    assert summary["configured"] is False
    assert summary["source"] == "host_smoke_evidence"
    assert summary["reasons"] == ["evidence_not_configured"]


def test_collect_operator_evidence_valid_bundle(tmp_path: Path) -> None:
    if not _dir_fd_operations_available():
        pytest.skip("descriptor-safe directory operations are unavailable")
    _write_evidence(tmp_path, _valid_payload())

    summary = collect_operator_evidence(
        environ={ENV_VZ_EVIDENCE_DIR: str(tmp_path)},
        now=NOW,
    )

    assert summary["configured"] is True
    assert summary["available"] is True
    assert summary["valid"] is True
    assert summary["schema_version"] == 1
    assert summary["age_seconds"] == 60
    assert summary["final_exit_code"] == 0
    assert summary["skip_flags"]["include_failure_drills"] is False
    assert "helper_path" not in summary.get("runtime_pointers", {})
    assert summary["expected_files"]["host-smoke-evidence.json"]["readable"] is True
