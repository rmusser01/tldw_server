from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from tldw_Server_API.app.core.Sandbox.operator_evidence import (
    DISPLAY_MAX_CHARS,
    ENV_VZ_EVIDENCE_DIR,
    JSON_MAX_BYTES,
    MAX_PHASES,
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


@pytest.fixture
def safe_evidence_root(tmp_path: Path) -> Path:
    """Return an evidence root only when descriptor-safe reads are available."""

    probe = tmp_path / "safe-open-probe"
    _write_evidence(probe, _valid_payload())
    summary = collect_operator_evidence(
        environ={ENV_VZ_EVIDENCE_DIR: str(probe)},
        now=NOW,
    )
    if "evidence_safe_open_unavailable" in summary.get("reasons", []):
        pytest.skip("descriptor-safe directory operations are unavailable")
    assert summary["valid"] is True
    return tmp_path / "evidence"


def test_collect_operator_evidence_unconfigured() -> None:
    summary = collect_operator_evidence(environ={}, now=NOW)

    assert summary["configured"] is False
    assert summary["source"] == "host_smoke_evidence"
    assert summary["reasons"] == ["evidence_not_configured"]


def test_collect_operator_evidence_valid_bundle(safe_evidence_root: Path) -> None:
    _write_evidence(safe_evidence_root, _valid_payload())

    summary = collect_operator_evidence(
        environ={ENV_VZ_EVIDENCE_DIR: str(safe_evidence_root)},
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


def test_collect_operator_evidence_rejects_nul_path() -> None:
    summary = collect_operator_evidence(
        environ={ENV_VZ_EVIDENCE_DIR: "/tmp/bad\0path"},
        now=NOW,
    )

    assert summary["configured"] is True
    assert summary["available"] is False
    assert "evidence_path_contains_nul" in summary["reasons"]


def test_collect_operator_evidence_reports_missing_directory(
    safe_evidence_root: Path,
) -> None:
    missing = safe_evidence_root
    summary = collect_operator_evidence(
        environ={ENV_VZ_EVIDENCE_DIR: str(missing)},
        now=NOW,
    )

    assert summary["available"] is False
    assert "evidence_directory_missing" in summary["reasons"]


def test_collect_operator_evidence_rejects_directory_symlink(
    safe_evidence_root: Path,
) -> None:
    target = safe_evidence_root.parent / "target"
    target.mkdir()
    link = safe_evidence_root.parent / "link"
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation is unavailable on this platform")

    summary = collect_operator_evidence(
        environ={ENV_VZ_EVIDENCE_DIR: str(link)},
        now=NOW,
    )

    assert summary["available"] is False
    assert "evidence_directory_symlink" in summary["reasons"]


def test_collect_operator_evidence_rejects_json_symlink(
    safe_evidence_root: Path,
) -> None:
    _write_evidence(safe_evidence_root, _valid_payload())
    (safe_evidence_root / "target.json").write_text("{}", encoding="utf-8")
    (safe_evidence_root / "host-smoke-evidence.json").unlink()
    try:
        (safe_evidence_root / "host-smoke-evidence.json").symlink_to(
            safe_evidence_root / "target.json"
        )
    except OSError:
        pytest.skip("symlink creation is unavailable on this platform")

    summary = collect_operator_evidence(
        environ={ENV_VZ_EVIDENCE_DIR: str(safe_evidence_root)},
        now=NOW,
    )

    assert summary["valid"] is False
    assert "evidence_json_symlink" in summary["reasons"]


def test_collect_operator_evidence_fails_closed_without_safe_open(
    tmp_path: Path,
) -> None:
    _write_evidence(tmp_path, _valid_payload())

    summary = collect_operator_evidence(
        environ={ENV_VZ_EVIDENCE_DIR: str(tmp_path)},
        now=NOW,
        safe_open_available=lambda: False,
    )

    assert summary["available"] is False
    assert "evidence_safe_open_unavailable" in summary["reasons"]


def test_collect_operator_evidence_rejects_oversized_json(
    safe_evidence_root: Path,
) -> None:
    _write_evidence(safe_evidence_root, _valid_payload())
    (safe_evidence_root / "host-smoke-evidence.json").write_text(
        " " * (JSON_MAX_BYTES + 1),
        encoding="utf-8",
    )

    summary = collect_operator_evidence(
        environ={ENV_VZ_EVIDENCE_DIR: str(safe_evidence_root)},
        now=NOW,
    )

    assert summary["valid"] is False
    assert "evidence_json_oversized" in summary["reasons"]


def test_collect_operator_evidence_rejects_malformed_utf8(
    safe_evidence_root: Path,
) -> None:
    _write_evidence(safe_evidence_root, _valid_payload())
    (safe_evidence_root / "host-smoke-evidence.json").write_bytes(b"\xff")

    summary = collect_operator_evidence(
        environ={ENV_VZ_EVIDENCE_DIR: str(safe_evidence_root)},
        now=NOW,
    )

    assert summary["valid"] is False
    assert "evidence_json_malformed_utf8" in summary["reasons"]
    assert summary["expected_files"]["host-smoke-evidence.json"]["readable"] is True


def test_collect_operator_evidence_rejects_malformed_json_without_leaking_raw(
    safe_evidence_root: Path,
) -> None:
    _write_evidence(safe_evidence_root, _valid_payload())
    raw_json = '{"schema_version": 1, "secret_raw": "must-not-leak"'
    (safe_evidence_root / "host-smoke-evidence.json").write_text(
        raw_json,
        encoding="utf-8",
    )

    summary = collect_operator_evidence(
        environ={ENV_VZ_EVIDENCE_DIR: str(safe_evidence_root)},
        now=NOW,
    )

    assert summary["valid"] is False
    assert "evidence_json_malformed" in summary["reasons"]
    assert summary["expected_files"]["host-smoke-evidence.json"]["readable"] is True
    assert "must-not-leak" not in str(summary)


def test_collect_operator_evidence_rejects_top_level_non_object(
    safe_evidence_root: Path,
) -> None:
    _write_evidence(safe_evidence_root, _valid_payload())
    (safe_evidence_root / "host-smoke-evidence.json").write_text(
        "[]",
        encoding="utf-8",
    )

    summary = collect_operator_evidence(
        environ={ENV_VZ_EVIDENCE_DIR: str(safe_evidence_root)},
        now=NOW,
    )

    assert summary["valid"] is False
    assert "evidence_json_top_level_not_object" in summary["reasons"]
    assert summary["expected_files"]["host-smoke-evidence.json"]["readable"] is True


def test_collect_operator_evidence_requires_supported_schema(
    safe_evidence_root: Path,
) -> None:
    payload = _valid_payload()
    payload.pop("schema_version")
    _write_evidence(safe_evidence_root, payload)

    missing = collect_operator_evidence(
        environ={ENV_VZ_EVIDENCE_DIR: str(safe_evidence_root)},
        now=NOW,
    )
    assert missing["valid"] is False
    assert "evidence_schema_version_missing" in missing["reasons"]

    _write_evidence(safe_evidence_root, _valid_payload(schema_version=2))
    unsupported = collect_operator_evidence(
        environ={ENV_VZ_EVIDENCE_DIR: str(safe_evidence_root)},
        now=NOW,
    )
    assert unsupported["valid"] is False
    assert "evidence_schema_version_unsupported" in unsupported["reasons"]


def test_collect_operator_evidence_rejects_invalid_exit_code_and_skip_flags(
    safe_evidence_root: Path,
) -> None:
    _write_evidence(
        safe_evidence_root,
        _valid_payload(final_exit_code=True, skip_build="false"),
    )

    summary = collect_operator_evidence(
        environ={ENV_VZ_EVIDENCE_DIR: str(safe_evidence_root)},
        now=NOW,
    )

    assert summary["valid"] is False
    assert summary["final_exit_code"] is None
    assert summary["skip_flags"]["skip_build"] is None
    assert "evidence_final_exit_code_invalid" in summary["reasons"]
    assert "evidence_skip_flag_invalid" in summary["reasons"]


def test_collect_operator_evidence_bounds_and_allowlists_metadata(
    safe_evidence_root: Path,
) -> None:
    phases = {
        f"phase-{index}": {
            "status": "ok",
            "exit_code": 0,
            "timestamp": "2026-06-19T11:59:10Z",
            "raw_output": "must-not-leak",
        }
        for index in range(MAX_PHASES + 9)
    }
    _write_evidence(
        safe_evidence_root,
        _valid_payload(
            helper_path="/secret/helper",
            unexpected_path="/secret/other",
            smoke_run_id="x" * (DISPLAY_MAX_CHARS + 50),
            phases=phases,
            nested={"raw": "must-not-leak"},
        ),
    )

    summary = collect_operator_evidence(
        environ={ENV_VZ_EVIDENCE_DIR: str(safe_evidence_root)},
        now=NOW,
    )

    assert len(summary["smoke_run_id"]) <= DISPLAY_MAX_CHARS
    assert summary["smoke_run_id"].endswith("...")
    assert "helper_path" not in summary["runtime_pointers"]
    assert "unexpected_path" not in summary["runtime_pointers"]
    assert len(summary["phases"]) == MAX_PHASES
    assert "raw_output" not in next(iter(summary["phases"].values()))
    assert "nested" not in summary
    assert "must-not-leak" not in str(summary)


def test_collect_operator_evidence_classifies_stale_evidence(
    safe_evidence_root: Path,
) -> None:
    _write_evidence(
        safe_evidence_root,
        _valid_payload(created_at="2026-06-01T00:00:00+00:00"),
    )

    summary = collect_operator_evidence(
        environ={ENV_VZ_EVIDENCE_DIR: str(safe_evidence_root)},
        now=NOW,
    )

    assert summary["valid"] is True
    assert summary["stale"] is True
    assert summary["age_seconds"] > 7 * 24 * 60 * 60


@pytest.mark.parametrize(
    ("created_at", "reason"),
    [
        ("2026-06-19T11:59:00", "evidence_created_at_malformed"),
        ("not-a-date", "evidence_created_at_malformed"),
        ("2026-06-20T00:00:00+00:00", "evidence_created_at_in_future"),
    ],
)
def test_collect_operator_evidence_rejects_invalid_timestamps(
    safe_evidence_root: Path,
    created_at: str,
    reason: str,
) -> None:
    _write_evidence(safe_evidence_root, _valid_payload(created_at=created_at))

    summary = collect_operator_evidence(
        environ={ENV_VZ_EVIDENCE_DIR: str(safe_evidence_root)},
        now=NOW,
    )

    assert summary["valid"] is False
    assert reason in summary["reasons"]
    assert summary.get("age_seconds") is None or summary["age_seconds"] >= 0
