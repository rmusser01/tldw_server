from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


pytestmark = pytest.mark.unit

IMAGE_DIR = Path(__file__).resolve().parents[1]
SUMMARY_SCRIPT = IMAGE_DIR / "scripts" / "summarize-host-e2e-evidence.py"
EXPECTED_EVIDENCE_FILES = {
    "host-smoke-evidence.json",
    "source-bundle-hashes-before.txt",
    "source-bundle-hashes-after.txt",
    "run-bundle-hashes.txt",
    "runtime-paths.txt",
    "cleanup-status.txt",
}


def _run_summary(
    evidence_dir: Path,
    *,
    summary_path: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    if summary_path is not None:
        env["GITHUB_STEP_SUMMARY"] = str(summary_path)
    else:
        env.pop("GITHUB_STEP_SUMMARY", None)
    return subprocess.run(
        [sys.executable, str(SUMMARY_SCRIPT), "--evidence-dir", str(evidence_dir)],
        cwd=IMAGE_DIR,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )


def _write_complete_evidence(evidence_dir: Path) -> None:
    evidence_dir.mkdir()
    for evidence_file in EXPECTED_EVIDENCE_FILES - {"host-smoke-evidence.json"}:
        (evidence_dir / evidence_file).write_text(f"{evidence_file}\n", encoding="utf-8")
    (evidence_dir / "host-smoke-evidence.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "created_at": "2026-06-17T00:00:00Z",
                "source_bundle_path": "/private/source/bundle",
                "run_bundle_path": "/private/run/bundle",
                "image_store_root": "/private/image-store",
                "smoke_run_id": "ci-123",
                "socket_path": "/private/runtime/helper.sock",
                "serial_log_dir": "/private/runtime/serial",
                "evidence_dir": str(evidence_dir),
                "helper_path": "/private/helper",
                "helper_pid_file": "/private/helper.pid",
                "skip_build": False,
                "skip_sign": False,
                "include_failure_drills": False,
                "final_exit_code": 7,
                "phases": {
                    "real_host_smoke": {
                        "status": "failed",
                        "exit_code": 7,
                        "timestamp": "2026-06-17T00:00:01Z",
                    },
                    "cleanup": {
                        "status": "ok",
                        "exit_code": 0,
                        "timestamp": "2026-06-17T00:00:02Z",
                    },
                },
                "cleanup": {
                    "status": 0,
                    "helper_pid": "123",
                    "helper_running_after_cleanup": False,
                    "socket_present_after_cleanup": False,
                },
                "evidence_files": {
                    evidence_file: str(evidence_dir / evidence_file)
                    for evidence_file in EXPECTED_EVIDENCE_FILES
                },
                "log_artifacts": [
                    {
                        "path": "/private/runtime/serial/vm.log",
                        "size_bytes": 128,
                        "sha256": "a" * 64,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def _assert_advisory_success(result: subprocess.CompletedProcess[str]) -> None:
    assert result.returncode == 0, result.stderr
    assert "Traceback" not in result.stdout
    assert "Traceback" not in result.stderr


def test_summary_reports_complete_evidence(tmp_path: Path) -> None:
    evidence_dir = tmp_path / "evidence"
    _write_complete_evidence(evidence_dir)

    result = _run_summary(evidence_dir)

    _assert_advisory_success(result)
    assert "VZ Linux Host Smoke Evidence Summary" in result.stdout
    assert "Advisory only" in result.stdout
    assert "final_exit_code" in result.stdout
    assert "7" in result.stdout
    assert "smoke_run_id" in result.stdout
    assert "ci-123" in result.stdout
    assert "real_host_smoke" in result.stdout
    assert "cleanup" in result.stdout
    assert "status" in result.stdout
    assert "helper_running_after_cleanup" in result.stdout
    assert "/private/run/bundle" in result.stdout
    assert "/private/runtime/serial/vm.log" in result.stdout
    assert "128" in result.stdout
    assert "vz-linux-host-gated-evidence" in result.stdout
    assert "vz-linux-host-gated-helper-logs" in result.stdout
    for evidence_file in EXPECTED_EVIDENCE_FILES:
        assert evidence_file in result.stdout


def test_missing_evidence_directory_warns_and_lists_expected_files(tmp_path: Path) -> None:
    result = _run_summary(tmp_path / "missing-evidence")

    _assert_advisory_success(result)
    assert "warning" in result.stdout.lower()
    assert "missing" in result.stdout.lower()
    for evidence_file in EXPECTED_EVIDENCE_FILES:
        assert evidence_file in result.stdout


def test_evidence_path_that_is_regular_file_warns_without_traceback(tmp_path: Path) -> None:
    evidence_path = tmp_path / "evidence"
    evidence_path.write_text("not a directory\n", encoding="utf-8")

    result = _run_summary(evidence_path)

    _assert_advisory_success(result)
    assert "warning" in result.stdout.lower()
    assert "not a directory" in result.stdout
    for evidence_file in EXPECTED_EVIDENCE_FILES:
        assert evidence_file in result.stdout


def test_partial_evidence_without_json_warns_and_checks_files(tmp_path: Path) -> None:
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir()
    (evidence_dir / "cleanup-status.txt").write_text("cleanup_status=0\n", encoding="utf-8")

    result = _run_summary(evidence_dir)

    _assert_advisory_success(result)
    assert "warning" in result.stdout.lower()
    assert "structured metadata" in result.stdout
    assert "cleanup-status.txt" in result.stdout
    assert "present" in result.stdout
    assert "host-smoke-evidence.json" in result.stdout
    assert "missing" in result.stdout


def test_malformed_json_warns_without_raw_json_dump(tmp_path: Path) -> None:
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir()
    raw_json = '{"secret_raw_json": "do not echo",'
    (evidence_dir / "host-smoke-evidence.json").write_text(raw_json, encoding="utf-8")

    result = _run_summary(evidence_dir)

    _assert_advisory_success(result)
    assert "warning" in result.stdout.lower()
    assert "JSONDecodeError" in result.stdout
    assert "host-smoke-evidence.json" in result.stdout
    assert "secret_raw_json" not in result.stdout
    assert raw_json not in result.stdout


def test_oversized_json_warns_and_is_not_parsed(tmp_path: Path) -> None:
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir()
    oversized_marker = "oversized-marker-should-not-appear"
    (evidence_dir / "host-smoke-evidence.json").write_text(
        json.dumps({"smoke_run_id": oversized_marker}) + (" " * (1024 * 1024)),
        encoding="utf-8",
    )

    result = _run_summary(evidence_dir)

    _assert_advisory_success(result)
    assert "warning" in result.stdout.lower()
    assert "exceeds" in result.stdout
    assert oversized_marker not in result.stdout


def test_json_symlink_is_skipped_without_reading_target(tmp_path: Path) -> None:
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir()
    target = tmp_path / "target.json"
    target_secret = "symlink-target-content-should-not-appear"
    target.write_text(json.dumps({"smoke_run_id": target_secret}), encoding="utf-8")
    (evidence_dir / "host-smoke-evidence.json").symlink_to(target)

    result = _run_summary(evidence_dir)

    _assert_advisory_success(result)
    assert "warning" in result.stdout.lower()
    assert "symlink" in result.stdout
    assert "host-smoke-evidence.json" in result.stdout
    assert target_secret not in result.stdout


def test_expected_evidence_file_that_is_directory_is_warned_and_skipped(tmp_path: Path) -> None:
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir()
    (evidence_dir / "cleanup-status.txt").mkdir()

    result = _run_summary(evidence_dir)

    _assert_advisory_success(result)
    assert "warning" in result.stdout.lower()
    assert "cleanup-status.txt" in result.stdout
    assert "non-regular file skipped" in result.stdout


def test_valid_github_step_summary_path_appends_markdown(tmp_path: Path) -> None:
    evidence_dir = tmp_path / "evidence"
    _write_complete_evidence(evidence_dir)
    summary_path = tmp_path / "step-summary.md"
    summary_path.write_text("existing summary\n", encoding="utf-8")

    result = _run_summary(evidence_dir, summary_path=summary_path)

    _assert_advisory_success(result)
    assert result.stdout == ""
    assert result.stderr == ""
    summary = summary_path.read_text(encoding="utf-8")
    assert summary.startswith("existing summary\n")
    assert "VZ Linux Host Smoke Evidence Summary" in summary
    assert "final_exit_code" in summary


def test_invalid_github_step_summary_path_falls_back_to_stdout_and_stderr(tmp_path: Path) -> None:
    evidence_dir = tmp_path / "evidence"
    _write_complete_evidence(evidence_dir)
    summary_path = tmp_path / "summary-directory"
    summary_path.mkdir()

    result = _run_summary(evidence_dir, summary_path=summary_path)

    _assert_advisory_success(result)
    assert "warning: unable to append to GITHUB_STEP_SUMMARY" in result.stderr
    assert "VZ Linux Host Smoke Evidence Summary" in result.stdout
    assert "final_exit_code" in result.stdout
