from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from typing import get_args

import pytest

from tldw_Server_API.app.api.v1.schemas.sandbox_schemas import (
    SandboxAdminRunSummary,
    SandboxRunStatus,
)
from tldw_Server_API.app.core.Sandbox import run_status_taxonomy
from tldw_Server_API.app.core.Sandbox.models import RunPhase, RunStatus, RuntimeType
from tldw_Server_API.app.core.Sandbox.run_status_taxonomy import (
    RUN_STATUS_REASON_METADATA,
    RunStatusReasonCode,
    _validate_run_status_reason_metadata,
    normalize_run_status_reason,
    run_status_reason_details,
)
from tldw_Server_API.app.core.Sandbox.store import SQLiteStore


def test_run_status_reason_taxonomy_maps_common_status_shapes() -> None:
    assert normalize_run_status_reason(
        phase=RunPhase.queued,
        message=None,
        exit_code=None,
        resource_usage=None,
    ) == "queued"
    assert normalize_run_status_reason(
        phase=RunPhase.completed,
        message="Docker execution finished",
        exit_code=0,
        resource_usage=None,
    ) == "completed"
    assert normalize_run_status_reason(
        phase=RunPhase.failed,
        message="Docker execution failed (exit=2)",
        exit_code=2,
        resource_usage=None,
    ) == "nonzero_exit"
    assert normalize_run_status_reason(
        phase=RunPhase.failed,
        message="lima_policy_failed",
        exit_code=None,
        resource_usage=None,
    ) == "policy_failed"
    assert normalize_run_status_reason(
        phase=RunPhase.timed_out,
        message="startup_timeout",
        exit_code=None,
        resource_usage=None,
    ) == "startup_timeout"
    assert normalize_run_status_reason(
        phase=RunPhase.timed_out,
        message="execution_timeout",
        exit_code=None,
        resource_usage=None,
    ) == "execution_timeout"
    assert normalize_run_status_reason(
        phase=RunPhase.killed,
        message="canceled_by_user",
        exit_code=None,
        resource_usage=None,
    ) == "canceled_by_user"
    assert normalize_run_status_reason(
        phase=RunPhase.failed,
        message="queue_ttl_expired",
        exit_code=None,
        resource_usage=None,
    ) == "queue_ttl_expired"


def test_run_status_reason_taxonomy_reports_limit_signals() -> None:
    assert normalize_run_status_reason(
        phase=RunPhase.completed,
        message="Docker execution finished",
        exit_code=0,
        resource_usage={"stdout_truncated": 1, "artifact_files_skipped": 0},
    ) == "limits_applied"
    assert normalize_run_status_reason(
        phase=RunPhase.completed,
        message="Docker execution finished",
        exit_code=0,
        resource_usage={"artifact_files_skipped": 2},
    ) == "limits_applied"
    assert normalize_run_status_reason(
        phase=RunPhase.failed,
        message="Docker execution failed (exit=2)",
        exit_code=2,
        resource_usage={"stdout_truncated": 1},
    ) == "limits_applied"
    assert normalize_run_status_reason(
        phase=RunPhase.timed_out,
        message="execution_timeout",
        exit_code=None,
        resource_usage={"artifact_skip_total_limit": 1},
    ) == "limits_applied"
    assert normalize_run_status_reason(
        phase=RunPhase.completed,
        message="worktree execution finished",
        exit_code=0,
        resource_usage={"log_truncated": 1, "log_limit_bytes": 5},
    ) == "limits_applied"


def test_run_status_reason_taxonomy_maps_known_policy_failures() -> None:
    for message in (
        "lima_policy_failed",
        "vz_linux_policy_failed",
        "vz_macos_policy_failed",
        "seatbelt_policy_failed",
        "worktree_policy_failed",
    ):
        assert normalize_run_status_reason(
            phase=RunPhase.failed,
            message=message,
            exit_code=None,
            resource_usage=None,
        ) == "policy_failed"


def test_run_status_reason_taxonomy_prefers_policy_failed_over_runtime_unavailable() -> None:
    for message in (
        "lima_policy_failed runtime_unavailable",
        "Runtime unavailable after VZ_MACOS_POLICY_FAILED",
    ):
        assert normalize_run_status_reason(
            phase=RunPhase.failed,
            message=message,
            exit_code=None,
            resource_usage=None,
        ) == "policy_failed"


def test_run_status_reason_taxonomy_distinguishes_runtime_unavailable() -> None:
    for message in (
        "runtime_unavailable",
        "runtime unavailable",
        "docker_unavailable",
        "firecracker_unavailable",
        "vz_linux_unavailable",
        "vz_macos_unavailable",
        "seatbelt_unavailable",
        "worktree_unavailable",
    ):
        assert normalize_run_status_reason(
            phase=RunPhase.failed,
            message=message,
            exit_code=None,
            resource_usage=None,
        ) == "runtime_unavailable"
    assert normalize_run_status_reason(
        phase=RunPhase.failed,
        message="runtime provisioning template missing",
        exit_code=None,
        resource_usage=None,
    ) == "runtime_unavailable"
    assert normalize_run_status_reason(
        phase=RunPhase.failed,
        message="command not found in PATH: python",
        exit_code=None,
        resource_usage=None,
    ) == "runtime_error"
    assert normalize_run_status_reason(
        phase=RunPhase.failed,
        message="artifact manifest missing",
        exit_code=None,
        resource_usage=None,
    ) == "runtime_error"


def test_run_status_reason_taxonomy_accepts_string_phase_values() -> None:
    assert normalize_run_status_reason(
        phase="running",
        message=None,
        exit_code=None,
        resource_usage=None,
    ) == "running"
    assert normalize_run_status_reason(
        phase="failed",
        message=f"runtime error at {datetime.now(timezone.utc).isoformat()}",
        exit_code=None,
        resource_usage=None,
    ) == "runtime_error"
    assert normalize_run_status_reason(
        phase="failed",
        message="bad exit code row",
        exit_code="not-an-int",
        resource_usage=None,
    ) == "runtime_error"


def test_run_status_reason_metadata_covers_every_reason_code() -> None:
    """Ensure every public reason code has structured metadata."""

    assert set(RUN_STATUS_REASON_METADATA) == set(get_args(RunStatusReasonCode))
    for key, metadata in RUN_STATUS_REASON_METADATA.items():
        assert metadata.code == key


def test_run_status_reason_metadata_validation_rejects_code_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject metadata entries whose internal code does not match their key."""

    mismatched_metadata = dict(RUN_STATUS_REASON_METADATA)
    mismatched_metadata["queued"] = replace(
        RUN_STATUS_REASON_METADATA["queued"],
        code="running",
    )
    monkeypatch.setattr(
        run_status_taxonomy,
        "RUN_STATUS_REASON_METADATA",
        mismatched_metadata,
    )

    with pytest.raises(RuntimeError, match="code mismatch"):
        _validate_run_status_reason_metadata()


def test_run_status_reason_details_exposes_stable_metadata() -> None:
    """Verify representative status reason details are stable and complete."""

    runtime_unavailable = run_status_reason_details("runtime_unavailable")
    assert runtime_unavailable.category == "runtime"
    assert runtime_unavailable.severity == "error"
    assert runtime_unavailable.retryable is True
    assert runtime_unavailable.operator_action == "check_runtime_readiness"
    assert runtime_unavailable.terminal is True
    assert runtime_unavailable.user_message_key == "sandbox.status.runtime_unavailable"

    assert run_status_reason_details("policy_failed").operator_action == "review_policy"
    assert run_status_reason_details("limits_applied").severity == "warning"
    assert run_status_reason_details("unknown").category == "unknown"
    assert run_status_reason_details("not-a-real-code").code == "unknown"


def test_public_and_admin_status_schemas_expose_reason_code() -> None:
    public_schema = SandboxRunStatus.model_json_schema()
    admin_schema = SandboxAdminRunSummary.model_json_schema()

    assert "status_reason_code" in public_schema["properties"]
    assert "status_reason_code" in admin_schema["properties"]
    assert "status_reason_details" in public_schema["properties"]
    assert "status_reason_details" in admin_schema["properties"]


def test_public_and_admin_status_models_preserve_reason_details() -> None:
    """Ensure public and admin status models serialize structured details."""

    details = run_status_reason_details("limits_applied")
    public_status = SandboxRunStatus(
        id="run-1",
        runtime="docker",
        phase="completed",
        status_reason_code=details.code,
        status_reason_details=details,
    )
    admin_status = SandboxAdminRunSummary(
        id="run-1",
        runtime="docker",
        phase="completed",
        status_reason_code=details.code,
        status_reason_details=details,
    )

    assert public_status.model_dump()["status_reason_details"] == {
        "code": "limits_applied",
        "category": "limits",
        "severity": "warning",
        "terminal": True,
        "retryable": False,
        "operator_action": "review_limits",
        "user_message_key": "sandbox.status.limits_applied",
    }
    assert admin_status.model_dump()["status_reason_details"]["code"] == "limits_applied"


def test_sqlite_run_listing_preserves_resource_usage_for_admin_reason_codes(tmp_path) -> None:
    store = SQLiteStore(str(tmp_path / "sandbox.db"))
    status = RunStatus(
        id="limited-run",
        phase=RunPhase.completed,
        runtime=RuntimeType.docker,
        exit_code=0,
        message="Docker execution finished",
        resource_usage={"stdout_truncated": 1, "artifact_files_skipped": 0},
    )

    store.put_run("user-1", status)
    rows = store.list_runs(limit=10, offset=0)

    assert rows[0]["resource_usage"] == {"stdout_truncated": 1, "artifact_files_skipped": 0}
    assert normalize_run_status_reason(
        phase=rows[0]["phase"],
        message=rows[0]["message"],
        exit_code=rows[0]["exit_code"],
        resource_usage=rows[0]["resource_usage"],
    ) == "limits_applied"
