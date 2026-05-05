from __future__ import annotations

from datetime import datetime, timezone

from tldw_Server_API.app.api.v1.schemas.sandbox_schemas import (
    SandboxAdminRunSummary,
    SandboxRunStatus,
)
from tldw_Server_API.app.core.Sandbox.models import RunPhase, RunStatus, RuntimeType
from tldw_Server_API.app.core.Sandbox.run_status_taxonomy import (
    normalize_run_status_reason,
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


def test_public_and_admin_status_schemas_expose_reason_code() -> None:
    public_schema = SandboxRunStatus.model_json_schema()
    admin_schema = SandboxAdminRunSummary.model_json_schema()

    assert "status_reason_code" in public_schema["properties"]
    assert "status_reason_code" in admin_schema["properties"]


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
