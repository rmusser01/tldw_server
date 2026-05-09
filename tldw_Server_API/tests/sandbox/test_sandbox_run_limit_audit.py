from __future__ import annotations

from datetime import datetime

import tldw_Server_API.app.core.Sandbox.service as service_module
from tldw_Server_API.app.core.Audit.unified_audit_service import (
    AuditEventCategory,
    AuditEventType,
    AuditSeverity,
)
from tldw_Server_API.app.core.Sandbox.audit_metadata import (
    build_run_completion_audit_metadata,
)
from tldw_Server_API.app.core.Sandbox.limits import (
    build_limit_audit_metadata,
    limit_event_actions,
)
from tldw_Server_API.app.core.Sandbox.models import (
    RunPhase,
    RunSpec,
    RunStatus,
    RuntimeType,
    TrustLevel,
)
from tldw_Server_API.app.core.Sandbox.service import SandboxService


def _completed_status(
    *,
    base_image: str = "vz_linux:test",
    resource_usage: dict[str, object] | None = None,
) -> RunStatus:
    started = datetime.utcnow()
    return RunStatus(
        id="run-audit-contract",
        phase=RunPhase.completed,
        runtime=RuntimeType.vz_linux,
        base_image=base_image,
        policy_hash="policy-hash",
        exit_code=0,
        started_at=started,
        finished_at=started,
        resource_usage=resource_usage or {},
    )


def test_build_limit_audit_metadata_is_aggregate_and_path_minimized() -> None:
    metadata = build_limit_audit_metadata(
        {
            "output_limit_bytes": 5,
            "stdout_truncated": 1,
            "stderr_truncated": 0,
            "artifact_files_skipped": 2,
            "artifact_skip_file_limit": 1,
            "artifact_skip_total_limit": 1,
            "log_limit_bytes": 5,
            "log_truncated": 1,
            "artifact_paths": ["secret.txt"],
        }
    )

    assert metadata["output_truncated"] is True
    assert metadata["output_limit_bytes"] == 5
    assert metadata["artifact_files_skipped"] == 2
    assert metadata["artifact_skip_reasons"] == ["file_limit", "total_limit"]
    assert metadata["log_limit_bytes"] == 5
    assert metadata["log_truncated"] == 1
    assert "artifact_paths" not in metadata


def test_limit_event_actions_reports_only_affected_limits() -> None:
    assert limit_event_actions({"stdout_truncated": 1}) == ["output_truncated"]
    assert limit_event_actions({"log_truncated": 1}) == ["log_truncated"]
    assert limit_event_actions({"artifact_files_skipped": 1}) == ["artifacts_limited"]
    assert limit_event_actions({"stdout_truncated": 0, "artifact_files_skipped": 0}) == []


def test_build_run_completion_audit_metadata_includes_runtime_policy_context() -> None:
    status = _completed_status(
        resource_usage={
            "output_limit_bytes": 5,
            "stdout_truncated": 1,
            "artifact_files_skipped": 1,
            "artifact_skip_file_limit": 1,
        },
    )

    metadata = build_run_completion_audit_metadata(
        status=status,
        spec_version="1.1",
        requested_runtime=RuntimeType.vz_linux,
        trust_level=TrustLevel.untrusted,
        network_policy="deny_all",
        capture_patterns=["reports/*.json", "logs/*.txt"],
    )

    assert metadata["runtime"] == "vz_linux"
    assert metadata["effective_runtime"] == "vz_linux"
    assert metadata["requested_runtime"] == "vz_linux"
    assert metadata["trust_level"] == "untrusted"
    assert metadata["network_policy"] == "deny_all"
    assert metadata["policy_hash"] == "policy-hash"
    assert metadata["spec_version"] == "1.1"
    assert metadata["exit_code"] == 0
    assert metadata["outcome"] == "success"
    assert metadata["status_reason_code"] == "limits_applied"
    assert metadata["capture_pattern_count"] == 2
    assert "capture_patterns" not in metadata


def test_build_run_completion_audit_metadata_includes_bounded_limit_metadata() -> None:
    status = _completed_status(
        resource_usage={
            "output_limit_bytes": 5,
            "stdout_truncated": 1,
            "artifact_files_skipped": 1,
            "artifact_skip_file_limit": 1,
            "artifact_paths": ["/Users/operator/private-output.txt"],
        },
    )

    metadata = build_run_completion_audit_metadata(
        status=status,
        spec_version="1.1",
        requested_runtime=RuntimeType.vz_linux,
        trust_level=TrustLevel.untrusted,
        network_policy="deny_all",
        capture_patterns=["reports/*.json", "logs/*.txt"],
    )

    assert metadata["output_truncated"] is True
    assert metadata["artifact_skip_reasons"] == ["file_limit"]
    assert "artifact_paths" not in metadata
    assert "/Users/operator" not in str(metadata)


def test_build_run_completion_audit_metadata_minimizes_posix_host_paths() -> None:
    status = _completed_status(base_image="/Users/operator/private-rootfs.img")

    metadata = build_run_completion_audit_metadata(
        status=status,
        spec_version="1.1",
        requested_runtime=RuntimeType.vz_linux,
    )

    assert metadata["base_image_kind"] == "host_path"
    assert metadata["base_image"] is None
    assert "/Users/operator" not in str(metadata)


def test_build_run_completion_audit_metadata_minimizes_windows_drive_relative_paths() -> None:
    status = _completed_status(base_image=r"C:Users\operator\private-rootfs.img")

    metadata = build_run_completion_audit_metadata(
        status=status,
        spec_version="1.1",
        requested_runtime=RuntimeType.vz_linux,
    )

    assert metadata["base_image_kind"] == "host_path"
    assert metadata["base_image"] is None
    assert "operator" not in str(metadata)


def test_build_run_completion_audit_metadata_preserves_omitted_requested_runtime() -> None:
    status = _completed_status()

    metadata = build_run_completion_audit_metadata(
        status=status,
        spec_version="1.1",
        requested_runtime=None,
    )

    assert metadata["effective_runtime"] == "vz_linux"
    assert metadata["requested_runtime"] is None


def test_sandbox_service_audits_aggregate_limit_events(monkeypatch) -> None:
    events: list[dict[str, object]] = []

    class _FakeAuditService:
        def __init__(self, db_path: str | None = None) -> None:
            self.db_path = db_path

        async def initialize(self, *, start_background_tasks: bool = False) -> None:
            assert start_background_tasks is False

        async def log_event(self, **kwargs) -> None:
            events.append(dict(kwargs))

        async def stop(self) -> None:
            return None

    monkeypatch.setattr(service_module, "UnifiedAuditService", _FakeAuditService)

    started = datetime.utcnow()
    status = RunStatus(
        id="run-limit-audit",
        phase=RunPhase.completed,
        runtime=RuntimeType.vz_linux,
        base_image="vz_linux:test",
        policy_hash="policy-hash",
        exit_code=0,
        started_at=started,
        finished_at=started,
        resource_usage={
            "output_limit_bytes": 5,
            "stdout_truncated": 1,
            "stderr_truncated": 0,
            "artifact_files_skipped": 2,
            "artifact_skip_file_limit": 1,
            "artifact_skip_total_limit": 1,
        },
    )
    spec = RunSpec(
        session_id="session-1",
        runtime=RuntimeType.vz_linux,
        base_image="vz_linux:test",
        command=["echo", "ok"],
        trust_level=TrustLevel.standard,
        network_policy="deny_all",
        capture_patterns=["reports/*.json"],
    )

    SandboxService()._audit_run_completion(
        user_id=None,
        run_id=status.id,
        status=status,
        spec_version="1.0",
        session_id="session-1",
        spec=spec,
    )

    assert [event["action"] for event in events] == ["run", "output_truncated", "artifacts_limited"]
    completion_metadata = events[0]["metadata"]
    assert completion_metadata["effective_runtime"] == "vz_linux"
    assert completion_metadata["requested_runtime"] == "vz_linux"
    assert completion_metadata["trust_level"] == "standard"
    assert completion_metadata["network_policy"] == "deny_all"
    assert completion_metadata["status_reason_code"] == "limits_applied"
    assert completion_metadata["outcome"] == "success"
    assert completion_metadata["capture_pattern_count"] == 1
    assert completion_metadata["output_truncated"] is True
    assert completion_metadata["artifact_skip_reasons"] == ["file_limit", "total_limit"]
    assert "artifact_paths" not in completion_metadata

    for event in events[1:]:
        assert event["event_type"] == AuditEventType.API_RESPONSE
        assert event["category"] == AuditEventCategory.API_CALL
        assert event["severity"] == AuditSeverity.WARNING
        assert event["resource_type"] == "sandbox.run"
        assert event["resource_id"] == "run-limit-audit"
        assert event["result"] == "limited"
        assert event["metadata"]["artifact_skip_reasons"] == ["file_limit", "total_limit"]
        assert "artifact_paths" not in event["metadata"]
