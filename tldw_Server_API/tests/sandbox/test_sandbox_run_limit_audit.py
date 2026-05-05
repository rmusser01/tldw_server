from __future__ import annotations

from datetime import datetime

import tldw_Server_API.app.core.Sandbox.service as service_module
from tldw_Server_API.app.core.Audit.unified_audit_service import (
    AuditEventCategory,
    AuditEventType,
    AuditSeverity,
)
from tldw_Server_API.app.core.Sandbox.limits import (
    build_limit_audit_metadata,
    limit_event_actions,
)
from tldw_Server_API.app.core.Sandbox.models import RunPhase, RunStatus, RuntimeType
from tldw_Server_API.app.core.Sandbox.service import SandboxService


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

    SandboxService()._audit_run_completion(
        user_id=None,
        run_id=status.id,
        status=status,
        spec_version="1.0",
        session_id="session-1",
    )

    assert [event["action"] for event in events] == ["run", "output_truncated", "artifacts_limited"]
    completion_metadata = events[0]["metadata"]
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
