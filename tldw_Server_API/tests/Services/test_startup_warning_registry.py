from __future__ import annotations

from datetime import UTC, datetime

import pytest


pytestmark = pytest.mark.unit


def _warning(*, code: str, startup_action: str = "warn"):
    from tldw_Server_API.app.services.startup_warning_models import StartupWarningRecord

    return StartupWarningRecord(
        component="sandbox.vz_linux",
        severity="error" if startup_action == "block_startup" else "warning",
        startup_action=startup_action,
        code=code,
        summary=f"summary for {code}",
        remediation="follow the operator notes",
        details={"count": 1},
        detected_at=datetime(2026, 4, 30, 12, 0, tzinfo=UTC),
    )


def test_startup_warning_registry_starts_empty() -> None:
    from tldw_Server_API.app.services.startup_warning_registry import (
        StartupWarningRegistry,
    )

    registry = StartupWarningRegistry(startup_id="boot-1")

    assert registry.list_warnings() == []
    assert registry.summary() == {
        "startup_id": "boot-1",
        "total": 0,
        "blocking_total": 0,
        "has_blocking": False,
        "by_component": {},
        "by_severity": {},
        "by_action": {},
    }
    assert registry.should_block_startup() is False


def test_startup_warning_registry_groups_and_detects_blockers() -> None:
    from tldw_Server_API.app.services.startup_warning_registry import (
        StartupWarningRegistry,
    )

    registry = StartupWarningRegistry(startup_id="boot-1")

    registry.add_warning(_warning(code="vz_helper_unavailable"))
    registry.add_warning(
        _warning(
            code="vz_helper_protocol_mismatch",
            startup_action="block_startup",
        )
    )

    assert [item.code for item in registry.list_warnings()] == [
        "vz_helper_protocol_mismatch",
        "vz_helper_unavailable",
    ]
    assert registry.summary() == {
        "startup_id": "boot-1",
        "total": 2,
        "blocking_total": 1,
        "has_blocking": True,
        "by_component": {"sandbox.vz_linux": 2},
        "by_severity": {"error": 1, "warning": 1},
        "by_action": {"block_startup": 1, "warn": 1},
    }
    assert registry.should_block_startup() is True


def test_startup_warning_registry_clear_resets_state() -> None:
    from tldw_Server_API.app.services.startup_warning_registry import (
        StartupWarningRegistry,
    )

    registry = StartupWarningRegistry(startup_id="boot-1")
    registry.add_warning(_warning(code="vz_orphaned_vms_detected"))

    registry.clear()

    assert registry.list_warnings() == []
    assert registry.summary()["total"] == 0
    assert registry.summary()["has_blocking"] is False
    assert registry.should_block_startup() is False


def test_startup_warning_registry_filters_by_component_prefix() -> None:
    from tldw_Server_API.app.services.startup_warning_registry import (
        StartupWarningRegistry,
    )
    from tldw_Server_API.app.services.startup_warning_models import (
        StartupWarningRecord,
    )

    registry = StartupWarningRegistry(startup_id="boot-1")
    registry.add_warning(_warning(code="vz_orphaned_vms_detected"))
    registry.add_warning(
        StartupWarningRecord(
            component="jobs.integrity",
            severity="warning",
            startup_action="warn",
            code="jobs_integrity_drift_detected",
            summary="jobs drift",
            remediation="inspect jobs",
            details={},
            detected_at=datetime(2026, 4, 30, 12, 0, tzinfo=UTC),
        )
    )

    sandbox_records = registry.list_warnings(component_prefix="sandbox.")

    assert [record.code for record in sandbox_records] == [
        "vz_orphaned_vms_detected"
    ]
    assert registry.summary(component_prefix="sandbox.") == {
        "startup_id": "boot-1",
        "total": 1,
        "blocking_total": 0,
        "has_blocking": False,
        "by_component": {"sandbox.vz_linux": 1},
        "by_severity": {"warning": 1},
        "by_action": {"warn": 1},
    }
