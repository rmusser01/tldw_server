from __future__ import annotations

import pytest

from tldw_Server_API.app.core.Sandbox.macos_virtualization.helper_client import (
    MacOSVirtualizationHelperProtocolError,
    MacOSVirtualizationHelperUnavailable,
)
from tldw_Server_API.app.core.Sandbox.macos_virtualization.models import (
    HelperVMListReply,
    HelperVMMetadata,
    HelperVMStatusReply,
)


pytestmark = pytest.mark.unit


class _FakeOrchestrator:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self._rows = rows
        self.list_calls = 0
        self.mutating_calls: list[str] = []

    def list_vz_session_controls(self) -> list[dict[str, object]]:
        self.list_calls += 1
        return list(self._rows)

    def delete_vz_session_control(self, session_id: str) -> None:
        self.mutating_calls.append(session_id)


class _FakeHelper:
    def __init__(self, vms: list[HelperVMStatusReply]) -> None:
        self._vms = vms
        self.terminated_vm_ids: list[str] = []
        self.deleted_vm_ids: list[str] = []

    def list_vms(self) -> HelperVMListReply:
        return HelperVMListReply(
            protocol_version="1",
            helper_version="0.1.0",
            vms=list(self._vms),
        )

    def terminate_vm(self, vm_id: str) -> bool:
        self.terminated_vm_ids.append(vm_id)
        return True

    def delete_vm(self, vm_id: str) -> bool:
        self.deleted_vm_ids.append(vm_id)
        return True


class _UnavailableHelper:
    def list_vms(self) -> HelperVMListReply:
        raise MacOSVirtualizationHelperUnavailable("helper socket missing")


class _ProtocolMismatchHelper:
    def list_vms(self) -> HelperVMListReply:
        raise MacOSVirtualizationHelperProtocolError("helper protocol mismatch")


class _FailureHelper:
    def list_vms(self) -> HelperVMListReply:
        from tldw_Server_API.app.core.Sandbox.macos_virtualization.helper_client import (
            MacOSVirtualizationHelperFailure,
        )

        raise MacOSVirtualizationHelperFailure("helper_internal_error", "list failed")


class _RaisingOrchestrator:
    def list_vz_session_controls(self) -> list[dict[str, object]]:
        raise RuntimeError("store unavailable")


def _metadata(
    *,
    owner: str = "tldw",
    runtime: str = "vz_linux",
    run_id: str = "run-owned",
    session_id: str = "",
    session_mode: bool = False,
    created_at: str = "2026-04-30T18:00:00Z",
) -> HelperVMMetadata:
    return HelperVMMetadata(
        owner=owner,
        runtime=runtime,
        run_id=run_id,
        session_id=session_id,
        session_mode=session_mode,
        template_path="/tmp/template",
        workspace_path="/tmp/workspace",
        created_at=created_at,
    )


def _vm(
    vm_id: str,
    *,
    state: str = "running",
    healthy: bool = True,
    metadata: HelperVMMetadata | None = None,
) -> HelperVMStatusReply:
    return HelperVMStatusReply(
        protocol_version="1",
        helper_version="0.1.0",
        vm_id=vm_id,
        state=state,
        healthy=healthy,
        metadata=metadata or HelperVMMetadata(),
    )


def test_sandbox_startup_producer_emits_warning_for_stale_and_orphaned_state() -> None:
    from tldw_Server_API.app.services.startup_warning_registry import (
        StartupWarningRegistry,
    )
    from tldw_Server_API.app.services.startup_warning_sandbox import (
        produce_sandbox_startup_warnings,
    )

    registry = StartupWarningRegistry(startup_id="boot-1")
    helper = _FakeHelper(
        [
            _vm("vm-live", healthy=True),
            _vm("vm-unhealthy", healthy=False),
            _vm("vm-orphan", metadata=_metadata(run_id="run-orphan")),
        ]
    )

    produce_sandbox_startup_warnings(
        orchestrator=_FakeOrchestrator(
            [
                {"id": "sess-live", "vm_id": "vm-live"},
                {"id": "sess-stale", "vm_id": "vm-missing"},
                {"id": "sess-unhealthy", "vm_id": "vm-unhealthy"},
                {"id": "sess-active", "vm_id": "vm-active-missing"},
            ]
        ),
        helper_client=helper,
        active_session_checker=lambda session_id: session_id == "sess-active",
        registry=registry,
    )

    assert [item.code for item in registry.list_warnings()] == [
        "vz_orphaned_vms_detected",
        "vz_skipped_active_reconciliation_items_detected",
        "vz_stale_session_controls_detected",
        "vz_unhealthy_session_controls_detected",
    ]
    assert registry.summary()["total"] == 4
    assert registry.should_block_startup() is False


def test_sandbox_startup_producer_emits_blocker_for_protocol_mismatch() -> None:
    from tldw_Server_API.app.services.startup_warning_registry import (
        StartupWarningRegistry,
    )
    from tldw_Server_API.app.services.startup_warning_sandbox import (
        produce_sandbox_startup_warnings,
    )

    registry = StartupWarningRegistry(startup_id="boot-1")

    produce_sandbox_startup_warnings(
        orchestrator=_FakeOrchestrator([{"id": "sess-live", "vm_id": "vm-live"}]),
        helper_client=_ProtocolMismatchHelper(),
        registry=registry,
    )

    warnings = registry.list_warnings()
    assert [item.code for item in warnings] == ["vz_helper_protocol_mismatch"]
    assert warnings[0].startup_action == "block_startup"
    assert warnings[0].severity == "error"
    assert registry.should_block_startup() is True


def test_sandbox_startup_producer_helper_unavailable_is_warning_only() -> None:
    from tldw_Server_API.app.services.startup_warning_registry import (
        StartupWarningRegistry,
    )
    from tldw_Server_API.app.services.startup_warning_sandbox import (
        produce_sandbox_startup_warnings,
    )

    registry = StartupWarningRegistry(startup_id="boot-1")

    produce_sandbox_startup_warnings(
        orchestrator=_FakeOrchestrator([{"id": "sess-live", "vm_id": "vm-live"}]),
        helper_client=_UnavailableHelper(),
        registry=registry,
    )

    warnings = registry.list_warnings()
    assert [item.code for item in warnings] == ["vz_helper_unavailable_at_startup"]
    assert warnings[0].startup_action == "warn"
    assert registry.should_block_startup() is False


def test_sandbox_startup_producer_does_not_mutate_runtime_state() -> None:
    from tldw_Server_API.app.services.startup_warning_registry import (
        StartupWarningRegistry,
    )
    from tldw_Server_API.app.services.startup_warning_sandbox import (
        produce_sandbox_startup_warnings,
    )

    orchestrator = _FakeOrchestrator([{"id": "sess-stale", "vm_id": "vm-missing"}])
    helper = _FakeHelper([_vm("vm-orphan", metadata=_metadata(run_id="run-orphan"))])
    registry = StartupWarningRegistry(startup_id="boot-1")

    produce_sandbox_startup_warnings(
        orchestrator=orchestrator,
        helper_client=helper,
        registry=registry,
    )

    assert orchestrator.list_calls == 1
    assert orchestrator.mutating_calls == []
    assert helper.terminated_vm_ids == []
    assert helper.deleted_vm_ids == []


def test_sandbox_startup_producer_reports_helper_failure() -> None:
    from tldw_Server_API.app.services.startup_warning_registry import (
        StartupWarningRegistry,
    )
    from tldw_Server_API.app.services.startup_warning_sandbox import (
        produce_sandbox_startup_warnings,
    )

    registry = StartupWarningRegistry(startup_id="boot-1")

    produce_sandbox_startup_warnings(
        orchestrator=_FakeOrchestrator([{"id": "sess-live", "vm_id": "vm-live"}]),
        helper_client=_FailureHelper(),
        registry=registry,
    )

    warnings = registry.list_warnings()
    assert [item.code for item in warnings] == ["vz_helper_failure_at_startup"]
    assert warnings[0].startup_action == "warn"


def test_sandbox_startup_producer_reports_reconciliation_unavailable() -> None:
    from tldw_Server_API.app.services.startup_warning_registry import (
        StartupWarningRegistry,
    )
    from tldw_Server_API.app.services.startup_warning_sandbox import (
        produce_sandbox_startup_warnings,
    )

    registry = StartupWarningRegistry(startup_id="boot-1")

    produce_sandbox_startup_warnings(
        orchestrator=_RaisingOrchestrator(),
        helper_client=_FakeHelper([]),
        registry=registry,
    )

    warnings = registry.list_warnings()
    assert [item.code for item in warnings] == [
        "vz_reconciliation_unavailable_at_startup"
    ]
    assert warnings[0].startup_action == "warn"
