from __future__ import annotations

import importlib
import sys

import pytest


pytestmark = pytest.mark.unit


def _import_shutdown_unified_audit_services():
    sys.modules.pop("tldw_Server_API.app.services.shutdown_unified_audit_services", None)
    return importlib.import_module("tldw_Server_API.app.services.shutdown_unified_audit_services")


@pytest.mark.asyncio
async def test_shutdown_unified_audit_services_runs_steps_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shutdown_audit = _import_shutdown_unified_audit_services()
    calls: list[str] = []

    async def _record_cached_audit_services():
        calls.append("cached-audit")

    async def _record_sharing_audit_service(*, guard_exceptions):
        assert guard_exceptions == (RuntimeError, ImportError, ModuleNotFoundError)
        calls.append("sharing")

    async def _record_embeddings_audit_adapter(*, guard_exceptions):
        assert guard_exceptions == (RuntimeError, LookupError)
        calls.append("embeddings")

    async def _record_evaluations_audit_adapter(*, guard_exceptions):
        assert guard_exceptions == (RuntimeError, LookupError)
        calls.append("evaluations")

    monkeypatch.setattr(shutdown_audit, "_shutdown_cached_audit_services", _record_cached_audit_services)
    monkeypatch.setattr(shutdown_audit, "_shutdown_sharing_audit_service", _record_sharing_audit_service)
    monkeypatch.setattr(shutdown_audit, "_shutdown_embeddings_audit_adapter", _record_embeddings_audit_adapter)
    monkeypatch.setattr(shutdown_audit, "_shutdown_evaluations_audit_adapter", _record_evaluations_audit_adapter)

    await shutdown_audit.shutdown_unified_audit_services(
        startup_guard_exceptions=(RuntimeError,),
        import_exceptions=(LookupError,),
    )

    assert calls == ["cached-audit", "sharing", "embeddings", "evaluations"]


@pytest.mark.asyncio
async def test_shutdown_unified_audit_services_handles_import_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shutdown_audit = _import_shutdown_unified_audit_services()

    async def _failing_cached_audit_services():
        raise LookupError("boom")

    monkeypatch.setattr(
        shutdown_audit,
        "_shutdown_cached_audit_services",
        _failing_cached_audit_services,
    )

    await shutdown_audit.shutdown_unified_audit_services(
        startup_guard_exceptions=(RuntimeError,),
        import_exceptions=(LookupError,),
    )


@pytest.mark.asyncio
async def test_shutdown_sharing_audit_service_handles_guard_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shutdown_audit = _import_shutdown_unified_audit_services()

    async def _failing_shutdown_sharing_audit_service():
        raise RuntimeError("boom")

    monkeypatch.setattr(
        shutdown_audit,
        "_shutdown_sharing_audit_service_service",
        _failing_shutdown_sharing_audit_service,
    )

    await shutdown_audit._shutdown_sharing_audit_service(
        guard_exceptions=(RuntimeError,),
    )


@pytest.mark.asyncio
async def test_shutdown_embeddings_audit_adapter_handles_guard_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shutdown_audit = _import_shutdown_unified_audit_services()

    def _failing_shutdown_embeddings_audit_adapter():
        raise RuntimeError("boom")

    monkeypatch.setattr(
        shutdown_audit,
        "_shutdown_embeddings_audit_adapter_service",
        _failing_shutdown_embeddings_audit_adapter,
    )

    await shutdown_audit._shutdown_embeddings_audit_adapter(
        guard_exceptions=(RuntimeError,),
    )


@pytest.mark.asyncio
async def test_shutdown_evaluations_audit_adapter_handles_guard_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shutdown_audit = _import_shutdown_unified_audit_services()

    def _failing_shutdown_evaluations_audit_adapter():
        raise RuntimeError("boom")

    monkeypatch.setattr(
        shutdown_audit,
        "_shutdown_evaluations_audit_adapter_service",
        _failing_shutdown_evaluations_audit_adapter,
    )

    await shutdown_audit._shutdown_evaluations_audit_adapter(
        guard_exceptions=(RuntimeError,),
    )
