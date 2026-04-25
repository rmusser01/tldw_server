from __future__ import annotations

import asyncio

import pytest

from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
from tldw_Server_API.app.services import workflows_db_maintenance as maintenance_mod


@pytest.mark.asyncio
async def test_postgres_maintenance_uses_backend_vacuum_without_transaction(monkeypatch):
    calls = {"vacuum": 0, "transaction": 0}

    class _StubBackend:
        def vacuum(self, connection=None):  # noqa: ANN001
            assert connection is None
            calls["vacuum"] += 1

        def transaction(self):  # noqa: ANN201
            calls["transaction"] += 1
            raise AssertionError("maintenance should not open a transaction for VACUUM")

    class _StubDB:
        backend = _StubBackend()
        backend_type = BackendType.POSTGRESQL

    monkeypatch.setenv("WORKFLOWS_DB_MAINTENANCE_INTERVAL_SEC", "1")
    monkeypatch.setenv("WORKFLOWS_POSTGRES_VACUUM", "true")
    monkeypatch.setattr(maintenance_mod, "get_content_backend_instance", lambda: object())
    monkeypatch.setattr(maintenance_mod, "create_workflows_database", lambda backend=None: _StubDB())

    stop_event = asyncio.Event()
    task = asyncio.create_task(maintenance_mod.run_workflows_db_maintenance(stop_event))
    await asyncio.sleep(0.1)
    stop_event.set()
    await asyncio.wait_for(task, timeout=2)

    assert calls["vacuum"] >= 1
    assert calls["transaction"] == 0
