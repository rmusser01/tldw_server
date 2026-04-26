from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

REQUIRED_ROOT_SHIM_SYMBOLS = {
    "_is_postgres_backend",
    "_get_rbac_repo",
    "_ensure_sqlite_authnz_ready_if_test_mode",
    "_emit_admin_audit_event",
    "emit_budget_audit_event",
    "_enforce_admin_user_scope",
    "_is_platform_admin",
    "_require_platform_admin",
    "_get_admin_org_ids",
    "_load_bulk_user_candidates",
}


def test_admin_root_shim_exports_contract() -> None:
    from tldw_Server_API.app.api.v1.endpoints import admin as admin_mod

    missing = sorted(symbol for symbol in REQUIRED_ROOT_SHIM_SYMBOLS if not hasattr(admin_mod, symbol))
    assert not missing, f"Missing required admin shim symbols: {missing}"

    non_callables = sorted(symbol for symbol in REQUIRED_ROOT_SHIM_SYMBOLS if not callable(getattr(admin_mod, symbol)))
    assert not non_callables, f"Admin shim symbols must be callable: {non_callables}"


@pytest.mark.asyncio
async def test_admin_root_postgres_alias_uses_pool_state(monkeypatch) -> None:
    from tldw_Server_API.app.api.v1.endpoints import admin as admin_mod

    class _Pool:
        def __init__(self, pool_obj):
            self.pool = pool_obj

    async def _fake_pg_pool():
        return _Pool(object())

    async def _fake_sqlite_pool():
        return _Pool(None)

    monkeypatch.setattr(admin_mod, "get_db_pool", _fake_pg_pool)
    assert await admin_mod._is_postgres_backend() is True

    monkeypatch.setattr(admin_mod, "get_db_pool", _fake_sqlite_pool)
    assert await admin_mod._is_postgres_backend() is False


@pytest.mark.asyncio
async def test_admin_root_postgres_alias_sanitizes_pool_fallback_log(monkeypatch) -> None:
    from tldw_Server_API.app.api.v1.endpoints import admin as admin_mod

    debug_records: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    async def _raise_pool_error():
        raise RuntimeError("pool exploded at /private/users.db")

    monkeypatch.setattr(admin_mod, "get_db_pool", _raise_pool_error)
    monkeypatch.setattr(
        admin_mod.logger,
        "debug",
        lambda message, *args, **kwargs: debug_records.append((str(message), args, kwargs)),
    )

    assert await admin_mod._is_postgres_backend() is False
    assert debug_records == [("Admin backend detection falling back to SQLite due to pool error", (), {})]


@pytest.mark.asyncio
async def test_admin_root_ensure_sqlite_ready_sanitizes_outer_fallback_log(monkeypatch) -> None:
    from tldw_Server_API.app.api.v1.endpoints import admin as admin_mod

    debug_records: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    def _raise_test_mode_error() -> bool:
        raise RuntimeError("test-mode detection exploded at /private/authnz.db")

    monkeypatch.setattr(admin_mod, "is_test_mode", _raise_test_mode_error)
    monkeypatch.setattr(
        admin_mod.logger,
        "debug",
        lambda message, *args, **kwargs: debug_records.append((str(message), args, kwargs)),
    )

    await admin_mod._ensure_sqlite_authnz_ready_if_test_mode()

    assert debug_records == [("AuthNZ test ensure skipped/failed", (), {})]


@pytest.mark.asyncio
async def test_admin_root_ensure_sqlite_ready_sanitizes_table_check_log(
    monkeypatch,
    tmp_path,
) -> None:
    from tldw_Server_API.app.api.v1.endpoints import admin as admin_mod
    from tldw_Server_API.app.core.AuthNZ import database as authnz_database
    from tldw_Server_API.app.core.AuthNZ import migrations as authnz_migrations

    debug_records: list[tuple[str, tuple[object, ...], dict[str, object]]] = []
    ensured_paths: list[object] = []

    class _Acquire:
        async def __aenter__(self):
            raise RuntimeError("sqlite table check exploded at /private/authnz.db")

        async def __aexit__(self, *_exc_info):
            return False

    class _Pool:
        pool = None
        _sqlite_fs_path = str(tmp_path / "authnz.db")

        def acquire(self):
            return _Acquire()

    async def _fake_get_db_pool():
        return _Pool()

    def _fake_ensure_authnz_tables(path):
        ensured_paths.append(path)

    monkeypatch.setattr(admin_mod, "is_test_mode", lambda: True)
    monkeypatch.setattr(authnz_database, "get_db_pool", _fake_get_db_pool)
    monkeypatch.setattr(authnz_migrations, "ensure_authnz_tables", _fake_ensure_authnz_tables)
    monkeypatch.setattr(
        admin_mod.logger,
        "debug",
        lambda message, *args, **kwargs: debug_records.append((str(message), args, kwargs)),
    )

    await admin_mod._ensure_sqlite_authnz_ready_if_test_mode()

    assert debug_records == [("AuthNZ test ensure table check failed", (), {})]
    assert [str(path) for path in ensured_paths] == [str(tmp_path / "authnz.db")]


@pytest.mark.asyncio
async def test_admin_root_ensure_sqlite_ready_sanitizes_mkdir_log(
    monkeypatch,
    tmp_path,
) -> None:
    from tldw_Server_API.app.api.v1.endpoints import admin as admin_mod
    from tldw_Server_API.app.core.AuthNZ import database as authnz_database
    from tldw_Server_API.app.core.AuthNZ import migrations as authnz_migrations

    debug_records: list[tuple[str, tuple[object, ...], dict[str, object]]] = []
    ensured_paths: list[object] = []
    file_parent = tmp_path / "not-a-dir"
    file_parent.write_text("blocks mkdir")

    class _Cursor:
        async def fetchone(self):
            return None

    class _Connection:
        async def execute(self, _query: str):
            return _Cursor()

    class _Acquire:
        async def __aenter__(self):
            return _Connection()

        async def __aexit__(self, *_exc_info):
            return False

    class _Pool:
        pool = None
        _sqlite_fs_path = str(file_parent / "authnz.db")

        def acquire(self):
            return _Acquire()

    async def _fake_get_db_pool():
        return _Pool()

    def _fake_ensure_authnz_tables(path):
        ensured_paths.append(path)

    monkeypatch.setattr(admin_mod, "is_test_mode", lambda: True)
    monkeypatch.setattr(authnz_database, "get_db_pool", _fake_get_db_pool)
    monkeypatch.setattr(authnz_migrations, "ensure_authnz_tables", _fake_ensure_authnz_tables)
    monkeypatch.setattr(
        admin_mod.logger,
        "debug",
        lambda message, *args, **kwargs: debug_records.append((str(message), args, kwargs)),
    )

    await admin_mod._ensure_sqlite_authnz_ready_if_test_mode()

    assert debug_records == [("AuthNZ test ensure mkdir failed", (), {})]
    assert [str(path) for path in ensured_paths] == [str(file_parent / "authnz.db")]


def test_admin_rbac_repo_factory_uses_root_compat_shim(monkeypatch) -> None:
    from tldw_Server_API.app.api.v1.endpoints import admin as admin_mod
    from tldw_Server_API.app.api.v1.endpoints.admin import admin_rbac

    sentinel_repo = object()
    monkeypatch.setattr(admin_mod, "_get_rbac_repo", lambda: sentinel_repo)

    assert admin_rbac._get_rbac_repo() is sentinel_repo


@pytest.mark.asyncio
async def test_admin_rbac_scope_enforcement_uses_root_compat_shim(monkeypatch) -> None:
    from tldw_Server_API.app.api.v1.endpoints import admin as admin_mod
    from tldw_Server_API.app.api.v1.endpoints.admin import admin_rbac

    calls: dict[str, object] = {}

    async def _fake_scope_check(principal, target_user_id: int, *, require_hierarchy: bool) -> None:
        calls["principal"] = principal
        calls["target_user_id"] = target_user_id
        calls["require_hierarchy"] = require_hierarchy

    monkeypatch.setattr(admin_mod, "_enforce_admin_user_scope", _fake_scope_check)

    principal = SimpleNamespace(user_id=1)
    await admin_rbac._enforce_admin_user_scope(principal, 42, require_hierarchy=True)

    assert calls == {
        "principal": principal,
        "target_user_id": 42,
        "require_hierarchy": True,
    }


@pytest.mark.asyncio
async def test_admin_kanban_fts_maintenance_closes_db(monkeypatch) -> None:
    from tldw_Server_API.app.api.v1.endpoints.admin import admin_rbac

    calls: list[object] = []

    class _FakeKanbanDB:
        def optimize_fts(self) -> None:
            calls.append("optimize")

        def close(self) -> None:
            calls.append("close")

    async def _fake_scope_check(principal, target_user_id: int, *, require_hierarchy: bool) -> None:
        calls.append(
            {
                "principal": principal,
                "target_user_id": target_user_id,
                "require_hierarchy": require_hierarchy,
            }
        )

    monkeypatch.setattr(admin_rbac, "_enforce_admin_user_scope", _fake_scope_check)
    monkeypatch.setattr(admin_rbac, "_get_kanban_db_for_user_id", lambda user_id: _FakeKanbanDB())

    principal = SimpleNamespace(user_id=1)
    response = await admin_rbac.admin_kanban_fts_maintenance(
        "optimize",
        user_id=42,
        principal=principal,
    )

    assert response.status == "ok"
    assert calls == [
        {"principal": principal, "target_user_id": 42, "require_hierarchy": True},
        "optimize",
        "close",
    ]


@pytest.mark.asyncio
async def test_admin_kanban_fts_maintenance_closes_db_when_operation_fails(monkeypatch) -> None:
    from tldw_Server_API.app.api.v1.endpoints.admin import admin_rbac

    calls: list[str] = []

    class _FakeKanbanDB:
        def rebuild_fts(self) -> None:
            calls.append("rebuild")
            raise admin_rbac.KanbanDBError("boom")

        def close(self) -> None:
            calls.append("close")

    async def _fake_scope_check(*_args, **_kwargs) -> None:
        calls.append("scope")

    monkeypatch.setattr(admin_rbac, "_enforce_admin_user_scope", _fake_scope_check)
    monkeypatch.setattr(admin_rbac, "_get_kanban_db_for_user_id", lambda user_id: _FakeKanbanDB())

    with pytest.raises(HTTPException) as exc_info:
        await admin_rbac.admin_kanban_fts_maintenance(
            "rebuild",
            user_id=42,
            principal=SimpleNamespace(user_id=1),
        )

    assert exc_info.value.status_code == 500
    assert calls == ["scope", "rebuild", "close"]


def test_admin_backend_detector_is_loaded_via_root_compat_shim(monkeypatch) -> None:
    from tldw_Server_API.app.api.v1.endpoints import admin as admin_mod
    from tldw_Server_API.app.api.v1.endpoints.admin import admin_rate_limits, admin_rbac

    async def _fake_is_postgres_backend() -> bool:
        return True

    monkeypatch.setattr(admin_mod, "_is_postgres_backend", _fake_is_postgres_backend)

    assert admin_rbac._get_is_postgres_backend_fn() is _fake_is_postgres_backend
    assert admin_rate_limits._get_is_postgres_backend_fn() is _fake_is_postgres_backend


def test_admin_ops_platform_admin_check_uses_root_compat_shim(monkeypatch) -> None:
    from tldw_Server_API.app.api.v1.endpoints import admin as admin_mod
    from tldw_Server_API.app.api.v1.endpoints.admin import admin_ops

    calls: dict[str, object] = {}

    def _fake_require_platform_admin(principal) -> None:
        calls["principal"] = principal

    monkeypatch.setattr(admin_mod, "_require_platform_admin", _fake_require_platform_admin)

    principal = SimpleNamespace(user_id=99)
    admin_ops._require_platform_admin(principal)

    assert calls["principal"] is principal


@pytest.mark.asyncio
async def test_admin_ops_org_scope_loader_uses_root_compat_shim(monkeypatch) -> None:
    from tldw_Server_API.app.api.v1.endpoints import admin as admin_mod
    from tldw_Server_API.app.api.v1.endpoints.admin import admin_ops

    calls: dict[str, object] = {}

    async def _fake_get_admin_org_ids(principal) -> list[int]:
        calls["principal"] = principal
        return [11, 22]

    monkeypatch.setattr(admin_mod, "_get_admin_org_ids", _fake_get_admin_org_ids)

    principal = SimpleNamespace(user_id=7)
    org_ids = await admin_ops._get_admin_org_ids(principal)

    assert org_ids == [11, 22]
    assert calls["principal"] is principal


@pytest.mark.asyncio
async def test_admin_data_ops_scope_enforcement_uses_root_compat_shim(monkeypatch) -> None:
    from tldw_Server_API.app.api.v1.endpoints import admin as admin_mod
    from tldw_Server_API.app.api.v1.endpoints.admin import admin_data_ops

    calls: dict[str, object] = {}

    async def _fake_scope_check(principal, target_user_id: int, *, require_hierarchy: bool) -> None:
        calls["principal"] = principal
        calls["target_user_id"] = target_user_id
        calls["require_hierarchy"] = require_hierarchy

    monkeypatch.setattr(admin_mod, "_enforce_admin_user_scope", _fake_scope_check)

    principal = SimpleNamespace(user_id=5)
    await admin_data_ops._enforce_admin_user_scope(principal, 123, require_hierarchy=False)

    assert calls == {
        "principal": principal,
        "target_user_id": 123,
        "require_hierarchy": False,
    }


def test_admin_ensure_sqlite_resolvers_use_root_shim(monkeypatch) -> None:
    from tldw_Server_API.app.api.v1.endpoints import admin as admin_mod
    from tldw_Server_API.app.api.v1.endpoints.admin import admin_llm_providers, admin_orgs

    async def _fake_ensure_sqlite_ready() -> None:
        return None

    monkeypatch.setattr(admin_mod, "_ensure_sqlite_authnz_ready_if_test_mode", _fake_ensure_sqlite_ready)

    assert admin_orgs._get_ensure_sqlite_authnz_ready_if_test_mode() is _fake_ensure_sqlite_ready
    assert admin_llm_providers._get_ensure_sqlite_authnz_ready_if_test_mode() is _fake_ensure_sqlite_ready


@pytest.mark.asyncio
async def test_admin_data_and_ops_audit_wrappers_use_root_shim(monkeypatch) -> None:
    from tldw_Server_API.app.api.v1.endpoints import admin as admin_mod
    from tldw_Server_API.app.api.v1.endpoints.admin import admin_data_ops, admin_ops

    calls: list[dict[str, object]] = []

    async def _fake_emit(*args, **kwargs) -> None:
        calls.append({"args": args, "kwargs": kwargs})

    monkeypatch.setattr(admin_mod, "_emit_admin_audit_event", _fake_emit)

    request = SimpleNamespace()
    principal = SimpleNamespace(user_id=1)

    await admin_data_ops._emit_admin_audit_event(
        request,
        principal,
        event_type="config.changed",
        category="system",
        resource_type="backup",
        resource_id="id-1",
        action="backup.create",
        metadata={"source": "test"},
    )
    await admin_ops._emit_admin_audit_event(
        request,
        principal,
        event_type="ops.incident",
        category="system",
        resource_type="incident",
        resource_id="inc-1",
        action="incident.update",
        metadata={"source": "test"},
    )

    assert len(calls) == 2
    assert calls[0]["kwargs"]["action"] == "backup.create"
    assert calls[1]["kwargs"]["action"] == "incident.update"


@pytest.mark.asyncio
async def test_admin_root_emit_audit_event_sanitizes_noncritical_warning(monkeypatch) -> None:
    from tldw_Server_API.app.api.v1.API_Deps import Audit_DB_Deps
    from tldw_Server_API.app.api.v1.endpoints import admin as admin_mod

    warning_records: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    async def _raise_audit_service(_actor_id: int):
        raise RuntimeError("audit backend exploded at /private/audit.db")

    monkeypatch.setattr(
        Audit_DB_Deps,
        "get_or_create_audit_service_for_user_id",
        _raise_audit_service,
    )
    monkeypatch.setattr(
        admin_mod.logger,
        "warning",
        lambda message, *args, **kwargs: warning_records.append((str(message), args, kwargs)),
    )

    request = SimpleNamespace(
        state=SimpleNamespace(user_id=7),
        headers={},
        client=None,
        url=SimpleNamespace(path="/api/v1/admin/test"),
        method="POST",
    )
    principal = SimpleNamespace(user_id=7)

    await admin_mod._emit_admin_audit_event(
        request,
        principal,
        event_type="config.changed",
        category="system",
        resource_type="test",
        resource_id="id-1",
        action="test.update",
        metadata={"source": "test"},
    )

    assert warning_records == [("Admin audit emission failed", (), {})]


@pytest.mark.asyncio
async def test_admin_root_emit_budget_audit_event_forwards_to_service(monkeypatch) -> None:
    from tldw_Server_API.app.api.v1.endpoints import admin as admin_mod

    calls: list[dict[str, object]] = []

    async def _fake_emit_budget(*args, **kwargs) -> None:
        calls.append({"args": args, "kwargs": kwargs})

    monkeypatch.setattr(
        "tldw_Server_API.app.services.budget_audit_service.emit_budget_audit_event",
        _fake_emit_budget,
    )

    request = SimpleNamespace()
    principal = SimpleNamespace(user_id=1)
    await admin_mod.emit_budget_audit_event(
        request,
        principal,
        org_id=7,
        budget_updates={"daily_cap": 10.0},
        audit_changes=[],
        clear_budgets=False,
        actor_role="admin",
    )

    assert len(calls) == 1
    assert calls[0]["kwargs"]["org_id"] == 7
    assert calls[0]["kwargs"]["actor_role"] == "admin"


def test_admin_root_platform_admin_helpers_delegate_to_scope_service(monkeypatch) -> None:
    from tldw_Server_API.app.api.v1.endpoints import admin as admin_mod
    from tldw_Server_API.app.services import admin_scope_service

    principal = SimpleNamespace(user_id=13)
    calls: dict[str, object] = {}

    def _fake_is_platform_admin(p) -> bool:
        calls["is_platform_admin"] = p
        return True

    def _fake_require_platform_admin(p) -> None:
        calls["require_platform_admin"] = p

    monkeypatch.setattr(admin_scope_service, "is_platform_admin", _fake_is_platform_admin)
    monkeypatch.setattr(admin_scope_service, "require_platform_admin", _fake_require_platform_admin)

    assert admin_mod._is_platform_admin(principal) is True
    admin_mod._require_platform_admin(principal)
    assert calls["is_platform_admin"] is principal
    assert calls["require_platform_admin"] is principal


@pytest.mark.asyncio
async def test_admin_root_org_ids_and_bulk_candidates_delegate_to_services(monkeypatch) -> None:
    from tldw_Server_API.app.api.v1.endpoints import admin as admin_mod
    from tldw_Server_API.app.services import admin_profiles_service, admin_scope_service

    principal = SimpleNamespace(user_id=21)
    calls: dict[str, object] = {}

    async def _fake_get_admin_org_ids(p) -> list[int]:
        calls["org_principal"] = p
        return [1, 2]

    async def _fake_load_bulk_user_candidates(*, principal, org_id, team_id, role, is_active, search, user_ids):
        calls["bulk"] = {
            "principal": principal,
            "org_id": org_id,
            "team_id": team_id,
            "role": role,
            "is_active": is_active,
            "search": search,
            "user_ids": user_ids,
        }
        return [31, 32]

    monkeypatch.setattr(admin_scope_service, "get_admin_org_ids", _fake_get_admin_org_ids)
    monkeypatch.setattr(admin_profiles_service, "_load_bulk_user_candidates", _fake_load_bulk_user_candidates)

    org_ids = await admin_mod._get_admin_org_ids(principal)
    bulk_ids = await admin_mod._load_bulk_user_candidates(
        principal=principal,
        org_id=1,
        team_id=None,
        role=None,
        is_active=True,
        search="alice",
        user_ids=[31, 32],
    )

    assert org_ids == [1, 2]
    assert bulk_ids == [31, 32]
    assert calls["org_principal"] is principal
    assert calls["bulk"]["principal"] is principal
