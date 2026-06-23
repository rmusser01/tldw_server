from __future__ import annotations

import importlib
import sys
from types import ModuleType, SimpleNamespace

import pytest


pytestmark = pytest.mark.unit


def _install_module(
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
    **attributes: object,
) -> ModuleType:
    module = ModuleType(module_name)
    for key, value in attributes.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, module_name, module)
    return module


def _import_startup_auth() -> ModuleType:
    sys.modules.pop("tldw_Server_API.app.services.startup_auth", None)
    return importlib.import_module("tldw_Server_API.app.services.startup_auth")


def _noop_reset_llm_provider_overrides_cache(_overrides: object = None) -> None:
    return None


def test_startup_auth_exception_guards_match_lifespan_contract() -> None:
    startup_auth = _import_startup_auth()

    assert startup_auth._STARTUP_GUARD_EXCEPTIONS == (
        AttributeError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    )
    assert startup_auth._IMPORT_EXCEPTIONS == (
        AssertionError,
        ImportError,
        ModuleNotFoundError,
        AttributeError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    )


@pytest.mark.asyncio
async def test_init_auth_services_runs_sqlite_startup_chain(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[object] = []
    db_pool = SimpleNamespace()

    async def _fake_get_db_pool():
        calls.append("get_db_pool")
        return db_pool

    async def _fake_ensure_schema():
        calls.append("ensure_schema")

    async def _fake_seed():
        calls.append("seed")

    async def _fake_refresh(pool):
        calls.append(("refresh", pool))

    _install_module(
        monkeypatch,
        "tldw_Server_API.app.core.AuthNZ.database",
        get_db_pool=_fake_get_db_pool,
    )
    _install_module(
        monkeypatch,
        "tldw_Server_API.app.core.AuthNZ.initialize",
        ensure_authnz_schema_ready_once=_fake_ensure_schema,
        ensure_single_user_rbac_seed_if_needed=_fake_seed,
    )
    _install_module(
        monkeypatch,
        "tldw_Server_API.app.core.AuthNZ.llm_provider_overrides",
        refresh_llm_provider_overrides=_fake_refresh,
        set_llm_provider_overrides_cache_for_tests=_noop_reset_llm_provider_overrides_cache,
    )

    startup_auth = _import_startup_auth()

    result = await startup_auth.init_auth_services()

    assert result is db_pool
    assert calls == [
        "get_db_pool",
        "ensure_schema",
        "seed",
        ("refresh", db_pool),
    ]


@pytest.mark.asyncio
async def test_init_auth_services_runs_pg_extras_when_pool_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_pool = SimpleNamespace(pool=object())
    pg_calls: list[str] = []

    async def _fake_get_db_pool():
        return db_pool

    async def _fake_noop():
        return None

    async def _fake_refresh(_pool):
        return None

    def _make_pg_ensure(label: str):
        async def _ensure(_pool):
            pg_calls.append(label)
            return True

        return _ensure

    _install_module(
        monkeypatch,
        "tldw_Server_API.app.core.AuthNZ.database",
        get_db_pool=_fake_get_db_pool,
    )
    _install_module(
        monkeypatch,
        "tldw_Server_API.app.core.AuthNZ.initialize",
        ensure_authnz_schema_ready_once=_fake_noop,
        ensure_single_user_rbac_seed_if_needed=_fake_noop,
    )
    _install_module(
        monkeypatch,
        "tldw_Server_API.app.core.AuthNZ.llm_provider_overrides",
        refresh_llm_provider_overrides=_fake_refresh,
        set_llm_provider_overrides_cache_for_tests=_noop_reset_llm_provider_overrides_cache,
    )
    _install_module(
        monkeypatch,
        "tldw_Server_API.app.core.AuthNZ.pg_migrations_extra",
        ensure_authnz_core_tables_pg=_make_pg_ensure("authnz_core"),
        ensure_generated_files_table_pg=_make_pg_ensure("generated_files"),
        ensure_tool_catalogs_tables_pg=_make_pg_ensure("tool_catalogs"),
        ensure_privilege_snapshots_table_pg=_make_pg_ensure("privilege_snapshots"),
        ensure_api_keys_tables_pg=_make_pg_ensure("api_keys"),
        ensure_usage_tables_pg=_make_pg_ensure("usage"),
        ensure_virtual_key_counters_pg=_make_pg_ensure("virtual_key_counters"),
        ensure_llm_provider_overrides_pg=_make_pg_ensure("provider_overrides"),
    )

    startup_auth = _import_startup_auth()

    result = await startup_auth.init_auth_services()

    assert result is db_pool
    assert pg_calls == [
        "authnz_core",
        "generated_files",
        "tool_catalogs",
        "privilege_snapshots",
        "api_keys",
        "usage",
        "virtual_key_counters",
        "provider_overrides",
    ]


@pytest.mark.asyncio
async def test_init_auth_services_raises_auth_startup_error_when_db_pool_init_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _failing_get_db_pool():
        raise RuntimeError("db boom")

    _install_module(
        monkeypatch,
        "tldw_Server_API.app.core.AuthNZ.database",
        get_db_pool=_failing_get_db_pool,
    )

    startup_auth = _import_startup_auth()

    with pytest.raises(
        startup_auth.AuthStartupError,
        match="AUTHNZ_DB_POOL_STARTUP_FAILED",
    ) as exc_info:
        await startup_auth.init_auth_services()

    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert str(exc_info.value.__cause__) == "db boom"


@pytest.mark.asyncio
async def test_init_auth_services_raises_auth_startup_error_when_db_pool_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _missing_get_db_pool():
        return None

    _install_module(
        monkeypatch,
        "tldw_Server_API.app.core.AuthNZ.database",
        get_db_pool=_missing_get_db_pool,
    )

    startup_auth = _import_startup_auth()

    with pytest.raises(
        startup_auth.AuthStartupError,
        match="AUTHNZ_DB_POOL_STARTUP_RETURNED_NONE",
    ):
        await startup_auth.init_auth_services()


@pytest.mark.asyncio
async def test_init_auth_services_warns_when_schema_ensure_is_skipped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_pool = SimpleNamespace()

    async def _fake_get_db_pool():
        return db_pool

    async def _failing_ensure_schema():
        raise RuntimeError("migration unavailable")

    async def _fake_noop():
        return None

    async def _fake_refresh(_pool):
        return None

    class _FakeLogger:
        def __init__(self) -> None:
            self.warnings: list[str] = []

        def info(self, _message: str) -> None:
            pass

        def error(self, _message: str) -> None:
            pass

        def debug(self, _message: str) -> None:
            pass

        def warning(self, message: str) -> None:
            self.warnings.append(message)

    _install_module(
        monkeypatch,
        "tldw_Server_API.app.core.AuthNZ.database",
        get_db_pool=_fake_get_db_pool,
    )
    _install_module(
        monkeypatch,
        "tldw_Server_API.app.core.AuthNZ.initialize",
        ensure_authnz_schema_ready_once=_failing_ensure_schema,
        ensure_single_user_rbac_seed_if_needed=_fake_noop,
    )
    _install_module(
        monkeypatch,
        "tldw_Server_API.app.core.AuthNZ.llm_provider_overrides",
        refresh_llm_provider_overrides=_fake_refresh,
        set_llm_provider_overrides_cache_for_tests=_noop_reset_llm_provider_overrides_cache,
    )

    startup_auth = _import_startup_auth()
    fake_logger = _FakeLogger()
    monkeypatch.setattr(startup_auth, "logger", fake_logger)

    result = await startup_auth.init_auth_services()

    assert result is db_pool
    assert fake_logger.warnings == [
        "App Startup: Skipped AuthNZ SQLite migration ensure: migration unavailable"
    ]


@pytest.mark.asyncio
async def test_init_auth_services_skips_provider_override_runtime_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_pool = SimpleNamespace()

    async def _fake_get_db_pool():
        return db_pool

    async def _fake_noop():
        return None

    async def _failing_refresh(_pool):
        raise RuntimeError("override cache unavailable")

    _install_module(
        monkeypatch,
        "tldw_Server_API.app.core.AuthNZ.database",
        get_db_pool=_fake_get_db_pool,
    )
    _install_module(
        monkeypatch,
        "tldw_Server_API.app.core.AuthNZ.initialize",
        ensure_authnz_schema_ready_once=_fake_noop,
        ensure_single_user_rbac_seed_if_needed=_fake_noop,
    )
    _install_module(
        monkeypatch,
        "tldw_Server_API.app.core.AuthNZ.llm_provider_overrides",
        refresh_llm_provider_overrides=_failing_refresh,
        set_llm_provider_overrides_cache_for_tests=_noop_reset_llm_provider_overrides_cache,
    )

    startup_auth = _import_startup_auth()

    result = await startup_auth.init_auth_services()

    assert result is db_pool
