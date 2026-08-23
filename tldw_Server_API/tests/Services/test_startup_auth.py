from __future__ import annotations

import importlib
import io
import sys
from types import ModuleType, SimpleNamespace

import pytest

from tldw_Server_API.app.core.AuthNZ.exceptions import DatabaseError

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


def _noop_start_llm_provider_override_refresh_service() -> None:
    return None


@pytest.mark.unit
def test_startup_auth_exception_guards_match_lifespan_contract() -> None:
    startup_auth = _import_startup_auth()

    assert (
        AttributeError,
        DatabaseError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) == startup_auth._STARTUP_GUARD_EXCEPTIONS
    assert (
        AssertionError,
        ImportError,
        ModuleNotFoundError,
        AttributeError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) == startup_auth._IMPORT_EXCEPTIONS


@pytest.mark.unit
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

    def _fake_start_refresh_service():
        calls.append("start_override_refresh_service")

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
        start_llm_provider_override_refresh_service=_fake_start_refresh_service,
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
        "start_override_refresh_service",
    ]


@pytest.mark.unit
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
        start_llm_provider_override_refresh_service=(
            _noop_start_llm_provider_override_refresh_service
        ),
        set_llm_provider_overrides_cache_for_tests=_noop_reset_llm_provider_overrides_cache,
    )
    _install_module(
        monkeypatch,
        "tldw_Server_API.app.core.AuthNZ.pg_migrations_extra",
        ensure_user_timestamp_timezones_pg=_make_pg_ensure("user_timestamps"),
        ensure_authnz_core_tables_pg=_make_pg_ensure("authnz_core"),
        ensure_admin_webhook_canonical_tables_pg=_make_pg_ensure(
            "admin_webhook_canonical"
        ),
        ensure_notification_permissions_pg=_make_pg_ensure("notification_permissions"),
        ensure_sharing_tables_pg=_make_pg_ensure("sharing"),
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
        "user_timestamps",
        "authnz_core",
        "admin_webhook_canonical",
        "sharing",
        "notification_permissions",
        "generated_files",
        "tool_catalogs",
        "privilege_snapshots",
        "api_keys",
        "usage",
        "virtual_key_counters",
        "provider_overrides",
    ]


@pytest.mark.unit
async def test_pg_ensure_false_emits_high_signal_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _successful_ensure(_pool: object) -> bool:
        return True

    async def _failed_ensure(_pool: object) -> bool:
        return False

    pg_ensures = {
        "ensure_user_timestamp_timezones_pg": _successful_ensure,
        "ensure_authnz_core_tables_pg": _successful_ensure,
        "ensure_admin_webhook_canonical_tables_pg": _successful_ensure,
        "ensure_notification_permissions_pg": _failed_ensure,
        "ensure_sharing_tables_pg": _successful_ensure,
        "ensure_generated_files_table_pg": _successful_ensure,
        "ensure_tool_catalogs_tables_pg": _successful_ensure,
        "ensure_privilege_snapshots_table_pg": _successful_ensure,
        "ensure_api_keys_tables_pg": _successful_ensure,
        "ensure_usage_tables_pg": _successful_ensure,
        "ensure_virtual_key_counters_pg": _successful_ensure,
        "ensure_llm_provider_overrides_pg": _successful_ensure,
    }
    _install_module(
        monkeypatch,
        "tldw_Server_API.app.core.AuthNZ.pg_migrations_extra",
        **pg_ensures,
    )
    startup_auth = _import_startup_auth()
    warnings: list[str] = []
    monkeypatch.setattr(startup_auth.logger, "warning", warnings.append)

    await startup_auth._ensure_pg_extras(SimpleNamespace(pool=object()))

    assert warnings == [
        "App Startup: PG notification permissions ensure returned False; "
        "canonical database state may be incomplete"
    ]


@pytest.mark.unit
async def test_pg_authnz_core_readiness_failure_blocks_startup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _successful_ensure(_pool: object) -> bool:
        return True

    async def _profile_version_not_ready(_pool: object) -> bool:
        raise RuntimeError("AuthNZ profile_version readiness validation failed")

    pg_ensures = {
        "ensure_user_timestamp_timezones_pg": _successful_ensure,
        "ensure_authnz_core_tables_pg": _profile_version_not_ready,
        "ensure_admin_webhook_canonical_tables_pg": _successful_ensure,
        "ensure_sharing_tables_pg": _successful_ensure,
        "ensure_notification_permissions_pg": _successful_ensure,
        "ensure_generated_files_table_pg": _successful_ensure,
        "ensure_tool_catalogs_tables_pg": _successful_ensure,
        "ensure_privilege_snapshots_table_pg": _successful_ensure,
        "ensure_api_keys_tables_pg": _successful_ensure,
        "ensure_usage_tables_pg": _successful_ensure,
        "ensure_virtual_key_counters_pg": _successful_ensure,
        "ensure_llm_provider_overrides_pg": _successful_ensure,
    }
    _install_module(
        monkeypatch,
        "tldw_Server_API.app.core.AuthNZ.pg_migrations_extra",
        **pg_ensures,
    )
    startup_auth = _import_startup_auth()

    with pytest.raises(
        startup_auth.AuthStartupError,
        match="AUTHNZ_CORE_SCHEMA_NOT_READY",
    ) as exc_info:
        await startup_auth._ensure_pg_extras(SimpleNamespace(pool=object()))

    assert exc_info.value.__cause__ is None


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_mode", ["false", "exception"])
@pytest.mark.unit
async def test_pg_user_timestamp_readiness_failure_blocks_startup(
    monkeypatch: pytest.MonkeyPatch,
    failure_mode: str,
) -> None:
    async def _successful_ensure(_pool: object) -> bool:
        return True

    async def _timestamp_ensure(_pool: object) -> bool:
        if failure_mode == "exception":
            raise RuntimeError("private timestamp migration failure")
        return False

    pg_ensures = {
        "ensure_user_timestamp_timezones_pg": _timestamp_ensure,
        "ensure_authnz_core_tables_pg": _successful_ensure,
        "ensure_admin_webhook_canonical_tables_pg": _successful_ensure,
        "ensure_sharing_tables_pg": _successful_ensure,
        "ensure_notification_permissions_pg": _successful_ensure,
        "ensure_generated_files_table_pg": _successful_ensure,
        "ensure_tool_catalogs_tables_pg": _successful_ensure,
        "ensure_privilege_snapshots_table_pg": _successful_ensure,
        "ensure_api_keys_tables_pg": _successful_ensure,
        "ensure_usage_tables_pg": _successful_ensure,
        "ensure_virtual_key_counters_pg": _successful_ensure,
        "ensure_llm_provider_overrides_pg": _successful_ensure,
    }
    _install_module(
        monkeypatch,
        "tldw_Server_API.app.core.AuthNZ.pg_migrations_extra",
        **pg_ensures,
    )
    startup_auth = _import_startup_auth()

    with pytest.raises(
        startup_auth.AuthStartupError,
        match="AUTHNZ_USER_TIMESTAMPS_NOT_READY",
    ) as exc_info:
        await startup_auth._ensure_pg_extras(SimpleNamespace(pool=object()))

    assert exc_info.value.__cause__ is None


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_mode", ["false", "exception"])
@pytest.mark.unit
async def test_pg_sharing_readiness_failure_blocks_startup(
    monkeypatch: pytest.MonkeyPatch,
    failure_mode: str,
) -> None:
    async def _successful_ensure(_pool: object) -> bool:
        return True

    async def _failed_sharing_ensure(_pool: object) -> bool:
        if failure_mode == "exception":
            raise RuntimeError("private sharing migration failure")
        return False

    pg_ensures = {
        "ensure_user_timestamp_timezones_pg": _successful_ensure,
        "ensure_authnz_core_tables_pg": _successful_ensure,
        "ensure_admin_webhook_canonical_tables_pg": _successful_ensure,
        "ensure_sharing_tables_pg": _failed_sharing_ensure,
        "ensure_notification_permissions_pg": _successful_ensure,
        "ensure_generated_files_table_pg": _successful_ensure,
        "ensure_tool_catalogs_tables_pg": _successful_ensure,
        "ensure_privilege_snapshots_table_pg": _successful_ensure,
        "ensure_api_keys_tables_pg": _successful_ensure,
        "ensure_usage_tables_pg": _successful_ensure,
        "ensure_virtual_key_counters_pg": _successful_ensure,
        "ensure_llm_provider_overrides_pg": _successful_ensure,
    }
    _install_module(
        monkeypatch,
        "tldw_Server_API.app.core.AuthNZ.pg_migrations_extra",
        **pg_ensures,
    )
    startup_auth = _import_startup_auth()

    with pytest.raises(
        startup_auth.AuthStartupError,
        match="AUTHNZ_PG_SHARING_SCHEMA_NOT_READY",
    ) as exc_info:
        await startup_auth._ensure_pg_extras(SimpleNamespace(pool=object()))

    assert exc_info.value.__cause__ is None


@pytest.mark.asyncio
@pytest.mark.unit
async def test_init_auth_services_raises_auth_startup_error_when_db_pool_init_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _failing_get_db_pool():
        raise DatabaseError("db boom")

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

    assert exc_info.value.__cause__ is None


@pytest.mark.unit
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


@pytest.mark.unit
async def test_db_pool_startup_failure_log_does_not_expose_exception_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = "postgres://admin:secret@private-host/auth"

    async def _failing_get_db_pool():
        raise RuntimeError(marker)

    _install_module(
        monkeypatch,
        "tldw_Server_API.app.core.AuthNZ.database",
        get_db_pool=_failing_get_db_pool,
    )
    startup_auth = _import_startup_auth()
    output = io.StringIO()
    sink = startup_auth.logger.add(output, format="{message} {extra}")
    try:
        with pytest.raises(startup_auth.AuthStartupError):
            await startup_auth.init_auth_services()
    finally:
        startup_auth.logger.remove(sink)

    assert marker not in output.getvalue()

@pytest.mark.asyncio
@pytest.mark.unit
async def test_init_auth_services_aborts_when_schema_readiness_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_pool = SimpleNamespace()
    calls: list[str] = []
    readiness_failure = RuntimeError("profile_version readiness failed")

    async def _fake_get_db_pool():
        calls.append("get_db_pool")
        return db_pool

    async def _failing_ensure_schema():
        calls.append("ensure_schema")
        raise readiness_failure

    async def _unexpected_pg_extras(_pool: object) -> None:
        calls.append("pg_extras")

    _install_module(
        monkeypatch,
        "tldw_Server_API.app.core.AuthNZ.database",
        get_db_pool=_fake_get_db_pool,
    )
    _install_module(
        monkeypatch,
        "tldw_Server_API.app.core.AuthNZ.initialize",
        ensure_authnz_schema_ready_once=_failing_ensure_schema,
    )

    startup_auth = _import_startup_auth()
    monkeypatch.setattr(startup_auth, "_ensure_pg_extras", _unexpected_pg_extras)

    with pytest.raises(
        startup_auth.AuthStartupError,
        match="AUTHNZ_SCHEMA_NOT_READY",
    ) as exc_info:
        await startup_auth.init_auth_services()

    assert exc_info.value.__cause__ is None
    assert calls == ["get_db_pool", "ensure_schema"]


@pytest.mark.asyncio
async def test_init_auth_services_aborts_when_schema_readiness_import_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_pool = SimpleNamespace()

    async def _fake_get_db_pool():
        return db_pool

    _install_module(
        monkeypatch,
        "tldw_Server_API.app.core.AuthNZ.database",
        get_db_pool=_fake_get_db_pool,
    )
    _install_module(
        monkeypatch,
        "tldw_Server_API.app.core.AuthNZ.initialize",
        ensure_single_user_rbac_seed_if_needed=None,
    )

    startup_auth = _import_startup_auth()

    with pytest.raises(
        startup_auth.AuthStartupError,
        match="AUTHNZ_SCHEMA_READINESS_UNAVAILABLE",
    ) as exc_info:
        await startup_auth.init_auth_services()

    assert exc_info.value.__cause__ is None


@pytest.mark.unit
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

    refresh_service_starts: list[str] = []

    def _fake_start_refresh_service():
        refresh_service_starts.append("started")

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
        start_llm_provider_override_refresh_service=_fake_start_refresh_service,
        set_llm_provider_overrides_cache_for_tests=_noop_reset_llm_provider_overrides_cache,
    )

    startup_auth = _import_startup_auth()

    warnings: list[str] = []
    infos: list[str] = []
    monkeypatch.setattr(startup_auth.logger, "warning", warnings.append)
    monkeypatch.setattr(startup_auth.logger, "info", infos.append)

    result = await startup_auth.init_auth_services()

    assert result is db_pool
    assert warnings == [
        "App Startup: LLM provider overrides unavailable; server fallback disabled"
    ]
    assert "App Startup: Loaded LLM provider overrides" not in infos
    assert refresh_service_starts == ["started"]
