import importlib
import os

import pytest


def _has_psycopg() -> bool:


    try:
        import psycopg  # noqa: F401
        return True
    except Exception:
        return False


def test_store_env_overrides_stale_settings_attrs(monkeypatch, tmp_path):
    from tldw_Server_API.app.core.config import clear_config_cache, settings as app_settings

    db_path = str(tmp_path / "sandbox_store.db")
    monkeypatch.setenv("SANDBOX_STORE_BACKEND", "sqlite")
    monkeypatch.setenv("SANDBOX_STORE_DB_PATH", db_path)
    if hasattr(app_settings, "SANDBOX_STORE_BACKEND"):
        monkeypatch.setattr(app_settings, "SANDBOX_STORE_BACKEND", "memory")
    if hasattr(app_settings, "SANDBOX_STORE_DB_PATH"):
        monkeypatch.setattr(app_settings, "SANDBOX_STORE_DB_PATH", str(tmp_path / "stale.db"))
    clear_config_cache()

    from tldw_Server_API.app.core.Sandbox import store as sbx_store
    importlib.reload(sbx_store)

    assert sbx_store.get_store_mode() == "sqlite"
    st = sbx_store.get_store()
    assert isinstance(st, sbx_store.SQLiteStore)
    assert st.db_path == db_path


def test_cluster_mode_without_postgres_is_fail_fast(monkeypatch):


     # Request cluster without DSN; expect explicit startup/runtime failure
    from tldw_Server_API.app.core.config import clear_config_cache

    monkeypatch.setenv("SANDBOX_STORE_BACKEND", "cluster")
    monkeypatch.delenv("SANDBOX_STORE_PG_DSN", raising=False)
    monkeypatch.delenv("SANDBOX_PG_DSN", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    clear_config_cache()

    from tldw_Server_API.app.core.Sandbox import store as sbx_store
    importlib.reload(sbx_store)

    with pytest.raises(RuntimeError):
        _ = sbx_store.get_store_mode()
    with pytest.raises(RuntimeError):
        _ = sbx_store.get_store()


@pytest.mark.integration
def test_cluster_mode_smoke_with_postgres(monkeypatch):
     # Requires SANDBOX_TEST_PG_DSN and psycopg installed
    dsn = os.getenv("SANDBOX_TEST_PG_DSN")
    if not dsn or not _has_psycopg():
        pytest.skip("Postgres DSN not provided or psycopg not installed")

    from tldw_Server_API.app.core.config import clear_config_cache

    monkeypatch.setenv("SANDBOX_STORE_BACKEND", "cluster")
    monkeypatch.setenv("SANDBOX_STORE_PG_DSN", dsn)
    clear_config_cache()

    from tldw_Server_API.app.core.Sandbox import store as sbx_store
    importlib.reload(sbx_store)

    assert sbx_store.get_store_mode() == "cluster"

    # Basic connectivity check: construct store and call a simple method
    st = sbx_store.get_store()
    assert hasattr(st, "count_runs")
    # Should not raise
    _ = int(st.count_runs())
