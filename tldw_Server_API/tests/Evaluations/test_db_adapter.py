import importlib

import pytest


@pytest.mark.unit
def test_sqlite_adapter_uses_shared_sqlite_policy_helper(tmp_path, monkeypatch):
    db_adapter_module = importlib.import_module(
        "tldw_Server_API.app.core.Evaluations.db_adapter"
    )
    webhook_module = importlib.import_module(
        "tldw_Server_API.app.core.Evaluations.webhook_manager"
    )
    original_database_type = db_adapter_module.DatabaseType
    assert webhook_module.DatabaseType is original_database_type
    calls: list[dict[str, object]] = []

    def fake_configure(conn, **kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(db_adapter_module, "configure_sqlite_connection", fake_configure)
    config = db_adapter_module.DatabaseConfig(
        db_type=webhook_module.DatabaseType.SQLITE,
        connection_string=str(tmp_path / "evaluations.db"),
    )
    adapter = db_adapter_module.DatabaseAdapterFactory.create(config)
    try:
        assert isinstance(adapter, db_adapter_module.SQLiteAdapter)
        assert adapter.conn.execute("PRAGMA mmap_size").fetchone()[0] == 268435456
    finally:
        adapter.close()

    assert calls == [{"busy_timeout_ms": 30000}]
    assert db_adapter_module.DatabaseType is original_database_type
    assert webhook_module.DatabaseType is db_adapter_module.DatabaseType
