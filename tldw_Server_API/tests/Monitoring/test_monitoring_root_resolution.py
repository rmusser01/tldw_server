import builtins
from pathlib import Path

import pytest

from tldw_Server_API.app.api.v1.endpoints import monitoring as monitoring_ep
from tldw_Server_API.app.core.DB_Management.TopicMonitoring_DB import TopicMonitoringDB
from tldw_Server_API.app.core.Monitoring.topic_monitoring_service import (
    get_topic_monitoring_service,
    _reset_topic_monitoring_service,
)


pytestmark = pytest.mark.unit


def test_find_project_root_walks_up_to_pyproject(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    nested = repo_root / "tldw_Server_API" / "app" / "api" / "v1" / "endpoints"
    nested.mkdir(parents=True)
    (repo_root / "pyproject.toml").write_text("[tool]\n", encoding="utf-8")

    found = monitoring_ep._find_project_root(nested / "monitoring.py")
    assert found == repo_root


def test_get_topic_monitoring_db_raises_on_invalid_env_value(monkeypatch: pytest.MonkeyPatch) -> None:
    class _BadPath:
        pass

    def _fake_getenv(key: str, default: str | None = None):  # type: ignore[override]
        if key == "MONITORING_ALERTS_DB":
            return _BadPath()
        return default

    monkeypatch.setattr(monitoring_ep.os, "getenv", _fake_getenv)

    with pytest.raises(RuntimeError) as exc_info:
        monitoring_ep.get_topic_monitoring_db()
    assert "Invalid MONITORING_ALERTS_DB" in str(exc_info.value)
    assert "_BadPath" in str(exc_info.value)


def test_get_topic_monitoring_db_uses_fallback_root_when_get_project_root_import_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _StubDB:
        def __init__(self, db_path: str) -> None:
            self.db_path = db_path

    original_import = builtins.__import__

    def _fake_import(name, globals=None, locals=None, fromlist=(), level=0):  # type: ignore[no-untyped-def]
        if name == "tldw_Server_API.app.core.Utils.Utils":
            raise ImportError("import boom")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("tldw_Server_API.app.core.DB_Management.TopicMonitoring_DB.TopicMonitoringDB", _StubDB)
    monkeypatch.setattr(builtins, "__import__", _fake_import)
    monkeypatch.setenv("MONITORING_ALERTS_DB", "relative/alerts.db")
    monkeypatch.setattr(monitoring_ep, "_find_project_root", lambda _start: tmp_path)

    db = monitoring_ep.get_topic_monitoring_db()
    assert db.db_path == str((tmp_path / "relative/alerts.db").resolve())


def test_get_topic_monitoring_db_raises_when_import_and_fallback_root_search_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__

    def _fake_import(name, globals=None, locals=None, fromlist=(), level=0):  # type: ignore[no-untyped-def]
        if name == "tldw_Server_API.app.core.Utils.Utils":
            raise ImportError("import boom")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    monkeypatch.setenv("MONITORING_ALERTS_DB", "relative/alerts.db")
    monkeypatch.setattr(monitoring_ep, "_find_project_root", lambda _start: None)

    with pytest.raises(RuntimeError) as exc_info:
        monitoring_ep.get_topic_monitoring_db()
    assert "importing get_project_root failed: import boom" in str(exc_info.value)


def test_get_topic_monitoring_db_rejects_relative_path_without_directory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("MONITORING_ALERTS_DB", "alerts.db")
    monitoring_ep._TOPIC_MONITORING_DB = None

    with pytest.raises(RuntimeError) as exc_info:
        monitoring_ep.get_topic_monitoring_db()
    msg = str(exc_info.value)
    assert "MONITORING_ALERTS_DB" in msg
    assert "directory" in msg


def test_topic_monitoring_service_rejects_watchlists_path_without_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MONITORING_ALERTS_DB", str(tmp_path / "alerts.db"))
    monkeypatch.setenv("MONITORING_WATCHLISTS_FILE", "watchlists.json")
    _reset_topic_monitoring_service()

    with pytest.raises(RuntimeError) as exc_info:
        get_topic_monitoring_service()
    msg = str(exc_info.value)
    assert "MONITORING_WATCHLISTS_FILE" in msg
    assert "directory" in msg


def test_topic_monitoring_db_rejects_relative_path_without_directory() -> None:
    with pytest.raises(RuntimeError) as exc_info:
        TopicMonitoringDB(db_path="monitoring_alerts.db")
    assert "directory" in str(exc_info.value)
