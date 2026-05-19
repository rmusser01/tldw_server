from pathlib import Path

import pytest

from tldw_Server_API.app.core.DB_Management.Collections_DB import CollectionsDatabase
from tldw_Server_API.app.core.DB_Management.UserDatabase_v2 import UserDatabase
from tldw_Server_API.app.core.DB_Management.Workflows_Scheduler_DB import WorkflowsSchedulerDB
from tldw_Server_API.app.core.DB_Management.backends import factory as factory_mod
from tldw_Server_API.app.core.DB_Management.backends.base import BackendType, DatabaseConfig
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths


class _RecordingLogger:
    def __init__(self) -> None:
        self.info_calls: list[str] = []
        self.debug_calls: list[str] = []
        self.warning_calls: list[str] = []
        self.error_calls: list[str] = []
        self.exception_calls: list[str] = []

    def info(self, message, *args, **kwargs) -> None:
        self.info_calls.append(message.format(*args))

    def debug(self, message, *args, **kwargs) -> None:
        self.debug_calls.append(message.format(*args))

    def warning(self, message, *args, **kwargs) -> None:
        self.warning_calls.append(message.format(*args))

    def error(self, message, *args, **kwargs) -> None:
        self.error_calls.append(message.format(*args))

    def exception(self, message, *args, **kwargs) -> None:
        self.exception_calls.append(message.format(*args))


@pytest.fixture(autouse=True)
def _reset_backend_caches() -> None:
    factory_mod.close_all_backends()
    yield
    factory_mod.close_all_backends()


def _close_scheduler_backend(db: WorkflowsSchedulerDB) -> None:
    try:
        db.backend.get_pool().close_all()
    except Exception:
        pass


def test_user_database_emits_owner_backend_target_info_on_sqlite_init(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """UserDatabase should emit one contextual INFO line with backend and target metadata."""
    from tldw_Server_API.app.core.DB_Management import UserDatabase_v2 as user_db_module

    recorder = _RecordingLogger()
    monkeypatch.setattr(user_db_module, "logger", recorder, raising=True)
    monkeypatch.setattr(
        user_db_module.UserDatabase,
        "_initialize_schema",
        lambda self: None,
        raising=True,
    )

    sqlite_path = str(tmp_path / "users.db")
    cfg = DatabaseConfig(backend_type=BackendType.SQLITE, sqlite_path=sqlite_path)
    expected_target = str(Path(sqlite_path).resolve())

    UserDatabase(config=cfg, client_id="owner-log-test")

    assert recorder.info_calls == [
        f"UserDatabase initialized backend=sqlite target={expected_target} client_id=owner-log-test"
    ]


def test_user_database_postgres_log_target_avoids_credentials(monkeypatch) -> None:
    from tldw_Server_API.app.core.DB_Management import UserDatabase_v2 as user_db_module

    class _FakePostgresBackend:
        backend_type = BackendType.POSTGRESQL

        def __init__(self) -> None:
            self.config = DatabaseConfig(
                backend_type=BackendType.POSTGRESQL,
                connection_string="postgresql://owner:secret@db.internal:5433/owner_logs",
                pg_host="db.internal",
                pg_port=5433,
                pg_database="owner_logs",
            )

    recorder = _RecordingLogger()
    monkeypatch.setattr(user_db_module, "logger", recorder, raising=True)
    monkeypatch.setattr(
        user_db_module.UserDatabase,
        "_initialize_schema",
        lambda self: None,
        raising=True,
    )

    UserDatabase(backend=_FakePostgresBackend(), client_id="owner-log-test")

    assert recorder.info_calls == [
        "UserDatabase initialized backend=postgresql target=db.internal:5433/owner_logs client_id=owner-log-test"
    ]


def test_collections_database_for_user_constructor_is_info_silent(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """for_user(...) should stay silent on INFO for successful construction."""
    from tldw_Server_API.app.core.DB_Management import Collections_DB as collections_module

    recorder = _RecordingLogger()
    monkeypatch.setattr(collections_module, "logger", recorder, raising=True)
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path / "user_databases"))

    db = CollectionsDatabase.for_user(17)
    try:
        assert recorder.info_calls == []
    finally:
        db.close()


def test_workflows_scheduler_constructor_logs_single_sqlite_path_info(tmp_path: Path, monkeypatch) -> None:
    """Scheduler constructor should emit only one effective-path INFO line on init."""
    from tldw_Server_API.app.core.DB_Management import Workflows_Scheduler_DB as scheduler_module

    recorder = _RecordingLogger()
    monkeypatch.setattr(scheduler_module, "logger", recorder, raising=True)
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path / "user_databases"))
    monkeypatch.setenv("DATABASE_URL", "sqlite:///./Databases/users.db")
    monkeypatch.delenv("WORKFLOWS_SCHEDULER_DATABASE_URL", raising=False)
    monkeypatch.delenv("WORKFLOWS_SCHEDULER_SQLITE_PATH", raising=False)

    user_id = 42
    db = WorkflowsSchedulerDB(user_id=user_id)
    try:
        assert len(recorder.info_calls) == 1
        assert recorder.info_calls[0] == (
            f"WorkflowsSchedulerDB using SQLite path: "
            f"{DatabasePaths.get_workflows_scheduler_db_path(user_id).resolve()}"
        )
    finally:
        _close_scheduler_backend(db)


def test_workflows_scheduler_injected_sqlite_backend_logs_one_success_line(monkeypatch) -> None:
    from tldw_Server_API.app.core.DB_Management import Workflows_Scheduler_DB as scheduler_module

    class _FakeBackend:
        def __init__(self) -> None:
            self.config = DatabaseConfig(
                backend_type=BackendType.SQLITE,
                sqlite_path="/tmp/injected-workflows.db",
            )

    recorder = _RecordingLogger()
    monkeypatch.setattr(scheduler_module, "logger", recorder, raising=True)
    monkeypatch.setattr(
        scheduler_module.WorkflowsSchedulerDB,
        "_ensure_schema",
        lambda self: None,
        raising=True,
    )

    WorkflowsSchedulerDB(backend=_FakeBackend())

    expected_path = str(Path("/tmp/injected-workflows.db").resolve())
    assert recorder.info_calls == [
        f"WorkflowsSchedulerDB using SQLite path: {expected_path}"
    ]


def test_workflows_scheduler_does_not_log_success_info_when_init_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from tldw_Server_API.app.core.DB_Management import Workflows_Scheduler_DB as scheduler_module

    recorder = _RecordingLogger()
    monkeypatch.setattr(scheduler_module, "logger", recorder, raising=True)
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path / "user_databases"))
    monkeypatch.delenv("WORKFLOWS_SCHEDULER_DATABASE_URL", raising=False)
    monkeypatch.delenv("WORKFLOWS_SCHEDULER_SQLITE_PATH", raising=False)
    monkeypatch.setattr(
        scheduler_module.WorkflowsSchedulerDB,
        "_ensure_schema",
        lambda self: (_ for _ in ()).throw(RuntimeError("schema boom")),
        raising=True,
    )

    with pytest.raises(RuntimeError, match="schema boom"):
        WorkflowsSchedulerDB(user_id=9)

    assert recorder.info_calls == []
