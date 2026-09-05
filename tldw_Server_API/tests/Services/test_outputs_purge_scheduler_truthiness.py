import asyncio
import contextlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import tldw_Server_API.app.services.outputs_purge_scheduler as scheduler

pytestmark = pytest.mark.unit


class _LoggerStub:
    def __init__(self) -> None:
        self.debugs: list[str] = []
        self.infos: list[str] = []
        self.warnings: list[str] = []
        self.binds: list[dict[str, Any]] = []

    def bind(self, **kwargs: Any):
        self.binds.append(kwargs)
        return self

    def debug(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.debugs.append(message.format(*args) if args else message)

    def info(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.infos.append(message.format(*args) if args else message)

    def warning(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.warnings.append(message.format(*args) if args else message)


class _MetricsStub:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def increment(self, metric: str, **kwargs: Any) -> None:
        self.calls.append((metric, kwargs))


@pytest.mark.asyncio
async def test_outputs_purge_scheduler_disabled_by_default(monkeypatch):
    monkeypatch.delenv("OUTPUTS_PURGE_ENABLED", raising=False)
    task = await scheduler.start_outputs_purge_scheduler()
    assert task is None


@pytest.mark.asyncio
async def test_outputs_purge_scheduler_accepts_y_flags(monkeypatch):
    monkeypatch.setenv("OUTPUTS_PURGE_ENABLED", "y")
    monkeypatch.setenv("OUTPUTS_PURGE_DELETE_FILES", "y")
    monkeypatch.setenv("OUTPUTS_PURGE_INTERVAL_SEC", "1")

    calls: list[tuple[int, bool, int]] = []

    monkeypatch.setattr(scheduler, "_enumerate_user_ids", lambda: [42])

    async def _fake_purge_for_user(user_id: int, delete_files: bool, grace_days: int):
        calls.append((user_id, delete_files, grace_days))
        return (0, 0)

    monkeypatch.setattr(scheduler, "_purge_for_user", _fake_purge_for_user)

    sleep_calls = {"count": 0}

    async def _fake_sleep(_seconds: float):
        sleep_calls["count"] += 1
        # First call is startup delay, second call is loop interval.
        # Cancel the task after one run.
        if sleep_calls["count"] >= 2:
            raise asyncio.CancelledError
        return None

    monkeypatch.setattr(scheduler.asyncio, "sleep", _fake_sleep)

    task = await scheduler.start_outputs_purge_scheduler()
    assert task is not None
    with contextlib.suppress(asyncio.CancelledError):
        await task

    assert calls
    assert calls[0] == (42, True, 30)


@pytest.mark.asyncio
async def test_outputs_purge_scheduler_invalid_env_logs_are_sanitized(monkeypatch):
    monkeypatch.setenv("OUTPUTS_PURGE_ENABLED", "true")
    monkeypatch.setenv("OUTPUTS_PURGE_INTERVAL_SEC", "not-int /private/output-token sk-live-interval")
    monkeypatch.setenv("OUTPUTS_PURGE_GRACE_DAYS", "not-int /private/grace-token sk-live-grace")

    logger = _LoggerStub()
    monkeypatch.setattr(scheduler, "logger", logger)

    created = {}

    def _fake_create_task(coro, *, name=None):
        created["name"] = name
        coro.close()
        return SimpleNamespace(name=name)

    monkeypatch.setattr(scheduler.asyncio, "create_task", _fake_create_task)

    task = await scheduler.start_outputs_purge_scheduler()

    assert task is not None
    assert created == {"name": "outputs_purge_scheduler"}
    assert logger.debugs == [
        "outputs_purge: invalid OUTPUTS_PURGE_INTERVAL_SEC; using default",
        "outputs_purge: invalid OUTPUTS_PURGE_GRACE_DAYS; using default",
    ]
    assert logger.binds == [{"error_type": "ValueError"}, {"error_type": "ValueError"}]
    logged = "\n".join(logger.debugs + logger.infos)
    assert "/private/output-token" not in logged
    assert "/private/grace-token" not in logged
    assert "sk-live-interval" not in logged
    assert "sk-live-grace" not in logged


@pytest.mark.asyncio
async def test_outputs_purge_scheduler_runner_failure_log_is_sanitized(monkeypatch):
    monkeypatch.setenv("OUTPUTS_PURGE_ENABLED", "true")
    monkeypatch.setenv("OUTPUTS_PURGE_INTERVAL_SEC", "1")

    logger = _LoggerStub()
    metrics = _MetricsStub()
    monkeypatch.setattr(scheduler, "logger", logger)
    monkeypatch.setattr(scheduler, "get_metrics_registry", lambda: metrics)

    def _fail_enumerate_user_ids():
        raise RuntimeError("secret /tmp/outputs-runner-path sk-live-runner")

    monkeypatch.setattr(scheduler, "_enumerate_user_ids", _fail_enumerate_user_ids)

    sleep_calls = {"count": 0}

    async def _fake_sleep(_seconds: float):
        sleep_calls["count"] += 1
        if sleep_calls["count"] >= 2:
            raise asyncio.CancelledError
        return None

    monkeypatch.setattr(scheduler.asyncio, "sleep", _fake_sleep)

    task = await scheduler.start_outputs_purge_scheduler()
    assert task is not None
    with contextlib.suppress(asyncio.CancelledError):
        await task

    assert logger.debugs == ["Outputs purge run failed"]
    assert logger.binds[-1] == {"error_type": "RuntimeError"}
    logged = "\n".join(logger.debugs + logger.infos + logger.warnings)
    assert "/tmp/outputs-runner-path" not in logged
    assert "sk-live-runner" not in logged
    assert metrics.calls == [
        (
            "app_exception_events_total",
            {"labels": {"component": "outputs_purge", "event": "purge_run_failed"}},
        )
    ]


def test_enumerate_user_ids_base_dir_failure_log_is_sanitized(monkeypatch):
    logger = _LoggerStub()
    metrics = _MetricsStub()
    monkeypatch.setattr(scheduler, "logger", logger)
    monkeypatch.setattr(scheduler, "get_metrics_registry", lambda: metrics)

    def _fail_get_user_db_base_dir():
        raise RuntimeError("cannot inspect /tmp/outputs-secret-token")

    monkeypatch.setattr(scheduler.DatabasePaths, "get_user_db_base_dir", _fail_get_user_db_base_dir)

    assert scheduler._enumerate_user_ids() == []
    assert logger.debugs[-1] == "outputs_purge: failed to resolve user db base dir"
    assert logger.binds[-1] == {"error_type": "RuntimeError"}
    assert "/tmp/outputs-secret-token" not in "\n".join(logger.debugs)
    assert metrics.calls == [
        (
            "app_warning_events_total",
            {"labels": {"component": "outputs_purge", "event": "settings_user_db_dir_read_failed"}},
        )
    ]


def test_enumerate_user_ids_single_user_fallback_log_is_sanitized(monkeypatch, tmp_path):
    logger = _LoggerStub()
    metrics = _MetricsStub()
    monkeypatch.setattr(scheduler, "logger", logger)
    monkeypatch.setattr(scheduler, "get_metrics_registry", lambda: metrics)
    monkeypatch.setattr(scheduler.DatabasePaths, "get_user_db_base_dir", lambda: tmp_path)

    def _fail_get_single_user_id():
        raise RuntimeError("cannot derive /tmp/outputs-single-user-secret")

    monkeypatch.setattr(scheduler.DatabasePaths, "get_single_user_id", _fail_get_single_user_id)

    assert scheduler._enumerate_user_ids() == []
    assert logger.debugs[-1] == "outputs_purge: failed to derive single_user_id"
    assert logger.binds[-1] == {"error_type": "RuntimeError"}
    assert "/tmp/outputs-single-user-secret" not in "\n".join(logger.debugs)
    assert metrics.calls == [
        (
            "app_warning_events_total",
            {"labels": {"component": "outputs_purge", "event": "single_user_id_fallback_failed"}},
        )
    ]


def test_enumerate_user_ids_skips_non_int_dir_without_echoing_name(monkeypatch, tmp_path):
    (tmp_path / "17").mkdir()
    (tmp_path / "sk-live-output-dir").mkdir()

    logger = _LoggerStub()
    metrics = _MetricsStub()
    monkeypatch.setattr(scheduler, "logger", logger)
    monkeypatch.setattr(scheduler, "get_metrics_registry", lambda: metrics)
    monkeypatch.setattr(scheduler.DatabasePaths, "get_user_db_base_dir", lambda: tmp_path)

    assert scheduler._enumerate_user_ids() == [17]
    assert logger.debugs[-1] == "outputs_purge: skipping non-int user dir"
    assert "sk-live-output-dir" not in "\n".join(logger.debugs)
    assert "invalid literal" not in "\n".join(logger.debugs)
    assert metrics.calls == [
        (
            "app_warning_events_total",
            {"labels": {"component": "outputs_purge", "event": "invalid_user_dir_name"}},
        )
    ]


@pytest.fixture
def purge_cdb(monkeypatch):
    from tldw_Server_API.app.core.DB_Management.Collections_DB import DeletedOutput

    cdb = SimpleNamespace(
        user_id="7",
        find_outputs_to_purge=lambda *_args: {12: "file.txt"},
        delete_output_artifact_record=lambda *_args, **_kwargs: DeletedOutput("file.txt", False),
    )
    monkeypatch.setattr(scheduler.CollectionsDatabase, "for_user", lambda _user_id: cdb)

    @contextlib.contextmanager
    def media_context(*_args, **_kwargs):
        yield SimpleNamespace(mark_tts_history_artifacts_deleted_for_output=lambda **_kw: None)

    monkeypatch.setattr(scheduler, "managed_media_database", media_context)
    return cdb


@pytest.mark.asyncio
async def test_purge_for_user_candidate_query_warning_is_sanitized(monkeypatch, purge_cdb):
    def fail_scan(*_args):
        raise RuntimeError("secret /tmp/retention-path sk-live-retention")

    purge_cdb.find_outputs_to_purge = fail_scan
    logger = _LoggerStub()
    metrics = _MetricsStub()
    monkeypatch.setattr(scheduler, "logger", logger)
    monkeypatch.setattr(scheduler, "get_metrics_registry", lambda: metrics)

    assert await scheduler._purge_for_user(7, False, 30) == (0, 0)
    assert logger.warnings == ["outputs_purge: error selecting purge candidates"]
    assert logger.binds == [{"error_type": "RuntimeError"}]
    assert metrics.calls == [
        (
            "app_exception_events_total",
            {"labels": {"component": "outputs_purge", "event": "select_purge_candidates_failed"}},
        )
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid", [False, True])
async def test_purge_for_user_file_failure_is_sanitized(monkeypatch, tmp_path, purge_cdb, invalid):
    from tldw_Server_API.app.core.exceptions import InvalidStoragePathError
    from tldw_Server_API.app.services import outputs_service

    logger = _LoggerStub()
    monkeypatch.setattr(outputs_service, "logger", logger)
    monkeypatch.setattr(scheduler.DatabasePaths, "get_user_outputs_dir", lambda _user_id: tmp_path)
    (tmp_path / "file.txt").write_text("payload")
    if invalid:

        def invalid_path(*_args):
            raise InvalidStoragePathError("bad /tmp/secret-path sk-live-secret")

        monkeypatch.setattr(outputs_service, "normalize_output_storage_path", invalid_path)
    else:
        original = Path.unlink

        def fail_unlink(path, *args, **kwargs):
            if path == tmp_path / "file.txt":
                raise PermissionError("bad /tmp/secret-path sk-live-secret")
            return original(path, *args, **kwargs)

        monkeypatch.setattr(Path, "unlink", fail_unlink)

    assert await scheduler._purge_for_user(7, True, 30) == (1, 0)
    assert logger.warnings == ["outputs.delete: failed to delete file"]
    assert (tmp_path / "file.txt").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("open_failure", [False, True])
async def test_purge_for_user_media_failures_are_sanitized(monkeypatch, purge_cdb, open_failure):
    logger = _LoggerStub()
    monkeypatch.setattr(scheduler, "logger", logger)

    def fail_history(**_kwargs):
        raise RuntimeError("secret /tmp/tts-history-path sk-live-secret")

    @contextlib.contextmanager
    def media_context(*_args, **_kwargs):
        if open_failure:
            raise RuntimeError("secret /tmp/media-open-path sk-live-secret")
        yield SimpleNamespace(mark_tts_history_artifacts_deleted_for_output=fail_history)

    monkeypatch.setattr(scheduler, "managed_media_database", media_context)
    assert await scheduler._purge_for_user(7, False, 30) == (1, 0)
    assert logger.debugs == [
        (
            "outputs_purge: failed to open Media DB for history update"
            if open_failure
            else "outputs_purge: failed to update tts_history for output 12"
        )
    ]
    assert logger.binds == [{"error_type": "RuntimeError"}]


@pytest.mark.asyncio
async def test_purge_for_user_db_failure_does_not_mark_history(monkeypatch, purge_cdb):
    def fail_delete(*_args, **_kwargs):
        raise RuntimeError("cannot delete /tmp/db-delete-secret sk-live-db-delete")

    def unexpected_history(*_args, **_kwargs):
        pytest.fail("failed deletion must not open history DB")

    purge_cdb.delete_output_artifact_record = fail_delete
    logger = _LoggerStub()
    metrics = _MetricsStub()
    monkeypatch.setattr(scheduler, "logger", logger)
    monkeypatch.setattr(scheduler, "get_metrics_registry", lambda: metrics)
    monkeypatch.setattr(scheduler, "managed_media_database", unexpected_history)

    assert await scheduler._purge_for_user(7, False, 30) == (0, 0)
    assert logger.warnings == ["outputs_purge: DB delete failed"]
    assert logger.binds == [{"error_type": "RuntimeError"}]
    assert metrics.calls == [
        ("app_exception_events_total", {"labels": {"component": "outputs_purge", "event": "db_delete_failed"}})
    ]


@pytest.mark.asyncio
async def test_purge_for_user_uses_managed_media_database_after_delete(monkeypatch, purge_cdb):
    events = []
    delete = purge_cdb.delete_output_artifact_record

    def record_delete(*args, **kwargs):
        events.append("delete")
        return delete(*args, **kwargs)

    purge_cdb.delete_output_artifact_record = record_delete

    @contextlib.contextmanager
    def media_context(client_id, **kwargs):
        events.append(("open", client_id, kwargs))
        yield SimpleNamespace(mark_tts_history_artifacts_deleted_for_output=lambda **kw: events.append(("mark", kw)))

    monkeypatch.setattr(scheduler, "managed_media_database", media_context)
    monkeypatch.setattr(scheduler.DatabasePaths, "get_media_db_path", lambda _uid: "/tmp/media-7.db")
    assert await scheduler._purge_for_user(7, False, 30) == (1, 0)
    assert events == [
        "delete",
        ("open", "outputs_purge", {"db_path": "/tmp/media-7.db", "initialize": False}),
        ("mark", {"user_id": "7", "output_id": 12}),
    ]
