import asyncio
import contextlib
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


@pytest.mark.asyncio
async def test_purge_for_user_retention_candidate_query_warning_is_sanitized(monkeypatch):
    class _FailingBackend:
        def execute(self, query, params):
            if "retention_until" in query:
                raise RuntimeError("secret /tmp/retention-path sk-live-retention")
            if "deleted = 1" in query:
                return SimpleNamespace(rows=[])
            raise AssertionError(f"Unexpected query: {query}")

    logger = _LoggerStub()
    metrics = _MetricsStub()
    monkeypatch.setattr(scheduler, "logger", logger)
    monkeypatch.setattr(scheduler, "get_metrics_registry", lambda: metrics)
    monkeypatch.setattr(
        scheduler.CollectionsDatabase,
        "for_user",
        lambda user_id: SimpleNamespace(backend=_FailingBackend()),
    )

    removed, files_deleted = await scheduler._purge_for_user(user_id=7, delete_files=False, grace_days=30)

    assert (removed, files_deleted) == (0, 0)
    assert logger.warnings == ["outputs_purge: error selecting retention candidates for user 7"]
    assert logger.binds[-1] == {"error_type": "RuntimeError"}
    logged = "\n".join(logger.debugs + logger.infos + logger.warnings)
    assert "/tmp/retention-path" not in logged
    assert "sk-live-retention" not in logged
    assert metrics.calls == [
        (
            "app_exception_events_total",
            {"labels": {"component": "outputs_purge", "event": "select_retention_candidates_failed"}},
        )
    ]


@pytest.mark.asyncio
async def test_purge_for_user_deleted_candidate_query_warning_is_sanitized(monkeypatch):
    class _FailingBackend:
        def execute(self, query, params):
            if "retention_until" in query:
                return SimpleNamespace(rows=[])
            if "deleted = 1" in query:
                raise RuntimeError("secret /tmp/deleted-path sk-live-deleted")
            raise AssertionError(f"Unexpected query: {query}")

    logger = _LoggerStub()
    metrics = _MetricsStub()
    monkeypatch.setattr(scheduler, "logger", logger)
    monkeypatch.setattr(scheduler, "get_metrics_registry", lambda: metrics)
    monkeypatch.setattr(
        scheduler.CollectionsDatabase,
        "for_user",
        lambda user_id: SimpleNamespace(backend=_FailingBackend()),
    )

    removed, files_deleted = await scheduler._purge_for_user(user_id=7, delete_files=False, grace_days=30)

    assert (removed, files_deleted) == (0, 0)
    assert logger.warnings == ["outputs_purge: error selecting deleted candidates for user 7"]
    assert logger.binds[-1] == {"error_type": "RuntimeError"}
    logged = "\n".join(logger.debugs + logger.infos + logger.warnings)
    assert "/tmp/deleted-path" not in logged
    assert "sk-live-deleted" not in logged
    assert metrics.calls == [
        (
            "app_exception_events_total",
            {"labels": {"component": "outputs_purge", "event": "select_deleted_candidates_failed"}},
        )
    ]


@pytest.mark.asyncio
async def test_purge_for_user_uses_managed_media_database(monkeypatch):
    events = []

    class _FakeBackend:
        def __init__(self):
            self.delete_calls = []

        def execute(self, query, params):
            if "retention_until" in query:
                return SimpleNamespace(rows=[{"id": 12, "storage_path": "reports/file.txt"}])
            if "deleted = 1" in query:
                return SimpleNamespace(rows=[])
            if query.startswith("DELETE FROM outputs"):
                self.delete_calls.append((query, params))
                return SimpleNamespace(rows=[])
            raise AssertionError(f"Unexpected query: {query}")

    class _FakeMediaDb:
        def mark_tts_history_artifacts_deleted_for_output(self, **kwargs):
            events.append(("mark", kwargs))

    @contextlib.contextmanager
    def _fake_managed_media_database(client_id, **kwargs):
        events.append(("open", client_id, kwargs))
        yield _FakeMediaDb()

    fake_backend = _FakeBackend()
    fake_cdb = SimpleNamespace(backend=fake_backend)

    monkeypatch.setattr(
        scheduler.CollectionsDatabase,
        "for_user",
        lambda user_id: fake_cdb,
    )
    monkeypatch.setattr(
        scheduler.DatabasePaths,
        "get_media_db_path",
        lambda user_id: f"/tmp/media-{user_id}.db",
    )
    monkeypatch.setattr(
        scheduler,
        "MediaDatabase",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("outputs_purge should not construct MediaDatabase directly")
        ),
        raising=False,
    )
    monkeypatch.setattr(
        scheduler,
        "managed_media_database",
        _fake_managed_media_database,
        raising=False,
    )
    monkeypatch.setattr(
        scheduler,
        "get_metrics_registry",
        lambda: SimpleNamespace(increment=lambda *args, **kwargs: None),
    )

    removed, files_deleted = await scheduler._purge_for_user(
        user_id=42,
        delete_files=False,
        grace_days=30,
    )

    assert (removed, files_deleted) == (1, 0)
    assert events == [
        ("open", "outputs_purge", {"db_path": "/tmp/media-42.db", "initialize": False}),
        (
            "mark",
            {
                "user_id": "42",
                "output_id": 12,
            },
        ),
    ]
    assert len(fake_backend.delete_calls) == 1
