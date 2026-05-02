from types import SimpleNamespace

import pytest
from loguru import logger

from tldw_Server_API.app.services import tts_history_cleanup_service as cleanup

pytestmark = pytest.mark.unit

_LEAK = "not-an-int /tmp/secret-token"


def _assert_safe_log(rendered: str) -> None:
    assert "not-an-int" not in rendered
    assert "/tmp/secret-token" not in rendered
    assert "invalid literal" not in rendered


def _clear_history_cleanup_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TTS_HISTORY_PURGE_INTERVAL_HOURS", raising=False)
    monkeypatch.delenv("TTS_HISTORY_RETENTION_DAYS", raising=False)
    monkeypatch.delenv("TTS_HISTORY_MAX_ROWS_PER_USER", raising=False)


def test_resolve_cleanup_settings_uses_settings_when_env_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_history_cleanup_env(monkeypatch)
    monkeypatch.setattr(cleanup.settings, "TTS_HISTORY_PURGE_INTERVAL_HOURS", 12, raising=False)
    monkeypatch.setattr(cleanup.settings, "TTS_HISTORY_RETENTION_DAYS", 45, raising=False)
    monkeypatch.setattr(cleanup.settings, "TTS_HISTORY_MAX_ROWS_PER_USER", 678, raising=False)

    assert cleanup._resolve_cleanup_settings() == (12, 45, 678)


def test_resolve_cleanup_settings_env_overrides_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_history_cleanup_env(monkeypatch)
    monkeypatch.setattr(cleanup.settings, "TTS_HISTORY_PURGE_INTERVAL_HOURS", 24, raising=False)
    monkeypatch.setattr(cleanup.settings, "TTS_HISTORY_RETENTION_DAYS", 90, raising=False)
    monkeypatch.setattr(cleanup.settings, "TTS_HISTORY_MAX_ROWS_PER_USER", 10000, raising=False)
    monkeypatch.setenv("TTS_HISTORY_PURGE_INTERVAL_HOURS", "6")
    monkeypatch.setenv("TTS_HISTORY_RETENTION_DAYS", "14")
    monkeypatch.setenv("TTS_HISTORY_MAX_ROWS_PER_USER", "321")

    assert cleanup._resolve_cleanup_settings() == (6, 14, 321)


def test_invalid_cleanup_setting_log_omits_raw_value_and_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_history_cleanup_env(monkeypatch)
    monkeypatch.setenv("TTS_HISTORY_RETENTION_DAYS", _LEAK)

    records: list[str] = []
    sink_id = logger.add(lambda message: records.append(str(message)), format="{message} {extra}")
    try:
        assert cleanup._resolve_cleanup_settings() == (24, 90, 10000)
    finally:
        logger.remove(sink_id)

    rendered = "\n".join(records)
    _assert_safe_log(rendered)
    assert "invalid TTS_HISTORY_RETENTION_DAYS value" in rendered
    assert "ValueError" in rendered


def test_enumerate_user_ids_base_dir_failure_log_is_sanitized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records: list[str] = []
    sink_id = logger.add(lambda message: records.append(str(message)), format="{message} {extra}")

    def _fail_get_user_db_base_dir():
        raise RuntimeError("cannot inspect /tmp/tts-secret-token")

    monkeypatch.setattr(cleanup.DatabasePaths, "get_user_db_base_dir", _fail_get_user_db_base_dir)

    try:
        assert cleanup._enumerate_user_ids_from_fs() == []
    finally:
        logger.remove(sink_id)

    rendered = "\n".join(records)
    assert "cannot inspect" not in rendered
    assert "/tmp/tts-secret-token" not in rendered
    assert "tts_history_cleanup: failed to resolve user db base dir" in rendered
    assert "RuntimeError" in rendered


def test_enumerate_user_ids_single_user_fallback_log_is_sanitized(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    records: list[str] = []
    sink_id = logger.add(lambda message: records.append(str(message)), format="{message} {extra}")
    monkeypatch.setattr(cleanup.DatabasePaths, "get_user_db_base_dir", lambda: tmp_path)

    def _fail_get_single_user_id():
        raise RuntimeError("cannot derive /tmp/tts-single-user-secret")

    monkeypatch.setattr(cleanup.DatabasePaths, "get_single_user_id", _fail_get_single_user_id)

    try:
        assert cleanup._enumerate_user_ids_from_fs() == []
    finally:
        logger.remove(sink_id)

    rendered = "\n".join(records)
    assert "cannot derive" not in rendered
    assert "/tmp/tts-single-user-secret" not in rendered
    assert "tts_history_cleanup: single_user_id fallback failed" in rendered
    assert "RuntimeError" in rendered


def test_enumerate_user_ids_skips_non_int_dir_without_echoing_name(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    bad_dir = tmp_path / "sk-live-tts-dir"
    good_dir = tmp_path / "7"
    bad_dir.mkdir()
    good_dir.mkdir()

    records: list[str] = []
    sink_id = logger.add(lambda message: records.append(str(message)), format="{message} {extra}")
    monkeypatch.setattr(cleanup.DatabasePaths, "get_user_db_base_dir", lambda: tmp_path)

    try:
        assert cleanup._enumerate_user_ids_from_fs() == ["7"]
    finally:
        logger.remove(sink_id)

    rendered = "\n".join(records)
    assert "sk-live-tts-dir" not in rendered
    assert "invalid literal" not in rendered
    assert "tts_history_cleanup: skipping non-int user dir" in rendered


def test_purge_with_db_failure_log_is_sanitized() -> None:
    records: list[str] = []
    sink_id = logger.add(lambda message: records.append(str(message)), format="{message} {extra}")

    class _FakeDb:
        def purge_tts_history_for_user(self, *, user_id: str, retention_days: int, max_rows: int) -> int:
            assert retention_days == 30
            assert max_rows == 100
            if user_id == "11":
                return 4
            raise RuntimeError("db locked at /tmp/tts-purge-secret-token")

    try:
        assert cleanup._purge_with_db(_FakeDb(), ["11", "22"], retention_days=30, max_rows=100) == 4
    finally:
        logger.remove(sink_id)

    rendered = "\n".join(records)
    assert "db locked" not in rendered
    assert "/tmp/tts-purge-secret-token" not in rendered
    assert "tts_history_cleanup: purge failed for user 22" in rendered
    assert "RuntimeError" in rendered


@pytest.mark.asyncio
async def test_cleanup_loop_disabled_when_interval_nonpositive(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_history_cleanup_env(monkeypatch)
    monkeypatch.setattr(cleanup.settings, "TTS_HISTORY_PURGE_INTERVAL_HOURS", 0, raising=False)
    monkeypatch.setattr(cleanup.settings, "TTS_HISTORY_RETENTION_DAYS", 90, raising=False)
    monkeypatch.setattr(cleanup.settings, "TTS_HISTORY_MAX_ROWS_PER_USER", 10000, raising=False)

    sleep_called = False

    async def _fake_sleep(_seconds: float) -> None:
        nonlocal sleep_called
        sleep_called = True

    monkeypatch.setattr(cleanup.asyncio, "sleep", _fake_sleep)

    await cleanup.run_tts_history_cleanup_loop()

    assert sleep_called is False


@pytest.mark.asyncio
async def test_cleanup_loop_disabled_when_retention_and_rows_nonpositive(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_history_cleanup_env(monkeypatch)
    monkeypatch.setattr(cleanup.settings, "TTS_HISTORY_PURGE_INTERVAL_HOURS", 24, raising=False)
    monkeypatch.setattr(cleanup.settings, "TTS_HISTORY_RETENTION_DAYS", 0, raising=False)
    monkeypatch.setattr(cleanup.settings, "TTS_HISTORY_MAX_ROWS_PER_USER", 0, raising=False)

    sleep_called = False

    async def _fake_sleep(_seconds: float) -> None:
        nonlocal sleep_called
        sleep_called = True

    monkeypatch.setattr(cleanup.asyncio, "sleep", _fake_sleep)

    await cleanup.run_tts_history_cleanup_loop()

    assert sleep_called is False


@pytest.mark.asyncio
async def test_cleanup_loop_honors_stop_event_during_initial_delay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stop_event = cleanup.asyncio.Event()
    events: list[object] = []

    monkeypatch.setattr(cleanup, "_resolve_cleanup_settings", lambda: (1, 30, 100))

    async def _fail_sleep(_seconds: float) -> None:
        raise AssertionError("initial delay should wait on the stop event")

    async def _fake_wait_for(coro, timeout: float):
        events.append(("wait_for", timeout))
        stop_event.set()
        await coro
        return True

    monkeypatch.setattr(cleanup.asyncio, "sleep", _fail_sleep)
    monkeypatch.setattr(cleanup.asyncio, "wait_for", _fake_wait_for)
    monkeypatch.setattr(
        cleanup.DatabasePaths,
        "get_media_db_path",
        lambda _uid: (_ for _ in ()).throw(AssertionError("cleanup should not run")),
    )

    await cleanup.run_tts_history_cleanup_loop(stop_event=stop_event)

    assert events == [("wait_for", 60)]


@pytest.mark.asyncio
async def test_cleanup_loop_uses_create_media_database_for_sqlite_users(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    stop_event = cleanup.asyncio.Event()
    events: list[object] = []
    user_db_paths = {
        "1": tmp_path / "probe.sqlite3",
        "11": tmp_path / "11.sqlite3",
        "22": tmp_path / "22.sqlite3",
    }
    user_removed = {"11": 3, "22": 5}

    monkeypatch.setattr(cleanup, "_resolve_cleanup_settings", lambda: (1, 30, 100))

    async def _fake_sleep(_seconds: float) -> None:
        return None

    initial_wait_seen = False

    async def _fake_wait_for(coro, timeout: float):
        nonlocal initial_wait_seen
        if not initial_wait_seen:
            initial_wait_seen = True
            events.append(("initial_wait_for", timeout))
            if hasattr(coro, "close"):
                coro.close()
            raise cleanup.asyncio.TimeoutError
        events.append(("wait_for", timeout))
        stop_event.set()
        await coro
        return True

    def _fake_create_media_database(client_id: str, **kwargs):
        db_path = kwargs.get("db_path")
        events.append(("create", client_id, db_path))
        if db_path == str(user_db_paths["1"]):
            return SimpleNamespace(
                backend_type=SimpleNamespace(name="sqlite"),
                close_connection=lambda: events.append(("close", "probe")),
            )

        user_id = next(uid for uid, path in user_db_paths.items() if str(path) == db_path)
        return SimpleNamespace(
            purge_tts_history_for_user=lambda **purge_kwargs: events.append(
                ("purge", user_id, purge_kwargs)
            ) or user_removed[user_id]
            ,
            close_connection=lambda: events.append(("close", user_id)),
        )

    monkeypatch.setattr(cleanup.asyncio, "sleep", _fake_sleep)
    monkeypatch.setattr(cleanup.asyncio, "wait_for", _fake_wait_for)
    monkeypatch.setattr(cleanup, "create_media_database", _fake_create_media_database)
    monkeypatch.setattr(cleanup.DatabasePaths, "get_single_user_id", lambda: "1")
    monkeypatch.setattr(cleanup.DatabasePaths, "get_media_db_path", lambda uid: user_db_paths[str(uid)])
    monkeypatch.setattr(cleanup, "_enumerate_user_ids_from_fs", lambda: ["11", "22"])

    await cleanup.run_tts_history_cleanup_loop(stop_event=stop_event)

    assert events == [
        ("initial_wait_for", 60),
        ("create", "tts_history_cleanup", str(user_db_paths["1"])),
        ("close", "probe"),
        ("create", "tts_history_cleanup", str(user_db_paths["11"])),
        (
            "purge",
            "11",
            {"user_id": "11", "retention_days": 30, "max_rows": 100},
        ),
        ("close", "11"),
        ("create", "tts_history_cleanup", str(user_db_paths["22"])),
        (
            "purge",
            "22",
            {"user_id": "22", "retention_days": 30, "max_rows": 100},
        ),
        ("close", "22"),
        ("wait_for", 3600),
    ]


@pytest.mark.asyncio
async def test_cleanup_loop_uses_create_media_database_for_postgres(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    stop_event = cleanup.asyncio.Event()
    events: list[object] = []
    probe_path = tmp_path / "probe.sqlite3"
    probe_db = SimpleNamespace(
        backend_type=SimpleNamespace(name="postgresql"),
        list_tts_history_user_ids=lambda: ["7", "8"],
        close_connection=lambda: events.append(("close", "probe")),
    )

    monkeypatch.setattr(cleanup, "_resolve_cleanup_settings", lambda: (1, 14, 55))

    async def _fake_sleep(_seconds: float) -> None:
        return None

    initial_wait_seen = False

    async def _fake_wait_for(coro, timeout: float):
        nonlocal initial_wait_seen
        if not initial_wait_seen:
            initial_wait_seen = True
            events.append(("initial_wait_for", timeout))
            if hasattr(coro, "close"):
                coro.close()
            raise cleanup.asyncio.TimeoutError
        events.append(("wait_for", timeout))
        stop_event.set()
        await coro
        return True

    def _fake_create_media_database(client_id: str, **kwargs):
        events.append(("create", client_id, kwargs.get("db_path")))
        return probe_db

    def _fake_purge_with_db(db, user_ids, retention_days: int, max_rows: int) -> int:
        events.append(("purge_with_db", db, list(user_ids), retention_days, max_rows))
        return 9

    monkeypatch.setattr(cleanup.asyncio, "sleep", _fake_sleep)
    monkeypatch.setattr(cleanup.asyncio, "wait_for", _fake_wait_for)
    monkeypatch.setattr(cleanup, "create_media_database", _fake_create_media_database)
    monkeypatch.setattr(cleanup, "_purge_with_db", _fake_purge_with_db)
    monkeypatch.setattr(cleanup.DatabasePaths, "get_single_user_id", lambda: "1")
    monkeypatch.setattr(cleanup.DatabasePaths, "get_media_db_path", lambda uid: probe_path)

    await cleanup.run_tts_history_cleanup_loop(stop_event=stop_event)

    assert events == [
        ("initial_wait_for", 60),
        ("create", "tts_history_cleanup", str(probe_path)),
        ("purge_with_db", probe_db, ["7", "8"], 14, 55),
        ("close", "probe"),
        ("wait_for", 3600),
    ]


@pytest.mark.asyncio
async def test_cleanup_loop_failure_log_is_sanitized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stop_event = cleanup.asyncio.Event()
    metrics_calls: list[tuple[str, dict[str, object]]] = []
    records: list[str] = []
    sink_id = logger.add(lambda message: records.append(str(message)), format="{message} {extra}")

    monkeypatch.setattr(cleanup, "_resolve_cleanup_settings", lambda: (1, 14, 55))

    async def _fake_sleep(_seconds: float) -> None:
        return None

    initial_wait_seen = False

    async def _fake_wait_for(coro, timeout: float):
        nonlocal initial_wait_seen
        if not initial_wait_seen:
            initial_wait_seen = True
            if hasattr(coro, "close"):
                coro.close()
            raise cleanup.asyncio.TimeoutError
        stop_event.set()
        await coro
        return True

    monkeypatch.setattr(cleanup.asyncio, "sleep", _fake_sleep)
    monkeypatch.setattr(cleanup.asyncio, "wait_for", _fake_wait_for)
    monkeypatch.setattr(cleanup.DatabasePaths, "get_single_user_id", lambda: "1")

    def _fail_get_media_db_path(_uid):
        raise RuntimeError("cannot open /tmp/tts-loop-secret")

    monkeypatch.setattr(cleanup.DatabasePaths, "get_media_db_path", _fail_get_media_db_path)
    monkeypatch.setattr(
        cleanup,
        "get_metrics_registry",
        lambda: SimpleNamespace(
            increment=lambda metric, **kwargs: metrics_calls.append((metric, kwargs))
        ),
    )

    try:
        await cleanup.run_tts_history_cleanup_loop(stop_event=stop_event)
    finally:
        logger.remove(sink_id)

    rendered = "\n".join(records)
    assert "cannot open" not in rendered
    assert "/tmp/tts-loop-secret" not in rendered
    assert "TTS history cleanup loop failed" in rendered
    assert "RuntimeError" in rendered
    assert metrics_calls == [
        (
            "app_exception_events_total",
            {"labels": {"component": "tts_history_cleanup", "event": "cleanup_failed"}},
        )
    ]
