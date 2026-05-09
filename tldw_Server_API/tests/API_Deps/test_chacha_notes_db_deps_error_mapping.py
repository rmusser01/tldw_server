from collections.abc import Iterator
import sqlite3
import threading

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.API_Deps import ChaCha_Notes_DB_Deps as chacha_deps


pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def clear_chacha_dependency_state() -> Iterator[None]:
    with chacha_deps._chacha_db_lock:
        chacha_deps._chacha_db_instances.clear()
        chacha_deps._chacha_db_init_events.clear()
        chacha_deps._chacha_db_init_errors.clear()
    yield
    with chacha_deps._chacha_db_lock:
        chacha_deps._chacha_db_instances.clear()
        chacha_deps._chacha_db_init_events.clear()
        chacha_deps._chacha_db_init_errors.clear()


@pytest.mark.asyncio
async def test_chacha_init_sanitizes_runtime_errors(monkeypatch, tmp_path):
    user_dir = tmp_path / "user-123"

    def _raise_init_error(_user_id, _client_id):
        raise RuntimeError("chacha backend exploded at /private/db/path")

    monkeypatch.setattr(chacha_deps.DatabasePaths, "get_user_base_directory", lambda _user_id: user_dir)
    monkeypatch.setattr(chacha_deps, "_create_and_prepare_db", _raise_init_error)

    with pytest.raises(HTTPException) as exc_info:
        await chacha_deps._get_or_init_db_instance(123, "client-123")

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Could not initialize character & notes database for user"


@pytest.mark.asyncio
async def test_chacha_init_classifies_sqlite_corruption_without_path_leak(monkeypatch, tmp_path):
    user_dir = tmp_path / "user-321"

    def _raise_init_error(_user_id, _client_id):
        raise sqlite3.DatabaseError("database disk image is malformed at /private/db/path")

    monkeypatch.setattr(chacha_deps.DatabasePaths, "get_user_base_directory", lambda _user_id: user_dir)
    monkeypatch.setattr(chacha_deps, "_create_and_prepare_db", _raise_init_error)

    with pytest.raises(HTTPException) as exc_info:
        await chacha_deps._get_or_init_db_instance(321, "client-321")

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == chacha_deps._CHACHA_CORRUPTION_ERROR_DETAIL
    assert "/private/db/path" not in exc_info.value.detail
    assert chacha_deps.get_chacha_health_snapshot()["last_error"] == "sqlite_corruption"


@pytest.mark.asyncio
async def test_chacha_waiter_sanitizes_cached_init_errors(monkeypatch, tmp_path):
    user_dir = tmp_path / "user-456"
    init_event = threading.Event()
    init_event.set()

    monkeypatch.setattr(chacha_deps.DatabasePaths, "get_user_base_directory", lambda _user_id: user_dir)
    with chacha_deps._chacha_db_lock:
        chacha_deps._chacha_db_init_events[str(user_dir)] = init_event
        chacha_deps._chacha_db_init_errors[str(user_dir)] = RuntimeError(
            "chacha backend exploded at /private/db/path"
        )

    with pytest.raises(HTTPException) as exc_info:
        await chacha_deps._get_or_init_db_instance(456, "client-456")

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Could not initialize character & notes database for user"


@pytest.mark.asyncio
async def test_chacha_waiter_classifies_cached_sqlite_corruption(monkeypatch, tmp_path):
    user_dir = tmp_path / "user-654"
    init_event = threading.Event()
    init_event.set()

    monkeypatch.setattr(chacha_deps.DatabasePaths, "get_user_base_directory", lambda _user_id: user_dir)
    with chacha_deps._chacha_db_lock:
        chacha_deps._chacha_db_init_events[str(user_dir)] = init_event
        chacha_deps._chacha_db_init_errors[str(user_dir)] = sqlite3.DatabaseError(
            "malformed database schema at /private/db/path"
        )

    with pytest.raises(HTTPException) as exc_info:
        await chacha_deps._get_or_init_db_instance(654, "client-654")

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == chacha_deps._CHACHA_CORRUPTION_ERROR_DETAIL
    assert "/private/db/path" not in exc_info.value.detail


@pytest.mark.asyncio
async def test_chacha_waiter_normalizes_shutdown_abort_detail(monkeypatch, tmp_path):
    user_dir = tmp_path / "user-789"
    init_event = threading.Event()
    init_event.set()

    monkeypatch.setattr(chacha_deps.DatabasePaths, "get_user_base_directory", lambda _user_id: user_dir)
    with chacha_deps._chacha_db_lock:
        chacha_deps._chacha_db_init_events[str(user_dir)] = init_event
        chacha_deps._chacha_db_init_errors[str(user_dir)] = chacha_deps._ChaChaInitializationAborted(
            "shutdown sentinel leaked /private/db/path"
        )

    with pytest.raises(HTTPException) as exc_info:
        await chacha_deps._get_or_init_db_instance(789, "client-789")

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == chacha_deps._CHACHA_SHUTDOWN_INIT_ERROR_DETAIL


def test_create_and_prepare_db_preflight_refuses_malformed_existing_file(monkeypatch, tmp_path):
    db_path = tmp_path / "user-987" / "ChaChaNotes.db"
    db_path.parent.mkdir(parents=True)
    db_path.write_bytes(b"not a sqlite database")

    class _UnexpectedCharactersRAGDB:
        def __init__(self, *args, **kwargs):
            raise AssertionError("CharactersRAGDB should not be constructed for a corrupt existing DB")

    monkeypatch.setattr(chacha_deps.DatabasePaths, "get_chacha_db_path", lambda _user_id: db_path)
    monkeypatch.setattr(chacha_deps, "CharactersRAGDB", _UnexpectedCharactersRAGDB)

    with pytest.raises(chacha_deps.ChaChaDatabaseCorruptionError) as exc_info:
        chacha_deps._create_and_prepare_db(987, "client-987")

    assert str(db_path) not in str(exc_info.value)
