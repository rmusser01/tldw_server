"""The app-service container holds the ChaCha runtime, which owns the executor, cache and init state."""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.API_Deps import ChaCha_Notes_DB_Deps as deps
from tldw_Server_API.app.core.DB_Management.chacha.runtime import (
    ChaChaDatabaseCorruptionError,
    ChaChaInitAbortedError,
    ChaChaInitFailedError,
    ChaChaInitTimeoutError,
    ChaChaRuntimeManager,
    ChaChaRuntimeUnavailableError,
    get_chacha_runtime,
)
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.Utils.app_services import app_services

pytestmark = pytest.mark.unit


class _FakeDB:
    def __init__(self) -> None:
        self.closed = False

    def close_all_connections(self) -> None:
        self.closed = True


class _RaisingRuntime(ChaChaRuntimeManager):
    def __init__(self, error: Exception) -> None:
        super().__init__()
        self.error = error

    async def get_db(self, *_args, **_kwargs):
        raise self.error


def _failed_from(cause: Exception | None) -> ChaChaInitFailedError:
    error = ChaChaInitFailedError()
    error.__cause__ = cause
    return error


def test_get_chacha_runtime_is_the_container_slot(fresh_services) -> None:
    assert get_chacha_runtime() is fresh_services.chacha_runtime is get_chacha_runtime()


@pytest.mark.parametrize(
    ("raised", "status", "detail"),
    [
        (ChaChaRuntimeUnavailableError("ChaChaNotes shutdown in progress"), 503, "ChaChaNotes shutdown in progress"),
        (ChaChaInitAbortedError("shutdown sentinel leaked /private/db/path"), 503, "ChaChaNotes shutdown in progress"),
        (ChaChaInitTimeoutError(), 503, "ChaChaNotes initialization timed out"),
        (
            _failed_from(ChaChaDatabaseCorruptionError()),
            503,
            "ChaChaNotes DB corruption detected; repair or restore required",
        ),
        (_failed_from(OSError("/private/db/path")), 500, "Could not initialize character & notes database for user"),
        (
            _failed_from(None),
            500,
            "Could not initialize character & notes database for user: unknown error",
        ),
    ],
    ids=["unavailable", "aborted", "timeout", "corruption", "other-cause", "no-cause"],
)
async def test_runtime_errors_map_to_the_same_http_responses(
    fresh_services, monkeypatch, tmp_path, raised: Exception, status: int, detail: str
) -> None:
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path))
    monkeypatch.setattr(app_services(), "chacha_runtime", _RaisingRuntime(raised))

    with pytest.raises(HTTPException) as exc_info:
        await deps._get_or_init_db_instance(1, "1")

    assert exc_info.value.status_code == status
    assert exc_info.value.detail == detail


async def test_runtime_owns_the_executor_and_cache(fresh_services) -> None:
    runtime = get_chacha_runtime()
    fake_db = _FakeDB()

    db = await runtime.get_db("k", lambda: fake_db, user_id=1)

    assert db is fake_db
    assert runtime.cached_instances() == (fake_db,)
    await runtime.aclose()
    assert runtime.cached_instances() == ()
    assert fake_db.closed
    runtime.resume()
    assert runtime.executor().submit(lambda: 42).result(timeout=5) == 42


class _RecordingRuntime(ChaChaRuntimeManager):
    """Builds through the factory on every call and records each cache key."""

    def __init__(self) -> None:
        super().__init__()
        self.keys: list[str] = []

    async def get_db(self, key, factory, *, user_id):
        self.keys.append(key)
        return factory()


async def _no_default_character(*_args) -> None:
    return None


async def test_core_accessors_bind_the_tenant_client_id(fresh_services, monkeypatch, tmp_path) -> None:
    """client_id is the PostgreSQL tenant key: "voice_assistant" once pooled every user into one tenant."""
    from tldw_Server_API.app.core.Character_Chat import chacha_db_provider

    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path))
    runtime = _RecordingRuntime()
    monkeypatch.setattr(app_services(), "chacha_runtime", runtime)
    built: list[tuple[int, str]] = []

    def create(user_id: int, client_id: str) -> _FakeDB:
        built.append((user_id, client_id))
        return _FakeDB()

    monkeypatch.setattr(chacha_db_provider, "create_and_prepare_chacha_db", create)
    monkeypatch.setattr(chacha_db_provider, "_ensure_default_character_async", _no_default_character)

    db = await chacha_db_provider.get_chacha_db_for_user_id(7, client_id="voice_assistant")
    await chacha_db_provider.get_chacha_db_for_owner(7)
    await chacha_db_provider.warm_chacha_db_for_user(7, "voice_assistant")

    assert isinstance(db, _FakeDB)
    assert built == [(7, "7")] * 3
    assert runtime.keys == [str(DatabasePaths.get_user_base_directory(7))] * 3


async def test_two_users_never_share_a_cached_handle(fresh_services, monkeypatch, tmp_path) -> None:
    """The cache key is the user's own directory, so each user gets, and keeps, a separate handle."""
    from tldw_Server_API.app.core.Character_Chat import chacha_db_provider

    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path))
    runtime = get_chacha_runtime()
    monkeypatch.setattr(runtime, "_probe", lambda _db: True)
    monkeypatch.setattr(chacha_db_provider, "create_and_prepare_chacha_db", lambda _user_id, _client_id: _FakeDB())
    monkeypatch.setattr(chacha_db_provider, "_ensure_default_character_async", _no_default_character)

    seven = await chacha_db_provider.get_chacha_db_for_user_id(7)
    nine = await chacha_db_provider.get_chacha_db_for_user_id(9)

    assert seven is not nine
    assert await chacha_db_provider.get_chacha_db_for_user_id(7) is seven
    assert await chacha_db_provider.get_chacha_db_for_owner(9) is nine
    assert sorted(runtime._instances) == sorted(str(DatabasePaths.get_user_base_directory(u)) for u in (7, 9))


async def test_core_accessor_raises_core_errors_not_http(fresh_services, monkeypatch, tmp_path) -> None:
    from tldw_Server_API.app.core.Character_Chat import chacha_db_provider

    with pytest.raises(ValueError, match=r"^Invalid owner_user_id\.$"):
        await chacha_db_provider.get_chacha_db_for_user_id(0)
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path))
    monkeypatch.setattr(app_services(), "chacha_runtime", _RaisingRuntime(ChaChaInitTimeoutError()))
    with pytest.raises(ChaChaInitTimeoutError):
        await chacha_db_provider.get_chacha_db_for_user_id(1)


@pytest.mark.parametrize(
    "call",
    [
        lambda provider: provider.get_chacha_db_for_user_id(4242),
        lambda provider: provider.get_chacha_db_for_owner(-1),
        lambda provider: provider.get_chacha_db_for_owner(0),
    ],
    ids=["user-4242", "owner-minus-1", "owner-0"],
)
async def test_closed_runtime_refuses_before_resolving_the_user_directory(
    fresh_services, monkeypatch, tmp_path, call
) -> None:
    """Resolving the directory creates it, so a closed runtime refuses first (the 503 path)."""
    from tldw_Server_API.app.core.Character_Chat import chacha_db_provider

    runtime = ChaChaRuntimeManager()
    runtime.shutdown()
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path))
    monkeypatch.setattr(app_services(), "chacha_runtime", runtime)

    with pytest.raises(ChaChaRuntimeUnavailableError):
        await call(chacha_db_provider)

    assert list(tmp_path.iterdir()) == []
