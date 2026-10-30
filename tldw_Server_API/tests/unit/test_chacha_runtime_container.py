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
