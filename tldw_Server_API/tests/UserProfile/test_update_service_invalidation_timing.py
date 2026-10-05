"""invalidate_user must run after the write (anchor.finalize), not interleaved with it (spec 2 review A8)."""

import pytest

from tldw_Server_API.app.core.UserProfiles import update_service as update_service_module

pytestmark = pytest.mark.unit


class _FakeGateway:
    """Records its final_touch call; capture_floor is a no-op."""

    def __init__(self, calls: list[str]) -> None:
        self._calls = calls

    async def capture_floor(self, _db_conn: object, *, user_id: int) -> str:
        """A fixed, non-None floor so anchor.finalize() will call final_touch."""
        return "floor"

    async def final_touch(self, _db_conn: object, *, user_id: int, version_floor: object) -> None:
        """Record the version-bump write anchor.finalize() performs."""
        self._calls.append("final_touch")


class _FakeOverridesRepo:
    """Records writes to the limits.* overrides table; no real DB involved."""

    def __init__(self, calls: list[str]) -> None:
        self._calls = calls

    async def ensure_tables(self, *, db_conn: object) -> None:
        """No-op: the fake has no schema to create."""

    async def upsert_override(self, *, user_id: int, key: str, value: object, updated_by: object, db_conn: object) -> None:
        """Record the override write."""
        self._calls.append("upsert_override")

    async def delete_override(self, *, user_id: int, key: str, db_conn: object) -> None:
        """Record the override delete."""
        self._calls.append("delete_override")


@pytest.fixture()
def calls(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Patch the gateway, repo, and invalidate_user to append to one ordered list."""
    recorded: list[str] = []

    def _recording_invalidate(_user_id: int) -> None:
        """Record the cache invalidation."""
        recorded.append("invalidate_user")

    monkeypatch.setattr(update_service_module, "invalidate_user", _recording_invalidate)
    monkeypatch.setattr(update_service_module, "VersionedUserWriteGateway", lambda _backend, *, clock: _FakeGateway(recorded))
    monkeypatch.setattr(update_service_module, "UserProfileOverridesRepo", lambda _pool: _FakeOverridesRepo(recorded))
    return recorded


async def test_invalidate_runs_after_final_touch_for_a_limits_write(calls: list[str]) -> None:
    """Writing a limits.* value: upsert, then final_touch (the version bump), then invalidate."""
    service = update_service_module.UserProfileUpdateService(db_pool=object())
    await service.apply_updates(
        user_id=7,
        updates=[("limits.rag_queries_per_day", 5)],
        roles={"platform_admin"},
        dry_run=False,
        db_conn=object(),
        updated_by=1,
    )
    assert calls == ["upsert_override", "final_touch", "invalidate_user"]


async def test_invalidate_runs_after_final_touch_for_a_limits_null_delete(calls: list[str]) -> None:
    """Null-deleting a limits.* value: delete, then final_touch, then invalidate."""
    service = update_service_module.UserProfileUpdateService(db_pool=object())
    await service.apply_updates(
        user_id=7,
        updates=[("limits.rag_queries_per_day", None)],
        roles={"platform_admin"},
        dry_run=False,
        db_conn=object(),
        updated_by=1,
    )
    assert calls == ["delete_override", "final_touch", "invalidate_user"]
