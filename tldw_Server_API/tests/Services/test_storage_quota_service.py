from pathlib import Path

import pytest

from tldw_Server_API.app.services.storage_quota_service import StorageQuotaService


class DummySettings:
    def __init__(self, base_path: str):
        self.USER_DATA_BASE_PATH = base_path
        self.CHROMADB_BASE_PATH = ""  # disable chroma in tests


class _DummyTransCtx:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def execute(self, query, params):
        if str(query).lstrip().lower().startswith("with target_user as"):
            return _DummyCursor(
                [
                    (
                        "user",
                        int(params[0]),
                        "2026-07-26T12:00:00.000000Z",
                    )
                ]
            )
        return _DummyCursor()

    async def commit(self):
        return None


class _DummyCursor:
    rowcount = 1

    def __init__(self, rows=None):
        self._rows = rows or []

    async def fetchall(self):
        return self._rows


class FakePool:
    def __init__(self, quota_mb: int = 1000, used_mb: float = 0.0):
        self._quota_mb = quota_mb
        self._used_mb = used_mb
        self.fetchone_calls = 0

    def transaction(self):

        return _DummyTransCtx()

    async def fetchone(self, query: str, *args):
        # Return consistent shape expected by service
        self.fetchone_calls += 1
        return {"storage_used_mb": self._used_mb, "storage_quota_mb": self._quota_mb}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_calculate_user_storage_cache_hit_and_miss(tmp_path: Path):
    # Arrange: create user data directory with one file
    user_id = 42
    base = tmp_path / "user_databases"
    user_dir = base / str(user_id)
    (user_dir / "media").mkdir(parents=True, exist_ok=True)
    f1 = user_dir / "media" / "a.txt"
    data1 = b"x" * 1024  # 1 KiB
    f1.write_bytes(data1)

    svc = StorageQuotaService(db_pool=FakePool(), settings=DummySettings(str(base)))
    await svc.initialize()

    # Act: first calculation (miss -> scans filesystem)
    r1 = await svc.calculate_user_storage(user_id=user_id, update_database=False)
    assert r1["total_bytes"] >= len(data1)

    # Mutate filesystem: add another file
    f2 = user_dir / "media" / "b.txt"
    data2 = b"y" * 2048  # 2 KiB
    f2.write_bytes(data2)

    # Second calculation without updating DB should hit cache and ignore new file
    r2 = await svc.calculate_user_storage(user_id=user_id, update_database=False)
    assert r2["total_bytes"] == r1["total_bytes"], "expected cache hit to return same result"

    # Third calculation with update_database True should bypass cache and see new bytes
    r3 = await svc.calculate_user_storage(user_id=user_id, update_database=True)
    assert r3["total_bytes"] >= r1["total_bytes"] + len(data2)


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("blocking_level", ["user", "team", "org"])
@pytest.mark.parametrize("raise_on_exceed", [False, True], ids=["report", "raise"])
async def test_combined_quota_denial_reports_or_raises_storage_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    blocking_level: str, raise_on_exceed: bool,
) -> None:
    """Every denied scope returns its level or raises the real quota exception."""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from tldw_Server_API.app.core.AuthNZ.exceptions import QuotaExceededError

    monkeypatch.setenv("USAGE_QUOTAS_ENABLED", "1")
    svc = StorageQuotaService(
        db_pool=FakePool(quota_mb=100, used_mb=90.0 if blocking_level == "user" else 10.0),
        settings=DummySettings(str(tmp_path)),
    )

    async def can_allocate(
        new_bytes: int, *, team_id: int | None = None, org_id: int | None = None,
    ) -> tuple[bool, str]:
        """Supply the repository admission result for the selected scope."""
        level = "team" if team_id is not None else "org"
        return level != blocking_level, f"{level} allocation"

    async def check_quota_status(
        *, team_id: int | None = None, org_id: int | None = None,
    ) -> dict[str, int | float]:
        """Supply the repository usage snapshot without replacing service logic."""
        level = "team" if team_id is not None else "org"
        return {"quota_mb": 100, "used_mb": 90.0 if level == blocking_level else 10.0}

    repo = SimpleNamespace(can_allocate=can_allocate, check_quota_status=check_quota_status)
    monkeypatch.setattr(svc, "get_storage_quotas_repo", AsyncMock(return_value=repo))

    if raise_on_exceed:
        with pytest.raises(QuotaExceededError) as exc:
            await svc.check_combined_quota(
                42, 20 * 1024 * 1024, team_id=7, org_id=9, raise_on_exceed=True,
            )
        assert (exc.value.used_mb, exc.value.quota_mb) == (20.0, 100)
    else:
        allowed, info = await svc.check_combined_quota(
            42, 20 * 1024 * 1024, team_id=7, org_id=9,
        )
        assert (allowed, info["has_quota"], info["blocking_level"]) == (False, False, blocking_level)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_combined_quota_does_not_convert_transaction_failure_to_denial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unrelated database failure remains a transaction error, not quota denial."""
    from unittest.mock import AsyncMock

    from tldw_Server_API.app.core.AuthNZ.exceptions import TransactionError

    monkeypatch.setenv("USAGE_QUOTAS_ENABLED", "1")
    failure = TransactionError("read storage usage")
    pool = FakePool()
    monkeypatch.setattr(pool, "fetchone", AsyncMock(side_effect=failure))
    svc = StorageQuotaService(db_pool=pool, settings=DummySettings(str(tmp_path)))

    with pytest.raises(TransactionError) as exc:
        await svc.check_combined_quota(42, 20 * 1024 * 1024, raise_on_exceed=True)
    assert exc.value is failure
