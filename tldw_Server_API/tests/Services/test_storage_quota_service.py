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
