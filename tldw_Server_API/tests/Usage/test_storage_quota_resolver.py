"""Per-user storage quota comes from the spec 2 resolver, not users.storage_quota_mb (spec 2 §5)."""

import pytest

from tldw_Server_API.app.core import config as config_module
from tldw_Server_API.app.core.Storage import generated_file_helpers
from tldw_Server_API.app.core.AuthNZ.exceptions import QuotaExceededError
from tldw_Server_API.app.core.Usage import quota_resolver
from tldw_Server_API.app.services import storage_quota_service as sqs

pytestmark = pytest.mark.unit
MB = 1024 * 1024


class _Pool:
    """Minimal pool: one users row (used 100 MB, legacy column 5120)."""

    def __init__(self) -> None:
        """Seed one user."""
        self.rows = {7: {"storage_used_mb": 100.0, "storage_quota_mb": 5120}}

    async def fetchone(self, _sql: str, user_id: int):
        """Return the users row."""
        return self.rows.get(int(user_id))


@pytest.fixture()
def service(monkeypatch: pytest.MonkeyPatch) -> sqs.StorageQuotaService:
    """A service whose resolver answers from a dict the test controls."""
    limits: dict[int, float | None] = {}

    async def _user_quota(user_id: int, key: str):
        """The test's limit for the user (storage key only)."""
        return limits.get(user_id) if key == sqs.STORAGE_QUOTA_KEY else None

    monkeypatch.setattr(quota_resolver, "user_quota", _user_quota)
    svc = sqs.StorageQuotaService(db_pool=_Pool())
    svc._initialized = True
    svc.limits = limits  # type: ignore[attr-defined]
    return svc


async def test_unlimited_ignores_stale_column(service) -> None:
    """No limits value: a 5 GB + 1 MB upload is admitted although the column says 5120."""
    ok, info = await service.check_quota(7, 5 * 1024 * MB + MB)
    assert ok is True
    assert info["quota_mb"] is None and info["available_mb"] is None and info["usage_percentage"] is None


async def test_resolved_value_denies_one_mb_over(service) -> None:
    """A limits value of 150 MB with 100 MB used refuses 51 MB and admits 50 MB."""
    service.limits[7] = 150
    assert (await service.check_quota(7, 51 * MB))[0] is False
    service.invalidate_user_cache(7)
    assert (await service.check_quota(7, 50 * MB))[0] is True


async def test_zero_blocks_any_upload(service) -> None:
    """limits.storage_quota_mb = 0 refuses even a 1-byte upload."""
    service.limits[7] = 0
    ok, info = await service.check_quota(7, 1)
    assert ok is False and info["quota_mb"] == 0


async def test_quota_change_applies_without_waiting_for_usage_cache(service) -> None:
    """The 300 s cache holds usage only, so a new limit applies on the next check."""
    assert (await service.check_quota(7, 10 * MB))[0] is True
    service.limits[7] = 105
    assert (await service.check_quota(7, 10 * MB))[0] is False


def test_quota_view_unlimited_and_limited() -> None:
    """quota_view yields None fields when unlimited and arithmetic when limited."""
    assert sqs.quota_view(10.0, None) == {"quota_mb": None, "available_mb": None, "usage_percentage": None}
    assert sqs.quota_view(25.0, 100) == {"quota_mb": 100, "available_mb": 75.0, "usage_percentage": 25.0}
    for used in (0.0, 5.0):
        assert sqs.quota_view(used, 0) == {"quota_mb": 0, "available_mb": 0.0, "usage_percentage": 100.0}


async def test_switch_off_admits_past_stale_column(monkeypatch: pytest.MonkeyPatch) -> None:
    """Quotas off (the OSS default): the real resolver returns None, so 5 GB + 1 MB is admitted despite the 5120 column."""
    monkeypatch.delenv("USAGE_QUOTAS_ENABLED", raising=False)
    monkeypatch.delenv("LIMIT_ENFORCEMENT_ENABLED", raising=False)
    monkeypatch.setattr(config_module, "load_comprehensive_config", lambda *a, **k: None)
    svc = sqs.StorageQuotaService(db_pool=_Pool())
    svc._initialized = True
    ok, info = await svc.check_quota(7, 5 * 1024 * MB + MB)
    assert ok is True and info["quota_mb"] is None


async def test_generated_file_preflight_admits_past_stale_column(monkeypatch: pytest.MonkeyPatch) -> None:
    """Quotas off: the generated-file admit path ignores the 5120 column and admits 5 GB + 1 MB."""
    monkeypatch.delenv("USAGE_QUOTAS_ENABLED", raising=False)
    monkeypatch.delenv("LIMIT_ENFORCEMENT_ENABLED", raising=False)
    monkeypatch.setattr(config_module, "load_comprehensive_config", lambda *a, **k: None)
    svc = sqs.StorageQuotaService(db_pool=_Pool())
    svc._initialized = True

    async def _get_service():
        """The real service over the stub pool."""
        return svc

    monkeypatch.setattr(generated_file_helpers, "get_storage_service", _get_service)
    returned = await generated_file_helpers._preflight_generated_file_write(
        user_id=7, file_size_bytes=5 * 1024 * MB + MB, org_id=None, team_id=None, check_quota=True
    )
    assert returned is svc


async def test_combined_pool_denial_raises_quota_exceeded(service, monkeypatch: pytest.MonkeyPatch) -> None:
    """A blocking team pool with raise_on_exceed=True raises QuotaExceededError (it used to raise TypeError)."""

    class _PoolRepo:
        """A team pool that is full."""

        async def can_allocate(self, new_bytes, team_id=None, org_id=None):
            """Refuse."""
            return False, "team pool full"

        async def check_quota_status(self, team_id=None, org_id=None):
            """Report the full pool."""
            return {"quota_mb": 1, "used_mb": 1.0}

    async def _repo():
        """The full pool repo."""
        return _PoolRepo()

    monkeypatch.setattr(service, "get_storage_quotas_repo", _repo)
    with pytest.raises(QuotaExceededError) as exc:
        await service.check_combined_quota(7, MB, team_id=3, raise_on_exceed=True)
    assert exc.value.quota_mb == 1 and exc.value.used_mb == 2.0


async def test_combined_user_denial_reports_user_projection(service) -> None:
    """When the user level blocks, the error carries the user's projected usage and quota."""
    service.limits[7] = 150
    with pytest.raises(QuotaExceededError) as exc:
        await service.check_combined_quota(7, 51 * MB, raise_on_exceed=True)
    assert exc.value.quota_mb == 150 and exc.value.used_mb == 151.0


async def test_user_quota_status_is_read_only_and_has_quota_means_set(service) -> None:
    """user_quota_status reports has_quota only when a value is set, and leaves the usage cache untouched."""
    assert await service.user_quota_status(7) == {
        "quota_mb": None, "used_mb": 100.0, "remaining_mb": None, "usage_pct": 0.0, "has_quota": False,
    }
    service.limits[7] = 400
    status = await service.user_quota_status(7)
    assert status["has_quota"] is True and status["remaining_mb"] == 300.0 and status["usage_pct"] == 25.0
    assert "quota:7" not in service.quota_cache


async def test_set_user_quota_writes_deletes_and_invalidates(service, monkeypatch: pytest.MonkeyPatch) -> None:
    """set_user_quota upserts the override, deletes it on None, rejects negatives, and clears the resolver entry."""
    calls: list[tuple] = []

    class _Repo:
        """Records override writes."""

        def __init__(self, _pool) -> None:
            """Ignore the pool."""

        async def ensure_tables(self) -> None:
            """Nothing to ensure."""

        async def upsert_override(self, **kw) -> None:
            """Record an upsert."""
            calls.append(("upsert", kw["user_id"], kw["key"], kw["value"]))

        async def delete_override(self, **kw) -> None:
            """Record a delete."""
            calls.append(("delete", kw["user_id"], kw["key"]))

    invalidated: list[int] = []
    monkeypatch.setattr(sqs, "UserProfileOverridesRepo", _Repo)
    monkeypatch.setattr(quota_resolver, "invalidate_user", invalidated.append)
    service.limits[7] = 0
    out = await service.set_user_quota(7, 0)
    assert calls == [("upsert", 7, sqs.STORAGE_QUOTA_KEY, 0)] and invalidated == [7]
    assert out["storage_quota_mb"] == 0 and out["storage_used_mb"] == 100.0
    service.limits.pop(7)
    out = await service.set_user_quota(7, None)
    assert calls[-1] == ("delete", 7, sqs.STORAGE_QUOTA_KEY) and out["storage_quota_mb"] is None
    with pytest.raises(ValueError):
        await service.set_user_quota(7, -1)
