"""Media, storage, workflows and chatbooks quotas admit everything when usage quotas are off (spec 2)."""

from types import SimpleNamespace

import pytest

from tldw_Server_API.app.api.v1.API_Deps import storage_quota_guard
from tldw_Server_API.app.api.v1.endpoints import workflows as workflows_ep
from tldw_Server_API.app.core.Chatbooks.quota_manager import QuotaManager
from tldw_Server_API.app.core.Usage import quota_resolver
from tldw_Server_API.app.services.storage_quota_service import StorageQuotaService

pytestmark = pytest.mark.unit


@pytest.fixture()
def quotas_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """A stock install: neither switch spelling is set."""
    monkeypatch.delenv("USAGE_QUOTAS_ENABLED", raising=False)
    monkeypatch.delenv("LIMIT_ENFORCEMENT_ENABLED", raising=False)
    monkeypatch.setattr("tldw_Server_API.app.core.config.load_comprehensive_config", lambda: None)


@pytest.fixture()
def quotas_on(monkeypatch: pytest.MonkeyPatch) -> None:
    """An operator who turned usage quotas on."""
    monkeypatch.setenv("USAGE_QUOTAS_ENABLED", "1")


def _full_storage_service(monkeypatch: pytest.MonkeyPatch) -> StorageQuotaService:
    """A storage service whose user is exactly at the old 5 GB default."""
    service = StorageQuotaService(db_pool=object(), settings=SimpleNamespace())
    service._initialized = True

    async def _info(_user_id: int) -> dict:
        """A storage-info stand-in reporting usage exactly at the 5 GB quota."""
        return {"storage_used_mb": 5120.0, "storage_quota_mb": 5120}

    monkeypatch.setattr(service, "_get_user_storage_info", _info)
    return service


async def test_storage_never_raises_when_quotas_off(quotas_off: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """With quotas off, a user already at the storage cap still passes both the plain and combined quota checks even with raise_on_exceed."""
    service = _full_storage_service(monkeypatch)
    has_quota, info = await service.check_quota(1, 1024 * 1024, raise_on_exceed=True)
    assert has_quota is True and info["has_quota"] is True
    has_combined, combined = await service.check_combined_quota(1, 1024 * 1024, raise_on_exceed=True)
    assert has_combined is True and combined["blocking_level"] is None


async def test_storage_still_enforced_when_quotas_on(quotas_on: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """With quotas on, a user already at the storage cap fails the quota check.

    The quota now comes from the spec 2 resolver, not the row's legacy
    storage_quota_mb column, so the resolver is stubbed with the same 5 GB value.
    """
    service = _full_storage_service(monkeypatch)

    async def _user_quota(_user_id: int, _key: str):
        """The user's limit, matching the old 5 GB default the row used to carry."""
        return 5120

    monkeypatch.setattr(quota_resolver, "user_quota", _user_quota)
    has_quota, _info = await service.check_quota(1, 1024 * 1024)
    assert has_quota is False


def test_storage_pool_guard_disabled_when_quotas_off(quotas_off: None) -> None:
    """The storage quota pool guard reports itself disabled when usage quotas are off."""
    assert storage_quota_guard._is_enabled() is False


async def test_workflows_cap_skipped_when_quotas_off(quotas_off: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """With quotas off, the workflows daily cap check never inspects the request or db it was
    given, never refuses, and still records the run (gate the check, never the record; round 2).

    Raising on attribute access doesn't work here: every lookup inside the
    cap logic is wrapped in a try/except over `_WORKFLOWS_NONCRITICAL_EXCEPTIONS`,
    which includes `AssertionError`, so a raise is swallowed. Recording the
    accessed names instead lets the test tell "never inspected" apart from
    "inspected, then the exception was swallowed".
    """
    from tldw_Server_API.app.core.Workflows import daily_ledger

    accessed: list[str] = []

    class _RecordingRequest:
        """Any attribute access is recorded; it never means the cap logic stopped."""

        def __getattr__(self, name: str) -> object:
            """Record the accessed attribute name and return None."""
            accessed.append(name)
            return None

    recorded: list[dict] = []

    async def _fake_consume(
        *, entity_scope: str, entity_value: str, run_id: str, daily_cap: int | None
    ) -> tuple[bool, int]:
        """Record the call; quotas-off must still reach this, with daily_cap=None."""
        recorded.append(
            {"entity_scope": entity_scope, "entity_value": entity_value, "run_id": run_id, "daily_cap": daily_cap}
        )
        return True, 0

    monkeypatch.setattr(daily_ledger, "consume_workflow_run_if_within_cap", _fake_consume)

    await workflows_ep._enforce_workflows_daily_cap(
        request=_RecordingRequest(), current_user=SimpleNamespace(id=1), db=None, run_id="run-quotas-off"
    )
    assert accessed == []
    assert recorded == [
        {"entity_scope": "user", "entity_value": "1", "run_id": "run-quotas-off", "daily_cap": None}
    ]


def test_chatbooks_quotas_follow_the_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    """A QuotaManager's disabled flag tracks USAGE_QUOTAS_ENABLED as it's flipped off and on."""
    for name in ("CHATBOOKS_DISABLE_QUOTAS", "TEST_MODE", "TESTING", "PYTEST_CURRENT_TEST", "LIMIT_ENFORCEMENT_ENABLED"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("USAGE_QUOTAS_ENABLED", "0")
    assert QuotaManager(1, "free")._quotas_disabled is True
    monkeypatch.setenv("USAGE_QUOTAS_ENABLED", "1")
    assert QuotaManager(1, "free")._quotas_disabled is False
