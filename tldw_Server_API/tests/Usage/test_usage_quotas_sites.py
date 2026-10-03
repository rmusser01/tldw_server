"""Media, storage, workflows and chatbooks quotas admit everything when usage quotas are off (spec 2)."""

from types import SimpleNamespace

import pytest

from tldw_Server_API.app.api.v1.API_Deps import storage_quota_guard
from tldw_Server_API.app.api.v1.endpoints import workflows as workflows_ep
from tldw_Server_API.app.core.Chatbooks.quota_manager import QuotaManager
from tldw_Server_API.app.core.Ingestion_Media_Processing import persistence
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


def _media_request() -> SimpleNamespace:
    """A request whose app carries a governor and a policy with media caps."""
    loader = SimpleNamespace(get_policy=lambda _pid: {"jobs": {"max_concurrent": 2}})
    app = SimpleNamespace(state=SimpleNamespace(rg_governor=object(), rg_policy_loader=loader))
    return SimpleNamespace(app=app, state=SimpleNamespace())


def test_media_budget_context_is_empty_when_quotas_off(quotas_off: None) -> None:
    gov, _policy_id, policy, entity = persistence._resolve_media_budget_context(
        request=_media_request(), current_user=SimpleNamespace(id=1)
    )
    assert gov is None and policy == {} and entity == ""


def test_media_budget_context_unchanged_when_quotas_on(quotas_on: None) -> None:
    gov, _policy_id, policy, entity = persistence._resolve_media_budget_context(
        request=_media_request(), current_user=SimpleNamespace(id=1)
    )
    assert gov is not None and policy["jobs"]["max_concurrent"] == 2 and entity == "user:1"


def _full_storage_service(monkeypatch: pytest.MonkeyPatch) -> StorageQuotaService:
    """A storage service whose user is exactly at the old 5 GB default."""
    service = StorageQuotaService(db_pool=object(), settings=SimpleNamespace())
    service._initialized = True

    async def _info(_user_id: int) -> dict:
        return {"storage_used_mb": 5120.0, "storage_quota_mb": 5120}

    monkeypatch.setattr(service, "_get_user_storage_info", _info)
    return service


async def test_storage_never_raises_when_quotas_off(quotas_off: None, monkeypatch: pytest.MonkeyPatch) -> None:
    service = _full_storage_service(monkeypatch)
    has_quota, info = await service.check_quota(1, 1024 * 1024, raise_on_exceed=True)
    assert has_quota is True and info["has_quota"] is True
    has_combined, combined = await service.check_combined_quota(1, 1024 * 1024, raise_on_exceed=True)
    assert has_combined is True and combined["blocking_level"] is None


async def test_storage_still_enforced_when_quotas_on(quotas_on: None, monkeypatch: pytest.MonkeyPatch) -> None:
    service = _full_storage_service(monkeypatch)
    has_quota, _info = await service.check_quota(1, 1024 * 1024)
    assert has_quota is False


def test_storage_pool_guard_disabled_when_quotas_off(quotas_off: None) -> None:
    assert storage_quota_guard._is_enabled() is False


async def test_workflows_cap_skipped_when_quotas_off(quotas_off: None) -> None:
    class _ExplodingRequest:
        """Any attribute access means the cap logic ran."""

        def __getattr__(self, name: str) -> object:
            raise AssertionError(f"the workflows cap must not inspect the request ({name})")

    await workflows_ep._enforce_workflows_daily_cap(
        request=_ExplodingRequest(), current_user=SimpleNamespace(id=1), db=None
    )


def test_chatbooks_quotas_follow_the_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("CHATBOOKS_DISABLE_QUOTAS", "TEST_MODE", "TESTING", "PYTEST_CURRENT_TEST", "LIMIT_ENFORCEMENT_ENABLED"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("USAGE_QUOTAS_ENABLED", "0")
    assert QuotaManager(1, "free")._quotas_disabled is True
    monkeypatch.setenv("USAGE_QUOTAS_ENABLED", "1")
    assert QuotaManager(1, "free")._quotas_disabled is False
