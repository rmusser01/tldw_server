from types import SimpleNamespace

import pytest

from tldw_Server_API.app.services.enhanced_web_scraping_service import WebScrapingService

pytestmark = pytest.mark.unit


def test_is_admin_user_rejects_boolean_only_shape() -> None:
    svc = WebScrapingService()
    user = {"id": 1, "roles": ["user"], "permissions": [], "is_admin": True}
    assert svc._is_admin_user(user) is False


def test_is_admin_user_accepts_claim_permissions() -> None:
    svc = WebScrapingService()
    user = {"id": 1, "roles": ["user"], "permissions": ["system.configure"]}
    assert svc._is_admin_user(user) is True


def test_can_access_job_allows_admin_claims_for_cross_user() -> None:
    svc = WebScrapingService()
    user = SimpleNamespace(id=1, roles=["user"], permissions=["*"], role="user")
    job = SimpleNamespace(user_id=22)
    assert svc._can_access_job(job, user) is True


@pytest.mark.parametrize("role", ["admin", "owner", "super_admin"])
def test_canonical_platform_roles_allow_cross_user_access(role) -> None:
    svc = WebScrapingService()
    assert svc._can_access_job(SimpleNamespace(user_id=22), SimpleNamespace(id=1, roles=[role])) is True
