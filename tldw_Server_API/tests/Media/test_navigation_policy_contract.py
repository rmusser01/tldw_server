from __future__ import annotations

from tldw_Server_API.app.api.v1.utils.media_navigation_policy import (
    MEDIA_NAVIGATION_RATE_LIMIT_RESOURCE,
    MEDIA_NAVIGATION_ROUTE_POLICY,
)


def test_media_navigation_rate_limit_resource_is_stable() -> None:
    assert MEDIA_NAVIGATION_RATE_LIMIT_RESOURCE == "media.navigation"


def test_media_navigation_policy_contract_values() -> None:
    assert MEDIA_NAVIGATION_ROUTE_POLICY.auth_dependency_name == "get_request_user"
    assert MEDIA_NAVIGATION_ROUTE_POLICY.db_dependency_name == "get_media_db_for_user"
    assert MEDIA_NAVIGATION_ROUTE_POLICY.rate_limit_factory_name == "rbac_rate_limit"
    assert MEDIA_NAVIGATION_ROUTE_POLICY.rate_limit_resource == MEDIA_NAVIGATION_RATE_LIMIT_RESOURCE
