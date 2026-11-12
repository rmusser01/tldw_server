"""A service account granted the "admin" PERMISSION is an administrator everywhere.

Service-account tokens carry permissions and no roles --
jwt_service.create_service_account_token has no roles parameter -- so the "admin"
permission is the only way to make one an administrator. It is distinct from the
"admin" ROLE, which is how interactive users are made administrators.

That grant used to be honoured only at principal resolution. The admin permission set
was written out in twenty-five places and twenty-four omitted "admin", so a service
account made administrator was refused by BYOK, Claims and every endpoint that
re-derived the answer from claims. Decided under TASK-13353: it is an administrator
everywhere. is_admin is left False on these principals deliberately, so each site is
tested on its own derivation rather than on the resolver's answer.
"""

from __future__ import annotations

import pytest

from tldw_Server_API.app.core.AuthNZ import byok_helpers
from tldw_Server_API.app.core.AuthNZ.platform_admin import (
    PLATFORM_ADMIN_PERMISSIONS,
    PLATFORM_ADMIN_ROLES,
)
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.Claims_Extraction import claims_service


def _service(*permissions: str) -> AuthPrincipal:
    return AuthPrincipal(
        kind="service", subject="service:worker", permissions=list(permissions), roles=[]
    )


_SITES = [
    pytest.param(byok_helpers._principal_has_platform_admin_claims, id="byok"),
    pytest.param(claims_service._principal_has_platform_admin_claims, id="claims"),
]


@pytest.mark.parametrize("is_admin_at", _SITES)
def test_service_account_admin_grant_is_honoured(is_admin_at) -> None:
    assert is_admin_at(_service("tools.execute:foo", "admin")) is True


@pytest.mark.parametrize("is_admin_at", _SITES)
def test_ordinary_service_permissions_do_not_confer_admin(is_admin_at) -> None:
    assert is_admin_at(_service("tools.execute:foo", "media.read")) is False


@pytest.mark.parametrize("permission", ["*", "system.configure", "admin"])
@pytest.mark.parametrize("is_admin_at", _SITES)
def test_every_admin_permission_is_honoured(is_admin_at, permission: str) -> None:
    assert is_admin_at(_service(permission)) is True


def test_admin_is_both_a_role_and_a_permission_and_they_are_different_sets() -> None:
    """The word is shared; the sets are not. "owner" is a role only, "*" a permission only."""
    assert "admin" in PLATFORM_ADMIN_ROLES and "admin" in PLATFORM_ADMIN_PERMISSIONS
    assert "owner" not in PLATFORM_ADMIN_PERMISSIONS
    assert "*" not in PLATFORM_ADMIN_ROLES


def test_the_resolver_and_the_consumers_agree() -> None:
    from tldw_Server_API.app.core.AuthNZ import auth_principal_resolver

    assert auth_principal_resolver._claims_mark_admin(roles=[], permissions=["admin"]) is True
    assert byok_helpers._principal_has_platform_admin_claims(_service("admin")) is True
    assert claims_service._principal_has_platform_admin_claims(_service("admin")) is True
