"""The one definition of which roles are platform administrators.

This set was spelled out identically in four places -- AuthNZ/auth_principal_resolver.py,
AuthNZ/byok_helpers.py, Claims_Extraction/claims_service.py, and (briefly)
MCP_unified/protocol_types.py. Adding a platform admin role meant finding every copy,
and missing one produced a silent under-grant: TASK-13338 fixed exactly that in MCP,
where a principal holding the AuthNZ role "owner" was refused permanent media delete,
permanent note delete and every kanban policy operation while being an administrator
everywhere else in the product.

The module deliberately imports nothing, so it can be imported from anywhere --
including MCP_unified, which previously reached for the private name in
auth_principal_resolver and thereby pulled in FastAPI's Request for a frozenset.

PERMISSIONS. ``PLATFORM_ADMIN_PERMISSIONS`` is the companion set: permission claims that
make a principal a platform administrator. Note that "admin" appears in BOTH sets and
means two different things:

* the ``admin`` ROLE is how an interactive user is made an administrator;
* the ``admin`` PERMISSION is how a SERVICE ACCOUNT is. Service-account tokens carry
  permissions and no roles -- jwt_service.create_service_account_token has no roles
  parameter -- so a permission is the only way to grant one administrator status.

That second meaning used to exist in exactly one place. The permission set was written
out in twenty-two places; twenty-one of them omitted "admin", so a service account
granted it was an administrator at principal resolution and nowhere else -- not in
BYOK, Claims, billing, org, setup or any endpoint that re-derived the answer from
claims. Decided (TASK-13353): it is an administrator everywhere, so every copy now
aliases this one.

MCP_unified is deliberately narrower and does not use this set: per ADR-048 it accepts
only "*", because a configuration permission should not authorise destroying another
user's data.
"""

from __future__ import annotations

__all__ = ["PLATFORM_ADMIN_PERMISSIONS", "PLATFORM_ADMIN_ROLES"]

#: Roles that make a principal a platform administrator, lowercase.
PLATFORM_ADMIN_ROLES: frozenset[str] = frozenset({"admin", "owner", "super_admin"})

#: Permission claims that make a principal a platform administrator. "admin" here is
#: the service-account grant, not the role -- see the module docstring.
PLATFORM_ADMIN_PERMISSIONS: frozenset[str] = frozenset({"*", "system.configure", "admin"})
