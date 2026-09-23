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

NOT INCLUDED HERE, on purpose: the companion permission set. The three copies of
``_ADMIN_CLAIM_PERMISSIONS`` do NOT agree, and never have -- all three were introduced
in the same commit (d0654d0cfb) with one already different:

    auth_principal_resolver.py   {"*", "system.configure", "admin"}
    byok_helpers.py              {"*", "system.configure"}
    claims_service.py            {"*", "system.configure"}

So the ``admin`` PERMISSION grants administrator status during principal resolution but
not in BYOK or Claims. Unifying those would widen or narrow real authorisation, which is
a policy decision rather than a deduplication, and it is tracked separately. MCP
diverges further still and deliberately: per ADR-048 it accepts only ``*``, because a
configuration permission should not authorise destroying another user's data.
"""

from __future__ import annotations

__all__ = ["PLATFORM_ADMIN_ROLES"]

#: Roles that make a principal a platform administrator, lowercase.
PLATFORM_ADMIN_ROLES: frozenset[str] = frozenset({"admin", "owner", "super_admin"})
