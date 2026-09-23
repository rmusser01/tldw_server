"""One admin predicate, one claim matrix.

Six _is_admin definitions had grown independently across MCP -- media, notes, kanban,
sandbox, mcp_discovery and protocol_types -- with six different claim sets, gating a
cross-user resource check and two irreversible deletes. They disagreed in both
directions:

* Under-grant: AuthNZ treats "owner" and "super_admin" as platform administrators and
  server.py writes them straight into metadata["roles"], but every MCP predicate tested
  the literal "admin". A platform owner was refused permanent media delete, permanent
  note delete and every kanban policy operation.
* Over-grant: sandbox_module alone accepted the "system.configure" permission and used
  it to pass its CROSS-USER session gate, while protocol_types -- MCP's own
  trusted-claims predicate -- said that same caller was not an admin.

Roles now come from AuthNZ's _PLATFORM_ADMIN_ROLES so the two cannot drift again.
Permissions deliberately stay narrower than AuthNZ's _ADMIN_CLAIM_PERMISSIONS: only
"*" grants admin, not "system.configure" or the "admin" permission. See
Docs/ADR/048-mcp-admin-claims.md.
"""

from __future__ import annotations

import os
from typing import Any

import pytest

os.environ.setdefault("TEST_MODE", "true")
os.environ.setdefault("ENABLE_TRACING", "false")

from tldw_Server_API.app.core.AuthNZ.auth_principal_resolver import _PLATFORM_ADMIN_ROLES
from tldw_Server_API.app.core.MCP_unified.modules.base import BaseModule, ModuleConfig
from tldw_Server_API.app.core.MCP_unified.protocol_types import metadata_has_admin_claims


class _Probe(BaseModule):
    async def on_initialize(self) -> None:
        return None

    async def on_shutdown(self) -> None:
        return None

    async def check_health(self) -> dict[str, bool]:
        return {"ok": True}

    async def get_tools(self) -> list[dict[str, Any]]:
        return []

    async def execute_tool(self, tool_name: str, arguments: dict[str, Any], context: Any = None) -> Any:
        return None


class _Ctx:
    def __init__(self, metadata: Any) -> None:
        self.metadata = metadata
        self.user_id = "7"


@pytest.fixture
def is_admin():
    module = _Probe(ModuleConfig(name="probe"))
    return lambda metadata: module.caller_is_admin(_Ctx(metadata))


# metadata, expected, why
CLAIM_MATRIX: list[tuple[dict[str, Any], bool, str]] = [
    # --- platform admin roles: all three count, and used not to ---
    ({"roles": ["admin"]}, True, "the only role the old predicates accepted"),
    ({"roles": ["owner"]}, True, "AuthNZ platform admin; was refused by all six"),
    ({"roles": ["super_admin"]}, True, "AuthNZ platform admin; was refused by all six"),
    ({"roles": ["OWNER"]}, True, "role matching is case-insensitive"),
    ({"roles": ["  owner  "]}, True, "role matching strips whitespace"),
    ({"roles": "owner"}, True, "a bare string claim is one role, not six characters"),
    ({"roles": ("owner",)}, True, "tuples are accepted like lists"),
    ({"roles": ["user", "owner"]}, True, "any platform admin role in the set is enough"),
    # --- non-admin roles ---
    ({"roles": ["user"]}, False, "an ordinary role is not admin"),
    ({"roles": ["administrator"]}, False, "no prefix or fuzzy role matching"),
    ({"roles": []}, False, "empty roles"),
    ({"roles": None}, False, "null roles"),
    # --- permissions: only the wildcard ---
    ({"permissions": ["*"]}, True, "wildcard permission means everything"),
    ({"permissions": "*"}, True, "bare string wildcard"),
    (
        {"permissions": ["system.configure"]},
        False,
        "DELIBERATE divergence from AuthNZ: a configuration permission must not "
        "authorise permanent deletes of another user's media and notes, nor pass "
        "sandbox's cross-user session gate",
    ),
    (
        {"permissions": ["admin"]},
        False,
        "the 'admin' PERMISSION is not the 'admin' ROLE; AuthNZ accepts it, MCP does not",
    ),
    ({"permissions": ["media.read"]}, False, "an ordinary permission is not admin"),
    # --- shapes and junk ---
    ({}, False, "no claims at all"),
    ({"roles": {"nested": "dict"}}, False, "a dict is not a claim list"),
    ({"roles": [None, ""]}, False, "empty and null entries are skipped, not matched"),
    ({"roles": [123]}, False, "non-string entries do not match"),
]


@pytest.mark.parametrize(
    ("metadata", "expected", "why"),
    CLAIM_MATRIX,
    ids=[f"{i}:{why[:48]}" for i, (_, _, why) in enumerate(CLAIM_MATRIX)],
)
def test_claim_matrix(is_admin, metadata: dict[str, Any], expected: bool, why: str) -> None:
    assert is_admin(metadata) is expected, why


def test_non_dict_metadata_is_not_admin(is_admin) -> None:
    for metadata in (None, "roles", 42, ["admin"]):
        assert is_admin(metadata) is False


def test_missing_metadata_attribute_is_not_admin() -> None:
    module = _Probe(ModuleConfig(name="probe"))

    class _Bare:
        pass

    assert module.caller_is_admin(_Bare()) is False
    assert module.caller_is_admin(None) is False


def test_dead_is_admin_attribute_is_not_consulted() -> None:
    """kanban and sandbox probed context.is_admin; RequestContext has no such field.

    server.py drops principal.is_admin when building the context, so the probe was
    always False. Nothing may resurrect it as a backdoor claim.
    """
    module = _Probe(ModuleConfig(name="probe"))

    class _Claiming:
        metadata = {"roles": ["user"]}
        is_admin = True

    assert module.caller_is_admin(_Claiming()) is False


def test_roles_are_taken_from_authnz() -> None:
    """The role set is imported, not restated, so MCP cannot drift from AuthNZ."""
    module = _Probe(ModuleConfig(name="probe"))
    for role in _PLATFORM_ADMIN_ROLES:
        assert module.caller_is_admin(_Ctx({"roles": [role]})) is True


def test_every_module_shares_the_one_predicate() -> None:
    """No module may re-declare its own admin check; that is how six of them appeared."""
    from tldw_Server_API.app.core.MCP_unified.modules.implementations import (
        kanban_module,
        media_module,
        notes_module,
        sandbox_module,
    )

    for mod, cls_name in (
        (media_module, "MediaModule"),
        (notes_module, "NotesModule"),
        (kanban_module, "KanbanModule"),
        (sandbox_module, "SandboxModule"),
    ):
        cls = getattr(mod, cls_name)
        assert "_is_admin" not in vars(cls), f"{cls_name} re-declares _is_admin"
        assert cls.caller_is_admin is BaseModule.caller_is_admin


def test_base_helper_and_canonical_predicate_agree(is_admin) -> None:
    for metadata, expected, why in CLAIM_MATRIX:
        assert metadata_has_admin_claims(metadata) is expected, why
        assert is_admin(metadata) is expected, why


@pytest.mark.asyncio
@pytest.mark.parametrize("roles", ["admin", "owner", "user", ["admin"], ["user"]])
async def test_discovery_claims_half_matches_every_other_module(roles: Any) -> None:
    """mcp_discovery keeps its own async _is_admin for the user_roles fallback.

    With no stored role its answer must be exactly the shared predicate's, including
    for a bare-string roles claim -- which used to split sandbox/discovery (grant)
    from media/notes/kanban (deny) for the same principal.
    """
    from tldw_Server_API.app.core.MCP_unified.modules.implementations.mcp_discovery_module import (
        MCPDiscoveryModule,
    )

    class _NoStoredRole:
        async def fetchone(self, *_args: Any) -> None:
            return None

    ctx = _Ctx({"roles": roles})
    discovery = MCPDiscoveryModule(ModuleConfig(name="discovery"))
    shared = _Probe(ModuleConfig(name="probe")).caller_is_admin(ctx)
    assert await discovery._is_admin(ctx, _NoStoredRole()) is shared
