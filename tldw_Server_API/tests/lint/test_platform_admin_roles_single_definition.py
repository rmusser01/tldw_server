"""Ratchet: the platform admin role set has exactly one definition.

``frozenset({"admin", "owner", "super_admin"})`` was written out in six places --
AuthNZ/auth_principal_resolver.py, AuthNZ/byok_helpers.py,
Claims_Extraction/claims_service.py, services/admin_budgets_service.py,
services/admin_profiles_service.py, and endpoints/admin/admin_ops.py (as
_INCIDENT_ASSIGNABLE_ROLES). Adding a platform admin role meant finding every copy.

Missing one is not hypothetical: TASK-13338 fixed exactly that failure in MCP, where a
principal holding the AuthNZ role "owner" was refused permanent media delete, permanent
note delete and every kanban policy operation while being an administrator everywhere
else in the product.

They now all alias AuthNZ/platform_admin.PLATFORM_ADMIN_ROLES. This test fails if a
seventh copy appears.

It does NOT police the companion permission set. Those three copies genuinely disagree
and always have -- see the platform_admin module docstring -- and reconciling them
widens or narrows real authorisation, so it is a policy decision rather than a
deduplication.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
APP_ROOT = REPO_ROOT / "tldw_Server_API" / "app"
CANONICAL = APP_ROOT / "core" / "AuthNZ" / "platform_admin.py"

_ROLES = {"admin", "owner", "super_admin"}


def _defines_role_set_literally(tree: ast.AST) -> bool:
    """True when the module builds the platform admin role set from literals."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
        if name != "frozenset" or not node.args:
            continue
        arg = node.args[0]
        elements = arg.elts if isinstance(arg, (ast.Set, ast.List, ast.Tuple)) else []
        values = {
            el.value
            for el in elements
            if isinstance(el, ast.Constant) and isinstance(el.value, str)
        }
        if values == _ROLES:
            return True
    return False


def test_platform_admin_roles_defined_exactly_once() -> None:
    offenders = []
    for path in sorted(APP_ROOT.rglob("*.py")):
        if path == CANONICAL:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:  # pragma: no cover - a broken file fails elsewhere
            continue
        if _defines_role_set_literally(tree):
            offenders.append(str(path.relative_to(REPO_ROOT)))

    assert not offenders, (  # nosec B101
        "These modules restate the platform admin role set instead of importing "
        "PLATFORM_ADMIN_ROLES from core/AuthNZ/platform_admin.py. A copy that is missed "
        "when a role is added silently under-grants that role (see TASK-13338): "
        f"{offenders}"
    )


def test_canonical_module_still_defines_it() -> None:
    """Guard against the ratchet passing because the definition moved or vanished."""
    tree = ast.parse(CANONICAL.read_text(encoding="utf-8"))
    assert _defines_role_set_literally(tree)  # nosec B101


def test_canonical_module_has_no_imports() -> None:
    """It must stay importable from anywhere, including MCP_unified.

    MCP previously reached for the private name in auth_principal_resolver, which pulled
    FastAPI's Request in for the sake of a frozenset.
    """
    tree = ast.parse(CANONICAL.read_text(encoding="utf-8"))
    imports = [
        node
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
        and not (isinstance(node, ast.ImportFrom) and node.module == "__future__")
    ]
    assert not imports, (  # nosec B101
        f"core/AuthNZ/platform_admin.py must import nothing; found {len(imports)}"
    )
