from __future__ import annotations

import ast
import re
import sqlite3
import textwrap
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import pytest

PROFILE_VISIBLE_COLUMNS = frozenset(
    {
        "uuid",
        "username",
        "email",
        "role",
        "is_superuser",
        "is_active",
        "is_verified",
        "two_factor_enabled",
        "last_login",
        "storage_quota_mb",
        "storage_used_mb",
    }
)
MEMBERSHIP_TABLES = frozenset({"org_members", "team_members"})
PARENT_SCOPE_TABLES = frozenset({"organizations", "teams"})
DIRECT_MEMBERSHIP_CALL_NAMES = frozenset(
    {
        "add_org_member",
        "remove_org_member",
        "update_org_member_role",
        "add_team_member",
        "remove_team_member",
        "update_team_member_role",
    }
)
DIRECT_MEMBERSHIP_PROXY_CALLS = frozenset(
    {
        (
            "tldw_Server_API/app/api/v1/endpoints/admin/admin_orgs.py",
            "admin_orgs_service",
            call_name,
        )
        for call_name in DIRECT_MEMBERSHIP_CALL_NAMES
    }
    | {
        (
            "tldw_Server_API/tests/AuthNZ/unit/test_orgs_endpoint_sanitization.py",
            "orgs",
            "add_org_member",
        )
    }
)
SERVING_MEMBERSHIP_CONTEXT_CATEGORIES = {
    "tldw_Server_API/app/api/v1/endpoints/auth.py": "trusted",
    "tldw_Server_API/app/api/v1/endpoints/admin/admin_tenant_provisioning.py": "actor",
    "tldw_Server_API/app/api/v1/endpoints/orgs.py": "actor",
    "tldw_Server_API/app/services/admin_e2e_support_service.py": "trusted",
    "tldw_Server_API/app/services/admin_orgs_service.py": "actor",
    "tldw_Server_API/app/services/registration_service.py": "trusted",
    "tldw_Server_API/app/services/org_invite_service.py": "trusted",
    "tldw_Server_API/app/core/AuthNZ/federation/provisioning_service.py": "trusted",
    "tldw_Server_API/app/core/AuthNZ/orgs_teams.py": frozenset(
        {"passthrough", "trusted"}
    ),
}
EXPECTED_TRUSTED_MEMBERSHIP_REASONS = {
    "tldw_Server_API/app/api/v1/endpoints/auth.py": frozenset({"BOOTSTRAP"}),
    "tldw_Server_API/app/core/AuthNZ/orgs_teams.py": frozenset({"BOOTSTRAP"}),
    "tldw_Server_API/app/core/AuthNZ/federation/provisioning_service.py": frozenset(
        {"BOOTSTRAP"}
    ),
    "tldw_Server_API/app/services/admin_e2e_support_service.py": frozenset(
        {"BOOTSTRAP"}
    ),
    "tldw_Server_API/app/services/org_invite_service.py": frozenset(
        {"REGISTRATION"}
    ),
    "tldw_Server_API/app/services/registration_service.py": frozenset(
        {"REGISTRATION"}
    ),
}
ACTOR_MEMBERSHIP_CONTEXT_FACTORIES = frozenset(
    {"_membership_context", "_membership_write_context"}
)
REPO_ROOT = Path(__file__).resolve().parents[3]
APP_ROOT = REPO_ROOT / "tldw_Server_API" / "app"
SQL_CALL_NAMES = frozenset(
    {
        "execute",
        "execute_many",
        "executemany",
        "executescript",
        "fetch",
        "fetchrow",
        "fetchval",
        "execute_query",
        "_execute_compat",
        "_execute_membership_scope_sql",
        "_mint_membership_scope_sql",
    }
)
PRIVILEGED_MEMBERSHIP_SCOPE_SQL_ENTRYPOINTS = frozenset(
    {
        "_execute_membership_scope_sql",
        "_execute_postgres_membership_timestamp_repair",
        "_mint_membership_scope_sql",
    }
)
_DYNAMIC_PRIVILEGED_MEMBERSHIP_SCOPE_SQL = "<dynamic_membership_scope_sql>"
_WRAPPED_PRIVILEGED_MEMBERSHIP_SCOPE_SQL = "<wrapped_membership_scope_sql>"
OFFLINE_MIGRATION_PATHS = frozenset(
    {
        "tldw_Server_API/app/core/AuthNZ/migrations.py",
        "tldw_Server_API/app/core/AuthNZ/pg_migrations_extra.py",
        "tldw_Server_API/app/core/AuthNZ/migrate_to_multiuser.py",
    }
)
_WRITE_RE = re.compile(
    r"\b(?P<verb>INSERT(?:\s+OR\s+\w+)?\s+INTO|UPDATE|DELETE\s+FROM)\s+"
    r"(?P<table>(?:[A-Za-z_]\w*\.)?[\"`\[]?[A-Za-z_]\w*[\"`\]]?)",
    re.IGNORECASE,
)
_UPDATE_COLUMNS_RE = re.compile(
    r"\bSET\s+(?P<columns>.*?)(?:\bWHERE\b|\bRETURNING\b|$)",
    re.IGNORECASE | re.DOTALL,
)
_INSERT_COLUMNS_RE = re.compile(
    r"^[^(]*\((?P<columns>.*?)\)\s*VALUES\b",
    re.IGNORECASE | re.DOTALL,
)
_POSTGRES_ROUTINE_DECLARATION_RE = re.compile(
    r"\bCREATE\s+(?:OR\s+REPLACE\s+)?"
    r"(?P<kind>FUNCTION|PROCEDURE|TRIGGER)\s+"
    r"(?:IF\s+NOT\s+EXISTS\s+)?"
    r"(?P<name>(?:[A-Za-z_]\w*|\"[^\"]+\")"
    r"(?:\s*\.\s*(?:[A-Za-z_]\w*|\"[^\"]+\"))*)",
    re.IGNORECASE,
)


@dataclass(frozen=True, order=True)
class ExpectedWrite:
    path: str
    function: str
    operation: str


@dataclass(frozen=True, order=True)
class ObservedWrite:
    path: str
    function: str
    operation: str
    line: int

    @property
    def expected(self) -> ExpectedWrite:
        return ExpectedWrite(self.path, self.function, self.operation)

    def diagnostic(self) -> str:
        return f"{self.path}:{self.line} {self.function} -> {self.operation}"


EXPECTED_MEMBERSHIP_WRITES = (
    ExpectedWrite(
        "tldw_Server_API/app/core/AuthNZ/membership_writer.py",
        "MembershipWriter._insert_membership",
        "INSERT org_members",
    ),
    ExpectedWrite(
        "tldw_Server_API/app/core/AuthNZ/membership_writer.py",
        "MembershipWriter._insert_membership",
        "INSERT org_members",
    ),
    ExpectedWrite(
        "tldw_Server_API/app/core/AuthNZ/membership_writer.py",
        "MembershipWriter._insert_membership",
        "INSERT team_members",
    ),
    ExpectedWrite(
        "tldw_Server_API/app/core/AuthNZ/membership_writer.py",
        "MembershipWriter._insert_membership",
        "INSERT team_members",
    ),
    ExpectedWrite(
        "tldw_Server_API/app/core/AuthNZ/membership_writer.py",
        "MembershipWriter._delete_membership",
        "DELETE org_members",
    ),
    ExpectedWrite(
        "tldw_Server_API/app/core/AuthNZ/membership_writer.py",
        "MembershipWriter._delete_membership",
        "DELETE org_members",
    ),
    ExpectedWrite(
        "tldw_Server_API/app/core/AuthNZ/membership_writer.py",
        "MembershipWriter._delete_membership",
        "DELETE team_members",
    ),
    ExpectedWrite(
        "tldw_Server_API/app/core/AuthNZ/membership_writer.py",
        "MembershipWriter._delete_membership",
        "DELETE team_members",
    ),
    ExpectedWrite(
        "tldw_Server_API/app/core/AuthNZ/membership_writer.py",
        "MembershipWriter._update_membership_role",
        "UPDATE org_members",
    ),
    ExpectedWrite(
        "tldw_Server_API/app/core/AuthNZ/membership_writer.py",
        "MembershipWriter._update_membership_role",
        "UPDATE org_members",
    ),
    ExpectedWrite(
        "tldw_Server_API/app/core/AuthNZ/membership_writer.py",
        "MembershipWriter._update_membership_role",
        "UPDATE team_members",
    ),
    ExpectedWrite(
        "tldw_Server_API/app/core/AuthNZ/membership_writer.py",
        "MembershipWriter._update_membership_role",
        "UPDATE team_members",
    ),
)

EXPECTED_PARENT_SCOPE_DELETES = (
    ExpectedWrite(
        "tldw_Server_API/app/core/AuthNZ/repos/orgs_teams_repo.py",
        "AuthnzOrgsTeamsRepo.delete_organization_with_provider_secrets",
        "DELETE organizations",
    ),
    ExpectedWrite(
        "tldw_Server_API/app/core/AuthNZ/repos/orgs_teams_repo.py",
        "AuthnzOrgsTeamsRepo.delete_organization_with_provider_secrets",
        "DELETE organizations",
    ),
    ExpectedWrite(
        "tldw_Server_API/app/core/AuthNZ/repos/orgs_teams_repo.py",
        "AuthnzOrgsTeamsRepo.delete_team_with_provider_secrets",
        "DELETE teams",
    ),
    ExpectedWrite(
        "tldw_Server_API/app/core/AuthNZ/repos/orgs_teams_repo.py",
        "AuthnzOrgsTeamsRepo.delete_team_with_provider_secrets",
        "DELETE teams",
    ),
)

EXPECTED_MEMBERSHIP_SCOPE_SQL_HELPER_CALLERS = (
    *(
        (
            "tldw_Server_API/app/core/AuthNZ/membership_writer.py",
            "MembershipWriter._insert_membership",
        )
        for _ in range(4)
    ),
    *(
        (
            "tldw_Server_API/app/core/AuthNZ/membership_writer.py",
            "MembershipWriter._update_membership_role",
        )
        for _ in range(4)
    ),
    *(
        (
            "tldw_Server_API/app/core/AuthNZ/membership_writer.py",
            "MembershipWriter._delete_membership",
        )
        for _ in range(4)
    ),
    *(
        (
            "tldw_Server_API/app/core/AuthNZ/repos/orgs_teams_repo.py",
            "AuthnzOrgsTeamsRepo.delete_organization_with_provider_secrets",
        )
        for _ in range(2)
    ),
    *(
        (
            "tldw_Server_API/app/core/AuthNZ/repos/orgs_teams_repo.py",
            "AuthnzOrgsTeamsRepo.delete_team_with_provider_secrets",
        )
        for _ in range(2)
    ),
)
EXPECTED_PRIVILEGED_MEMBERSHIP_SCOPE_SQL_CALLERS = (
    *(
        ("_execute_membership_scope_sql", path, function)
        for path, function in EXPECTED_MEMBERSHIP_SCOPE_SQL_HELPER_CALLERS
    ),
    (
        "_mint_membership_scope_sql",
        "tldw_Server_API/app/core/AuthNZ/profile_user_write_guard.py",
        "_execute_membership_scope_sql",
    ),
    (
        "_execute_membership_scope_sql",
        "tldw_Server_API/app/core/AuthNZ/profile_user_write_guard.py",
        "_execute_postgres_membership_timestamp_repair",
    ),
    (
        "_execute_postgres_membership_timestamp_repair",
        "tldw_Server_API/app/core/AuthNZ/profile_candidate_schema.py",
        "repair_postgres_profile_candidate_timestamps",
    ),
)
APPROVED_PRIVILEGED_MEMBERSHIP_SCOPE_SQL_CALLERS = frozenset(
    EXPECTED_PRIVILEGED_MEMBERSHIP_SCOPE_SQL_CALLERS
)
EXPECTED_PRIVILEGED_MEMBERSHIP_SCOPE_SQL_IMPORTS = (
    (
        "_execute_membership_scope_sql",
        "_execute_membership_scope_sql",
        "tldw_Server_API/app/core/AuthNZ/membership_writer.py",
        "<module>",
    ),
    (
        "_execute_membership_scope_sql",
        "_execute_membership_scope_sql",
        "tldw_Server_API/app/core/AuthNZ/repos/orgs_teams_repo.py",
        "<module>",
    ),
    (
        "_execute_postgres_membership_timestamp_repair",
        "_execute_postgres_membership_timestamp_repair",
        "tldw_Server_API/app/core/AuthNZ/profile_candidate_schema.py",
        "<module>",
    ),
)
APPROVED_PRIVILEGED_MEMBERSHIP_SCOPE_SQL_IMPORTS = frozenset(
    EXPECTED_PRIVILEGED_MEMBERSHIP_SCOPE_SQL_IMPORTS
)

EXPECTED_EXCLUDED_WRITES = (
    ExpectedWrite(
        "tldw_Server_API/app/core/AuthNZ/migrations.py",
        "migration_025_team_members_added_at",
        "UPDATE team_members",
    ),
    ExpectedWrite(
        "tldw_Server_API/app/core/AuthNZ/migrations.py",
        "migration_093_harmonize_users_write_columns",
        "UPDATE users (uuid)",
    ),
)


def _relative_path(path: Path, repo_root: Path = REPO_ROOT) -> str:
    return path.relative_to(repo_root).as_posix()


def _normalized_schema_sql(sql: str) -> str:
    return " ".join(sql.split())


def _sqlite_users_routine_inventory(
    schema_sql: str,
) -> tuple[tuple[str, str, str], ...]:
    connection = sqlite3.connect(":memory:")
    try:
        connection.executescript(schema_sql)
        rows = connection.execute(
            """
            SELECT name, tbl_name, sql
              FROM sqlite_master
             WHERE type = 'trigger'
               AND instr(lower(sql), 'users') > 0
             ORDER BY name
            """
        ).fetchall()
    finally:
        connection.close()
    return tuple(
        (str(name), str(table), _normalized_schema_sql(str(sql)))
        for name, table, sql in rows
    )


def _postgres_stored_routine_declarations(
    schema_sql: str,
) -> tuple[tuple[str, str], ...]:
    declarations: list[tuple[str, str]] = []
    for match in _POSTGRES_ROUTINE_DECLARATION_RE.finditer(schema_sql):
        name = ".".join(
            part.strip().strip('"').lower()
            for part in match.group("name").split(".")
        )
        declarations.append((match.group("kind").lower(), name))
    return tuple(sorted(declarations))


def _qualified_scope(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> str:
    names: list[str] = []
    current: ast.AST | None = node
    while current is not None:
        if isinstance(current, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            names.append(current.name)
        current = parents.get(current)
    return ".".join(reversed(names)) or "<module>"


def _nodes_in_scope_and_parents(
    tree: ast.AST,
) -> tuple[dict[ast.AST, list[ast.AST]], dict[ast.AST, ast.AST]]:
    grouped: dict[ast.AST, list[ast.AST]] = defaultdict(list)
    parents: dict[ast.AST, ast.AST] = {}
    enclosing_scope: dict[ast.AST, ast.AST] = {tree: tree}
    for parent in ast.walk(tree):
        scope = enclosing_scope[parent]
        if not isinstance(
            parent,
            (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
        ):
            grouped[scope].append(parent)
        child_scope = (
            parent
            if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef))
            else scope
        )
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent
            enclosing_scope[child] = child_scope
    return grouped, parents


def _assignment_targets(node: ast.AST) -> Iterable[tuple[str, ast.AST]]:
    if isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name):
                yield target.id, node.value
    elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        if node.value is not None:
            yield node.target.id, node.value
    elif isinstance(node, ast.NamedExpr) and isinstance(node.target, ast.Name):
        yield node.target.id, node.value


def _render_joined_string(
    node: ast.JoinedStr,
    assignments: dict[str, list[ast.AST]],
    globals_: dict[str, list[ast.AST]],
    seen: frozenset[str],
) -> set[str]:
    rendered = ""
    for value in node.values:
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            rendered += value.value
        elif isinstance(value, ast.FormattedValue):
            rendered += "{" + ast.unparse(value.value) + "}"
    return {rendered}


def _resolve_container_nodes(
    node: ast.AST,
    assignments: dict[str, list[ast.AST]],
    globals_: dict[str, list[ast.AST]],
    seen: frozenset[str] = frozenset(),
) -> tuple[ast.AST, ...]:
    if isinstance(node, ast.Name):
        if node.id in seen:
            return ()
        containers: list[ast.AST] = []
        values = assignments.get(node.id) or globals_.get(node.id) or []
        for value in values:
            containers.extend(
                _resolve_container_nodes(
                    value,
                    assignments,
                    globals_,
                    seen | {node.id},
                )
            )
        return tuple(containers)
    if isinstance(node, ast.IfExp):
        return _resolve_container_nodes(
            node.body,
            assignments,
            globals_,
            seen,
        ) + _resolve_container_nodes(
            node.orelse,
            assignments,
            globals_,
            seen,
        )
    if isinstance(node, (ast.Dict, ast.List, ast.Tuple)):
        return (node,)
    return ()


def _resolve_subscript(
    node: ast.Subscript,
    assignments: dict[str, list[ast.AST]],
    globals_: dict[str, list[ast.AST]],
    seen: frozenset[str],
) -> set[str]:
    try:
        key = ast.literal_eval(node.slice)
    except (ValueError, TypeError):
        return set()

    containers = _resolve_container_nodes(
        node.value,
        assignments,
        globals_,
        seen,
    )

    resolved: set[str] = set()
    for container in containers:
        if isinstance(container, ast.Dict):
            for key_node, value_node in zip(container.keys, container.values):
                if key_node is None:
                    continue
                try:
                    candidate = ast.literal_eval(key_node)
                except (ValueError, TypeError):
                    continue
                if candidate == key:
                    resolved.update(
                        _resolve_strings(
                            value_node,
                            assignments,
                            globals_,
                            seen,
                        )
                    )
        elif isinstance(container, (ast.List, ast.Tuple)) and isinstance(key, int):
            try:
                value_node = container.elts[key]
            except IndexError:
                continue
            resolved.update(
                _resolve_strings(value_node, assignments, globals_, seen)
            )
    return resolved


def _resolve_string_sequences(
    node: ast.AST,
    assignments: dict[str, list[ast.AST]],
    globals_: dict[str, list[ast.AST]],
    seen: frozenset[str],
) -> set[tuple[str, ...]]:
    if isinstance(node, ast.Name):
        if node.id in seen:
            return set()
        resolved: set[tuple[str, ...]] = set()
        values = assignments.get(node.id) or globals_.get(node.id) or []
        for value in values:
            resolved.update(
                _resolve_string_sequences(
                    value,
                    assignments,
                    globals_,
                    seen | {node.id},
                )
            )
        return resolved
    if isinstance(node, (ast.List, ast.Tuple)):
        combinations: set[tuple[str, ...]] = {()}
        for element in node.elts:
            values = _resolve_strings(element, assignments, globals_, seen)
            if not values:
                return set()
            combinations = {
                prefix + (value,)
                for prefix in combinations
                for value in values
            }
        return combinations
    if isinstance(node, ast.IfExp):
        return _resolve_string_sequences(
            node.body,
            assignments,
            globals_,
            seen,
        ) | _resolve_string_sequences(
            node.orelse,
            assignments,
            globals_,
            seen,
        )
    return set()


def _resolve_strings(
    node: ast.AST,
    assignments: dict[str, list[ast.AST]],
    globals_: dict[str, list[ast.AST]],
    seen: frozenset[str] = frozenset(),
) -> set[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return {node.value}
    if isinstance(node, ast.JoinedStr):
        return _render_joined_string(node, assignments, globals_, seen)
    if isinstance(node, ast.Subscript):
        return _resolve_subscript(node, assignments, globals_, seen)
    if isinstance(node, ast.Name):
        if node.id in seen:
            return set()
        values = assignments.get(node.id) or globals_.get(node.id) or []
        resolved: set[str] = set()
        for value in values:
            resolved.update(
                _resolve_strings(value, assignments, globals_, seen | {node.id})
            )
        return resolved
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _resolve_strings(node.left, assignments, globals_, seen)
        right = _resolve_strings(node.right, assignments, globals_, seen)
        return {lhs + rhs for lhs in left for rhs in right}
    if isinstance(node, ast.IfExp):
        return _resolve_strings(
            node.body, assignments, globals_, seen
        ) | _resolve_strings(node.orelse, assignments, globals_, seen)
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Attribute):
            if node.func.attr in {"format", "format_map"}:
                return _resolve_strings(
                    node.func.value,
                    assignments,
                    globals_,
                    seen,
                )
            if node.func.attr == "join" and node.args:
                separators = _resolve_strings(
                    node.func.value,
                    assignments,
                    globals_,
                    seen,
                )
                sequences = _resolve_string_sequences(
                    node.args[0],
                    assignments,
                    globals_,
                    seen,
                )
                return {
                    separator.join(sequence)
                    for separator in separators
                    for sequence in sequences
                }
        if node.args:
            return _resolve_strings(node.args[0], assignments, globals_, seen)
    return set()


def _call_name(node: ast.Call) -> str:
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    if isinstance(node.func, ast.Name):
        return node.func.id
    return ""


def _privileged_membership_scope_sql_import_aliases(
    nodes: Iterable[ast.AST],
) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for node in nodes:
        if not isinstance(node, ast.ImportFrom):
            continue
        for imported in node.names:
            if imported.name not in PRIVILEGED_MEMBERSHIP_SCOPE_SQL_ENTRYPOINTS:
                continue
            aliases[imported.asname or imported.name] = imported.name
    return aliases


def _profile_user_write_guard_module_aliases(
    nodes: Iterable[ast.AST],
) -> frozenset[str]:
    aliases: set[str] = set()
    for node in nodes:
        if isinstance(node, ast.Import):
            for imported in node.names:
                if imported.name.endswith(".profile_user_write_guard"):
                    aliases.add(imported.asname or imported.name.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom) and (
            node.module or ""
        ).endswith(".AuthNZ"):
            for imported in node.names:
                if imported.name == "profile_user_write_guard":
                    aliases.add(imported.asname or imported.name)
    return frozenset(aliases)


def _privileged_wrapper_import_aliases(
    nodes: Iterable[ast.AST],
) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for node in nodes:
        if not isinstance(node, ast.ImportFrom):
            continue
        module = node.module or ""
        for imported in node.names:
            if module == "functools" and imported.name == "partial":
                aliases[imported.asname or imported.name] = "partial"
            elif module == "builtins" and imported.name == "getattr":
                aliases[imported.asname or imported.name] = "getattr"
            elif module == "builtins" and imported.name == "vars":
                aliases[imported.asname or imported.name] = "vars"
            elif module == "builtins" and imported.name == "__import__":
                aliases[imported.asname or imported.name] = "__import__"
            elif module == "importlib" and imported.name == "import_module":
                aliases[imported.asname or imported.name] = "import_module"
            elif module == "operator" and imported.name == "attrgetter":
                aliases[imported.asname or imported.name] = "attrgetter"
    return aliases


def _resolve_privileged_wrapper_call_names(
    expression: ast.AST,
    assignments: dict[str, list[ast.AST]],
    globals_: dict[str, list[ast.AST]],
    aliases: dict[str, str],
    global_aliases: dict[str, str],
    seen: frozenset[str] = frozenset(),
) -> set[str]:
    if isinstance(expression, ast.Call):
        return _resolve_privileged_wrapper_call_names(
            expression.func,
            assignments,
            globals_,
            aliases,
            global_aliases,
            seen,
        )
    if isinstance(expression, ast.Attribute):
        return (
            {expression.attr}
            if expression.attr
            in {
                "__getattribute__",
                "__import__",
                "getattr",
                "attrgetter",
                "partial",
                "import_module",
                "vars",
            }
            else set()
        )
    if not isinstance(expression, ast.Name) or expression.id in seen:
        return set()

    resolved: set[str] = set()
    if expression.id in {
        "__getattribute__",
        "__import__",
        "getattr",
        "attrgetter",
        "partial",
        "import_module",
        "vars",
    }:
        resolved.add(expression.id)
    imported_name = aliases.get(expression.id) or global_aliases.get(expression.id)
    if imported_name is not None:
        resolved.add(imported_name)
    for value in assignments.get(expression.id) or globals_.get(expression.id) or []:
        resolved.update(
            _resolve_privileged_wrapper_call_names(
                value,
                assignments,
                globals_,
                aliases,
                global_aliases,
                seen | {expression.id},
            )
        )
    return resolved


def _resolve_attrgetter_attribute_names(
    expression: ast.AST,
    assignments: dict[str, list[ast.AST]],
    globals_: dict[str, list[ast.AST]],
    aliases: dict[str, str],
    global_aliases: dict[str, str],
    seen: frozenset[str] = frozenset(),
) -> tuple[bool, set[str]]:
    if isinstance(expression, ast.Call):
        wrapper_names = _resolve_privileged_wrapper_call_names(
            expression.func,
            assignments,
            globals_,
            aliases,
            global_aliases,
        )
        if "attrgetter" not in wrapper_names:
            return False, set()
        if not expression.args:
            return True, set()
        attribute_names: set[str] = set()
        for argument in expression.args:
            resolved = _resolve_strings(argument, assignments, globals_)
            if not resolved:
                attribute_names.add(_DYNAMIC_PRIVILEGED_MEMBERSHIP_SCOPE_SQL)
            attribute_names.update(resolved)
        return True, attribute_names
    if not isinstance(expression, ast.Name) or expression.id in seen:
        return False, set()

    matched = False
    attribute_names: set[str] = set()
    for value in assignments.get(expression.id) or globals_.get(expression.id) or []:
        value_matched, value_names = _resolve_attrgetter_attribute_names(
            value,
            assignments,
            globals_,
            aliases,
            global_aliases,
            seen | {expression.id},
        )
        matched = matched or value_matched
        attribute_names.update(value_names)
    return matched, attribute_names


def _privileged_membership_scope_sql_imports(
    tree: ast.AST,
    *,
    relative_path: str,
    parents: dict[ast.AST, ast.AST],
) -> tuple[tuple[tuple[str, str, str, str], int], ...]:
    imports: list[tuple[tuple[str, str, str, str], int]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        for imported in node.names:
            if imported.name not in PRIVILEGED_MEMBERSHIP_SCOPE_SQL_ENTRYPOINTS:
                continue
            imports.append(
                (
                    (
                        imported.name,
                        imported.asname or imported.name,
                        relative_path,
                        _qualified_scope(node, parents),
                    ),
                    node.lineno,
                )
            )
    return tuple(sorted(imports))


def _is_profile_user_write_guard_module(
    expression: ast.AST,
    aliases: frozenset[str],
    global_aliases: frozenset[str],
    assignments: dict[str, list[ast.AST]],
    globals_: dict[str, list[ast.AST]],
    wrapper_aliases: dict[str, str],
    global_wrapper_aliases: dict[str, str],
    seen: frozenset[str] = frozenset(),
) -> bool:
    if isinstance(expression, ast.Name):
        if expression.id in aliases or expression.id in global_aliases:
            return True
        if expression.id in seen:
            return False
        return any(
            _is_profile_user_write_guard_module(
                value,
                aliases,
                global_aliases,
                assignments,
                globals_,
                wrapper_aliases,
                global_wrapper_aliases,
                seen | {expression.id},
            )
            for value in assignments.get(expression.id)
            or globals_.get(expression.id)
            or []
        )
    if isinstance(expression, ast.Call) and expression.args:
        call_names = _resolve_privileged_wrapper_call_names(
            expression.func,
            assignments,
            globals_,
            wrapper_aliases,
            global_wrapper_aliases,
        )
        if call_names & {"__import__", "import_module"}:
            module_names = _resolve_strings(
                expression.args[0],
                assignments,
                globals_,
            )
            return bool(module_names) and all(
                name.endswith(".profile_user_write_guard")
                for name in module_names
            )
    return (
        isinstance(expression, ast.Attribute)
        and expression.attr == "profile_user_write_guard"
    )


def _resolve_privileged_membership_scope_sql_entrypoints(
    expression: ast.AST,
    assignments: dict[str, list[ast.AST]],
    globals_: dict[str, list[ast.AST]],
    aliases: dict[str, str],
    global_aliases: dict[str, str],
    module_aliases: frozenset[str],
    global_module_aliases: frozenset[str],
    wrapper_aliases: dict[str, str],
    global_wrapper_aliases: dict[str, str],
    seen: frozenset[str] = frozenset(),
) -> set[str]:
    def _is_guard_namespace(candidate: ast.AST) -> bool:
        if (
            isinstance(candidate, ast.Attribute)
            and candidate.attr == "__dict__"
        ):
            return _is_profile_user_write_guard_module(
                candidate.value,
                module_aliases,
                global_module_aliases,
                assignments,
                globals_,
                wrapper_aliases,
                global_wrapper_aliases,
            )
        if isinstance(candidate, ast.Call) and candidate.args:
            call_names = _resolve_privileged_wrapper_call_names(
                candidate.func,
                assignments,
                globals_,
                wrapper_aliases,
                global_wrapper_aliases,
            )
            return "vars" in call_names and _is_profile_user_write_guard_module(
                candidate.args[0],
                module_aliases,
                global_module_aliases,
                assignments,
                globals_,
                wrapper_aliases,
                global_wrapper_aliases,
            )
        return False

    if isinstance(expression, ast.Subscript):
        mapping = expression.value
        if _is_guard_namespace(mapping):
            attribute_names = _resolve_strings(
                expression.slice,
                assignments,
                globals_,
            )
            if not attribute_names:
                return {_DYNAMIC_PRIVILEGED_MEMBERSHIP_SCOPE_SQL}
            return {
                attribute_name
                for attribute_name in attribute_names
                if attribute_name in PRIVILEGED_MEMBERSHIP_SCOPE_SQL_ENTRYPOINTS
            }
        return set()
    if isinstance(expression, ast.Attribute):
        if expression.attr in PRIVILEGED_MEMBERSHIP_SCOPE_SQL_ENTRYPOINTS:
            return {expression.attr}
        return set()
    if isinstance(expression, ast.Call):
        if (
            isinstance(expression.func, ast.Attribute)
            and expression.func.attr == "get"
            and expression.args
            and _is_guard_namespace(expression.func.value)
        ):
            attribute_names = _resolve_strings(
                expression.args[0],
                assignments,
                globals_,
            )
            if not attribute_names:
                return {_DYNAMIC_PRIVILEGED_MEMBERSHIP_SCOPE_SQL}
            return {
                attribute_name
                for attribute_name in attribute_names
                if attribute_name in PRIVILEGED_MEMBERSHIP_SCOPE_SQL_ENTRYPOINTS
            }
        wrapper_call_names = _resolve_privileged_wrapper_call_names(
            expression.func,
            assignments,
            globals_,
            wrapper_aliases,
            global_wrapper_aliases,
        )
        is_attrgetter, attrgetter_names = _resolve_attrgetter_attribute_names(
            expression.func,
            assignments,
            globals_,
            wrapper_aliases,
            global_wrapper_aliases,
        )
        if (
            is_attrgetter
            and expression.args
            and _is_profile_user_write_guard_module(
                expression.args[0],
                module_aliases,
                global_module_aliases,
                assignments,
                globals_,
                wrapper_aliases,
                global_wrapper_aliases,
            )
        ):
            attrgetter_segments = {
                segment
                for attribute_name in attrgetter_names
                for segment in attribute_name.split(".")
                if segment
            }
            if (
                not attrgetter_names
                or _DYNAMIC_PRIVILEGED_MEMBERSHIP_SCOPE_SQL in attrgetter_names
                or attrgetter_segments & {"__dict__", "__getattribute__"}
            ):
                return {_DYNAMIC_PRIVILEGED_MEMBERSHIP_SCOPE_SQL}
            return attrgetter_segments & PRIVILEGED_MEMBERSHIP_SCOPE_SQL_ENTRYPOINTS
        getattribute_name_index: int | None = None
        if "__getattribute__" in wrapper_call_names and expression.args:
            if (
                isinstance(expression.func, ast.Attribute)
                and _is_profile_user_write_guard_module(
                    expression.func.value,
                    module_aliases,
                    global_module_aliases,
                    assignments,
                    globals_,
                    wrapper_aliases,
                    global_wrapper_aliases,
                )
            ):
                getattribute_name_index = 0
            elif len(expression.args) >= 2 and _is_profile_user_write_guard_module(
                expression.args[0],
                module_aliases,
                global_module_aliases,
                assignments,
                globals_,
                wrapper_aliases,
                global_wrapper_aliases,
            ):
                getattribute_name_index = 1
        if getattribute_name_index is not None:
            attribute_names = _resolve_strings(
                expression.args[getattribute_name_index],
                assignments,
                globals_,
            )
            if not attribute_names:
                return {_DYNAMIC_PRIVILEGED_MEMBERSHIP_SCOPE_SQL}
            return {
                attribute_name
                for attribute_name in attribute_names
                if attribute_name in PRIVILEGED_MEMBERSHIP_SCOPE_SQL_ENTRYPOINTS
            }
        if (
            "getattr" in wrapper_call_names
            and len(expression.args) >= 2
            and _is_profile_user_write_guard_module(
                expression.args[0],
                module_aliases,
                global_module_aliases,
                assignments,
                globals_,
                wrapper_aliases,
                global_wrapper_aliases,
            )
        ):
            attribute_names = _resolve_strings(
                expression.args[1],
                assignments,
                globals_,
            )
            if not attribute_names:
                return {_DYNAMIC_PRIVILEGED_MEMBERSHIP_SCOPE_SQL}
            return {
                attribute_name
                for attribute_name in attribute_names
                if attribute_name in PRIVILEGED_MEMBERSHIP_SCOPE_SQL_ENTRYPOINTS
            }
        if "partial" in wrapper_call_names and expression.args:
            wrapped = _resolve_privileged_membership_scope_sql_entrypoints(
                expression.args[0],
                assignments,
                globals_,
                aliases,
                global_aliases,
                module_aliases,
                global_module_aliases,
                wrapper_aliases,
                global_wrapper_aliases,
                seen,
            )
            if wrapped:
                return wrapped | {_WRAPPED_PRIVILEGED_MEMBERSHIP_SCOPE_SQL}
        return set()
    if not isinstance(expression, ast.Name) or expression.id in seen:
        return set()

    resolved: set[str] = set()
    if expression.id in PRIVILEGED_MEMBERSHIP_SCOPE_SQL_ENTRYPOINTS:
        resolved.add(expression.id)
    imported_name = aliases.get(expression.id) or global_aliases.get(expression.id)
    if imported_name is not None:
        resolved.add(imported_name)
    for value in assignments.get(expression.id) or globals_.get(expression.id) or []:
        resolved.update(
            _resolve_privileged_membership_scope_sql_entrypoints(
                value,
                assignments,
                globals_,
                aliases,
                global_aliases,
                module_aliases,
                global_module_aliases,
                wrapper_aliases,
                global_wrapper_aliases,
                seen | {expression.id},
            )
        )
    return resolved


def _direct_membership_import_aliases(tree: ast.AST) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        for imported in node.names:
            if imported.name not in DIRECT_MEMBERSHIP_CALL_NAMES:
                continue
            aliases[imported.asname or imported.name] = imported.name
    return aliases


def _trusted_membership_context_symbols(tree: ast.AST) -> frozenset[str]:
    symbols: set[str] = set()
    for node in ast.walk(tree):
        target: ast.AST | None = None
        value: ast.AST | None = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = node.targets[0], node.value
        elif isinstance(node, ast.AnnAssign):
            target, value = node.target, node.value
        if not isinstance(target, ast.Name) or not isinstance(value, ast.Call):
            continue
        if _call_name(value) != "TrustedMembershipWriteContext":
            continue
        if any(keyword.arg == "trusted_reason" for keyword in value.keywords):
            symbols.add(target.id)
    return frozenset(symbols)


def _trusted_membership_reasons(tree: ast.AST) -> frozenset[str]:
    reasons: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if _call_name(node) != "TrustedMembershipWriteContext":
            continue
        reason_keywords = [
            keyword for keyword in node.keywords if keyword.arg == "trusted_reason"
        ]
        if len(reason_keywords) != 1:
            reasons.add("<missing>")
            continue
        expression = reason_keywords[0].value
        if (
            isinstance(expression, ast.Attribute)
            and isinstance(expression.value, ast.Name)
            and expression.value.id == "TrustedMembershipReason"
        ):
            reasons.add(expression.attr)
        else:
            reasons.add("<dynamic>")
    return frozenset(reasons)


def _membership_context_category(
    expression: ast.AST,
    *,
    trusted_symbols: frozenset[str],
) -> str | None:
    if isinstance(expression, ast.Name):
        if expression.id in trusted_symbols:
            return "trusted"
        if expression.id == "context":
            return "passthrough"
        return None
    if not isinstance(expression, ast.Call):
        return None
    call_name = _call_name(expression)
    if call_name in ACTOR_MEMBERSHIP_CONTEXT_FACTORIES or call_name == (
        "ActorMembershipWriteContext"
    ):
        return "actor"
    if call_name == "TrustedMembershipWriteContext" and any(
        keyword.arg == "trusted_reason" for keyword in expression.keywords
    ):
        return "trusted"
    return None


def _attribute_base_name(node: ast.Call) -> str:
    if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
        return node.func.value.id
    return ""


def _query_arguments(
    node: ast.Call,
    *,
    call_name: str,
) -> tuple[ast.AST, ...]:
    keyword_arguments = tuple(
        keyword.value
        for keyword in node.keywords
        if keyword.arg in {"query", "sql", "statement"}
    )
    if keyword_arguments:
        return keyword_arguments
    argument_index = (
        1
        if call_name in {"_execute_compat", "_execute_membership_scope_sql"}
        else 0
    )
    return tuple(node.args[argument_index : argument_index + 1])


def _subscript_candidate_strings(
    node: ast.Subscript,
    assignments: dict[str, list[ast.AST]],
    globals_: dict[str, list[ast.AST]],
) -> set[str]:
    containers = _resolve_container_nodes(
        node.value,
        assignments,
        globals_,
    )

    candidates: set[str] = set()
    for container in containers:
        if isinstance(container, ast.Dict):
            values = container.values
        elif isinstance(container, (ast.List, ast.Tuple)):
            values = container.elts
        else:
            continue
        for value in values:
            candidates.update(_resolve_strings(value, assignments, globals_))
    return candidates


def _contains_inventory_write(sql: str) -> bool:
    for verb, table, columns in _parse_operations(sql, is_script=True):
        if table == "users" and verb in {"INSERT", "UPDATE"}:
            if "<dynamic>" in columns or PROFILE_VISIBLE_COLUMNS & set(columns):
                return True
        elif table in MEMBERSHIP_TABLES or (
            table in PARENT_SCOPE_TABLES and verb == "DELETE"
        ):
            return True
    return False


def _requires_static_resolution(
    node: ast.AST,
    assignments: dict[str, list[ast.AST]],
    globals_: dict[str, list[ast.AST]],
    seen: frozenset[str] = frozenset(),
) -> bool:
    if isinstance(node, ast.Name):
        if node.id in seen:
            return False
        values = assignments.get(node.id) or globals_.get(node.id) or []
        return any(
            _requires_static_resolution(
                value,
                assignments,
                globals_,
                seen | {node.id},
            )
            for value in values
        )
    if isinstance(node, ast.Constant):
        return isinstance(node.value, str)
    if isinstance(node, ast.JoinedStr):
        return True
    if isinstance(node, ast.Subscript):
        return any(
            _contains_inventory_write(candidate)
            for candidate in _subscript_candidate_strings(
                node,
                assignments,
                globals_,
            )
        )
    if isinstance(node, ast.BinOp):
        return _requires_static_resolution(
            node.left,
            assignments,
            globals_,
            seen,
        ) and _requires_static_resolution(
            node.right,
            assignments,
            globals_,
            seen,
        )
    if isinstance(node, ast.IfExp):
        return _requires_static_resolution(
            node.body,
            assignments,
            globals_,
            seen,
        ) and _requires_static_resolution(
            node.orelse,
            assignments,
            globals_,
            seen,
        )
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Attribute):
            if node.func.attr in {"format", "format_map"}:
                return _requires_static_resolution(
                    node.func.value,
                    assignments,
                    globals_,
                    seen,
                )
            if node.func.attr == "join" and node.args:
                return bool(
                    _resolve_strings(
                        node.func.value,
                        assignments,
                        globals_,
                        seen,
                    )
                    and _resolve_string_sequences(
                        node.args[0],
                        assignments,
                        globals_,
                        seen,
                    )
                )
        return bool(node.args) and _requires_static_resolution(
            node.args[0],
            assignments,
            globals_,
            seen,
        )
    return False


def _clean_identifier(raw: str) -> str:
    identifier = raw.strip().strip('"`[]')
    return identifier.rsplit(".", 1)[-1].lower()


def _column_names(raw: str) -> tuple[str, ...]:
    parts: list[str] = []
    start = 0
    depth = 0
    for index, character in enumerate(raw):
        if character == "(":
            depth += 1
        elif character == ")" and depth:
            depth -= 1
        elif character == "," and depth == 0:
            parts.append(raw[start:index])
            start = index + 1
    parts.append(raw[start:])

    columns: list[str] = []
    for part in parts:
        candidate = part.strip()
        if not candidate:
            continue
        if candidate.startswith("{"):
            columns.append("<dynamic>")
            continue
        candidate = candidate.split("=", 1)[0].strip()
        candidate = _clean_identifier(candidate)
        if re.fullmatch(r"[a-z_]\w*", candidate):
            columns.append(candidate)
        else:
            columns.append("<dynamic>")
    return tuple(columns)


def _parse_operation(sql: str) -> tuple[str, str, tuple[str, ...]] | None:
    normalized = " ".join(sql.split())
    match = _WRITE_RE.search(normalized)
    if not match:
        return None
    verb_token = match.group("verb").upper()
    verb = "INSERT" if verb_token.startswith("INSERT") else verb_token.split()[0]
    table = _clean_identifier(match.group("table"))
    tail = normalized[match.end() :]
    columns: tuple[str, ...] = ()
    if verb == "INSERT":
        columns_match = _INSERT_COLUMNS_RE.search(tail)
        if columns_match:
            columns = _column_names(columns_match.group("columns"))
    elif verb == "UPDATE":
        columns_match = _UPDATE_COLUMNS_RE.search(tail)
        if columns_match:
            columns = _column_names(columns_match.group("columns"))
    return verb, table, columns


def _split_sql_statements(script: str) -> tuple[str, ...]:
    statements: list[str] = []
    current: list[str] = []
    quote: str | None = None
    index = 0
    while index < len(script):
        character = script[index]
        following = script[index + 1] if index + 1 < len(script) else ""
        if quote is not None:
            current.append(character)
            closing = "]" if quote == "[" else quote
            if character == closing:
                if following == closing and quote != "[":
                    current.append(following)
                    index += 1
                else:
                    quote = None
            index += 1
            continue
        if character in {"'", '"', "`", "["}:
            quote = character
            current.append(character)
            index += 1
            continue
        if character == "-" and following == "-":
            newline = script.find("\n", index + 2)
            if newline == -1:
                break
            current.append(" ")
            index = newline
            continue
        if character == "/" and following == "*":
            end = script.find("*/", index + 2)
            if end == -1:
                current.append(script[index:])
                break
            current.append(" ")
            index = end + 2
            continue
        if character == ";":
            statement = "".join(current).strip()
            if statement:
                statements.append(statement)
            current.clear()
        else:
            current.append(character)
        index += 1
    statement = "".join(current).strip()
    if statement:
        statements.append(statement)
    return tuple(statements)


def _parse_operations(
    sql: str,
    *,
    is_script: bool,
) -> tuple[tuple[str, str, tuple[str, ...]], ...]:
    statements = _split_sql_statements(sql) if is_script else (sql,)
    return tuple(
        operation
        for statement in statements
        if (operation := _parse_operation(statement)) is not None
    )


def _operation_label(
    verb: str,
    table: str,
    columns: tuple[str, ...],
) -> str:
    if table in MEMBERSHIP_TABLES:
        return f"{verb} {table}"
    if columns:
        return f"{verb} {table} ({', '.join(columns)})"
    return f"{verb} {table}"


def _scan_python_tree(
    *,
    path: Path,
    tree: ast.AST,
    repo_root: Path,
) -> tuple[ObservedWrite, ...]:
    observed: list[ObservedWrite] = []
    relative_path = _relative_path(path, repo_root)
    grouped, parents = _nodes_in_scope_and_parents(tree)
    module_assignments: dict[str, list[ast.AST]] = defaultdict(list)
    module_nodes = grouped.get(tree, [])
    module_aliases = _privileged_membership_scope_sql_import_aliases(module_nodes)
    module_guard_aliases = _profile_user_write_guard_module_aliases(module_nodes)
    module_wrapper_aliases = _privileged_wrapper_import_aliases(module_nodes)
    for node in module_nodes:
        for name, value in _assignment_targets(node):
            module_assignments[name].append(value)
            exported_entrypoints = (
                _resolve_privileged_membership_scope_sql_entrypoints(
                    value,
                    module_assignments,
                    module_assignments,
                    module_aliases,
                    module_aliases,
                    module_guard_aliases,
                    module_guard_aliases,
                    module_wrapper_aliases,
                    module_wrapper_aliases,
                )
            )
            if exported_entrypoints:
                raise AssertionError(
                    "Membership-scope SQL capabilities must not be re-exported "
                    f"at {relative_path}:{node.lineno}: {name} resolves to "
                    f"{sorted(exported_entrypoints)}."
                )

    for scope, nodes in grouped.items():
        assignments: dict[str, list[ast.AST]] = defaultdict(list)
        aliases = _privileged_membership_scope_sql_import_aliases(nodes)
        guard_aliases = _profile_user_write_guard_module_aliases(nodes)
        wrapper_aliases = _privileged_wrapper_import_aliases(nodes)
        for node in nodes:
            for name, value in _assignment_targets(node):
                assignments[name].append(value)
        for node in nodes:
            if not isinstance(node, ast.Call):
                continue
            privileged_entrypoints = (
                _resolve_privileged_membership_scope_sql_entrypoints(
                    node.func,
                    assignments,
                    module_assignments,
                    aliases,
                    module_aliases,
                    guard_aliases,
                    module_guard_aliases,
                    wrapper_aliases,
                    module_wrapper_aliases,
                )
            )
            privileged_entrypoints.update(
                _resolve_privileged_membership_scope_sql_entrypoints(
                    node,
                    assignments,
                    module_assignments,
                    aliases,
                    module_aliases,
                    guard_aliases,
                    module_guard_aliases,
                    wrapper_aliases,
                    module_wrapper_aliases,
                )
            )
            function = _qualified_scope(scope, parents)
            indirect_entrypoints = privileged_entrypoints & {
                _DYNAMIC_PRIVILEGED_MEMBERSHIP_SCOPE_SQL,
                _WRAPPED_PRIVILEGED_MEMBERSHIP_SCOPE_SQL,
            }
            if indirect_entrypoints:
                raise AssertionError(
                    "Indirect membership-scope SQL capability caller at "
                    f"{relative_path}:{node.lineno} in {function}: "
                    f"{sorted(indirect_entrypoints)}."
                )
            if len(privileged_entrypoints) > 1:
                raise AssertionError(
                    "Ambiguous membership-scope SQL capability caller at "
                    f"{relative_path}:{node.lineno} in {function}: "
                    f"{sorted(privileged_entrypoints)}."
                )
            privileged_entrypoint = next(iter(privileged_entrypoints), None)
            call_name = privileged_entrypoint or _call_name(node)
            if call_name not in SQL_CALL_NAMES:
                continue
            if privileged_entrypoint is not None and (
                privileged_entrypoint,
                relative_path,
                function,
            ) not in APPROVED_PRIVILEGED_MEMBERSHIP_SCOPE_SQL_CALLERS:
                raise AssertionError(
                    "Unapproved membership-scope SQL capability caller "
                    f"{privileged_entrypoint} at {relative_path}:{node.lineno} "
                    f"in {function}."
                )
            sql_values: set[str] = set()
            for argument in _query_arguments(node, call_name=call_name):
                resolved = _resolve_strings(
                    argument,
                    assignments,
                    module_assignments,
                )
                if not resolved and _requires_static_resolution(
                    argument,
                    assignments,
                    module_assignments,
                ):
                    raise AssertionError(
                        "Unable to statically resolve SQL expression at "
                        f"{relative_path}:{node.lineno} in {function} "
                        f"for {call_name}(); use a resolvable SQL constant "
                        "or extend the AST resolver."
                    )
                sql_values.update(resolved)
            call_observed: dict[tuple[int, str], ObservedWrite] = {}
            for sql in sql_values:
                operations = _parse_operations(
                    sql, is_script=call_name == "executescript"
                )
                for statement_index, (verb, table, columns) in enumerate(
                    operations
                ):
                    if table == "users" and verb in {"INSERT", "UPDATE"}:
                        if "<dynamic>" not in columns and not (
                            PROFILE_VISIBLE_COLUMNS & set(columns)
                        ):
                            continue
                    elif table in MEMBERSHIP_TABLES or (
                        table in PARENT_SCOPE_TABLES and verb == "DELETE"
                    ):
                        pass
                    else:
                        continue
                    operation = _operation_label(verb, table, columns)
                    call_observed.setdefault(
                        (statement_index, operation),
                        ObservedWrite(
                            path=relative_path,
                            function=_qualified_scope(scope, parents),
                            operation=operation,
                            line=node.lineno,
                        ),
                    )
            observed.extend(call_observed.values())
    for imported, line in _privileged_membership_scope_sql_imports(
        tree,
        relative_path=relative_path,
        parents=parents,
    ):
        if imported not in APPROVED_PRIVILEGED_MEMBERSHIP_SCOPE_SQL_IMPORTS:
            entrypoint, local_name, _, function = imported
            raise AssertionError(
                "Unapproved membership-scope SQL capability import "
                f"{entrypoint} as {local_name} at {relative_path}:{line} "
                f"in {function}."
            )
    return tuple(sorted(observed))


def _scan_python_source(
    *,
    path: Path,
    source: str,
    repo_root: Path,
) -> tuple[ObservedWrite, ...]:
    tree = ast.parse(source, filename=str(path))
    return _scan_python_tree(path=path, tree=tree, repo_root=repo_root)


@cache
def _scan_sql_calls() -> tuple[ObservedWrite, ...]:
    return _scan_python_root(app_root=APP_ROOT, repo_root=REPO_ROOT)


def _scan_python_root(
    *,
    app_root: Path,
    repo_root: Path,
) -> tuple[ObservedWrite, ...]:
    observed: list[ObservedWrite] = []
    for path in sorted(app_root.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        observed.extend(
            _scan_python_source(
                path=path,
                source=source,
                repo_root=repo_root,
            )
        )
    return tuple(sorted(observed))


def _membership_scope_sql_helper_callers() -> tuple[tuple[str, str, str], ...]:
    observed: list[tuple[str, str, str]] = []
    for path in sorted(APP_ROOT.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        grouped, parents = _nodes_in_scope_and_parents(tree)
        relative_path = _relative_path(path)
        module_nodes = grouped.get(tree, [])
        module_aliases = _privileged_membership_scope_sql_import_aliases(
            module_nodes
        )
        module_guard_aliases = _profile_user_write_guard_module_aliases(
            module_nodes
        )
        module_wrapper_aliases = _privileged_wrapper_import_aliases(module_nodes)
        module_assignments: dict[str, list[ast.AST]] = defaultdict(list)
        for node in module_nodes:
            for name, value in _assignment_targets(node):
                module_assignments[name].append(value)
        for scope, nodes in grouped.items():
            assignments: dict[str, list[ast.AST]] = defaultdict(list)
            aliases = _privileged_membership_scope_sql_import_aliases(nodes)
            guard_aliases = _profile_user_write_guard_module_aliases(nodes)
            wrapper_aliases = _privileged_wrapper_import_aliases(nodes)
            for node in nodes:
                for name, value in _assignment_targets(node):
                    assignments[name].append(value)
            for node in nodes:
                if not isinstance(node, ast.Call):
                    continue
                entrypoints = _resolve_privileged_membership_scope_sql_entrypoints(
                    node.func,
                    assignments,
                    module_assignments,
                    aliases,
                    module_aliases,
                    guard_aliases,
                    module_guard_aliases,
                    wrapper_aliases,
                    module_wrapper_aliases,
                )
                entrypoints.update(
                    _resolve_privileged_membership_scope_sql_entrypoints(
                        node,
                        assignments,
                        module_assignments,
                        aliases,
                        module_aliases,
                        guard_aliases,
                        module_guard_aliases,
                        wrapper_aliases,
                        module_wrapper_aliases,
                    )
                )
                indirect_entrypoints = entrypoints & {
                    _DYNAMIC_PRIVILEGED_MEMBERSHIP_SCOPE_SQL,
                    _WRAPPED_PRIVILEGED_MEMBERSHIP_SCOPE_SQL,
                }
                if indirect_entrypoints:
                    raise AssertionError(
                        "Indirect membership-scope SQL capability caller at "
                        f"{relative_path}:{node.lineno} in "
                        f"{_qualified_scope(scope, parents)}: "
                        f"{sorted(indirect_entrypoints)}."
                    )
                observed.extend(
                    (
                        entrypoint,
                        relative_path,
                        _qualified_scope(scope, parents),
                    )
                    for entrypoint in entrypoints
                )
    return tuple(sorted(observed))


def _membership_scope_sql_helper_imports() -> tuple[tuple[str, str, str, str], ...]:
    observed: list[tuple[str, str, str, str]] = []
    for path in sorted(APP_ROOT.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        _, parents = _nodes_in_scope_and_parents(tree)
        observed.extend(
            imported
            for imported, _line in _privileged_membership_scope_sql_imports(
                tree,
                relative_path=_relative_path(path),
                parents=parents,
            )
        )
    return tuple(sorted(observed))


def _partition_inventory() -> dict[str, tuple[ObservedWrite, ...]]:
    groups: dict[str, list[ObservedWrite]] = defaultdict(list)
    for write in _scan_sql_calls():
        if write.path in OFFLINE_MIGRATION_PATHS:
            groups["excluded"].append(write)
        elif any(
            write.operation.startswith(f"{verb} users")
            for verb in ("INSERT", "UPDATE")
        ):
            groups["profile"].append(write)
        elif any(table in write.operation for table in MEMBERSHIP_TABLES):
            groups["membership"].append(write)
        else:
            groups["parent_delete"].append(write)
    return {name: tuple(values) for name, values in groups.items()}


def _format_expected(write: ObservedWrite) -> str:
    return (
        "ExpectedWrite("
        f"{write.path!r}, {write.function!r}, {write.operation!r}),"
    )


def _assert_inventory(
    *,
    label: str,
    observed: tuple[ObservedWrite, ...],
    expected: tuple[ExpectedWrite, ...],
) -> None:
    observed_counter = Counter(write.expected for write in observed)
    expected_counter = Counter(expected)
    if observed_counter == expected_counter:
        return

    remaining_expected = expected_counter.copy()
    unexpected_lines: list[str] = []
    for write in observed:
        if remaining_expected[write.expected] > 0:
            remaining_expected[write.expected] -= 1
        else:
            unexpected_lines.append(write.diagnostic())
    missing_counter = +remaining_expected
    message = [f"{label} inventory changed."]
    if unexpected_lines:
        message.append("Unexpected call sites:")
        message.extend(f"  {line}" for line in unexpected_lines)
    if missing_counter:
        message.append("Missing expected operations:")
        for write, count in sorted(missing_counter.items()):
            message.append(f"  {count}x {write}")
    message.append("Freeze this discovered inventory:")
    message.extend(f"    {_format_expected(write)}" for write in observed)
    raise AssertionError("\n".join(message))


def test_profile_visible_authnz_users_writer_inventory_is_frozen() -> None:
    inventory = _partition_inventory()
    forbidden = tuple(
        write
        for write in inventory.get("profile", ())
        if not (
            write.path
            == "tldw_Server_API/app/core/AuthNZ/profile_version.py"
            and write.function.startswith("VersionedUserWriteGateway.")
        )
    )
    assert not forbidden, "Forbidden profile-visible users writes:\n" + "\n".join(
        f"  {write.diagnostic()}" for write in forbidden
    )


def test_membership_dml_inventory_is_frozen() -> None:
    inventory = _partition_inventory()
    _assert_inventory(
        label="Membership DML writer",
        observed=inventory.get("membership", ()),
        expected=EXPECTED_MEMBERSHIP_WRITES,
    )


def test_membership_scope_sql_capability_callers_are_frozen() -> None:
    assert Counter(_membership_scope_sql_helper_callers()) == Counter(
        EXPECTED_PRIVILEGED_MEMBERSHIP_SCOPE_SQL_CALLERS
    )


def test_membership_scope_sql_capability_imports_are_frozen() -> None:
    assert Counter(_membership_scope_sql_helper_imports()) == Counter(
        EXPECTED_PRIVILEGED_MEMBERSHIP_SCOPE_SQL_IMPORTS
    )


def test_direct_membership_callers_supply_explicit_context() -> None:
    missing_context: list[str] = []
    wrong_context_category: list[str] = []
    for root in (APP_ROOT, REPO_ROOT / "tldw_Server_API" / "tests"):
        for path in sorted(root.rglob("*.py")):
            relative_path = _relative_path(path)
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            aliases = _direct_membership_import_aliases(tree)
            trusted_symbols = _trusted_membership_context_symbols(tree)
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                call_name = aliases.get(_call_name(node), _call_name(node))
                if call_name not in DIRECT_MEMBERSHIP_CALL_NAMES:
                    continue
                if (
                    relative_path,
                    _attribute_base_name(node),
                    call_name,
                ) in DIRECT_MEMBERSHIP_PROXY_CALLS:
                    continue
                context_keywords = [
                    keyword for keyword in node.keywords if keyword.arg == "context"
                ]
                if not context_keywords:
                    missing_context.append(f"{relative_path}:{node.lineno}")
                    continue
                expected_category = SERVING_MEMBERSHIP_CONTEXT_CATEGORIES.get(
                    relative_path
                )
                if expected_category is None:
                    continue
                observed_category = _membership_context_category(
                    context_keywords[0].value,
                    trusted_symbols=trusted_symbols,
                )
                expected_categories = (
                    expected_category
                    if isinstance(expected_category, frozenset)
                    else frozenset({expected_category})
                )
                if observed_category not in expected_categories:
                    wrong_context_category.append(
                        f"{relative_path}:{node.lineno} expected "
                        f"{sorted(expected_categories)}, found "
                        f"{observed_category or 'unknown'}"
                    )

    assert not missing_context, (
        "Direct membership calls must supply an explicit write context:\n  "
        + "\n  ".join(missing_context)
    )
    assert not wrong_context_category, (
        "Serving membership adapters must use the expected context category:\n  "
        + "\n  ".join(wrong_context_category)
    )


def test_shared_membership_writer_calls_supply_complete_explicit_contract() -> None:
    incomplete: list[str] = []
    required_keywords = {
        "conn",
        "context",
        "mutations",
        "anchor_ownership",
        "operation_time",
    }
    for path in sorted(APP_ROOT.rglob("*.py")):
        relative_path = _relative_path(path)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if _call_name(node) != "apply_membership_mutations":
                continue
            supplied = {keyword.arg for keyword in node.keywords}
            missing = sorted(required_keywords - supplied)
            if missing:
                incomplete.append(
                    f"{relative_path}:{node.lineno} missing {', '.join(missing)}"
                )
    assert not incomplete, (
        "Shared membership writer calls must supply the complete explicit contract:\n  "
        + "\n  ".join(incomplete)
    )


def test_runtime_trusted_membership_reasons_are_exact_and_path_owned() -> None:
    observed: dict[str, frozenset[str]] = {}
    for path in sorted(APP_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        reasons = _trusted_membership_reasons(tree)
        if reasons:
            observed[_relative_path(path)] = reasons

    assert observed == EXPECTED_TRUSTED_MEMBERSHIP_REASONS


def test_task8_membership_paths_do_not_import_work_package3_pipeline() -> None:
    task8_paths = (
        APP_ROOT / "core/AuthNZ/membership_writer.py",
        APP_ROOT / "core/AuthNZ/repos/orgs_teams_repo.py",
        APP_ROOT / "core/AuthNZ/federation/provisioning_service.py",
        APP_ROOT / "services/registration_service.py",
        APP_ROOT / "services/org_invite_service.py",
        APP_ROOT / "api/v1/endpoints/admin/admin_tenant_provisioning.py",
    )
    forbidden_modules = {
        "tldw_Server_API.app.core.UserProfiles.contracts",
        "tldw_Server_API.app.core.UserProfiles.effects",
        "tldw_Server_API.app.core.UserProfiles.executor",
        "tldw_Server_API.app.core.UserProfiles.planner",
    }
    leaked: list[str] = []
    for path in task8_paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module in forbidden_modules:
                leaked.append(f"{_relative_path(path)}:{node.lineno} {node.module}")
            elif isinstance(node, ast.Import):
                for imported in node.names:
                    if imported.name in forbidden_modules:
                        leaked.append(
                            f"{_relative_path(path)}:{node.lineno} {imported.name}"
                        )
    assert not leaked, "Work Package 3 imports leaked into Task 8:\n  " + "\n  ".join(
        leaked
    )


def test_parent_scope_delete_inventory_is_frozen() -> None:
    inventory = _partition_inventory()
    _assert_inventory(
        label="Membership parent-delete",
        observed=inventory.get("parent_delete", ()),
        expected=EXPECTED_PARENT_SCOPE_DELETES,
    )


def test_only_offline_migrations_are_excluded() -> None:
    inventory = _partition_inventory()
    excluded = inventory.get("excluded", ())
    for write in excluded:
        assert write.path in OFFLINE_MIGRATION_PATHS, write.diagnostic()
    _assert_inventory(
        label="Excluded offline/content-database write",
        observed=excluded,
        expected=EXPECTED_EXCLUDED_WRITES,
    )


def test_authnz_bootstrap_users_stored_routine_inventory_is_frozen() -> None:
    sqlite_schema = (
        REPO_ROOT
        / "tldw_Server_API/Databases/SQLite/Schema/sqlite_users.sql"
    ).read_text(encoding="utf-8")
    postgres_schema = (
        REPO_ROOT
        / "tldw_Server_API/Databases/Postgres/Schema/postgresql_users.sql"
    ).read_text(encoding="utf-8")

    assert _sqlite_users_routine_inventory(sqlite_schema) == (
        (
            "update_users_timestamp",
            "users",
            "CREATE TRIGGER update_users_timestamp AFTER UPDATE ON users "
            "FOR EACH ROW BEGIN UPDATE users SET updated_at = CURRENT_TIMESTAMP "
            "WHERE id = NEW.id; END",
        ),
    )
    assert _postgres_stored_routine_declarations(postgres_schema) == ()


def test_stored_routine_inventory_surfaces_future_users_writers() -> None:
    sqlite_schema = """
        CREATE TABLE users (id INTEGER PRIMARY KEY, email TEXT);
        CREATE TABLE source_rows (id INTEGER PRIMARY KEY);
        CREATE TRIGGER future_profile_write
        AFTER UPDATE ON source_rows
        BEGIN
            UPDATE users SET email = 'changed' WHERE id = NEW.id;
        END;
    """
    postgres_schema = """
        CREATE OR REPLACE FUNCTION future_profile_write() RETURNS trigger AS $$
        BEGIN
            UPDATE users SET email = 'changed' WHERE id = NEW.id;
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
    """

    sqlite_inventory = _sqlite_users_routine_inventory(sqlite_schema)
    assert sqlite_inventory[0][:2] == ("future_profile_write", "source_rows")
    assert _postgres_stored_routine_declarations(postgres_schema) == (
        ("function", "future_profile_write"),
    )


def _scan_fixture_source(
    tmp_path: Path,
    source: str,
) -> tuple[ObservedWrite, ...]:
    fixture_root = tmp_path / "fixture_app"
    source_path = fixture_root / "scanner_case.py"
    return _scan_python_source(
        path=source_path,
        source=textwrap.dedent(source).lstrip(),
        repo_root=tmp_path,
    )


def _scan_fixture_files(
    tmp_path: Path,
    sources: dict[str, str],
) -> tuple[ObservedWrite, ...]:
    fixture_root = tmp_path / "fixture_app"
    observed: list[ObservedWrite] = []
    for relative_path, source in sorted(sources.items()):
        observed.extend(
            _scan_python_source(
                path=fixture_root / relative_path,
                source=textwrap.dedent(source).lstrip(),
                repo_root=tmp_path,
            )
        )
    return tuple(sorted(observed))


def _assert_all_write_families(observed: tuple[ObservedWrite, ...]) -> None:
    assert {(write.function, write.operation) for write in observed} == {
        ("writes", "UPDATE users (email)"),
        ("writes", "INSERT org_members"),
        ("writes", "DELETE organizations"),
    }


def test_scanner_parses_every_executescript_statement(
    tmp_path: Path,
) -> None:
    observed = _scan_fixture_source(
        tmp_path,
        '''
        async def writes(db):
            await db.executescript(
                """\
                UPDATE users SET email = 'next@example.com' WHERE id = 1;
                INSERT INTO org_members (org_id, user_id) VALUES (1, 2);
                DELETE FROM organizations WHERE id = 1;
                """
            )
        ''',
    )

    _assert_all_write_families(observed)


def test_scanner_supports_project_execute_many(
    tmp_path: Path,
) -> None:
    observed = _scan_fixture_source(
        tmp_path,
        """
        async def writes(db):
            await db.execute_many(
                "UPDATE users SET email = ? WHERE id = ?", []
            )
            await db.execute_many(
                "INSERT INTO org_members (org_id, user_id) VALUES (?, ?)", []
            )
            await db.execute_many(
                "DELETE FROM organizations WHERE id = ?", []
            )
        """,
    )

    _assert_all_write_families(observed)


def test_scanner_resolves_subscripted_sql_constants(
    tmp_path: Path,
) -> None:
    observed = _scan_fixture_source(
        tmp_path,
        """
        SQL = {
            "profile": "UPDATE users SET email = ? WHERE id = ?",
            "membership": (
                "INSERT INTO org_members (org_id, user_id) VALUES (?, ?)"
            ),
            "parent": "DELETE FROM organizations WHERE id = ?",
        }

        async def writes(db):
            await db.execute(SQL["profile"])
            await db.execute(SQL["membership"])
            await db.execute(SQL["parent"])
        """,
    )

    _assert_all_write_families(observed)


def test_scanner_resolves_static_string_join_construction(
    tmp_path: Path,
) -> None:
    observed = _scan_fixture_source(
        tmp_path,
        """
        async def writes(db):
            await db.execute("".join((
                "UPDATE users SET ", "email = ? WHERE id = ?"
            )))
            await db.execute("".join((
                "INSERT INTO org_members ",
                "(org_id, user_id) VALUES (?, ?)",
            )))
            await db.execute("".join((
                "DELETE FROM ", "organizations WHERE id = ?"
            )))
        """,
    )

    _assert_all_write_families(observed)


def test_scanner_fails_closed_for_unresolved_static_query_expression(
    tmp_path: Path,
) -> None:
    with pytest.raises(AssertionError) as exc_info:
        _scan_fixture_source(
            tmp_path,
            """
            SQL_BY_KIND = {
                "profile": "UPDATE users SET email = ? WHERE id = ?",
            }

            async def writes(db, kind):
                await db.execute(SQL_BY_KIND[kind])
            """,
        )

    diagnostic = str(exc_info.value)
    assert "fixture_app/scanner_case.py:6" in diagnostic
    assert "writes" in diagnostic
    assert "execute" in diagnostic


def test_plain_unresolved_query_parameter_is_delegated_to_runtime_guard(
    tmp_path: Path,
) -> None:
    from tldw_Server_API.app.core.AuthNZ.profile_user_write_guard import (
        ProfileUserWriteRejected,
        _guard_sql,
    )

    observed = _scan_fixture_source(
        tmp_path,
        """
        async def writes(db, statement):
            await db.execute(statement)
        """,
    )
    assert observed == ()

    with pytest.raises(ProfileUserWriteRejected):
        _guard_sql(
            "UPDATE users SET email = ? WHERE id = ?",
            backend="sqlite",
            connection_identity=object(),
            operation="execute",
        )
    with pytest.raises(ProfileUserWriteRejected):
        _guard_sql(
            "DELETE FROM org_members WHERE org_id = ? AND user_id = ?",
            backend="sqlite",
            connection_identity=object(),
            operation="execute",
        )
    with pytest.raises(ProfileUserWriteRejected):
        _guard_sql(
            "DELETE FROM organizations WHERE id = ?",
            backend="sqlite",
            connection_identity=object(),
            operation="execute",
        )


def test_unapproved_membership_scope_sql_capability_caller_is_rejected(
    tmp_path: Path,
) -> None:
    with pytest.raises(AssertionError) as exc_info:
        _scan_fixture_source(
            tmp_path,
            """
            from tldw_Server_API.app.core.AuthNZ.profile_user_write_guard import (
                _execute_membership_scope_sql,
            )

            async def unapproved_writer(conn, statement):
                await _execute_membership_scope_sql(
                    conn,
                    statement,
                    backend="sqlite",
                )
            """,
        )

    diagnostic = str(exc_info.value)
    assert "fixture_app/scanner_case.py:6" in diagnostic
    assert "unapproved_writer" in diagnostic
    assert "Unapproved membership-scope SQL capability caller" in diagnostic


@pytest.mark.parametrize(
    ("source", "line", "entrypoint"),
    (
        (
            """
            from tldw_Server_API.app.core.AuthNZ.profile_user_write_guard import _execute_membership_scope_sql as execute_alias

            async def unapproved_writer(conn, statement):
                await execute_alias(conn, statement, backend="sqlite")
            """,
            4,
            "_execute_membership_scope_sql",
        ),
        (
            """
            from tldw_Server_API.app.core.AuthNZ.profile_user_write_guard import _execute_membership_scope_sql

            async def unapproved_writer(conn, statement):
                local_execute = _execute_membership_scope_sql
                await local_execute(conn, statement, backend="sqlite")
            """,
            5,
            "_execute_membership_scope_sql",
        ),
        (
            """
            from tldw_Server_API.app.core.AuthNZ.profile_user_write_guard import _mint_membership_scope_sql

            def unapproved_writer(statement, connection):
                return _mint_membership_scope_sql(statement, backend="sqlite", connection_identity=connection, execution_mode="execute")
            """,
            4,
            "_mint_membership_scope_sql",
        ),
        (
            """
            from tldw_Server_API.app.core.AuthNZ.profile_user_write_guard import _mint_membership_scope_sql as mint_alias

            def unapproved_writer(statement, connection):
                return mint_alias(statement, backend="sqlite", connection_identity=connection, execution_mode="execute")
            """,
            4,
            "_mint_membership_scope_sql",
        ),
        (
            """
            from tldw_Server_API.app.core.AuthNZ.profile_user_write_guard import _mint_membership_scope_sql

            def unapproved_writer(statement, connection):
                local_mint = _mint_membership_scope_sql
                return local_mint(statement, backend="sqlite", connection_identity=connection, execution_mode="execute")
            """,
            5,
            "_mint_membership_scope_sql",
        ),
    ),
)
def test_privileged_membership_scope_sql_aliases_are_rejected(
    tmp_path: Path,
    source: str,
    line: int,
    entrypoint: str,
) -> None:
    with pytest.raises(AssertionError) as exc_info:
        _scan_fixture_source(tmp_path, source)

    diagnostic = str(exc_info.value)
    assert f"fixture_app/scanner_case.py:{line}" in diagnostic
    assert "unapproved_writer" in diagnostic
    assert entrypoint in diagnostic


def test_privileged_membership_scope_sql_renamed_reexport_is_rejected(
    tmp_path: Path,
) -> None:
    with pytest.raises(AssertionError) as exc_info:
        _scan_fixture_files(
            tmp_path,
            {
                "bridge.py": """
                    from tldw_Server_API.app.core.AuthNZ.profile_user_write_guard import (
                        _execute_membership_scope_sql as forwarded_write,
                    )
                """,
                "consumer.py": """
                    from fixture_app.bridge import forwarded_write

                    async def unapproved_writer(conn, statement):
                        await forwarded_write(conn, statement, backend="sqlite")
                """,
            },
        )

    diagnostic = str(exc_info.value)
    assert "fixture_app/bridge.py:1" in diagnostic
    assert "_execute_membership_scope_sql" in diagnostic


@pytest.mark.parametrize(
    "source",
    (
        """
        import tldw_Server_API.app.core.AuthNZ.profile_user_write_guard as guard

        async def unapproved_writer(conn, statement):
            helper = getattr(guard, "_execute_membership_scope_sql")
            await helper(conn, statement, backend="sqlite")
        """,
        """
        import tldw_Server_API.app.core.AuthNZ.profile_user_write_guard as guard

        async def unapproved_writer(conn, statement, attribute_name):
            helper = getattr(guard, attribute_name)
            await helper(conn, statement, backend="sqlite")
        """,
        """
        import tldw_Server_API.app.core.AuthNZ.profile_user_write_guard as guard

        async def unapproved_writer(conn, statement, attribute_name):
            guard_alias = guard
            helper = getattr(guard_alias, attribute_name)
            await helper(conn, statement, backend="sqlite")
        """,
        """
        import functools
        import tldw_Server_API.app.core.AuthNZ.profile_user_write_guard as guard

        async def unapproved_writer(conn, statement):
            helper = functools.partial(
                guard._execute_membership_scope_sql,
                conn,
            )
            await helper(statement, backend="sqlite")
        """,
        """
        from functools import partial as bind
        import tldw_Server_API.app.core.AuthNZ.profile_user_write_guard as guard

        async def unapproved_writer(conn, statement):
            helper = bind(guard._execute_membership_scope_sql, conn)
            await helper(statement, backend="sqlite")
        """,
        """
        from builtins import getattr as lookup
        import tldw_Server_API.app.core.AuthNZ.profile_user_write_guard as guard

        async def unapproved_writer(conn, statement, attribute_name):
            helper = lookup(guard, attribute_name)
            await helper(conn, statement, backend="sqlite")
        """,
        """
        import tldw_Server_API.app.core.AuthNZ.profile_user_write_guard as guard

        async def unapproved_writer(conn, statement):
            helper = guard.__dict__["_execute_membership_scope_sql"]
            await helper(conn, statement, backend="sqlite")
        """,
        """
        import tldw_Server_API.app.core.AuthNZ.profile_user_write_guard as guard

        async def unapproved_writer(conn, statement):
            helper = vars(guard)["_execute_membership_scope_sql"]
            await helper(conn, statement, backend="sqlite")
        """,
        """
        import tldw_Server_API.app.core.AuthNZ.profile_user_write_guard as guard

        async def unapproved_writer(conn, statement):
            helper = guard.__dict__.get("_execute_membership_scope_sql")
            await helper(conn, statement, backend="sqlite")
        """,
        """
        from importlib import import_module

        async def unapproved_writer(conn, statement, attribute_name):
            guard = import_module(
                "tldw_Server_API.app.core.AuthNZ.profile_user_write_guard"
            )
            helper = getattr(guard, attribute_name)
            await helper(conn, statement, backend="sqlite")
        """,
        """
        async def unapproved_writer(conn, statement):
            guard = __import__(
                "tldw_Server_API.app.core.AuthNZ.profile_user_write_guard",
                fromlist=["_execute_membership_scope_sql"],
            )
            helper = getattr(guard, "_execute_membership_scope_sql")
            await helper(conn, statement, backend="sqlite")
        """,
        """
        async def unapproved_writer(conn, statement):
            importer = __import__
            guard = importer(
                "tldw_Server_API.app.core.AuthNZ.profile_user_write_guard",
                fromlist=["_execute_membership_scope_sql"],
            )
            helper = getattr(guard, "_execute_membership_scope_sql")
            await helper(conn, statement, backend="sqlite")
        """,
        """
        import tldw_Server_API.app.core.AuthNZ.profile_user_write_guard as guard

        async def unapproved_writer(conn, statement):
            helper = guard.__getattribute__("_execute_membership_scope_sql")
            await helper(conn, statement, backend="sqlite")
        """,
        """
        import tldw_Server_API.app.core.AuthNZ.profile_user_write_guard as guard

        async def unapproved_writer(conn, statement, attribute_name):
            helper = guard.__getattribute__(attribute_name)
            await helper(conn, statement, backend="sqlite")
        """,
        """
        import tldw_Server_API.app.core.AuthNZ.profile_user_write_guard as guard

        async def unapproved_writer(conn, statement):
            helper = object.__getattribute__(
                guard,
                "_execute_membership_scope_sql",
            )
            await helper(conn, statement, backend="sqlite")
        """,
        """
        import tldw_Server_API.app.core.AuthNZ.profile_user_write_guard as guard

        async def unapproved_writer(conn, statement, attribute_name):
            helper = object.__getattribute__(guard, attribute_name)
            await helper(conn, statement, backend="sqlite")
        """,
        """
        import tldw_Server_API.app.core.AuthNZ.profile_user_write_guard as guard

        async def unapproved_writer(conn, statement):
            helper = type(guard).__getattribute__(
                guard,
                "_execute_membership_scope_sql",
            )
            await helper(conn, statement, backend="sqlite")
        """,
        """
        import operator
        import tldw_Server_API.app.core.AuthNZ.profile_user_write_guard as guard

        async def unapproved_writer(conn, statement):
            helper = operator.attrgetter("_execute_membership_scope_sql")(guard)
            await helper(conn, statement, backend="sqlite")
        """,
        """
        from operator import attrgetter as select_attribute
        import tldw_Server_API.app.core.AuthNZ.profile_user_write_guard as guard

        async def unapproved_writer(conn, statement):
            selector = select_attribute("_execute_membership_scope_sql")
            helper = selector(guard)
            await helper(conn, statement, backend="sqlite")
        """,
        """
        import operator
        import tldw_Server_API.app.core.AuthNZ.profile_user_write_guard as guard

        async def unapproved_writer(conn, statement, attribute_name):
            helper = operator.attrgetter(attribute_name)(guard)
            await helper(conn, statement, backend="sqlite")
        """,
        """
        import operator
        import tldw_Server_API.app.core.AuthNZ.profile_user_write_guard as guard

        async def unapproved_writer(conn, statement):
            _, helper = operator.attrgetter(
                "__name__",
                "_execute_membership_scope_sql",
            )(guard)
            await helper(conn, statement, backend="sqlite")
        """,
        """
        import operator
        import tldw_Server_API.app.core.AuthNZ.profile_user_write_guard as guard

        async def unapproved_writer(conn, statement, attribute_name):
            _, helper = operator.attrgetter("__name__", attribute_name)(guard)
            await helper(conn, statement, backend="sqlite")
        """,
        """
        import operator
        import tldw_Server_API.app.core.AuthNZ.profile_user_write_guard as guard

        async def unapproved_writer(conn, statement):
            namespace = operator.attrgetter("__dict__")(guard)
            helper = namespace["_execute_membership_scope_sql"]
            await helper(conn, statement, backend="sqlite")
        """,
        """
        from operator import attrgetter as select_attribute
        import tldw_Server_API.app.core.AuthNZ.profile_user_write_guard as guard

        async def unapproved_writer(conn, statement):
            namespace_selector = select_attribute("__dict__")
            namespace = namespace_selector(guard)
            helper = namespace["_execute_membership_scope_sql"]
            await helper(conn, statement, backend="sqlite")
        """,
        """
        import operator
        import tldw_Server_API.app.core.AuthNZ.profile_user_write_guard as guard

        async def unapproved_writer(conn, statement):
            _, namespace = operator.attrgetter("__name__", "__dict__")(guard)
            helper = namespace["_execute_membership_scope_sql"]
            await helper(conn, statement, backend="sqlite")
        """,
        """
        import operator
        import tldw_Server_API.app.core.AuthNZ.profile_user_write_guard as guard

        async def unapproved_writer(conn, statement):
            lookup = operator.attrgetter("__dict__.get")(guard)
            helper = lookup("_execute_membership_scope_sql")
            await helper(conn, statement, backend="sqlite")
        """,
        """
        from operator import attrgetter as select_attribute
        import tldw_Server_API.app.core.AuthNZ.profile_user_write_guard as guard

        async def unapproved_writer(conn, statement):
            selector = select_attribute("__dict__.get")
            lookup = selector(guard)
            helper = lookup("_execute_membership_scope_sql")
            await helper(conn, statement, backend="sqlite")
        """,
        """
        import operator
        import tldw_Server_API.app.core.AuthNZ.profile_user_write_guard as guard

        async def unapproved_writer(conn, statement):
            _, lookup = operator.attrgetter("__name__", "__dict__.get")(guard)
            helper = lookup("_execute_membership_scope_sql")
            await helper(conn, statement, backend="sqlite")
        """,
        """
        import operator
        import tldw_Server_API.app.core.AuthNZ.profile_user_write_guard as guard

        async def unapproved_writer(conn, statement):
            lookup = operator.attrgetter("__getattribute__")(guard)
            helper = lookup("_execute_membership_scope_sql")
            await helper(conn, statement, backend="sqlite")
        """,
        """
        from operator import attrgetter as select_attribute
        import tldw_Server_API.app.core.AuthNZ.profile_user_write_guard as guard

        async def unapproved_writer(conn, statement):
            selector = select_attribute("__getattribute__")
            lookup = selector(guard)
            helper = lookup("_execute_membership_scope_sql")
            await helper(conn, statement, backend="sqlite")
        """,
        """
        import operator
        import tldw_Server_API.app.core.AuthNZ.profile_user_write_guard as guard

        async def unapproved_writer(conn, statement):
            _, lookup = operator.attrgetter(
                "__name__",
                "__class__.__getattribute__",
            )(guard)
            helper = lookup(guard, "_execute_membership_scope_sql")
            await helper(conn, statement, backend="sqlite")
        """,
        """
        import operator
        import tldw_Server_API.app.core.AuthNZ.profile_user_write_guard as guard

        async def unapproved_writer(conn, statement):
            helper = operator.attrgetter(
                "_execute_membership_scope_sql.__call__"
            )(guard)
            await helper(conn, statement, backend="sqlite")
        """,
    ),
)
def test_privileged_membership_scope_sql_dynamic_wrappers_are_rejected(
    tmp_path: Path,
    source: str,
) -> None:
    with pytest.raises(AssertionError) as exc_info:
        _scan_fixture_source(tmp_path, source)

    diagnostic = str(exc_info.value)
    assert "fixture_app/scanner_case.py" in diagnostic
    assert "unapproved_writer" in diagnostic


def test_global_scanner_rejects_split_dynamic_guard_and_helper_names(
    tmp_path: Path,
) -> None:
    fixture_root = tmp_path / "fixture_app"
    fixture_root.mkdir()
    (fixture_root / "split_dynamic.py").write_text(
        textwrap.dedent(
            """
            from importlib import import_module

            async def unapproved_writer(conn, statement):
                module_name = (
                    "tldw_Server_API.app.core.AuthNZ.profile_user_"
                    + "write_guard"
                )
                helper_name = "_execute_membership_" + "scope_sql"
                guard = import_module(module_name)
                helper = getattr(guard, helper_name)
                await helper(conn, statement, backend="sqlite")
            """
        ).lstrip(),
        encoding="utf-8",
    )

    with pytest.raises(AssertionError):
        _scan_python_root(app_root=fixture_root, repo_root=tmp_path)


def test_scanner_fails_closed_through_container_alias_and_cycle(
    tmp_path: Path,
) -> None:
    with pytest.raises(AssertionError) as exc_info:
        _scan_fixture_source(
            tmp_path,
            """
            SQL_BY_KIND = {
                "profile": "UPDATE users SET email = ? WHERE id = ?",
            }
            ALIASED_SQL = SQL_BY_KIND
            ALIASED_SQL = ALIASED_SQL

            async def writes(db, kind):
                await db.execute(ALIASED_SQL[kind])
            """,
        )

    diagnostic = str(exc_info.value)
    assert "fixture_app/scanner_case.py:8" in diagnostic
    assert "writes" in diagnostic
    assert "execute" in diagnostic


def test_scanner_preserves_identical_executescript_statement_multiplicity(
    tmp_path: Path,
) -> None:
    observed = _scan_fixture_source(
        tmp_path,
        '''
        async def writes(db):
            await db.executescript(
                """\
                UPDATE users SET email = 'next@example.com' WHERE id = 1;
                UPDATE users SET email = 'next@example.com' WHERE id = 1;
                """
            )
        ''',
    )

    assert [write.operation for write in observed] == [
        "UPDATE users (email)",
        "UPDATE users (email)",
    ]


def test_scanner_deduplicates_identical_sql_from_resolver_paths(
    tmp_path: Path,
) -> None:
    observed = _scan_fixture_source(
        tmp_path,
        """
        PROFILE_SQL = "UPDATE users SET email = ? WHERE id = ?"
        PROFILE_SQL_ALIAS = "UPDATE users SET email = $1 WHERE id = $2"

        async def writes(db, use_alias):
            query = PROFILE_SQL_ALIAS if use_alias else PROFILE_SQL
            await db.execute(query)
        """,
    )

    assert [write.operation for write in observed] == ["UPDATE users (email)"]


def test_inventory_diagnostic_reports_only_excess_duplicate_location() -> None:
    expected = ExpectedWrite("writers.py", "write_profile", "UPDATE users (email)")
    observed = tuple(
        ObservedWrite(
            path=expected.path,
            function=expected.function,
            operation=expected.operation,
            line=line,
        )
        for line in (10, 20, 30)
    )

    with pytest.raises(AssertionError) as exc_info:
        _assert_inventory(
            label="Duplicate diagnostic",
            observed=observed,
            expected=(expected, expected),
        )

    diagnostic = str(exc_info.value)
    assert "writers.py:30" in diagnostic
    assert "writers.py:10" not in diagnostic
    assert "writers.py:20" not in diagnostic
