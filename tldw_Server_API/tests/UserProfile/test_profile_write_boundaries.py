from __future__ import annotations

import ast
import io
import re
import sqlite3
import textwrap
import tokenize
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
    "tldw_Server_API/app/api/v1/endpoints/orgs.py": "actor",
    "tldw_Server_API/app/services/admin_e2e_support_service.py": "trusted",
    "tldw_Server_API/app/services/admin_orgs_service.py": "actor",
    "tldw_Server_API/app/services/org_invite_service.py": "trusted",
    "tldw_Server_API/app/core/AuthNZ/federation/provisioning_service.py": "trusted",
    "tldw_Server_API/app/core/AuthNZ/orgs_teams.py": "passthrough",
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
    }
)
_SQL_CALL_IDENTIFIER_RE = re.compile(
    rf"\b(?:{'|'.join(sorted(map(re.escape, SQL_CALL_NAMES)))})\b"
)
_IGNORED_CALL_TOKENS = frozenset(
    {
        tokenize.NL,
        tokenize.NEWLINE,
        tokenize.INDENT,
        tokenize.DEDENT,
        tokenize.COMMENT,
    }
)
OFFLINE_MIGRATION_PATHS = frozenset(
    {
        "tldw_Server_API/app/core/AuthNZ/migrations.py",
        "tldw_Server_API/app/core/AuthNZ/pg_migrations_extra.py",
        "tldw_Server_API/app/core/AuthNZ/migrate_to_multiuser.py",
    }
)
UNRELATED_CONTENT_DATABASE_PATHS = frozenset(
    {
        "tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py",
        "tldw_Server_API/app/core/DB_Management/Prompts_DB.py",
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
        "tldw_Server_API/app/api/v1/endpoints/admin/admin_tenant_provisioning.py",
        "provision_tenant",
        "INSERT org_members",
    ),
    ExpectedWrite(
        "tldw_Server_API/app/api/v1/endpoints/admin/admin_tenant_provisioning.py",
        "provision_tenant",
        "INSERT org_members",
    ),
    ExpectedWrite(
        "tldw_Server_API/app/core/AuthNZ/repos/orgs_teams_repo.py",
        "AuthnzOrgsTeamsRepo._ensure_user_in_default_team",
        "INSERT team_members",
    ),
    ExpectedWrite(
        "tldw_Server_API/app/core/AuthNZ/repos/orgs_teams_repo.py",
        "AuthnzOrgsTeamsRepo._ensure_user_in_default_team",
        "INSERT team_members",
    ),
    ExpectedWrite(
        "tldw_Server_API/app/core/AuthNZ/repos/orgs_teams_repo.py",
        "AuthnzOrgsTeamsRepo._remove_user_from_default_team",
        "DELETE team_members",
    ),
    ExpectedWrite(
        "tldw_Server_API/app/core/AuthNZ/repos/orgs_teams_repo.py",
        "AuthnzOrgsTeamsRepo._remove_user_from_default_team",
        "DELETE team_members",
    ),
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
        "tldw_Server_API/app/core/AuthNZ/repos/orgs_teams_repo.py",
        "AuthnzOrgsTeamsRepo.transfer_organization_ownership",
        "UPDATE org_members",
    ),
    ExpectedWrite(
        "tldw_Server_API/app/core/AuthNZ/repos/orgs_teams_repo.py",
        "AuthnzOrgsTeamsRepo.transfer_organization_ownership",
        "UPDATE org_members",
    ),
    ExpectedWrite(
        "tldw_Server_API/app/core/AuthNZ/repos/orgs_teams_repo.py",
        "AuthnzOrgsTeamsRepo.transfer_organization_ownership",
        "UPDATE org_members",
    ),
    ExpectedWrite(
        "tldw_Server_API/app/core/AuthNZ/repos/orgs_teams_repo.py",
        "AuthnzOrgsTeamsRepo.transfer_organization_ownership",
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
    ExpectedWrite(
        "tldw_Server_API/app/services/registration_service.py",
        "RegistrationService._ensure_org_membership",
        "INSERT org_members",
    ),
    ExpectedWrite(
        "tldw_Server_API/app/services/registration_service.py",
        "RegistrationService._ensure_org_membership",
        "INSERT org_members",
    ),
    ExpectedWrite(
        "tldw_Server_API/app/services/registration_service.py",
        "RegistrationService._ensure_org_membership",
        "INSERT team_members",
    ),
    ExpectedWrite(
        "tldw_Server_API/app/services/registration_service.py",
        "RegistrationService._ensure_org_membership",
        "INSERT team_members",
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

EXPECTED_EXCLUDED_WRITES = (
    ExpectedWrite(
        "tldw_Server_API/app/core/AuthNZ/migrations.py",
        "migration_025_team_members_added_at",
        "UPDATE team_members",
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
    if call_name in ACTOR_MEMBERSHIP_CONTEXT_FACTORIES:
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


def _contains_sql_call_candidate(source: str) -> bool:
    if _SQL_CALL_IDENTIFIER_RE.search(source) is None:
        return False
    pending_call_name = False
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if pending_call_name:
            if token.type in _IGNORED_CALL_TOKENS or (
                token.type == tokenize.OP and token.string == ")"
            ):
                continue
            if token.type == tokenize.OP and token.string == "(":
                return True
            pending_call_name = False
        if token.type == tokenize.NAME and token.string in SQL_CALL_NAMES:
            pending_call_name = True
    return False


def _query_arguments(node: ast.Call) -> tuple[ast.AST, ...]:
    keyword_arguments = tuple(
        keyword.value
        for keyword in node.keywords
        if keyword.arg in {"query", "sql", "statement"}
    )
    if keyword_arguments:
        return keyword_arguments
    argument_index = 1 if _call_name(node) == "_execute_compat" else 0
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
    for node in grouped.get(tree, []):
        for name, value in _assignment_targets(node):
            module_assignments[name].append(value)

    for scope, nodes in grouped.items():
        assignments: dict[str, list[ast.AST]] = defaultdict(list)
        for node in nodes:
            for name, value in _assignment_targets(node):
                assignments[name].append(value)
        for node in nodes:
            if not isinstance(node, ast.Call):
                continue
            call_name = _call_name(node)
            if call_name not in SQL_CALL_NAMES:
                continue
            sql_values: set[str] = set()
            for argument in _query_arguments(node):
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
                    function = _qualified_scope(scope, parents)
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
    observed: list[ObservedWrite] = []
    for path in sorted(APP_ROOT.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        if not _contains_sql_call_candidate(source):
            continue
        observed.extend(
            _scan_python_source(
                path=path,
                source=source,
                repo_root=REPO_ROOT,
            )
        )
    return tuple(sorted(observed))


def _partition_inventory() -> dict[str, tuple[ObservedWrite, ...]]:
    groups: dict[str, list[ObservedWrite]] = defaultdict(list)
    for write in _scan_sql_calls():
        if write.path in (
            OFFLINE_MIGRATION_PATHS | UNRELATED_CONTENT_DATABASE_PATHS
        ):
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
                if observed_category != expected_category:
                    wrong_context_category.append(
                        f"{relative_path}:{node.lineno} expected "
                        f"{expected_category}, found {observed_category or 'unknown'}"
                    )

    assert not missing_context, (
        "Direct membership calls must supply an explicit write context:\n  "
        + "\n  ".join(missing_context)
    )
    assert not wrong_context_category, (
        "Serving membership adapters must use the expected context category:\n  "
        + "\n  ".join(wrong_context_category)
    )


def test_parent_scope_delete_inventory_is_frozen() -> None:
    inventory = _partition_inventory()
    _assert_inventory(
        label="Membership parent-delete",
        observed=inventory.get("parent_delete", ()),
        expected=EXPECTED_PARENT_SCOPE_DELETES,
    )


def test_only_offline_migrations_or_content_databases_are_excluded() -> None:
    inventory = _partition_inventory()
    excluded = inventory.get("excluded", ())
    for write in excluded:
        assert (
            write.path in OFFLINE_MIGRATION_PATHS
            or write.path in UNRELATED_CONTENT_DATABASE_PATHS
        ), write.diagnostic()
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
