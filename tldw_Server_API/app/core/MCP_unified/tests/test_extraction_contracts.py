from __future__ import annotations

import ast
import asyncio
import contextlib
import inspect
import re
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any, get_type_hints

import pytest

MCP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[5]
STANDALONE_MCP_ROOT = REPO_ROOT / "apps" / "mcp-unified" / "src" / "mcp_unified"
MCP_PACKAGE = "tldw_Server_API.app.core.MCP_unified"
EXPECTED_INTERFACE_FILES = {"runtime.py", "policy.py", "storage.py"}
EXPECTED_FAILURE_SYMBOLS = {"ExpectedToolFailure", "ExpectedToolFailureReason"}
EXPECTED_FAILURE_REASON_CODES = {
    "dependency_unavailable",
    "idempotency_in_progress",
    "idempotency_unavailable",
    "rate_limit_unavailable",
    "stale_prepared_call",
}
DOMAIN_BRANCH_TOKENS = {"model", "models", "provider", "providers", "skill", "skills"}
IDENTIFIER_WORD_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z]|\d|$)|[A-Z]?[a-z]+|\d+")
LOGGER_METHODS = {
    "bind",
    "contextualize",
    "critical",
    "debug",
    "error",
    "exception",
    "info",
    "log",
    "opt",
    "patch",
    "success",
    "trace",
    "warning",
}
SAFE_EXCEPTION_LOG_HELPERS = {
    "_safe_error_type",
    "_safe_exception_family",
    "get_expected_tool_failure_reason",
}
LEXICAL_SCOPE_NODES = (
    ast.Module,
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.Lambda,
    ast.ClassDef,
)
RUNTIME_POLICY_FIELD_PATHS = {
    "effect": ("policy", "effect"),
    "rate_limit_category": ("policy", "rate_limit_category"),
    "rate_limit_fail_closed": ("policy", "rate_limit_fail_closed"),
    "inject_argument": ("policy", "idempotency", "inject_argument"),
    "ttl_seconds": ("policy", "idempotency", "ttl_seconds"),
    "contention_wait_seconds": ("policy", "idempotency", "contention_wait_seconds"),
    "finalize_seconds": ("policy", "idempotency", "finalize_seconds"),
    "lock_ttl_seconds": ("policy", "idempotency", "lock_ttl_seconds"),
    "max_entries": ("policy", "idempotency", "max_entries"),
    "max_result_bytes": ("policy", "idempotency", "max_result_bytes"),
}
MUTABLE_EXECUTION_FIELDS = set(RUNTIME_POLICY_FIELD_PATHS) | {
    "category",
    "idempotency_cache_size",
    "idempotency_finalize_seconds",
    "idempotency_result_max_bytes",
    "idempotency_ttl_seconds",
    "idempotency_wait_seconds",
    "inputSchema",
    "input_schema",
    "is_write",
    "module_timeout",
    "timeout_seconds",
    "uses_network",
}


def _fake_runtime_dependencies() -> SimpleNamespace:
    lifecycle_guard = _RecordingLifecycleGuard()
    return SimpleNamespace(
        module_registry=object(),
        rbac_policy=object(),
        rate_limiter=object(),
        metrics_collector=object(),
        telemetry_provider=object(),
        database_path_resolver=object(),
        tool_catalog_provider=object(),
        api_key_scope_normalizer=object(),
        effective_policy_resolver=object(),
        approval_evaluator=object(),
        path_scope_enforcer=object(),
        external_access_evaluator=object(),
        redis_client_factory=object(),
        circuit_breaker_factory=object(),
        auth_provider=_FakeAuthProvider(),
        lifecycle_guard=lifecycle_guard,
        permission_seeder=object(),
        module_config_provider=object(),
        policy_context_provider=object(),
        environment_flags_provider=_FakeEnvironmentFlagsProvider(),
        websocket_stream_factory=_RecordingWebSocketStreamFactory(),
    )


class _RecordingLifecycleGuard:
    """Lifecycle guard double that records shutdown-family registrations."""

    def __init__(self) -> None:
        self.registered: list[dict[str, Any]] = []

    def register_shutdown_transport_family(self, family: str, **kwargs: Any) -> None:
        self.registered.append({"family": family, **kwargs})

    def assert_may_start_work(self, app: Any, family: str) -> None:
        del app, family


class _RecordingModuleRegistry:
    """Small module registry double for MCPServer dependency-injection tests."""

    def __init__(self, events: list[str] | None = None) -> None:
        self.started = 0
        self.registrations: list[dict[str, Any]] = []
        self.events = events

    async def start_health_monitoring(self) -> None:
        self.started += 1

    async def register_module(self, module_id: str, module_type: type[Any], config: Any) -> None:
        self.registrations.append(
            {"module_id": module_id, "module_type": module_type, "config": config}
        )

    async def shutdown_all(self) -> None:
        if self.events is not None:
            self.events.append("modules")


class _ExoticShutdownError(Exception):
    def __getattribute__(self, name: str) -> Any:
        if name == "__class__":
            raise RuntimeError("hostile shutdown exception class access")
        return super().__getattribute__(name)


class _NoopMetrics:
    """Metrics double with async lifecycle methods used by server tests."""

    async def start_collection(self) -> None:
        return None

    async def stop_collection(self) -> None:
        return None

    def __getattr__(self, _name: str) -> Callable[..., None]:
        return lambda *args, **kwargs: None


class _RecordingPermissionSeeder:
    """Permission seeder double that records whether initialize invoked it."""

    def __init__(self) -> None:
        self.calls = 0

    async def seed_default_tool_permissions(self) -> None:
        self.calls += 1


class _FakeModuleConfigProvider:
    """Module config provider double with deterministic default paths."""

    def __init__(self) -> None:
        self.media_db_path = "/tmp/mcp-stage3-media.db"

    def default_media_db_path(self) -> str:
        return self.media_db_path


class _FakePolicyContextProvider:
    """Policy-context provider double with a caller-controlled flag."""

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled

    def is_policy_context_enabled(self) -> bool:
        return self.enabled


class _FakeEnvironmentFlagsProvider:
    """Environment flag provider double with caller-controlled results."""

    def __init__(
        self,
        *,
        flags: dict[str, bool] | None = None,
        test_mode: bool = False,
        explicit_pytest_runtime: bool = False,
        truthy_values: dict[Any, bool] | None = None,
    ) -> None:
        self.flags = flags or {}
        self.test_mode = test_mode
        self.explicit_pytest_runtime = explicit_pytest_runtime
        self.truthy_values = truthy_values or {}
        self.flag_calls: list[str] = []
        self.truthy_calls: list[Any] = []

    def env_flag_enabled(self, name: str) -> bool:
        self.flag_calls.append(name)
        return self.flags.get(name, False)

    def is_test_mode(self) -> bool:
        return self.test_mode

    def is_explicit_pytest_runtime(self) -> bool:
        return self.explicit_pytest_runtime

    def is_truthy(self, value: Any) -> bool:
        self.truthy_calls.append(value)
        return self.truthy_values.get(value, False)


class _FakeWebSocketStream:
    """Sentinel stream object returned by the injected WebSocket stream factory."""

    def __init__(self) -> None:
        self.ws = SimpleNamespace(close=lambda *args, **kwargs: None)
        self.sent: list[dict[str, Any]] = []
        self.started = False
        self.stopped = False
        self.activity_marks = 0

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    def mark_activity(self) -> None:
        self.activity_marks += 1

    async def send_json(self, payload: dict[str, Any]) -> None:
        self.sent.append(payload)


class _RecordingWebSocketStreamFactory:
    """WebSocket stream factory double that records construction calls."""

    def __init__(self) -> None:
        self.stream = _FakeWebSocketStream()
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self,
        websocket: Any,
        *,
        heartbeat_interval_s: float | None,
        idle_timeout_s: float | None,
        close_on_done: bool,
        labels: dict[str, str],
    ) -> _FakeWebSocketStream:
        self.calls.append(
            {
                "websocket": websocket,
                "heartbeat_interval_s": heartbeat_interval_s,
                "idle_timeout_s": idle_timeout_s,
                "close_on_done": close_on_done,
                "labels": labels,
            }
        )
        return self.stream


class _FakeAuthProvider:
    """Auth provider double for API-key permission normalization."""

    def get_mcp_jwt_manager(self) -> object:
        return object()

    def is_authnz_access_token(self, _token: str) -> bool:
        return False

    async def authenticate_authnz_websocket_token(
        self,
        _token: str,
        *,
        websocket: Any,
    ) -> None:
        del websocket
        return None

    async def validate_api_key(
        self,
        _api_key: str,
        *,
        ip_address: str | None = None,
    ) -> None:
        del ip_address
        return None

    def normalize_api_key_permissions(self, info: dict[str, Any] | None) -> list[str]:
        return ["fake.scope"] if info else []


class _TelemetryManagerDouble:
    """Telemetry manager double that exposes trace_context calls."""

    def __init__(self, label: str) -> None:
        self.label = label
        self.trace_calls: list[tuple[str, dict[str, Any] | None]] = []

    def trace_context(
        self,
        operation_name: str,
        attributes: dict[str, Any] | None = None,
    ) -> Any:
        self.trace_calls.append((operation_name, attributes))
        return contextlib.nullcontext(self.label)


def _server_runtime_dependencies() -> SimpleNamespace:
    deps = _fake_runtime_dependencies()
    deps.module_registry = _RecordingModuleRegistry()
    deps.metrics_collector = _NoopMetrics()
    deps.permission_seeder = _RecordingPermissionSeeder()
    deps.module_config_provider = _FakeModuleConfigProvider()
    deps.policy_context_provider = _FakePolicyContextProvider(enabled=False)
    deps.auth_provider = _FakeAuthProvider()
    return deps


def _real_server_runtime_dependencies() -> Any:
    from mcp_unified.interfaces.runtime import MCPRuntimeDependencies

    return MCPRuntimeDependencies(**vars(_server_runtime_dependencies()))


def _interface_boundary_violations_for(path: Path, interface_dir: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    relative_parent = path.parent.relative_to(interface_dir)
    relative_depth = 0 if relative_parent == Path(".") else len(relative_parent.parts)
    max_interface_relative_level = relative_depth + 1
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            violations.extend(
                alias.name
                for alias in node.names
                if alias.name == "tldw_Server_API"
                or alias.name.startswith("tldw_Server_API.")
            )
        elif isinstance(node, ast.ImportFrom):
            if node.level > max_interface_relative_level:
                violations.append(
                    f"relative import escapes interfaces package: level={node.level}"
                )
            elif (
                node.module
                and node.level == 0
                and (
                    node.module == "tldw_Server_API"
                    or node.module.startswith("tldw_Server_API.")
                )
            ):
                violations.append(node.module)
    return violations


def _resolve_import_from_source(package: str, module: str | None, level: int) -> str:
    """Return an absolute module path for an ``ast.ImportFrom`` source."""
    if level == 0:
        return module or ""

    package_parts = package.split(".")
    base_parts = package_parts[: len(package_parts) - level + 1]
    if module:
        base_parts.extend(module.split("."))
    return ".".join(base_parts)


def _resolved_import_sources_for(
    path: Path,
    package: str,
    *,
    top_level_only: bool = False,
) -> list[str]:
    """Return import module sources from Python AST, resolving relative imports."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    nodes = tree.body if top_level_only else ast.walk(tree)
    imports: list[str] = []
    for node in nodes:
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(_resolve_import_from_source(package, node.module, node.level))
    return imports


def _attribute_path(node: ast.AST) -> tuple[str, ...]:
    """Return a dotted name/attribute path when an expression has one."""

    parts: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if not isinstance(current, ast.Name):
        return ()
    parts.append(current.id)
    return tuple(reversed(parts))


def _display_node(node: ast.AST) -> str:
    return ast.unparse(node)


def _node_mentions_symbols(node: ast.AST, symbols: set[str]) -> bool:
    return any(
        (isinstance(candidate, ast.Name) and candidate.id in symbols)
        or (isinstance(candidate, ast.Attribute) and candidate.attr in symbols)
        for candidate in ast.walk(node)
    )


def _lexical_scope_nodes(scope: ast.AST) -> tuple[list[ast.AST], list[ast.AST]]:
    body = [scope.body] if isinstance(scope, ast.Lambda) else list(scope.body)
    nodes: list[ast.AST] = []
    nested_scopes: list[ast.AST] = []
    pending = list(reversed(body))
    while pending:
        node = pending.pop()
        nodes.append(node)
        if isinstance(node, LEXICAL_SCOPE_NODES):
            nested_scopes.append(node)
            continue
        pending.extend(reversed(list(ast.iter_child_nodes(node))))
    return nodes, nested_scopes


def _lexically_bound_names(scope: ast.AST) -> set[str]:
    names: set[str] = set()
    external_names: set[str] = set()
    if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
        arguments = scope.args
        names.update(
            argument.arg
            for argument in [
                *arguments.posonlyargs,
                *arguments.args,
                *arguments.kwonlyargs,
            ]
        )
        if arguments.vararg is not None:
            names.add(arguments.vararg.arg)
        if arguments.kwarg is not None:
            names.add(arguments.kwarg.arg)

    scope_nodes, _ = _lexical_scope_nodes(scope)
    for node in scope_nodes:
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
        elif isinstance(node, ast.alias):
            names.add(node.asname or node.name.split(".", maxsplit=1)[0])
        elif (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            or isinstance(node, ast.ExceptHandler)
            and isinstance(node.name, str)
            or isinstance(node, (ast.MatchAs, ast.MatchStar))
            and node.name
        ):
            names.add(node.name)
        elif isinstance(node, ast.MatchMapping) and node.rest:
            names.add(node.rest)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            external_names.update(node.names)
    return names - external_names


def _stored_names(target: ast.AST) -> set[str]:
    return {
        candidate.id
        for candidate in ast.walk(target)
        if isinstance(candidate, ast.Name) and isinstance(candidate.ctx, ast.Store)
    }


def _assignment_pairs(target: ast.AST, value: ast.AST) -> list[tuple[ast.AST, ast.AST]]:
    if (
        isinstance(target, (ast.Tuple, ast.List))
        and isinstance(value, (ast.Tuple, ast.List))
        and len(target.elts) == len(value.elts)
    ):
        return [
            pair
            for child_target, child_value in zip(
                target.elts,
                value.elts,
                strict=True,
            )
            for pair in _assignment_pairs(child_target, child_value)
        ]
    return [(target, value)]


def _contains_expected_error_result(
    nodes: list[ast.AST],
    aliases: set[str] | None = None,
) -> bool:
    error_result_aliases = aliases or set()
    for node in nodes:
        if _node_mentions_symbols(node, EXPECTED_FAILURE_SYMBOLS):
            return True
        pending = [node]
        while pending:
            candidate = pending.pop()
            if isinstance(candidate, ast.Name):
                if candidate.id in error_result_aliases or candidate.id in {
                    "_complete_expected_failure",
                    "_expected_failure_payload",
                }:
                    return True
            if isinstance(candidate, ast.Constant) and candidate.value in EXPECTED_FAILURE_REASON_CODES:
                return True
            if isinstance(candidate, ast.Dict):
                for key, value in zip(candidate.keys, candidate.values, strict=True):
                    if (
                        isinstance(key, ast.Constant)
                        and key.value == "isError"
                        and isinstance(value, ast.Constant)
                        and value.value is True
                    ):
                        return True
            if not isinstance(candidate, LEXICAL_SCOPE_NODES):
                pending.extend(ast.iter_child_nodes(candidate))
    return False


def _expected_error_result_aliases_by_node(tree: ast.Module) -> dict[int, set[str]]:
    aliases_by_node: dict[int, set[str]] = {}

    def collect_scope(scope: ast.AST, inherited_aliases: set[str]) -> None:
        scope_nodes, nested_scopes = _lexical_scope_nodes(scope)
        assignments: list[tuple[list[ast.AST], ast.AST]] = []
        for node in scope_nodes:
            if isinstance(node, ast.Assign):
                assignments.append((node.targets, node.value))
            elif isinstance(node, (ast.AnnAssign, ast.NamedExpr)) and node.value is not None:
                assignments.append(([node.target], node.value))

        aliases: set[str] = set()
        changed = True
        while changed:
            changed = False
            for targets, value in assignments:
                for target in targets:
                    target_pairs = [(target, value)]
                    while target_pairs:
                        candidate_target, candidate_value = target_pairs.pop()
                        if (
                            isinstance(candidate_target, (ast.Tuple, ast.List))
                            and isinstance(candidate_value, (ast.Tuple, ast.List))
                            and len(candidate_target.elts) == len(candidate_value.elts)
                        ):
                            target_pairs.extend(
                                zip(
                                    candidate_target.elts,
                                    candidate_value.elts,
                                    strict=True,
                                )
                            )
                            continue
                        if not _contains_expected_error_result(
                            [candidate_value],
                            aliases,
                        ):
                            continue
                        new_aliases = _stored_names(candidate_target) - aliases
                        if new_aliases:
                            aliases.update(new_aliases)
                            changed = True

        visible_aliases = (
            inherited_aliases - _lexically_bound_names(scope)
        ) | aliases
        for node in scope_nodes:
            aliases_by_node[id(node)] = visible_aliases
        for nested_scope in nested_scopes:
            collect_scope(nested_scope, visible_aliases)

    collect_scope(tree, set())
    return aliases_by_node


def _expected_failure_symbol_aliases_by_node(tree: ast.Module) -> dict[int, set[str]]:
    aliases_by_node: dict[int, set[str]] = {}

    def collect_scope(scope: ast.AST, inherited_aliases: set[str]) -> None:
        scope_nodes, nested_scopes = _lexical_scope_nodes(scope)
        aliases: set[str] = set()
        assignments: list[tuple[ast.AST, ast.AST]] = []
        for node in scope_nodes:
            if isinstance(node, ast.ImportFrom):
                aliases.update(
                    alias.asname or alias.name
                    for alias in node.names
                    if alias.name in EXPECTED_FAILURE_SYMBOLS
                )
            elif isinstance(node, ast.Assign):
                assignments.extend((target, node.value) for target in node.targets)
            elif isinstance(node, (ast.AnnAssign, ast.NamedExpr)) and node.value is not None:
                assignments.append((node.target, node.value))

        changed = True
        while changed:
            changed = False
            visible_symbols = EXPECTED_FAILURE_SYMBOLS | inherited_aliases | aliases
            for target, value in assignments:
                target_pairs = [(target, value)]
                while target_pairs:
                    candidate_target, candidate_value = target_pairs.pop()
                    if (
                        isinstance(candidate_target, (ast.Tuple, ast.List))
                        and isinstance(candidate_value, (ast.Tuple, ast.List))
                        and len(candidate_target.elts) == len(candidate_value.elts)
                    ):
                        target_pairs.extend(
                            zip(
                                candidate_target.elts,
                                candidate_value.elts,
                                strict=True,
                            )
                        )
                        continue
                    path = _attribute_path(candidate_value)
                    if not (
                        isinstance(candidate_value, ast.Name)
                        and candidate_value.id in visible_symbols
                        or path
                        and path[-1] in visible_symbols
                    ):
                        continue
                    new_aliases = _stored_names(candidate_target) - aliases
                    if new_aliases:
                        aliases.update(new_aliases)
                        changed = True

        visible_aliases = (
            inherited_aliases - _lexically_bound_names(scope)
        ) | aliases
        for node in scope_nodes:
            aliases_by_node[id(node)] = visible_aliases
        for nested_scope in nested_scopes:
            collect_scope(nested_scope, visible_aliases)

    collect_scope(tree, set())
    return aliases_by_node


def _contains_domain_token(node: ast.AST) -> bool:
    for candidate in ast.walk(node):
        value: str | None = None
        if isinstance(candidate, ast.Constant) and isinstance(candidate.value, str):
            value = candidate.value
        elif isinstance(candidate, ast.Name):
            value = candidate.id
        elif isinstance(candidate, ast.Attribute):
            value = candidate.attr
        if value is None:
            continue
        normalized = value
        for separator in ".-_/:":
            normalized = normalized.replace(separator, " ")
        words = {
            word.lower()
            for segment in normalized.split()
            for word in IDENTIFIER_WORD_RE.findall(segment)
        }
        if DOMAIN_BRANCH_TOKENS.intersection(words):
            return True
    return False


def _protocol_expected_failure_branch_violations_for(path: Path) -> list[str]:
    """Find expected-failure handling that belongs in the extracted runtime."""

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    error_result_aliases = _expected_error_result_aliases_by_node(tree)
    expected_symbol_aliases = _expected_failure_symbol_aliases_by_node(tree)
    violations: list[str] = []
    for node in ast.walk(tree):
        node_error_aliases = error_result_aliases.get(id(node), set())
        node_expected_symbols = EXPECTED_FAILURE_SYMBOLS | expected_symbol_aliases.get(
            id(node),
            set(),
        )
        if isinstance(node, ast.ExceptHandler) and node.type is not None:
            if _node_mentions_symbols(node.type, node_expected_symbols):
                violations.append(
                    f"{path.name}:{node.lineno} catches ExpectedToolFailure in protocol facade"
                )
            elif _contains_domain_token(node.type) and _contains_expected_error_result(
                node.body,
                node_error_aliases,
            ):
                violations.append(
                    f"{path.name}:{node.lineno} contains a Skills/model/provider "
                    "exception error-result branch"
                )
            continue

        if isinstance(node, ast.Match):
            if _node_mentions_symbols(node.subject, node_expected_symbols):
                violations.append(
                    f"{path.name}:{node.lineno} branches on ExpectedToolFailure in protocol facade"
                )
            elif _contains_domain_token(node.subject) and _contains_expected_error_result(
                list(node.cases),
                node_error_aliases,
            ):
                violations.append(
                    f"{path.name}:{node.lineno} contains a Skills/model/provider error-result branch"
                )
            for case in node.cases:
                predicates = [case.pattern]
                if case.guard is not None:
                    predicates.append(case.guard)
                if any(
                    _node_mentions_symbols(predicate, node_expected_symbols)
                    for predicate in predicates
                ):
                    violations.append(
                        f"{path.name}:{case.pattern.lineno} branches on ExpectedToolFailure "
                        "in protocol facade"
                    )
                elif any(_contains_domain_token(predicate) for predicate in predicates) and (
                    _contains_expected_error_result(case.body, node_error_aliases)
                ):
                    violations.append(
                        f"{path.name}:{case.pattern.lineno} contains a Skills/model/provider "
                        "error-result branch"
                    )
            continue

        predicate: ast.AST | None = None
        bodies: list[ast.AST] = []
        if isinstance(node, ast.If):
            predicate = node.test
            bodies = [*node.body, *node.orelse]
        elif isinstance(node, ast.IfExp):
            predicate = node.test
            bodies = [node.body, node.orelse]
        if predicate is None:
            continue
        if _node_mentions_symbols(predicate, node_expected_symbols):
            violations.append(
                f"{path.name}:{node.lineno} branches on ExpectedToolFailure in protocol facade"
            )
        elif _contains_domain_token(predicate) and _contains_expected_error_result(
            bodies,
            node_error_aliases,
        ):
            violations.append(
                f"{path.name}:{node.lineno} contains a Skills/model/provider error-result branch"
            )
    return violations


def _is_policy_binding_node(
    node: ast.AST,
    execution: ast.FunctionDef | ast.AsyncFunctionDef,
) -> bool:
    if (
        isinstance(node, ast.Name)
        and node.id == "policy"
        and isinstance(node.ctx, (ast.Store, ast.Del))
    ):
        return True
    return (
        isinstance(node, ast.arg)
        and node.arg == "policy"
        or isinstance(node, ast.ExceptHandler)
        and node.name == "policy"
        or isinstance(node, (ast.MatchAs, ast.MatchStar))
        and node.name == "policy"
        or isinstance(node, ast.MatchMapping)
        and node.rest == "policy"
        or isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        and node is not execution
        and node.name == "policy"
        or isinstance(node, ast.alias)
        and (node.asname == "policy" or (node.asname is None and node.name == "policy"))
        or isinstance(node, (ast.Global, ast.Nonlocal))
        and "policy" in node.names
    )


def _runtime_mutable_authority_violations_for(path: Path) -> list[str]:
    """Find runtime decisions that bypass the immutable prepared policy."""

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    execution = next(
        (
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "execute_prepared_tool_call"
        ),
        None,
    )
    if execution is None:
        return [f"{path.name}: missing execute_prepared_tool_call for policy scan"]

    violations: list[str] = []
    execution_nodes = list(ast.walk(execution))
    policy_bindings = [
        node for node in execution_nodes if _is_policy_binding_node(node, execution)
    ]
    direct_policy_targets: set[int] = set()
    for node in execution_nodes:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "policy"
            and _attribute_path(node.value) == ("prepared", "policy")
        ):
            direct_policy_targets.add(id(node.targets[0]))
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "policy"
            and node.value is not None
            and _attribute_path(node.value) == ("prepared", "policy")
        ):
            direct_policy_targets.add(id(node.target))

    direct_policy_bindings = [
        binding for binding in policy_bindings if id(binding) in direct_policy_targets
    ]
    legitimate_policy_binding = (
        direct_policy_bindings[0] if direct_policy_bindings else None
    )
    valid_policy_binding = (
        len(policy_bindings) == 1
        and len(direct_policy_bindings) == 1
        and policy_bindings[0] is legitimate_policy_binding
    )
    if not policy_bindings:
        violations.append(
            f"{path.name}:{execution.lineno} must bind policy once directly from prepared.policy"
        )
    elif legitimate_policy_binding is None:
        violations.append(
            f"{path.name}:{policy_bindings[0].lineno} must bind policy directly from prepared.policy"
        )
    for binding in policy_bindings:
        if binding is legitimate_policy_binding:
            continue
        violations.append(
            f"{path.name}:{binding.lineno} reassigns policy; only one prepared.policy binding is allowed"
        )

    for node in execution_nodes:
        if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
            field = node.attr
            if field not in MUTABLE_EXECUTION_FIELDS:
                continue
            path_parts = _attribute_path(node)
            allowed_path = RUNTIME_POLICY_FIELD_PATHS.get(field)
            if (
                allowed_path is None
                or path_parts != allowed_path
                or not valid_policy_binding
            ):
                violations.append(
                    f"{path.name}:{node.lineno} reads {_display_node(node)} outside prepared policy"
                )
        elif isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Load):
            key = node.slice.value if isinstance(node.slice, ast.Constant) else None
            if key in MUTABLE_EXECUTION_FIELDS:
                violations.append(
                    f"{path.name}:{node.lineno} reads mutable mapping field {key!r}"
                )
        elif isinstance(node, ast.Call):
            key: object | None = None
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "get"
                and node.args
                and isinstance(node.args[0], ast.Constant)
            ):
                key = node.args[0].value
            elif (
                isinstance(node.func, ast.Name)
                and node.func.id == "getattr"
                and len(node.args) >= 2
                and isinstance(node.args[1], ast.Constant)
            ):
                key = node.args[1].value
            if key in MUTABLE_EXECUTION_FIELDS:
                violations.append(
                    f"{path.name}:{node.lineno} reads mutable field {key!r} via dynamic lookup"
                )
    operator_modules, operator_getitems = _operator_getitem_bindings_for(tree)
    authority_visitor = _PreparedAuthorityAliasVisitor(
        path,
        operator_modules=operator_modules,
        operator_getitems=operator_getitems,
    )
    for statement in execution.body:
        authority_visitor.visit(statement)
    violations.extend(authority_visitor.violations)
    return violations


def _contains_prepared_tool_authority(
    node: ast.AST,
    aliases: set[str] | None = None,
) -> bool:
    authority_aliases = aliases or set()
    return any(
        (
            isinstance(candidate, ast.Attribute)
            and _attribute_path(candidate) == ("prepared", "tool_def")
        )
        or isinstance(candidate, ast.Name)
        and candidate.id in authority_aliases
        for candidate in ast.walk(node)
    )


def _is_operator_getitem_reference(
    node: ast.AST,
    operator_modules: set[str],
    operator_getitems: set[str],
) -> bool:
    path = _attribute_path(node)
    return bool(
        isinstance(node, ast.Name)
        and node.id in operator_getitems
        or len(path) == 2
        and path[0] in operator_modules
        and path[1] == "getitem"
    )


def _operator_getitem_bindings_for(tree: ast.Module) -> tuple[set[str], set[str]]:
    operator_modules: set[str] = set()
    operator_getitems: set[str] = set()
    assignments: list[tuple[ast.AST, ast.AST]] = []
    scope_nodes, _ = _lexical_scope_nodes(tree)
    for statement in scope_nodes:
        if isinstance(statement, ast.Import):
            for alias in statement.names:
                if alias.name == "operator":
                    operator_modules.add(alias.asname or alias.name)
        elif isinstance(statement, ast.ImportFrom) and statement.module == "operator":
            for alias in statement.names:
                if alias.name == "getitem":
                    operator_getitems.add(alias.asname or alias.name)
        elif isinstance(statement, ast.Assign):
            assignments.extend((target, statement.value) for target in statement.targets)
        elif isinstance(statement, ast.AnnAssign) and statement.value is not None:
            assignments.append((statement.target, statement.value))

    changed = True
    while changed:
        changed = False
        for target, value in assignments:
            for candidate_target, candidate_value in _assignment_pairs(target, value):
                if (
                    isinstance(candidate_target, ast.Name)
                    and candidate_target.id not in operator_modules
                    and isinstance(candidate_value, ast.Name)
                    and candidate_value.id in operator_modules
                ):
                    operator_modules.add(candidate_target.id)
                    changed = True
                if (
                    isinstance(candidate_target, ast.Name)
                    and candidate_target.id not in operator_getitems
                    and _is_operator_getitem_reference(
                        candidate_value,
                        operator_modules,
                        operator_getitems,
                    )
                ):
                    operator_getitems.add(candidate_target.id)
                    changed = True
    return operator_modules, operator_getitems


class _PreparedAuthorityAliasVisitor(ast.NodeVisitor):
    def __init__(
        self,
        path: Path,
        *,
        operator_modules: set[str],
        operator_getitems: set[str],
    ) -> None:
        self.path = path
        self.aliases: set[str] = set()
        self.operator_modules = set(operator_modules)
        self.operator_getitems = set(operator_getitems)
        self.violations: list[str] = []

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        for default in [*node.args.defaults, *node.args.kw_defaults]:
            if default is not None:
                self.visit(default)
        previous_aliases = set(self.aliases)
        previous_modules = set(self.operator_modules)
        previous_getitems = set(self.operator_getitems)
        bound_names = _lexically_bound_names(node)
        self.aliases.difference_update(bound_names)
        self.operator_modules.difference_update(bound_names)
        self.operator_getitems.difference_update(bound_names)
        for statement in node.body:
            self.visit(statement)
        self.aliases = previous_aliases
        self.operator_modules = previous_modules
        self.operator_getitems = previous_getitems

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802
        self._visit_function(node)

    def visit_Lambda(self, node: ast.Lambda) -> None:  # noqa: N802
        for default in [*node.args.defaults, *node.args.kw_defaults]:
            if default is not None:
                self.visit(default)
        previous_aliases = set(self.aliases)
        previous_modules = set(self.operator_modules)
        previous_getitems = set(self.operator_getitems)
        bound_names = _lexically_bound_names(node)
        self.aliases.difference_update(bound_names)
        self.operator_modules.difference_update(bound_names)
        self.operator_getitems.difference_update(bound_names)
        self.visit(node.body)
        self.aliases = previous_aliases
        self.operator_modules = previous_modules
        self.operator_getitems = previous_getitems

    def _update_alias(self, target: ast.AST, value: ast.AST) -> None:
        for candidate_target, candidate_value in _assignment_pairs(target, value):
            if _contains_prepared_tool_authority(candidate_value, self.aliases):
                self.aliases.update(_stored_names(candidate_target))
        # A later assignment may be conditional, so tainted-name reuse stays tainted.

    def _update_operator_getitem_alias(self, target: ast.AST, value: ast.AST) -> None:
        for candidate_target, candidate_value in _assignment_pairs(target, value):
            if isinstance(candidate_target, ast.Name) and _is_operator_getitem_reference(
                candidate_value,
                self.operator_modules,
                self.operator_getitems,
            ):
                self.operator_getitems.add(candidate_target.id)

    def _update_operator_module_alias(self, target: ast.AST, value: ast.AST) -> None:
        for candidate_target, candidate_value in _assignment_pairs(target, value):
            if (
                isinstance(candidate_target, ast.Name)
                and isinstance(candidate_value, ast.Name)
                and candidate_value.id in self.operator_modules
            ):
                self.operator_modules.add(candidate_target.id)

    def visit_Assign(self, node: ast.Assign) -> None:  # noqa: N802
        self.visit(node.value)
        for target in node.targets:
            self._update_alias(target, node.value)
            self._update_operator_module_alias(target, node.value)
            self._update_operator_getitem_alias(target, node.value)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:  # noqa: N802
        if node.value is not None:
            self.visit(node.value)
            self._update_alias(node.target, node.value)
            self._update_operator_module_alias(node.target, node.value)
            self._update_operator_getitem_alias(node.target, node.value)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:  # noqa: N802
        self.visit(node.value)
        self._update_alias(node.target, node.value)
        self._update_operator_module_alias(node.target, node.value)
        self._update_operator_getitem_alias(node.target, node.value)

    def visit_Import(self, node: ast.Import) -> None:  # noqa: N802
        for alias in node.names:
            if alias.name == "operator":
                self.operator_modules.add(alias.asname or alias.name)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # noqa: N802
        if node.module != "operator":
            return
        for alias in node.names:
            if alias.name == "getitem":
                self.operator_getitems.add(alias.asname or alias.name)

    def visit_Subscript(self, node: ast.Subscript) -> None:  # noqa: N802
        if (
            isinstance(node.ctx, ast.Load)
            and not isinstance(node.slice, ast.Constant)
            and _contains_prepared_tool_authority(node.value, self.aliases)
        ):
            self.violations.append(
                f"{self.path.name}:{node.lineno} performs dynamic prepared authority read "
                f"{_display_node(node)}"
            )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        dynamic_get = (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and node.args
            and not isinstance(node.args[0], ast.Constant)
            and _contains_prepared_tool_authority(node.func.value, self.aliases)
        )
        dynamic_getattr = (
            isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and len(node.args) >= 2
            and not isinstance(node.args[1], ast.Constant)
            and _contains_prepared_tool_authority(node.args[0], self.aliases)
        )
        operator_getitem = (
            _is_operator_getitem_reference(
                node.func,
                self.operator_modules,
                self.operator_getitems,
            )
            and len(node.args) >= 2
            and _contains_prepared_tool_authority(node.args[0], self.aliases)
            and (
                not isinstance(node.args[1], ast.Constant)
                or node.args[1].value in MUTABLE_EXECUTION_FIELDS
            )
        )
        if dynamic_get or dynamic_getattr or operator_getitem:
            self.violations.append(
                f"{self.path.name}:{node.lineno} performs dynamic prepared authority read "
                f"{_display_node(node)}"
            )
        self.generic_visit(node)


class _DefaultStrVisitor(ast.NodeVisitor):
    def __init__(self, path: Path) -> None:
        self.path = path
        self.aliases: set[str] = {"str"}
        self.default_mapping_aliases: set[str] = set()
        self.builtins_modules: set[str] = {"builtins"}
        self.violations: list[str] = []

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        for default in [*node.args.defaults, *node.args.kw_defaults]:
            if default is not None:
                self.visit(default)
        previous_aliases = set(self.aliases)
        previous_mappings = set(self.default_mapping_aliases)
        previous_builtins = set(self.builtins_modules)
        bound_names = _lexically_bound_names(node)
        self.aliases.difference_update(bound_names)
        self.default_mapping_aliases.difference_update(bound_names)
        self.builtins_modules.difference_update(bound_names)
        for statement in node.body:
            self.visit(statement)
        self.aliases = previous_aliases
        self.default_mapping_aliases = previous_mappings
        self.builtins_modules = previous_builtins

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802
        self._visit_function(node)

    def visit_Lambda(self, node: ast.Lambda) -> None:  # noqa: N802
        for default in [*node.args.defaults, *node.args.kw_defaults]:
            if default is not None:
                self.visit(default)
        previous_aliases = set(self.aliases)
        previous_mappings = set(self.default_mapping_aliases)
        previous_builtins = set(self.builtins_modules)
        bound_names = _lexically_bound_names(node)
        self.aliases.difference_update(bound_names)
        self.default_mapping_aliases.difference_update(bound_names)
        self.builtins_modules.difference_update(bound_names)
        self.visit(node.body)
        self.aliases = previous_aliases
        self.default_mapping_aliases = previous_mappings
        self.builtins_modules = previous_builtins

    def _is_str_expression(self, value: ast.AST) -> bool:
        return bool(
            isinstance(value, ast.Name)
            and value.id in self.aliases
            or isinstance(value, ast.Attribute)
            and isinstance(value.value, ast.Name)
            and value.value.id in self.builtins_modules
            and value.attr == "str"
        )

    def _update_alias(self, target: ast.AST, value: ast.AST) -> None:
        for candidate_target, candidate_value in _assignment_pairs(target, value):
            if isinstance(candidate_target, ast.Name) and self._is_str_expression(
                candidate_value
            ):
                self.aliases.add(candidate_target.id)
        # A later assignment may be conditional, so tainted-name reuse stays tainted.

    def _contains_default_str_mapping(self, value: ast.AST) -> bool:
        if isinstance(value, ast.Name):
            return value.id in self.default_mapping_aliases
        if isinstance(value, ast.Dict):
            return any(
                isinstance(key, ast.Constant)
                and key.value == "default"
                and self._is_str_expression(item)
                or key is None
                and self._contains_default_str_mapping(item)
                for key, item in zip(value.keys, value.values, strict=True)
            )
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id == "dict"
        ):
            return any(
                keyword.arg == "default" and self._is_str_expression(keyword.value)
                or keyword.arg is None
                and self._contains_default_str_mapping(keyword.value)
                for keyword in value.keywords
            ) or any(
                self._contains_default_str_mapping(argument)
                for argument in value.args[:1]
            )
        if isinstance(value, ast.BinOp) and isinstance(value.op, ast.BitOr):
            return self._contains_default_str_mapping(
                value.left
            ) or self._contains_default_str_mapping(value.right)
        return False

    def _update_default_mapping_alias(self, target: ast.AST, value: ast.AST) -> None:
        for candidate_target, candidate_value in _assignment_pairs(target, value):
            if isinstance(candidate_target, ast.Name) and self._contains_default_str_mapping(
                candidate_value
            ):
                self.default_mapping_aliases.add(candidate_target.id)
        # Expanded kwargs use the same conservative may-be-tainted name rule.

    def _update_builtins_module_alias(self, target: ast.AST, value: ast.AST) -> None:
        for candidate_target, candidate_value in _assignment_pairs(target, value):
            if (
                isinstance(candidate_target, ast.Name)
                and isinstance(candidate_value, ast.Name)
                and candidate_value.id in self.builtins_modules
            ):
                self.builtins_modules.add(candidate_target.id)

    def visit_Assign(self, node: ast.Assign) -> None:  # noqa: N802
        self.visit(node.value)
        for target in node.targets:
            self._update_alias(target, node.value)
            self._update_default_mapping_alias(target, node.value)
            self._update_builtins_module_alias(target, node.value)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:  # noqa: N802
        if node.value is not None:
            self.visit(node.value)
            self._update_alias(node.target, node.value)
            self._update_default_mapping_alias(node.target, node.value)
            self._update_builtins_module_alias(node.target, node.value)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:  # noqa: N802
        self.visit(node.value)
        self._update_alias(node.target, node.value)
        self._update_default_mapping_alias(node.target, node.value)
        self._update_builtins_module_alias(node.target, node.value)

    def visit_Import(self, node: ast.Import) -> None:  # noqa: N802
        for alias in node.names:
            if alias.name == "builtins":
                self.builtins_modules.add(alias.asname or "builtins")

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # noqa: N802
        if node.module != "builtins":
            return
        for alias in node.names:
            if alias.name == "str":
                self.aliases.add(alias.asname or alias.name)

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        uses_default_str = any(
            keyword.arg == "default" and self._is_str_expression(keyword.value)
            or keyword.arg is None
            and self._contains_default_str_mapping(keyword.value)
            for keyword in node.keywords
        )
        if uses_default_str:
            self.violations.append(
                f"{self.path.name}:{node.lineno} uses coercing default=str serialization"
            )
        self.generic_visit(node)


def _default_str_violations_for(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    visitor = _DefaultStrVisitor(path)
    visitor.visit(tree)
    return visitor.violations


def _is_logger_expression(
    node: ast.AST,
    logger_names: set[str] | None = None,
    logger_modules: set[str] | None = None,
) -> bool:
    names = {"logger"} if logger_names is None else logger_names
    modules = {"loguru"} if logger_modules is None else logger_modules
    if isinstance(node, ast.Name):
        return node.id in names
    if isinstance(node, ast.Attribute):
        return (
            node.attr == "logger"
            or _is_logger_expression(node.value, names, modules)
        )
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        return _is_logger_expression(node.func.value, names, modules)
    return False


def _helper_uses_parameter_unsafely(
    node: ast.AST,
    parameter: str,
    helper_name: str,
) -> bool:
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "type"
        and len(node.args) == 1
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == parameter
        and not node.keywords
    ):
        return False
    if (
        isinstance(node, ast.Call)
        and _attribute_path(node.func) == ("object", "__getattribute__")
        and len(node.args) == 2
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == parameter
        and isinstance(node.args[1], ast.Constant)
        and node.args[1].value == "_reason"
        and helper_name == "get_expected_tool_failure_reason"
        and not node.keywords
    ):
        return False
    if (
        isinstance(node, ast.Attribute)
        and _attribute_path(node) == (parameter, "__class__", "__name__")
    ):
        return False
    if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id == parameter:
        return True
    return any(
        _helper_uses_parameter_unsafely(child, parameter, helper_name)
        for child in ast.iter_child_nodes(node)
    )


def _is_safe_helper_import(node: ast.ImportFrom) -> bool:
    return bool(
        node.module == "execution_outcomes"
        and node.level > 0
        or node.module == f"{MCP_PACKAGE}.execution_outcomes"
    )


def _verified_safe_exception_log_helpers(scope: ast.AST) -> set[str]:
    definitions: dict[str, list[ast.FunctionDef | ast.AsyncFunctionDef]] = {}
    imported: set[str] = set()
    scope_nodes, _ = _lexical_scope_nodes(scope)
    for node in scope_nodes:
        if (
            isinstance(scope, ast.Module)
            and
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name in SAFE_EXCEPTION_LOG_HELPERS
        ):
            definitions.setdefault(node.name, []).append(node)
        elif (
            isinstance(node, ast.ImportFrom)
            and _is_safe_helper_import(node)
        ):
            imported.update(
                alias.asname or alias.name
                for alias in node.names
                if alias.name == "get_expected_tool_failure_reason"
            )

    verified = set(imported)
    for name, candidates in definitions.items():
        all_safe = True
        for candidate in candidates:
            arguments = [*candidate.args.posonlyargs, *candidate.args.args]
            if not arguments:
                all_safe = False
                break
            parameter = arguments[0].arg
            if any(
                _helper_uses_parameter_unsafely(statement, parameter, name)
                for statement in candidate.body
            ):
                all_safe = False
                break
        if all_safe:
            verified.add(name)
    return verified


def _is_safe_exception_log_expression(
    node: ast.AST,
    exception_names: set[str],
    safe_helper_names: set[str],
) -> bool:
    if isinstance(node, ast.Call):
        function_path = _attribute_path(node.func)
        if (
            function_path
            and function_path[-1] in safe_helper_names
            and len(node.args) == 1
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id in exception_names
            and not node.keywords
        ):
            return True
    return False


def _uses_raw_exception_unsafely(
    node: ast.AST,
    exception_names: set[str],
    safe_helper_names: set[str],
) -> bool:
    if _is_safe_exception_log_expression(node, exception_names, safe_helper_names):
        return False
    if isinstance(node, ast.Name) and node.id in exception_names:
        return True
    return any(
        _uses_raw_exception_unsafely(child, exception_names, safe_helper_names)
        for child in ast.iter_child_nodes(node)
    )


class _UnsafeExceptionLogVisitor(ast.NodeVisitor):
    """Track raw exception names only within their lexical logging scope."""

    def __init__(self, path: Path, *, safe_helper_names: set[str]) -> None:
        self.path = path
        self.exception_names: set[str] = set()
        self.logger_names: set[str] = {"logger"}
        self.logger_modules: set[str] = {"loguru"}
        self.safe_helper_names = set(safe_helper_names)
        self.violations: list[str] = []

    @staticmethod
    def _annotated_exception_arguments(
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> set[str]:
        names: set[str] = set()
        for argument in [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]:
            annotation = argument.annotation
            if annotation is not None and _node_mentions_symbols(
                annotation,
                {"BaseException", "Exception"},
            ):
                names.add(argument.arg)
        return names

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        for default in [*node.args.defaults, *node.args.kw_defaults]:
            if default is not None:
                self.visit(default)
        previous_exceptions = set(self.exception_names)
        previous_loggers = set(self.logger_names)
        previous_logger_modules = set(self.logger_modules)
        previous_safe_helpers = set(self.safe_helper_names)
        bound_names = _lexically_bound_names(node)
        self.exception_names = (
            previous_exceptions - bound_names
        ) | self._annotated_exception_arguments(node)
        self.logger_names.difference_update(bound_names)
        self.logger_modules.difference_update(bound_names)
        self.safe_helper_names = (
            previous_safe_helpers - bound_names
        ) | _verified_safe_exception_log_helpers(node)
        for statement in node.body:
            self.visit(statement)
        self.exception_names = previous_exceptions
        self.logger_names = previous_loggers
        self.logger_modules = previous_logger_modules
        self.safe_helper_names = previous_safe_helpers

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802
        self._visit_function(node)

    def visit_Lambda(self, node: ast.Lambda) -> None:  # noqa: N802
        for default in [*node.args.defaults, *node.args.kw_defaults]:
            if default is not None:
                self.visit(default)
        previous_exceptions = set(self.exception_names)
        previous_loggers = set(self.logger_names)
        previous_logger_modules = set(self.logger_modules)
        previous_safe_helpers = set(self.safe_helper_names)
        bound_names = _lexically_bound_names(node)
        self.exception_names.difference_update(bound_names)
        self.logger_names.difference_update(bound_names)
        self.logger_modules.difference_update(bound_names)
        self.safe_helper_names.difference_update(bound_names)
        self.visit(node.body)
        self.exception_names = previous_exceptions
        self.logger_names = previous_loggers
        self.logger_modules = previous_logger_modules
        self.safe_helper_names = previous_safe_helpers

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:  # noqa: N802
        previous = set(self.exception_names)
        self.exception_names = set(previous)
        if isinstance(node.name, str):
            self.exception_names.add(node.name)
        for statement in node.body:
            self.visit(statement)
        if isinstance(node.name, str):
            self.exception_names.discard(node.name)

    @staticmethod
    def _assigned_names(targets: list[ast.AST]) -> set[str]:
        return set().union(*(_stored_names(target) for target in targets))

    def _update_exception_aliases(self, targets: list[ast.AST], value: ast.AST) -> None:
        for target in targets:
            for candidate_target, candidate_value in _assignment_pairs(target, value):
                if _uses_raw_exception_unsafely(
                    candidate_value,
                    self.exception_names,
                    self.safe_helper_names,
                ):
                    self.exception_names.update(_stored_names(candidate_target))
        # A later assignment may be conditional, so tainted-name reuse stays tainted.

    def _update_logger_aliases(self, targets: list[ast.AST], value: ast.AST) -> None:
        for target in targets:
            for candidate_target, candidate_value in _assignment_pairs(target, value):
                if _is_logger_expression(
                    candidate_value,
                    self.logger_names,
                    self.logger_modules,
                ):
                    self.logger_names.update(_stored_names(candidate_target))
        # A later assignment may be conditional, so logger-name reuse stays tainted.

    def _update_logger_module_aliases(self, targets: list[ast.AST], value: ast.AST) -> None:
        for target in targets:
            for candidate_target, candidate_value in _assignment_pairs(target, value):
                if (
                    isinstance(candidate_value, ast.Name)
                    and candidate_value.id in self.logger_modules
                ):
                    self.logger_modules.update(_stored_names(candidate_target))

    def _update_safe_helper_aliases(self, targets: list[ast.AST], value: ast.AST) -> None:
        for target in targets:
            for candidate_target, candidate_value in _assignment_pairs(target, value):
                assigned_names = _stored_names(candidate_target)
                value_is_safe = (
                    isinstance(candidate_value, ast.Name)
                    and candidate_value.id in self.safe_helper_names
                )
                self.safe_helper_names.difference_update(assigned_names)
                if value_is_safe:
                    self.safe_helper_names.update(assigned_names)

    def visit_Assign(self, node: ast.Assign) -> None:  # noqa: N802
        self.visit(node.value)
        self._update_exception_aliases(node.targets, node.value)
        self._update_logger_aliases(node.targets, node.value)
        self._update_logger_module_aliases(node.targets, node.value)
        self._update_safe_helper_aliases(node.targets, node.value)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:  # noqa: N802
        if node.value is not None:
            self.visit(node.value)
            self._update_exception_aliases([node.target], node.value)
            self._update_logger_aliases([node.target], node.value)
            self._update_logger_module_aliases([node.target], node.value)
            self._update_safe_helper_aliases([node.target], node.value)

    def visit_Import(self, node: ast.Import) -> None:  # noqa: N802
        for alias in node.names:
            if alias.name == "loguru":
                self.logger_modules.add(alias.asname or alias.name)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # noqa: N802
        if node.module == "loguru":
            for alias in node.names:
                if alias.name == "logger":
                    self.logger_names.add(alias.asname or alias.name)
        if _is_safe_helper_import(node):
            self.safe_helper_names.update(
                alias.asname or alias.name
                for alias in node.names
                if alias.name == "get_expected_tool_failure_reason"
            )

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:  # noqa: N802
        self.visit(node.value)
        self._update_exception_aliases([node.target], node.value)
        self._update_logger_aliases([node.target], node.value)
        self._update_logger_module_aliases([node.target], node.value)
        self._update_safe_helper_aliases([node.target], node.value)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:  # noqa: N802
        self.visit(node.value)
        assigned_names = self._assigned_names([node.target])
        if assigned_names.intersection(self.exception_names) or _uses_raw_exception_unsafely(
            node.value,
            self.exception_names,
            self.safe_helper_names,
        ):
            self.exception_names.update(assigned_names)

    def _visit_for(self, node: ast.For | ast.AsyncFor) -> None:
        self.visit(node.iter)
        self._update_exception_aliases([node.target], node.iter)
        for statement in [*node.body, *node.orelse]:
            self.visit(statement)

    def visit_For(self, node: ast.For) -> None:  # noqa: N802
        self._visit_for(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:  # noqa: N802
        self._visit_for(node)

    def _visit_with(self, node: ast.With | ast.AsyncWith) -> None:
        for item in node.items:
            self.visit(item.context_expr)
            if item.optional_vars is not None:
                self._update_exception_aliases([item.optional_vars], item.context_expr)
        for statement in node.body:
            self.visit(statement)

    def visit_With(self, node: ast.With) -> None:  # noqa: N802
        self._visit_with(node)

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:  # noqa: N802
        self._visit_with(node)

    @staticmethod
    def _match_bound_names(pattern: ast.pattern) -> set[str]:
        names: set[str] = set()
        for candidate in ast.walk(pattern):
            if isinstance(candidate, (ast.MatchAs, ast.MatchStar)) and candidate.name:
                names.add(candidate.name)
            elif isinstance(candidate, ast.MatchMapping) and candidate.rest:
                names.add(candidate.rest)
        return names

    def visit_Match(self, node: ast.Match) -> None:  # noqa: N802
        self.visit(node.subject)
        subject_is_exception = _uses_raw_exception_unsafely(
            node.subject,
            self.exception_names,
            self.safe_helper_names,
        )
        previous = set(self.exception_names)
        branch_states: list[set[str]] = []
        for case in node.cases:
            self.exception_names = set(previous)
            bound_names = self._match_bound_names(case.pattern)
            if subject_is_exception:
                self.exception_names.update(bound_names)
            else:
                self.exception_names.difference_update(bound_names)
            if case.guard is not None:
                self.visit(case.guard)
            for statement in case.body:
                self.visit(statement)
            branch_states.append(set(self.exception_names))
        self.exception_names = previous.union(*branch_states)

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        is_logger_call = (
            isinstance(node.func, ast.Attribute)
            and node.func.attr in LOGGER_METHODS
            and _is_logger_expression(
                node.func.value,
                self.logger_names,
                self.logger_modules,
            )
        )
        if is_logger_call:
            if node.func.attr == "exception" and self.exception_names:
                self.violations.append(
                    f"{self.path.name}:{node.lineno} uses logger.exception with execution context"
                )
            for keyword in node.keywords:
                if (
                    keyword.arg in {"exception", "exc_info"}
                    and self.exception_names
                    and not (
                        isinstance(keyword.value, ast.Constant)
                        and keyword.value.value in {False, None}
                    )
                ):
                    self.violations.append(
                        f"{self.path.name}:{node.lineno} enables raw {keyword.arg} logging"
                    )
            values = [
                *node.args,
                *(
                    keyword.value
                    for keyword in node.keywords
                    if keyword.arg not in {"exception", "exc_info"}
                ),
            ]
            for value in values:
                if _uses_raw_exception_unsafely(
                    value,
                    self.exception_names,
                    self.safe_helper_names,
                ):
                    self.violations.append(
                        f"{self.path.name}:{node.lineno} logs raw exception expression "
                        f"{_display_node(value)}"
                    )
        self.generic_visit(node)


def _unsafe_exception_log_violations_for(path: Path) -> list[str]:
    """Return unsafe raw-exception logging violations in one Python module."""

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    visitor = _UnsafeExceptionLogVisitor(
        path,
        safe_helper_names=_verified_safe_exception_log_helpers(tree),
    )
    visitor.visit(tree)
    return visitor.violations


def _defined_class_names_for(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}


def test_new_interface_modules_do_not_import_tldw_server_api() -> None:
    interface_dir = MCP_ROOT / "interfaces"
    assert interface_dir.exists()
    interface_files = {path.name for path in interface_dir.rglob("*.py")}
    assert EXPECTED_INTERFACE_FILES.issubset(interface_files)
    offenders: dict[str, list[str]] = {}
    for path in interface_dir.rglob("*.py"):
        violations = _interface_boundary_violations_for(path, interface_dir)
        if violations:
            offenders[str(path)] = violations
    assert offenders == {}


def test_protocol_uses_runtime_dependencies_for_stage3b_host_services() -> None:
    forbidden_imports = {
        "tldw_Server_API.app.core.Infrastructure.redis_factory",
        "tldw_Server_API.app.core.Metrics.telemetry",
        "tldw_Server_API.app.core.testing",
    }

    imports = _resolved_import_sources_for(MCP_ROOT / "protocol.py", MCP_PACKAGE)
    offenders = sorted(
        source
        for source in imports
        if source in forbidden_imports
        or any(source.startswith(f"{forbidden}.") for forbidden in forbidden_imports)
    )

    assert offenders == []


def test_protocol_catalog_lookup_uses_runtime_dependencies() -> None:
    forbidden_import = "tldw_Server_API.app.core.AuthNZ.database"

    imports = _resolved_import_sources_for(MCP_ROOT / "protocol.py", MCP_PACKAGE)
    offenders = sorted(
        source
        for source in imports
        if source == forbidden_import or source.startswith(f"{forbidden_import}.")
    )

    assert offenders == []


def test_tldw_runtime_catalog_provider_does_not_inline_tool_catalog_sql() -> None:
    adapter = MCP_ROOT / "adapters" / "tldw_runtime.py"
    tree = ast.parse(adapter.read_text(encoding="utf-8"), filename=str(adapter))
    table_sql_constants = sorted(
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and (
            "tool_catalogs" in node.value
            or "tool_catalog_entries" in node.value
        )
    )

    assert table_sql_constants == []


def test_server_uses_runtime_dependencies_for_stage3c_host_services() -> None:
    forbidden_imports = {
        "tldw_Server_API.app.core.AuthNZ.exceptions",
        "tldw_Server_API.app.core.Streaming.streams",
        "tldw_Server_API.app.core.testing",
    }

    imports = _resolved_import_sources_for(MCP_ROOT / "server.py", MCP_PACKAGE)
    offenders = sorted(
        source
        for source in imports
        if source in forbidden_imports
        or any(source.startswith(f"{forbidden}.") for forbidden in forbidden_imports)
    )

    assert offenders == []


def test_security_and_config_use_package_local_environment_helpers() -> None:
    forbidden_import = "tldw_Server_API.app.core.testing"
    targets = {
        MCP_ROOT / "config.py": MCP_PACKAGE,
        MCP_ROOT / "security" / "ip_filter.py": f"{MCP_PACKAGE}.security",
        MCP_ROOT / "security" / "request_guards.py": f"{MCP_PACKAGE}.security",
    }

    offenders: dict[str, list[str]] = {}
    for path, package in targets.items():
        imports = _resolved_import_sources_for(path, package)
        blocked = sorted(
            source
            for source in imports
            if source == forbidden_import or source.startswith(f"{forbidden_import}.")
        )
        if blocked:
            offenders[str(path.relative_to(MCP_ROOT))] = blocked

    assert offenders == {}


def test_mcp_discovery_module_uses_package_local_environment_helpers() -> None:
    forbidden_import = "tldw_Server_API.app.core.testing"
    discovery_module = MCP_ROOT / "modules" / "implementations" / "mcp_discovery_module.py"

    imports = _resolved_import_sources_for(
        discovery_module,
        f"{MCP_PACKAGE}.modules.implementations",
    )
    offenders = sorted(
        source
        for source in imports
        if source == forbidden_import or source.startswith(f"{forbidden_import}.")
    )

    assert offenders == []


def test_base_module_does_not_import_host_circuit_breaker() -> None:
    forbidden_import = "tldw_Server_API.app.core.Infrastructure.circuit_breaker"
    imports = _resolved_import_sources_for(
        MCP_ROOT / "modules" / "base.py",
        f"{MCP_PACKAGE}.modules",
    )
    offenders = sorted(
        source
        for source in imports
        if source == forbidden_import or source.startswith(f"{forbidden_import}.")
    )

    assert offenders == []


def test_catalog_loader_uses_standalone_catalog_schema() -> None:
    """Catalog loading must depend on standalone MCP package schemas."""
    forbidden_import = "tldw_Server_API.app.api.v1.schemas.archetype_schemas"
    imports = _resolved_import_sources_for(MCP_ROOT / "catalog_loader.py", MCP_PACKAGE)
    offenders = sorted(
        source
        for source in imports
        if source == forbidden_import or source.startswith(f"{forbidden_import}.")
    )

    assert offenders == []


def test_catalog_loader_delegates_to_standalone_package_loader() -> None:
    """Host catalog loader should be a thin wrapper around package loader code."""
    imports = _resolved_import_sources_for(
        MCP_ROOT / "catalog_loader.py",
        MCP_PACKAGE,
        top_level_only=True,
    )
    forbidden_sources = {"yaml", "loguru", "pydantic"}
    offenders = sorted(
        source
        for source in imports
        if source in forbidden_sources
        or any(source.startswith(f"{forbidden}.") for forbidden in forbidden_sources)
    )

    assert "mcp_unified.federation.catalog_loader" in imports
    assert offenders == []


def test_external_server_config_schema_delegates_to_standalone_package_schema() -> None:
    """Host external registry config schema should be a package-wrapper seam."""
    imports = _resolved_import_sources_for(
        MCP_ROOT / "external_servers" / "config_schema.py",
        f"{MCP_PACKAGE}.external_servers",
        top_level_only=True,
    )
    forbidden_sources = {"fnmatch", "json", "loguru", "pydantic", "yaml"}
    offenders = sorted(
        source
        for source in imports
        if source in forbidden_sources
        or any(source.startswith(f"{forbidden}.") for forbidden in forbidden_sources)
    )

    assert "mcp_unified.federation.config_schema" in imports
    assert offenders == []


def test_standalone_catalog_loader_uses_loguru_not_stdlib_logging() -> None:
    """Package catalog loader should use the project-standard Loguru logger."""
    imports = _resolved_import_sources_for(
        STANDALONE_MCP_ROOT / "federation" / "catalog_loader.py",
        "mcp_unified.federation",
        top_level_only=True,
    )

    assert "loguru" in imports
    assert "logging" not in imports


def test_protocol_boundary_scan_resolves_relative_imports(tmp_path: Path) -> None:
    sample = tmp_path / "sample.py"
    sample.write_text(
        "from ..testing import is_truthy\n"
        "from .adapters.tldw_runtime import build_default_runtime_dependencies\n",
        encoding="utf-8",
    )

    imports = _resolved_import_sources_for(sample, MCP_PACKAGE)

    assert imports == [
        "tldw_Server_API.app.core.testing",
        "tldw_Server_API.app.core.MCP_unified.adapters.tldw_runtime",
    ]


def test_protocol_import_time_boundary_does_not_load_tldw_runtime_adapter() -> None:
    imports = _resolved_import_sources_for(
        MCP_ROOT / "protocol.py",
        MCP_PACKAGE,
        top_level_only=True,
    )

    assert "tldw_Server_API.app.core.MCP_unified.adapters.tldw_runtime" not in imports


def test_default_runtime_dependency_builder_exposes_core_dependencies() -> None:
    from tldw_Server_API.app.core.MCP_unified.adapters.tldw_runtime import (
        build_default_runtime_dependencies,
    )

    deps = build_default_runtime_dependencies()
    assert hasattr(deps, "module_registry")
    assert hasattr(deps, "rbac_policy")


@pytest.mark.asyncio
async def test_tldw_auth_provider_fails_closed_for_unencodable_ws_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.MCP_unified.adapters import tldw_runtime

    verify_calls = 0

    async def _unexpected_verify(*_args: Any, **_kwargs: Any) -> None:
        nonlocal verify_calls
        verify_calls += 1
        raise AssertionError("verify_jwt_and_fetch_user should not be called")

    monkeypatch.setattr(tldw_runtime, "get_jwt_manager", lambda: object())
    monkeypatch.setattr(
        "tldw_Server_API.app.core.AuthNZ.User_DB_Handling.verify_jwt_and_fetch_user",
        _unexpected_verify,
    )

    provider = tldw_runtime.TldwServerAuthProvider()
    websocket = SimpleNamespace(headers={"x-bad": "snowman \u2603"}, client=None)

    assert (
        await provider.authenticate_authnz_websocket_token(
            "token",
            websocket=websocket,
        )
        is None
    )
    assert verify_calls == 0


@pytest.mark.asyncio
async def test_tldw_auth_provider_fails_closed_when_authnz_ws_verify_rejects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.AuthNZ.exceptions import InvalidTokenError
    from tldw_Server_API.app.core.MCP_unified.adapters import tldw_runtime

    async def _reject_token(*_args: Any, **_kwargs: Any) -> None:
        raise InvalidTokenError("bad token")

    monkeypatch.setattr(tldw_runtime, "get_jwt_manager", lambda: object())
    monkeypatch.setattr(
        "tldw_Server_API.app.core.AuthNZ.User_DB_Handling.verify_jwt_and_fetch_user",
        _reject_token,
    )

    provider = tldw_runtime.TldwServerAuthProvider()
    websocket = SimpleNamespace(headers={}, client=None)

    assert (
        await provider.authenticate_authnz_websocket_token(
            "token",
            websocket=websocket,
        )
        is None
    )


def test_stage3_runtime_contracts_are_exported_by_interface_packages() -> None:
    import mcp_unified.interfaces as standalone_interfaces

    import tldw_Server_API.app.core.MCP_unified.interfaces as compat_interfaces

    names = [
        "AuthenticatedIdentity",
        "LifecycleGuard",
        "NoopToolCallHookManager",
        "NoopToolUseRecorder",
        "ModuleConfigProvider",
        "PermissionSeeder",
        "PolicyContextProvider",
        "ServerAuthProvider",
        "ToolCallHookManager",
        "ToolCatalogProvider",
        "ToolHookAction",
        "ToolHookCallContext",
        "ToolHookDecision",
        "ToolUseRecorder",
        "EnvironmentFlagsProvider",
        "WebSocketStream",
        "WebSocketStreamFactory",
    ]

    for name in names:
        assert getattr(standalone_interfaces, name) is getattr(compat_interfaces, name)


def test_mcp_protocol_accepts_runtime_dependencies() -> None:
    from mcp_unified.tool_use_reporting.recorder import NoopToolUseRecorder

    from tldw_Server_API.app.core.MCP_unified.protocol import MCPProtocol

    deps = _fake_runtime_dependencies()
    protocol = MCPProtocol(dependencies=deps)
    assert protocol.module_registry is deps.module_registry
    assert protocol.rbac_policy is deps.rbac_policy
    assert isinstance(protocol._tool_use_recorder, NoopToolUseRecorder)


def test_mcp_protocol_treats_none_tool_use_recorder_as_unconfigured() -> None:
    from mcp_unified.tool_use_reporting.recorder import NoopToolUseRecorder

    from tldw_Server_API.app.core.MCP_unified.protocol import MCPProtocol

    deps = _fake_runtime_dependencies()
    deps.tool_use_recorder = None

    protocol = MCPProtocol(dependencies=deps)

    assert isinstance(protocol._tool_use_recorder, NoopToolUseRecorder)


def test_runtime_dependencies_default_to_noop_tool_use_recorder() -> None:
    from mcp_unified.interfaces.runtime import MCPRuntimeDependencies, NoopToolCallHookManager
    from mcp_unified.tool_use_reporting.recorder import NoopToolUseRecorder

    deps = _fake_runtime_dependencies()
    dependency_kwargs = vars(deps).copy()
    runtime_dependencies = MCPRuntimeDependencies(**dependency_kwargs)

    assert isinstance(runtime_dependencies.tool_use_recorder, NoopToolUseRecorder)
    assert isinstance(runtime_dependencies.tool_call_hook_manager, NoopToolCallHookManager)


def test_mcp_server_accepts_runtime_dependencies() -> None:
    from tldw_Server_API.app.core.MCP_unified.server import MCPServer

    deps = _fake_runtime_dependencies()
    server = MCPServer(dependencies=deps)
    assert server.protocol.module_registry is deps.module_registry
    assert server.protocol.rbac_policy is deps.rbac_policy
    assert server.module_registry is deps.module_registry
    assert server.rbac_policy is deps.rbac_policy
    assert server.metrics_collector is deps.metrics_collector
    assert server.auth_provider is deps.auth_provider
    assert server.lifecycle_guard is deps.lifecycle_guard
    assert server.permission_seeder is deps.permission_seeder
    assert server.module_config_provider is deps.module_config_provider
    assert server.policy_context_provider is deps.policy_context_provider
    assert server.environment_flags_provider is deps.environment_flags_provider
    assert server.websocket_stream_factory is deps.websocket_stream_factory


def test_mcp_server_registers_shutdown_family_through_injected_lifecycle_guard() -> None:
    from tldw_Server_API.app.core.MCP_unified.server import MCPServer

    deps = _fake_runtime_dependencies()
    MCPServer(dependencies=deps)

    assert deps.lifecycle_guard.registered
    registration = deps.lifecycle_guard.registered[-1]
    assert registration["family"] == "mcp.websocket"
    assert callable(registration["active_count"])
    assert callable(registration["drain"])


@pytest.mark.asyncio
async def test_mcp_server_drains_idempotency_before_module_registry_shutdown() -> None:
    from tldw_Server_API.app.core.MCP_unified.server import MCPServer

    events: list[str] = []
    deps = _real_server_runtime_dependencies()
    deps.module_registry = _RecordingModuleRegistry(events)
    server = MCPServer(dependencies=deps)
    idempotency = server.protocol._idempotency
    finalizer_started = asyncio.Event()
    release_finalizer = asyncio.Event()

    async def _finalize() -> str:
        finalizer_started.set()
        await release_finalizer.wait()
        events.append("idempotency")
        return "local"

    finalizer = idempotency._create_finalizer(_finalize, bound=1.0)
    assert finalizer is not None
    await asyncio.wait_for(finalizer_started.wait(), timeout=0.5)
    shutdown = asyncio.create_task(server.shutdown())
    await asyncio.sleep(0)
    events_before_release = list(events)
    release_finalizer.set()
    await asyncio.wait_for(shutdown, timeout=1.0)
    await asyncio.wait_for(finalizer, timeout=0.5)

    assert not hasattr(deps, "idempotency")
    assert events_before_release == []
    assert events == ["idempotency", "modules"]
    assert idempotency._finalizers == set()


@pytest.mark.asyncio
async def test_mcp_server_shutdown_contains_exotic_error_before_module_teardown() -> None:
    from tldw_Server_API.app.core.MCP_unified.server import MCPServer

    events: list[str] = []
    deps = _real_server_runtime_dependencies()
    deps.module_registry = _RecordingModuleRegistry(events)
    server = MCPServer(dependencies=deps)

    async def _fail_protocol_shutdown() -> None:
        events.append("idempotency")
        raise _ExoticShutdownError("private shutdown detail")

    server.protocol.shutdown = _fail_protocol_shutdown

    await server.shutdown()

    assert events == ["idempotency", "modules"]
    assert server.initialized is False


@pytest.mark.asyncio
async def test_mcp_server_shutdown_defers_repeated_cancellation_through_teardown() -> None:
    from tldw_Server_API.app.core.MCP_unified.server import MCPServer

    events: list[str] = []
    deps = _real_server_runtime_dependencies()
    deps.module_registry = _RecordingModuleRegistry(events)
    server = MCPServer(dependencies=deps)
    idempotency = server.protocol._idempotency
    finalizer_started = asyncio.Event()
    release_finalizer = asyncio.Event()
    protocol_started = asyncio.Event()
    protocol_tasks: list[asyncio.Task[Any]] = []
    module_tasks: list[asyncio.Task[Any]] = []

    async def _finalize() -> str:
        finalizer_started.set()
        await release_finalizer.wait()
        events.append("idempotency")
        return "local"

    original_protocol_shutdown = server.protocol.shutdown

    async def _shutdown_protocol() -> None:
        task = asyncio.current_task()
        assert task is not None
        protocol_tasks.append(task)
        events.append("protocol_start")
        protocol_started.set()
        await original_protocol_shutdown()
        events.append("protocol_done")

    async def _shutdown_modules() -> None:
        task = asyncio.current_task()
        assert task is not None
        module_tasks.append(task)
        events.append("modules")

    server.protocol.shutdown = _shutdown_protocol
    deps.module_registry.shutdown_all = _shutdown_modules
    finalizer = idempotency._create_finalizer(_finalize, bound=1.0)
    assert finalizer is not None
    await asyncio.wait_for(finalizer_started.wait(), timeout=0.5)
    shutdown = asyncio.create_task(server.shutdown())
    await asyncio.wait_for(protocol_started.wait(), timeout=0.5)

    shutdown.cancel("first cancellation")
    await asyncio.sleep(0)
    survived_first_cancellation = not shutdown.done()
    shutdown.cancel("second cancellation")
    await asyncio.sleep(0)
    survived_second_cancellation = not shutdown.done()
    release_finalizer.set()

    with pytest.raises(asyncio.CancelledError) as caught:
        await asyncio.wait_for(shutdown, timeout=1.0)
    await asyncio.wait_for(finalizer, timeout=0.5)

    assert caught.value.args == ("first cancellation",)
    assert survived_first_cancellation is True
    assert survived_second_cancellation is True
    assert events == ["protocol_start", "idempotency", "protocol_done", "modules"]
    assert protocol_tasks and protocol_tasks[0] is not shutdown
    assert module_tasks and module_tasks[0] is not shutdown
    assert all(task.done() for task in protocol_tasks + module_tasks)
    assert idempotency._finalizers == set()
    assert server.initialized is False


@pytest.mark.asyncio
async def test_mcp_server_shutdown_defers_cancellation_during_connection_close() -> None:
    from tldw_Server_API.app.core.MCP_unified.server import MCPServer

    events: list[str] = []
    deps = _real_server_runtime_dependencies()
    deps.module_registry = _RecordingModuleRegistry(events)
    server = MCPServer(dependencies=deps)
    close_started = asyncio.Event()
    release_close = asyncio.Event()
    cleanup_tasks: list[asyncio.Task[Any]] = []

    async def _close_connections() -> None:
        task = asyncio.current_task()
        assert task is not None
        cleanup_tasks.append(task)
        events.append("close_start")
        close_started.set()
        await release_close.wait()
        events.append("close_done")

    original_protocol_shutdown = server.protocol.shutdown

    async def _shutdown_protocol() -> None:
        events.append("protocol")
        await original_protocol_shutdown()

    async def _shutdown_modules() -> None:
        events.append("modules")

    server._close_all_connections = _close_connections
    server.protocol.shutdown = _shutdown_protocol
    deps.module_registry.shutdown_all = _shutdown_modules
    shutdown = asyncio.create_task(server.shutdown())
    await asyncio.wait_for(close_started.wait(), timeout=0.5)

    shutdown.cancel("close cancellation")
    await asyncio.sleep(0)
    survived_first_cancellation = not shutdown.done()
    shutdown.cancel("repeated cancellation")
    await asyncio.sleep(0)
    survived_second_cancellation = not shutdown.done()
    release_close.set()

    with pytest.raises(asyncio.CancelledError) as caught:
        await asyncio.wait_for(shutdown, timeout=1.0)

    assert caught.value.args == ("close cancellation",)
    assert survived_first_cancellation is True
    assert survived_second_cancellation is True
    assert events == ["close_start", "close_done", "protocol", "modules"]
    assert cleanup_tasks and cleanup_tasks[0] is not shutdown
    assert all(task.done() for task in cleanup_tasks)
    assert server.protocol._idempotency._finalizers == set()
    assert server.initialized is False


@pytest.mark.asyncio
async def test_owned_shutdown_helper_preserves_cancellation_and_child_failure() -> None:
    from tldw_Server_API.app.core.MCP_unified.server import _await_owned_shutdown_task

    child_started = asyncio.Event()
    release_child = asyncio.Event()
    child_tasks: list[asyncio.Task[None]] = []

    async def _fail_cleanup() -> None:
        child_started.set()
        await release_child.wait()
        raise _ExoticShutdownError("private cleanup detail")

    async def _wait_for_cleanup() -> tuple[asyncio.CancelledError | None, Exception | None]:
        child = asyncio.create_task(_fail_cleanup())
        child_tasks.append(child)
        return await _await_owned_shutdown_task(child, None)

    waiter = asyncio.create_task(_wait_for_cleanup())
    await asyncio.wait_for(child_started.wait(), timeout=0.5)
    release_child.set()
    await asyncio.sleep(0)
    assert child_tasks[0].done() is True
    waiter.cancel("original cancellation")

    cancellation, error = await asyncio.wait_for(waiter, timeout=0.5)

    assert cancellation is not None
    assert cancellation.args == ("original cancellation",)
    assert type(error) is _ExoticShutdownError
    assert child_tasks[0].done() is True


@pytest.mark.asyncio
async def test_public_shutdown_retries_failed_module_cleanup_single_flight() -> None:
    from tldw_Server_API.app.core.MCP_unified.server import MCPServer

    deps = _real_server_runtime_dependencies()
    server = MCPServer(dependencies=deps)
    server.initialized = True
    protocol_shutdown_calls = 0
    module_shutdown_calls = 0
    retry_started = asyncio.Event()
    release_retry = asyncio.Event()
    messages: list[str] = []

    original_protocol_shutdown = server.protocol.shutdown

    async def _shutdown_protocol() -> None:
        nonlocal protocol_shutdown_calls
        protocol_shutdown_calls += 1
        await original_protocol_shutdown()

    async def _shutdown_modules() -> None:
        nonlocal module_shutdown_calls
        module_shutdown_calls += 1
        if module_shutdown_calls == 1:
            raise _ExoticShutdownError("private module shutdown detail")
        retry_started.set()
        await release_retry.wait()

    from loguru import logger

    server.protocol.shutdown = _shutdown_protocol
    deps.module_registry.shutdown_all = _shutdown_modules
    sink_id = logger.add(lambda message: messages.append(str(message)), format="{message}")
    retries: list[asyncio.Task[None]] = []
    try:
        await server.shutdown()
        await asyncio.sleep(0)
        first_resource_task = server._resource_shutdown_task

        assert first_resource_task is not None
        assert first_resource_task.done() is True
        resource_error = first_resource_task.exception()
        assert resource_error is not None
        assert type(resource_error).__name__ == "_ModuleShutdownFailure"
        assert str(resource_error) == ""
        assert resource_error.__cause__ is None
        assert resource_error.__context__ is None
        assert server.initialized is False
        assert server._module_shutdown_task is None
        assert server._module_shutdown_complete is False
        assert protocol_shutdown_calls == 1
        assert module_shutdown_calls == 1

        retries = [
            asyncio.create_task(server.shutdown()),
            asyncio.create_task(server.shutdown()),
        ]
        await asyncio.wait_for(retry_started.wait(), timeout=0.5)
        assert protocol_shutdown_calls == 2
        assert module_shutdown_calls == 2
        assert all(not task.done() for task in retries)
        release_retry.set()
        await asyncio.wait_for(asyncio.gather(*retries), timeout=0.5)
        await asyncio.sleep(0)
    finally:
        release_retry.set()
        await asyncio.gather(*retries, return_exceptions=True)
        logger.remove(sink_id)

    assert server._resource_shutdown_task is not first_resource_task
    assert server._resource_shutdown_task is not None
    assert server._resource_shutdown_task.done() is True
    assert server._module_shutdown_task is None
    assert server._module_shutdown_complete is True
    assert protocol_shutdown_calls == 2
    assert module_shutdown_calls == 2
    assert all("private module shutdown detail" not in message for message in messages)
    assert any("error_type=_ExoticShutdownError" in message for message in messages)


@pytest.mark.asyncio
async def test_deferred_module_shutdown_failure_retries_after_shared_resource_success() -> None:
    from tldw_Server_API.app.core.MCP_unified.server import MCPServer

    deps = _real_server_runtime_dependencies()
    server = MCPServer(dependencies=deps)
    manager = server.protocol._idempotency
    finalizer_started = asyncio.Event()
    release_finalizer = asyncio.Event()
    first_module_attempted = asyncio.Event()
    retry_started = asyncio.Event()
    release_retry = asyncio.Event()
    protocol_shutdown_calls = 0
    module_shutdown_calls = 0

    async def _finalize() -> str:
        finalizer_started.set()
        while not release_finalizer.is_set():
            try:
                await release_finalizer.wait()
            except asyncio.CancelledError:
                continue
        return "local"

    original_protocol_shutdown = server.protocol.shutdown

    async def _shutdown_protocol() -> None:
        nonlocal protocol_shutdown_calls
        protocol_shutdown_calls += 1
        await original_protocol_shutdown()

    async def _shutdown_modules() -> None:
        nonlocal module_shutdown_calls
        module_shutdown_calls += 1
        if module_shutdown_calls == 1:
            first_module_attempted.set()
            raise _ExoticShutdownError("private deferred module detail")
        retry_started.set()
        await release_retry.wait()

    server.protocol.shutdown = _shutdown_protocol
    deps.module_registry.shutdown_all = _shutdown_modules
    finalizer = manager._create_finalizer(_finalize, bound=0.01)
    await asyncio.wait_for(finalizer_started.wait(), timeout=0.5)
    retries: list[asyncio.Task[None]] = []
    try:
        await asyncio.wait_for(server.shutdown(), timeout=0.5)
        resource_task = server._resource_shutdown_task

        assert resource_task is not None
        assert resource_task.done() is True
        assert protocol_shutdown_calls == 1
        assert module_shutdown_calls == 0
        assert manager.has_pending_shutdown_work is True

        release_finalizer.set()
        await asyncio.wait_for(finalizer, timeout=0.5)
        await asyncio.wait_for(first_module_attempted.wait(), timeout=0.5)
        await asyncio.sleep(0)

        assert server._resource_shutdown_task is resource_task
        assert server._module_shutdown_task is None
        assert server._module_shutdown_complete is False

        retries = [
            asyncio.create_task(server.shutdown()),
            asyncio.create_task(server.shutdown()),
        ]
        await asyncio.wait_for(retry_started.wait(), timeout=0.5)
        assert protocol_shutdown_calls == 1
        assert module_shutdown_calls == 2
        release_retry.set()
        await asyncio.wait_for(asyncio.gather(*retries), timeout=0.5)
        await asyncio.sleep(0)
    finally:
        release_finalizer.set()
        release_retry.set()
        await asyncio.gather(finalizer, *retries, return_exceptions=True)

    assert protocol_shutdown_calls == 1
    assert module_shutdown_calls == 2
    assert manager.has_pending_shutdown_work is False
    assert server._module_shutdown_task is None
    assert server._module_shutdown_complete is True


@pytest.mark.asyncio
async def test_mcp_server_initialize_uses_injected_permission_seeder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.MCP_unified.server import MCPServer

    monkeypatch.setenv("MCP_ENABLE_FILESYSTEM_MODULE", "0")
    monkeypatch.setenv("MCP_MODULES_CONFIG", "/tmp/mcp-stage3-no-modules.yaml")
    monkeypatch.setattr(
        "tldw_Server_API.app.core.MCP_unified.server.validate_config",
        lambda: True,
    )
    deps = _server_runtime_dependencies()
    server = MCPServer(dependencies=deps)
    server.config.metrics_enabled = False

    await server.initialize()

    assert deps.permission_seeder.calls == 1


@pytest.mark.asyncio
async def test_mcp_server_default_media_module_path_uses_injected_module_config_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.MCP_unified.server import MCPServer

    monkeypatch.setenv("MCP_ENABLE_MEDIA_MODULE", "1")
    monkeypatch.setenv("MCP_ENABLE_FILESYSTEM_MODULE", "0")
    monkeypatch.setenv("MCP_MODULES_CONFIG", "/tmp/mcp-stage3-no-modules.yaml")
    deps = _server_runtime_dependencies()
    deps.environment_flags_provider = _FakeEnvironmentFlagsProvider(
        flags={"MCP_ENABLE_MEDIA_MODULE": True}
    )
    server = MCPServer(dependencies=deps)

    await server._register_default_modules()  # noqa: SLF001

    media_registration = next(
        row for row in deps.module_registry.registrations if row["module_id"] == "media"
    )
    assert media_registration["config"].settings["db_path"] == (
        deps.module_config_provider.media_db_path
    )


@pytest.mark.asyncio
async def test_mcp_server_default_module_configs_use_injected_circuit_breaker_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.MCP_unified.server import MCPServer

    monkeypatch.setenv("MCP_ENABLE_MEDIA_MODULE", "1")
    monkeypatch.setenv("MCP_ENABLE_FILESYSTEM_MODULE", "0")
    monkeypatch.setenv("MCP_MODULES_CONFIG", "/tmp/mcp-stage3-no-modules.yaml")
    deps = _server_runtime_dependencies()
    deps.environment_flags_provider = _FakeEnvironmentFlagsProvider(
        flags={"MCP_ENABLE_MEDIA_MODULE": True}
    )
    server = MCPServer(dependencies=deps)

    await server._register_default_modules()  # noqa: SLF001

    assert deps.module_registry.registrations
    for registration in deps.module_registry.registrations:
        assert registration["config"].circuit_breaker_factory is deps.circuit_breaker_factory


def test_mcp_server_uses_injected_auth_and_policy_context_helpers() -> None:
    from tldw_Server_API.app.core.MCP_unified.server import MCPServer

    deps = _server_runtime_dependencies()
    server = MCPServer(dependencies=deps)

    assert server._extract_api_key_permissions({"scopes": "ignored"}) == ["fake.scope"]  # noqa: SLF001
    assert server._policy_context_enabled() is False  # noqa: SLF001


def test_websocket_stream_factory_declares_returned_stream_contract() -> None:
    from mcp_unified.interfaces import runtime as standalone_runtime

    hints = get_type_hints(standalone_runtime.WebSocketStreamFactory.__call__)

    assert hints["return"] is standalone_runtime.WebSocketStream
    assert hasattr(standalone_runtime.WebSocketStream, "start")
    assert hasattr(standalone_runtime.WebSocketStream, "stop")
    assert hasattr(standalone_runtime.WebSocketStream, "mark_activity")
    assert hasattr(standalone_runtime.WebSocketStream, "send_json")


def test_mcp_server_websocket_stream_annotations_use_runtime_contract() -> None:
    from tldw_Server_API.app.core.MCP_unified.interfaces import WebSocketStream
    from tldw_Server_API.app.core.MCP_unified.server import MCPServer

    create_hints = get_type_hints(MCPServer._create_websocket_stream)
    handle_hints = get_type_hints(MCPServer._handle_websocket_messages)

    assert create_hints["return"] is WebSocketStream
    assert handle_hints["stream"] is WebSocketStream
    assert handle_hints["return"] is type(None)


def test_mcp_server_uses_injected_environment_flags_provider() -> None:
    from tldw_Server_API.app.core.MCP_unified.server import MCPServer

    deps = _server_runtime_dependencies()
    deps.environment_flags_provider = _FakeEnvironmentFlagsProvider(
        flags={"MCP_ENABLE_DEMO_AUTH": True},
        test_mode=True,
        explicit_pytest_runtime=True,
        truthy_values={"yes": True},
    )
    server = MCPServer(dependencies=deps)

    assert server._env_flag_enabled("MCP_ENABLE_DEMO_AUTH") is True  # noqa: SLF001
    assert server._is_test_mode() is True  # noqa: SLF001
    assert server._is_explicit_pytest_runtime() is True  # noqa: SLF001
    assert server._is_truthy("yes") is True  # noqa: SLF001


def test_mcp_server_uses_injected_websocket_stream_factory() -> None:
    from tldw_Server_API.app.core.MCP_unified.server import MCPServer

    deps = _server_runtime_dependencies()
    factory = _RecordingWebSocketStreamFactory()
    deps.websocket_stream_factory = factory
    server = MCPServer(dependencies=deps)
    server.config.ws_ping_interval = 7
    server.config.ws_idle_timeout_seconds = 11
    websocket = object()

    stream = server._create_websocket_stream(websocket)  # noqa: SLF001

    assert stream is factory.stream
    assert factory.calls == [
        {
            "websocket": websocket,
            "heartbeat_interval_s": 7.0,
            "idle_timeout_s": 11.0,
            "close_on_done": True,
            "labels": {"component": "mcp", "endpoint": "mcp_ws"},
        }
    ]


def test_mcp_server_logs_host_adapter_fallbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.MCP_unified import server as server_mod
    from tldw_Server_API.app.core.MCP_unified.server import MCPServer

    class _FailingAuthProvider(_FakeAuthProvider):
        def normalize_api_key_permissions(self, info: dict[str, Any] | None) -> list[str]:
            del info
            raise RuntimeError("scope failure")

    class _FailingPolicyContextProvider:
        def is_policy_context_enabled(self) -> bool:
            raise RuntimeError("policy failure")

    class _FailingModuleConfigProvider:
        def default_media_db_path(self) -> str:
            raise RuntimeError("path failure")

    warnings: list[str] = []

    def _record_warning(message: str, *args: Any, **_kwargs: Any) -> None:
        warnings.append(message.format(*args))

    monkeypatch.setattr(server_mod.logger, "warning", _record_warning)
    deps = _server_runtime_dependencies()
    deps.auth_provider = _FailingAuthProvider()
    deps.policy_context_provider = _FailingPolicyContextProvider()
    deps.module_config_provider = _FailingModuleConfigProvider()
    server = MCPServer(dependencies=deps)

    assert server._extract_api_key_permissions({"scopes": "ignored"}) == []  # noqa: SLF001
    assert server._policy_context_enabled() is False  # noqa: SLF001
    assert server._default_media_db_path() == ""  # noqa: SLF001
    assert warnings == [
        "MCP API-key permission normalization failed: scope failure",
        "MCP policy-context flag resolution failed: policy failure",
        "MCP default media DB path resolution failed: path failure",
    ]


@pytest.mark.asyncio
async def test_tldw_permission_seeder_uses_shared_rbac_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.MCP_unified.adapters import tldw_runtime

    class _TransactionContext:
        def __init__(self, conn: object) -> None:
            self.conn = conn
            self.entered = False
            self.exited = False

        async def __aenter__(self) -> object:
            self.entered = True
            return self.conn

        async def __aexit__(self, *_exc_info: Any) -> None:
            self.exited = True
            return None

    class _Pool:
        def __init__(self) -> None:
            self.conn = object()
            self.context = _TransactionContext(self.conn)

        def acquire(self) -> None:
            raise AssertionError("permission seed must not use an unowned connection")

        def transaction(self) -> _TransactionContext:
            return self.context

    pool = _Pool()
    ensure_calls: list[tuple[object, bool, bool]] = []

    async def _get_db_pool() -> _Pool:
        return pool

    async def _ensure_baseline_rbac_seed(
        db: object,
        *,
        include_mcp_permissions: bool,
        is_postgres: bool | None = None,
    ) -> None:
        ensure_calls.append((db, include_mcp_permissions, bool(is_postgres)))

    monkeypatch.setattr(
        "tldw_Server_API.app.core.AuthNZ.database.get_db_pool",
        _get_db_pool,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.AuthNZ.rbac_seed.ensure_baseline_rbac_seed",
        _ensure_baseline_rbac_seed,
    )

    await tldw_runtime.TldwPermissionSeeder().seed_default_tool_permissions()

    assert ensure_calls == [
        (pool.conn, True, False)
    ]
    assert pool.context.entered is True
    assert pool.context.exited is True


@pytest.mark.asyncio
async def test_tldw_permission_seeder_propagates_failure_and_rolls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.MCP_unified.adapters import tldw_runtime

    class _TransactionContext:
        def __init__(self, pool: _Pool) -> None:
            self.pool = pool
            self.snapshot: list[str] = []

        async def __aenter__(self) -> object:
            self.snapshot = list(self.pool.rows)
            return self.pool.conn

        async def __aexit__(self, exc_type: object, *_exc_info: Any) -> None:
            if exc_type is not None:
                self.pool.rows[:] = self.snapshot
                self.pool.rolled_back = True
            return None

    class _Pool:
        def __init__(self) -> None:
            self.conn = object()
            self.rows: list[str] = []
            self.rolled_back = False

        def acquire(self) -> None:
            raise AssertionError("permission seed must not use an unowned connection")

        def transaction(self) -> _TransactionContext:
            return _TransactionContext(self)

    pool = _Pool()

    async def _get_db_pool() -> _Pool:
        return pool

    async def _failing_seed(*_args: Any, **_kwargs: Any) -> None:
        pool.rows.append("partial-catalog")
        raise RuntimeError("forced seed failure")

    monkeypatch.setattr(
        "tldw_Server_API.app.core.AuthNZ.database.get_db_pool",
        _get_db_pool,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.AuthNZ.rbac_seed.ensure_baseline_rbac_seed",
        _failing_seed,
    )

    with pytest.raises(RuntimeError, match="forced seed failure"):
        await tldw_runtime.TldwPermissionSeeder().seed_default_tool_permissions()

    assert pool.rows == []
    assert pool.rolled_back is True


def test_authnz_access_token_helper_documents_boolean_semantics() -> None:
    from tldw_Server_API.app.core.MCP_unified.server import _is_authnz_access_token

    doc = inspect.getdoc(_is_authnz_access_token)

    assert doc is not None
    assert "Return True" in doc
    assert "MCP AuthNZ access token" in doc


def test_default_server_protocol_uses_current_telemetry_manager(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.MCP_unified.adapters import tldw_runtime
    from tldw_Server_API.app.core.MCP_unified.server import MCPServer

    first_manager = _TelemetryManagerDouble("first")
    second_manager = _TelemetryManagerDouble("second")
    current = {"manager": first_manager}

    monkeypatch.setattr(tldw_runtime, "get_telemetry_manager", lambda: current["manager"])

    server = MCPServer()

    with server.protocol.telemetry.trace_context("first-op", {"generation": 1}) as span:
        assert span == "first"
    current["manager"] = second_manager
    with server.protocol.telemetry.trace_context("second-op", {"generation": 2}) as span:
        assert span == "second"
    assert first_manager.trace_calls == [("first-op", {"generation": 1})]
    assert second_manager.trace_calls == [("second-op", {"generation": 2})]


def test_protocol_instances_do_not_share_prepared_call_secrets() -> None:
    from tldw_Server_API.app.core.MCP_unified.protocol import MCPProtocol

    first = MCPProtocol()
    second = MCPProtocol()
    assert first is not second
    assert first._prepared_call_secret != second._prepared_call_secret  # noqa: SLF001


def test_protocol_reexports_tool_execution_shared_symbols() -> None:
    from tldw_Server_API.app.core.MCP_unified import protocol
    from tldw_Server_API.app.core.MCP_unified.modules.base import BaseModule

    expected = {
        "AuthenticatedExecutionScope",
        "RequestContext",
        "PreparedToolCall",
        "PreparedExecutionPolicy",
        "InvalidParamsException",
        "GovernanceDeniedError",
        "ApprovalRequiredError",
        "IdempotencyManager",
        "ToolExecutionCoordinator",
        "ToolExecutionDependencies",
        "ToolExecutionReporter",
        "ToolExecutionRuntime",
        "ToolExecutionSecurity",
        "_trusted_compat_claims_metadata",
    }

    missing = sorted(name for name in expected if not hasattr(protocol, name))
    assert missing == [], f"protocol.py dropped compatibility exports: {missing}"

    hints = get_type_hints(protocol.PreparedToolCall)
    assert hints["module"] is BaseModule


def test_idempotency_manager_is_extracted_with_protocol_compatibility() -> None:
    from tldw_Server_API.app.core.MCP_unified import protocol

    protocol_classes = _defined_class_names_for(MCP_ROOT / "protocol.py")

    assert "IdempotencyManager" not in protocol_classes, (
        "protocol.py must only re-export the extracted IdempotencyManager"
    )

    from tldw_Server_API.app.core.MCP_unified.tool_execution import IdempotencyManager, idempotency

    assert protocol.IdempotencyManager is IdempotencyManager
    assert inspect.getmodule(protocol.IdempotencyManager) is idempotency


def test_protocol_class_scan_detects_nested_idempotency_manager(tmp_path: Path) -> None:
    sample = tmp_path / "nested_idempotency_manager.py"
    sample.write_text(
        "def build_manager():\n"
        "    class IdempotencyManager:\n"
        "        pass\n"
        "    return IdempotencyManager\n",
        encoding="utf-8",
    )

    assert "IdempotencyManager" in _defined_class_names_for(sample)


def test_tool_execution_package_does_not_import_protocol_facade() -> None:
    package_dir = MCP_ROOT / "tool_execution"
    assert package_dir.is_dir()
    forbidden = {
        f"{MCP_PACKAGE}.protocol",
        "tldw_Server_API.app.core.MCP_unified.protocol",
    }
    offenders: dict[str, list[str]] = {}

    def package_for(path: Path) -> str:
        relative_parent = path.relative_to(MCP_ROOT).parent
        return ".".join((MCP_PACKAGE, *relative_parent.parts))

    for path in package_dir.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        current_package = package_for(path)
        imports = _resolved_import_sources_for(path, current_package)
        bad_imports = [
            source
            for source in imports
            if source in forbidden
            or source.endswith(".MCP_unified.protocol")
        ]
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                source = _resolve_import_from_source(current_package, node.module, node.level)
                if source == MCP_PACKAGE and any(alias.name == "protocol" for alias in node.names):
                    bad_imports.append(f"{source}.protocol")
        if bad_imports:
            offenders[str(path.relative_to(MCP_ROOT))] = bad_imports

        uses_facade = [
            "MCPProtocol"
            for node in ast.walk(tree)
            if (
                isinstance(node, ast.Name)
                and node.id == "MCPProtocol"
            )
            or (
                isinstance(node, ast.Attribute)
                and node.attr == "MCPProtocol"
            )
        ]
        if uses_facade:
            offenders.setdefault(str(path.relative_to(MCP_ROOT)), []).append("MCPProtocol")

    assert offenders == {}, (
        "tool_execution must not depend on the protocol facade; offenders: "
        f"{offenders}"
    )


def test_protocol_has_no_expected_failure_or_domain_specific_error_result_branch() -> None:
    violations = _protocol_expected_failure_branch_violations_for(MCP_ROOT / "protocol.py")

    assert violations == [], (
        "Expected tool failures belong in the host-neutral execution runtime, not protocol.py: "
        f"{violations}"
    )


def test_protocol_branch_scan_detects_domain_specific_error_result(tmp_path: Path) -> None:
    unsafe = tmp_path / "unsafe_protocol.py"
    unsafe.write_text(
        "def handle(tool_name):\n"
        "    if tool_name == 'skills.run':\n"
        "        return {'isError': True}\n"
        "    return {}\n",
        encoding="utf-8",
    )
    safe = tmp_path / "safe_protocol.py"
    safe.write_text(
        "def configure(model_validator):\n"
        "    if model_validator is not None:\n"
        "        return model_validator\n"
        "    return None\n",
        encoding="utf-8",
    )

    assert _protocol_expected_failure_branch_violations_for(safe) == []
    violations = _protocol_expected_failure_branch_violations_for(unsafe)
    assert len(violations) == 1
    assert "Skills/model/provider" in violations[0]


def test_protocol_branch_scan_detects_prebuilt_error_result_alias(
    tmp_path: Path,
) -> None:
    sample = tmp_path / "prebuilt_error_result.py"
    sample.write_text(
        "def handle(tool_name):\n"
        "    error_result = {'isError': True}\n"
        "    if tool_name == 'skills.run':\n"
        "        return error_result\n"
        "    return {}\n",
        encoding="utf-8",
    )

    violations = _protocol_expected_failure_branch_violations_for(sample)

    assert len(violations) == 1
    assert "Skills/model/provider" in violations[0]


def test_protocol_branch_scan_allows_prebuilt_non_error_result_alias(
    tmp_path: Path,
) -> None:
    sample = tmp_path / "prebuilt_success_result.py"
    sample.write_text(
        "def handle(tool_name):\n"
        "    success_result = {'content': []}\n"
        "    if tool_name == 'skills.run':\n"
        "        return success_result\n"
        "    return {}\n",
        encoding="utf-8",
    )

    assert _protocol_expected_failure_branch_violations_for(sample) == []


@pytest.mark.parametrize(
    ("filename", "error_scope"),
    [
        (
            "function_scope.py",
            "def build_error():\n"
            "    result = {'isError': True}\n"
            "    return result\n",
        ),
        (
            "lambda_scope.py",
            "build_error = lambda: (result := {'isError': True})\n",
        ),
        (
            "class_scope.py",
            "class ErrorFactory:\n"
            "    result = {'isError': True}\n",
        ),
    ],
)
def test_protocol_branch_scan_isolates_expected_error_aliases_by_lexical_scope(
    tmp_path: Path,
    filename: str,
    error_scope: str,
) -> None:
    sample = tmp_path / filename
    sample.write_text(
        f"{error_scope}"
        "def handle(tool_name):\n"
        "    result = {'content': []}\n"
        "    if tool_name == 'skills.run':\n"
        "        return result\n"
        "    return {}\n",
        encoding="utf-8",
    )

    assert _protocol_expected_failure_branch_violations_for(sample) == []


@pytest.mark.parametrize(
    ("filename", "source"),
    [
        (
            "identifier_domain.py",
            "SKILLS_TOOLS = {'skills.run'}\n"
            "def handle(tool_name):\n"
            "    if tool_name in SKILLS_TOOLS:\n"
            "        return {'isError': True}\n"
            "    return {}\n",
        ),
        (
            "match_pattern_domain.py",
            "def handle(tool_name):\n"
            "    match tool_name:\n"
            "        case 'model.run':\n"
            "            return {'isError': True}\n"
            "    return {}\n",
        ),
        (
            "match_guard_domain.py",
            "PROVIDER_TOOLS = {'provider.run'}\n"
            "def handle(tool_name):\n"
            "    match tool_name:\n"
            "        case candidate if candidate in PROVIDER_TOOLS:\n"
            "            return {'isError': True}\n"
            "    return {}\n",
        ),
    ],
)
def test_protocol_branch_scan_detects_domain_identifiers_patterns_and_guards(
    tmp_path: Path,
    filename: str,
    source: str,
) -> None:
    sample = tmp_path / filename
    sample.write_text(source, encoding="utf-8")

    violations = _protocol_expected_failure_branch_violations_for(sample)

    assert len(violations) == 1
    assert "Skills/model/provider" in violations[0]


@pytest.mark.parametrize(
    ("filename", "source"),
    [
        (
            "camel_domain.py",
            "def handle(tool_name):\n"
            "    if tool_name in SkillsRunnerTool:\n"
            "        return {'isError': True}\n"
            "    return {}\n",
        ),
        (
            "attribute_domain.py",
            "def handle(tool_name):\n"
            "    if tool_name in catalog.ProviderRunnerTool:\n"
            "        return {'isError': True}\n"
            "    return {}\n",
        ),
        (
            "domain_exception.py",
            "def handle():\n"
            "    try:\n"
            "        run_tool()\n"
            "    except SkillsRunnerError:\n"
            "        return {'isError': True}\n"
            "    return {}\n",
        ),
    ],
)
def test_protocol_branch_scan_detects_camel_attributes_and_domain_exceptions(
    tmp_path: Path,
    filename: str,
    source: str,
) -> None:
    sample = tmp_path / filename
    sample.write_text(source, encoding="utf-8")

    violations = _protocol_expected_failure_branch_violations_for(sample)

    assert len(violations) == 1
    assert "Skills/model/provider" in violations[0]


def test_protocol_branch_scan_uses_identifier_word_boundaries(tmp_path: Path) -> None:
    sample = tmp_path / "non_domain_identifier.py"
    sample.write_text(
        "def handle(tool_name):\n"
        "    if tool_name in ModelingRunnerTools:\n"
        "        return {'isError': True}\n"
        "    return {}\n",
        encoding="utf-8",
    )

    assert _protocol_expected_failure_branch_violations_for(sample) == []


def test_runtime_security_decisions_only_read_immutable_prepared_policy() -> None:
    violations = _runtime_mutable_authority_violations_for(
        MCP_ROOT / "tool_execution" / "runtime.py"
    )

    assert violations == [], (
        "execute_prepared_tool_call must not reread mutable metadata, schema, config, "
        f"idempotency, or effect fields: {violations}"
    )


def test_runtime_policy_scan_distinguishes_typed_policy_from_mutable_reads(
    tmp_path: Path,
) -> None:
    safe = tmp_path / "safe_runtime.py"
    safe.write_text(
        "async def execute_prepared_tool_call(prepared):\n"
        "    policy = prepared.policy\n"
        "    return (policy.effect, policy.rate_limit_fail_closed, "
        "policy.idempotency.ttl_seconds)\n",
        encoding="utf-8",
    )
    unsafe = tmp_path / "unsafe_runtime.py"
    unsafe.write_text(
        "async def execute_prepared_tool_call(prepared):\n"
        "    category = prepared.tool_def.get('metadata', {}).get('category')\n"
        "    return prepared.is_write, category\n",
        encoding="utf-8",
    )

    assert _runtime_mutable_authority_violations_for(safe) == []
    violations = _runtime_mutable_authority_violations_for(unsafe)
    assert len(violations) == 3
    assert any("prepared.policy" in violation for violation in violations)
    assert any("'category'" in violation for violation in violations)
    assert any("prepared.is_write" in violation for violation in violations)


def test_runtime_policy_scan_rejects_policy_bound_to_mutable_metadata(
    tmp_path: Path,
) -> None:
    sample = tmp_path / "mutable_policy_alias.py"
    sample.write_text(
        "async def execute_prepared_tool_call(prepared):\n"
        "    policy = prepared.tool_def['metadata']\n"
        "    return policy.effect\n",
        encoding="utf-8",
    )

    violations = _runtime_mutable_authority_violations_for(sample)

    assert any("prepared.policy" in violation for violation in violations)


def test_runtime_policy_scan_rejects_policy_rebinding(tmp_path: Path) -> None:
    sample = tmp_path / "rebound_policy.py"
    sample.write_text(
        "async def execute_prepared_tool_call(prepared):\n"
        "    policy = prepared.policy\n"
        "    policy = prepared.policy\n"
        "    return policy.effect\n",
        encoding="utf-8",
    )

    violations = _runtime_mutable_authority_violations_for(sample)

    assert any("reassigns policy" in violation for violation in violations)


def test_runtime_policy_scan_rejects_nested_policy_shadowing(tmp_path: Path) -> None:
    sample = tmp_path / "shadowed_policy.py"
    sample.write_text(
        "async def execute_prepared_tool_call(prepared):\n"
        "    policy = prepared.policy\n"
        "    def effect_from_metadata():\n"
        "        policy = prepared.tool_def['metadata']\n"
        "        return policy.effect\n"
        "    return effect_from_metadata()\n",
        encoding="utf-8",
    )

    violations = _runtime_mutable_authority_violations_for(sample)

    assert any("reassigns policy" in violation for violation in violations)


@pytest.mark.parametrize(
    ("filename", "source"),
    [
        (
            "nested_parameter_shadow.py",
            "async def execute_prepared_tool_call(prepared):\n"
            "    policy = prepared.policy\n"
            "    def inner(policy=prepared.tool_def['metadata']):\n"
            "        return policy.effect\n"
            "    return inner()\n",
        ),
        (
            "lambda_parameter_shadow.py",
            "async def execute_prepared_tool_call(prepared):\n"
            "    policy = prepared.policy\n"
            "    inner = lambda policy=prepared.tool_def['metadata']: policy.effect\n"
            "    return inner()\n",
        ),
        (
            "except_alias_shadow.py",
            "async def execute_prepared_tool_call(prepared):\n"
            "    policy = prepared.policy\n"
            "    try:\n"
            "        raise RuntimeError\n"
            "    except Exception as policy:\n"
            "        return policy.effect\n",
        ),
    ],
)
def test_runtime_policy_scan_rejects_parameter_and_except_alias_shadowing(
    tmp_path: Path,
    filename: str,
    source: str,
) -> None:
    sample = tmp_path / filename
    sample.write_text(source, encoding="utf-8")

    violations = _runtime_mutable_authority_violations_for(sample)

    assert any("reassigns policy" in violation for violation in violations)


@pytest.mark.parametrize(
    ("filename", "expression"),
    [
        ("dynamic_subscript.py", "prepared.tool_def['metadata'][field]"),
        ("dynamic_get.py", "prepared.tool_def['metadata'].get(field)"),
        ("dynamic_getattr.py", "getattr(prepared.tool_def['metadata'], field)"),
    ],
)
def test_runtime_policy_scan_rejects_dynamic_prepared_authority_reads(
    tmp_path: Path,
    filename: str,
    expression: str,
) -> None:
    sample = tmp_path / filename
    sample.write_text(
        "async def execute_prepared_tool_call(prepared):\n"
        "    policy = prepared.policy\n"
        "    field = 'category'\n"
        f"    return {expression}\n",
        encoding="utf-8",
    )

    violations = _runtime_mutable_authority_violations_for(sample)

    assert any("dynamic prepared authority" in violation for violation in violations)


def test_runtime_policy_scan_allows_unrelated_dynamic_lookup(tmp_path: Path) -> None:
    sample = tmp_path / "unrelated_dynamic_lookup.py"
    sample.write_text(
        "async def execute_prepared_tool_call(prepared):\n"
        "    policy = prepared.policy\n"
        "    field = 'category'\n"
        "    payload = {'category': 'safe'}\n"
        "    return payload[field], policy.effect\n",
        encoding="utf-8",
    )

    assert _runtime_mutable_authority_violations_for(sample) == []


@pytest.mark.parametrize(
    ("filename", "assignments", "expression"),
    [
        (
            "tool_def_alias.py",
            "    tool_def = prepared.tool_def\n",
            "tool_def[field]",
        ),
        (
            "metadata_alias.py",
            "    metadata = prepared.tool_def['metadata']\n",
            "metadata.get(field)",
        ),
        (
            "authority_alias_chain.py",
            "    tool_def = prepared.tool_def\n"
            "    metadata = tool_def['metadata']\n",
            "getattr(metadata, field)",
        ),
    ],
)
def test_runtime_policy_scan_rejects_dynamic_prepared_authority_aliases(
    tmp_path: Path,
    filename: str,
    assignments: str,
    expression: str,
) -> None:
    sample = tmp_path / filename
    sample.write_text(
        "async def execute_prepared_tool_call(prepared):\n"
        "    policy = prepared.policy\n"
        f"{assignments}"
        "    field = 'category'\n"
        f"    return {expression}\n",
        encoding="utf-8",
    )

    violations = _runtime_mutable_authority_violations_for(sample)

    assert any("dynamic prepared authority" in violation for violation in violations)


def test_runtime_policy_scan_rejects_conditional_authority_alias_reassignment(
    tmp_path: Path,
) -> None:
    sample = tmp_path / "conditional_authority_alias.py"
    sample.write_text(
        "async def execute_prepared_tool_call(prepared, cond):\n"
        "    policy = prepared.policy\n"
        "    metadata = prepared.tool_def['metadata']\n"
        "    if cond:\n"
        "        metadata = {}\n"
        "    field = 'category'\n"
        "    return metadata[field], policy.effect\n",
        encoding="utf-8",
    )

    violations = _runtime_mutable_authority_violations_for(sample)

    assert any("dynamic prepared authority" in violation for violation in violations)


def test_runtime_policy_scan_allows_distinct_untainted_dynamic_lookup(
    tmp_path: Path,
) -> None:
    sample = tmp_path / "distinct_untainted_authority_alias.py"
    sample.write_text(
        "async def execute_prepared_tool_call(prepared):\n"
        "    policy = prepared.policy\n"
        "    metadata = prepared.tool_def['metadata']\n"
        "    payload = {'category': 'safe'}\n"
        "    field = 'category'\n"
        "    return payload[field], policy.effect\n",
        encoding="utf-8",
    )

    assert _runtime_mutable_authority_violations_for(sample) == []


@pytest.mark.parametrize(
    ("filename", "imports", "lookup"),
    [
        ("operator_module.py", "import operator as op\n", "op.getitem"),
        (
            "operator_import_alias.py",
            "from operator import getitem as read_item\n",
            "read_item",
        ),
    ],
)
def test_runtime_policy_scan_rejects_operator_getitem_authority_reads(
    tmp_path: Path,
    filename: str,
    imports: str,
    lookup: str,
) -> None:
    sample = tmp_path / filename
    sample.write_text(
        f"{imports}"
        "async def execute_prepared_tool_call(prepared):\n"
        "    policy = prepared.policy\n"
        f"    metadata = {lookup}(prepared.tool_def, 'metadata')\n"
        "    field = 'category'\n"
        f"    return {lookup}(metadata, field), policy.effect\n",
        encoding="utf-8",
    )

    violations = _runtime_mutable_authority_violations_for(sample)

    assert any("dynamic prepared authority" in violation for violation in violations)


def test_runtime_policy_scan_rejects_destructured_authority_alias(
    tmp_path: Path,
) -> None:
    sample = tmp_path / "destructured_authority_alias.py"
    sample.write_text(
        "async def execute_prepared_tool_call(prepared):\n"
        "    policy = prepared.policy\n"
        "    metadata, unused = prepared.tool_def['metadata'], {}\n"
        "    field = 'category'\n"
        "    return metadata[field], policy.effect\n",
        encoding="utf-8",
    )

    violations = _runtime_mutable_authority_violations_for(sample)

    assert any("dynamic prepared authority" in violation for violation in violations)


@pytest.mark.parametrize(
    ("filename", "source"),
    [
        (
            "unrelated_operator_lookup.py",
            "import operator\n"
            "async def execute_prepared_tool_call(prepared):\n"
            "    policy = prepared.policy\n"
            "    payload = {'category': 'safe'}\n"
            "    field = 'category'\n"
            "    return operator.getitem(payload, field), policy.effect\n",
        ),
        (
            "unrelated_getitem_function.py",
            "def getitem(mapping, key):\n"
            "    return mapping[key]\n"
            "async def execute_prepared_tool_call(prepared):\n"
            "    policy = prepared.policy\n"
            "    metadata = getitem(prepared.tool_def, 'metadata')\n"
            "    field = 'category'\n"
            "    return getitem(metadata, field), policy.effect\n",
        ),
        (
            "unrelated_destructuring.py",
            "async def execute_prepared_tool_call(prepared):\n"
            "    policy = prepared.policy\n"
            "    payload, unused = {'category': 'safe'}, {}\n"
            "    field = 'category'\n"
            "    return payload[field], policy.effect\n",
        ),
    ],
)
def test_runtime_policy_scan_allows_unrelated_getitem_and_destructuring(
    tmp_path: Path,
    filename: str,
    source: str,
) -> None:
    sample = tmp_path / filename
    sample.write_text(source, encoding="utf-8")

    assert _runtime_mutable_authority_violations_for(sample) == []


def test_runtime_has_no_default_str_serialization_fallback() -> None:
    violations = _default_str_violations_for(MCP_ROOT / "tool_execution" / "runtime.py")

    assert violations == [], (
        "runtime serialization must reject unsupported values instead of coercing them: "
        f"{violations}"
    )


def test_default_str_scan_rejects_lexical_alias(tmp_path: Path) -> None:
    sample = tmp_path / "aliased_default_str.py"
    sample.write_text(
        "import json\n"
        "def encode(value):\n"
        "    fallback = str\n"
        "    return json.dumps(value, default=fallback)\n",
        encoding="utf-8",
    )

    violations = _default_str_violations_for(sample)

    assert len(violations) == 1
    assert "default=str" in violations[0]


def test_default_str_scan_rejects_conditional_alias_reassignment(tmp_path: Path) -> None:
    sample = tmp_path / "conditional_default.py"
    sample.write_text(
        "import json\n"
        "def encode(value, cond):\n"
        "    fallback = str\n"
        "    if cond:\n"
        "        fallback = None\n"
        "    return json.dumps(value, default=fallback)\n",
        encoding="utf-8",
    )

    violations = _default_str_violations_for(sample)

    assert len(violations) == 1
    assert "default=str" in violations[0]


def test_default_str_scan_allows_distinct_untainted_default(tmp_path: Path) -> None:
    sample = tmp_path / "distinct_untainted_default.py"
    sample.write_text(
        "import json\n"
        "def encode(value):\n"
        "    coercing_fallback = str\n"
        "    fallback = None\n"
        "    return json.dumps(value, default=fallback)\n",
        encoding="utf-8",
    )

    assert _default_str_violations_for(sample) == []


@pytest.mark.parametrize(
    ("filename", "assignments", "expanded_name"),
    [
        (
            "expanded_default.py",
            "    options = {'default': str}\n",
            "options",
        ),
        (
            "expanded_default_alias.py",
            "    options = {'default': str}\n"
            "    kwargs = options\n"
            "    if cond:\n"
            "        kwargs = {}\n",
            "kwargs",
        ),
    ],
)
def test_default_str_scan_rejects_expanded_kwargs_aliases(
    tmp_path: Path,
    filename: str,
    assignments: str,
    expanded_name: str,
) -> None:
    sample = tmp_path / filename
    sample.write_text(
        "import json\n"
        "def encode(value, cond=False):\n"
        f"{assignments}"
        f"    return json.dumps(value, **{expanded_name})\n",
        encoding="utf-8",
    )

    violations = _default_str_violations_for(sample)

    assert len(violations) == 1
    assert "default=str" in violations[0]


def test_default_str_scan_allows_unrelated_expanded_kwargs(tmp_path: Path) -> None:
    sample = tmp_path / "expanded_safe_options.py"
    sample.write_text(
        "import json\n"
        "def encode(value):\n"
        "    options = {'sort_keys': True}\n"
        "    kwargs = options\n"
        "    return json.dumps(value, **kwargs)\n",
        encoding="utf-8",
    )

    assert _default_str_violations_for(sample) == []


@pytest.mark.parametrize(
    ("filename", "imports", "assignment"),
    [
        ("builtins_attribute.py", "import builtins\n", "    fallback = builtins.str\n"),
        (
            "builtins_import_alias.py",
            "from builtins import str as fallback\n",
            "",
        ),
    ],
)
def test_default_str_scan_rejects_builtins_aliases(
    tmp_path: Path,
    filename: str,
    imports: str,
    assignment: str,
) -> None:
    sample = tmp_path / filename
    sample.write_text(
        "import json\n"
        f"{imports}"
        "def encode(value):\n"
        f"{assignment}"
        "    return json.dumps(value, default=fallback)\n",
        encoding="utf-8",
    )

    violations = _default_str_violations_for(sample)

    assert len(violations) == 1
    assert "default=str" in violations[0]


def test_default_str_scan_allows_non_builtin_attribute(tmp_path: Path) -> None:
    sample = tmp_path / "non_builtin_attribute.py"
    sample.write_text(
        "import json\n"
        "def encode(value, converters):\n"
        "    fallback = converters.str\n"
        "    return json.dumps(value, default=fallback)\n",
        encoding="utf-8",
    )

    assert _default_str_violations_for(sample) == []


def test_standalone_reporting_package_does_not_import_host_execution_facade() -> None:
    package_dir = STANDALONE_MCP_ROOT / "tool_use_reporting"
    offenders: dict[str, list[str]] = {}
    for path in package_dir.rglob("*.py"):
        relative_parent = path.relative_to(STANDALONE_MCP_ROOT).parent
        current_package = ".".join(("mcp_unified", *relative_parent.parts))
        blocked = sorted(
            source
            for source in _resolved_import_sources_for(path, current_package)
            if source == "tldw_Server_API" or source.startswith("tldw_Server_API.")
        )
        if blocked:
            offenders[str(path.relative_to(STANDALONE_MCP_ROOT))] = blocked

    assert offenders == {}, (
        "standalone tool-use reporting must remain shape-based and host-neutral: "
        f"{offenders}"
    )


def test_standalone_expected_failure_classifier_export_remains_available() -> None:
    from mcp_unified import tool_use_reporting
    from mcp_unified.tool_use_reporting import builders

    assert "classify_tool_use_exception" in tool_use_reporting.__all__
    assert tool_use_reporting.classify_tool_use_exception is builders.classify_tool_use_exception


def test_exception_log_scan_rejects_raw_exception_interpolation(tmp_path: Path) -> None:
    sample = tmp_path / "unsafe_logging.py"
    sample.write_text(
        "from loguru import logger\n"
        "try:\n"
        "    raise RuntimeError('private provider detail')\n"
        "except Exception as exc:\n"
        "    logger.error('execution failed: {error}', error=str(exc))\n",
        encoding="utf-8",
    )

    violations = _unsafe_exception_log_violations_for(sample)

    assert len(violations) == 1
    assert "str(exc)" in violations[0]


def test_exception_log_scan_rejects_exception_alias_and_repr(tmp_path: Path) -> None:
    sample = tmp_path / "unsafe_exception_alias.py"
    sample.write_text(
        "from loguru import logger\n"
        "try:\n"
        "    raise RuntimeError('private provider detail')\n"
        "except Exception as exc:\n"
        "    failure = exc\n"
        "    logger.error('execution failed: {error}', error=failure)\n"
        "    logger.error('execution failed: {error}', error=repr(failure))\n",
        encoding="utf-8",
    )

    violations = _unsafe_exception_log_violations_for(sample)

    assert len(violations) == 2
    assert any("failure" in violation for violation in violations)
    assert any("repr(failure)" in violation for violation in violations)


def test_exception_log_scan_rejects_conditional_alias_reassignment(
    tmp_path: Path,
) -> None:
    sample = tmp_path / "conditional_exception_alias.py"
    sample.write_text(
        "from loguru import logger\n"
        "def run(cond):\n"
        "    try:\n"
        "        raise RuntimeError('private provider detail')\n"
        "    except Exception as exc:\n"
        "        failure = exc\n"
        "        if cond:\n"
        "            failure = 'safe'\n"
        "        logger.error('{}', failure)\n",
        encoding="utf-8",
    )

    violations = _unsafe_exception_log_violations_for(sample)

    assert len(violations) == 1
    assert "failure" in violations[0]


def test_exception_log_scan_allows_distinct_untainted_log_value(
    tmp_path: Path,
) -> None:
    sample = tmp_path / "distinct_untainted_exception_value.py"
    sample.write_text(
        "from loguru import logger\n"
        "try:\n"
        "    raise RuntimeError('private provider detail')\n"
        "except Exception as exc:\n"
        "    failure = exc\n"
        "    message = 'safe'\n"
        "    logger.error('{}', message)\n",
        encoding="utf-8",
    )

    assert _unsafe_exception_log_violations_for(sample) == []


@pytest.mark.parametrize(
    ("filename", "statement"),
    [
        (
            "for_exception_alias.py",
            "        for failure in [exc]:\n"
            "            logger.error('{}', repr(failure))\n",
        ),
        (
            "async_for_exception_alias.py",
            "        async for failure in stream(exc):\n"
            "            logger.error('{}', repr(failure))\n",
        ),
    ],
)
def test_exception_log_scan_rejects_loop_target_aliases(
    tmp_path: Path,
    filename: str,
    statement: str,
) -> None:
    sample = tmp_path / filename
    sample.write_text(
        "from loguru import logger\n"
        "async def run():\n"
        "    try:\n"
        "        raise RuntimeError('private provider detail')\n"
        "    except Exception as exc:\n"
        f"{statement}",
        encoding="utf-8",
    )

    violations = _unsafe_exception_log_violations_for(sample)

    assert len(violations) == 1
    assert "repr(failure)" in violations[0]


@pytest.mark.parametrize(
    ("filename", "source"),
    [
        (
            "logger_alias.py",
            "from loguru import logger\n"
            "log = logger\n"
            "try:\n"
            "    raise RuntimeError('private provider detail')\n"
            "except Exception as exc:\n"
            "    log.error('{}', repr(exc))\n",
        ),
        (
            "match_exception_alias.py",
            "from loguru import logger\n"
            "try:\n"
            "    raise RuntimeError('private provider detail')\n"
            "except Exception as exc:\n"
            "    match exc:\n"
            "        case _ as failure:\n"
            "            logger.error('{}', repr(failure))\n",
        ),
    ],
)
def test_exception_log_scan_rejects_logger_and_match_aliases(
    tmp_path: Path,
    filename: str,
    source: str,
) -> None:
    sample = tmp_path / filename
    sample.write_text(source, encoding="utf-8")

    violations = _unsafe_exception_log_violations_for(sample)

    assert len(violations) == 1


def test_exception_log_scan_rejects_loguru_import_alias(tmp_path: Path) -> None:
    sample = tmp_path / "loguru_import_alias.py"
    sample.write_text(
        "from loguru import logger as log\n"
        "try:\n"
        "    raise RuntimeError('private provider detail')\n"
        "except Exception as exc:\n"
        "    log.error('{}', repr(exc))\n",
        encoding="utf-8",
    )

    violations = _unsafe_exception_log_violations_for(sample)

    assert len(violations) == 1
    assert "repr(exc)" in violations[0]


def test_exception_log_scan_rejects_shadowed_safe_helper(tmp_path: Path) -> None:
    sample = tmp_path / "shadowed_safe_helper.py"
    sample.write_text(
        "from loguru import logger\n"
        "def _safe_error_type(exc):\n"
        "    return str(exc)\n"
        "try:\n"
        "    raise RuntimeError('private provider detail')\n"
        "except Exception as exc:\n"
        "    logger.error('{}', _safe_error_type(exc))\n",
        encoding="utf-8",
    )

    violations = _unsafe_exception_log_violations_for(sample)

    assert len(violations) == 1
    assert "_safe_error_type(exc)" in violations[0]


@pytest.mark.parametrize(
    ("filename", "source"),
    [
        (
            "safe_loguru_alias.py",
            "from loguru import logger as log\n"
            "try:\n"
            "    raise RuntimeError('private provider detail')\n"
            "except Exception as exc:\n"
            "    log.error('{}', 'Exception')\n",
        ),
        (
            "verified_safe_helper.py",
            "from loguru import logger\n"
            "def _safe_error_type(exc):\n"
            "    try:\n"
            "        name = type(exc).__name__\n"
            "        return name if type(name) is str and name.isascii() and len(name) <= 64 else 'Exception'\n"
            "    except BaseException:\n"
            "        return 'Exception'\n"
            "try:\n"
            "    raise RuntimeError('private provider detail')\n"
            "except Exception as exc:\n"
            "    logger.error('{}', _safe_error_type(exc))\n",
        ),
    ],
)
def test_exception_log_scan_allows_verified_logger_and_helper_bindings(
    tmp_path: Path,
    filename: str,
    source: str,
) -> None:
    sample = tmp_path / filename
    sample.write_text(source, encoding="utf-8")

    assert _unsafe_exception_log_violations_for(sample) == []


@pytest.mark.parametrize(
    ("filename", "source"),
    [
        (
            "distinct_logger_name.py",
            "from loguru import logger\n"
            "logger_alias = logger\n"
            "log = sink\n"
            "try:\n"
            "    raise RuntimeError('private provider detail')\n"
            "except Exception as exc:\n"
            "    log.error('{}', repr(exc))\n",
        ),
        (
            "safe_match_alias.py",
            "from loguru import logger\n"
            "try:\n"
            "    raise RuntimeError('private provider detail')\n"
            "except Exception as exc:\n"
            "    status = 'safe'\n"
            "    match status:\n"
            "        case _ as failure:\n"
            "            logger.error('{}', repr(failure))\n",
        ),
    ],
)
def test_exception_log_scan_allows_distinct_logger_and_safe_match_aliases(
    tmp_path: Path,
    filename: str,
    source: str,
) -> None:
    sample = tmp_path / filename
    sample.write_text(source, encoding="utf-8")

    assert _unsafe_exception_log_violations_for(sample) == []


def test_exception_log_scan_rejects_loguru_exception_context(tmp_path: Path) -> None:
    sample = tmp_path / "unsafe_exception_context.py"
    sample.write_text(
        "from loguru import logger\n"
        "try:\n"
        "    raise RuntimeError('private provider detail')\n"
        "except Exception as exc:\n"
        "    logger.opt(exception=True).error('execution failed')\n"
        "    logger.error('execution failed', exc_info=exc)\n",
        encoding="utf-8",
    )

    violations = _unsafe_exception_log_violations_for(sample)

    assert len(violations) == 2
    assert any("exception logging" in violation for violation in violations)
    assert any("exc_info logging" in violation for violation in violations)


def test_exception_log_scan_allows_verified_exception_family_helper(tmp_path: Path) -> None:
    sample = tmp_path / "safe_logging.py"
    sample.write_text(
        "from loguru import logger\n"
        "def _safe_exception_family(exc):\n"
        "    name = type(exc).__name__\n"
        "    return name if type(name) is str else 'Exception'\n"
        "try:\n"
        "    raise RuntimeError('private provider detail')\n"
        "except Exception as exc:\n"
        "    logger.opt(exception=False).bind(\n"
        "        error_type=_safe_exception_family(exc),\n"
        "    ).error('execution failed')\n",
        encoding="utf-8",
    )

    assert _unsafe_exception_log_violations_for(sample) == []


def test_exception_log_scan_rejects_direct_exception_family_descriptors(
    tmp_path: Path,
) -> None:
    sample = tmp_path / "unsafe_exception_family.py"
    sample.write_text(
        "from loguru import logger\n"
        "try:\n"
        "    raise RuntimeError('private provider detail')\n"
        "except Exception as exc:\n"
        "    logger.error('failed {}', exc.__class__.__name__)\n"
        "    logger.error('failed {}', type(exc).__name__)\n",
        encoding="utf-8",
    )

    violations = _unsafe_exception_log_violations_for(sample)

    assert len(violations) == 2


def test_protocol_branch_scan_resolves_expected_failure_import_alias(
    tmp_path: Path,
) -> None:
    sample = tmp_path / "expected_failure_import_alias.py"
    sample.write_text(
        "from package.execution_outcomes import ExpectedToolFailure as ETF\n"
        "def handle(provider):\n"
        "    try:\n"
        "        provider.run()\n"
        "    except ETF:\n"
        "        return {'isError': True}\n",
        encoding="utf-8",
    )

    violations = _protocol_expected_failure_branch_violations_for(sample)

    assert any("ExpectedToolFailure" in violation for violation in violations)


def test_protocol_branch_scan_resolves_destructured_expected_failure_alias(
    tmp_path: Path,
) -> None:
    sample = tmp_path / "destructured_expected_failure_alias.py"
    sample.write_text(
        "from package.execution_outcomes import ExpectedToolFailure\n"
        "ETF, = (ExpectedToolFailure,)\n"
        "def handle(provider):\n"
        "    try:\n"
        "        provider.run()\n"
        "    except ETF:\n"
        "        return {'isError': True}\n",
        encoding="utf-8",
    )

    violations = _protocol_expected_failure_branch_violations_for(sample)

    assert any("ExpectedToolFailure" in violation for violation in violations)


def test_protocol_branch_scan_resolves_destructured_error_result_alias(
    tmp_path: Path,
) -> None:
    sample = tmp_path / "destructured_error_result.py"
    sample.write_text(
        "def handle(tool_name):\n"
        "    result, = ({'isError': True},)\n"
        "    if tool_name == 'skills.run':\n"
        "        return result\n"
        "    return {}\n",
        encoding="utf-8",
    )

    violations = _protocol_expected_failure_branch_violations_for(sample)

    assert any("Skills/model/provider" in violation for violation in violations)


def test_runtime_policy_scan_resolves_operator_module_assignment_alias(
    tmp_path: Path,
) -> None:
    sample = tmp_path / "operator_module_assignment_alias.py"
    sample.write_text(
        "import operator\n"
        "op = operator\n"
        "async def execute_prepared_tool_call(prepared):\n"
        "    policy = prepared.policy\n"
        "    field = 'category'\n"
        "    return op.getitem(prepared.tool_def, field), policy.effect\n",
        encoding="utf-8",
    )

    violations = _runtime_mutable_authority_violations_for(sample)

    assert any("dynamic prepared authority" in violation for violation in violations)


@pytest.mark.parametrize(
    ("filename", "source"),
    [
        (
            "shadowed_operator_parameter.py",
            "import operator\n"
            "op = operator\n"
            "async def execute_prepared_tool_call(prepared):\n"
            "    policy = prepared.policy\n"
            "    def inspect(op):\n"
            "        field = 'category'\n"
            "        return op.getitem(prepared.tool_def, field)\n"
            "    return inspect(safe_reader), policy.effect\n",
        ),
        (
            "shadowed_authority_parameter.py",
            "async def execute_prepared_tool_call(prepared):\n"
            "    policy = prepared.policy\n"
            "    metadata = prepared.tool_def['metadata']\n"
            "    def inspect(metadata):\n"
            "        field = 'category'\n"
            "        return metadata[field]\n"
            "    return inspect({'category': 'safe'}), policy.effect\n",
        ),
    ],
)
def test_runtime_policy_scan_honors_nested_parameter_shadowing(
    tmp_path: Path,
    filename: str,
    source: str,
) -> None:
    sample = tmp_path / filename
    sample.write_text(source, encoding="utf-8")

    assert _runtime_mutable_authority_violations_for(sample) == []


def test_default_str_scan_rejects_positional_dict_mapping(tmp_path: Path) -> None:
    sample = tmp_path / "positional_default_mapping.py"
    sample.write_text(
        "import json\n"
        "def encode(value):\n"
        "    options = dict({'default': str})\n"
        "    return json.dumps(value, **options)\n",
        encoding="utf-8",
    )

    violations = _default_str_violations_for(sample)

    assert len(violations) == 1
    assert "default=str" in violations[0]


@pytest.mark.parametrize(
    ("filename", "source"),
    [
        (
            "shadowed_str_parameter.py",
            "import json\n"
            "def encode(value, str):\n"
            "    return json.dumps(value, default=str)\n",
        ),
        (
            "shadowed_options_parameter.py",
            "import json\n"
            "options = {'default': str}\n"
            "def encode(value, options):\n"
            "    return json.dumps(value, **options)\n",
        ),
    ],
)
def test_default_str_scan_honors_parameter_shadowing(
    tmp_path: Path,
    filename: str,
    source: str,
) -> None:
    sample = tmp_path / filename
    sample.write_text(source, encoding="utf-8")

    assert _default_str_violations_for(sample) == []


@pytest.mark.parametrize(
    ("filename", "binding"),
    [
        (
            "rebound_safe_helper.py",
            "    _safe_error_type = lambda value: str(value)\n",
        ),
        (
            "parameter_shadowed_safe_helper.py",
            "def run(_safe_error_type):\n",
        ),
    ],
)
def test_exception_log_scan_rejects_rebound_safe_helper(
    tmp_path: Path,
    filename: str,
    binding: str,
) -> None:
    function_header = "def run():\n" if binding.startswith("    ") else binding
    function_binding = binding if binding.startswith("    ") else ""
    sample = tmp_path / filename
    sample.write_text(
        "from loguru import logger\n"
        "def _safe_error_type(exc):\n"
        "    return type(exc).__name__\n"
        f"{function_header}"
        f"{function_binding}"
        "    try:\n"
        "        raise RuntimeError('private provider detail')\n"
        "    except Exception as exc:\n"
        "        logger.error('{}', _safe_error_type(exc))\n",
        encoding="utf-8",
    )

    violations = _unsafe_exception_log_violations_for(sample)

    assert len(violations) == 1
    assert "_safe_error_type(exc)" in violations[0]


def test_exception_log_scan_honors_logger_parameter_shadowing(tmp_path: Path) -> None:
    sample = tmp_path / "shadowed_logger_parameter.py"
    sample.write_text(
        "from loguru import logger as log\n"
        "def run(log):\n"
        "    try:\n"
        "        raise RuntimeError('private provider detail')\n"
        "    except Exception as exc:\n"
        "        log.error('{}', repr(exc))\n",
        encoding="utf-8",
    )

    assert _unsafe_exception_log_violations_for(sample) == []


def test_exception_log_scan_honors_direct_logger_parameter_shadowing(
    tmp_path: Path,
) -> None:
    sample = tmp_path / "shadowed_direct_logger_parameter.py"
    sample.write_text(
        "def run(logger):\n"
        "    try:\n"
        "        raise RuntimeError('private provider detail')\n"
        "    except Exception as exc:\n"
        "        logger.error('{}', repr(exc))\n",
        encoding="utf-8",
    )

    assert _unsafe_exception_log_violations_for(sample) == []


def test_exception_log_scan_resolves_loguru_module_assignment_alias(
    tmp_path: Path,
) -> None:
    sample = tmp_path / "loguru_module_assignment_alias.py"
    sample.write_text(
        "import loguru\n"
        "logs = loguru\n"
        "try:\n"
        "    raise RuntimeError('private provider detail')\n"
        "except Exception as exc:\n"
        "    logs.logger.error('{}', repr(exc))\n",
        encoding="utf-8",
    )

    violations = _unsafe_exception_log_violations_for(sample)

    assert len(violations) == 1
    assert "repr(exc)" in violations[0]


@pytest.mark.parametrize(
    ("filename", "source", "scanner"),
    [
        (
            "authority_walrus.py",
            "def execute_prepared_tool_call(prepared, field):\n"
            "    policy = prepared.policy\n"
            "    if authority := prepared.tool_def:\n"
            "        return authority[field], policy.effect\n",
            _runtime_mutable_authority_violations_for,
        ),
        (
            "operator_destructure.py",
            "import operator\n"
            "def execute_prepared_tool_call(prepared, field):\n"
            "    policy = prepared.policy\n"
            "    op, other = (operator, None)\n"
            "    return op.getitem(prepared.tool_def, field), policy.effect\n",
            _runtime_mutable_authority_violations_for,
        ),
        (
            "default_str_destructure.py",
            "import json\n"
            "default, other = (str, None)\n"
            "def encode(value):\n"
            "    return json.dumps(value, default=default)\n",
            _default_str_violations_for,
        ),
        (
            "builtins_assignment_alias.py",
            "import builtins\n"
            "import json\n"
            "b = builtins\n"
            "def encode(value):\n"
            "    return json.dumps(value, default=b.str)\n",
            _default_str_violations_for,
        ),
        (
            "logger_destructure.py",
            "from loguru import logger\n"
            "log, other = (logger, None)\n"
            "def run():\n"
            "    try:\n"
            "        raise RuntimeError('secret')\n"
            "    except Exception as exc:\n"
            "        log.error('failed: {}', exc)\n",
            _unsafe_exception_log_violations_for,
        ),
    ],
)
def test_contract_scanners_propagate_assignment_forms_consistently(
    tmp_path: Path,
    filename: str,
    source: str,
    scanner: Callable[[Path], list[str]],
) -> None:
    sample = tmp_path / filename
    sample.write_text(source, encoding="utf-8")

    assert scanner(sample)


@pytest.mark.parametrize(
    ("filename", "helper_body"),
    [
        (
            "helper_getattribute_args.py",
            "def _safe_error_type(error):\n"
            "    return object.__getattribute__(error, 'args')\n",
        ),
        (
            "helper_closure.py",
            "def _safe_error_type(_ignored):\n"
            "            return str(exc)\n",
        ),
    ],
)
def test_exception_log_scan_rejects_helper_disclosure_paths(
    tmp_path: Path,
    filename: str,
    helper_body: str,
) -> None:
    nested = filename == "helper_closure.py"
    prefix = "        " if nested else ""
    helper_source = "".join(prefix + line for line in helper_body.splitlines(keepends=True))
    sample = tmp_path / filename
    sample.write_text(
        "from loguru import logger\n"
        + ("def run():\n    try:\n        raise RuntimeError('secret')\n    except Exception as exc:\n" if nested else "")
        + helper_source
        + ("        logger.error('failed: {}', _safe_error_type(exc))\n" if nested else "def run():\n    try:\n        raise RuntimeError('secret')\n    except Exception as exc:\n        logger.error('failed: {}', _safe_error_type(exc))\n"),
        encoding="utf-8",
    )

    assert _unsafe_exception_log_violations_for(sample)


def test_exception_log_scan_taints_destructured_values_elementwise(
    tmp_path: Path,
) -> None:
    sample = tmp_path / "exception_destructure_control.py"
    sample.write_text(
        "from loguru import logger\n"
        "def run():\n"
        "    try:\n"
        "        raise RuntimeError('secret')\n"
        "    except Exception as exc:\n"
        "        unsafe, safe = (exc, 'constant')\n"
        "        logger.error('safe: {}', safe)\n",
        encoding="utf-8",
    )

    assert _unsafe_exception_log_violations_for(sample) == []


def test_breaker_runtime_and_idempotency_logs_do_not_expose_exception_text() -> None:
    targets = (
        MCP_ROOT / "modules" / "base.py",
        MCP_ROOT / "tool_execution" / "runtime.py",
        MCP_ROOT / "tool_execution" / "idempotency.py",
        MCP_ROOT / "tool_execution" / "security.py",
    )
    offenders = {
        str(path.relative_to(MCP_ROOT)): violations
        for path in targets
        if (violations := _unsafe_exception_log_violations_for(path))
    }

    assert offenders == {}, (
        "MCP execution logs may use safe exception-family and bounded structural fields only: "
        f"{offenders}"
    )
