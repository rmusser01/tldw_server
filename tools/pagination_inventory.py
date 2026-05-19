#!/usr/bin/env python3
"""Generate a static inventory of API pagination candidates.

This tool intentionally avoids importing the FastAPI app. Importing the app can
start configuration/database side effects, so route metadata here is static and
marked as inferred when it cannot be confirmed from decorators alone.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
ENDPOINTS_ROOT = ROOT / "tldw_Server_API" / "app" / "api" / "v1" / "endpoints"
SCHEMAS_ROOT = ROOT / "tldw_Server_API" / "app" / "api" / "v1" / "schemas"
TESTS_ROOT = ROOT / "tldw_Server_API" / "tests"

HTTP_METHODS = {"get", "post", "put", "patch", "delete"}
PAGINATION_PARAM_NAMES = {
    "after",
    "after_id",
    "before",
    "cursor",
    "ending_before",
    "last_id",
    "limit",
    "next_cursor",
    "offset",
    "page",
    "page_size",
    "per_page",
    "starting_after",
}
RESPONSE_PAGINATION_FIELDS = {
    "count",
    "cursor",
    "has_more",
    "limit",
    "next_cursor",
    "next_offset",
    "next_page",
    "offset",
    "page",
    "page_size",
    "pagination",
    "per_page",
    "prev_cursor",
    "prev_page",
    "total",
    "total_count",
    "total_items",
    "total_pages",
}
LIST_ROUTE_KEYWORDS = {
    "collections",
    "entries",
    "events",
    "feeds",
    "history",
    "items",
    "jobs",
    "list",
    "messages",
    "results",
    "runs",
    "search",
    "versions",
}
PROVIDER_COMPATIBLE_HINTS = {
    "anthropic",
    "audio_transcriptions",
    "chat",
    "embeddings",
    "llm_providers",
    "openai",
}


@dataclass(frozen=True)
class SchemaInfo:
    """Static field inventory for a schema class."""

    name: str
    file: Path
    fields: frozenset[str]


@dataclass
class RouteInfo:
    """Static route inventory extracted from endpoint decorators."""

    method: str
    path: str
    path_confidence: str
    endpoint_file: Path
    function_name: str
    response_model: str
    query_params: list[str] = field(default_factory=list)
    response_fields: set[str] = field(default_factory=set)
    family: str = "unknown"
    status: str = "needs-confirmation"
    count_strategy: str = "needs-confirmation"
    recommended_tranche: str = "classify-manually"
    test_candidates: list[str] = field(default_factory=list)


def _parse_python(path: Path) -> ast.Module | None:
    try:
        return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (SyntaxError, UnicodeDecodeError):
        return None


def _name_of(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Subscript):
        return _name_of(node.value)
    if isinstance(node, ast.Call):
        return _name_of(node.func)
    return ""


def _unparse(node: ast.AST | None) -> str:
    if node is None:
        return ""
    try:
        return ast.unparse(node)
    except Exception:
        return ""


def _string_literal(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _decorator_call(node: ast.AST) -> tuple[str, ast.Call] | None:
    if not isinstance(node, ast.Call):
        return None
    method = _name_of(node.func).lower()
    if method not in HTTP_METHODS:
        return None
    return method.upper(), node


def _router_prefixes(module: ast.Module) -> dict[str, str]:
    prefixes: dict[str, str] = {}
    for node in module.body:
        if not isinstance(node, ast.Assign):
            continue
        if not isinstance(node.value, ast.Call):
            continue
        if _name_of(node.value.func) != "APIRouter":
            continue
        prefix = ""
        for kw in node.value.keywords:
            if kw.arg == "prefix":
                prefix = _string_literal(kw.value) or ""
        for target in node.targets:
            if isinstance(target, ast.Name):
                prefixes[target.id] = prefix
    return prefixes


def _decorator_router_name(call: ast.Call) -> str | None:
    func = call.func
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        return func.value.id
    return None


def _join_paths(prefix: str, path: str) -> str:
    if not prefix:
        return path or "unknown"
    if not path:
        return prefix
    return f"{prefix.rstrip('/')}/{path.lstrip('/')}"


def _extract_response_model(call: ast.Call) -> str:
    for kw in call.keywords:
        if kw.arg == "response_model":
            return _unparse(kw.value)
    return ""


def _query_params(node: ast.AsyncFunctionDef | ast.FunctionDef) -> list[str]:
    args = [*node.args.args, *node.args.kwonlyargs]
    names = sorted({arg.arg for arg in args if arg.arg in PAGINATION_PARAM_NAMES})
    return names


def _field_names(node: ast.ClassDef) -> set[str]:
    fields: set[str] = set()
    for item in node.body:
        target: ast.expr | None = None
        if isinstance(item, ast.AnnAssign):
            target = item.target
        elif isinstance(item, ast.Assign) and len(item.targets) == 1:
            target = item.targets[0]
        if isinstance(target, ast.Name) and not target.id.startswith("_"):
            fields.add(target.id)
    return fields


def _is_schema_class(node: ast.ClassDef) -> bool:
    if not node.bases:
        return False
    for base in node.bases:
        base_name = _name_of(base)
        if base_name.endswith("BaseModel") or base_name.endswith("Response") or base_name.endswith("Schema"):
            return True
    return False


def collect_schemas() -> dict[str, SchemaInfo]:
    schemas: dict[str, SchemaInfo] = {}
    for path in sorted(SCHEMAS_ROOT.rglob("*.py")):
        module = _parse_python(path)
        if module is None:
            continue
        for node in ast.walk(module):
            if isinstance(node, ast.ClassDef) and _is_schema_class(node):
                fields = _field_names(node)
                if fields:
                    schemas[node.name] = SchemaInfo(
                        name=node.name,
                        file=path.relative_to(ROOT),
                        fields=frozenset(fields),
                    )
    return schemas


def _model_tokens(response_model: str) -> list[str]:
    if not response_model:
        return []
    return re.findall(r"[A-Za-z_][A-Za-z0-9_]*", response_model)


def _response_fields(response_model: str, schemas: dict[str, SchemaInfo]) -> set[str]:
    fields: set[str] = set()
    for token in _model_tokens(response_model):
        schema = schemas.get(token)
        if schema is not None:
            fields.update(schema.fields)
    return fields


def _is_raw_list(response_model: str) -> bool:
    normalized = response_model.replace("typing.", "")
    return bool(re.search(r"(^|[\[\s,|])(?:list|List|Sequence)\s*\[", normalized))


def _has_list_keyword(*values: str) -> bool:
    tokens: set[str] = set()
    for value in values:
        tokens.update(token for token in re.split(r"[^a-z0-9]+", value.lower()) if token)
    return bool(tokens & LIST_ROUTE_KEYWORDS)


def _is_candidate(route: RouteInfo) -> bool:
    if route.query_params:
        return True
    if route.response_fields & RESPONSE_PAGINATION_FIELDS:
        return True
    if _is_raw_list(route.response_model):
        return route.method == "GET" or _has_list_keyword(route.path, route.function_name, route.response_model)
    if route.method != "GET":
        return False
    if "{" in route.path and not _has_list_keyword(route.function_name):
        return False
    return _has_list_keyword(route.path, route.function_name, route.response_model)


def _classify_family(route: RouteInfo) -> str:
    response_model = route.response_model
    fields = route.response_fields
    params = set(route.query_params)
    file_text = route.endpoint_file.as_posix().lower()
    haystack = f"{route.path} {route.function_name} {file_text} {response_model}".lower()

    if _is_raw_list(response_model):
        return "raw-list"
    if any(hint in haystack for hint in PROVIDER_COMPATIBLE_HINTS) and not (params or fields & RESPONSE_PAGINATION_FIELDS):
        return "provider"
    if {"cursor", "after", "after_id", "before", "starting_after", "ending_before"} & params:
        return "cursor"
    if {"next_cursor", "prev_cursor", "cursor"} & fields:
        return "cursor"
    if {"page", "per_page", "page_size"} & params:
        return "page"
    if {"total_pages", "next_page", "prev_page"} & fields:
        return "page"
    if {"limit", "offset"} <= params or "offset" in params:
        return "offset"
    if {"next_offset", "has_more"} & fields:
        return "offset"
    if "pagination" in fields:
        return "custom"
    if params or fields & {"total", "total_count", "count", "total_items"}:
        return "custom"
    if any(keyword in haystack for keyword in LIST_ROUTE_KEYWORDS):
        return "unknown"
    return "not-paginated"


def _count_strategy(route: RouteInfo) -> str:
    fields = route.response_fields
    if route.family in {"provider", "raw-list", "not-paginated"}:
        return "not-applicable"
    if {"total", "total_count", "total_items"} & fields:
        return "known-total-or-provider-total"
    if route.family in {"offset", "cursor"} and ({"has_more", "next_cursor", "next_offset"} & fields or route.query_params):
        return "overfetch-or-token"
    if "pagination" in fields:
        return "inspect-schema"
    return "needs-confirmation"


def _status(route: RouteInfo) -> str:
    if route.family in {"provider", "raw-list", "not-paginated"}:
        return f"exempt-{route.family}"
    if "pagination" in route.response_fields:
        return "canonical-present-or-custom-pagination"
    if route.family == "unknown":
        return "needs-confirmation"
    return "migration-candidate"


def _recommended_tranche(route: RouteInfo) -> str:
    file_text = route.endpoint_file.as_posix().lower()
    if route.family == "raw-list":
        return "exempt-raw-list-or-versioned-route"
    if route.family == "provider":
        return "exempt-provider-compatible"
    if route.family == "not-paginated":
        return "exempt-not-paginated"
    if "prompt_studio" in file_text:
        return "page-prompt-studio"
    if "paper_search" in file_text or "research" in file_text:
        return "page-research"
    if "/media/" in file_text:
        return "page-media"
    if "/audio/" in file_text:
        return "cursor-audio"
    if "workflows" in file_text or "jobs_admin" in file_text:
        return "cursor-workflows-jobs"
    if any(name in file_text for name in ("watchlists", "kanban", "sandbox", "chatbooks", "mcp_hub")):
        return "custom-envelope"
    if route.family == "page":
        return "page-family"
    if route.family == "cursor":
        return "cursor-family"
    if route.family == "offset":
        return "offset-cleanup"
    return "classify-manually"


def collect_routes(schemas: dict[str, SchemaInfo]) -> list[RouteInfo]:
    routes: list[RouteInfo] = []
    for path in sorted(ENDPOINTS_ROOT.rglob("*.py")):
        module = _parse_python(path)
        if module is None:
            continue
        prefixes = _router_prefixes(module)
        for node in ast.walk(module):
            if not isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)):
                continue
            for decorator in node.decorator_list:
                decorated = _decorator_call(decorator)
                if decorated is None:
                    continue
                method, call = decorated
                raw_path = _string_literal(call.args[0]) if call.args else ""
                router_name = _decorator_router_name(call)
                prefix = prefixes.get(router_name or "", "")
                route_path = _join_paths(prefix, raw_path or "")
                path_confidence = "decorator" if raw_path is not None else "unknown"
                response_model = _extract_response_model(call)
                route = RouteInfo(
                    method=method,
                    path=route_path,
                    path_confidence=path_confidence,
                    endpoint_file=path.relative_to(ROOT),
                    function_name=node.name,
                    response_model=response_model or "unknown",
                    query_params=_query_params(node),
                )
                route.response_fields = _response_fields(response_model, schemas)
                route.family = _classify_family(route)
                route.status = _status(route)
                route.count_strategy = _count_strategy(route)
                route.recommended_tranche = _recommended_tranche(route)
                if _is_candidate(route):
                    routes.append(route)
    return routes


def _test_files() -> list[Path]:
    return sorted(TESTS_ROOT.rglob("test_*.py")) if TESTS_ROOT.exists() else []


def _tokens_for_path(path: Path) -> list[str]:
    tokens = re.split(r"[_/\-.]+", path.with_suffix("").as_posix().lower())
    ignored = {"app", "api", "v1", "endpoints", "endpoint", "test", "tests", "py"}
    return [token for token in tokens if len(token) >= 4 and token not in ignored]


def attach_test_candidates(routes: Iterable[RouteInfo]) -> None:
    tests = _test_files()
    scored_tests = [(test, set(_tokens_for_path(test.relative_to(ROOT)))) for test in tests]
    for route in routes:
        route_tokens = set(_tokens_for_path(route.endpoint_file)) | set(route.function_name.lower().split("_"))
        scored: list[tuple[int, Path]] = []
        for test, test_tokens in scored_tests:
            score = len(route_tokens & test_tokens)
            if score:
                scored.append((score, test))
        scored.sort(key=lambda item: (-item[0], item[1].as_posix()))
        route.test_candidates = [path.relative_to(ROOT).as_posix() for _, path in scored[:3]]


def _md(text: object) -> str:
    value = str(text) if text is not None else ""
    value = value.replace("|", "\\|").replace("\n", " ")
    return f"`{value}`" if value else "`unknown`"


def render_markdown(routes: list[RouteInfo]) -> str:
    lines = [
        "# Pagination Completion Matrix",
        "",
        "Generated by `tools/pagination_inventory.py` using static AST analysis.",
        "Route paths are local decorator paths and may omit prefixes added by parent routers.",
        "Rows with inferred or incomplete metadata should be confirmed before source migrations.",
        "",
        f"Total candidate rows: {len(routes)}",
        "",
        "| Method | Local Path | Path Source | Endpoint | Response Model | Query Params | Response Pagination Fields | Family | Status | Count Strategy | Recommended Tranche | Test Candidates |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for route in sorted(routes, key=lambda item: (item.recommended_tranche, item.endpoint_file.as_posix(), item.function_name)):
        response_fields = sorted(route.response_fields & RESPONSE_PAGINATION_FIELDS)
        cells = [
            _md(route.method),
            _md(route.path),
            _md(route.path_confidence),
            _md(f"{route.endpoint_file}:{route.function_name}"),
            _md(route.response_model),
            _md(", ".join(route.query_params) or "none"),
            _md(", ".join(response_fields) or "none"),
            _md(route.family),
            _md(route.status),
            _md(route.count_strategy),
            _md(route.recommended_tranche),
            _md("<br>".join(route.test_candidates) or "needs-test-confirmation"),
        ]
        lines.append(f"| {' | '.join(cells)} |")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `raw-list` rows are response-shape sensitive; do not wrap them without a versioning decision.",
            "- `provider` rows should preserve provider-compatible wire shapes unless explicitly approved.",
            "- `unknown` rows need manual review before being used as implementation scope.",
            "- Test candidates are heuristic and must be confirmed before editing endpoints.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    schemas = collect_schemas()
    routes = collect_routes(schemas)
    attach_test_candidates(routes)
    print(render_markdown(routes), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
