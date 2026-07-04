from __future__ import annotations

from collections.abc import Iterable

import pytest

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal
from tldw_Server_API.app.api.v1.endpoints.evaluations.evaluations_auth import (
    get_eval_request_user,
    verify_api_key,
)
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user


def _walk_dependants(deps: Iterable[object]) -> Iterable[object]:
    for dep in deps:
        yield dep
        nested = getattr(dep, "dependencies", None) or []
        yield from _walk_dependants(nested)


def _dependency_calls_for_route(route: object) -> list[object]:
    dependant = getattr(route, "dependant", None)
    if dependant is None:
        return []
    deps = list(_walk_dependants(getattr(dependant, "dependencies", []) or []))
    return [getattr(dep, "call", None) for dep in deps]


def _find_route(app: object, method: str, path: str) -> object | None:
    method = method.upper()
    for route in getattr(app, "routes", []):
        if str(getattr(route, "path", "")) != path:
            continue
        methods = {str(item).upper() for item in (getattr(route, "methods", []) or [])}
        if method in methods:
            return route
    return None


def test_scoped_routes_include_auth_dependency_chain() -> None:
    # Import lazily to avoid module-level startup overhead in unrelated test runs.
    from tldw_Server_API.app.main import app

    auth_calls = {get_request_user, get_eval_request_user, verify_api_key, get_auth_principal}

    scoped_routes: list[str] = []
    missing_auth_chain: list[str] = []

    for route in app.routes:
        dependant = getattr(route, "dependant", None)
        if dependant is None:
            continue

        deps = list(_walk_dependants(getattr(dependant, "dependencies", []) or []))
        dep_calls = [getattr(dep, "call", None) for dep in deps]

        if not any(getattr(call, "_tldw_token_scope", False) for call in dep_calls):
            continue

        methods = sorted(getattr(route, "methods", []) or [])
        path = str(getattr(route, "path", ""))
        scoped_routes.append(path)

        if not any(call in auth_calls for call in dep_calls):
            missing_auth_chain.append(f"{','.join(methods)} {path}")

    assert scoped_routes, "Expected at least one route with require_token_scope metadata."
    assert not missing_auth_chain, (
        "Scoped routes missing auth dependency chain:\n" + "\n".join(missing_auth_chain)
    )


@pytest.mark.parametrize(
    ("method", "path", "endpoint_id", "count_as"),
    [
        ("POST", "/api/v1/rag/ablate", "rag.search", "call"),
        ("POST", "/api/v1/rag/batch", "rag.search", "call"),
        ("POST", "/api/v1/rag/batch/resume/{checkpoint_id}", "rag.search", "call"),
        ("GET", "/api/v1/rag/simple", "rag.search", "call"),
        ("POST", "/api/v1/rag/search/stream", "rag.search", "call"),
        ("GET", "/api/v1/rag/advanced", "rag.search", "call"),
        ("POST", "/api/v1/chats/{chat_id}/complete", "chat.completions", "call"),
        ("POST", "/api/v1/chats/{chat_id}/completions", "chat.completions", "call"),
        ("POST", "/api/v1/chats/{chat_id}/complete-v2", "chat.completions", "call"),
        ("POST", "/api/v1/chat/documents/generate", "chat.completions", "call"),
        ("POST", "/api/v1/chat/documents/bulk", "chat.completions", "call"),
        ("POST", "/api/v1/embeddings", "embeddings", "call"),
        ("POST", "/api/v1/embeddings/batch", "embeddings", "call"),
    ],
)
def test_resource_spending_routes_declare_token_scope_guards(
    method: str,
    path: str,
    endpoint_id: str,
    count_as: str,
) -> None:
    # Import lazily to use the same routed app graph as HTTP requests.
    from tldw_Server_API.app.main import app

    route = _find_route(app, method, path)

    assert route is not None, f"Expected route {method} {path} to be registered"

    token_scope_calls = [
        call
        for call in _dependency_calls_for_route(route)
        if getattr(call, "_tldw_token_scope", False)
    ]

    assert token_scope_calls, f"Expected {method} {path} to declare TokenScopeGuard"
    assert any(
        getattr(call, "_tldw_endpoint_id", None) == endpoint_id
        and getattr(call, "_tldw_count_as", None) == count_as
        and getattr(call, "_tldw_token_scope_required", None) == "any"
        for call in token_scope_calls
    ), (
        f"Expected {method} {path} to declare endpoint_id={endpoint_id!r} "
        f"and count_as={count_as!r}"
    )
