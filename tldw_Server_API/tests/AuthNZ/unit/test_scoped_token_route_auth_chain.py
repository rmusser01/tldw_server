from __future__ import annotations

from collections.abc import Iterable

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal
from tldw_Server_API.app.api.v1.endpoints.evaluations.evaluations_auth import (
    get_eval_request_user,
    verify_api_key,
)
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user
from tldw_Server_API.app.core.Utils.fastapi_routes import iter_served_routes


def _walk_dependants(deps: Iterable[object]) -> Iterable[object]:
    for dep in deps:
        yield dep
        nested = getattr(dep, "dependencies", None) or []
        yield from _walk_dependants(nested)


def test_scoped_routes_include_auth_dependency_chain() -> None:
    # Import lazily to avoid module-level startup overhead in unrelated test runs.
    from tldw_Server_API.app.main import app

    auth_calls = {get_request_user, get_eval_request_user, verify_api_key, get_auth_principal}

    scoped_routes: list[str] = []
    missing_auth_chain: list[str] = []

    for route in iter_served_routes(app.routes):
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
