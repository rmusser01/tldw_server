from __future__ import annotations

import importlib
import inspect
from collections.abc import Callable

import pytest
from fastapi.routing import APIRoute

from tldw_Server_API.app.core.AuthNZ.permissions import MEDIA_CREATE


pytestmark = pytest.mark.unit


ROUTE_CASES = [
    ("process_audios", "/process-audios"),
    ("process_documents", "/process-documents"),
    ("process_pdfs", "/process-pdfs"),
    ("process_ebooks", "/process-ebooks"),
    ("process_code", "/process-code"),
    ("process_emails", "/process-emails"),
    ("process_mediawiki", "/mediawiki/ingest-dump"),
    ("process_mediawiki", "/mediawiki/process-dump"),
]


def _route_for(module_name: str, path: str) -> APIRoute:
    module = importlib.import_module(
        f"tldw_Server_API.app.api.v1.endpoints.media.{module_name}"
    )
    for route in module.router.routes:
        if isinstance(route, APIRoute) and route.path == path:
            return route
    raise AssertionError(f"{module_name} does not define route {path}")


def _requires_permission(call: Callable[..., object], permission: str) -> bool:
    closure = inspect.getclosurevars(call)
    for value in closure.nonlocals.values():
        if isinstance(value, (list, tuple, set)) and permission in value:
            return True
    return False


@pytest.mark.parametrize(("module_name", "path"), ROUTE_CASES)
def test_processing_route_requires_media_create_and_rbac_rate_limit(
    module_name: str,
    path: str,
) -> None:
    route = _route_for(module_name, path)
    dependency_calls = [dependency.call for dependency in route.dependant.dependencies]

    assert any(_requires_permission(call, MEDIA_CREATE) for call in dependency_calls)
    assert any(
        getattr(call, "_tldw_rate_limit_resource", None) == "media.create"
        for call in dependency_calls
    )
