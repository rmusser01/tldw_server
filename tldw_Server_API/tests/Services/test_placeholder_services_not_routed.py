from __future__ import annotations

import pytest

from tldw_Server_API.app.core.Utils.fastapi_routes import iter_served_routes


@pytest.mark.unit
def test_placeholder_services_not_bound_to_active_routes():
    from tldw_Server_API.app.main import app

    placeholder_modules = {
        "tldw_Server_API.app.services.document_processing_service",
        "tldw_Server_API.app.services.ebook_processing_service",
        "tldw_Server_API.app.services.podcast_processing_service",
        "tldw_Server_API.app.services.xml_processing_service",
    }

    violations: list[str] = []
    for route in iter_served_routes(app.routes):
        endpoint = getattr(route, "endpoint", None)
        module_name = getattr(endpoint, "__module__", None)
        if module_name in placeholder_modules:
            route_path = getattr(route, "path", "<unknown>")
            violations.append(f"{route_path} -> {module_name}")

    assert not violations, (
        "Placeholder services should not be wired to active API routes.\n"
        + "\n".join(violations)
    )
