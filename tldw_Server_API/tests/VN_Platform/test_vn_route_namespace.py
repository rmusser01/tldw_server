from __future__ import annotations

import pytest
from fastapi import FastAPI

from tldw_Server_API.app.api.v1.router_groups.content import iter_content_router_specs
from tldw_Server_API.app.api.v1.router_registry import register_router_specs

pytestmark = pytest.mark.integration


def test_vn_routes_are_registered_under_canonical_namespace() -> None:
    app = FastAPI()
    vn_specs = [
        spec
        for spec in iter_content_router_specs()
        if spec.route_key in {"vn-capabilities", "vn-assets", "vn-play"}
    ]

    register_router_specs(app, vn_specs)

    paths = {getattr(route, "path", "") for route in app.routes}
    assert "/api/v1/vn/vn-capabilities" in paths
    assert "/api/v1/vn/vn-assets/packs" in paths
    assert "/api/v1/vn/vn-play/sessions" in paths
    assert "/api/v1/vn-assets/packs" not in paths
    assert "/api/v1/vn-play/sessions" not in paths
