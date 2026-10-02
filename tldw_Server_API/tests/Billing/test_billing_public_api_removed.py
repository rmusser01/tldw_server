from __future__ import annotations

import importlib
import os


os.environ["ALLOWED_ORIGINS"] = "http://localhost:3000"

from tldw_Server_API.app.core.Utils.fastapi_routes import iter_served_routes
from tldw_Server_API.app import main as app_main

app_main = importlib.reload(app_main)


def test_public_app_registers_no_billing_routes() -> None:
    """OSS should not expose any public billing routes."""
    route_paths = {getattr(route, "path", "") for route in iter_served_routes(app_main.app.routes)}

    assert "/api/v1/billing/webhooks/stripe" not in route_paths
    assert not any(path.startswith("/api/v1/billing") for path in route_paths)
