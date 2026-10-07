"""Keep governor routes mounted when the endpoint is imported before main."""

import os
import subprocess  # nosec B404 - a fresh interpreter is required to test import order.
import sys

import pytest


@pytest.mark.unit
def test_endpoint_first_import_registers_protected_governor_routes():
    env = os.environ.copy()
    env.update(
        TEST_MODE="1", TLDW_TEST_MODE="1", MINIMAL_TEST_APP="1",
        AUTH_MODE="single_user", SINGLE_USER_API_KEY="test-rg-import-order",
    )
    result = subprocess.run(  # nosec B603 - fixed interpreter and literal Python code; no input.
        [sys.executable, "-c", """
from tldw_Server_API.app.api.v1.endpoints import resource_governor
from tldw_Server_API.app.main import app
from tldw_Server_API.app.core.Utils.fastapi_routes import iter_served_routes
routes = list(iter_served_routes(app.routes))
route = next((r for r in routes if getattr(r, "path", None) == "/api/v1/resource-governor/diag/capabilities"), None)
assert route is not None, "endpoint-first import lost the governor router"
assert "GET" in route.methods
assert route.dependencies, "governor diagnostics must retain the admin dependency"
assert resource_governor._get_app() is app
"""],
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
