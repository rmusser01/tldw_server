"""The FastAPI application must import cleanly and register its routes.

This is the most basic assertion a server project can make, and until now the
suite made it only by accident: an autouse fixture in the root conftest ran
``from tldw_Server_API.app.main import app`` before every test purely to reach
the app object it wanted to reset. Importing ``app.main`` executes the whole
route-registration graph, which is roughly nine points of the global coverage
floor -- the fixture no longer needs that import, and removing it dropped the
measured total from 12.91% to 4.11% against a 12% gate.

The import belongs somewhere that says what it is for, so it lives here as a
test rather than as a side effect of unrelated fixture code. It is at module
scope so the app is loaded during collection, which is when the fixture used to
load it.

Note that the coverage this contributes is import-time execution, not exercised
behaviour. That the floor leans on it this heavily is worth addressing on its
own; this file at least makes the dependency visible.
"""

from __future__ import annotations

import pytest

from tldw_Server_API.app.main import app


@pytest.mark.unit
def test_app_imports_and_registers_routes() -> None:
    """A broken import or an empty router is a total outage, not a subtle bug."""
    assert app.routes, "the application registered no routes"


@pytest.mark.unit
def test_app_exposes_the_versioned_api_prefix() -> None:
    """Every documented endpoint hangs off this prefix."""
    paths = {getattr(route, "path", "") for route in app.routes}
    assert any(path.startswith("/api/v1/") for path in paths), (
        "no /api/v1 routes are registered; the v1 router did not load"
    )
