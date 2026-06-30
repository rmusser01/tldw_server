"""
Regression test: the media list endpoint must serve both the no-slash and
trailing-slash forms directly, without a 307 redirect.

FastAPI's default ``redirect_slashes=True`` would 307 ``/api/v1/media`` ->
``/api/v1/media/``. Frontend clients normalize media-list URLs to the no-slash
form, so the redirect turned every media-list call into a redirect round-trip
(two network requests). The listing router now registers both ``""`` and ``"/"``
on the same handler to avoid that.
"""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.endpoints.media.listing import router as listing_router


def test_listing_router_serves_both_slash_forms():
    get_paths = {
        route.path
        for route in listing_router.routes
        if "GET" in (getattr(route, "methods", None) or set())
    }
    # Canonical slash form plus the no-slash alias that avoids the 307.
    assert "/" in get_paths  # nosec B101
    assert "" in get_paths  # nosec B101


@pytest.fixture
def listing_client():
    app = FastAPI()
    app.include_router(listing_router, prefix="/api/v1/media", tags=["media"])
    # follow_redirects=False so a 307 surfaces instead of being transparently followed.
    with TestClient(app, follow_redirects=False) as client:
        yield client


@pytest.mark.parametrize("path", ["/api/v1/media", "/api/v1/media/"])
def test_media_list_does_not_307_redirect(listing_client, path):
    resp = listing_client.get(f"{path}?page=1&results_per_page=5")
    # The request must reach the route handler (any non-redirect status), not be
    # bounced with a 307 trailing-slash redirect.
    assert resp.status_code != 307, (  # nosec B101
        f"{path} should be served directly, got a {resp.status_code} redirect to "
        f"{resp.headers.get('location')!r}"
    )
