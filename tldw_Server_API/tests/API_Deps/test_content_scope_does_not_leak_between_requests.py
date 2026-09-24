"""The content scope ContextVar must not survive into another request.

auth_deps._activate_scope_context calls set_scope and keeps no reset token,
which reads like a leak: a per-request identity set and never cleared, feeding
the PostgreSQL tenant GUCs. It is not one, because Starlette runs every request
in its own copied context -- threadpool handlers included -- so a value set in
one request is invisible to the next.

That is an assumption about framework behaviour, and this pins it. If a
Starlette upgrade ever changed it, an unreset scope would become a genuine
cross-account leak and these assertions would catch it.

Work that outlives a request is deliberately not covered: a task spawned inside
one captures the context at creation, so resetting at request end would not
help it anyway. Such workers establish their own scope.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.core.DB_Management.scope_context import get_scope, set_scope

pytestmark = pytest.mark.unit


@pytest.fixture()
def client():
    app = FastAPI()

    @app.get("/set-async")
    async def set_async():
        set_scope(user_id=111)  # set and never reset, exactly as auth_deps does
        return {"ok": True}

    @app.get("/read-async")
    async def read_async():
        scope = get_scope()
        return {"seen": None if scope is None else scope.user_id}

    @app.get("/set-sync")
    def set_sync():
        set_scope(user_id=222)
        return {"ok": True}

    @app.get("/read-sync")
    def read_sync():
        scope = get_scope()
        return {"seen": None if scope is None else scope.user_id}

    return TestClient(app)


def test_async_request_does_not_inherit_a_previous_scope(client):
    client.get("/set-async")

    assert client.get("/read-async").json()["seen"] is None


def test_threadpool_request_does_not_inherit_a_previous_scope(client):
    """Sync handlers run on a shared threadpool, the likeliest place to leak."""
    client.get("/set-sync")

    assert client.get("/read-sync").json()["seen"] is None


def test_scope_does_not_cross_between_sync_and_async_handlers(client):
    client.get("/set-sync")

    assert client.get("/read-async").json()["seen"] is None


def test_repeated_threadpool_requests_never_see_a_stale_scope(client):
    """One pass could miss a leak by landing on a fresh worker thread."""
    client.get("/set-sync")

    seen = [client.get("/read-sync").json()["seen"] for _ in range(25)]

    assert seen == [None] * 25
