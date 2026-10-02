import contextlib

from fastapi.testclient import TestClient


@contextlib.contextmanager
def _with_request_id_middleware(app):
    """Guarantee RequestIDMiddleware is active on the shared app for this test.

    `app` is the process-wide FastAPI singleton. Other autouse test fixtures
    (e.g. AuthNZ's `reset_singletons`, which becomes global for the whole
    xdist worker once any file opts into the `authnz_full_fixtures` plugin)
    intentionally strip RequestIDMiddleware from `app.user_middleware` for
    the duration of *their own* test to quiet background-task noise -- with
    no way for this unrelated test to opt out of that scope leak (TASK-13400).
    Re-add it here, scoped to just this test, mirroring the
    `_with_rg_middleware` pattern used by the Resource_Governance e2e tests.
    """
    from starlette.middleware import Middleware

    from tldw_Server_API.app.core.Security.request_id_middleware import RequestIDMiddleware

    original_user_middleware = getattr(app, "user_middleware", [])[:]
    changed = False
    try:
        already = any(getattr(m, "cls", None) is RequestIDMiddleware for m in original_user_middleware)
        if not already:
            app.user_middleware = [Middleware(RequestIDMiddleware), *original_user_middleware]
            changed = True
            app.middleware_stack = app.build_middleware_stack()
        yield
    finally:
        if changed:
            app.user_middleware = original_user_middleware
            app.middleware_stack = app.build_middleware_stack()


def test_health_includes_trace_headers(monkeypatch):


     # Reduce startup work
    monkeypatch.setenv("TEST_MODE", "true")

    from tldw_Server_API.app.main import app

    with _with_request_id_middleware(app):
        client = TestClient(app)

        r = client.get("/health")
        assert r.status_code == 200
        # RequestIDMiddleware should set request id
        assert "X-Request-ID" in r.headers
        # Trace headers middleware should attach trace headers
        assert "traceparent" in r.headers
        assert "X-Trace-Id" in r.headers
