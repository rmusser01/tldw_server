"""
Tests for Resource Governor endpoint coverage audit.

Covers:
- Correct route classification (protected vs unprotected)
- Coverage percentage calculation
- Custom excluded prefixes
- Edge cases (empty app, all excluded, none excluded)
"""
from __future__ import annotations

import pytest

from tldw_Server_API.app.core.Resource_Governance.coverage_audit import (
    DEFAULT_EXCLUDED_PREFIXES,
    audit_governor_coverage,
)


# ---------------------------------------------------------------------------
# Helpers: mock FastAPI app with routes
# ---------------------------------------------------------------------------


class _MockRoute:
    """Minimal mock of a FastAPI route."""

    def __init__(self, path: str, methods: set[str] | None = None):
        self.path = path
        self.methods = methods or {"GET"}
        self.tags = []


class _MockApp:
    """Minimal mock of a FastAPI app."""

    def __init__(self, routes: list[_MockRoute] | None = None, *, user_middleware=None, state=None):
        self.routes = routes or []
        self.user_middleware = user_middleware or []
        self.state = state


class _Middleware:
    def __init__(self, cls):
        self.cls = cls


class _RGSimpleMiddleware:
    pass


class _Snap:
    def __init__(self, route_map):
        self.route_map = route_map


class _Loader:
    def __init__(self, route_map):
        self._snap = _Snap(route_map)

    def get_snapshot(self):
        return self._snap


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAuditGovernorCoverage:
    """Tests for audit_governor_coverage."""

    def test_empty_app_returns_zero_coverage(self):
        app = _MockApp(routes=[])
        result = audit_governor_coverage(app)

        assert result["total_routes"] == 0
        assert result["protected_count"] == 0
        assert result["unprotected_count"] == 0
        assert result["coverage_pct"] == 0.0

    def test_all_protected(self):
        """All routes under /api/ are considered protected."""
        app = _MockApp(routes=[
            _MockRoute("/api/v1/chat", {"POST"}),
            _MockRoute("/api/v1/media", {"GET"}),
            _MockRoute("/api/v1/rag/search", {"POST"}),
        ], user_middleware=[_Middleware(_RGSimpleMiddleware)], state=type("State", (), {
            "rg_policy_loader": _Loader({"by_path": {"/api/v1/*": "api.default"}})
        })())
        result = audit_governor_coverage(app)

        assert result["total_routes"] == 3
        assert result["protected_count"] == 3
        assert result["unprotected_count"] == 0
        assert result["coverage_pct"] == 100.0

    def test_excluded_routes_are_unprotected(self):
        """Routes under excluded prefixes are classified as unprotected."""
        app = _MockApp(routes=[
            _MockRoute("/api/v1/chat", {"POST"}),
            _MockRoute("/docs", {"GET"}),
            _MockRoute("/healthz", {"GET"}),
            _MockRoute("/openapi.json", {"GET"}),
        ], user_middleware=[_Middleware(_RGSimpleMiddleware)], state=type("State", (), {
            "rg_policy_loader": _Loader({"by_path": {"/api/v1/chat": "chat.default"}})
        })())
        result = audit_governor_coverage(app)

        assert result["total_routes"] == 4
        assert result["protected_count"] == 1
        assert result["unprotected_count"] == 3
        assert result["coverage_pct"] == 25.0

    def test_custom_excluded_prefixes(self):
        """Custom excluded prefixes override defaults."""
        app = _MockApp(routes=[
            _MockRoute("/api/v1/chat", {"POST"}),
            _MockRoute("/api/v1/internal/debug", {"GET"}),
            _MockRoute("/docs", {"GET"}),
        ], user_middleware=[_Middleware(_RGSimpleMiddleware)], state=type("State", (), {
            "rg_policy_loader": _Loader({"by_path": {
                "/api/v1/chat": "chat.default",
                "/docs": "docs.default",
            }})
        })())
        result = audit_governor_coverage(
            app, excluded_prefixes=["/api/v1/internal"]
        )

        # /docs is NOT excluded with custom prefixes (unless included)
        assert result["protected_count"] == 2  # /api/v1/chat + /docs
        assert result["unprotected_count"] == 1  # /api/v1/internal/debug

    def test_multiple_methods_counted_separately(self):
        """Each HTTP method on a route counts as a separate entry."""
        app = _MockApp(routes=[
            _MockRoute("/api/v1/items", {"GET", "POST", "DELETE"}),
        ], user_middleware=[_Middleware(_RGSimpleMiddleware)], state=type("State", (), {
            "rg_policy_loader": _Loader({"by_path": {"/api/v1/items": "items.default"}})
        })())
        result = audit_governor_coverage(app)

        assert result["total_routes"] == 3
        assert result["protected_count"] == 3

    def test_default_excluded_prefixes_used(self):
        """Default excluded prefixes are applied when none specified."""
        app = _MockApp(routes=[
            _MockRoute("/health", {"GET"}),
            _MockRoute("/readyz", {"GET"}),
            _MockRoute("/api/v1/chat", {"POST"}),
        ], user_middleware=[_Middleware(_RGSimpleMiddleware)], state=type("State", (), {
            "rg_policy_loader": _Loader({"by_path": {"/api/v1/chat": "chat.default"}})
        })())
        result = audit_governor_coverage(app)

        assert result["excluded_prefixes"] == DEFAULT_EXCLUDED_PREFIXES
        assert result["unprotected_count"] == 2
        assert result["protected_count"] == 1

    def test_unprotected_routes_capped_at_50(self):
        """Unprotected routes list is capped at 50 for readability."""
        routes = [_MockRoute(f"/docs/page{i}", {"GET"}) for i in range(60)]
        app = _MockApp(routes=routes)
        result = audit_governor_coverage(app)

        assert len(result["unprotected_routes"]) == 50
        assert result["unprotected_count"] == 60

    def test_protected_routes_capped_at_50(self):
        """Protected routes list is capped at 50 for readability."""
        routes = [_MockRoute(f"/api/v1/endpoint{i}", {"GET"}) for i in range(60)]
        app = _MockApp(routes=routes, user_middleware=[_Middleware(_RGSimpleMiddleware)], state=type("State", (), {
            "rg_policy_loader": _Loader({"by_path": {"/api/v1/*": "api.default"}})
        })())
        result = audit_governor_coverage(app)

        assert len(result["protected_routes"]) == 50
        assert result["protected_count"] == 60

    def test_route_limit_returns_full_lists(self):
        """route_limit lets admin callers fetch the complete audit (#2890)."""
        routes = [_MockRoute(f"/docs/page{i}", {"GET"}) for i in range(60)]
        app = _MockApp(routes=routes)
        result = audit_governor_coverage(app, route_limit=1000)

        assert len(result["unprotected_routes"]) == 60
        assert result["unprotected_count"] == 60
        assert result["route_list_limit"] == 1000

    def test_route_list_limit_reported_for_default_cap(self):
        """Clients can detect truncation from the reported list limit (#2890)."""
        routes = [_MockRoute(f"/docs/page{i}", {"GET"}) for i in range(60)]
        app = _MockApp(routes=routes)
        result = audit_governor_coverage(app)

        assert result["route_list_limit"] == 50
        assert len(result["unprotected_routes"]) == result["route_list_limit"]

    def test_routes_without_methods_skipped(self):
        """Routes without methods attribute are skipped (e.g., Mount)."""

        class _MountRoute:
            def __init__(self, path: str):
                self.path = path
                # No 'methods' attribute

        app = _MockApp(routes=[
            _MockRoute("/api/v1/chat", {"POST"}),
            _MountRoute("/static"),
        ], user_middleware=[_Middleware(_RGSimpleMiddleware)], state=type("State", (), {
            "rg_policy_loader": _Loader({"by_path": {"/api/v1/chat": "chat.default"}})
        })())
        result = audit_governor_coverage(app)

        assert result["total_routes"] == 1

    def test_coverage_percentage_rounded(self):
        """Coverage percentage is rounded to 1 decimal."""
        app = _MockApp(routes=[
            _MockRoute("/api/v1/a", {"GET"}),
            _MockRoute("/api/v1/b", {"GET"}),
            _MockRoute("/docs", {"GET"}),
        ], user_middleware=[_Middleware(_RGSimpleMiddleware)], state=type("State", (), {
            "rg_policy_loader": _Loader({"by_path": {"/api/v1/*": "api.default"}})
        })())
        result = audit_governor_coverage(app)

        # 2/3 = 66.666...% -> 66.7
        assert result["coverage_pct"] == 66.7

    def test_api_route_without_middleware_or_route_map_is_unprotected(self):
        app = _MockApp(routes=[
            _MockRoute("/api/v1/chat", {"POST"}),
        ])

        result = audit_governor_coverage(app)

        assert result["protected_count"] == 0
        assert result["unprotected_count"] == 1
        assert result["coverage_pct"] == 0.0
        assert result["unprotected_routes"][0]["reason"] == "rg_middleware_missing"
