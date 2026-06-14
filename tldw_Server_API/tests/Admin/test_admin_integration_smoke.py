"""Admin integration smoke tests.

Verifies that all admin sub-routers are registered and that key endpoints
appear in the application route table.  These tests do NOT hit a live
database -- they inspect the FastAPI app object directly.
"""
from __future__ import annotations

import importlib

import pytest

# ---------------------------------------------------------------------------
# Helper: collect all route paths from the app
# ---------------------------------------------------------------------------

def _get_all_route_paths() -> set[str]:
    """Import the FastAPI app and return the set of registered route paths."""
    from tldw_Server_API.app.main import app
    return {route.path for route in app.routes if hasattr(route, "path")}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAdminRouterRegistration:
    """Verify that every admin sub-router is included in the main router."""

    def test_minimal_test_app_requires_admin_router(self) -> None:
        """Minimal test app should fail fast if the admin aggregate is unavailable."""
        from tldw_Server_API.app.api.v1.router_groups import minimal

        specs = {spec.name: spec for spec in minimal.iter_minimal_test_router_specs()}

        assert "admin" in minimal.MINIMAL_REQUIRED_ROUTER_NAMES  # nosec B101
        assert "admin" in specs  # nosec B101
        assert specs["admin"].skip_exceptions == minimal.REQUIRED_ROUTER_SKIP_EXCEPTIONS  # nosec B101

    def test_admin_init_does_not_import_admin_billing_router(self) -> None:
        """OSS admin router assembly should not import admin billing revenue operations."""
        admin_init = importlib.import_module(
            "tldw_Server_API.app.api.v1.endpoints.admin"
        )
        expected_submodules = [
            "admin_api_keys",
            "admin_budgets",
            "admin_bundle_ops",
            "admin_byok",
            "admin_circuit_breakers",
            "admin_data_ops",
            "admin_llm_providers",
            "admin_monitoring",
            "admin_network",
            "admin_ops",
            "admin_orgs",
            "admin_personalization",
            "admin_profiles",
            "admin_rate_limits",
            "admin_rbac",
            "admin_registration",
            "admin_sessions_mfa",
            "admin_settings",
            "admin_system",
            "admin_tools",
            "admin_usage",
            "admin_router_analytics",
            "admin_acp_agents",
            "admin_events_stream",
            "admin_storage_quotas",
            "admin_user",
            "admin_tenant_provisioning",
            "admin_impersonation",
        ]
        for submod_name in expected_submodules:
            attr_name = f"{submod_name}_endpoints"
            assert hasattr(admin_init, attr_name), (
                f"admin __init__ missing attribute '{attr_name}' -- "
                f"sub-router '{submod_name}' may not be imported"
            )  # nosec B101
        assert not hasattr(admin_init, "admin_billing_endpoints")  # nosec B101


class TestAdminKeyEndpointsExist:
    """Spot-check that critical admin endpoints are in the route table."""

    EXPECTED_ENDPOINTS = [
        "/api/v1/admin/roles",
        "/api/v1/admin/permissions",
        "/api/v1/admin/cleanup-settings",
        "/api/v1/admin/registration-settings",
        "/api/v1/admin/rate-limits",
        "/api/v1/admin/roles/{role_id}/rate-limits",
        "/api/v1/admin/users",
    ]

    def test_key_endpoints_present(self) -> None:
        paths = _get_all_route_paths()
        for ep in self.EXPECTED_ENDPOINTS:
            assert ep in paths, f"Expected endpoint {ep!r} not found in route table"  # nosec B101

    @pytest.mark.parametrize(
        "path",
        [
            "/api/v1/admin/billing/overview",
            "/api/v1/admin/billing/subscriptions",
            "/api/v1/admin/billing/subscriptions/{user_id}",
            "/api/v1/admin/billing/subscriptions/{user_id}/override",
            "/api/v1/admin/billing/subscriptions/{user_id}/credits",
            "/api/v1/admin/billing/events",
        ],
    )
    def test_admin_billing_endpoints_absent_from_oss_route_table(self, path: str) -> None:
        paths = _get_all_route_paths()
        assert path not in paths, f"OSS route table should not expose {path!r}"  # nosec B101


class TestAdminSchemaImports:
    """Verify that key admin response/request schemas can be imported."""

    def test_cleanup_settings_schemas(self) -> None:
        from tldw_Server_API.app.api.v1.endpoints.admin import admin_settings as mod
        assert hasattr(mod, "router")  # nosec B101

    def test_registration_settings_schemas(self) -> None:
        from tldw_Server_API.app.api.v1.endpoints.admin import admin_registration as mod
        assert hasattr(mod, "router")  # nosec B101

    def test_tenant_provisioning_schemas(self) -> None:
        from tldw_Server_API.app.api.v1.endpoints.admin import admin_tenant_provisioning as mod
        assert hasattr(mod, "router")  # nosec B101
