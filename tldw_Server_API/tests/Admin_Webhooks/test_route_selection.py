"""Startup route-selection tests for canonical and temporary legacy webhooks."""

from __future__ import annotations

import pytest
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.endpoints import admin as admin_endpoints
from tldw_Server_API.app.api.v1.endpoints.admin import admin_ops
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal


def _pairs(router: APIRouter) -> list[tuple[str, str]]:
    return [(method, route.path) for route in router.routes for method in (route.methods or set())]


def _build(environ: dict[str, str]) -> APIRouter:
    router = APIRouter(prefix="/admin")
    admin_endpoints._mount_admin_webhook_routes(router, environ)
    return router


@pytest.mark.parametrize(
    "environ",
    (
        {"TLDW_ADMIN_WEBHOOKS_MODE": "off", "TLDW_ADMIN_WEBHOOKS_LEGACY_COMPAT": "false"},
        {"TLDW_ADMIN_WEBHOOKS_MODE": "off", "TLDW_ADMIN_WEBHOOKS_LEGACY_COMPAT": "true"},
        {"TLDW_ADMIN_WEBHOOKS_MODE": "migrate", "TLDW_ADMIN_WEBHOOKS_LEGACY_COMPAT": "false"},
        {"TLDW_ADMIN_WEBHOOKS_MODE": "on", "TLDW_ADMIN_WEBHOOKS_LEGACY_COMPAT": "false"},
    ),
)
def test_selected_router_has_no_duplicate_method_path_pairs(
    environ: dict[str, str],
) -> None:
    pairs = _pairs(_build(environ))
    assert len(pairs) == len(set(pairs))


def test_canonical_selection_excludes_legacy_delivery_routes() -> None:
    pairs = set(
        _pairs(
            _build(
                {
                    "TLDW_ADMIN_WEBHOOKS_MODE": "off",
                    "TLDW_ADMIN_WEBHOOKS_LEGACY_COMPAT": "false",
                }
            )
        )
    )

    assert ("GET", "/admin/webhooks/status") in pairs
    assert ("GET", "/admin/webhooks/catalog") in pairs
    assert ("POST", "/admin/webhooks/{webhook_id}/rotate-secret") in pairs
    assert ("POST", "/admin/webhooks/{webhook_id}/test") not in pairs
    assert ("GET", "/admin/webhooks/{webhook_id}/deliveries") not in pairs
    assert ("POST", "/admin/incidents/{incident_id}/notify-webhooks") not in pairs


def test_legacy_selection_excludes_canonical_catalog_and_rotation() -> None:
    pairs = set(
        _pairs(
            _build(
                {
                    "TLDW_ADMIN_WEBHOOKS_MODE": "off",
                    "TLDW_ADMIN_WEBHOOKS_LEGACY_COMPAT": "true",
                }
            )
        )
    )

    assert ("GET", "/admin/webhooks/status") in pairs
    assert ("GET", "/admin/webhooks") in pairs
    assert ("POST", "/admin/webhooks/{webhook_id}/test") in pairs
    assert ("GET", "/admin/webhooks/{webhook_id}/deliveries") in pairs
    assert ("POST", "/admin/incidents/{incident_id}/notify-webhooks") in pairs
    assert ("GET", "/admin/webhooks/catalog") not in pairs
    assert ("POST", "/admin/webhooks/{webhook_id}/rotate-secret") not in pairs


@pytest.mark.parametrize("mode", ("migrate", "on"))
def test_legacy_compatibility_is_rejected_outside_off_mode(mode: str) -> None:
    with pytest.raises(ValueError, match="Legacy webhook compatibility"):
        _build(
            {
                "TLDW_ADMIN_WEBHOOKS_MODE": mode,
                "TLDW_ADMIN_WEBHOOKS_LEGACY_COMPAT": "true",
            }
        )


def test_canonical_off_warning_is_fixed_and_emitted_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(admin_endpoints.logger, "warning", lambda *args, **kwargs: calls.append(args))
    _build(
        {
            "TLDW_ADMIN_WEBHOOKS_MODE": "off",
            "TLDW_ADMIN_WEBHOOKS_LEGACY_COMPAT": "false",
        }
    )

    assert calls == [
        (
            "Canonical admin webhook mode is off; historical webhook CRUD, test, delivery, and incident-notify routes are disabled. Explicitly select temporary compatibility or migration before use.",
        )
    ]


def test_legacy_create_and_update_audits_omit_target_and_event_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canary = "compatibility-audit-canary"
    principal = AuthPrincipal(
        kind="user",
        user_id=7,
        roles=["admin"],
        is_admin=True,
    )
    audit_calls: list[dict[str, object]] = []

    async def auth_dependency() -> AuthPrincipal:
        return principal

    async def capture_audit(*args, **kwargs) -> None:
        del args
        audit_calls.append(kwargs)

    monkeypatch.setattr(admin_ops, "_require_platform_admin", lambda _principal: None)
    monkeypatch.setattr(admin_ops, "_emit_admin_audit_event", capture_audit)
    monkeypatch.setattr(
        admin_ops,
        "svc_create_webhook",
        lambda **_kwargs: {
            "id": "wh_test",
            "url": f"https://receiver.example/private?token={canary}",
            "secret": canary,
            "events": [f"incident.{canary}"],
            "enabled": True,
            "created_at": None,
            "updated_at": None,
        },
    )
    monkeypatch.setattr(
        admin_ops,
        "svc_update_webhook",
        lambda **_kwargs: {
            "id": "wh_test",
            "url": f"https://receiver.example/changed?token={canary}",
            "events": [f"user.{canary}", "incident.created"],
            "enabled": False,
            "created_at": None,
            "updated_at": None,
        },
    )

    app = FastAPI()
    app.dependency_overrides[admin_ops.get_auth_principal] = auth_dependency
    app.include_router(admin_ops.legacy_webhooks_router, prefix="/api/v1/admin")
    client = TestClient(app)

    created = client.post(
        "/api/v1/admin/webhooks",
        json={
            "url": f"https://receiver.example/private?token={canary}",
            "events": ["incident.created"],
            "enabled": True,
        },
    )
    updated = client.patch(
        "/api/v1/admin/webhooks/wh_test",
        json={
            "url": f"https://receiver.example/changed?token={canary}",
            "events": ["incident.created", "user.created"],
            "enabled": False,
        },
    )

    assert created.status_code == 200
    assert updated.status_code == 200
    assert [call["metadata"] for call in audit_calls] == [
        {"event_count": 1, "enabled": True},
        {"event_count": 2, "enabled": False},
    ]
    assert canary not in repr(audit_calls)
