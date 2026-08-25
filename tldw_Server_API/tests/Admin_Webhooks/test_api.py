"""Request-level tests for the canonical admin-webhook control plane."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.endpoints.admin import admin_webhooks
from tldw_Server_API.app.core.Admin_Webhooks.audit import MutationAudit
from tldw_Server_API.app.core.Admin_Webhooks.control_plane import (
    MutationResult,
    SecretMutationResult,
    WebhookCatalog,
)
from tldw_Server_API.app.core.Admin_Webhooks.domain import (
    WebhookError,
    WebhookErrorCode,
    WebhookLimits,
    WebhookMigrationSummary,
    WebhookRegistration,
    WebhookStatus,
    parse_registration_etag,
)
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal

NOW = datetime(2026, 8, 22, 12, 0, tzinfo=timezone.utc)
REQUEST_ID = "4aa1324c-7fb7-49cf-9058-ce0df25d5932"
IDEMPOTENCY_KEY = "0123456789abcdef0123456789abcdef"


def _registration(*, revision: int = 1) -> WebhookRegistration:
    return WebhookRegistration(
        id=41,
        description="Incident receiver",
        target_display="https://receiver.example",
        target_hostname="receiver.example",
        event_types=("incident.created",),
        active=False,
        timeout_seconds=10,
        revision=revision,
        delivery_config_version=1,
        target_version=1,
        secret_version=1,
        secret_rotation_required=False,
        created_by_user_id=7,
        updated_by_user_id=7,
        created_at=NOW,
        updated_at=NOW,
    )


def _status(*, route_selection: str = "canonical") -> WebhookStatus:
    return WebhookStatus(
        mode="on",
        route_selection=route_selection,
        schema_ready=True,
        key_state="available",
        delivery_capability_ready=False,
        limits=WebhookLimits(
            registrations=100,
            active_registrations=25,
            current_registrations=1,
            current_active_registrations=0,
        ),
        migration=WebhookMigrationSummary(
            phase="complete",
            imported_count=0,
            unresolved_count=0,
            rejected_count=0,
            secret_rotation_required_count=0,
            legacy_file_restore_permitted=True,
            rollback_expires_at=NOW + timedelta(days=7),
        ),
    )


class _FakeControlPlane:
    def __init__(self) -> None:
        self.registration = _registration()
        self.calls: list[tuple[str, object]] = []

    async def status(self) -> WebhookStatus:
        self.calls.append(("status", None))
        return _status()

    async def catalog(self) -> WebhookCatalog:
        from tldw_Server_API.app.core.Admin_Webhooks.catalog import (
            EVENT_API_VERSION,
            EVENT_CATALOG,
        )

        self.calls.append(("catalog", None))
        return WebhookCatalog(
            api_version=EVENT_API_VERSION,
            events=EVENT_CATALOG,
            registration_limit=100,
            active_limit=25,
        )

    async def list_page(self, *, limit: int, offset: int) -> SimpleNamespace:
        self.calls.append(("list_page", {"limit": limit, "offset": offset}))
        return SimpleNamespace(
            items=(self.registration,),
            total=1,
            limit=limit,
            offset=offset,
        )

    async def get(self, webhook_id: int) -> WebhookRegistration:
        self.calls.append(("get", webhook_id))
        return self.registration

    async def create(self, command, *, audit_sink) -> SecretMutationResult:
        self.calls.append(("create", command))
        await audit_sink(
            MutationAudit(
                actor_id=command.actor_id,
                action="admin_webhook.create",
                webhook_id=self.registration.id,
                target_hostname=self.registration.target_hostname,
                event_types=self.registration.event_types,
                outcome="accepted",
                request_id=command.request_id,
                reason_code=None,
            )
        )
        return SecretMutationResult(
            registration=self.registration,
            secret="whsec_" + ("a" * 64),
            replayed=False,
        )

    async def patch(self, command, *, audit_sink) -> MutationResult:
        self.calls.append(("patch", command))
        parse_registration_etag(command.if_match, expected_webhook_id=command.webhook_id)
        return MutationResult(registration=_registration(revision=2), changed=True)

    async def delete(self, command, *, audit_sink) -> MutationResult:
        self.calls.append(("delete", command))
        parse_registration_etag(command.if_match, expected_webhook_id=command.webhook_id)
        return MutationResult(registration=_registration(revision=2), changed=True)

    async def rotate_secret(self, command, *, audit_sink) -> SecretMutationResult:
        self.calls.append(("rotate_secret", command))
        parse_registration_etag(command.if_match, expected_webhook_id=command.webhook_id)
        return SecretMutationResult(
            registration=_registration(revision=2),
            secret="whsec_" + ("b" * 64),
            replayed=True,
        )


def _principal(*, admin: bool = True, user_id: int | None = 7) -> AuthPrincipal:
    return AuthPrincipal(
        kind="user" if user_id is not None else "service",
        user_id=user_id,
        subject=(f"user:{user_id}" if user_id is not None else "service:operator"),
        roles=["admin"] if admin else ["user"],
        is_admin=admin,
    )


def _api_key_principal() -> AuthPrincipal:
    return AuthPrincipal(
        kind="api_key",
        user_id=7,
        api_key_id=19,
        roles=["admin"],
        is_admin=True,
    )


def _client(
    monkeypatch: pytest.MonkeyPatch,
    *,
    principal: AuthPrincipal | None = None,
    auth_error: HTTPException | None = None,
) -> tuple[TestClient, _FakeControlPlane, AsyncMock, AsyncMock]:
    app = FastAPI()

    @app.middleware("http")
    async def request_id_middleware(request, call_next):
        request.state.request_id = REQUEST_ID
        return await call_next(request)

    async def auth_dependency() -> AuthPrincipal:
        if auth_error is not None:
            raise auth_error
        return principal or _principal()

    def require_platform_admin(value: AuthPrincipal) -> None:
        if not value.is_admin:
            raise HTTPException(
                status_code=403,
                detail="forbidden-canary",
                headers={"X-Injected": "forbidden-canary"},
            )

    service = _FakeControlPlane()
    mandatory_audit = AsyncMock()
    read_audit = AsyncMock()
    monkeypatch.setattr(admin_webhooks, "get_admin_webhook_control_plane", AsyncMock(return_value=service))
    monkeypatch.setattr(admin_webhooks, "_require_platform_admin", require_platform_admin)
    monkeypatch.setattr(admin_webhooks, "emit_mandatory_webhook_audit", mandatory_audit)
    monkeypatch.setattr(admin_webhooks, "_emit_admin_audit_event", read_audit)
    app.dependency_overrides[admin_webhooks.get_auth_principal] = auth_dependency
    app.include_router(admin_webhooks.status_router, prefix="/api/v1/admin")
    app.include_router(admin_webhooks.canonical_router, prefix="/api/v1/admin")
    return TestClient(app, raise_server_exceptions=False), service, mandatory_audit, read_audit


@pytest.mark.unit
def test_create_returns_one_time_secret_etag_and_no_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, service, mandatory_audit, _ = _client(monkeypatch)

    response = client.post(
        "/api/v1/admin/webhooks",
        headers={"Idempotency-Key": IDEMPOTENCY_KEY},
        json={
            "url": "https://receiver.example/hooks/private?token=fake",
            "event_types": ["incident.created"],
            "description": "Incident receiver",
        },
    )

    assert response.status_code == 201
    assert response.headers["etag"] == '"admin-webhook-41-r1"'
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["pragma"] == "no-cache"
    assert response.json()["signing_secret"] == "whsec_" + ("a" * 64)
    assert "url" not in response.json()["registration"]
    assert service.calls[0][0] == "create"
    mandatory_audit.assert_awaited_once()


@pytest.mark.unit
def test_patch_requires_current_etag_and_returns_new_etag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _, _, _ = _client(monkeypatch)

    missing = client.patch(
        "/api/v1/admin/webhooks/41",
        json={"description": "new"},
    )
    assert missing.status_code == 428
    assert missing.json()["error"]["code"] == "precondition_required"

    updated = client.patch(
        "/api/v1/admin/webhooks/41",
        headers={"If-Match": '"admin-webhook-41-r1"'},
        json={"description": "new"},
    )
    assert updated.status_code == 200
    assert updated.headers["etag"] == '"admin-webhook-41-r2"'


@pytest.mark.unit
def test_get_and_rotate_return_current_etags_and_rotate_is_never_cached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _, _, _ = _client(monkeypatch, principal=_api_key_principal())

    fetched = client.get("/api/v1/admin/webhooks/41")
    rotated = client.post(
        "/api/v1/admin/webhooks/41/rotate-secret",
        headers={
            "If-Match": '"admin-webhook-41-r1"',
            "Idempotency-Key": IDEMPOTENCY_KEY,
        },
    )

    assert fetched.status_code == 200
    assert fetched.headers["etag"] == '"admin-webhook-41-r1"'
    assert rotated.status_code == 200
    assert rotated.headers["etag"] == '"admin-webhook-41-r2"'
    assert rotated.headers["cache-control"] == "no-store"
    assert rotated.headers["pragma"] == "no-cache"
    assert rotated.json()["replayed"] is True


@pytest.mark.parametrize(
    ("method", "path", "kwargs"),
    (
        ("get", "/api/v1/admin/webhooks/status", {}),
        ("get", "/api/v1/admin/webhooks/catalog", {}),
        ("get", "/api/v1/admin/webhooks", {}),
        ("get", "/api/v1/admin/webhooks/41", {}),
        (
            "post",
            "/api/v1/admin/webhooks",
            {
                "headers": {"Idempotency-Key": IDEMPOTENCY_KEY},
                "json": {
                    "url": "https://receiver.example/hook",
                    "event_types": ["incident.created"],
                },
            },
        ),
        (
            "patch",
            "/api/v1/admin/webhooks/41",
            {"headers": {"If-Match": '"admin-webhook-41-r1"'}, "json": {"active": False}},
        ),
        (
            "delete",
            "/api/v1/admin/webhooks/41",
            {"headers": {"If-Match": '"admin-webhook-41-r1"'}},
        ),
        (
            "post",
            "/api/v1/admin/webhooks/41/rotate-secret",
            {
                "headers": {
                    "If-Match": '"admin-webhook-41-r1"',
                    "Idempotency-Key": IDEMPOTENCY_KEY,
                }
            },
        ),
    ),
)
@pytest.mark.unit
def test_every_canonical_route_denies_non_platform_admin(
    monkeypatch: pytest.MonkeyPatch,
    method: str,
    path: str,
    kwargs: dict[str, object],
) -> None:
    client, _, _, _ = _client(monkeypatch, principal=_principal(admin=False))

    response = client.request(method, path, **kwargs)

    assert response.status_code == 403
    assert response.json() == {
        "error": {
            "code": "platform_admin_required",
            "message": "Platform administrator access is required",
            "request_id": REQUEST_ID,
        }
    }
    assert "forbidden-canary" not in response.text
    assert "x-injected" not in response.headers


@pytest.mark.parametrize(
    ("method", "path", "kwargs"),
    (
        (
            "post",
            "/api/v1/admin/webhooks",
            {
                "headers": {"Idempotency-Key": IDEMPOTENCY_KEY},
                "json": {
                    "url": "https://receiver.example/hook",
                    "event_types": ["incident.created"],
                },
            },
        ),
        (
            "patch",
            "/api/v1/admin/webhooks/41",
            {"headers": {"If-Match": '"admin-webhook-41-r1"'}, "json": {"active": False}},
        ),
        (
            "delete",
            "/api/v1/admin/webhooks/41",
            {"headers": {"If-Match": '"admin-webhook-41-r1"'}},
        ),
        (
            "post",
            "/api/v1/admin/webhooks/41/rotate-secret",
            {
                "headers": {
                    "If-Match": '"admin-webhook-41-r1"',
                    "Idempotency-Key": IDEMPOTENCY_KEY,
                }
            },
        ),
    ),
)
@pytest.mark.unit
def test_mutations_require_user_backed_platform_admin(
    monkeypatch: pytest.MonkeyPatch,
    method: str,
    path: str,
    kwargs: dict[str, object],
) -> None:
    client, _, _, _ = _client(monkeypatch, principal=_principal(user_id=None))

    response = client.request(method, path, **kwargs)

    assert response.status_code == 403
    assert response.json()["error"]["code"] == "admin_webhook_user_principal_required"


@pytest.mark.unit
def test_validation_error_never_reflects_destination_or_forbidden_secret(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    canary = "request-secret-canary"
    client, _, _, _ = _client(monkeypatch)

    response = client.post(
        "/api/v1/admin/webhooks",
        headers={"Idempotency-Key": IDEMPOTENCY_KEY},
        json={
            "url": f"https://receiver.example/private?token={canary}",
            "event_types": ["incident.created"],
            "secret": canary,
        },
    )

    assert response.status_code == 422
    assert response.json() == {
        "error": {
            "code": "admin_webhook_validation_failed",
            "message": "Webhook request validation failed",
            "request_id": REQUEST_ID,
        }
    }
    assert response.headers["x-request-id"] == REQUEST_ID
    assert response.headers["cache-control"] == "no-store"
    assert canary not in response.text
    assert canary not in str(response.headers)
    assert canary not in caplog.text


@pytest.mark.unit
def test_authentication_error_preserves_only_exact_bearer_challenge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _, _, _ = _client(
        monkeypatch,
        auth_error=HTTPException(
            status_code=401,
            detail="authentication-canary",
            headers={
                "WWW-Authenticate": "Bearer",
                "X-Injected": "authentication-canary",
            },
        ),
    )

    response = client.get("/api/v1/admin/webhooks/status")

    assert response.status_code == 401
    assert response.json()["error"] == {
        "code": "authentication_required",
        "message": "Authentication is required",
        "request_id": REQUEST_ID,
    }
    assert response.headers["www-authenticate"] == "Bearer"
    assert "x-injected" not in response.headers
    assert "authentication-canary" not in response.text


@pytest.mark.unit
def test_unmapped_http_exception_and_auth_unavailability_drop_injected_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unavailable, _, _, _ = _client(
        monkeypatch,
        auth_error=HTTPException(
            status_code=503,
            detail="unavailable-canary",
            headers={"Retry-After": "60", "X-Injected": "unavailable-canary"},
        ),
    )
    unavailable_response = unavailable.get("/api/v1/admin/webhooks/status")
    assert unavailable_response.status_code == 503
    assert unavailable_response.json()["error"]["code"] == "authentication_unavailable"
    assert "retry-after" not in unavailable_response.headers
    assert "x-injected" not in unavailable_response.headers

    rejected, _, _, _ = _client(
        monkeypatch,
        auth_error=HTTPException(
            status_code=418,
            detail="rejected-canary",
            headers={"X-Injected": "rejected-canary"},
        ),
    )
    rejected_response = rejected.get("/api/v1/admin/webhooks/status")
    assert rejected_response.status_code == 418
    assert rejected_response.json()["error"]["code"] == "admin_webhook_request_rejected"
    assert "rejected-canary" not in rejected_response.text
    assert "x-injected" not in rejected_response.headers


@pytest.mark.parametrize(
    ("retry_after", "expected"),
    (("60", "60"), ("00060", "60"), (" 60", None), ("+60", None), ("86401", None)),
)
@pytest.mark.unit
def test_rate_limit_header_is_strictly_bounded(
    monkeypatch: pytest.MonkeyPatch,
    retry_after: str,
    expected: str | None,
) -> None:
    client, _, _, _ = _client(
        monkeypatch,
        auth_error=HTTPException(
            status_code=429,
            detail="rate-canary",
            headers={"Retry-After": retry_after, "X-Injected": "rate-canary"},
        ),
    )

    response = client.get("/api/v1/admin/webhooks/status")

    assert response.status_code == 429
    assert response.json()["error"]["code"] == "authentication_rate_limited"
    assert response.headers.get("retry-after") == expected
    assert "x-injected" not in response.headers


@pytest.mark.unit
def test_list_uses_server_total_without_synthesizing_etag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, service, _, read_audit = _client(monkeypatch)

    response = client.get("/api/v1/admin/webhooks?limit=25&offset=10")

    assert response.status_code == 200
    assert response.json()["total"] == 1
    assert response.json()["limit"] == 25
    assert response.json()["offset"] == 10
    assert "etag" not in response.headers
    assert service.calls == [("list_page", {"limit": 25, "offset": 10})]
    metadata = read_audit.await_args.kwargs["metadata"]
    assert metadata == {
        "outcome": "succeeded",
        "request_id": REQUEST_ID,
        "result_count": 1,
    }


@pytest.mark.unit
def test_read_audit_failure_does_not_change_successful_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _, _, read_audit = _client(monkeypatch)
    read_audit.side_effect = RuntimeError("audit-canary")
    warning = Mock()
    monkeypatch.setattr(
        admin_webhooks,
        "logger",
        SimpleNamespace(warning=warning),
        raising=False,
    )

    response = client.get("/api/v1/admin/webhooks/catalog")

    assert response.status_code == 200
    assert response.json()["api_version"] == "2026-07-01"
    assert "audit-canary" not in response.text
    warning.assert_called_once_with(
        "Admin webhook read audit failed action={} request_id={} error_type={}",
        "admin_webhook.catalog.read",
        REQUEST_ID,
        "RuntimeError",
    )
    assert "audit-canary" not in repr(warning.call_args)


@pytest.mark.unit
def test_domain_failures_use_closed_status_code_and_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, service, _, _ = _client(monkeypatch)
    service.status = AsyncMock(side_effect=WebhookError(WebhookErrorCode.KEY_CONFIGURATION_MISMATCH))

    response = client.get("/api/v1/admin/webhooks/status")

    assert response.status_code == 503
    assert response.json()["error"] == {
        "code": "admin_webhook_key_configuration_mismatch",
        "message": "Webhook encryption key configuration does not match durable state",
        "request_id": REQUEST_ID,
    }


@pytest.mark.unit
def test_control_plane_factory_failure_is_read_audited(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _, _, read_audit = _client(monkeypatch)
    monkeypatch.setattr(
        admin_webhooks,
        "get_admin_webhook_control_plane",
        AsyncMock(side_effect=WebhookError(WebhookErrorCode.KEY_UNAVAILABLE)),
    )

    response = client.get("/api/v1/admin/webhooks/status")

    assert response.status_code == 503
    assert read_audit.await_args.kwargs["metadata"] == {
        "outcome": "failed",
        "request_id": REQUEST_ID,
        "reason_code": "admin_webhook_key_unavailable",
    }
