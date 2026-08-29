"""Request-level tests for the canonical admin-webhook control plane."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

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
    AttemptState,
    DeliveryKind,
    DeliveryState,
    WebhookDelivery,
    WebhookDeliveryAttempt,
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
DELIVERY_ID = "e8614dc3-8f6f-4e06-89b5-b6cded708aa4"
SOURCE_DELIVERY_ID = "cf5e58a3-2f55-437d-8bfd-f2e5987c4449"
ATTEMPT_ID = "1a0094d8-94a5-4544-890c-4b7281fdc741"


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


def _delivery(
    *,
    delivery_id: str = DELIVERY_ID,
    kind: DeliveryKind = DeliveryKind.TEST,
    state: DeliveryState = DeliveryState.SUCCEEDED,
    redelivery_of_id: str | None = None,
) -> WebhookDelivery:
    return WebhookDelivery(
        id=delivery_id,
        event_id="56dcb152-8dda-46d4-880f-e71b3b99b6af",
        webhook_id=41,
        kind=kind,
        state=state,
        delivery_config_version=1,
        secret_version=1,
        attempt_count=1 if kind is DeliveryKind.TEST else 0,
        status_code=204 if state is DeliveryState.SUCCEEDED else None,
        latency_ms=8 if state is DeliveryState.SUCCEEDED else None,
        reason_code=None,
        expires_at=NOW + timedelta(hours=72),
        created_at=NOW,
        updated_at=NOW,
        terminal_at=NOW if state is DeliveryState.SUCCEEDED else None,
        redelivery_of_id=redelivery_of_id,
    )


def _attempt(
    *,
    state: AttemptState = AttemptState.SUCCEEDED,
) -> WebhookDeliveryAttempt:
    return WebhookDeliveryAttempt(
        id=ATTEMPT_ID,
        delivery_id=DELIVERY_ID,
        attempt_number=1,
        state=state,
        request_timeout_seconds=10,
        status_code=204 if state is AttemptState.SUCCEEDED else None,
        latency_ms=8 if state is AttemptState.SUCCEEDED else None,
        reason_code=None,
        requested_retry_delay_seconds=None,
        started_at=NOW,
        finished_at=NOW if state is not AttemptState.PROCESSING else None,
    )


class _FakeDeliveryService:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []
        self.processing = False
        self.fail_test_before_audit = False
        self.fail_redelivery_before_audit = False

    async def list_delivery_history(
        self,
        webhook_id: int,
        *,
        limit: int,
        offset: int,
    ) -> SimpleNamespace:
        self.calls.append(
            ("list_delivery_history", (webhook_id, limit, offset))
        )
        item = SimpleNamespace(
            delivery=_delivery(),
            event_type="webhook.test",
            completed_after_config_change=False,
            attempts=(_attempt(),),
        )
        return SimpleNamespace(items=(item,), total=1, limit=limit, offset=offset)

    async def test_webhook(self, command, *, audit_sink) -> SimpleNamespace:
        from tldw_Server_API.app.core.Admin_Webhooks import delivery as delivery_module

        self.calls.append(("test_webhook", command))
        if self.fail_test_before_audit:
            raise WebhookError(WebhookErrorCode.PRECONDITION_FAILED)
        state = AttemptState.PROCESSING if self.processing else AttemptState.SUCCEEDED
        delivery_state = (
            DeliveryState.PROCESSING if self.processing else DeliveryState.SUCCEEDED
        )
        await audit_sink(
            delivery_module.TestWebhookAudit(
                actor_id=command.actor_id,
                webhook_id=command.webhook_id,
                delivery_id=DELIVERY_ID,
                attempt_id=ATTEMPT_ID,
                target_hostname="receiver.example",
                request_id=command.request_id,
                outcome="accepted" if self.processing else "succeeded",
                status_code=None if self.processing else 204,
                reason_code=None,
            )
        )
        return SimpleNamespace(
            delivery=_delivery(state=delivery_state),
            attempt=_attempt(state=state),
            idempotent_replay=self.processing,
            in_progress=self.processing,
            retry_after_seconds=5 if self.processing else None,
            completed_after_config_change=False,
        )

    async def redeliver_webhook(self, command, *, audit_sink) -> SimpleNamespace:
        from tldw_Server_API.app.core.Admin_Webhooks import audit as audit_module

        self.calls.append(("redeliver_webhook", command))
        if self.fail_redelivery_before_audit:
            raise WebhookError(WebhookErrorCode.NOT_FOUND)
        delivery = _delivery(
            kind=DeliveryKind.MANUAL,
            state=DeliveryState.PENDING,
            redelivery_of_id=command.source_delivery_id,
        )
        await audit_sink(
            audit_module.DeliveryMutationAudit(
                actor_id=command.actor_id,
                action="admin_webhook.redeliver",
                webhook_id=command.webhook_id,
                source_delivery_id=command.source_delivery_id,
                delivery_id=delivery.id,
                attempt_id=None,
                target_hostname="receiver.example",
                source_config_version=1,
                current_config_version=1,
                redelivery_to_changed_config=False,
                status_code=None,
                outcome="accepted",
                request_id=command.request_id,
                reason_code=None,
            )
        )
        return SimpleNamespace(
            delivery=delivery,
            event_type="user.created",
            completed_after_config_change=False,
            idempotent_replay=False,
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
    delivery_service = _FakeDeliveryService()
    mandatory_audit = AsyncMock()
    read_audit = AsyncMock()
    monkeypatch.setattr(admin_webhooks, "get_admin_webhook_control_plane", AsyncMock(return_value=service))
    monkeypatch.setattr(admin_webhooks, "_require_platform_admin", require_platform_admin)
    monkeypatch.setattr(admin_webhooks, "emit_mandatory_webhook_audit", mandatory_audit)
    monkeypatch.setattr(admin_webhooks, "_emit_admin_audit_event", read_audit)
    app.dependency_overrides[admin_webhooks.get_auth_principal] = auth_dependency
    if hasattr(admin_webhooks, "get_admin_webhook_delivery_service"):
        app.dependency_overrides[
            admin_webhooks.get_admin_webhook_delivery_service
        ] = lambda: delivery_service
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
        ("get", "/api/v1/admin/webhooks/41/deliveries", {}),
        (
            "post",
            "/api/v1/admin/webhooks/41/test",
            {
                "headers": {
                    "If-Match": '"admin-webhook-41-r1"',
                    "Idempotency-Key": IDEMPOTENCY_KEY,
                },
                "json": {"delivery_config_version": 1},
            },
        ),
        (
            "post",
            f"/api/v1/admin/webhooks/41/deliveries/{SOURCE_DELIVERY_ID}/redeliver",
            {
                "headers": {
                    "If-Match": '"admin-webhook-41-r1"',
                    "Idempotency-Key": IDEMPOTENCY_KEY,
                },
                "json": {
                    "delivery_config_version": 1,
                    "confirm_changed_configuration": False,
                },
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
        (
            "post",
            "/api/v1/admin/webhooks/41/test",
            {
                "headers": {
                    "If-Match": '"admin-webhook-41-r1"',
                    "Idempotency-Key": IDEMPOTENCY_KEY,
                },
                "json": {"delivery_config_version": 1},
            },
        ),
        (
            "post",
            f"/api/v1/admin/webhooks/41/deliveries/{SOURCE_DELIVERY_ID}/redeliver",
            {
                "headers": {
                    "If-Match": '"admin-webhook-41-r1"',
                    "Idempotency-Key": IDEMPOTENCY_KEY,
                },
                "json": {
                    "delivery_config_version": 1,
                    "confirm_changed_configuration": False,
                },
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

    response = client.get("/api/v1/admin/webhooks/catalog")

    assert response.status_code == 200
    assert response.json()["api_version"] == "2026-07-01"
    assert "audit-canary" not in response.text


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


def _delivery_client(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[TestClient, _FakeDeliveryService, AsyncMock, AsyncMock]:
    app = FastAPI()

    @app.middleware("http")
    async def request_id_middleware(request, call_next):
        request.state.request_id = REQUEST_ID
        return await call_next(request)

    async def auth_dependency() -> AuthPrincipal:
        return _principal()

    service = _FakeDeliveryService()
    mandatory_audit = AsyncMock()
    read_audit = AsyncMock()
    monkeypatch.setattr(admin_webhooks, "_require_platform_admin", lambda _value: None)
    monkeypatch.setattr(
        admin_webhooks,
        "emit_mandatory_webhook_delivery_audit",
        mandatory_audit,
        raising=False,
    )
    monkeypatch.setattr(admin_webhooks, "_emit_admin_audit_event", read_audit)
    app.dependency_overrides[admin_webhooks.get_auth_principal] = auth_dependency
    assert hasattr(admin_webhooks, "get_admin_webhook_delivery_service"), (
        "Task 10 delivery-service dependency is missing"
    )
    app.dependency_overrides[
        admin_webhooks.get_admin_webhook_delivery_service
    ] = lambda: service
    app.include_router(admin_webhooks.status_router, prefix="/api/v1/admin")
    app.include_router(admin_webhooks.canonical_router, prefix="/api/v1/admin")
    return TestClient(app, raise_server_exceptions=False), service, mandatory_audit, read_audit


@pytest.mark.unit
def test_delivery_history_route_is_sanitized_audited_and_never_cached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, service, _, read_audit = _delivery_client(monkeypatch)

    response = client.get("/api/v1/admin/webhooks/41/deliveries?limit=25&offset=7")

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-request-id"] == REQUEST_ID
    assert response.json()["total"] == 1
    assert response.json()["items"][0]["delivery"]["event_type"] == "webhook.test"
    assert response.json()["items"][0]["attempts"][0]["sequence"] == 1
    assert service.calls == [("list_delivery_history", (41, 25, 7))]
    assert read_audit.await_args.kwargs["metadata"] == {
        "outcome": "succeeded",
        "request_id": REQUEST_ID,
        "result_count": 1,
    }
    for forbidden in (
        "target_url",
        "secret_ciphertext",
        "request_headers",
        "request_body",
        "jobs",
        "token",
    ):
        assert forbidden not in response.text.lower()


@pytest.mark.unit
def test_test_and_redelivery_routes_use_typed_audit_and_exact_status_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, service, mandatory_audit, _ = _delivery_client(monkeypatch)
    headers = {
        "If-Match": '"admin-webhook-41-r1"',
        "Idempotency-Key": IDEMPOTENCY_KEY,
    }

    terminal = client.post(
        "/api/v1/admin/webhooks/41/test",
        headers=headers,
        json={"delivery_config_version": 1},
    )
    assert terminal.status_code == 200
    assert "retry-after" not in terminal.headers
    assert terminal.headers["cache-control"] == "no-store"
    assert terminal.headers["x-request-id"] == REQUEST_ID

    service.processing = True
    processing = client.post(
        "/api/v1/admin/webhooks/41/test",
        headers=headers,
        json={"delivery_config_version": 1},
    )
    assert processing.status_code == 202
    assert processing.headers["retry-after"] == "5"
    assert processing.json()["in_progress"] is True

    redelivery = client.post(
        f"/api/v1/admin/webhooks/41/deliveries/{SOURCE_DELIVERY_ID}/redeliver",
        headers=headers,
        json={
            "delivery_config_version": 1,
            "confirm_changed_configuration": False,
        },
    )
    assert redelivery.status_code == 202
    assert redelivery.headers["cache-control"] == "no-store"
    assert redelivery.headers["x-request-id"] == REQUEST_ID
    assert redelivery.json()["delivery"]["redelivery_of_id"] == SOURCE_DELIVERY_ID
    assert mandatory_audit.await_count == 3


@pytest.mark.unit
def test_delivery_routes_emit_once_when_service_fails_before_internal_audit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, service, mandatory_audit, _ = _delivery_client(monkeypatch)
    headers = {
        "If-Match": '"admin-webhook-41-r1"',
        "Idempotency-Key": IDEMPOTENCY_KEY,
    }
    service.fail_test_before_audit = True

    test_response = client.post(
        "/api/v1/admin/webhooks/41/test",
        headers=headers,
        json={"delivery_config_version": 1},
    )

    assert test_response.status_code == 412
    assert mandatory_audit.await_count == 1
    test_audit = mandatory_audit.await_args.args[0]
    assert test_audit.action == "admin_webhook.test"
    assert test_audit.outcome == "denied"
    assert test_audit.reason_code is WebhookErrorCode.PRECONDITION_FAILED

    service.fail_redelivery_before_audit = True
    redelivery_response = client.post(
        f"/api/v1/admin/webhooks/41/deliveries/{SOURCE_DELIVERY_ID}/redeliver",
        headers=headers,
        json={
            "delivery_config_version": 1,
            "confirm_changed_configuration": False,
        },
    )

    assert redelivery_response.status_code == 404
    assert mandatory_audit.await_count == 2
    redelivery_audit = mandatory_audit.await_args.args[0]
    assert redelivery_audit.action == "admin_webhook.redeliver"
    assert redelivery_audit.outcome == "denied"
    assert redelivery_audit.reason_code is WebhookErrorCode.NOT_FOUND


@pytest.mark.unit
def test_delivery_operation_validation_is_closed_and_does_not_audit_framework_failures(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    client, _, mandatory_audit, _ = _delivery_client(monkeypatch)
    canary = "task10-secret-canary"

    response = client.post(
        f"/api/v1/admin/webhooks/41/deliveries/{SOURCE_DELIVERY_ID}/redeliver",
        headers={
            "If-Match": canary,
            "Idempotency-Key": canary,
        },
        json={
            "delivery_config_version": "1",
            "confirm_changed_configuration": 1,
            "secret": canary,
        },
    )

    assert response.status_code == 422
    assert response.json()["error"] == {
        "code": "admin_webhook_validation_failed",
        "message": "Webhook request validation failed",
        "request_id": REQUEST_ID,
    }
    assert canary not in response.text
    assert canary not in str(response.headers)
    assert canary not in caplog.text
    mandatory_audit.assert_not_awaited()
