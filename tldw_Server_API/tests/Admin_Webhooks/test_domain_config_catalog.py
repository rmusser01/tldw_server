from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from tldw_Server_API.app.core import exceptions as core_exceptions
from tldw_Server_API.app.core.Admin_Webhooks import domain
from tldw_Server_API.app.core.Admin_Webhooks.catalog import (
    EVENT_API_VERSION,
    EVENT_CATALOG,
    normalize_subscriptions,
    validate_subscriptions,
)
from tldw_Server_API.app.core.Admin_Webhooks.config import (
    AdminWebhookMode,
    AdminWebhookSettings,
    WebhookRouteSelection,
    is_production_environment_mapping,
)
from tldw_Server_API.app.core.Admin_Webhooks.domain import (
    WebhookError,
    WebhookErrorCode,
    build_idempotency_scope,
    build_registration_etag,
    canonical_request_hash,
    idempotency_lookup_digest,
    normalize_request_id,
    parse_registration_etag,
    redact_target,
    validate_idempotency_key,
    validate_webhook_target,
)
from tldw_Server_API.app.core.DB_Management.admin_webhooks_repository import (
    AdminWebhookRepository,
)
from tldw_Server_API.app.core.Security.egress import URLPolicyResult

pytestmark = pytest.mark.unit


def test_settings_default_off_and_validate_bounds() -> None:
    settings = AdminWebhookSettings.from_environment({})

    assert settings.mode is AdminWebhookMode.OFF
    assert settings.route_selection is WebhookRouteSelection.CANONICAL
    assert settings.registration_limit == 100
    assert settings.active_limit == 25
    assert settings.allow_http_dev is False
    assert settings.idempotency_ttl_seconds == 86_400
    assert settings.rollback_window_days == 7


@pytest.mark.parametrize("value", ["", "enabled", "disabled"])
def test_settings_reject_invalid_mode(value: str) -> None:
    with pytest.raises(ValueError, match="TLDW_ADMIN_WEBHOOKS_MODE"):
        AdminWebhookSettings.from_environment({"TLDW_ADMIN_WEBHOOKS_MODE": value})


@pytest.mark.parametrize("value", ["yes", "0", "1", "disabled"])
def test_settings_reject_noncanonical_boolean(value: str) -> None:
    with pytest.raises(ValueError, match="TLDW_ADMIN_WEBHOOKS_LEGACY_COMPAT"):
        AdminWebhookSettings.from_environment({"TLDW_ADMIN_WEBHOOKS_LEGACY_COMPAT": value})


@pytest.mark.parametrize("mode", ["migrate", "on"])
def test_legacy_compatibility_requires_off_mode(mode: str) -> None:
    with pytest.raises(ValueError, match="requires canonical mode off"):
        AdminWebhookSettings.from_environment(
            {
                "TLDW_ADMIN_WEBHOOKS_MODE": mode,
                "TLDW_ADMIN_WEBHOOKS_LEGACY_COMPAT": "true",
            }
        )


def test_legacy_compatibility_selects_only_legacy_routes() -> None:
    settings = AdminWebhookSettings.from_environment(
        {"TLDW_ADMIN_WEBHOOKS_LEGACY_COMPAT": "true"}
    )

    assert settings.mode is AdminWebhookMode.OFF
    assert settings.route_selection is WebhookRouteSelection.LEGACY


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("TLDW_ADMIN_WEBHOOK_REGISTRATION_LIMIT", "0"),
        ("TLDW_ADMIN_WEBHOOK_REGISTRATION_LIMIT", "1001"),
        ("TLDW_ADMIN_WEBHOOK_ACTIVE_LIMIT", "-1"),
        ("TLDW_ADMIN_WEBHOOK_ACTIVE_LIMIT", "1001"),
        ("TLDW_ADMIN_WEBHOOK_ROLLBACK_WINDOW_DAYS", "0"),
        ("TLDW_ADMIN_WEBHOOK_ROLLBACK_WINDOW_DAYS", "31"),
    ],
)
def test_settings_reject_out_of_range_integers(name: str, value: str) -> None:
    with pytest.raises(ValueError, match=name):
        AdminWebhookSettings.from_environment({name: value})


def test_settings_reject_active_limit_above_registration_limit() -> None:
    with pytest.raises(ValueError, match="cannot exceed"):
        AdminWebhookSettings.from_environment(
            {
                "TLDW_ADMIN_WEBHOOK_REGISTRATION_LIMIT": "3",
                "TLDW_ADMIN_WEBHOOK_ACTIVE_LIMIT": "4",
            }
        )


@pytest.mark.parametrize(
    "environ",
    [
        {"tldw_production": "y"},
        {"TLDW_PRODUCTION": "true"},
        {"ENV": "prod"},
        {"APP_ENV": "production"},
        {"TLDW_ENV": "PROD"},
        {"ENVIRONMENT": "Production"},
    ],
)
def test_production_environment_mapping_matches_supported_markers(
    environ: dict[str, str],
) -> None:
    assert is_production_environment_mapping(environ) is True


def test_http_override_is_rejected_in_production() -> None:
    with pytest.raises(ValueError, match="forbidden in production"):
        AdminWebhookSettings.from_environment(
            {
                "ENVIRONMENT": "production",
                "TLDW_ADMIN_WEBHOOKS_ALLOW_HTTP_DEV": "true",
            }
        )


def test_settings_are_immutable() -> None:
    settings = AdminWebhookSettings.from_environment({})

    with pytest.raises(FrozenInstanceError):
        settings.mode = AdminWebhookMode.ON  # type: ignore[misc]


def test_catalog_is_explicit_and_rejects_wildcard() -> None:
    assert EVENT_API_VERSION == "2026-07-01"
    assert tuple(item.event_type for item in EVENT_CATALOG) == (
        "user.created",
        "user.deleted",
        "incident.created",
        "incident.updated",
        "incident.resolved",
        "incident.notify",
    )
    assert all(item.description for item in EVENT_CATALOG)

    with pytest.raises(WebhookError) as exc_info:
        validate_subscriptions(["*"])
    assert exc_info.value.code is WebhookErrorCode.EVENT_UNSUPPORTED


@pytest.mark.parametrize(
    "subscriptions",
    [[], ["user.created", "user.created"], ["webhook.test"]],
)
def test_catalog_rejects_empty_duplicate_and_reserved_subscriptions(
    subscriptions: list[str],
) -> None:
    with pytest.raises(WebhookError):
        validate_subscriptions(subscriptions)


def test_catalog_normalizes_sets_to_catalog_order() -> None:
    first = normalize_subscriptions(
        ["incident.notify", "user.created", "incident.updated"]
    )
    second = normalize_subscriptions(
        ["incident.updated", "incident.notify", "user.created"]
    )

    assert first == second == (
        "user.created",
        "incident.updated",
        "incident.notify",
    )


def test_etag_is_strong_and_round_trips() -> None:
    value = build_registration_etag(webhook_id=41, revision=7)

    assert value == '"admin-webhook-41-r7"'
    assert parse_registration_etag(value, expected_webhook_id=41) == 7


@pytest.mark.parametrize(
    "value",
    [
        'W/"admin-webhook-41-r7"',
        '"admin-webhook-0-r7"',
        '"admin-webhook-41-r0"',
        '"admin-webhook-41-r7", "admin-webhook-41-r8"',
        "*",
        "admin-webhook-41-r7",
        "",
    ],
)
def test_etag_parser_rejects_weak_wildcard_and_malformed_values(value: str) -> None:
    with pytest.raises(WebhookError) as exc_info:
        parse_registration_etag(value, expected_webhook_id=41)
    assert exc_info.value.code is WebhookErrorCode.PRECONDITION_FAILED


def test_etag_parser_rejects_another_registration() -> None:
    with pytest.raises(WebhookError) as exc_info:
        parse_registration_etag(
            '"admin-webhook-42-r7"',
            expected_webhook_id=41,
        )
    assert exc_info.value.code is WebhookErrorCode.PRECONDITION_FAILED


@pytest.mark.parametrize(
    "value",
    ["short", "has space in it", "x" * 256, "line\nbreak", "unicode-é"],
)
def test_idempotency_key_rejects_weak_or_malformed_values(value: str) -> None:
    with pytest.raises(WebhookError) as exc_info:
        validate_idempotency_key(value)
    assert exc_info.value.code is WebhookErrorCode.IDEMPOTENCY_KEY_INVALID


def test_idempotency_digests_are_deterministic_and_domain_separated() -> None:
    key = "0123456789abcdef0123456789abcdef"
    scope = build_idempotency_scope(
        actor_id=9,
        operation="rotate_secret",
        route="/api/v1/admin/webhooks/{webhook_id}/rotate-secret",
        webhook_id=41,
    )
    body = {"url": "https://receiver.example/private?token=canary"}

    lookup = idempotency_lookup_digest(key, scope)
    fingerprint = canonical_request_hash(
        key,
        scope=scope,
        body=body,
        conditional_version=7,
    )

    assert lookup == idempotency_lookup_digest(key, scope)
    assert lookup.startswith("sha256:")
    assert fingerprint.startswith("hmac-sha256:")
    assert lookup != fingerprint
    assert "receiver.example" not in lookup
    assert "receiver.example" not in fingerprint
    assert "canary" not in lookup
    assert "canary" not in fingerprint


def test_idempotency_fingerprint_separates_key_body_scope_and_condition() -> None:
    key = "0123456789abcdef0123456789abcdef"
    scope = build_idempotency_scope(
        actor_id=9,
        operation="create",
        route="/api/v1/admin/webhooks/",
    )
    baseline = canonical_request_hash(
        key,
        scope=scope,
        body={"description": "first"},
        conditional_version=None,
    )

    assert canonical_request_hash(
        "fedcba9876543210fedcba9876543210",
        scope=scope,
        body={"description": "first"},
        conditional_version=None,
    ) != baseline
    assert canonical_request_hash(
        key,
        scope=scope,
        body={"description": "second"},
        conditional_version=None,
    ) != baseline
    assert canonical_request_hash(
        key,
        scope=build_idempotency_scope(
            actor_id=10,
            operation="create",
            route="/api/v1/admin/webhooks/",
        ),
        body={"description": "first"},
        conditional_version=None,
    ) != baseline
    assert canonical_request_hash(
        key,
        scope=scope,
        body={"description": "first"},
        conditional_version=1,
    ) != baseline


def test_request_id_accepts_only_bounded_safe_values() -> None:
    generated = "00000000-0000-4000-8000-000000000001"

    assert normalize_request_id("req_1:alpha") == "req_1:alpha"
    assert normalize_request_id(None, generator=lambda: generated) == generated
    assert normalize_request_id("bad value", generator=lambda: generated) == generated
    assert normalize_request_id("x" * 129, generator=lambda: generated) == generated


@pytest.mark.parametrize(
    "url",
    [
        "",
        "receiver.example/hook",
        "ftp://receiver.example/hook",
        "https://user:pass@receiver.example/hook",
        "https://receiver.example/hook#fragment",
        "https://receiver.example\\hook",
        "https://receiver.example/line\nbreak",
        "https://receiver.example:bad/hook",
        "https://[::1/hook",
        "https://.example/hook",
        "https://receiver.example/" + ("x" * 2_050),
    ],
)
def test_target_validation_rejects_unsafe_or_malformed_urls(
    url: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        domain,
        "evaluate_platform_webhook_url_policy",
        lambda _url: URLPolicyResult(True),
    )

    with pytest.raises(WebhookError) as exc_info:
        validate_webhook_target(url, allow_http_dev=False)
    assert exc_info.value.code is WebhookErrorCode.VALIDATION_FAILED


def test_target_validation_requires_https_without_dev_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        domain,
        "evaluate_platform_webhook_url_policy",
        lambda _url: URLPolicyResult(True),
    )

    with pytest.raises(WebhookError):
        validate_webhook_target("http://receiver.example/hook", allow_http_dev=False)

    target = validate_webhook_target(
        "http://receiver.example/hook",
        allow_http_dev=True,
    )
    assert target.hostname == "receiver.example"


def test_target_validation_delegates_and_returns_only_redacted_display(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[str] = []

    def _allow(url: str) -> URLPolicyResult:
        observed.append(url)
        return URLPolicyResult(True, resolved_ips=("93.184.216.34",))

    monkeypatch.setattr(domain, "evaluate_platform_webhook_url_policy", _allow)
    raw = "https://BÜCHER.example:8443/private/hook?token=secret"

    target = validate_webhook_target(raw, allow_http_dev=False)

    assert observed == [raw]
    assert target.url == raw
    assert target.hostname == "xn--bcher-kva.example"
    assert target.target_display == "https://xn--bcher-kva.example:8443"
    assert "private" not in target.target_display
    assert "secret" not in target.target_display


def test_target_validation_fails_closed_when_central_policy_denies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        domain,
        "evaluate_platform_webhook_url_policy",
        lambda _url: URLPolicyResult(False, "sensitive policy reason", reason_code="host_denied"),
    )

    with pytest.raises(WebhookError) as exc_info:
        validate_webhook_target("https://receiver.example/hook", allow_http_dev=False)
    assert exc_info.value.code is WebhookErrorCode.TARGET_REJECTED
    assert "sensitive policy reason" not in str(exc_info.value)


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://example.com/private?x=1", "https://example.com"),
        ("https://example.com:443/private", "https://example.com"),
        ("https://example.com:8443/private", "https://example.com:8443"),
        ("http://example.com:80/private", "http://example.com"),
        ("http://example.com:8080/private", "http://example.com:8080"),
    ],
)
def test_redact_target_returns_origin_without_default_port(
    url: str,
    expected: str,
) -> None:
    assert redact_target(url) == expected


def test_webhook_error_is_centralized_in_core_exceptions() -> None:
    assert core_exceptions.WebhookError is WebhookError


def test_repository_implementation_lives_in_db_management() -> None:
    assert AdminWebhookRepository.__module__ == (
        "tldw_Server_API.app.core.DB_Management.admin_webhooks_repository"
    )
