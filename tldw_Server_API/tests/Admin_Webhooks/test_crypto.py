from __future__ import annotations

import base64
import json
import os
from dataclasses import replace

import pytest

from tldw_Server_API.app.core.Admin_Webhooks.crypto import (
    MIGRATION_DOMAIN_DATABASE_RECORD,
    MIGRATION_DOMAIN_DATABASE_TABLE,
    MIGRATION_DOMAIN_SYSTEM_OPS_RECORD,
    MIGRATION_DOMAIN_SYSTEM_OPS_SUBTREE,
    ProtectedValue,
    WebhookKeyError,
    WebhookKeyErrorCode,
    WebhookKeyLoadCode,
    WebhookKeyRing,
    load_webhook_key_ring,
)


def _key(byte: int) -> str:
    return base64.b64encode(bytes([byte]) * 32).decode("ascii")


def _environment(*, primary: str = "primary") -> dict[str, str]:
    return {
        "TLDW_ADMIN_WEBHOOK_KEYS_JSON": json.dumps(
            {"primary": _key(1), "previous": _key(2)}
        ),
        "TLDW_ADMIN_WEBHOOK_PRIMARY_KEY_ID": primary,
    }


@pytest.fixture
def key_ring() -> WebhookKeyRing:
    return WebhookKeyRing.from_environment(_environment())


@pytest.mark.unit
def test_context_prevents_cross_row_substitution(key_ring: WebhookKeyRing) -> None:
    protected = key_ring.encrypt_text(
        purpose="registration.secret",
        identity={"registration_id": 7, "secret_version": 1},
        plaintext="whsec_" + "a" * 64,
    )

    with pytest.raises(
        WebhookKeyError,
        match="admin_webhook_envelope_context_mismatch",
    ):
        key_ring.decrypt_text(
            purpose="registration.secret",
            identity={"registration_id": 8, "secret_version": 1},
            protected=protected,
        )


@pytest.mark.parametrize(
    ("purpose", "identity"),
    [
        (
            "registration.target",
            {"registration_id": 7, "secret_version": 1},
        ),
        (
            "registration.secret",
            {"registration_id": 7, "secret_version": 2},
        ),
    ],
)
@pytest.mark.unit
def test_context_prevents_cross_purpose_and_version_substitution(
    key_ring: WebhookKeyRing,
    purpose: str,
    identity: dict[str, int],
) -> None:
    protected = key_ring.encrypt_text(
        purpose="registration.secret",
        identity={"registration_id": 7, "secret_version": 1},
        plaintext="whsec_" + "a" * 64,
    )

    with pytest.raises(WebhookKeyError) as exc_info:
        key_ring.decrypt_text(
            purpose=purpose,
            identity=identity,
            protected=protected,
        )
    assert exc_info.value.code is WebhookKeyErrorCode.CONTEXT_MISMATCH


@pytest.mark.unit
def test_runtime_key_ring_ignores_unrelated_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "BYOK_ENCRYPTION_KEY",
        "SESSION_ENCRYPTION_KEY",
        "JWT_SECRET_KEY",
        "SINGLE_USER_API_KEY",
        "API_KEY",
    ):
        monkeypatch.setenv(name, _key(9))
    monkeypatch.delenv("TLDW_ADMIN_WEBHOOK_KEYS_JSON", raising=False)
    monkeypatch.delenv("TLDW_ADMIN_WEBHOOK_PRIMARY_KEY_ID", raising=False)

    with pytest.raises(WebhookKeyError) as exc_info:
        WebhookKeyRing.from_environment(os.environ)
    assert exc_info.value.code is WebhookKeyErrorCode.KEY_UNAVAILABLE


@pytest.mark.parametrize(
    "raw_keys",
    [
        "{",
        "null",
        "[]",
        f'[["primary", "{_key(1)}"]]',
        '"not-an-object"',
        '{"primary": 1}',
        '{"": "' + _key(1) + '"}',
        '{"bad id": "' + _key(1) + '"}',
        '{"' + ("x" * 65) + '": "' + _key(1) + '"}',
        '{"primary": "not-base64"}',
        '{"primary": "' + base64.b64encode(b"short").decode("ascii") + '"}',
        '{"primary": "' + base64.b64encode(b"x" * 33).decode("ascii") + '"}',
        '{"primary": "' + _key(1) + '=="}',
    ],
)
@pytest.mark.unit
def test_key_ring_rejects_malformed_or_invalid_key_configuration(
    raw_keys: str,
) -> None:
    with pytest.raises(WebhookKeyError) as exc_info:
        WebhookKeyRing.from_environment(
            {
                "TLDW_ADMIN_WEBHOOK_KEYS_JSON": raw_keys,
                "TLDW_ADMIN_WEBHOOK_PRIMARY_KEY_ID": "primary",
            }
        )
    assert exc_info.value.code is WebhookKeyErrorCode.CONFIGURATION_INVALID
    assert raw_keys not in str(exc_info.value)


@pytest.mark.unit
def test_key_ring_rejects_duplicate_ids_before_object_construction() -> None:
    raw = '{"primary":"' + _key(1) + '","primary":"' + _key(2) + '"}'

    with pytest.raises(WebhookKeyError) as exc_info:
        WebhookKeyRing.from_environment(
            {
                "TLDW_ADMIN_WEBHOOK_KEYS_JSON": raw,
                "TLDW_ADMIN_WEBHOOK_PRIMARY_KEY_ID": "primary",
            }
        )
    assert exc_info.value.code is WebhookKeyErrorCode.CONFIGURATION_INVALID


@pytest.mark.parametrize(
    "environ",
    [
        {"TLDW_ADMIN_WEBHOOK_KEYS_JSON": "{}"},
        {
            "TLDW_ADMIN_WEBHOOK_KEYS_JSON": '{"primary":"' + _key(1) + '"}',
            "TLDW_ADMIN_WEBHOOK_PRIMARY_KEY_ID": "",
        },
    ],
)
@pytest.mark.unit
def test_empty_ring_or_primary_is_unavailable(environ: dict[str, str]) -> None:
    with pytest.raises(WebhookKeyError) as exc_info:
        WebhookKeyRing.from_environment(environ)
    assert exc_info.value.code is WebhookKeyErrorCode.KEY_UNAVAILABLE


@pytest.mark.unit
def test_primary_must_name_a_configured_key() -> None:
    with pytest.raises(WebhookKeyError) as exc_info:
        WebhookKeyRing.from_environment(_environment(primary="missing"))
    assert exc_info.value.code is WebhookKeyErrorCode.CONFIGURATION_INVALID


@pytest.mark.parametrize(
    "environ",
    [
        {},
        {
            "TLDW_ADMIN_WEBHOOK_KEYS_JSON": "{",
            "TLDW_ADMIN_WEBHOOK_PRIMARY_KEY_ID": "canary-primary",
        },
        {
            "TLDW_ADMIN_WEBHOOK_KEYS_JSON": '{"canary-id":"canary-value"}',
            "TLDW_ADMIN_WEBHOOK_PRIMARY_KEY_ID": "canary-id",
        },
    ],
)
@pytest.mark.unit
def test_runtime_loader_returns_only_closed_redacted_state(
    environ: dict[str, str],
) -> None:
    result = load_webhook_key_ring(environ)

    assert result.ring is None
    assert result.code in {
        WebhookKeyLoadCode.KEY_UNAVAILABLE,
        WebhookKeyLoadCode.CONFIGURATION_INVALID,
    }
    rendered = repr(result)
    assert "canary-primary" not in rendered
    assert "canary-id" not in rendered
    assert "canary-value" not in rendered

    with pytest.raises(WebhookKeyError) as exc_info:
        result.require_ring()
    assert exc_info.value.code.value == result.code.value


@pytest.mark.unit
def test_runtime_loader_returns_available_ring() -> None:
    result = load_webhook_key_ring(_environment())

    assert result.code is WebhookKeyLoadCode.AVAILABLE
    assert result.require_ring() is result.ring


@pytest.mark.unit
def test_runtime_loader_defaults_to_process_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name, value in _environment().items():
        monkeypatch.setenv(name, value)

    result = load_webhook_key_ring()

    assert result.code is WebhookKeyLoadCode.AVAILABLE
    assert result.require_ring().primary_id == "primary"


@pytest.mark.unit
def test_primary_only_writes_and_previous_key_reads() -> None:
    previous_primary = WebhookKeyRing.from_environment(_environment(primary="previous"))
    old_value = previous_primary.encrypt_text(
        purpose="registration.target",
        identity={"registration_id": 4, "target_version": 1},
        plaintext="https://receiver.example/private",
    )
    rotated_ring = WebhookKeyRing.from_environment(_environment(primary="primary"))

    assert old_value.key_id == "previous"
    assert rotated_ring.decrypt_text(
        purpose="registration.target",
        identity={"registration_id": 4, "target_version": 1},
        protected=old_value,
    ) == "https://receiver.example/private"
    assert rotated_ring.encrypt_text(
        purpose="registration.target",
        identity={"registration_id": 4, "target_version": 2},
        plaintext="https://receiver.example/new",
    ).key_id == "primary"


@pytest.mark.unit
def test_utf8_and_arbitrary_bytes_round_trip(key_ring: WebhookKeyRing) -> None:
    text = "receiver-☃-é"
    arbitrary = bytes(range(256))

    protected_text = key_ring.encrypt_text(
        purpose="registration.description",
        identity={"registration_id": 3, "revision": 1},
        plaintext=text,
    )
    protected_bytes = key_ring.encrypt_bytes(
        purpose="event.body",
        identity={"event_id": "evt-3", "api_version": "2026-07-01"},
        plaintext=arbitrary,
    )

    assert key_ring.decrypt_text(
        purpose="registration.description",
        identity={"registration_id": 3, "revision": 1},
        protected=protected_text,
    ) == text
    assert key_ring.decrypt_bytes(
        purpose="event.body",
        identity={"event_id": "evt-3", "api_version": "2026-07-01"},
        protected=protected_bytes,
    ) == arbitrary


@pytest.mark.unit
def test_event_body_limit_and_context(key_ring: WebhookKeyRing) -> None:
    accepted = key_ring.encrypt_event_body(
        event_id="evt-1",
        api_version="2026-07-01",
        body=b"x" * 65_536,
    )
    assert key_ring.decrypt_event_body(
        event_id="evt-1",
        api_version="2026-07-01",
        protected=accepted,
    ) == b"x" * 65_536

    with pytest.raises(WebhookKeyError) as exc_info:
        key_ring.encrypt_event_body(
            event_id="evt-2",
            api_version="2026-07-01",
            body=b"x" * 65_537,
        )
    assert exc_info.value.code is WebhookKeyErrorCode.EVENT_BODY_TOO_LARGE

    for event_id, api_version in (
        ("evt-other", "2026-07-01"),
        ("evt-1", "2026-08-01"),
    ):
        with pytest.raises(WebhookKeyError) as substitution:
            key_ring.decrypt_event_body(
                event_id=event_id,
                api_version=api_version,
                protected=accepted,
            )
        assert substitution.value.code is WebhookKeyErrorCode.CONTEXT_MISMATCH


@pytest.mark.unit
def test_event_body_decryption_rechecks_size_bound(key_ring: WebhookKeyRing) -> None:
    oversized = key_ring.encrypt_bytes(
        purpose="event.body",
        identity={"event_id": "evt-oversized", "api_version": "2026-07-01"},
        plaintext=b"x" * 65_537,
    )

    with pytest.raises(WebhookKeyError) as exc_info:
        key_ring.decrypt_event_body(
            event_id="evt-oversized",
            api_version="2026-07-01",
            protected=oversized,
        )
    assert exc_info.value.code is WebhookKeyErrorCode.EVENT_BODY_TOO_LARGE


@pytest.mark.unit
def test_noncanonical_outer_base64_is_rejected(key_ring: WebhookKeyRing) -> None:
    protected = key_ring.encrypt_text(
        purpose="registration.secret",
        identity={"registration_id": 1, "secret_version": 1},
        plaintext="whsec_" + "a" * 64,
    )
    envelope = json.loads(protected.ciphertext_json)
    envelope["nonce"] += "!"
    noncanonical = replace(
        protected,
        ciphertext_json=json.dumps(envelope, sort_keys=True),
    )

    with pytest.raises(WebhookKeyError) as exc_info:
        key_ring.decrypt_text(
            purpose="registration.secret",
            identity={"registration_id": 1, "secret_version": 1},
            protected=noncanonical,
        )
    assert exc_info.value.code is WebhookKeyErrorCode.DECRYPTION_FAILED


@pytest.mark.unit
def test_unknown_key_tamper_and_can_decrypt_fail_closed(
    key_ring: WebhookKeyRing,
) -> None:
    protected = key_ring.encrypt_text(
        purpose="registration.secret",
        identity={"registration_id": 1, "secret_version": 1},
        plaintext="whsec_" + "a" * 64,
    )
    unknown = replace(protected, key_id="missing")
    envelope = json.loads(protected.ciphertext_json)
    envelope["ct"] = base64.b64encode(b"tampered").decode("ascii")
    tampered = ProtectedValue(
        ciphertext_json=json.dumps(envelope),
        key_id=protected.key_id,
    )

    with pytest.raises(WebhookKeyError) as unknown_error:
        key_ring.decrypt_text(
            purpose="registration.secret",
            identity={"registration_id": 1, "secret_version": 1},
            protected=unknown,
        )
    assert unknown_error.value.code is WebhookKeyErrorCode.UNKNOWN_KEY
    assert key_ring.can_decrypt(
        purpose="registration.secret",
        identity={"registration_id": 1, "secret_version": 1},
        protected=unknown,
    ) is False

    with pytest.raises(WebhookKeyError) as tamper_error:
        key_ring.decrypt_text(
            purpose="registration.secret",
            identity={"registration_id": 1, "secret_version": 1},
            protected=tampered,
        )
    assert tamper_error.value.code is WebhookKeyErrorCode.DECRYPTION_FAILED
    assert "tampered" not in str(tamper_error.value)


@pytest.mark.unit
def test_reencrypt_to_configured_target_does_not_change_primary(
    key_ring: WebhookKeyRing,
) -> None:
    original = key_ring.encrypt_text(
        purpose="registration.target",
        identity={"registration_id": 5, "target_version": 1},
        plaintext="https://receiver.example/private",
    )

    rewritten = key_ring.reencrypt_to_key(
        original,
        purpose="registration.target",
        identity={"registration_id": 5, "target_version": 1},
        target_key_id="previous",
    )

    assert original.key_id == "primary"
    assert rewritten.key_id == "previous"
    assert key_ring.primary_id == "primary"
    assert key_ring.decrypt_text(
        purpose="registration.target",
        identity={"registration_id": 5, "target_version": 1},
        protected=rewritten,
    ) == "https://receiver.example/private"


@pytest.mark.unit
def test_migration_fingerprints_are_keyed_deterministic_and_domain_separated(
    key_ring: WebhookKeyRing,
) -> None:
    domains = (
        MIGRATION_DOMAIN_SYSTEM_OPS_SUBTREE,
        MIGRATION_DOMAIN_SYSTEM_OPS_RECORD,
        MIGRATION_DOMAIN_DATABASE_TABLE,
        MIGRATION_DOMAIN_DATABASE_RECORD,
    )
    values = {
        key_ring.fingerprint_migration_source(domain, b'{"secret":"weak"}')
        for domain in domains
    }

    assert len(values) == len(domains)
    key_id, digest = key_ring.fingerprint_migration_source(
        MIGRATION_DOMAIN_SYSTEM_OPS_SUBTREE,
        b'{"secret":"weak"}',
    )
    assert key_id == "primary"
    assert digest.startswith("hmac-sha256:")
    assert digest == key_ring.fingerprint_migration_source(
        MIGRATION_DOMAIN_SYSTEM_OPS_SUBTREE,
        b'{"secret":"weak"}',
    )[1]
    assert "weak" not in digest
    assert digest != key_ring.fingerprint_migration_source(
        MIGRATION_DOMAIN_SYSTEM_OPS_SUBTREE,
        b'{"secret":"changed"}',
    )[1]


@pytest.mark.unit
def test_migration_fingerprint_rejects_caller_controlled_domain(
    key_ring: WebhookKeyRing,
) -> None:
    with pytest.raises(WebhookKeyError) as exc_info:
        key_ring.fingerprint_migration_source("caller-controlled", b"source")
    assert exc_info.value.code is WebhookKeyErrorCode.FINGERPRINT_DOMAIN_INVALID
