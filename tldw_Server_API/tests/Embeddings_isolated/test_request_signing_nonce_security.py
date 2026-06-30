from datetime import datetime, timedelta

import pytest

from tldw_Server_API.app.core.Embeddings import request_signing as request_signing_module


@pytest.mark.unit
def test_nonce_manager_honors_ttl_even_without_cleanup_tick():
    manager = request_signing_module.NonceManager(ttl_seconds=1)
    expired_nonce = "expired-nonce"
    manager.used_nonces[expired_nonce] = datetime.utcnow() - timedelta(seconds=10)
    manager.last_cleanup = datetime.utcnow()

    # Expected behavior: expired nonce is treated as expired and can be reused.
    assert manager.is_valid_nonce(expired_nonce) is True


@pytest.mark.unit
def test_invalid_signature_does_not_consume_nonce(monkeypatch):
    monkeypatch.setattr(
        request_signing_module,
        "log_security_violation",
        lambda **_kwargs: None,
    )

    monkeypatch.setenv("EMBEDDINGS_REQUEST_SIGNING_SECRET", "k" * 32)
    monkeypatch.setattr(request_signing_module, "_request_signer", None, raising=False)
    monkeypatch.setattr(
        request_signing_module,
        "_nonce_manager",
        request_signing_module.NonceManager(ttl_seconds=300),
        raising=False,
    )

    user_id = "1"
    payload = {"text": "hello"}
    valid_headers = request_signing_module.create_signed_headers(user_id, payload)
    invalid_headers = dict(valid_headers)
    invalid_headers["X-Signature"] = "invalid-signature"

    first_ok, _first_error = request_signing_module.validate_signed_request(
        user_id=user_id,
        request_data=payload,
        headers=invalid_headers,
    )
    assert first_ok is False

    second_ok, second_error = request_signing_module.validate_signed_request(
        user_id=user_id,
        request_data=payload,
        headers=valid_headers,
    )
    assert second_ok is True
    assert second_error is None


@pytest.mark.unit
def test_valid_signature_replay_is_rejected(monkeypatch):
    monkeypatch.setattr(
        request_signing_module,
        "log_security_violation",
        lambda **_kwargs: None,
    )

    monkeypatch.setenv("EMBEDDINGS_REQUEST_SIGNING_SECRET", "k" * 32)
    monkeypatch.setattr(request_signing_module, "_request_signer", None, raising=False)
    monkeypatch.setattr(
        request_signing_module,
        "_nonce_manager",
        request_signing_module.NonceManager(ttl_seconds=300),
        raising=False,
    )

    user_id = "1"
    payload = {"text": "hello"}
    headers = request_signing_module.create_signed_headers(user_id, payload)

    first_ok, first_error = request_signing_module.validate_signed_request(
        user_id=user_id,
        request_data=payload,
        headers=headers,
    )
    assert first_ok is True
    assert first_error is None

    second_ok, second_error = request_signing_module.validate_signed_request(
        user_id=user_id,
        request_data=payload,
        headers=headers,
    )
    assert second_ok is False
    assert second_error == "Invalid or reused nonce"
