from unittest.mock import MagicMock

import pytest

from tldw_Server_API.app.core.External_Sources import connectors_service as svc
from tldw_Server_API.app.core.Security import crypto


pytestmark = pytest.mark.unit


def _assert_sanitized_debug(fake_logger: MagicMock, expected_message: str) -> None:
    fake_logger.debug.assert_called_once_with(expected_message)
    rendered = repr(fake_logger.debug.call_args)
    assert "connectors crypto exploded" not in rendered
    assert "/private/connectors.db" not in rendered


def test_protect_oauth_state_metadata_sanitizes_encryption_fallback_log(monkeypatch):
    fake_logger = MagicMock()

    def _fail_encrypt(_metadata):
        raise RuntimeError("connectors crypto exploded /private/connectors.db")

    monkeypatch.setattr(svc, "logger", fake_logger)
    monkeypatch.setattr(crypto, "encrypt_json_blob", _fail_encrypt)

    result = svc._protect_oauth_state_metadata({"nonce": "state-token"})

    assert result == {"nonce": "state-token"}
    _assert_sanitized_debug(
        fake_logger,
        "Failed to encrypt oauth state metadata",
    )


def test_unprotect_oauth_state_metadata_sanitizes_decryption_fallback_log(monkeypatch):
    fake_logger = MagicMock()

    def _fail_decrypt(_metadata):
        raise RuntimeError("connectors crypto exploded /private/connectors.db")

    monkeypatch.setattr(svc, "logger", fake_logger)
    monkeypatch.setattr(crypto, "decrypt_json_blob", _fail_decrypt)

    result = svc._unprotect_oauth_state_metadata({"_enc": "aesgcm:v1", "ct": "encrypted"})

    assert result == {}
    _assert_sanitized_debug(
        fake_logger,
        "Failed to decrypt oauth state metadata",
    )
