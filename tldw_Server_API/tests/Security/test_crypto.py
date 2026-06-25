import base64

import pytest

from tldw_Server_API.app.core.Security import crypto


pytestmark = pytest.mark.unit


class _CapturingLogger:
    def __init__(self):
        self.records = []

    def error(self, message, *args, **kwargs):
        self.records.append(("error", message, args, dict(kwargs)))


@pytest.mark.skipif(not crypto._HAS_CRYPTO, reason="Crypto backend not available")
def test_decrypt_json_blob_invalid_base64_returns_none(monkeypatch):
    key = base64.b64encode(b"a" * 32).decode("ascii")
    monkeypatch.setenv("WORKFLOWS_ARTIFACT_ENC_KEY", key)
    envelope = {"_enc": "aesgcm:v1", "nonce": "abc", "ct": "abc", "tag": "abc"}
    assert crypto.decrypt_json_blob(envelope) is None


@pytest.mark.skipif(not crypto._HAS_CRYPTO, reason="Crypto backend not available")
def test_decrypt_json_blob_with_key_invalid_base64_returns_none():
    key = base64.b64encode(b"b" * 32).decode("ascii")
    envelope = {"_enc": "aesgcm:v1", "nonce": "abc", "ct": "abc", "tag": "abc"}
    assert crypto.decrypt_json_blob_with_key(envelope, key) is None


@pytest.mark.skipif(not crypto._HAS_CRYPTO, reason="Crypto backend not available")
def test_encrypt_json_blob_rejects_invalid_env_key(monkeypatch):
    logger = _CapturingLogger()
    monkeypatch.setattr(crypto, "logger", logger)
    monkeypatch.setenv("WORKFLOWS_ARTIFACT_ENC_KEY", "not-a-valid-base64-key")

    assert crypto.encrypt_json_blob({"secret": "value"}) is None
    assert logger.records
    assert "WORKFLOWS_ARTIFACT_ENC_KEY is set but invalid" in logger.records[0][1]


@pytest.mark.skipif(not crypto._HAS_CRYPTO, reason="Crypto backend not available")
def test_encrypt_json_blob_with_key_rejects_wrong_length_key():
    short_key = base64.b64encode(b"short").decode("ascii")

    assert crypto.encrypt_json_blob_with_key({"secret": "value"}, short_key) is None
