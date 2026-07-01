from __future__ import annotations

import hmac

import pytest

from tldw_Server_API.app.core.Metrics.metrics_manager import MetricsRegistry

pytestmark = pytest.mark.unit


def test_sensitive_label_hash_uses_configured_hmac_sha256_namespace(monkeypatch) -> None:
    hash_key = "metrics-label-hash-test-key"
    monkeypatch.setenv("METRICS_SENSITIVE_LABEL_HASH_KEY", hash_key)

    first = MetricsRegistry.normalize_labels({"user_id": "user-123"})["user_hash"]
    second = MetricsRegistry.normalize_labels({"user_id": "user-123"})["user_hash"]
    other = MetricsRegistry.normalize_labels({"user_id": "user-456"})["user_hash"]

    assert first == second
    assert first != other
    assert first.startswith("u_")
    assert "user-123" not in first
    assert first == "u_" + hmac.digest(
        hash_key.encode("utf-8"),
        b"user-123",
        "sha256",
    ).hex()[:16]
