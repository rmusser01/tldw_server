from __future__ import annotations

import hmac

from tldw_Server_API.app.core.Metrics.metrics_manager import MetricsRegistry


def test_sensitive_label_hash_uses_configured_hmac_sha256_namespace(monkeypatch) -> None:
    hash_key = "metrics-label-hash-test-key"
    monkeypatch.setenv("METRICS_SENSITIVE_LABEL_HASH_KEY", hash_key)

    first = MetricsRegistry._hash_sensitive_label_value("user-123")
    second = MetricsRegistry._hash_sensitive_label_value("user-123")
    other = MetricsRegistry._hash_sensitive_label_value("user-456")

    assert first == second
    assert first != other
    assert first.startswith("u_")
    assert "user-123" not in first
    assert first == "u_" + hmac.digest(
        hash_key.encode("utf-8"),
        b"user-123",
        "sha256",
    ).hex()[:16]
