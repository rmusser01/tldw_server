from __future__ import annotations

import hmac

from tldw_Server_API.app.core.Metrics.metrics_manager import MetricsRegistry


def test_sensitive_label_hash_uses_hmac_sha256_namespace() -> None:
    first = MetricsRegistry._hash_sensitive_label_value("user-123")
    second = MetricsRegistry._hash_sensitive_label_value("user-123")
    other = MetricsRegistry._hash_sensitive_label_value("user-456")

    assert first == second
    assert first != other
    assert first.startswith("u_")
    assert "user-123" not in first
    assert first == "u_" + hmac.digest(
        MetricsRegistry._SENSITIVE_LABEL_HASH_KEY,
        b"user-123",
        "sha256",
    ).hex()[:16]
