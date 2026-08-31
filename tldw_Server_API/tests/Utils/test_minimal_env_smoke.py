"""Contract tests for the scrubbed minimal-environment startup probe."""

import pytest
from Helper_Scripts.ci import minimal_env_smoke


def test_public_liveness_accepts_exact_minimal_health_body() -> None:
    assert minimal_env_smoke._is_public_liveness_response(200, '{"status": "ok"}')


@pytest.mark.parametrize(
    ("status", "body"),
    [
        (503, '{"status": "ok"}'),
        (200, '{"status": "healthy"}'),
        (200, '{"status": "ok", "database": "healthy"}'),
        (200, "not-json"),
    ],
)
def test_public_liveness_rejects_noncanonical_response(status: int, body: str) -> None:
    assert not minimal_env_smoke._is_public_liveness_response(status, body)
