import pathlib

import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.main import app


@pytest.mark.unit
def test_metrics_text_contains_golden_subset(admin_user: object) -> None:
    """The metrics endpoint must still expose every name in the golden list.

    Args:
        admin_user: Requested for its side effect -- it installs the auth
            overrides. /api/v1/metrics/text requires authentication, and without
            it this returns 401 rather than the metrics body.

    Returns:
        None.
    """
    client = TestClient(app)
    r = client.get("/api/v1/metrics/text")
    assert r.status_code == 200
    body = r.text
    golden = (
        pathlib.Path(
            "tldw_Server_API/tests/Embeddings/golden/metrics_must_have.txt"
        )
        .read_text()
        .strip()
        .splitlines()
    )
    missing = [g for g in golden if g and g not in body]
    assert not missing, f"Missing required metrics in /metrics: {missing}"
