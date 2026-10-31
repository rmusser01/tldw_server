import pytest
from fastapi.testclient import TestClient


@pytest.mark.unit
def test_metrics_exposes_new_histograms(
    admin_user: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Importing BaseWorker must register its histograms on the default registry.

    Args:
        admin_user: Requested for its side effect -- /api/v1/metrics/text
            requires authentication and returns 401 without these overrides.
        monkeypatch: Used to trim startup work.

    Returns:
        None.
    """


     # Reduce startup work
    monkeypatch.setenv("TEST_MODE", "true")

    # Import BaseWorker to register histograms in the default REGISTRY
    from tldw_Server_API.app.core.Embeddings.workers.base_worker import BaseWorker  # noqa: F401
    from tldw_Server_API.app.main import app

    client = TestClient(app)
    r = client.get("/api/v1/metrics/text")
    assert r.status_code == 200
    body = r.text
    # Histograms should produce *_bucket names
    assert "embedding_stage_batch_size_bucket" in body
    assert "embedding_stage_payload_bytes_bucket" in body
