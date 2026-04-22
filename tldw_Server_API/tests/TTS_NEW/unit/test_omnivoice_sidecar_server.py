from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
import pytest


@pytest.fixture
def test_client():
    from tldw_Server_API.app.core.TTS.adapters.omnivoice_sidecar_server import create_app

    app = create_app(sidecar_token="test-sidecar-token")  # nosec B106
    with TestClient(app) as client:
        yield client


@pytest.mark.unit
def test_sidecar_requires_auth_header_for_health(test_client: TestClient):
    response = test_client.get("/health")

    assert response.status_code == 401  # nosec B101


@pytest.mark.unit
def test_sidecar_requires_auth_header_for_synthesize(test_client: TestClient):
    response = test_client.post("/v1/synthesize", json={"text": "hi", "mode": "auto"})

    assert response.status_code == 401  # nosec B101


@pytest.mark.unit
def test_sidecar_accepts_authorized_health_probe(test_client: TestClient):
    response = test_client.get("/health", headers={"X-TLDW-Sidecar-Token": "test-sidecar-token"})

    assert response.status_code == 200  # nosec B101
    payload = response.json()
    assert payload["status"] == "ok"  # nosec B101
    assert payload["ready"] is True  # nosec B101


@pytest.mark.unit
def test_sidecar_runtime_rejects_non_loopback_bind_host():
    from tldw_Server_API.app.core.TTS.adapters.omnivoice_sidecar_server import validate_loopback_host

    with pytest.raises(ValueError, match="loopback"):
        validate_loopback_host("0.0.0.0")  # nosec B104


@pytest.mark.unit
def test_sidecar_runtime_normalizes_localhost_to_loopback():
    from tldw_Server_API.app.core.TTS.adapters.omnivoice_sidecar_server import validate_loopback_host

    assert validate_loopback_host("localhost") == "127.0.0.1"  # nosec B101


@pytest.mark.unit
def test_sidecar_clone_mode_requires_reference_audio_path(test_client: TestClient):
    response = test_client.post(
        "/v1/synthesize",
        headers={"X-TLDW-Sidecar-Token": "test-sidecar-token"},
        json={"text": "hi", "mode": "clone", "sample_rate": 24000},
    )

    assert response.status_code == 422  # nosec B101


@pytest.mark.unit
def test_sidecar_clone_mode_rejects_directory_reference(test_client: TestClient, tmp_path: Path):
    response = test_client.post(
        "/v1/synthesize",
        headers={"X-TLDW-Sidecar-Token": "test-sidecar-token"},
        json={
            "text": "hi",
            "mode": "clone",
            "sample_rate": 24000,
            "reference_audio_path": str(tmp_path),
        },
    )

    assert response.status_code == 422  # nosec B101
