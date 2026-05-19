from fastapi.testclient import TestClient

from tldw_Server_API.app.main import app
from tldw_Server_API.app.core.TTS import tts_config as tts_config_module
from tldw_Server_API.app.core.TTS.tts_config import reload_tts_config


def _reset_tts_manager() -> None:

    tts_config_module._config_manager = None


def test_effective_config_requires_auth():

    with TestClient(app) as client:
        resp = client.get("/api/v1/admin/config/effective")
        assert resp.status_code in (401, 403)


def test_effective_config_redacts_tts_api_key(monkeypatch, auth_headers):

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-secret")
    _reset_tts_manager()
    reload_tts_config()

    with TestClient(app) as client:
        resp = client.get("/api/v1/admin/config/effective", headers=auth_headers)
        assert resp.status_code == 200
        payload = resp.json()

    tts_values = payload["values"].get("tts", {})
    entry = tts_values.get("providers.openai.api_key")
    assert entry is not None
    assert entry["redacted"] is True
    assert entry["value"] == "<redacted>"
    assert entry["source"] == "env"


def test_effective_config_sections_filter(monkeypatch, auth_headers):

    monkeypatch.setenv("TTS_DEFAULT_PROVIDER", "openai")
    _reset_tts_manager()
    reload_tts_config()

    with TestClient(app) as client:
        resp = client.get(
            "/api/v1/admin/config/effective",
            params={"sections": "tts", "include_defaults": "false"},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        payload = resp.json()

    assert set(payload["values"].keys()) == {"tts"}
    tts_values = payload["values"]["tts"]
    assert "default_provider" in tts_values
    assert tts_values["default_provider"]["source"] == "env"
    assert "strict_validation" not in tts_values


def test_effective_config_sanitizes_resolution_error(monkeypatch, auth_headers):

    from tldw_Server_API.app.api.v1.endpoints import config_admin

    def _raise_missing_config_root():
        raise FileNotFoundError("sensitive config path")

    monkeypatch.setattr(config_admin, "resolve_config_root", _raise_missing_config_root)

    with TestClient(app) as client:
        resp = client.get("/api/v1/admin/config/effective", headers=auth_headers)

    assert resp.status_code == 500
    assert resp.json()["detail"] == "Failed to resolve effective configuration"
