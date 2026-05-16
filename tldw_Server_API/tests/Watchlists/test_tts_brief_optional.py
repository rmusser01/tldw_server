import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user


pytestmark = pytest.mark.integration


@pytest.fixture()
def client_with_user(monkeypatch, tmp_path):
    async def override_user():
        return User(id=947, username="wluser", email="wl@example.com", is_active=True)

    base_dir = tmp_path / "test_user_dbs_tts_brief_optional"
    base_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("USER_DB_BASE_DIR", str(base_dir))
    monkeypatch.setenv("TEST_MODE", "1")

    from fastapi import FastAPI
    from tldw_Server_API.app.core.config import API_V1_PREFIX
    from tldw_Server_API.app.api.v1.endpoints.watchlists import router as watchlists_router

    app = FastAPI()
    app.include_router(watchlists_router, prefix=f"{API_V1_PREFIX}")
    app.dependency_overrides[get_request_user] = override_user
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()


@pytest.fixture()
def mock_tts_service(monkeypatch):
    class DummyTTS:
        async def generate_speech(self, req):  # noqa: ARG002
            yield b"FAKEAUDIO"

    async def _fake_get_tts_service_v2(*args, **kwargs):  # noqa: ARG002
        return DummyTTS()

    monkeypatch.setattr(
        "tldw_Server_API.app.services.outputs_service.get_tts_service_v2",
        _fake_get_tts_service_v2,
    )


def _create_run(c: TestClient, *, output_prefs: dict | None = None, suffix: str = "a") -> int:
    source = c.post(
        "/api/v1/watchlists/sources",
        json={
            "name": f"TTS Brief Feed {suffix}",
            "url": f"https://example.com/tts-brief-{suffix}.xml",
            "source_type": "rss",
        },
    )
    assert source.status_code == 200, source.text
    source_id = source.json()["id"]

    job_payload = {
        "name": f"TTS Brief Job {suffix}",
        "scope": {"sources": [source_id]},
        "active": True,
    }
    if output_prefs is not None:
        job_payload["output_prefs"] = output_prefs
    job = c.post("/api/v1/watchlists/jobs", json=job_payload)
    assert job.status_code == 200, job.text
    job_id = job.json()["id"]

    run = c.post(f"/api/v1/watchlists/jobs/{job_id}/run")
    assert run.status_code == 200, run.text
    return run.json()["id"]


def test_auto_tts_brief_generates_audio_for_small_runs(client_with_user: TestClient, mock_tts_service):
    c = client_with_user
    run_id = _create_run(
        c,
        output_prefs={"tts_brief": {"enabled": True, "max_items": 10}},
        suffix="auto",
    )

    create = c.post("/api/v1/watchlists/outputs", json={"run_id": run_id, "title": "Auto TTS Brief"})
    assert create.status_code == 200, create.text
    created_output = create.json()
    assert created_output.get("metadata", {}).get("tts_brief_auto") is True

    listed = c.get("/api/v1/watchlists/outputs", params={"run_id": run_id})
    assert listed.status_code == 200, listed.text
    outputs = listed.json()["items"]
    tts_outputs = [o for o in outputs if o.get("type") == "tts_audio"]
    assert len(tts_outputs) == 1
    assert tts_outputs[0].get("format") == "mp3"


def test_auto_tts_brief_respects_threshold_and_skips(client_with_user: TestClient, mock_tts_service):
    c = client_with_user
    run_id = _create_run(
        c,
        output_prefs={"tts_brief": {"enabled": True, "max_items": 0}},
        suffix="threshold",
    )

    create = c.post("/api/v1/watchlists/outputs", json={"run_id": run_id, "title": "No Auto TTS"})
    assert create.status_code == 200, create.text
    created_output = create.json()
    assert created_output.get("metadata", {}).get("tts_brief_auto") is None

    listed = c.get("/api/v1/watchlists/outputs", params={"run_id": run_id})
    assert listed.status_code == 200, listed.text
    outputs = listed.json()["items"]
    tts_outputs = [o for o in outputs if o.get("type") == "tts_audio"]
    assert not tts_outputs


def test_explicit_generate_tts_false_disables_auto_mode(client_with_user: TestClient, mock_tts_service):
    c = client_with_user
    run_id = _create_run(
        c,
        output_prefs={"tts_brief": {"enabled": True, "max_items": 10}},
        suffix="explicit-false",
    )

    create = c.post(
        "/api/v1/watchlists/outputs",
        json={"run_id": run_id, "title": "Explicit False", "generate_tts": False},
    )
    assert create.status_code == 200, create.text
    created_output = create.json()
    assert created_output.get("metadata", {}).get("tts_brief_auto") is None

    listed = c.get("/api/v1/watchlists/outputs", params={"run_id": run_id})
    assert listed.status_code == 200, listed.text
    outputs = listed.json()["items"]
    tts_outputs = [o for o in outputs if o.get("type") == "tts_audio"]
    assert not tts_outputs
