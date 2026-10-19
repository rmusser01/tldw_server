"""Transcription rejects a changed cookie principal before any processing."""

from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps import auth_deps
from tldw_Server_API.app.api.v1.endpoints.audio import audio_transcriptions
from tldw_Server_API.app.core.Utils.fastapi_routes import iter_served_routes

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("expected,status", [("42", 412), ("84", 200), (None, 200)])
def test_b05_transcription_owner_admission(expected: str | None, status: int, monkeypatch: pytest.MonkeyPatch) -> None:
    app = FastAPI()
    app.include_router(audio_transcriptions.router, prefix="/api/v1/audio")
    route = next(r for r in iter_served_routes(app.routes) if r.path == "/api/v1/audio/transcriptions")
    calls = []

    async def processing_probe(**_kwargs: object) -> dict[str, str]:
        calls.append(True)
        return {"text": "admitted unit probe"}

    async def no_side_effect() -> None:
        return None

    monkeypatch.setattr(route.dependant, "call", processing_probe)
    # Retain the actual owner guard; isolate unrelated quotas/DB/provider work.
    for dependency in route.dependant.dependencies:
        if dependency.call is not auth_deps.require_expected_user:
            monkeypatch.setitem(app.dependency_overrides, dependency.call, no_side_effect)
    monkeypatch.setitem(app.dependency_overrides, auth_deps.get_auth_principal, lambda: SimpleNamespace(user_id=84))
    headers = {} if expected is None else {"X-TLDW-Expected-User-ID": expected}
    with TestClient(app) as client:
        response = client.post(route.path, headers=headers, files={"file": ("unit.wav", b"unit bytes", "audio/wav")})
    assert response.status_code == status, response.text  # nosec B101 - Verify actual route admission.
    assert bool(calls) is (status == 200)  # nosec B101 - Rejected requests must not process audio.
    if status == 412:
        assert response.json()["detail"]["code"] == "request_config_scope_changed"  # nosec B101 - Verify structured recovery contract.
    if status == 200:
        assert response.json() == {"text": "admitted unit probe"}  # nosec B101 - The accepted owner reaches and returns the actual route probe.
