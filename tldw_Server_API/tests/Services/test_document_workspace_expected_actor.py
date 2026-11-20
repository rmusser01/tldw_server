"""Selected-text requests must keep the authenticated document account."""

from collections.abc import Callable
from types import SimpleNamespace
from unittest.mock import AsyncMock

from fastapi import APIRouter, FastAPI, HTTPException
from httpx import ASGITransport, AsyncClient
import pytest

from tldw_Server_API.app.api.v1.API_Deps import auth_deps
from tldw_Server_API.app.api.v1.endpoints import translate
from tldw_Server_API.app.api.v1.endpoints.audio import audio_tts


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("router", "path", "body", "generation_dependency"),
    [
        (
            translate.router, "/translate",
            {"text": "Selected passage", "target_language": "French"},
            translate.get_prompts_db_for_user,
        ),
        (
            audio_tts.router, "/speech", {"input": "Selected passage", "model": "fixture", "voice": "alloy"},
            audio_tts.get_tts_service,
        ),
    ],
)
async def test_rejects_selected_text_when_authenticated_account_changed(
    monkeypatch: pytest.MonkeyPatch,
    router: APIRouter,
    path: str,
    body: dict[str, str],
    generation_dependency: Callable[..., object],
) -> None:
    monkeypatch.setattr(
        auth_deps, "get_auth_principal", AsyncMock(return_value=SimpleNamespace(user_id=2))
    )
    generation_calls = []

    async def generation_unavailable() -> None:
        generation_calls.append(path)
        raise HTTPException(status_code=503)

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[auth_deps.get_request_user] = lambda: SimpleNamespace(id=2)
    app.dependency_overrides[auth_deps.check_rate_limit] = lambda: None
    app.dependency_overrides[generation_dependency] = generation_unavailable
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        result = await client.post(path, headers={"X-TLDW-Expected-User-ID": "1"}, json=body)

    assert result.status_code == 412
    assert result.json()["detail"]["code"] == "request_config_scope_changed"
    assert generation_calls == []
