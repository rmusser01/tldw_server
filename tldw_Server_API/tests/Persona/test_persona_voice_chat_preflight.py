"""Voice preparation must not preempt authenticated Chat credential admission."""

from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import Depends, FastAPI, HTTPException, Request

from tldw_Server_API.app.core.Persona import live_conversation as live

pytestmark = pytest.mark.unit


@pytest.fixture
def configured_target(monkeypatch: pytest.MonkeyPatch) -> Any:
    from tldw_Server_API.app.core.Chat import chat_service, chat_target_resolution

    target = SimpleNamespace(provider="deepseek", model="deepseek-chat")
    monkeypatch.setattr(chat_target_resolution, "resolve_chat_target", lambda **kwargs: target)

    def unexpected_static_credentials(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("Voice preparation cannot decide the caller's effective credentials from server keys")

    monkeypatch.setattr(chat_service, "resolve_static_provider_fallback", unexpected_static_credentials)
    return target


def test_voice_preflight_does_not_read_static_credentials(configured_target: Any) -> None:
    target = live.require_persona_voice_conversation_credentials()
    assert (target.provider, target.model) == ("deepseek", "deepseek-chat")


def test_voice_preflight_retains_invalid_target_error(monkeypatch: pytest.MonkeyPatch) -> None:
    from tldw_Server_API.app.core.Chat import chat_target_resolution
    from tldw_Server_API.app.core.Chat.Chat_Deps import ChatConfigurationError

    def invalid_target(**kwargs: Any) -> Any:
        raise ChatConfigurationError(provider="invalid", message="private configuration detail")

    monkeypatch.setattr(chat_target_resolution, "resolve_chat_target", invalid_target)
    with pytest.raises(live.PersonaConversationError, match="supported default Chat provider and model") as error:
        live.require_persona_voice_conversation_credentials()
    assert "private configuration detail" not in str(error.value)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("admission_status", "expected_error"),
    [(200, None), (401, "not authorized"), (403, "not authorized"), (429, "usage is limited")],
)
async def test_prepared_voice_dispatch_still_runs_authenticated_http_admission(
    configured_target: Any, admission_status: int, expected_error: str | None
) -> None:
    app = FastAPI()
    admissions = []

    async def admission(request: Request) -> None:
        admissions.append((request.headers.get("authorization"), request.client.host))
        if admission_status != 200:
            raise HTTPException(admission_status, "private admission detail")

    @app.post("/api/v1/chat/completions", dependencies=[Depends(admission)])
    async def chat(request: Request) -> Any:
        body = await request.json()
        assert (body["api_provider"], body["model"]) == ("deepseek", "deepseek-chat")
        return {"choices": [{"message": {"content": "Spoken reply"}}]}

    live.require_persona_voice_conversation_credentials()
    completion = live.complete_persona_conversation(
        app=app,
        headers={"authorization": "Bearer test-only"},
        client=("192.0.2.42", 1234),
        system_prompt="You are Migu.",
        turns=[{"role": "user", "content": "Hello"}],
    )
    if expected_error is None:
        assert await completion == "Spoken reply"
    else:
        with pytest.raises(live.PersonaConversationError, match=expected_error) as error:
            await completion
        assert "private admission detail" not in str(error.value)
    assert admissions == [("Bearer test-only", "192.0.2.42")]
