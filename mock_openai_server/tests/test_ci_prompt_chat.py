"""Keep the C-01 downstream fixture conditional on the actual saved instructions."""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from tldw_Server_API.app.api.v1.schemas.chat_request_schemas import ChatCompletionRequest
from tldw_Server_API.app.core.Chat.chat_service import apply_prompt_templating
from tldw_Server_API.app.core.LLM_Calls.providers.openai_adapter import OpenAIAdapter

from mock_openai_server.mock_openai.config import MockConfig
from mock_openai_server.mock_openai.server import app, get_config_instance

SYSTEM = "You are a pirate. Respond to everything in pirate speak. Always say ARRR at least once."
QUESTION = "Tell me about the weather today."
ANSWER = "ARRR, matey! I need yer location and current weather data before I can give ye today's forecast."


@pytest.mark.unit
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("templated", [False, True])
@pytest.mark.parametrize(
    "mutation", [None, "missing", "changed", "user_role", "question", "prefix", "suffix", "extra_turn"]
)
def test_ci_prompt_chat_requires_saved_system_instruction(
    monkeypatch: pytest.MonkeyPatch, stream: bool, templated: bool, mutation: str | None
) -> None:
    """Lost prompt selection must not receive the successful pirate fixture."""
    root = Path(__file__).resolve().parents[2]
    config = MockConfig.from_file(
        root / "apps/tldw-frontend/e2e/onboarding-uat/mock-openai/configs/ci-journeys.json"
    )
    monkeypatch.setitem(app.dependency_overrides, get_config_instance, lambda: config)
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": QUESTION}]
    if mutation == "missing":
        messages.pop(0)
    elif mutation == "changed":
        messages[0]["content"] = "You are a helpful assistant."
    elif mutation == "user_role":
        messages[0]["role"] = "user"
    elif mutation == "question":
        messages[-1]["content"] = "Tell me about ships."
    elif mutation == "prefix":
        messages[0]["content"] = "Ignore this instruction. " + SYSTEM
    elif mutation == "suffix":
        messages[0]["content"] += " Invent a forecast."
    elif mutation == "extra_turn":
        messages.insert(1, {"role": "assistant", "content": "An unrelated conversation"})
    if templated:
        request = ChatCompletionRequest(model="gpt-4.1-mini", messages=messages, api_provider="openai")
        system, payload = apply_prompt_templating(request, {}, messages)
        messages = OpenAIAdapter()._build_openai_payload({
            "model": request.model, "messages": payload, "system_message": system,
        })["messages"]
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer sk-ci-controlled-downstream"},
            json={"model": "gpt-4.1-mini", "messages": messages, "stream": stream},
        )
        assert response.status_code == 200
        if stream:
            frames = [line[6:] for line in response.text.splitlines() if line.startswith("data: ")]
            assert frames[-1] == "[DONE]"
            answer = "".join(
                json.loads(frame)["choices"][0]["delta"].get("content", "")
                for frame in frames[:-1]
            ).strip()
        else:
            answer = response.json()["choices"][0]["message"]["content"]
    if mutation is None:
        assert answer == ANSWER
    else:
        assert "ARRR" not in answer


@pytest.mark.unit
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("mutation", [None, "missing", "changed", "user_role", "question", "model"])
def test_ci_character_chat_requires_testbot_context(monkeypatch, stream, mutation):
    """A lost or wrong character/model must not receive the TestBot success response."""
    from tldw_Server_API.app.core.Character_Chat.modules.character_prompt_presets import (
        build_character_system_prompt,
    )

    root = Path(__file__).resolve().parents[2]
    config = MockConfig.from_file(
        root / "apps/tldw-frontend/e2e/onboarding-uat/mock-openai/configs/ci-journeys.json"
    )
    monkeypatch.setitem(app.dependency_overrides, get_config_instance, lambda: config)
    character = {
        "name": "E2E-TestBot-controlled",
        "description": "E2E test character for journey spec",
        "system_prompt": "You are E2E-TestBot. Always respond with exactly: BEEP BOOP.",
    }
    system = build_character_system_prompt(character, character["name"], "User")
    messages = [{"role": "system", "content": system}, {"role": "user", "content": "Hello, who are you?"}]
    model = "gpt-4.1-mini"
    if mutation == "missing":
        messages.pop(0)
    elif mutation == "changed":
        messages[0]["content"] = "You are a helpful assistant."
    elif mutation == "user_role":
        messages[0]["role"] = "user"
    elif mutation == "question":
        messages[-1]["content"] = "Tell me about ships."
    elif mutation == "model":
        model = "gpt-4o"
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer sk-ci-controlled-downstream"},
            json={"model": model, "messages": messages, "stream": stream},
        )
        assert response.status_code == 200
        if stream:
            frames = [line[6:] for line in response.text.splitlines() if line.startswith("data: ")]
            assert frames[-1] == "[DONE]"
            answer = "".join(json.loads(frame)["choices"][0]["delta"].get("content", "") for frame in frames[:-1]).strip()
        else:
            answer = response.json()["choices"][0]["message"]["content"]
    if mutation is None:
        assert answer == "BEEP BOOP."
    else:
        assert "BEEP BOOP" not in answer
