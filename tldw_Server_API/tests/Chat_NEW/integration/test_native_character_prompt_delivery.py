"""Join native Character history preparation to Ollama's outbound prompt payload."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from contextlib import contextmanager
from copy import deepcopy
from typing import Any

import httpx
import pytest
from starlette.requests import Request

from tldw_Server_API.app.api.v1.endpoints.chat import (
    _process_content_for_db_sync,
    _save_message_turn_to_db,
)
from tldw_Server_API.app.api.v1.schemas.chat_request_schemas import ChatCompletionRequest
from tldw_Server_API.app.core.AuthNZ.byok_runtime import (
    ByokResolutionStatus,
    ResolvedByokCredentials,
)
from tldw_Server_API.app.core.AuthNZ.provider_credential_runtime import (
    PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY,
    ProviderCredentialRuntime,
)
from tldw_Server_API.app.core.Character_Chat import character_conversation_factory
from tldw_Server_API.app.core.Chat import chat_service
from tldw_Server_API.app.core.Chat.chat_metrics import get_chat_metrics
from tldw_Server_API.app.core.Chat.history_selection import (
    resolve_history_selection,
    snapshot_to_wire,
)
from tldw_Server_API.app.core.Chat.persistence_service import (
    native_history_owner_key,
    prepare_native_history_message,
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.LLM_Calls.adapter_registry import ChatProviderRegistry


@pytest.fixture
def native_provider_configuration(monkeypatch):
    """Supply external provider configuration/inventory; keep binding and transport construction real."""
    model = "qwen2.5:0.5b"
    endpoint = "http://127.0.0.1:11434"
    app_config = {"ollama_api": {"api_url": endpoint, "model": model}}
    # Isolate configuration and inventory inputs; use the real model binding checks.
    monkeypatch.setattr(character_conversation_factory, "ensure_app_config", lambda: deepcopy(app_config))
    monkeypatch.setattr(chat_service, "load_comprehensive_config", lambda: deepcopy(app_config))
    monkeypatch.setattr(chat_service, "list_provider_models", lambda _provider: [model])
    return app_config


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("preset,separator", [("default", "\n"), ("st_default", "\n\n")])
async def test_saved_testbot_instruction_reaches_first_ollama_system_message(
    chacha_db: CharactersRAGDB, monkeypatch: pytest.MonkeyPatch, native_provider_configuration, preset, separator,
) -> None:
    """Observe prompt delivery independently of any model's generated answer."""
    instruction = "You are E2E-TestBot. Always respond with exactly: BEEP BOOP."
    app_config = native_provider_configuration
    model = app_config["ollama_api"]["model"]
    endpoint = app_config["ollama_api"]["api_url"]

    character_id = chacha_db.add_character_card({
        "name": "E2E-TestBot",
        "extensions": {"tldw": {"prompt_preset": preset}},
        "system_prompt": instruction,
        "first_message": "BEEP BOOP.",
    })
    conversation_id = character_conversation_factory.create_character_conversation(
        chacha_db,
        conversation_data={"character_id": character_id, "title": "TestBot first send"},
        provider="ollama",
        model=model,
        seed_first_message=True,
    )
    owner_key = native_history_owner_key(Request({
        "type": "http", "scheme": "http", "server": ("testserver", 80),
        "path": "/api/v1/chat/completions", "root_path": "", "headers": [],
    }), "1")
    snapshot = snapshot_to_wire(chacha_db.get_conversation_history_snapshot(
        conversation_id, owner_client_id="1", owner_key=owner_key,
    ))
    assert len(snapshot["nodes"]) == 1  # nosec B101
    # Pytest validates the real native greeting role before admitting a user turn.
    assert snapshot["nodes"][0]["role"] == "assistant"  # nosec B101
    resolved = resolve_history_selection(snapshot, {
        "owner_key": owner_key,
        "conversation_id": conversation_id,
        "interpretation": {"kind": "parent_graph_v1"},
        "cursor": {"kind": "after_message", "message_id": snapshot["nodes"][0]["id"]},
        "selection_revision": 1,
    }, "send", "testbot-first-send")
    assert resolved["status"] == "ready", resolved  # nosec B101
    request_data = ChatCompletionRequest(
        messages=[{"role": "user", "content": "Hello"}],
        conversation_id=conversation_id,
        model=model,
        api_provider="ollama",
        save_to_db=True,
        stream=True,
        tldw_history_selection_v1=resolved["selection"],
    )
    user_input = await prepare_native_history_message(
        request_data.messages[0].model_dump(exclude_none=True),
        conversation_id,
        _process_content_for_db_sync,
    )
    runtime_state = {
        "history_owner_key": owner_key,
        "history_owner_client_id": "1",
        "history_inputs": [user_input],
    }
    card, projected_id, accepted_id, created, messages, persists = await chat_service.build_context_and_messages(
        chacha_db, request_data, asyncio.get_running_loop(), get_chat_metrics(),
        True, conversation_id, _save_message_turn_to_db, runtime_state,
    )
    assert (projected_id, accepted_id, created, persists) == (  # nosec B101
        character_id, conversation_id, False, True,
    )
    system_prompt, templated_messages = chat_service.apply_prompt_templating(request_data, card, messages)
    call_params = chat_service.build_call_params_from_request(
        request_data, "ollama", None, templated_messages, system_prompt,
        app_config=app_config,
    )

    async def resolve_credentials(provider: str, **_kwargs: Any) -> ResolvedByokCredentials:
        """Provide only this test's nonsecret server endpoint to the real runtime."""
        return ResolvedByokCredentials(
            provider=provider, api_key=None, app_config=deepcopy(app_config),
            credential_fields={"base_url": endpoint}, source="server",
            allowlisted=True, status=ByokResolutionStatus.ABSENT,
        )

    credentials_runtime = ProviderCredentialRuntime(
        user_id=1, team_ids=(), org_ids=(), trusted_base_url_override=False,
        server_config_snapshot=app_config, resolver=resolve_credentials,
    )
    captured: list[dict[str, Any]] = []

    @contextmanager
    def capture_stream(**kwargs: Any) -> Iterator[httpx.Response]:
        # Capture at the transport boundary after the real adapter constructs its JSON.
        assert kwargs["configured_endpoint"].matches(kwargs["url"])  # nosec B101
        captured.append({"url": kwargs["url"], "payload": deepcopy(kwargs["json"])})
        response = httpx.Response(
            200, content='data: [DONE]\n\n',
            request=httpx.Request("POST", kwargs["url"]),
        )
        try:
            yield response
        finally:
            response.close()

    registry = ChatProviderRegistry()
    adapter = registry.get_adapter("ollama")
    assert adapter is not None  # nosec B101
    adapter.http_streamer = capture_stream
    monkeypatch.setattr(chat_service, "_get_llm_registry", lambda: registry)
    try:
        call_params[PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY] = await credentials_runtime.resolve("ollama", model=model)
        chunks = list(chat_service.perform_chat_api_call(**call_params))
    finally:
        await credentials_runtime.close()

    assert chunks == ["data: [DONE]\n\n"]  # nosec B101
    assert len(captured) == 1  # nosec B101
    assert captured[0]["url"] == endpoint + "/v1/chat/completions"  # nosec B101
    assert captured[0]["payload"]["model"] == model  # nosec B101
    assert captured[0]["payload"]["stream"] is True  # nosec B101
    assert captured[0]["payload"]["messages"] == [  # nosec B101
        {"role": "system", "content": "You are E2E-TestBot." + separator + instruction},
        {"role": "assistant", "content": [{"type": "text", "text": "BEEP BOOP."}]},
        {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
    ]
