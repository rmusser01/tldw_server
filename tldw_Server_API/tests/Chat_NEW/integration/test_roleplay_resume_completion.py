"""Authenticated protected Resume admission and generation contract coverage."""

from __future__ import annotations

import pytest

from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.core.Character_Chat.character_conversation_factory import create_character_conversation
from tldw_Server_API.app.core.Character_Chat.roleplay_resume_contract import roleplay_resume_fence
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB


@pytest.fixture
def resume_api(credentialed_test_client, populated_chacha_db, auth_headers):
    db = CharactersRAGDB(db_path=populated_chacha_db.db_path_str, client_id="1", owner_user_id="1")
    client = credentialed_test_client
    client.app.dependency_overrides[get_chacha_db_for_user] = lambda: db
    card = db.add_character_card(
        {"name": "Ari", "description": "Original saved archivist", "system_prompt": "Preserve continuity."}
    )
    cid = create_character_conversation(
        db,
        conversation_data={"character_id": card, "title": "Protected Resume", "client_id": "1"},
        provider="local-llm",
        model="local-test",
        initial_messages=[
            {"id": "resume-ancestor", "sender": "user", "content": "Earlier question"},
            {
                "id": "resume-prior",
                "sender": "assistant",
                "content": "Earlier answer",
                "parent_message_id": "resume-ancestor",
            },
        ],
    )
    yield client, db, cid, auth_headers
    client.app.dependency_overrides.pop(get_chacha_db_for_user, None)
    db.close_connection()


def input_request(db, cid):
    return {
        "message_id": "resume-chosen",
        "content": "Original proposal?\n  Keep spaces.  ",
        "expected": roleplay_resume_fence(db.get_roleplay_resume_state(cid)),
    }


def post_input(client, cid, headers, body):
    return client.post(f"/api/v1/chats/{cid}/roleplay/messages", headers=headers, json=body)


def test_idempotent_authenticated_append_returns_original_graph_and_fence(resume_api):
    client, db, cid, headers = resume_api
    body = input_request(db, cid)
    accepted = post_input(client, cid, headers, body)
    assert accepted.status_code == 201, accepted.text
    original = accepted.json()
    assert original["parent_message_id"] == "resume-prior"
    assert original["fence"]["tail"] == {"message_id": "resume-chosen", "message_version": 1}
    db.update_message("resume-ancestor", {"content": "Changed after admission"}, 1)
    replay = post_input(client, cid, headers, body)
    assert replay.status_code == 201, replay.text
    assert replay.json() == original
    changed = post_input(client, cid, headers, {**body, "content": "Conflicting replacement"})
    assert changed.status_code == 409, changed.text
    assert changed.json()["detail"] == {"status": "resume_conflict", "code": "idempotency_conflict"}
    assert db.count_messages_for_conversation(cid) == 3


@pytest.mark.parametrize("mutation", ["ancestor-edit", "append", "settings"])
def test_authenticated_resume_prior_fence_never_silently_branches(resume_api, mutation):
    client, db, cid, headers = resume_api
    body = input_request(db, cid)
    if mutation == "ancestor-edit":
        db.update_message("resume-ancestor", {"content": "Changed before admission"}, 1)
    elif mutation == "append":
        db.add_message(
            {
                "id": "other-tab",
                "conversation_id": cid,
                "sender": "user",
                "content": "Other tab",
                "parent_message_id": "resume-prior",
            }
        )
    else:
        db.upsert_conversation_settings(cid, db.get_roleplay_resume_state(cid)["settings"])
    before = db.count_messages_for_conversation(cid)
    response = post_input(client, cid, headers, body)
    assert response.status_code == 409, response.text
    assert response.json()["detail"]["code"] in {"history_fence_changed", "settings_changed"}
    assert db.get_message_by_id("resume-chosen") is None
    assert db.count_messages_for_conversation(cid) == before


def test_resume_append_is_authenticated_before_any_write(resume_api):
    client, db, cid, headers = resume_api
    response = post_input(client, cid, {}, input_request(db, cid))
    assert response.status_code == 401, response.text
    assert db.count_messages_for_conversation(cid) == 2


def test_enrolled_sync_refuses_resume_before_any_append(resume_api, monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints import character_messages as endpoint

    client, db, cid, headers = resume_api
    monkeypatch.setattr(endpoint, "_active_message_sync_service", lambda *args: object())
    response = post_input(client, cid, headers, input_request(db, cid))
    assert response.status_code == 409, response.text
    assert response.json()["detail"] == {"status": "resume_conflict", "code": "resume_sync_unsupported"}
    assert db.count_messages_for_conversation(cid) == 2


def test_append_quota_is_atomic_and_identical_replay_does_not_spend_it_again(resume_api, monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints import character_messages as endpoint
    from tldw_Server_API.app.core.Character_Chat.character_rate_limiter import CharacterRateLimiter

    client, db, cid, headers = resume_api
    limiter = CharacterRateLimiter(max_messages_per_chat=3, enabled=False)
    monkeypatch.setattr(endpoint, "get_character_rate_limiter", lambda: limiter)
    body = input_request(db, cid)
    accepted = post_input(client, cid, headers, body)
    assert accepted.status_code == 201, accepted.text
    assert post_input(client, cid, headers, body).json() == accepted.json()
    next_body = {
        **body,
        "message_id": "quota-refused",
        "expected": roleplay_resume_fence(db.get_roleplay_resume_state(cid)),
    }
    before = db.get_roleplay_resume_state(cid)["history_version"]
    refused = post_input(client, cid, headers, next_body)
    assert refused.status_code == 403, refused.text
    assert db.get_message_by_id("quota-refused") is None
    assert db.get_roleplay_resume_state(cid)["history_version"] == before
    assert db.count_messages_for_conversation(cid) == 3


def test_resume_append_conceals_foreign_owner_before_any_write(resume_api):
    client, db, cid, headers = resume_api
    body = input_request(db, cid)
    foreign = db.add_conversation({"id": "foreign-conversation", "title": "Foreign", "client_id": "another-owner"})
    response = post_input(client, foreign, headers, body)
    assert response.status_code == 404, response.text
    assert db.count_messages_for_conversation(foreign) == 0
    assert db.count_messages_for_conversation(cid) == 2


def test_global_resume_never_ignores_an_explicit_workspace_scope(resume_api):
    client, db, cid, headers = resume_api
    response = client.post(
        f"/api/v1/chats/{cid}/roleplay/messages?scope_type=workspace&workspace_id=other-workspace",
        headers=headers,
        json=input_request(db, cid),
    )
    assert response.status_code == 404, response.text
    assert db.count_messages_for_conversation(cid) == 2


@pytest.mark.parametrize("mutation", ["ancestor-edit", "settings", "input-id", "input-version", "sync"])
def test_resume_completion_fence_rejects_before_live_sources_or_dispatch(resume_api, monkeypatch, mutation):
    from tldw_Server_API.app.api.v1.endpoints import character_chat_sessions as endpoint

    client, db, cid, headers = resume_api
    accepted = post_input(client, cid, headers, input_request(db, cid))
    assert accepted.status_code == 201, accepted.text
    contract = {
        "version": 1,
        "input_message_id": "resume-chosen",
        "input_message_version": 1,
        "assistant_message_id": "resume-assistant",
        "expected": accepted.json()["fence"],
    }
    if mutation == "ancestor-edit":
        db.update_message("resume-ancestor", {"content": "Another tab"}, 1)
    elif mutation == "settings":
        db.upsert_conversation_settings(cid, db.get_roleplay_resume_state(cid)["settings"])
    elif mutation == "input-id":
        contract["input_message_id"] = "resume-ancestor"
    elif mutation == "input-version":
        contract["input_message_version"] = 2
    else:
        monkeypatch.setattr(endpoint, "_active_chat_sync_service", lambda *args: object())

    def forbidden(*args, **kwargs):
        raise AssertionError("Protected Resume fell through to current sources/provider")

    monkeypatch.setattr(db, "get_character_card_by_id", forbidden)
    monkeypatch.setattr(endpoint, "perform_chat_api_call", forbidden)
    response = client.post(
        f"/api/v1/chats/{cid}/complete-v2",
        headers=headers,
        json={"tldw_roleplay_resume_v1": contract},
    )
    assert response.status_code == 409, response.text
    assert response.json()["detail"]["status"] == "resume_conflict"
    assert db.get_message_by_id("resume-assistant") is None


def generation_request(client, db, cid, headers):
    accepted = post_input(client, cid, headers, input_request(db, cid))
    assert accepted.status_code == 201, accepted.text
    return {
        "tldw_roleplay_resume_v1": {
            "version": 1,
            "input_message_id": "resume-chosen",
            "input_message_version": 1,
            "assistant_message_id": "resume-assistant",
            "expected": accepted.json()["fence"],
        }
    }


def test_unchanged_protected_generation_dispatches_saved_prompt_and_persists_once(resume_api, monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints import character_chat_sessions as endpoint

    client, db, cid, headers = resume_api
    body = generation_request(client, db, cid, headers)
    calls = []
    content = "Original generated reply.\n  Exact spaces.  "

    def provider(**kwargs):
        calls.append(kwargs)
        return {"choices": [{"message": {"role": "assistant", "content": content}}]}

    monkeypatch.setattr(endpoint, "perform_chat_api_call", provider)

    def forbidden(*args, **kwargs):
        raise AssertionError("Loaded a changed source card")

    monkeypatch.setattr(db, "get_character_card_by_id", forbidden)
    response = client.post(f"/api/v1/chats/{cid}/complete-v2", headers=headers, json=body)
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["saved"] is True
    assert result["assistant_content"] == content
    assert result["assistant_message_id"] == "resume-assistant"
    assert result["tldw_roleplay_resume_v1"]["expected"] == body["tldw_roleplay_resume_v1"]["expected"]
    assert calls[0]["api_endpoint"] == "local-llm"
    assert calls[0]["model"] == "local-test"
    assert calls[0]["messages_payload"][-1]["content"] == input_request(db, cid)["content"]
    assert "Original saved archivist" in str(calls[0]["messages_payload"])
    db.update_message("resume-ancestor", {"content": "Later changed"}, 1)
    replay = client.post(f"/api/v1/chats/{cid}/complete-v2", headers=headers, json=body)
    assert replay.status_code == 200, replay.text
    assert replay.json() == result
    assert len(calls) == 1
    assert db.count_messages_for_conversation(cid) == 4


@pytest.mark.parametrize("mutation", ["ancestor-edit", "append", "settings", "input-delete", "sync"])
def test_concurrent_writer_retains_generated_content_without_wrong_save(resume_api, monkeypatch, mutation):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    from tldw_Server_API.app.api.v1.endpoints import character_chat_sessions as endpoint

    client, db, cid, headers = resume_api
    body = generation_request(client, db, cid, headers)
    started, release = Event(), Event()
    content = "Keep this generated content.\n  Exact spaces.  "

    def provider(**kwargs):
        started.set()
        assert release.wait(8), "Test did not release owned provider"
        return {"choices": [{"message": {"content": content}}]}

    monkeypatch.setattr(endpoint, "perform_chat_api_call", provider)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(client.post, f"/api/v1/chats/{cid}/complete-v2", headers=headers, json=body)
        try:
            assert started.wait(8), "Saved provider was never dispatched"
            other = CharactersRAGDB(db_path=db.db_path_str, client_id="1", owner_user_id="1")
            try:
                if mutation == "ancestor-edit":
                    other.update_message("resume-ancestor", {"content": "Another writer"}, 1)
                elif mutation == "append":
                    other.add_message(
                        {
                            "id": "resume-other",
                            "conversation_id": cid,
                            "sender": "user",
                            "content": "Other tab",
                            "parent_message_id": "resume-chosen",
                        }
                    )
                elif mutation == "settings":
                    other.upsert_conversation_settings(cid, other.get_roleplay_resume_state(cid)["settings"])
                elif mutation == "input-delete":
                    other.soft_delete_message("resume-chosen", expected_version=1)
                else:
                    monkeypatch.setattr(endpoint, "_active_chat_sync_service", lambda *args: object())
            finally:
                other.close_connection()
        finally:
            release.set()
        response = future.result(timeout=20)
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["saved"] is False
    assert result["assistant_content"] == content
    assert result["assistant_message_id"] is None
    assert result["tldw_roleplay_resume_v1"]["unsaved_reason"]
    assert db.get_message_by_id("resume-assistant") is None


@pytest.mark.parametrize(
    "provider_result",
    [
        {},
        {"choices": []},
        {"choices": [{"message": {"content": "   "}}]},
        {"error": {"message": "private-provider-detail"}},
    ],
)
def test_invalid_provider_result_is_never_saved_as_roleplay_text(resume_api, monkeypatch, provider_result):
    from tldw_Server_API.app.api.v1.endpoints import character_chat_sessions as endpoint

    client, db, cid, headers = resume_api
    body = generation_request(client, db, cid, headers)
    monkeypatch.setattr(endpoint, "perform_chat_api_call", lambda **kwargs: provider_result)
    response = client.post(f"/api/v1/chats/{cid}/complete-v2", headers=headers, json=body)
    assert response.status_code == 502, response.text
    assert "private-provider-detail" not in response.text
    assert db.get_message_by_id("resume-assistant") is None
    assert db.count_messages_for_conversation(cid) == 3


@pytest.mark.parametrize("action", ["block", "warn"])
def test_frozen_lore_obeys_current_segment_cost_policy(resume_api, monkeypatch, action):
    from tldw_Server_API.app.api.v1.endpoints import character_chat_sessions as endpoint
    from tldw_Server_API.app.core.Character_Chat.world_book_manager import (
        WorldBookService,
    )
    from tldw_Server_API.app.core.Chat.prompt_cost_guardrails import (
        PromptCostGuardrailConfig,
    )

    client, db, _, headers = resume_api
    card = db.add_character_card({"name": "Lio", "description": "Original traveller"})
    books = WorldBookService(db)
    book = books.create_world_book("Historical harbor")
    lore = "Saved harbor law. " * 12
    books.add_entry(book, ["Original"], lore)
    books.attach_to_character(book, card)
    cid = create_character_conversation(
        db,
        conversation_data={
            "character_id": card,
            "title": "Cost policy",
            "client_id": "1",
        },
        provider="local-llm",
        model="local-test",
    )
    body = generation_request(client, db, cid, headers)
    calls = []

    def provider(**kwargs):
        calls.append(kwargs)
        return {"choices": [{"message": {"content": "Actual reply"}}]}

    monkeypatch.setattr(endpoint, "perform_chat_api_call", provider)
    monkeypatch.setattr(
        endpoint,
        "load_prompt_cost_guardrail_config",
        lambda: PromptCostGuardrailConfig(enabled=True, default_action=action, warn_world_book_tokens=1),
    )
    before = db.count_messages_for_conversation(cid)
    response = client.post(f"/api/v1/chats/{cid}/complete-v2", headers=headers, json=body)
    if action == "block":
        assert response.status_code == 413, response.text
        assert calls == []
        assert response.json()["detail"]["prompt_guardrails"]["segment_token_totals"]["world_book"] > 1
        assert lore not in response.text
        assert db.get_message_by_id("resume-assistant") is None
        assert db.count_messages_for_conversation(cid) == before
    else:
        assert response.status_code == 200, response.text
        assert response.json()["saved"] is True
        assert lore.strip() in str(calls[0]["messages_payload"])


@pytest.mark.parametrize("saved_penalty", [1.0, 1.2])
def test_saved_kobold_sampling_reaches_actual_http_payload(resume_api, monkeypatch, saved_penalty):
    import httpx

    from tldw_Server_API.app.core.AuthNZ import provider_credential_runtime
    from tldw_Server_API.app.core.Chat import chat_service

    client, db, _, headers = resume_api
    card = db.add_character_card(
        {
            "name": "Kobold Ari",
            "extensions": {"tldw": {"generation": {"repetition_penalty": saved_penalty}}},
        }
    )
    cid = create_character_conversation(
        db,
        conversation_data={
            "character_id": card,
            "title": "Saved Kobold sampling",
            "client_id": "1",
        },
        provider="kobold",
        model="saved-kobold-model",
    )
    body = generation_request(client, db, cid, headers)
    contrary_extra = {
        "temperature": 0.01,
        "top_p": 0.01,
        "stop": ["CURRENT-STOP"],
        "rep_pen": 2.2,
        "seed": 7,
        "unsaved_sampling_sentinel": True,
    }
    config = {
        "extra_body": contrary_extra,
        "kobold_api": {"api_ip": "https://example.com/saved-kobold", "rep_pen": 2.2, "extra_body": contrary_extra},
    }
    monkeypatch.setattr(provider_credential_runtime, "load_server_config_snapshot", lambda: config)
    sent = []

    def fetch(**kwargs):
        sent.append(kwargs["json"])
        return httpx.Response(
            200,
            json={"results": [{"text": "Actual adapter reply"}]},
            request=httpx.Request("POST", kwargs["url"]),
        )

    adapter = chat_service._get_llm_registry().get_adapter("kobold")
    monkeypatch.setattr(adapter, "http_fetcher", fetch)
    frozen = db.prepare_roleplay_resume_generation(cid, body["tldw_roleplay_resume_v1"], owner_client_id="1")
    response = client.post(f"/api/v1/chats/{cid}/complete-v2", headers=headers, json=body)
    assert response.status_code == 200, response.text
    assert response.json()["saved"] is True
    assert sent[0]["rep_pen"] == saved_penalty
    assert sent[0]["temperature"] == frozen["context"]["provider_arguments"]["temp"]
    assert sent[0]["top_p"] == frozen["context"]["provider_arguments"]["top_p"]
    assert sent[0]["stop_sequence"] == frozen["context"]["provider_arguments"]["stop"]
    assert not {"seed", "unsaved_sampling_sentinel"} & sent[0].keys()
    assert response.json()["provider"] == "kobold"
    assert response.json()["model"] == "saved-kobold-model"


@pytest.mark.parametrize("provider", ["anthropic", "google", "cohere"])
def test_frozen_vendor_instructions_reach_actual_http_payload(resume_api, monkeypatch, provider):
    import httpx

    from tldw_Server_API.app.core.AuthNZ import provider_credential_runtime
    from tldw_Server_API.app.core.Character_Chat.world_book_manager import (
        WorldBookService,
    )
    from tldw_Server_API.app.core.LLM_Calls.providers import (
        anthropic_adapter,
        google_adapter,
    )

    client, db, _, headers = resume_api
    from tldw_Server_API.app.core.Chat.chat_service import (
        known_models_for_provider_cached,
    )

    saved_model = known_models_for_provider_cached(provider)[0]
    card = db.add_character_card(
        {
            "name": "Vendor Ava",
            "description": "Saved vendor identity.",
            "system_prompt": "Original saved instructions.",
        }
    )
    service = WorldBookService(db)
    book = service.create_world_book("Vendor frozen lore")
    service.add_entry(book, ["Original"], "Original frozen harbor law.")
    service.attach_to_character(book, card)
    cid = create_character_conversation(
        db,
        conversation_data={
            "character_id": card,
            "title": "Vendor saved effects",
            "client_id": "1",
        },
        provider=provider,
        model=saved_model,
        conversation_settings={"authorNote": "Original frozen note."},
    )
    body = generation_request(client, db, cid, headers)
    config = {
        provider
        + "_api": {
            "api_key": "roleplay-test-only-key",
            "api_retries": 0,
            "tools": [{"type": "function", "function": {"name": "current-tool"}}],
            "tool_choice": "required",
            "response_format": {"type": "json_object"},
            "n": 4,
            "num_generations": 4,
            "candidate_count": 4,
            "k": 40,
            "topK": 40,
        }
    }
    monkeypatch.setattr(provider_credential_runtime, "load_server_config_snapshot", lambda: config)
    monkeypatch.setenv("LLM_ADAPTERS_NATIVE_HTTP_ANTHROPIC", "1")
    sent = []
    reply = {
        "anthropic": {
            "type": "message",
            "content": [{"type": "text", "text": "Vendor reply"}],
        },
        "google": {"candidates": [{"content": {"parts": [{"text": "Vendor reply"}]}}]},
        "cohere": {
            "text": "Vendor reply",
            "generation_id": "vendor-generation",
            "meta": {},
        },
    }[provider]

    class Client:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def post(self, url, **kwargs):
            sent.append(kwargs["json"])
            return httpx.Response(200, json=reply, request=httpx.Request("POST", url))

        def close(self):
            pass

    if provider == "anthropic":
        monkeypatch.setattr(anthropic_adapter, "http_client_factory", lambda **kwargs: Client())
    elif provider == "google":
        monkeypatch.setattr(google_adapter, "http_client_factory", lambda **kwargs: Client())
    else:
        monkeypatch.setattr(
            "tldw_Server_API.app.core.LLM_Calls.chat_calls.create_session_with_retries",
            lambda **kwargs: Client(),
        )
    response = client.post(f"/api/v1/chats/{cid}/complete-v2", headers=headers, json=body)
    assert response.status_code == 200, response.text
    assert response.json()["saved"] is True
    payload = sent[0]
    for text in (
        "Saved vendor identity.",
        "Original saved instructions.",
        "Original frozen note.",
        "Original frozen harbor law.",
    ):
        assert text in str(payload)
    assert "current-tool" not in str(payload)
    assert "response_format" not in payload
    if provider == "google":
        assert payload["generationConfig"]["candidateCount"] == 1
        assert "responseMimeType" not in payload["generationConfig"]
    elif provider == "cohere":
        assert payload["num_generations"] == 1
        assert "k" not in payload


def test_resume_suppresses_current_local_tool_and_candidate_defaults_at_http_boundary(resume_api, monkeypatch):
    import httpx

    from tldw_Server_API.app.core.AuthNZ import provider_credential_runtime
    from tldw_Server_API.app.core.Chat import chat_service

    client, db, cid, headers = resume_api
    body = generation_request(client, db, cid, headers)
    config = {
        "local_llm": {
            "api_ip": "https://example.com/saved-local",
            "n": 4,
            "tools": [{"type": "function", "function": {"name": "current-tool"}}],
            "tool_choice": "required",
            "response_format": {"type": "json_object"},
            "stop": ["CURRENT-STOP"],
            "seed": 7,
            "top_k": 20,
            "min_p": 0.1,
            "presence_penalty": 0.8,
            "frequency_penalty": 0.9,
            "logit_bias": {"25": 50},
        }
    }
    monkeypatch.setattr(provider_credential_runtime, "load_server_config_snapshot", lambda: config)
    sent = []

    def fetch(**kwargs):
        sent.append(kwargs["json"])
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "Local reply"}}]},
            request=httpx.Request("POST", kwargs["url"]),
        )

    monkeypatch.setattr(chat_service._get_llm_registry().get_adapter("local-llm"), "http_fetcher", fetch)
    response = client.post(f"/api/v1/chats/{cid}/complete-v2", headers=headers, json=body)
    assert response.status_code == 200, response.text
    assert response.json()["saved"] is True
    assert "current-tool" not in str(sent[0])
    assert "response_format" not in sent[0]
    assert sent[0]["n"] == 1
    assert sent[0]["stop"] == []

    assert (
        not {
            "seed",
            "top_k",
            "min_p",
            "presence_penalty",
            "frequency_penalty",
            "logit_bias",
        }
        & sent[0].keys()
    )


@pytest.mark.parametrize("provider", ["qwen", "deepseek"])
def test_resume_suppresses_unsaved_sampling_modifiers_at_vendor_http_boundary(resume_api, monkeypatch, provider):
    import httpx

    from tldw_Server_API.app.core.AuthNZ import provider_credential_runtime
    from tldw_Server_API.app.core.Chat.chat_service import (
        known_models_for_provider_cached,
    )
    from tldw_Server_API.app.core.LLM_Calls.providers import (
        deepseek_adapter,
        qwen_adapter,
    )

    client, db, _, headers = resume_api
    card = db.add_character_card({"name": "Modifier Vendor Ava"})
    model = known_models_for_provider_cached(provider)[0]
    cid = create_character_conversation(
        db,
        conversation_data={
            "character_id": card,
            "title": "Saved sampling modifiers",
            "client_id": "1",
        },
        provider=provider,
        model=model,
    )
    body = generation_request(client, db, cid, headers)
    config = {
        provider
        + "_api": {
            "api_key": "roleplay-test-only-key",
            "seed": 7,
            "presence_penalty": 0.8,
            "frequency_penalty": 0.9,
            "logit_bias": {"25": 50},
        }
    }
    monkeypatch.setattr(provider_credential_runtime, "load_server_config_snapshot", lambda: config)
    sent = []

    class Client:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def close(self):
            pass

        def post(self, url, **kwargs):
            sent.append(kwargs["json"])
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": "Vendor reply"}}]},
                request=httpx.Request("POST", url),
            )

    monkeypatch.setattr(
        qwen_adapter if provider == "qwen" else deepseek_adapter,
        "http_client_factory",
        lambda **kwargs: Client(),
    )
    response = client.post(f"/api/v1/chats/{cid}/complete-v2", headers=headers, json=body)
    assert response.status_code == 200, response.text
    assert response.json()["saved"] is True
    assert not {"seed", "presence_penalty", "frequency_penalty", "logit_bias"} & sent[0].keys()
    assert sent[0]["model"] == model


_RESUME_FEATURES = {
    "snapshot_completion", "fenced_completion", "idempotent_user_append", "nonstream_assistant_persist"
}


def test_docs_info_attests_resume_only_to_authenticated_owner(credentialed_test_client, auth_headers):
    client = credentialed_test_client
    public = client.get("/api/v1/config/docs-info")
    assert public.status_code == 200
    assert "roleplay_resume_contract_version" not in public.json()["capabilities"]
    authenticated = client.get("/api/v1/config/docs-info", headers=auth_headers)
    assert authenticated.status_code == 200
    payload = authenticated.json()
    assert payload["capabilities"]["roleplay_resume_contract_version"] == 1
    assert set(payload["capabilities"]["roleplay_resume_features"]) == _RESUME_FEATURES
    assert payload["capabilities"] == payload["supported_features"]
    assert authenticated.headers["cache-control"] == "no-store"


@pytest.mark.parametrize("headers", [{"X-API-KEY": "invalid-key"}, {"Authorization": "Bearer invalid"},
                                     {"X-API-KEY": ""}, {"Authorization": ""}])
def test_docs_info_does_not_downgrade_explicit_invalid_credentials(credentialed_test_client, headers):
    response = credentialed_test_client.get("/api/v1/config/docs-info", headers=headers)
    assert response.status_code == 401


def test_docs_info_resume_checks_original_expected_owner(credentialed_test_client, auth_headers):
    response = credentialed_test_client.get(
        "/api/v1/config/docs-info", headers={**auth_headers, "X-TLDW-Expected-User-ID": "999999"}
    )
    assert response.status_code == 412
    assert response.json()["detail"]["code"] == "request_config_scope_changed"
    assert response.headers["cache-control"] == "no-store"


@pytest.mark.parametrize("disabled", ["character-chat-sessions", "character-messages"])
def test_docs_info_resume_requires_both_current_route_policies(
    credentialed_test_client, auth_headers, monkeypatch, disabled
):
    from tldw_Server_API.app.core import config as config_mod

    monkeypatch.setenv("ROUTES_DISABLE", disabled)
    config_mod._route_toggle_policy.cache_clear()
    try:
        response = credentialed_test_client.get("/api/v1/config/docs-info", headers=auth_headers)
        assert response.status_code == 200
        assert "roleplay_resume_contract_version" not in response.json()["capabilities"]
    finally:
        config_mod._route_toggle_policy.cache_clear()


@pytest.mark.parametrize("missing", ["append_roleplay_resume_input", "character_chat_completion"])
def test_docs_info_resume_requires_actual_registered_handlers(
    credentialed_test_client, auth_headers, monkeypatch, missing
):
    from fastapi import APIRouter, FastAPI
    from fastapi.testclient import TestClient

    from tldw_Server_API.app.api.v1.endpoints import character_chat_sessions, character_messages, config_info
    from tldw_Server_API.app.core.Utils.fastapi_routes import iter_served_routes

    app = FastAPI()
    for router, prefix in [(character_messages.router, "/api/v1"), (character_chat_sessions.router, "/api/v1/chats")]:
        retained = [route for route in router.routes if getattr(getattr(route, "endpoint", None), "__name__", None) != missing]
        app.include_router(APIRouter(routes=retained), prefix=prefix)
    app.include_router(config_info.router, prefix="/api/v1")
    assert all(getattr(route.endpoint, "__name__", None) != missing for route in iter_served_routes(app.routes))
    with TestClient(app) as client:
        response = client.get("/api/v1/config/docs-info", headers=auth_headers)
    assert response.status_code == 200
    assert "roleplay_resume_contract_version" not in response.json()["capabilities"]


def test_docs_info_resume_derivation_failure_remains_public_only(credentialed_test_client, auth_headers, monkeypatch):
    from tldw_Server_API.app.core import config as config_mod

    def failed_policy(*_args, **_kwargs):
        raise RuntimeError("unavailable-route-policy")

    monkeypatch.setattr(config_mod, "route_enabled", failed_policy)
    response = credentialed_test_client.get("/api/v1/config/docs-info", headers=auth_headers)
    assert response.status_code == 200
    assert "roleplay_resume_contract_version" not in response.json()["capabilities"]
    assert response.json()["capabilities"]["hasReadingSnapshotPagesV1"] is True


def test_docs_info_resume_overlay_survives_missing_public_config(
    credentialed_test_client, auth_headers, monkeypatch, tmp_path
):
    from tldw_Server_API.app.core import config as config_mod

    monkeypatch.setenv("TLDW_CONFIG_PATH", str(tmp_path / "absent-config.txt"))
    config_mod._route_toggle_policy.cache_clear()
    try:
        response = credentialed_test_client.get("/api/v1/config/docs-info", headers=auth_headers)
        assert response.status_code == 200
        assert response.json()["capabilities"]["roleplay_resume_contract_version"] == 1
    finally:
        config_mod._route_toggle_policy.cache_clear()
