"""Assigned dictionaries cross real owner settings, native admission and transport."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from copy import deepcopy
from functools import partial
from types import SimpleNamespace

import httpx
import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.api.v1.endpoints.chat import _process_content_for_db_sync, _save_message_turn_to_db
from tldw_Server_API.app.api.v1.schemas.chat_request_schemas import ChatCompletionRequest
from tldw_Server_API.app.core.AuthNZ.byok_runtime import ByokResolutionStatus, ResolvedByokCredentials
from tldw_Server_API.app.core.AuthNZ.provider_credential_runtime import (
    PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY,
    ProviderCredentialRuntime,
)
from tldw_Server_API.app.core.Character_Chat import character_conversation_factory
from tldw_Server_API.app.core.Character_Chat.world_book_manager import WorldBookService
from tldw_Server_API.app.core.Chat import chat_service
from tldw_Server_API.app.core.Chat.chat_metrics import get_chat_metrics
from tldw_Server_API.app.core.Chat.history_selection import resolve_history_selection
from tldw_Server_API.app.core.Chat.persistence_service import prepare_native_history_message
from tldw_Server_API.app.core.DB_Management.chacha.conversation_resume_store import build_materialized_behavior_settings
from tldw_Server_API.app.core.DB_Management.db_errors import NotFoundError
from tldw_Server_API.app.core.LLM_Calls import gateway
from tldw_Server_API.app.core.LLM_Calls.adapter_registry import ChatProviderRegistry
from tldw_Server_API.tests.AuthNZ.integration.test_single_user_cookie_session import _mint
from tldw_Server_API.tests.AuthNZ.integration.test_single_user_cookie_session import (
    single_user_cookie_client as single_user_cookie_client,
)

pytestmark = pytest.mark.integration
TEXT = "Echo UAT_IN_742; ORBIT-742."
MODEL = "qwen2.5:0.5b"
ENDPOINT = "http://127.0.0.1:11434"
CONFIG = {"ollama_api": {"api_url": ENDPOINT, "model": MODEL}}


@pytest.fixture
def dictionary_chat(single_user_cookie_client, chacha_db, monkeypatch):
    """Use normal authenticated app fixtures and private owner-scoped content DB."""
    client, api_key = single_user_cookie_client
    db = chacha_db
    assert _mint(client, api_key).status_code == 200  # nosec B101 - pytest contract
    auth_headers = {"X-CSRF-Token": client.cookies["csrf_token"]}
    previous = client.app.dependency_overrides.get(get_chacha_db_for_user)
    client.app.dependency_overrides[get_chacha_db_for_user] = lambda: db
    monkeypatch.setattr(character_conversation_factory, "ensure_app_config", lambda: deepcopy(CONFIG))
    monkeypatch.setattr(chat_service, "load_comprehensive_config", lambda: deepcopy(CONFIG))
    monkeypatch.setattr(chat_service, "list_provider_models", lambda _provider: [MODEL])
    monkeypatch.setattr(gateway, "list_provider_models", chat_service.list_provider_models)
    created = client.post("/api/v1/chat/dictionaries", headers=auth_headers, json={"name": "TokenSwap"})
    assert created.status_code == 201, created.text  # nosec B101 - pytest contract
    dictionary_id = created.json()["id"]
    entry = client.post(f"/api/v1/chat/dictionaries/{dictionary_id}/entries", headers=auth_headers,
                        json={"pattern": "UAT_IN_742", "replacement": "UAT_OUT_742", "type": "literal"})
    assert entry.status_code == 201, entry.text  # nosec B101 - pytest contract
    books = WorldBookService(db)
    character_id = db.add_character_card({"name": "LoreGuide", "system_prompt": "Preserve LoreGuide instructions."})
    book_id = books.create_world_book("ORBIT canon")
    books.add_world_book_entry(book_id, ["ORBIT-742"], "ORBIT-742 access phrase is amber kestrel.")
    books.attach_to_character(book_id, character_id)
    cid = character_conversation_factory.create_character_conversation(
        db, conversation_data={"character_id": character_id, "title": "Native dictionary"},
        provider="ollama", model=MODEL,
    )
    prior = db.add_message({"conversation_id": cid, "sender": "user", "content": "Earlier UAT_IN_742."})
    tail = db.add_message({"conversation_id": cid, "sender": "assistant", "content": "Earlier answer.",
                           "parent_message_id": prior})
    db.close_connection()
    try:
        yield SimpleNamespace(client=client, db=db, headers=auth_headers, cid=cid, tail=tail,
                              dictionary_id=dictionary_id, entry_id=entry.json()["id"])
    finally:
        if previous is None:
            client.app.dependency_overrides.pop(get_chacha_db_for_user, None)
        else:
            client.app.dependency_overrides[get_chacha_db_for_user] = previous


def assign(chat, ids=None):
    """Use the exact singleton Quick assign payload and real public writer."""
    ids = ids if ids is not None else [chat.dictionary_id]
    patch = {"chat_dictionary_ids": ids}
    if len(ids) == 1:
        patch["chat_dictionary_id"] = ids[0]
    response = chat.client.put(f"/api/v1/chats/{chat.cid}/settings?scope_type=global",
                               headers=chat.headers, json={"settings": patch})
    assert response.status_code == 200, response.text  # nosec B101 - pytest contract
    return response


def capture(chat):
    """Reload saved settings, then capture actual owner ancestry and its fences."""
    reopened = chat.client.get(f"/api/v1/chats/{chat.cid}/settings?scope_type=global", headers=chat.headers)
    assert reopened.status_code == 200, reopened.text  # nosec B101 - pytest contract
    response = chat.client.post(f"/api/v1/chat/conversations/{chat.cid}/history/selection",
                               headers=chat.headers, json={"purpose": "send", "view": {
                                   "view_session_id": "dictionary-view", "conversation_id": chat.cid,
                                   "interpretation": {"kind": "parent_graph_v1"},
                                   "cursor": {"kind": "after_message", "message_id": chat.tail},
                                   "selection_revision": 1,
                               }})
    assert response.status_code == 200, response.text  # nosec B101 - pytest contract
    body = response.json()
    return resolve_history_selection(body["snapshot"], body["view"], "send", "dictionary-request")["selection"]


async def prepare(chat, selection, *, text=TEXT, owner="1"):
    """Real native preparation keeps authored admission separate from provider text."""
    request = ChatCompletionRequest(model=MODEL, api_provider="ollama", conversation_id=chat.cid,
                                    save_to_db=True, stream=True, messages=[{"role": "user", "content": text}],
                                    tldw_history_selection_v1=selection)
    user_input = await prepare_native_history_message(request.messages[0].model_dump(exclude_none=True),
                                                      chat.cid, _process_content_for_db_sync)
    runtime = {"history_owner_key": selection["owner_key"], "history_owner_client_id": owner,
               "history_inputs": [user_input]}
    loop = asyncio.get_running_loop()
    try:
        built = await chat_service.build_context_and_messages(chat.db, request, loop, get_chat_metrics(),
                                                             True, chat.cid, _save_message_turn_to_db, runtime)
        system, messages = chat_service.apply_prompt_templating(request, built[0], built[4])
        return request, system, messages, runtime
    finally:
        await asyncio.to_thread(chat.db.close_connection)


async def transport(request, system, messages):
    """Inject only HTTP transport; retain real provider/runtime/payload construction."""
    captured = []
    registry = ChatProviderRegistry()
    adapter = registry.get_adapter("ollama")

    @contextmanager
    def http_stream(**kwargs):
        assert kwargs["configured_endpoint"].matches(kwargs["url"])  # nosec B101 - pytest contract
        captured.append(deepcopy(kwargs["json"]))
        response = httpx.Response(200, content="data: [DONE]\n\n", request=httpx.Request("POST", kwargs["url"]))
        try:
            yield response
        finally:
            response.close()

    async def resolve_credentials(provider, **_kwargs):
        return ResolvedByokCredentials(provider=provider, api_key=None, app_config=deepcopy(CONFIG),
                                       credential_fields={"base_url": ENDPOINT}, source="server",
                                       allowlisted=True, status=ByokResolutionStatus.ABSENT)

    adapter.http_streamer = http_stream
    credentials = ProviderCredentialRuntime(user_id=1, team_ids=(), org_ids=(), trusted_base_url_override=False,
                                            server_config_snapshot=CONFIG, resolver=resolve_credentials)
    params = chat_service.build_call_params_from_request(request, "ollama", None, messages, system, app_config=CONFIG)
    params[PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY] = await credentials.resolve("ollama", model=MODEL)
    # perform_chat_api_call accepts the explicit real adapter registry at this seam.
    previous = gateway._get_llm_registry
    gateway._get_llm_registry = lambda: registry
    try:
        list(chat_service.perform_chat_api_call(**params))
    finally:
        gateway._get_llm_registry = previous
        await credentials.close()
    return captured[0]["messages"]


@pytest.mark.parametrize("assigned", [False, True])
def test_native_provider_transforms_only_assigned_current_input_and_preserves_authored_storage(dictionary_chat, assigned):
    """Allowlisting without processing, global processing, or rewriting history fails."""
    chat = dictionary_chat
    if assigned:
        assign(chat)
    selection = capture(chat)
    request, system, messages, runtime = chat.client.portal.call(prepare, chat, selection)
    provider_messages = chat.client.portal.call(transport, request, system, messages)
    assert "Preserve LoreGuide instructions." in provider_messages[0]["content"]  # nosec B101 - pytest contract
    assert "amber kestrel" in provider_messages[0]["content"]  # nosec B101 - pytest contract
    assert provider_messages[1]["content"] == [{"type": "text", "text": "Earlier UAT_IN_742."}]  # nosec B101
    assert provider_messages[2]["content"] == [{"type": "text", "text": "Earlier answer."}]  # nosec B101
    expected = "Echo UAT_OUT_742; ORBIT-742." if assigned else TEXT
    assert provider_messages[-1]["content"] == [{"type": "text", "text": expected}]  # nosec B101 - pytest contract
    assert chat.db.get_message_by_id(runtime["user_message_id"])["content"] == TEXT  # nosec B101 - pytest contract
    assert chat.db.count_messages_for_conversation(chat.cid) == 3  # nosec B101 - pytest contract


def new_dictionary(chat, name, pattern, replacement, **metadata):
    """Create actual catalog rows with supported APIs for composition controls."""
    response = chat.client.post("/api/v1/chat/dictionaries", headers=chat.headers, json={"name": name, **metadata})
    assert response.status_code == 201, response.text  # nosec B101
    dictionary_id = response.json()["id"]
    response = chat.client.post(f"/api/v1/chat/dictionaries/{dictionary_id}/entries", headers=chat.headers,
                               json={"pattern": pattern, "replacement": replacement, "type": "literal"})
    assert response.status_code == 201, response.text  # nosec B101
    return dictionary_id


@pytest.mark.parametrize("foreign", ["client", "key"])
def test_foreign_native_owner_cannot_admit_assigned_input(dictionary_chat, foreign):
    chat = dictionary_chat
    assign(chat)
    selection = capture(chat)
    if foreign == "key":
        selection["owner_key"] = "foreign-owner"
    with pytest.raises((HTTPException, NotFoundError)):
        chat.client.portal.call(partial(prepare, chat, selection, owner="2" if foreign == "client" else "1"))
    assert chat.db.count_messages_for_conversation(chat.cid) == 2  # nosec B101


def test_foreign_scope_cannot_change_dictionary_assignment(dictionary_chat):
    chat = dictionary_chat
    response = chat.client.put(f"/api/v1/chats/{chat.cid}/settings?scope_type=workspace&workspace_id=foreign",
                               headers=chat.headers, json={"settings": {"chat_dictionary_ids": [chat.dictionary_id]}})
    assert response.status_code in (403, 404), response.text  # nosec B101
    assert "chat_dictionary_ids" not in chat.db.get_conversation_settings(chat.cid)["settings"]  # nosec B101


def test_changed_assignment_after_capture_refuses_before_native_append(dictionary_chat):
    chat = dictionary_chat
    assign(chat)
    selection = capture(chat)
    assign(chat, [])
    with pytest.raises(HTTPException) as failure:
        chat.client.portal.call(prepare, chat, selection)
    assert failure.value.status_code == 409  # nosec B101
    assert failure.value.detail["status"] == "stale_selection"  # nosec B101
    assert chat.db.count_messages_for_conversation(chat.cid) == 2  # nosec B101


@pytest.mark.parametrize("catalog_change", ["edit", "delete"])
def test_accepted_dictionary_snapshot_survives_current_catalog_change(dictionary_chat, catalog_change, monkeypatch):
    chat = dictionary_chat
    assign(chat)
    selection = capture(chat)
    if catalog_change == "edit":
        response = chat.client.put(f"/api/v1/chat/dictionaries/entries/{chat.entry_id}",
                                   headers=chat.headers, json={"replacement": "LIVE_CHANGE"})
    else:
        response = chat.client.delete(f"/api/v1/chat/dictionaries/{chat.dictionary_id}", headers=chat.headers)
    assert response.status_code in (200, 204), response.text  # nosec B101
    from tldw_Server_API.app.core.Character_Chat.chat_dictionary import ChatDictionaryService

    def no_live_catalog(*_args, **_kwargs):
        raise AssertionError("No live catalog reader is allowed after accepted assignment")

    monkeypatch.setattr(ChatDictionaryService, "get_entries", no_live_catalog)
    monkeypatch.setattr(ChatDictionaryService, "get_dictionary", no_live_catalog)
    monkeypatch.setattr(ChatDictionaryService, "snapshot_assigned_dictionaries", no_live_catalog)
    _, system, messages, _ = chat.client.portal.call(prepare, chat, selection)
    assert messages[-1]["content"] == [{"type": "text", "text": "Echo UAT_OUT_742; ORBIT-742."}]  # nosec B101
    assert "amber kestrel" in system  # nosec B101


def test_explicit_reassignment_adopts_new_catalog_definition(dictionary_chat):
    chat = dictionary_chat
    assign(chat)
    response = chat.client.put(f"/api/v1/chat/dictionaries/entries/{chat.entry_id}",
                               headers=chat.headers, json={"replacement": "REASSIGNED"})
    assert response.status_code == 200, response.text  # nosec B101
    assign(chat)
    _, _, messages, _ = chat.client.portal.call(prepare, chat, capture(chat))
    assert messages[-1]["content"] == [{"type": "text", "text": "Echo REASSIGNED; ORBIT-742."}]  # nosec B101


def test_assigned_include_order_runs_before_root_without_global_spillover(dictionary_chat):
    chat = dictionary_chat
    base = new_dictionary(chat, "Base", "UAT_IN_742", "BASE_TOKEN")
    root = new_dictionary(chat, "Root", "BASE_TOKEN", "ROOT_TOKEN", included_dictionary_ids=[base])
    new_dictionary(chat, "Unassigned global", "ROOT_TOKEN", "UNASSIGNED_MUST_NOT_APPEAR")
    assign(chat, [root])
    _, _, messages, _ = chat.client.portal.call(prepare, chat, capture(chat))
    assert messages[-1]["content"] == [{"type": "text", "text": "Echo ROOT_TOKEN; ORBIT-742."}]  # nosec B101


@pytest.mark.parametrize("disabled", ["dictionary", "entry"])
def test_disabled_assignment_effect_does_not_replace_current_input(dictionary_chat, disabled):
    chat = dictionary_chat
    path = f"/api/v1/chat/dictionaries/{chat.dictionary_id}" if disabled == "dictionary" else f"/api/v1/chat/dictionaries/entries/{chat.entry_id}"
    patch = {"is_active": False} if disabled == "dictionary" else {"enabled": False}
    response = chat.client.put(path, headers=chat.headers, json=patch)
    assert response.status_code == 200, response.text  # nosec B101
    assign(chat)
    _, _, messages, _ = chat.client.portal.call(prepare, chat, capture(chat))
    assert messages[-1]["content"] == [{"type": "text", "text": TEXT}]  # nosec B101


def test_malformed_materialized_dictionary_plan_refuses_before_append(dictionary_chat):
    chat = dictionary_chat
    assign(chat)
    row = chat.db.get_conversation_settings(chat.cid)
    settings = deepcopy(row["settings"])
    values = deepcopy(settings["roleplayBehaviorV1"]["values"])
    values["dictionaries"] = {"root_ids": [chat.dictionary_id], "dictionaries": "unreadable"}
    # Canonical materialized envelope is valid; its executable nested plan is not.
    settings["roleplayBehaviorV1"] = build_materialized_behavior_settings(values)
    assert chat.db.upsert_conversation_settings(chat.cid, settings, expected_settings_version=row["settings_version"])  # nosec B101
    before = chat.db.get_conversation_settings(chat.cid)
    response = chat.client.post(f"/api/v1/chat/conversations/{chat.cid}/history/selection", headers=chat.headers,
                               json={"purpose": "send", "view": {"view_session_id": "invalid-plan", "conversation_id": chat.cid,
                                   "interpretation": {"kind": "parent_graph_v1"}, "cursor": {"kind": "after_message", "message_id": chat.tail},
                                   "selection_revision": 1}})
    assert response.status_code == 409, response.text  # nosec B101
    assert chat.db.get_conversation_settings(chat.cid) == before  # nosec B101
    assert chat.db.count_messages_for_conversation(chat.cid) == 2  # nosec B101


def legacy_assignment(chat, *, extra=None):
    """Represent pre-fix owner settings using the supported DB abstraction."""
    row = chat.db.get_conversation_settings(chat.cid)
    settings = deepcopy(row["settings"])
    settings.update({"chat_dictionary_ids": [chat.dictionary_id], "chat_dictionary_id": chat.dictionary_id, **(extra or {})})
    values = settings["roleplayBehaviorV1"]["values"]
    values.pop("dictionaries", None)
    values["behavior_controls"]["applied_overrides"].pop("chat_dictionary_ids", None)
    values["behavior_controls"]["applied_overrides"].pop("chat_dictionary_id", None)
    settings["roleplayBehaviorV1"] = build_materialized_behavior_settings(values)
    assert chat.db.upsert_conversation_settings(chat.cid, settings, expected_settings_version=row["settings_version"])  # nosec B101
    return chat.db.get_conversation_settings(chat.cid)


def test_existing_legacy_assignment_binds_once_before_capture_and_send(dictionary_chat):
    chat = dictionary_chat
    old = legacy_assignment(chat)
    selection = capture(chat)
    bound = chat.db.get_conversation_settings(chat.cid)
    assert bound["settings_version"] == old["settings_version"] + 1  # nosec B101
    _, _, messages, _ = chat.client.portal.call(prepare, chat, selection)
    assert messages[-1]["content"] == [{"type": "text", "text": "Echo UAT_OUT_742; ORBIT-742."}]  # nosec B101
    assert chat.db.get_conversation_settings(chat.cid) == bound  # nosec B101


def test_unsupported_legacy_behavior_does_not_bind_or_overwrite_settings(dictionary_chat):
    chat = dictionary_chat
    old = legacy_assignment(chat, extra={"unknown_prompt_effect": True})
    response = chat.client.post(f"/api/v1/chat/conversations/{chat.cid}/history/selection", headers=chat.headers,
                               json={"purpose": "send", "view": {"view_session_id": "unsupported", "conversation_id": chat.cid,
                                   "interpretation": {"kind": "parent_graph_v1"}, "cursor": {"kind": "after_message", "message_id": chat.tail},
                                   "selection_revision": 1}})
    assert response.status_code == 409, response.text  # nosec B101
    assert chat.db.get_conversation_settings(chat.cid) == old  # nosec B101
    assert chat.db.count_messages_for_conversation(chat.cid) == 2  # nosec B101


def test_invalid_legacy_cursor_does_not_bind_settings(dictionary_chat):
    chat = dictionary_chat
    old = legacy_assignment(chat)
    response = chat.client.post(f"/api/v1/chat/conversations/{chat.cid}/history/selection", headers=chat.headers,
                               json={"purpose": "send", "view": {"view_session_id": "stale-view", "conversation_id": chat.cid,
                                   "interpretation": {"kind": "parent_graph_v1"}, "cursor": {"kind": "after_message", "message_id": "missing"},
                                   "selection_revision": 1}})
    assert response.status_code == 409 or response.json()["status"] != "captured"  # nosec B101
    assert chat.db.get_conversation_settings(chat.cid) == old  # nosec B101


def test_legacy_materialization_settings_cas_refuses_concurrent_assignment(dictionary_chat, monkeypatch):
    chat = dictionary_chat
    old = legacy_assignment(chat)
    upsert = chat.db.upsert_conversation_settings

    def intervening_assignment(cid, settings, **kwargs):
        changed = {**old["settings"], "chat_dictionary_ids": []}
        upsert(cid, changed, **kwargs)
        return upsert(cid, settings, **kwargs)

    monkeypatch.setattr(chat.db, "upsert_conversation_settings", intervening_assignment)
    response = chat.client.post(f"/api/v1/chat/conversations/{chat.cid}/history/selection", headers=chat.headers,
                               json={"purpose": "send", "view": {"view_session_id": "cas-race", "conversation_id": chat.cid,
                                   "interpretation": {"kind": "parent_graph_v1"}, "cursor": {"kind": "after_message", "message_id": chat.tail},
                                   "selection_revision": 1}})
    assert response.status_code == 409, response.text  # nosec B101
    assert response.json()["detail"]["code"] == "stale_settings"  # nosec B101
    assert chat.db.get_conversation_settings(chat.cid) == old  # nosec B101
    assert chat.db.count_messages_for_conversation(chat.cid) == 2  # nosec B101


@pytest.mark.parametrize("writer", ["singleton", "context"])
def test_supported_settings_writers_materialize_assigned_dictionary(dictionary_chat, writer):
    chat = dictionary_chat
    patch = ({"chat_dictionary_id": chat.dictionary_id} if writer == "singleton" else
             {"conversationContext": {"chat_dictionary_ids": [chat.dictionary_id], "world_book_ids": []}})
    response = chat.client.put(f"/api/v1/chats/{chat.cid}/settings?scope_type=global",
                               headers=chat.headers, json={"settings": patch})
    assert response.status_code == 200, response.text  # nosec B101
    _, _, messages, _ = chat.client.portal.call(prepare, chat, capture(chat))
    assert messages[-1]["content"] == [{"type": "text", "text": "Echo UAT_OUT_742; ORBIT-742."}]  # nosec B101


def test_native_frozen_default_budget_matches_real_process_endpoint(dictionary_chat):
    chat = dictionary_chat
    base = new_dictionary(chat, "Budget base", "UAT_IN_742", "LONG REPLACEMENT", default_token_budget=1)
    root = new_dictionary(chat, "Budget root", "LONG", "MUST_NOT_RUN", included_dictionary_ids=[base],
                          default_token_budget=1000)
    authored = "UAT_IN_742 UAT_IN_742"
    ordinary = chat.client.post("/api/v1/chat/dictionaries/process", headers=chat.headers,
                                json={"dictionary_id": root, "text": authored})
    assert ordinary.status_code == 200, ordinary.text  # nosec B101
    assert ordinary.json()["token_budget_exceeded"] is True  # nosec B101
    assign(chat, [root])
    selection = capture(chat)
    _, _, messages, runtime = chat.client.portal.call(partial(prepare, chat, selection, text=authored))
    assert messages[-1]["content"] == [{"type": "text", "text": ordinary.json()["processed_text"]}]  # nosec B101
    assert chat.db.get_message_by_id(runtime["user_message_id"])["content"] == authored  # nosec B101


def test_oversized_assignment_refuses_atomic_settings_write(dictionary_chat):
    from tldw_Server_API.app.core.Character_Chat.character_behavior_snapshot import DEFAULT_MAX_SNAPSHOT_BYTES

    chat = dictionary_chat
    response = chat.client.put(f"/api/v1/chat/dictionaries/entries/{chat.entry_id}", headers=chat.headers,
                               json={"replacement": "x" * DEFAULT_MAX_SNAPSHOT_BYTES})
    assert response.status_code == 200, response.text  # nosec B101
    before = chat.db.get_conversation_settings(chat.cid)
    response = chat.client.put(f"/api/v1/chats/{chat.cid}/settings?scope_type=global", headers=chat.headers,
                               json={"settings": {"chat_dictionary_ids": [chat.dictionary_id]}})
    assert response.status_code == 400, response.text  # nosec B101
    assert chat.db.get_conversation_settings(chat.cid) == before  # nosec B101
    assert chat.db.count_messages_for_conversation(chat.cid) == 2  # nosec B101


def test_unavailable_legacy_assignment_refuses_without_normalization(dictionary_chat):
    chat = dictionary_chat
    before = legacy_assignment(chat)
    response = chat.client.delete(f"/api/v1/chat/dictionaries/{chat.dictionary_id}", headers=chat.headers)
    assert response.status_code == 204, response.text  # nosec B101
    response = chat.client.post(f"/api/v1/chat/conversations/{chat.cid}/history/selection", headers=chat.headers,
                               json={"purpose": "send", "view": {"view_session_id": "missing-source", "conversation_id": chat.cid,
                                   "interpretation": {"kind": "parent_graph_v1"}, "cursor": {"kind": "after_message", "message_id": chat.tail},
                                   "selection_revision": 1}})
    assert response.status_code == 409, response.text  # nosec B101
    assert chat.db.get_conversation_settings(chat.cid) == before  # nosec B101


@pytest.mark.parametrize("malformed", ["envelope", "binding", "controls"])
def test_malformed_legacy_authority_is_not_replaced_by_compatibility_binding(dictionary_chat, malformed):
    chat = dictionary_chat
    row = legacy_assignment(chat)
    settings = deepcopy(row["settings"])
    authority = settings["roleplayBehaviorV1"]
    if malformed == "envelope":
        authority["digest"] = "sha256:" + "0" * 64
    else:
        values = authority["values"]
        if malformed == "binding":
            values["base_snapshot"]["digest"] = "sha256:" + "0" * 64
        else:
            values["behavior_controls"]["prompt_context"] = ["invalid"]
        settings["roleplayBehaviorV1"] = build_materialized_behavior_settings(values)
    assert chat.db.upsert_conversation_settings(chat.cid, settings, expected_settings_version=row["settings_version"])  # nosec B101
    before = chat.db.get_conversation_settings(chat.cid)
    response = chat.client.post(f"/api/v1/chat/conversations/{chat.cid}/history/selection", headers=chat.headers,
                               json={"purpose": "send", "view": {"view_session_id": "malformed-legacy", "conversation_id": chat.cid,
                                   "interpretation": {"kind": "parent_graph_v1"}, "cursor": {"kind": "after_message", "message_id": chat.tail},
                                   "selection_revision": 1}})
    assert response.status_code == 409, response.text  # nosec B101
    assert chat.db.get_conversation_settings(chat.cid) == before  # nosec B101
    assert chat.db.count_messages_for_conversation(chat.cid) == 2  # nosec B101


def dictionary_assignment_patch(chat, writer):
    if writer == "singleton":
        return {"chat_dictionary_id": chat.dictionary_id}
    if writer == "context":
        return {"conversationContext": {"chat_dictionary_ids": [chat.dictionary_id], "world_book_ids": []}}
    return {"chat_dictionary_ids": [chat.dictionary_id]}


@pytest.mark.parametrize("writer", ["list", "singleton", "context"])
@pytest.mark.parametrize("intent,stamp,accepted", [
    ("older", "2099-01-01T00:00:00Z", False),
    ("equal", "2099-01-02T00:00:00Z", False),
    ("newer", "2099-01-03T00:00:00Z", True),
    ("untimestamped", None, True),
])
def test_only_accepted_assignment_intent_refreshes_frozen_catalog(dictionary_chat, writer, intent, stamp, accepted):
    """Discarded API patches cannot import an edit from the live dictionary."""
    chat = dictionary_chat
    initial = {**dictionary_assignment_patch(chat, writer), "updatedAt": "2099-01-02T00:00:00Z"}
    response = chat.client.put(f"/api/v1/chats/{chat.cid}/settings?scope_type=global",
                               headers=chat.headers, json={"settings": initial})
    assert response.status_code == 200, response.text  # nosec B101 - real settings writer contract
    retained = deepcopy(chat.db.get_conversation_settings(chat.cid)["settings"]["roleplayBehaviorV1"]["values"]["dictionaries"])
    response = chat.client.put(f"/api/v1/chat/dictionaries/entries/{chat.entry_id}", headers=chat.headers,
                               json={"replacement": "ACCEPTED_REASSIGNMENT"})
    assert response.status_code == 200, response.text  # nosec B101 - actual catalog edit
    patch = dictionary_assignment_patch(chat, writer)
    if stamp is not None:
        patch["updatedAt"] = stamp
    response = chat.client.put(f"/api/v1/chats/{chat.cid}/settings?scope_type=global",
                               headers=chat.headers, json={"settings": patch})
    assert response.status_code == 200, response.text  # nosec B101 - no blanket stale-write refusal
    saved = chat.db.get_conversation_settings(chat.cid)["settings"]
    if not accepted:
        assert saved["roleplayBehaviorV1"]["values"]["dictionaries"] == retained, intent  # nosec B101 - rejected intent preserves authority
        assert saved["updatedAt"] == initial["updatedAt"]  # nosec B101 - existing server-wins rule
    _, _, messages, runtime = chat.client.portal.call(prepare, chat, capture(chat))
    replacement = "ACCEPTED_REASSIGNMENT" if accepted else "UAT_OUT_742"
    assert messages[-1]["content"] == [{"type": "text", "text": f"Echo {replacement}; ORBIT-742."}]  # nosec B101 - actual frozen provider projection
    assert chat.db.get_message_by_id(runtime["user_message_id"])["content"] == TEXT  # nosec B101 - authored input unchanged


@pytest.mark.parametrize("writer", ["list", "singleton", "context"])
def test_older_assignment_to_absent_field_retains_existing_merge_contract(dictionary_chat, writer):
    """Server-wins merges still accept keys the saved settings never contained."""
    chat = dictionary_chat
    response = chat.client.put(f"/api/v1/chats/{chat.cid}/settings?scope_type=global", headers=chat.headers,
                               json={"settings": {"updatedAt": "2099-01-02T00:00:00Z"}})
    assert response.status_code == 200, response.text  # nosec B101 - establish newer server timestamp
    response = chat.client.put(f"/api/v1/chats/{chat.cid}/settings?scope_type=global", headers=chat.headers,
                               json={"settings": {**dictionary_assignment_patch(chat, writer), "updatedAt": "2099-01-01T00:00:00Z"}})
    assert response.status_code == 200, response.text  # nosec B101 - accepted previously absent field
    _, _, messages, _ = chat.client.portal.call(prepare, chat, capture(chat))
    assert messages[-1]["content"] == [{"type": "text", "text": "Echo UAT_OUT_742; ORBIT-742."}]  # nosec B101 - accepted assignment is materialized
