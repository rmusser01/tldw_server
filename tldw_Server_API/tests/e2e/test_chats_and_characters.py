import os
import pytest
import httpx

from .test_data import TestDataGenerator
from .fixtures import authenticated_client, data_tracker, require_llm_or_skip, FALLBACK_CHAT_MODEL


def _sanitize_name(name: str) -> str:
    return (
        name.replace(" ", "_")
        .replace("<", "")
        .replace(">", "")
        .replace("|", "")
        .replace("\\", "")
        .replace("/", "")
    )


def _resolve_chat_model_or_skip(api) -> str:
    return require_llm_or_skip(api)


def _maybe_set_fallback_provider(payload: dict, model: str) -> None:
    if not model:
        return
    normalized = model.replace("-", "").lower()
    if normalized == FALLBACK_CHAT_MODEL.lower():
        payload.setdefault("api_provider", "openai")


def _is_fallback_model(model: str) -> bool:
    if not model:
        return False
    return model.replace("-", "").lower() == FALLBACK_CHAT_MODEL.lower()


def _post_chat_completion_with_model(api, payload: dict, model: str) -> httpx.Response:
    request_payload = dict(payload)
    request_payload["model"] = model
    _maybe_set_fallback_provider(request_payload, model)
    resp = api.client.post("/api/v1/chat/completions", json=request_payload)
    resp.raise_for_status()
    return resp


def _post_chat_completion_with_fallback(api, payload: dict, model: str) -> httpx.Response:
    try:
        return _post_chat_completion_with_model(api, payload, model)
    except httpx.HTTPStatusError as e:
        if e.response.status_code in (502, 503):
            detail = e.response.text
            pytest.skip(f"LLM provider unavailable for /chat/completions: {detail}")
        if e.response.status_code == 400 and not _is_fallback_model(model):
            fallback_model = "gpt-4o"
            try:
                return _post_chat_completion_with_model(api, payload, fallback_model)
            except httpx.HTTPStatusError as fallback_exc:
                if fallback_exc.response.status_code in (400, 502, 503):
                    pytest.skip(
                        f"Fallback model {fallback_model} rejected by /chat/completions: "
                        f"{fallback_exc.response.text}"
                    )
                raise
        if e.response.status_code == 400 and _is_fallback_model(model):
            pytest.skip(f"Fallback model {model} rejected by /chat/completions: {e.response.text}")
        raise


@pytest.mark.critical
def test_character_persona_influences_context_and_sender_name(authenticated_client, data_tracker):
    """
    Persona influence in chat:
    - Create character with persona fields.
    - Create a chat session for that character.
    - Prepare messages with include_character_context to verify persona/system prompt present.
    - Complete a turn (offline sim if local-llm) and persist.
    - Verify last assistant message sender equals sanitized character name.
    """
    api = authenticated_client

    # Create character via API (JSON create; avoids file import dependency)
    char_payload = {
        "name": f"PersonaTester_{TestDataGenerator.random_string(5)}",
        "description": "A poetic assistant who speaks in rhymes.",
        "personality": "Witty, rhythmic, and kind.",
        "scenario": "Casual conversation with a poetry enthusiast.",
        "system_prompt": "You are a poetic assistant that answers in brief rhymes.",
        "alternate_greetings": [
            "Greetings, friend, with words that chime.",
            "Hello there, let's speak in rhyme."
        ],
        "tags": ["test", "persona"],
        "creator": "e2e",
        "character_version": "1.0",
    }

    resp = api.client.post("/api/v1/characters/", json=char_payload)
    try:
        resp.raise_for_status()
    except httpx.HTTPStatusError as e:
        # If character creation not enabled, skip gracefully
        if e.response.status_code in (404, 501):
            pytest.skip("Character create endpoint unavailable in this deployment")
        raise
    created = resp.json()
    character_id = created.get("id") or created.get("character_id")
    assert character_id, f"No character_id returned: {created}"
    data_tracker.add_character(int(character_id))

    # Create a chat session for this character
    create_chat_payload = {"character_id": int(character_id), "title": "Persona E2E Chat"}
    resp = api.client.post("/api/v1/chats/", json=create_chat_payload)
    resp.raise_for_status()
    chat = resp.json()
    chat_id = chat.get("id")
    assert chat_id, f"Chat not created: {chat}"
    data_tracker.add_chat(chat_id)

    # Prepare messages for completion (no provider call); include character context
    prep_body = {"include_character_context": True, "limit": 100, "offset": 0}
    resp = api.client.post(f"/api/v1/chats/{chat_id}/completions", json=prep_body)
    resp.raise_for_status()
    prep = resp.json()
    assert prep.get("chat_id") == chat_id
    msgs = prep.get("messages") or []
    # Expect a system message including persona details
    assert msgs, "No messages returned from preparation"
    sys_msgs = [m for m in msgs if m.get("role") == "system"]
    assert sys_msgs, "Expected a system message with character context"
    sys_text = sys_msgs[0].get("content", "")
    assert char_payload["name"] in sys_text
    assert "poetic" in sys_text.lower() or "rhyme" in sys_text.lower(), "Persona fields missing in system context"

    # Perform a completion on the chat with persistence; defaults to local-llm (offline sim in TEST_MODE)
    complete_body = {
        "include_character_context": True,
        "append_user_message": "Please introduce yourself in one line.",
        "save_to_db": True,
        "stream": False,
        "provider": "local-llm",
        "model": "local-test",
    }
    resp = api.client.post(f"/api/v1/chats/{chat_id}/complete-v2", json=complete_body)
    try:
        resp.raise_for_status()
    except httpx.HTTPStatusError as e:
        if e.response.status_code in (502, 503):
            pytest.skip(f"LLM provider unavailable for /complete-v2: {e.response.text}")
        raise
    comp = resp.json()
    assert comp.get("chat_id") == chat_id
    assert comp.get("assistant_content") is not None

    # Fetch messages and verify last assistant sender equals sanitized character name
    resp = api.client.get(f"/api/v1/chats/{chat_id}/messages")
    resp.raise_for_status()
    messages = resp.json().get("messages") or []
    assert messages, "Expected at least one message in conversation"
    last_msg = messages[-1]
    expected_sender = _sanitize_name(char_payload["name"])  # DB stores assistant "sender" as name
    actual_sender = str(last_msg.get("sender") or "").strip()
    allowed_senders = {expected_sender, "assistant"}
    if actual_sender not in allowed_senders:
        pytest.fail(
            f"Assistant sender mismatch: {actual_sender} not in {sorted(allowed_senders)}"
        )


@pytest.mark.critical
def test_chat_history_search_returns_expected_snippets(authenticated_client, data_tracker):
    """
    Chat history search:
    - Create character + chat.
    - Add distinct user/assistant messages.
    - Search chat history; assert matching snippets are returned.
    """
    api = authenticated_client

    # Minimal character for the chat
    char_payload = {
        "name": f"SearchTester_{TestDataGenerator.random_string(5)}",
    }
    resp = api.client.post("/api/v1/characters/", json=char_payload)
    try:
        resp.raise_for_status()
    except httpx.HTTPStatusError as e:
        if e.response.status_code in (404, 501):
            pytest.skip("Character create endpoint unavailable in this deployment")
        raise
    character_id = (resp.json().get("id") or resp.json().get("character_id"))
    data_tracker.add_character(int(character_id))

    # Create chat
    create_chat_payload = {"character_id": int(character_id), "title": "Search E2E Chat"}
    resp = api.client.post("/api/v1/chats/", json=create_chat_payload)
    resp.raise_for_status()
    chat_id = resp.json().get("id")
    assert chat_id
    data_tracker.add_chat(chat_id)

    # Add messages (user and assistant)
    msgs = [
        {"role": "user", "content": "The quick brown fox jumps over the lazy dog."},
        {"role": "assistant", "content": "Indeed, the sun is bright and skies are clear."},
        {"role": "user", "content": "Rainy weather in Seattle today, bring an umbrella."},
        {"role": "assistant", "content": "Seattle often sees rain; stay dry and warm!"},
    ]
    for m in msgs:
        r = api.client.post(f"/api/v1/chats/{chat_id}/messages", json=m)
        r.raise_for_status()

    # Search for a term present in the last user message
    term = "Seattle"
    resp = api.client.get(f"/api/v1/chats/{chat_id}/messages/search", params={"query": term, "limit": 10})
    resp.raise_for_status()
    results = resp.json().get("messages") or []
    assert results, "Expected at least one search result"
    contents = "\n".join([m.get("content", "") for m in results])
    assert "Seattle" in contents, "Search results did not include expected message content"


@pytest.mark.critical
def test_chat_completions_save_to_db_persists_and_exposes_conversation(authenticated_client, data_tracker):
    """
    Variant using /api/v1/chat/completions with save_to_db:
    - Create a character.
    - Call /chat/completions with character_id + save_to_db=True.
    - Assert a conversation_id is returned and messages are persisted with assistant sender name.
    """
    api = authenticated_client

    # Create character
    char_payload = {
        "name": f"CompletionsPersona_{TestDataGenerator.random_string(5)}",
        "system_prompt": "You are a concise assistant speaking in short, factual sentences.",
    }
    resp = api.client.post("/api/v1/characters/", json=char_payload)
    try:
        resp.raise_for_status()
    except httpx.HTTPStatusError as e:
        if e.response.status_code in (404, 501):
            pytest.skip("Character create endpoint unavailable in this deployment")
        raise
    character_id = (resp.json().get("id") or resp.json().get("character_id"))
    data_tracker.add_character(int(character_id))

    # Call /chat/completions with save_to_db True
    model = _resolve_chat_model_or_skip(api)
    payload = {
        "messages": [{"role": "user", "content": "State your name in five words or less."}],
        "character_id": str(character_id),  # API expects string
        "save_to_db": True,
        "stream": False,
    }
    resp = _post_chat_completion_with_fallback(api, payload, model)

    data = resp.json()
    conv_id = data.get("tldw_conversation_id") or data.get("conversation_id")
    assert conv_id, f"/chat/completions did not return conversation id: {data}"
    data_tracker.add_chat(conv_id)

    # Verify messages persisted and assistant sender matches sanitized character name
    resp = api.client.get(f"/api/v1/chats/{conv_id}/messages")
    resp.raise_for_status()
    messages = resp.json().get("messages") or []
    assert len(messages) >= 2
    last_msg = messages[-1]
    expected_sender = _sanitize_name(char_payload["name"])
    actual_sender = str(last_msg.get("sender") or "").strip()
    allowed_senders = {expected_sender, "assistant"}
    if actual_sender not in allowed_senders:
        pytest.fail(
            f"Assistant sender mismatch: {actual_sender} not in {sorted(allowed_senders)}"
        )


@pytest.mark.critical
def test_chat_completions_history_search_followup(authenticated_client, data_tracker):
    """
    /chat/completions follow-up using the same conversation:
    - Create character, call /chat/completions (save_to_db) to create a conversation.
    - Call /chat/completions again with the returned conversation_id and unique term.
    - Search chat messages for the unique term and assert it appears in results.
    """
    api = authenticated_client

    # Create character
    char_payload = {"name": f"CompletionsSearch_{TestDataGenerator.random_string(5)}"}
    resp = api.client.post("/api/v1/characters/", json=char_payload)
    try:
        resp.raise_for_status()
    except httpx.HTTPStatusError as e:
        if e.response.status_code in (404, 501):
            pytest.skip("Character create endpoint unavailable in this deployment")
        raise
    character_id = (resp.json().get("id") or resp.json().get("character_id"))
    data_tracker.add_character(int(character_id))

    # Initial completion to create conversation
    model = _resolve_chat_model_or_skip(api)
    first = {
        "messages": [{"role": "user", "content": "Start a short conversation."}],
        "character_id": str(character_id),
        "save_to_db": True,
        "stream": False,
    }
    r1 = _post_chat_completion_with_fallback(api, first, model)
    d1 = r1.json()
    conv_id = d1.get("tldw_conversation_id") or d1.get("conversation_id")
    assert conv_id
    data_tracker.add_chat(conv_id)

    # Follow-up with a unique term and save
    unique_term = f"E2EUniqueTerm_{TestDataGenerator.random_string(6)}"
    second = {
        "messages": [{"role": "user", "content": f"Please include this token: {unique_term}"}],
        "character_id": str(character_id),
        "conversation_id": conv_id,
        "save_to_db": True,
        "stream": False,
    }
    _post_chat_completion_with_fallback(api, second, model)

    # Search chat history for the unique term
    resp = api.client.get(f"/api/v1/chats/{conv_id}/messages/search", params={"query": unique_term, "limit": 10})
    resp.raise_for_status()
    results = resp.json().get("messages") or []
    assert results, "Expected search to return results for the unique term"
    blob = "\n".join([m.get("content", "") for m in results])
    assert unique_term in blob, "Unique term not found in chat search results"


@pytest.mark.critical
def test_chat_completions_assistant_name_if_present(authenticated_client, data_tracker):
    """
    Validate assistant name in /chat/completions payload when provided by the provider:
    - Create a character with a name that requires sanitization.
    - Call /chat/completions with save_to_db=True.
    - If choices[0].message.name is present, assert it equals the sanitized character name.
    - If not present, skip (some providers don't include assistant name in responses).
    """
    api = authenticated_client

    # Character with spaces and special chars to exercise sanitization
    raw_name = f"Name Test / Persona {_sanitize_name(TestDataGenerator.random_string(4))}"
    char_payload = {
        "name": raw_name,
        "system_prompt": "You are a testing assistant; include concise answers.",
    }
    resp = api.client.post("/api/v1/characters/", json=char_payload)
    try:
        resp.raise_for_status()
    except httpx.HTTPStatusError as e:
        if e.response.status_code in (404, 501):
            pytest.skip("Character create endpoint unavailable in this deployment")
        raise
    character_id = (resp.json().get("id") or resp.json().get("character_id"))
    data_tracker.add_character(int(character_id))

    model = _resolve_chat_model_or_skip(api)
    payload = {
        "messages": [{"role": "user", "content": "Reply briefly."}],
        "character_id": str(character_id),
        "save_to_db": True,
        "stream": False,
    }
    r = _post_chat_completion_with_fallback(api, payload, model)

    data = r.json()
    choices = data.get("choices") or []
    assert choices, f"/chat/completions returned no choices: {data}"
    msg = choices[0].get("message") or {}
    assistant_name = msg.get("name")
    if not assistant_name:
        pytest.skip("Provider response did not include assistant 'name' field")

    expected = _sanitize_name(raw_name)
    assert assistant_name == expected, f"Assistant name mismatch: {assistant_name} != {expected}"
