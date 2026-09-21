"""Accepted failed Retry must end the actual adapter payload, without new writes."""
import asyncio
import base64
import io
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException
from PIL import Image

from tldw_Server_API.app.api.v1.schemas.chat_request_schemas import ChatCompletionRequest
from tldw_Server_API.app.core.Chat import chat_service
from tldw_Server_API.app.core.LLM_Calls.providers.custom_openai_adapter import CustomOpenAIAdapter
from tldw_Server_API.tests.Chat.unit.test_chat_history_and_streaming import DummyChatDB

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


class SavedDB(DummyChatDB):
    def __init__(self, records, kind="neutral"):
        super().__init__(records)
        self.kind = kind

    def get_conversation_by_id(self, _id):
        return {"id": "conv", "character_id": 1 if self.kind == "character" else None,
                "client_id": "client", "assistant_kind": "persona" if self.kind == "persona" else None,
                "assistant_id": "persona" if self.kind == "persona" else None}

    def get_message_metadata(self, message_id):
        return {"extra": {"client_message_id": "retry-turn"}} if message_id == "failed" else {}

    def get_persona_profile(self, _id, *, user_id):
        assert user_id == "client"
        return {"id": "persona", "name": "Persona", "system_prompt": "Be helpful."}


def records_for_retry():
    return [
        {"id": "first", "sender": "user", "content": "Who coordinates Cedar?", "timestamp": 1},
        {"id": "answer", "sender": "assistant", "content": "Jonah Patel.", "timestamp": 2},
        {"id": "second", "sender": "user", "content": "Greet the visitor.", "timestamp": 3},
        {"id": "second-answer", "sender": "assistant", "content": "Ahoy visitor!", "timestamp": 4},
        {"id": "failed", "sender": "user", "content": "Reply exactly: CEDAR RETRY READY.", "timestamp": 5},
    ]


async def provider_payload(request, records, kind="neutral"):
    save, state = AsyncMock(), {}
    result = await chat_service.build_context_and_messages(
        chat_db=SavedDB(records, kind), request_data=request, loop=asyncio.get_running_loop(),
        metrics=MagicMock(), default_save_to_db=False, final_conversation_id="conv",
        save_message_fn=save, runtime_state=state)
    system, messages = chat_service.apply_prompt_templating(request, result[0] or {}, result[4])
    system, messages = chat_service.inject_research_context_into_prompt(
        final_system_message=system, templated_llm_payload=messages, research_context=None)
    args = chat_service.build_call_params_from_request(
        request_data=request, target_api_provider="custom-openai-api",
        provider_api_key="synthetic-unused-key", templated_llm_payload=messages,
        final_system_message=system, app_config={}, resolved_model="synthetic")
    _, adapter_request, _ = chat_service._build_adapter_request_from_chat_args(args)
    payload = CustomOpenAIAdapter()._build_payload(adapter_request)
    return payload, save, state


def text_of(message):
    content = message["content"]
    return content if isinstance(content, str) else "".join(part.get("text", "") for part in content)


@pytest.mark.parametrize("order", ["asc", "desc"])
@pytest.mark.parametrize("full_history", [False, True])
@pytest.mark.parametrize("limit", [0, 1, 3, 20])
@pytest.mark.parametrize("kind", ["neutral", "persona", "character"])
async def test_failed_retry_is_last_once_in_final_provider_payload(order, full_history, limit, kind):
    records = records_for_retry()
    request = ChatCompletionRequest(
        model="synthetic", conversation_id="conv", save_to_db=True,
        history_message_order=order, history_message_limit=max(1, limit),
        messages=[{"role": row["sender"], "content": row["content"]}
                  for row in (records if full_history else records[-1:])],
        metadata={"tldw_retry_failed_turn": True, "tldw_client_message_id": "retry-turn"})
    if limit == 0:
        # The service supports zero internally; the public schema requires >=1.
        request = request.model_copy(update={"history_message_limit": 0})
    # Unmatched full client history must not bypass the existing unsaved-user guard.
    if full_history and (limit == 0 or (limit == 1 and order == "asc")):
        with pytest.raises(HTTPException) as error:
            await provider_payload(request, records, kind)
        assert error.value.status_code == 409
        return
    payload, save, state = await provider_payload(request, records, kind)
    non_system = [message for message in payload["messages"] if message["role"] != "system"]
    assert text_of(non_system[-1]) == records[-1]["content"]
    assert sum(text_of(message) == records[-1]["content"] for message in non_system) == 1
    ordered = records if order == "asc" else list(reversed(records))
    expected_prefix = [row["content"] for row in ordered[:limit] if row["id"] != "failed"]
    if full_history and order == "asc" and limit == 3:
        expected_prefix.append(records[3]["content"])
    assert [text_of(message) for message in non_system[:-1]] == expected_prefix
    assert state["user_message_id"] == "failed"
    assert save.await_count == (1 if full_history and order == "asc" and limit == 3 else 0)
    assert all(call.args[2]["role"] != "user" for call in save.await_args_list)
    assert "tldw_retry_failed_turn" not in str(payload)


@pytest.mark.parametrize("order", ["asc", "desc"])
@pytest.mark.parametrize("kind", ["neutral", "persona", "character"])
async def test_retry_reposition_preserves_literal_images_and_filters_saved_error(order, kind):
    records = records_for_retry()
    records[-1]["content"] = "Literal {{char}} question"
    buffer = io.BytesIO()
    Image.new("RGB", (2, 2), "red").save(buffer, format="PNG")
    raw = buffer.getvalue()
    image_url = "data:image/png;base64," + base64.b64encode(raw).decode()
    records[-1]["images"] = [{"image_data": raw, "image_mime_type": "image/png"}]
    records.append({"id": "error", "sender": "assistant", "timestamp": 6,
                    "content": '__tldw_error__:{"summary":"Failed","hint":"Retry"}'})
    request = ChatCompletionRequest(model="synthetic", conversation_id="conv", save_to_db=True,
        history_message_order=order, messages=[{"role": "user", "content": [
            {"type": "text", "text": "Literal {{char}} question"},
            {"type": "image_url", "image_url": {"url": image_url}}]}],
        metadata={"tldw_retry_failed_turn": True, "tldw_client_message_id": "retry-turn"})
    payload, save, state = await provider_payload(request, records, kind)
    final = payload["messages"][-1]
    expected = "Literal Assistant question" if kind == "character" else "Literal {{char}} question"
    assert text_of(final) == expected
    assert [part["image_url"]["url"] for part in final["content"] if part["type"] == "image_url"] == [image_url]
    assert "__tldw_error__" not in str(payload)
    assert state["user_message_id"] == "failed"
    save.assert_not_awaited()


@pytest.mark.parametrize("retry", [False, True])
async def test_retry_repositions_by_identity_without_deduplicating_equal_turns(retry):
    records = records_for_retry()
    records[-1]["content"] = records[0]["content"]
    request = ChatCompletionRequest(model="synthetic", conversation_id="conv", save_to_db=True,
        history_message_order="desc", messages=[{"role": "user", "content": records[-1]["content"]}],
        metadata={"tldw_retry_failed_turn": retry})
    payload, save, state = await provider_payload(request, records)
    messages = [message for message in payload["messages"] if message["role"] != "system"]
    historical = list(reversed(records[:-1] if retry else records))
    assert [text_of(message) for message in messages] == [row["content"] for row in historical] + [records[-1]["content"]]
    assert sum(text_of(message) == records[-1]["content"] for message in messages) == (2 if retry else 3)
    assert save.await_count == (0 if retry else 1)
    if retry:
        assert state["user_message_id"] == "failed"
