import asyncio
import json
from contextlib import asynccontextmanager, nullcontext
from typing import Any, Dict, Optional, List

import pytest
from loguru import logger
from unittest.mock import AsyncMock, MagicMock

pytestmark = pytest.mark.unit

from tldw_Server_API.app.core.Chat import chat_service
from tldw_Server_API.app.core.Chat.Chat_Deps import ChatAuthenticationError
from tldw_Server_API.app.core.Chat.streaming_utils import StreamingResponseHandler


class DummyRequestData:
    def __init__(self, history_message_limit=None, history_message_order=None, save_to_db=False):
        self.character_id = None
        self.history_message_limit = history_message_limit
        self.history_message_order = history_message_order
        self.save_to_db = save_to_db
        self.messages: List[Any] = []


class DummyChatDB:
    def __init__(self, records: List[Dict[str, Any]]):
        self._records = records
        self.client_id = "client"

    def get_messages_for_conversation(self, conversation_id: str, limit: int, offset: int, order: str):
        assert conversation_id == "conv"
        assert offset == 0
        assert order in ("ASC", "DESC")
        ordered = sorted(self._records, key=lambda r: r["timestamp"], reverse=(order == "DESC"))
        return ordered[:limit]

    def get_character_card_by_name(self, name):

        return {"id": 1, "name": name, "system_prompt": "Prompt"}

    def get_character_card_by_id(self, char_id):

        return {"id": char_id, "name": "Assistant", "system_prompt": "Prompt"}

    def get_message_metadata(self, _message_id: str):
        return {}

    def get_connection(self):

        raise RuntimeError("not needed")


class DummyChatDBWithMetadata(DummyChatDB):
    def get_message_metadata(self, _message_id: str):
        return {
            "tool_calls": None,
            "extra": {
                "sender_role": "system",
                "sender_name": "system-command",
            },
        }


@pytest.mark.parametrize("content", [["literal"], [None]])
def test_chat_error_envelope_leaves_malformed_content_parts_unchanged(content):
    assert chat_service._is_saved_chat_error_envelope({"role": "assistant", "content": content}) is False


@pytest.mark.asyncio
async def test_build_context_uses_history_knobs():
    records = [
        {
            "id": f"msg-{idx}",
            "sender": "user" if idx % 2 == 0 else "assistant",
            "content": f"text-{idx}",
            "timestamp": idx,
            "images": [],
        }
        for idx in range(10)
    ]
    db = DummyChatDB(records)
    request_data = DummyRequestData(history_message_limit=3, history_message_order="desc")
    request_data.messages = []
    loop = asyncio.get_running_loop()

    class DummyMetrics:
        def track_character_access(self, *args, **kwargs):
            pass

        def track_conversation(self, *args, **kwargs):

            pass

    result = await chat_service.build_context_and_messages(
        chat_db=db,
        request_data=request_data,
        loop=loop,
        metrics=DummyMetrics(),
        default_save_to_db=False,
        final_conversation_id="conv",
        save_message_fn=AsyncMock(),
    )

    history = result[4]
    assert len(history) == 3
    # History preserves requested order in the payload
    timestamps = [msg_part["content"][0]["text"].split("-")[-1] for msg_part in history if msg_part["content"]]
    assert timestamps == ["9", "8", "7"]


@pytest.mark.asyncio
async def test_build_context_history_limit_zero_skips_history():
    records = [
        {"id": f"msg-{idx}", "sender": "user", "content": f"text-{idx}", "timestamp": idx, "images": []}
        for idx in range(5)
    ]
    db = DummyChatDB(records)
    request_data = DummyRequestData(history_message_limit=0, history_message_order="asc")

    class DummyMessage:
        def __init__(self, role: str, content: Any):
            self.role = role
            self.content = content

        def model_dump(self, _exclude_none: bool = True, **_kwargs):
            return {"role": self.role, "content": self.content}

    request_data.messages = [DummyMessage("user", "current")]
    loop = asyncio.get_running_loop()

    class DummyMetrics:
        def track_character_access(self, *args, **kwargs):
            pass

        def track_conversation(self, *args, **kwargs):
            pass

    result = await chat_service.build_context_and_messages(
        chat_db=db,
        request_data=request_data,
        loop=loop,
        metrics=DummyMetrics(),
        default_save_to_db=False,
        final_conversation_id="conv",
        save_message_fn=AsyncMock(),
    )

    history = result[4]
    assert len(history) == 1
    assert history[0]["role"] == "user"


class DummySave:
    def __init__(self):
        self.calls: List[Dict[str, Any]] = []

    async def __call__(
        self,
        text: str,
        tool_calls: Optional[List[Dict[str, Any]]],
        function_call: Optional[Dict[str, Any]],
    ):
        self.calls.append({
            "text": text,
            "tool_calls": tool_calls,
            "function_call": function_call,
        })


@pytest.mark.asyncio
async def test_build_context_skips_tool_placeholder_replacement():
    records = [
        {
            "id": "msg-tool-1",
            "sender": "tool",
            "content": "{\"query\":\"{{char}}\"}",
            "timestamp": 1,
            "images": [],
        }
    ]
    class DummyChatDBWithToolId(DummyChatDB):
        def get_message_metadata(self, _message_id: str):
            return {
                "tool_calls": None,
                "extra": {
                    "tool_call_id": "tool-1",
                    "sender_role": "tool",
                },
            }

    db = DummyChatDBWithToolId(records)
    request_data = DummyRequestData(history_message_limit=5, history_message_order="asc")
    request_data.messages = []
    loop = asyncio.get_running_loop()

    class DummyMetrics:
        def track_character_access(self, *args, **kwargs):
            pass

        def track_conversation(self, *args, **kwargs):

            pass

    result = await chat_service.build_context_and_messages(
        chat_db=db,
        request_data=request_data,
        loop=loop,
        metrics=DummyMetrics(),
        default_save_to_db=False,
        final_conversation_id="conv",
        save_message_fn=AsyncMock(),
    )

    history = result[4]
    tool_msgs = [msg for msg in history if msg.get("role") == "tool"]
    assert tool_msgs
    assert tool_msgs[0]["content"] == "{\"query\":\"{{char}}\"}"


@pytest.mark.asyncio
async def test_build_context_skips_tool_message_without_tool_call_id():
    records = [
        {
            "id": "msg-tool-missing",
            "sender": "tool",
            "content": "{\"result\": 1}",
            "timestamp": 1,
            "images": [],
        }
    ]
    db = DummyChatDB(records)
    request_data = DummyRequestData(history_message_limit=5, history_message_order="asc")
    request_data.messages = []
    loop = asyncio.get_running_loop()

    class DummyMetrics:
        def track_character_access(self, *args, **kwargs):
            pass

        def track_conversation(self, *args, **kwargs):
            pass

    result = await chat_service.build_context_and_messages(
        chat_db=db,
        request_data=request_data,
        loop=loop,
        metrics=DummyMetrics(),
        default_save_to_db=False,
        final_conversation_id="conv",
        save_message_fn=AsyncMock(),
    )

    history = result[4]
    assert all(msg.get("role") != "tool" for msg in history)


@pytest.mark.asyncio
async def test_build_context_keeps_tool_message_with_tool_call_id():
    class DummyChatDBWithToolId(DummyChatDB):
        def get_message_metadata(self, _message_id: str):
            return {
                "tool_calls": None,
                "extra": {
                    "tool_call_id": "tool-1",
                    "sender_role": "tool",
                },
            }

    records = [
        {
            "id": "msg-tool-1",
            "sender": "tool",
            "content": "{\"result\": 1}",
            "timestamp": 1,
            "images": [],
        }
    ]
    db = DummyChatDBWithToolId(records)
    request_data = DummyRequestData(history_message_limit=5, history_message_order="asc")
    request_data.messages = []
    loop = asyncio.get_running_loop()

    class DummyMetrics:
        def track_character_access(self, *args, **kwargs):
            pass

        def track_conversation(self, *args, **kwargs):
            pass

    result = await chat_service.build_context_and_messages(
        chat_db=db,
        request_data=request_data,
        loop=loop,
        metrics=DummyMetrics(),
        default_save_to_db=False,
        final_conversation_id="conv",
        save_message_fn=AsyncMock(),
    )

    history = result[4]
    tool_msgs = [msg for msg in history if msg.get("role") == "tool"]
    assert tool_msgs
    assert tool_msgs[0].get("tool_call_id") == "tool-1"


@pytest.mark.asyncio
async def test_build_context_honors_sender_role_metadata():
    records = [
        {
            "id": "msg-system-1",
            "sender": "system-command",
            "content": "System note",
            "timestamp": 1,
            "images": [],
        }
    ]
    db = DummyChatDBWithMetadata(records)
    request_data = DummyRequestData(history_message_limit=5, history_message_order="asc")
    request_data.messages = []
    loop = asyncio.get_running_loop()

    class DummyMetrics:
        def track_character_access(self, *args, **kwargs):
            pass

        def track_conversation(self, *args, **kwargs):
            pass

    result = await chat_service.build_context_and_messages(
        chat_db=db,
        request_data=request_data,
        loop=loop,
        metrics=DummyMetrics(),
        default_save_to_db=False,
        final_conversation_id="conv",
        save_message_fn=AsyncMock(),
    )

    history = result[4]
    assert history
    assert history[0]["role"] == "system"


@pytest.mark.asyncio
@pytest.mark.parametrize("history_order", ["asc", "desc"])
@pytest.mark.parametrize(
    "last_role,last_content,retry_content,expected_writes,expected_history_count",
    [
        ("assistant", '__tldw_error__:{"summary":"Failed","hint":"Retry"}', "question", 0, 1),
        ("assistant", '__tldw_error__:{"summary":"Failed","hint":"Retry"}', "different", 1, 2),
        ("assistant", "Successful answer", "question", 1, 3),
        ("assistant", '__tldw_error__:{"summary":"Quoted example"}', "question", 1, 3),
        ("assistant", "__tldw_error__:not-json", "question", 1, 3),
        ("user", '__tldw_error__:{"summary":"Failed","hint":"Retry"}', "question", 1, 3),
    ],
)
async def test_failed_turn_retry_preserves_history_without_replaying_diagnostics(
    history_order, last_role, last_content, retry_content, expected_writes, expected_history_count, monkeypatch
):
    from tldw_Server_API.app.api.v1.schemas.chat_request_schemas import ChatCompletionRequest

    monkeypatch.delenv("CHAT_TRIM_USER_ONLY_OVERLAP", raising=False)

    class SavedChatDB(DummyChatDB):
        def get_conversation_by_id(self, conversation_id):
            return {"id": conversation_id, "character_id": 1, "client_id": "client"}

    db = SavedChatDB([
        {"id": "question", "sender": "user", "content": "question", "timestamp": 1},
        {"id": "last", "sender": last_role, "content": last_content, "timestamp": 2},
    ])
    request = ChatCompletionRequest(
        model="test-model", save_to_db=True, history_message_order=history_order,
        messages=[{"role": "user", "content": retry_content}],
    )
    save = AsyncMock()
    result = await chat_service.build_context_and_messages(
        chat_db=db, request_data=request, loop=asyncio.get_running_loop(), metrics=MagicMock(),
        default_save_to_db=False, final_conversation_id="conv", save_message_fn=save,
    )

    assert len(result[4]) == expected_history_count
    assert save.await_count == expected_writes
    assert db._records[-1]["content"] == last_content


@pytest.mark.asyncio
@pytest.mark.parametrize("history_limit,history_order", [(1, "asc"), (20, "asc"), (20, "desc")])
async def test_explicit_failed_retry_uses_actual_tail_even_outside_context_window(history_limit, history_order):
    from tldw_Server_API.app.api.v1.schemas.chat_request_schemas import ChatCompletionRequest

    class SavedDB(DummyChatDB):
        def get_conversation_by_id(self, _id):
            return {"id": "conv", "character_id": 1, "client_id": "client"}

    records = [
        {"id": "earlier", "sender": "assistant", "content": "Earlier reply", "timestamp": 1},
        {"id": "failed-user", "sender": "user", "content": "Question for {{char}}", "timestamp": 2},
    ]
    request = ChatCompletionRequest(
        model="test", conversation_id="conv", save_to_db=True,
        history_message_limit=history_limit, history_message_order=history_order,
        messages=[{"role": "user", "content": records[-1]["content"]}],
        metadata={"tldw_retry_failed_turn": True},
    )
    save = AsyncMock()
    state = {}
    result = await chat_service.build_context_and_messages(
        chat_db=SavedDB(records), request_data=request, loop=asyncio.get_running_loop(), metrics=MagicMock(),
        default_save_to_db=False, final_conversation_id="conv", save_message_fn=save, runtime_state=state,
    )
    save.assert_not_awaited()
    assert state["user_message_id"] == "failed-user"
    assert sum(message["role"] == "user" for message in result[4]) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("marker", ["client-user", "", "x" * 129, {"unsafe": "object"}])
async def test_client_correlation_is_bounded_and_only_attached_to_final_persisted_user(marker):
    from tldw_Server_API.app.api.v1.schemas.chat_request_schemas import ChatCompletionRequest

    class SavedDB(DummyChatDB):
        def get_conversation_by_id(self, _id):
            return {"id": "conv", "character_id": 1, "client_id": "client"}

    request = ChatCompletionRequest(model="test", save_to_db=True, conversation_id="conv",
        messages=[{"role": "user", "content": "Earlier"}, {"role": "user", "content": "Current"}],
        metadata={"tldw_client_message_id": marker})
    save = AsyncMock(return_value="saved-user")
    result = await chat_service.build_context_and_messages(chat_db=SavedDB([]), request_data=request,
        loop=asyncio.get_running_loop(), metrics=MagicMock(), default_save_to_db=False,
        final_conversation_id="conv", save_message_fn=save)
    stored = [call.args[2] for call in save.await_args_list]
    assert "client_message_id" not in stored[0]
    assert stored[1].get("client_message_id") == (marker if marker == "client-user" else None)
    assert all("client_message_id" not in message for message in result[4])


@pytest.mark.asyncio
@pytest.mark.parametrize("extra", [None, {}, {"client_message_id": "same"}, {"client_message_id": "different"}])
async def test_retry_client_identity_respects_saved_correlation_and_legacy_metadata(extra):
    from fastapi import HTTPException

    from tldw_Server_API.app.api.v1.schemas.chat_request_schemas import ChatCompletionRequest

    class SavedDB(DummyChatDB):
        def get_conversation_by_id(self, _id):
            return {"id": "conv", "character_id": 1, "client_id": "client"}
        def get_message_metadata(self, _id):
            return {"extra": extra}

    request = ChatCompletionRequest(model="test", save_to_db=True, conversation_id="conv",
        messages=[{"role": "user", "content": "Question"}],
        metadata={"tldw_client_message_id": "same", "tldw_retry_failed_turn": True})
    save = AsyncMock()
    state = {}
    call = chat_service.build_context_and_messages(chat_db=SavedDB([
        {"id": "user", "sender": "user", "content": "Question", "timestamp": 1}]), request_data=request,
        loop=asyncio.get_running_loop(), metrics=MagicMock(), default_save_to_db=False,
        final_conversation_id="conv", save_message_fn=save, runtime_state=state)
    if extra == {"client_message_id": "different"}:
        with pytest.raises(HTTPException) as error:
            await call
        assert error.value.status_code == 409
    else:
        await call
        assert state["user_message_id"] == "user"
    save.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["matching-image", "changed-image", "answered", "legacy-error", "not-persisted", "ordinary-repeat"])
async def test_failed_retry_identity_conflict_and_compatibility_controls(kind):
    from fastapi import HTTPException

    from tldw_Server_API.app.api.v1.schemas.chat_request_schemas import ChatCompletionRequest

    class SavedDB(DummyChatDB):
        def get_conversation_by_id(self, _id):
            return {"id": "conv", "character_id": 1, "client_id": "client"}

    records = [{"id": "failed-user", "sender": "user", "content": "Question", "timestamp": 1}]
    content = "Question"
    if "image" in kind:
        records[0]["images"] = [{"image_data": b"image", "image_mime_type": "image/png"}]
        content = [{"type": "text", "text": "Question"}, {"type": "image_url", "image_url": {
            "url": "data:image/png;base64," + ("aW1hZ2U=" if kind == "matching-image" else "b3RoZXI=")
        }}]
    if kind in {"answered", "legacy-error"}:
        records.append({"id": "assistant", "sender": "assistant", "timestamp": 2,
                        "content": "Answer" if kind == "answered" else '__tldw_error__:{"summary":"Failed","hint":"Retry"}'})
    if kind == "not-persisted":
        records = []
    request = ChatCompletionRequest(
        model="test", save_to_db=True, conversation_id="conv", messages=[{"role": "user", "content": content}],
        metadata={"tldw_retry_failed_turn": kind != "ordinary-repeat"},
    )
    save = AsyncMock(return_value="new-user")
    state = {}
    call = chat_service.build_context_and_messages(
        chat_db=SavedDB(records), request_data=request, loop=asyncio.get_running_loop(), metrics=MagicMock(),
        default_save_to_db=False, final_conversation_id="conv", save_message_fn=save, runtime_state=state,
    )
    if kind in {"changed-image", "answered"}:
        with pytest.raises(HTTPException) as error:
            await call
        assert error.value.status_code == 409
        save.assert_not_awaited()
    else:
        await call
        assert state["user_message_id"] == ("new-user" if kind in {"ordinary-repeat", "not-persisted"} else "failed-user")
        assert save.await_count == (1 if kind in {"ordinary-repeat", "not-persisted"} else 0)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "owner,requested_character,expected_id,expected_created",
    [
        ("client", None, "conv", False),
        ("client", "2", "fork", True),
        ("other-client", None, "fork", True),
    ],
)
async def test_build_context_preserves_owned_neutral_conversation_only_for_implicit_assistant(
    owner, requested_character, expected_id, expected_created
):
    class NeutralChatDB(DummyChatDB):
        def get_conversation_by_id(self, conversation_id):
            return {
                "id": conversation_id,
                "character_id": None,
                "assistant_kind": None,
                "assistant_id": None,
                "client_id": owner,
            }

        def transaction(self):
            return nullcontext()

        def add_conversation(self, conversation):
            return "fork"

    request = DummyRequestData(save_to_db=True)
    request.character_id = requested_character
    result = await chat_service.build_context_and_messages(
        chat_db=NeutralChatDB([]),
        request_data=request,
        loop=asyncio.get_running_loop(),
        metrics=MagicMock(),
        default_save_to_db=False,
        final_conversation_id="conv",
        save_message_fn=AsyncMock(),
    )

    assert (result[2], result[3], result[5]) == (expected_id, expected_created, True)
    assert result[0]["system_prompt"] == "Prompt"


@pytest.mark.asyncio
async def test_build_context_persists_full_transcript_when_enabled():
    class DummyChatDBWithConversation(DummyChatDB):
        def get_conversation_by_id(self, conversation_id: str):
            return {"id": conversation_id, "character_id": 1, "client_id": self.client_id}

    class DummyMessage:
        def __init__(self, role: str, content: Any):
            self.role = role
            self.content = content

        def model_dump(self, _exclude_none: bool = True, **_kwargs):
            return {"role": self.role, "content": self.content}

    db = DummyChatDBWithConversation([])
    request_data = DummyRequestData(save_to_db=True)
    request_data.messages = [
        DummyMessage("user", "hi"),
        DummyMessage("assistant", "hello"),
        DummyMessage("tool", "{\"result\": 1}"),
    ]
    loop = asyncio.get_running_loop()

    class DummyMetrics:
        def track_character_access(self, *args, **kwargs):
            pass

        def track_conversation(self, *args, **kwargs):
            pass

    save_message_fn = AsyncMock()

    await chat_service.build_context_and_messages(
        chat_db=db,
        request_data=request_data,
        loop=loop,
        metrics=DummyMetrics(),
        default_save_to_db=False,
        final_conversation_id="conv",
        save_message_fn=save_message_fn,
    )

    roles = [call.args[2]["role"] for call in save_message_fn.call_args_list]
    assert roles == ["user", "assistant", "tool"]


@pytest.mark.asyncio
async def test_build_context_sanitizes_message_names():
    class DummyChatDBWithConversation(DummyChatDB):
        def get_conversation_by_id(self, conversation_id: str):
            return {"id": conversation_id, "character_id": 1, "client_id": self.client_id}

    class DummyMessage:
        def __init__(self, role: str, content: Any, name: Optional[str]):
            self.role = role
            self.content = content
            self.name = name

        def model_dump(self, _exclude_none: bool = True, **_kwargs):
            data = {"role": self.role, "content": self.content}
            if self.name is not None:
                data["name"] = self.name
            return data

    db = DummyChatDBWithConversation([])
    request_data = DummyRequestData(save_to_db=False)
    request_data.messages = [
        DummyMessage("user", "hi", "User Name/Bad"),
    ]
    loop = asyncio.get_running_loop()

    class DummyMetrics:
        def track_character_access(self, *args, **kwargs):
            pass

        def track_conversation(self, *args, **kwargs):
            pass

    result = await chat_service.build_context_and_messages(
        chat_db=db,
        request_data=request_data,
        loop=loop,
        metrics=DummyMetrics(),
        default_save_to_db=False,
        final_conversation_id="conv",
        save_message_fn=AsyncMock(),
    )

    llm_messages = result[4]
    assert llm_messages
    assert llm_messages[0]["name"] == "User_NameBad"


@pytest.mark.asyncio
async def test_streaming_handler_persists_tool_calls():
    handler = StreamingResponseHandler(
        conversation_id="conv",
        model_name="model",
        idle_timeout=30,
        heartbeat_interval=30,
        max_response_size=1024,
    )

    handler._accumulate_tool_calls([
        {
            "index": 0,
            "id": "call",
            "type": "function",
            "function": {"name": "foo", "arguments": "{\\\"a\\\": "},
        }
    ])
    handler._accumulate_tool_calls([
        {
            "index": 0,
            "function": {"arguments": "1\\\"}"},
        }
    ])

    async def chunk_stream():
        yield "data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}\n\n"
        yield 'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
        yield "data: [DONE]\n\n"

    saver = DummySave()
    stream = handler.safe_stream_generator(chunk_stream(), save_callback=saver)
    collected = []
    async for item in stream:
        collected.append(item)
    assert any("Hello" in part for part in collected)
    assert saver.calls
    persisted = saver.calls[0]
    assert persisted["text"] == "Hello"
    assert persisted["tool_calls"] == [
        {
            "id": "call",
            "type": "function",
            "function": {"name": "foo", "arguments": "{\\\"a\\\": 1\\\"}"},
        }
    ]
    assert persisted["function_call"] is None


@pytest.mark.asyncio
async def test_streaming_topic_monitoring_runs_without_output_moderation(monkeypatch):
    class DummyPolicy:
        enabled = False
        output_enabled = False

    class DummyModeration:
        def get_effective_policy(self, *_args, **_kwargs):
            return DummyPolicy()

    class DummyMonitor:
        def __init__(self):
            self.calls: List[Dict[str, Any]] = []

        def schedule_evaluate_and_alert(self, **kwargs):

            self.calls.append(kwargs)

    dummy_monitor = DummyMonitor()
    monkeypatch.setattr(chat_service, "get_moderation_service", lambda: DummyModeration())
    monkeypatch.setattr(chat_service, "get_topic_monitoring_service", lambda: dummy_monitor)

    class DummyStreamTracker:
        def add_heartbeat(self):
            pass

        def add_chunk(self):

            pass

    class DummyMetrics:
        def track_llm_call(self, *args, **kwargs):
            pass

        def track_provider_fallback_success(self, *args, **kwargs):

            pass

        def track_rate_limit(self, *args, **kwargs):

            pass

        @asynccontextmanager
        async def track_streaming(self, *args, **kwargs):
            yield DummyStreamTracker()

    def fake_llm_call():

        def _gen():
            yield "data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}\n\n"
            yield "data: [DONE]\n\n"
        return _gen()

    response = await chat_service.execute_streaming_call(
        current_loop=asyncio.get_running_loop(),
        cleaned_args={"messages_payload": [], "api_endpoint": "openai"},
        selected_provider="openai",
        provider="openai",
        model="gpt",
        request_json="{}",
        request=None,
        metrics=DummyMetrics(),
        provider_manager=None,
        templated_llm_payload=[],
        should_persist=False,
        final_conversation_id="conv",
        character_card_for_context=None,
        chat_db=MagicMock(),
        save_message_fn=AsyncMock(),
        audit_service=None,
        audit_context=None,
        client_id="client",
        queue_execution_enabled=False,
        enable_provider_fallback=False,
        llm_call_func=fake_llm_call,
        refresh_provider_params=lambda *_args, **_kwargs: ({}, None),
        moderation_getter=None,
    )

    async for _chunk in response.body_iterator:
        pass

    assert dummy_monitor.calls


def test_document_generator_accepts_string_ids(tmp_path):
    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
    from tldw_Server_API.app.core.Chat.document_generator import DocumentGeneratorService, DocumentType

    db_path = tmp_path / "chacha.db"
    service_db = CharactersRAGDB(db_path=str(db_path), client_id="client")
    generator = DocumentGeneratorService(service_db, user_id="user")

    # Force table initialization
    generator._init_tables()

    # Insert with UUID conversation id
    conv_id = "550e8400-e29b-41d4-a716-446655440000"
    job_id = generator.create_generation_job(
        conv_id,
        DocumentType.SUMMARY,
        provider="openai",
        model="gpt",
        prompt_config={},
    )
    generator._save_generated_document(
        conversation_id=conv_id,
        document_type=DocumentType.SUMMARY,
        title="t",
        content="body",
        provider="openai",
        model="gpt",
        generation_time_ms=1,
    )

    docs = generator.get_generated_documents(conversation_id=conv_id)
    assert docs and docs[0]["conversation_id"] == conv_id

    job = generator.get_job_status(job_id)
    assert job is not None


_STREAM_ERROR_SENTINEL = "sk-stream-secret /private/provider.log https://provider.invalid/raw"


def _stream_error_payload(wire_shape: str, *, known: bool) -> object:
    code = "provider_authentication_failed" if known else "provider_private_failure"
    nested = {"error": {"code": code, "message": _STREAM_ERROR_SENTINEL}}
    if wire_shape == "dict":
        return nested
    if wire_shape == "bare":
        return {"code": code, "message": _STREAM_ERROR_SENTINEL}
    encoded = f"data: {json.dumps(nested)}\n\n"
    return encoded.encode() if wire_shape == "bytes" else encoded


def _error_frames(chunks: list[str]) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    for chunk in chunks:
        for line in chunk.splitlines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            try:
                payload = json.loads(line.removeprefix("data: "))
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict) and "error" in payload:
                frames.append(payload)
    return frames


@pytest.mark.asyncio
@pytest.mark.parametrize("stream_kind", ["sync", "async"])
@pytest.mark.parametrize("wire_shape", ["dict", "string", "bytes"])
@pytest.mark.parametrize("known", [True, False])
async def test_streaming_handler_sanitizes_every_inband_error_shape(
    stream_kind,
    wire_shape,
    known,
):
    payload = _stream_error_payload(wire_shape, known=known)
    closed = False

    if stream_kind == "async":

        async def source():
            nonlocal closed
            try:
                yield payload
            finally:
                closed = True

    else:

        def source():
            nonlocal closed
            try:
                yield payload
            finally:
                closed = True

    handler = StreamingResponseHandler("conv", "model", heartbeat_interval=0)
    logs: list[str] = []
    sink_id = logger.add(logs.append, format="{message}")
    try:
        chunks = [chunk async for chunk in handler.safe_stream_generator(source())]
    finally:
        logger.remove(sink_id)

    frames = _error_frames(chunks)
    expected_code = "provider_authentication_failed" if known else "provider_unavailable"
    assert len(frames) == 1
    assert frames[0]["error"] == {
        "code": expected_code,
        "type": expected_code,
        "message": (
            "The selected provider credentials could not be authenticated."
            if known
            else "The chat service provider is currently unavailable."
        ),
    }
    assert sum(chunk.strip().lower() == "data: [done]" for chunk in chunks) == 1
    assert closed is True
    assert _STREAM_ERROR_SENTINEL not in "".join(chunks)
    assert _STREAM_ERROR_SENTINEL not in "".join(logs)


@pytest.mark.asyncio
@pytest.mark.parametrize("stream_kind", ["sync", "async"])
@pytest.mark.parametrize("known", [True, False])
async def test_streaming_handler_sanitizes_raised_provider_errors(stream_kind, known):
    closed = False
    error = (
        ChatAuthenticationError(_STREAM_ERROR_SENTINEL, provider="openai")
        if known
        else RuntimeError(_STREAM_ERROR_SENTINEL)
    )

    if stream_kind == "async":

        async def source():
            nonlocal closed
            try:
                if False:
                    yield ""
                raise error
            finally:
                closed = True

    else:

        def source():
            nonlocal closed
            try:
                if False:
                    yield ""
                raise error
            finally:
                closed = True

    handler = StreamingResponseHandler("conv", "model", heartbeat_interval=0)
    logs: list[str] = []
    sink_id = logger.add(logs.append, format="{message}")
    try:
        chunks = [chunk async for chunk in handler.safe_stream_generator(source())]
    finally:
        logger.remove(sink_id)

    frames = _error_frames(chunks)
    expected_code = "provider_authentication_failed" if known else "provider_unavailable"
    assert len(frames) == 1
    assert frames[0]["error"]["code"] == expected_code
    assert closed is True
    assert _STREAM_ERROR_SENTINEL not in "".join(chunks)
    assert _STREAM_ERROR_SENTINEL not in "".join(logs)
