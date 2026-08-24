import pytest
from loguru import logger as loguru_logger

from tldw_Server_API.app.api.v1.schemas.chat_request_schemas import (
    ChatCompletionRequest,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)
from tldw_Server_API.app.core.Chat import chat_orchestrator
from tldw_Server_API.app.core.Chat.chat_service import apply_prompt_templating


def _extract_text_content(message):
    content = message.get("content")
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if part.get("type") == "text")
    return content


def test_apply_prompt_templating_strips_system_messages():
    request_data = ChatCompletionRequest(
        model="test-model",
        messages=[
            ChatCompletionSystemMessageParam(role="system", content="You are a helpful assistant."),
            ChatCompletionUserMessageParam(role="user", content="Hello there."),
        ],
    )
    llm_payload_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello there."},
    ]
    character_card = {"name": "Test", "system_prompt": "Default prompt"}

    final_system_message, payload = apply_prompt_templating(
        request_data=request_data,
        character_card=character_card,
        llm_payload_messages=llm_payload_messages,
    )

    assert final_system_message == "You are a helpful assistant."
    assert all(msg.get("role") != "system" for msg in payload)
    assert len(payload) == 1
    assert payload[0]["role"] == "user"
    assert _extract_text_content(payload[0]) == "Hello there."


def test_apply_prompt_templating_uses_payload_system_message():
    request_data = ChatCompletionRequest(
        model="test-model",
        messages=[ChatCompletionUserMessageParam(role="user", content="Hello there.")],
    )
    llm_payload_messages = [
        {"role": "system", "content": "Persisted system prompt"},
        {"role": "user", "content": "Hello there."},
    ]
    character_card = {"name": "Test", "system_prompt": "Default prompt"}

    final_system_message, payload = apply_prompt_templating(
        request_data=request_data,
        character_card=character_card,
        llm_payload_messages=llm_payload_messages,
    )

    assert final_system_message == "Persisted system prompt"
    assert all(msg.get("role") != "system" for msg in payload)
    assert len(payload) == 1
    assert payload[0]["role"] == "user"
    assert _extract_text_content(payload[0]) == "Hello there."


def test_apply_prompt_templating_combines_request_system_messages():
    request_data = ChatCompletionRequest(
        model="test-model",
        messages=[
            ChatCompletionSystemMessageParam(role="system", content="Primary system."),
            ChatCompletionSystemMessageParam(role="system", content="Injected system."),
            ChatCompletionUserMessageParam(role="user", content="Hello there."),
        ],
    )
    llm_payload_messages = [
        {"role": "system", "content": "Primary system."},
        {"role": "system", "content": "Injected system."},
        {"role": "user", "content": "Hello there."},
    ]
    character_card = {"name": "Test", "system_prompt": "Default prompt"}

    final_system_message, payload = apply_prompt_templating(
        request_data=request_data,
        character_card=character_card,
        llm_payload_messages=llm_payload_messages,
    )

    assert final_system_message == "Primary system.\n\nInjected system."
    assert all(msg.get("role") != "system" for msg in payload)
    assert len(payload) == 1
    assert payload[0]["role"] == "user"


def test_prompt_template_summary_omits_raw_prompt_text():
    from tldw_Server_API.app.core.Chat.chat_logging import prompt_template_summary

    summary = prompt_template_summary(
        template_name="raw",
        system_message="do not leak this system prompt",
        payload_system_messages=["payload secret"],
        request_system_messages=["request secret"],
        character_name="Tester",
    )

    rendered = repr(summary)
    assert "do not leak" not in rendered
    assert "payload secret" not in rendered
    assert "request secret" not in rendered
    assert summary["system_message"] == {"present": True, "type": "str", "chars": 30}
    assert summary["payload_system_messages"]["count"] == 1
    assert summary["payload_system_messages"]["items"][0]["chars"] == 14
    assert summary["request_system_messages"]["count"] == 1
    assert summary["request_system_messages"]["items"][0]["chars"] == 14
    assert summary["character_name"] == "Tester"


def test_message_payload_summary_omits_text_and_image_data_uri():
    from tldw_Server_API.app.core.Chat.chat_logging import message_payload_summary

    secret_text = "raw user text must not appear"
    secret_b64 = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAASECRET"
    secret_metadata_key = "private metadata key"
    secret_role = "secret role label"
    secret_part_type = "secret part type"
    summary = message_payload_summary(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": secret_text},
                    {"type": "image_url", "image_url": {"url": secret_b64}},
                    {"type": secret_part_type, "text": "hidden text under custom type"},
                ],
                "metadata": {secret_metadata_key: "private metadata value"},
            },
            {"role": "assistant", "content": "generated assistant content"},
            {"role": secret_role, "content": None},
        ]
    )

    rendered = repr(summary)
    assert secret_text not in rendered
    assert secret_b64 not in rendered
    assert secret_metadata_key not in rendered
    assert secret_role not in rendered
    assert secret_part_type not in rendered
    assert "generated assistant content" not in rendered
    assert summary["count"] == 3
    assert summary["roles"] == {"assistant": 1, "other": 1, "user": 1}
    assert summary["messages"][0]["content"]["kind"] == "parts"
    assert summary["messages"][0]["content"]["part_kinds"] == {"image_url": 1, "other": 1, "text": 1}
    assert summary["messages"][0]["content"]["text_chars"] == len(secret_text)
    assert summary["messages"][0]["content"]["image_count"] == 1
    assert summary["messages"][0]["has_metadata"] is True
    assert summary["messages"][0]["metadata_key_count"] == 1
    assert summary["messages"][1]["content"]["kind"] == "text"
    assert summary["messages"][1]["content"]["chars"] == len("generated assistant content")
    assert summary["messages"][2]["role"] == "other"


def test_tool_and_response_summaries_omit_sensitive_payloads():
    from tldw_Server_API.app.core.Chat.chat_logging import response_summary, tool_payload_summary

    secret_tool_key = "private tool key"
    secret_response_key = "private response key"
    secret_usage_key = "private usage key"
    tool_summary = tool_payload_summary(
        {
            "arguments": {"query": "sensitive tool argument"},
            "output": "sensitive tool output",
            "error": "sensitive execution detail",
            secret_tool_key: "value",
        }
    )
    response = {
        "choices": [
            {"message": {"content": "generated assistant content"}},
            {"message": {"tool_calls": [{"function": {"arguments": "secret args"}}]}},
        ],
        "usage": {secret_usage_key: 42},
        secret_response_key: "private response value",
    }
    response_info = response_summary(response)

    rendered = repr({"tool": tool_summary, "response": response_info})
    assert secret_tool_key not in rendered
    assert secret_response_key not in rendered
    assert secret_usage_key not in rendered
    assert "sensitive tool argument" not in rendered
    assert "sensitive tool output" not in rendered
    assert "sensitive execution detail" not in rendered
    assert "generated assistant content" not in rendered
    assert "secret args" not in rendered
    assert "private response value" not in rendered
    assert tool_summary["kind"] == "dict"
    assert tool_summary["key_count"] == 4
    assert tool_summary["has_arguments"] is True
    assert tool_summary["has_output"] is True
    assert tool_summary["has_error"] is True
    assert response_info["kind"] == "dict"
    assert response_info["key_count"] == 3
    assert response_info["has_choices"] is True
    assert response_info["has_usage"] is True
    assert response_info["has_error"] is False
    assert response_info["choices"] == {"count": 2}
    assert response_info["usage_key_count"] == 1


def test_chat_api_call_error_log_omits_raw_exception_message(monkeypatch):
    secret = "raw prompt and tool output must not enter logs"
    captured_logs: list[str] = []
    sink_id = loguru_logger.add(
        lambda message: captured_logs.append(str(message)),
        format="{message}",
        level="ERROR",
    )

    def fail_provider(**_kwargs):
        raise RuntimeError(secret)

    monkeypatch.setattr(chat_orchestrator, "perform_chat_api_call", fail_provider)

    try:
        with pytest.raises(chat_orchestrator.ChatAPIError):
            chat_orchestrator.chat_api_call(
                api_endpoint="openai",
                messages_payload=[{"role": "user", "content": "secret user message"}],
                api_key="sk-secret",
            )
    finally:
        loguru_logger.remove(sink_id)

    rendered = "\n".join(captured_logs)
    assert "RuntimeError" in rendered
    assert secret not in rendered
    assert "secret user message" not in rendered
    assert "sk-secret" not in rendered
