from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
from tldw_Server_API.app.main import app
from tldw_Server_API.app.core.Chat.chat_metrics import ChatMetricsCollector
from tldw_Server_API.app.core.Chat.prompt_template_manager import DEFAULT_RAW_PASSTHROUGH_TEMPLATE


pytestmark = pytest.mark.integration


@patch.dict("tldw_Server_API.app.api.v1.endpoints.chat.API_KEYS", {"openai": "test_key"})
@patch("tldw_Server_API.app.api.v1.endpoints.chat.perform_chat_api_call")
@patch("tldw_Server_API.app.api.v1.endpoints.chat.load_template")
def test_chat_endpoint_records_run_first_rollout_labels(
    mock_load_template,
    mock_chat_api_call,
    client,
    auth_headers,
    mock_media_db,
    mock_chacha_db,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.Chat import chat_service

    mock_load_template.return_value = DEFAULT_RAW_PASSTHROUGH_TEMPLATE
    mock_chat_api_call.return_value = {
        "id": "chatcmpl-run-first",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "ok",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "run", "arguments": "{\"command\":\"ls\"}"},
                        },
                        {
                            "id": "c2",
                            "type": "function",
                            "function": {"name": "notes_search", "arguments": "{\"query\":\"todo\"}"},
                        },
                    ],
                }
            }
        ],
    }

    collector = ChatMetricsCollector()
    collector.track_run_first_rollout = MagicMock()
    collector.track_run_first_first_tool = MagicMock()
    collector.track_run_first_fallback_after_run = MagicMock()
    collector.track_run_first_completion_proxy = MagicMock()

    monkeypatch.setattr("tldw_Server_API.app.api.v1.endpoints.chat.get_chat_metrics", lambda: collector)
    monkeypatch.setattr(chat_service, "resolve_chat_run_first_rollout_mode", lambda raw_mode=None, default="off": "default_on")
    monkeypatch.setattr(
        chat_service,
        "resolve_chat_run_first_presentation_variant",
        lambda raw_variant=None, default="chat_phase2a_v1": "chat_phase2b_v1",
    )
    monkeypatch.setattr(
        chat_service,
        "resolve_chat_run_first_provider_allowlist",
        lambda raw_allowlist=None: ["openai:gpt-4o-mini"],
    )
    monkeypatch.setattr(chat_service, "get_chat_tool_allow_catalog", lambda: ["run", "notes_search"])
    monkeypatch.setattr(chat_service, "should_auto_execute_tools", lambda: False)

    app.dependency_overrides[get_media_db_for_user] = lambda: mock_media_db
    app.dependency_overrides[get_chacha_db_for_user] = lambda: mock_chacha_db
    try:
        body = {
            "api_provider": "openai",
            "messages": [{"role": "user", "content": "hi"}],
            "model": "gpt-4o-mini",
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "notes_search",
                        "description": "Search notes for relevant passages.",
                        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "run",
                        "description": "Execute shell commands.",
                        "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
                    },
                },
            ],
        }

        csrf_token = client.get("/api/v1/health").cookies.get("csrf_token", "")
        request_headers = dict(auth_headers)
        request_headers["X-CSRF-Token"] = csrf_token
        response = client.post(
            "/api/v1/chat/completions",
            json=body,
            headers=request_headers,
        )

        assert response.status_code == 200, response.text
        called_kwargs = mock_chat_api_call.call_args.kwargs
        assert [tool["function"]["name"] for tool in called_kwargs["tools"]] == ["run", "notes_search"]
        assert "run(command)" in called_kwargs["system_message"]

        collector.track_run_first_rollout.assert_called_once()
        collector.track_run_first_first_tool.assert_called_once()
        collector.track_run_first_fallback_after_run.assert_called_once()
        collector.track_run_first_completion_proxy.assert_called_once()

        _, rollout_kwargs = collector.track_run_first_rollout.call_args
        assert rollout_kwargs["presentation_variant"] == "chat_phase2b_v1"
        assert rollout_kwargs["cohort"] == "default_on"
        assert rollout_kwargs["provider"] == "openai"
        assert rollout_kwargs["model"] == "gpt-4o-mini"
        assert rollout_kwargs["streaming"] is False
        assert rollout_kwargs["eligible"] is True

        _, completion_kwargs = collector.track_run_first_completion_proxy.call_args
        assert completion_kwargs["outcome"] == "success"

        _, fallback_kwargs = collector.track_run_first_fallback_after_run.call_args
        assert fallback_kwargs["fallback_tool"] == "notes_search"
    finally:
        app.dependency_overrides.pop(get_media_db_for_user, None)
        app.dependency_overrides.pop(get_chacha_db_for_user, None)
