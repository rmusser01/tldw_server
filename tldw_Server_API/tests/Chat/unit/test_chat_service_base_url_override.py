from __future__ import annotations

import pytest

from tldw_Server_API.app.core.Chat import chat_service
from tldw_Server_API.app.core.Chat.Chat_Deps import ChatBadRequestError
from tldw_Server_API.app.core.AuthNZ import byok_helpers


def _base_args() -> dict:
    return {
        "api_provider": "openai",
        "messages": [{"role": "user", "content": "hi"}],
        "model": "gpt-4o-mini",
        "api_key": "test-key",
        "app_config": {},
    }


def test_base_url_override_allowed(monkeypatch):
    monkeypatch.setattr(byok_helpers, "resolve_byok_base_url_allowlist", lambda: {"openai"})
    monkeypatch.setattr(byok_helpers, "validate_base_url_override", lambda value: value)
    args = _base_args()
    args.update({"base_url": "https://example.com/v1", "trusted_base_url_override": True})
    provider, request, _internal = chat_service._build_adapter_request_from_chat_args(args)
    assert provider == "openai"
    assert request["base_url"] == "https://example.com/v1"


def test_base_url_override_rejected_when_untrusted(monkeypatch):
    monkeypatch.setattr(byok_helpers, "resolve_byok_base_url_allowlist", lambda: {"openai"})
    args = _base_args()
    args.update({"base_url": "https://example.com/v1"})
    with pytest.raises(ChatBadRequestError):
        chat_service._build_adapter_request_from_chat_args(args)


def test_base_url_override_rejected_when_not_allowlisted(monkeypatch):
    monkeypatch.setattr(byok_helpers, "resolve_byok_base_url_allowlist", lambda: set())
    args = _base_args()
    args.update({"base_url": "https://example.com/v1", "trusted_base_url_override": True})
    with pytest.raises(ChatBadRequestError):
        chat_service._build_adapter_request_from_chat_args(args)


def test_build_adapter_request_omits_internal_chat_metadata() -> None:
    args = _base_args()
    args.update(
        {
            "_chat_effective_tool_names": ["run", "notes.search"],
            "_chat_run_first_eligible": True,
            "_chat_run_first_ineligible_reason": "provider_not_in_rollout_allowlist",
            "_chat_run_first_presentation_variant": "chat_phase2b_v1",
            "_chat_run_first_cohort": "default_on",
            "_chat_run_first_cohort": "gated",
        }
    )

    provider, request, _internal = chat_service._build_adapter_request_from_chat_args(args)

    assert provider == "openai"
    assert not any(key.startswith("_chat_") for key in request)
