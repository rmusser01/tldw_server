from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from tldw_Server_API.app.core.Claims_Extraction import claims_engine, claims_service
from tldw_Server_API.app.core.exceptions import LLMQuotaExceededError
from tldw_Server_API.app.core.RAG.rag_service.types import DataSource, Document


def test_fva_analyze_passes_response_format_and_returns_choice_content(monkeypatch):
    captured = {}

    def _fake_chat_call(**kwargs):
        captured.update(kwargs)
        return {"choices": [{"message": {"content": '{"label":"supported"}'}}]}

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Chat.chat_service.perform_chat_api_call",
        _fake_chat_call,
    )

    out = claims_service._fva_claims_analyze_call(
        "openai",
        "ignored input",
        "Use this prompt",
        None,
        "System message",
        0.12,
        False,
        False,
        False,
        None,
        model_override="gpt-test",
        response_format={"type": "json_object"},
    )

    assert out == '{"label":"supported"}'
    assert captured["api_endpoint"] == "openai"
    assert captured["model"] == "gpt-test"
    assert captured["response_format"] == {"type": "json_object"}
    assert captured["messages_payload"][0]["content"] == "Use this prompt"


def test_fva_analyze_uses_input_data_when_prompt_missing(monkeypatch):
    def _fake_chat_call(**kwargs):
        return {"response": "ok-from-input"}

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Chat.chat_service.perform_chat_api_call",
        _fake_chat_call,
    )

    out = claims_service._fva_claims_analyze_call(
        "openai",
        "input fallback text",
        None,
        None,
        None,
        0.2,
    )

    assert out == "ok-from-input"


def test_fva_analyze_returns_empty_on_provider_error(monkeypatch):
    def _fake_chat_call(**kwargs):
        raise RuntimeError("provider down")

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Chat.chat_service.perform_chat_api_call",
        _fake_chat_call,
    )

    out = claims_service._fva_claims_analyze_call(
        "openai",
        "input",
        "prompt",
        None,
        None,
        0.2,
    )

    assert out == ""


def _refused(**_kwargs):
    raise LLMQuotaExceededError(
        category="llm_tokens_month", current=0, limit=0, message="Monthly LLM token limit reached", retry_after=60
    )


def test_fva_analyze_lets_an_llm_quota_refusal_through(monkeypatch):
    """A refusal is not turned into an empty reply (TASK-13638)."""
    monkeypatch.setattr("tldw_Server_API.app.core.Chat.chat_service.perform_chat_api_call", _refused)

    with pytest.raises(LLMQuotaExceededError):
        claims_service._fva_claims_analyze_call("openai", "input", "prompt", None, None, 0.2)


@pytest.mark.asyncio
async def test_a_refused_verification_raises_and_records_no_usage(monkeypatch):
    """On dev the empty reply let the decision tree mark this claim verified, and wrote a claims_verify row (TASK-13638)."""
    recorder = AsyncMock()
    monkeypatch.setattr("tldw_Server_API.app.core.Chat.chat_service.perform_chat_api_call", _refused)
    monkeypatch.setattr(claims_engine, "_log_claims_llm_usage", recorder)
    verifier = claims_engine.HybridClaimVerifier(claims_service._fva_claims_analyze_call)
    evidence = Document(id="d1", content="The sky is blue.", metadata={}, source=DataSource.MEDIA_DB, score=0.9)

    with pytest.raises(LLMQuotaExceededError):
        await verifier.verify(
            claim=claims_engine.Claim(id="c1", text="The sky is green."),
            query="sky colour",
            base_documents=[evidence],
            mode="llm",
        )
    recorder.assert_not_awaited()


@pytest.mark.parametrize(
    ("provider", "model", "expected_provider", "expected_model"),
    [
        ("llamacpp", None, "llamacpp", None),
        ("llamacpp", "local-explicit", "llamacpp", "local-explicit"),
        (None, None, "openai", "standalone-claims-model"),
        (None, "explicit-claims-model", "openai", "explicit-claims-model"),
    ],
)
def test_x04_fva_provider_default_preserves_standalone_claims_defaults(
    monkeypatch, provider, model, expected_provider, expected_model
):
    from tldw_Server_API.app.core.config import settings

    monkeypatch.setitem(settings, "CLAIMS_LLM_PROVIDER", "openai")
    monkeypatch.setitem(settings, "CLAIMS_LLM_MODEL", "standalone-claims-model")
    dispatches = []

    def provider_io(**kwargs):
        dispatches.append((kwargs["api_endpoint"], kwargs["model"]))
        return {"choices": [{"message": {"content": "supported"}}]}

    monkeypatch.setattr("tldw_Server_API.app.core.Chat.chat_service.perform_chat_api_call", provider_io)
    result = claims_service._fva_claims_analyze_call(provider, "source", "verify", None, None, model_override=model)
    assert result == "supported"  # nosec B101 # Pytest behavioral oracle; not a runtime guard.
    assert dispatches == [(expected_provider, expected_model)]  # nosec B101 # Pytest behavioral oracle; not a runtime guard.


@pytest.mark.parametrize(
    "failure",
    [
        LLMQuotaExceededError(category="llm_tokens_month", current=0, limit=0, message="Refused", retry_after=60),
        asyncio.CancelledError(),
    ],
)
def test_x04_bound_verifier_preserves_quota_and_cancellation_identity(monkeypatch, failure):
    from tldw_Server_API.app.core.Claims_Extraction import artifact_verification

    def refused_io(**_kwargs):
        raise failure

    monkeypatch.setattr("tldw_Server_API.app.core.Chat.chat_service.perform_chat_api_call", refused_io)
    bound = artifact_verification._bound_verification_analyze_fn(
        artifact_verification._default_analyze_fn, verification_provider="llamacpp", verification_model=None
    )
    with pytest.raises(type(failure)) as caught:
        bound("openai", "source", "verify", None, None, model_override="unrelated-model")
    assert caught.value is failure  # nosec B101 # Pytest behavioral oracle; not a runtime guard.
