"""Explicit model results stay advisory, grounded and recoverable."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from uuid import uuid4

import pytest
from pydantic import ValidationError

from tldw_Server_API.app.api.v1.schemas.moderation_schemas import ModerationClassificationRequest
from tldw_Server_API.app.core.DB_Management.Moderation_Review_DB import ModerationReviewStore
from tldw_Server_API.app.core.Moderation import classification
from tldw_Server_API.app.core.Moderation.review_service import ModerationReviewService
from tldw_Server_API.app.core.Security.egress import ConfiguredEndpointScope


@pytest.fixture
def sample_case(tmp_path, monkeypatch):
    review = ModerationReviewService(ModerationReviewStore(db_path=str(tmp_path / "review.db")))
    request = ModerationClassificationRequest(
        request_id=uuid4(),
        provider="llama.cpp",
        model="existing-model",
        criterion="Flag an unpublished observatory access code.",
        text="The unpublished observatory access code is ROWAN-731.",
    )
    endpoint = SimpleNamespace(
        base_url="http://127.0.0.1:8010/v1",
        scope=ConfiguredEndpointScope.from_url("http://127.0.0.1:8010/v1"),
    )
    credentials = SimpleNamespace(
        trusted_endpoint=endpoint,
        api_key=None,
        app_config={},
        credentials_resolved=True,
    )
    calls = []

    async def provider(**kwargs):
        calls.append(kwargs)
        return response()

    monkeypatch.setattr(classification, "perform_chat_api_call_async", provider)
    monkeypatch.setattr(classification, "audited_transport", lambda _: True)
    return SimpleNamespace(review=review, request=request, credentials=credentials, calls=calls)


def response(**overrides):
    output = {
        "label": "flagged",
        "reason": "The sample discloses an unpublished access code.",
        "evidence": "ROWAN-731",
    }
    output.update(overrides)
    return {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(output)}}]}


async def classify(case, request=None):
    return await classification.classify_sample(
        request or case.request,
        actor_id="user:41",
        user_id=41,
        credentials=case.credentials,
        review=case.review,
    )


@pytest.mark.asyncio
async def test_grounded_result_is_saved_once_with_server_provenance_and_atomic_actor_audit(sample_case):
    case = sample_case
    item = await classify(case)
    replay = await classify(case)
    audit = case.review.list_audit(item_id=item["id"])["events"]
    assert (item, len(case.calls), audit[0]["actor_id"], audit[0]["action"]) == (replay, 1, "user:41", "item.created")  # nosec B101 -- Pytest checks classification, authority or resource retirement.
    assert (item["status"], item["decision_history"], item["excerpt"], item["recommended_action"]) == (  # nosec B101 -- Pytest checks classification, authority or resource retirement.
        "needs_review",
        [],
        case.request.text,
        "block",
    )
    assert item["context"]["classification"] == {  # nosec B101 -- Pytest checks classification, authority or resource retirement.
        "provider": "llama.cpp",
        "model": "existing-model",
        "criterion": case.request.criterion,
        "label": "flagged",
        "reason": "The sample discloses an unpublished access code.",
        "evidence": "ROWAN-731",
    }
    assert item["matches"][0]["confidence"] is None  # nosec B101 -- Pytest checks classification, authority or resource retirement.
    assert case.request.text not in json.dumps(audit)  # nosec B101 -- Pytest checks classification, authority or resource retirement.
    call = case.calls[0]
    assert (call["streaming"], call["tools"], call["timeout"], call["api_endpoint"]) == (False, None, 120, "llama.cpp")  # nosec B101 -- Pytest checks classification, authority or resource retirement.
    assert call["call_policy"].max_transport_attempts == 1  # nosec B101 -- Pytest checks classification, authority or resource retirement.


@pytest.mark.asyncio
async def test_benign_result_still_requires_human_review(sample_case, monkeypatch):
    async def provider(**_):
        return response(label="benign", reason="No access code is disclosed.", evidence=None)

    monkeypatch.setattr(classification, "perform_chat_api_call_async", provider)
    item = await classify(sample_case)
    assert (item["status"], item["recommended_action"], item["matches"]) == ("needs_review", "approve", [])  # nosec B101 -- Pytest checks classification, authority or resource retirement.


@pytest.mark.asyncio
async def test_same_receipt_with_different_input_refuses_without_provider_or_store_write(sample_case):
    case = sample_case
    item = await classify(case)
    with pytest.raises(classification.ClassificationError, match="receipt_conflict"):
        await classify(case, case.request.model_copy(update={"text": "Changed sample"}))
    assert (len(case.calls), case.review.get_item(item["id"])) == (1, item)  # nosec B101 -- Pytest checks classification, authority or resource retirement.


@pytest.mark.asyncio
async def test_actor_receipts_cannot_replay_another_actor_result(sample_case):
    case = sample_case
    first = await classify(case)
    second = await classification.classify_sample(
        case.request,
        actor_id="user:42",
        user_id=42,
        credentials=case.credentials,
        review=case.review,
    )
    assert first["id"] != second["id"]  # nosec B101 -- Pytest checks classification, authority or resource retirement.


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "raw",
    [
        {},
        {"choices": []},
        {"choices": [{"finish_reason": "length", "message": {"content": "{}"}}]},
        {"choices": [{"finish_reason": "stop", "message": {"content": "not JSON"}}]},
        {"choices": [{"finish_reason": "stop", "message": {"content": "[]"}}]},
        response(label="safe"),
        response(label=True),
        response(reason=""),
        response(evidence="invented words"),
        response(evidence=None),
        response(label="benign"),
        response(extra="untrusted"),
        response(reason="x" * 501),
        {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": json.dumps({"label": "benign", "reason": "Fine", "evidence": None}),
                        "refusal": "Denied",
                    },
                }
            ]
        },
        {"choices": [{"finish_reason": "stop", "message": {"content": "{}", "tool_calls": [{}]}}]},
    ],
)
async def test_invalid_refused_truncated_or_ungrounded_result_is_not_a_benign_or_saved_item(
    sample_case, monkeypatch, raw
):
    async def provider(**_):
        return raw

    monkeypatch.setattr(classification, "perform_chat_api_call_async", provider)
    with pytest.raises(classification.ClassificationError, match="invalid_result"):
        await classify(sample_case)
    assert sample_case.review.list_items()["items"] == []  # nosec B101 -- Pytest checks classification, authority or resource retirement.


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [RuntimeError("private provider payload"), TimeoutError("secret URL")])
async def test_provider_failure_is_truthful_private_and_not_saved(sample_case, monkeypatch, failure):
    async def provider(**_):
        raise failure

    monkeypatch.setattr(classification, "perform_chat_api_call_async", provider)
    with pytest.raises(classification.ClassificationError, match="provider_failed") as caught:
        await classify(sample_case)
    assert str(caught.value) == "provider_failed"  # nosec B101 -- Pytest checks classification, authority or resource retirement.
    assert sample_case.review.list_items()["items"] == []  # nosec B101 -- Pytest checks classification, authority or resource retirement.


@pytest.mark.asyncio
async def test_cancellation_propagates_without_saving_negative_classification(sample_case, monkeypatch):
    entered = asyncio.Event()
    retired = asyncio.Event()

    async def provider(**_):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            retired.set()

    monkeypatch.setattr(classification, "perform_chat_api_call_async", provider)
    task = asyncio.create_task(classify(sample_case))
    await asyncio.wait_for(entered.wait(), 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert retired.is_set() and sample_case.review.list_items()["items"] == []  # nosec B101 -- Pytest checks classification, authority or resource retirement.


@pytest.mark.asyncio
async def test_unaudited_transport_fails_before_provider(sample_case, monkeypatch):
    monkeypatch.setattr(classification, "audited_transport", lambda _: False)
    with pytest.raises(classification.ClassificationError, match="unsupported_provider"):
        await classify(sample_case)
    assert sample_case.calls == []  # nosec B101 -- Pytest checks classification, authority or resource retirement.


@pytest.mark.parametrize(
    "change",
    [
        {"text": ""},
        {"text": " " * 2},
        {"text": "x" * 4001},
        {"criterion": ""},
        {"criterion": "x" * 501},
        {"provider": "http://attacker.test"},
        {"model": "x" * 201},
        {"request_id": "not-uuid"},
        {"api_key": "not-allowed"},
        {"actor_id": "other"},
    ],
)
def test_request_trust_boundary_rejects_unbounded_or_forged_inputs(sample_case, change):
    with pytest.raises(ValidationError):
        ModerationClassificationRequest.model_validate({**sample_case.request.model_dump(), **change})


@pytest.mark.asyncio
async def test_llm_quota_refusal_keeps_original_http_contract(sample_case, monkeypatch):
    from tldw_Server_API.app.core.exceptions import LLMQuotaExceededError

    refusal = LLMQuotaExceededError(
        category="llm_tokens_per_month", current=5, limit=5, message="Monthly token quota reached.", retry_after=60
    )

    async def provider(**_):
        raise refusal

    monkeypatch.setattr(classification, "perform_chat_api_call_async", provider)
    with pytest.raises(LLMQuotaExceededError) as caught:
        await classify(sample_case)
    assert caught.value is refusal and sample_case.review.list_items()["items"] == []  # nosec B101 -- Pytest checks classification, authority or resource retirement.


@pytest.mark.parametrize("value", [None, [], "bad", 4, True])
def test_invalid_message_container_is_always_a_classification_error(value):
    with pytest.raises(classification.ClassificationError, match="invalid_result"):
        classification.parse_classification({"choices": [{"finish_reason": "stop", "message": value}]}, "Sample")


def test_deep_bounded_model_output_is_refused_as_invalid_not_server_error():
    content = '{"label":' + "[" * 3900 + "0" + "]" * 3900 + ',"reason":"bad","evidence":null}'
    with pytest.raises(classification.ClassificationError, match="invalid_result"):
        classification.parse_classification(
            {"choices": [{"finish_reason": "stop", "message": {"content": content}}]}, "Sample"
        )


def test_decoder_recursion_refusal_is_an_invalid_result(monkeypatch):
    def refuse(*_, **__):
        raise RecursionError("decoder nesting limit")

    monkeypatch.setattr(classification.json, "loads", refuse)
    with pytest.raises(classification.ClassificationError, match="invalid_result"):
        classification.parse_classification(
            {"choices": [{"finish_reason": "stop", "message": {"content": "{}"}}]}, "Sample"
        )
