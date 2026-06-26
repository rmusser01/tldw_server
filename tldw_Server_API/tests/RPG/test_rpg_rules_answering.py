from __future__ import annotations

from typing import Any

import pytest

from tldw_Server_API.app.core.Chat.Chat_Deps import ChatProviderError
from tldw_Server_API.app.core.RPG.rules.answering import (
    ChatRulesAnswerGenerator,
    RulesAnswerOptions,
    RulesAnswerResult,
)
from tldw_Server_API.app.core.RPG.rules.content_packs import RuleLookupCitation, RuleLookupItem
from tldw_Server_API.app.core.RPG.rules.lookup import RulesLookupService
from tldw_Server_API.app.core.RPG.rules.retrieval import RulesRetrievalResult


class FakeLookupRetriever:
    def __init__(self, items: list[RuleLookupItem]) -> None:
        self.items = items

    async def retrieve(self, **kwargs: Any) -> RulesRetrievalResult:
        return RulesRetrievalResult(
            items=self.items,
            ready_media_ids=[42] if self.items else [],
            skipped_refs=[],
            diagnostics={"retrieval_result_count": len(self.items)},
        )


class FakeAnswerGenerator:
    def __init__(self, result: RulesAnswerResult) -> None:
        self.result = result
        self.calls: list[dict[str, Any]] = []

    async def generate(self, **kwargs: Any) -> RulesAnswerResult:
        self.calls.append(kwargs)
        return self.result


def _item(snippet_id: str = "media:42:chunk:7", text: str = "Spend a stress box to reduce harm.") -> RuleLookupItem:
    return RuleLookupItem(
        origin="user_provided",
        text=text,
        citation=RuleLookupCitation(
            source_type="media_item",
            source_id=42,
            source_title="Fate Rules",
            source_url=None,
            license=None,
            license_url=None,
            attribution=None,
            trust_level="user_provided",
            content_hash="sha256:abc",
            snippet_id=snippet_id,
        ),
        score=0.9,
    )


@pytest.mark.asyncio
async def test_answer_mode_returns_not_requested_for_lookup_mode() -> None:
    generator = FakeAnswerGenerator(RulesAnswerResult(answer="unused", answer_status="answered", citation_ids=[]))
    lookup = RulesLookupService(retriever=FakeLookupRetriever([_item()]), answer_generator=generator)

    result = await lookup.lookup(
        owner_user_id=42,
        adapter_key="fate",
        query="stress",
        linked_rules_pack_refs=[{"source_type": "media_item", "source_id": 42}],
        mode="lookup",
    )

    assert result.answer is None  # nosec B101
    assert result.answer_status == "not_requested"  # nosec B101
    assert result.answer_citation_ids == []  # nosec B101
    assert generator.calls == []  # nosec B101


@pytest.mark.asyncio
async def test_answer_mode_returns_no_evidence_without_user_snippets() -> None:
    generator = FakeAnswerGenerator(RulesAnswerResult(answer="unused", answer_status="answered", citation_ids=[]))
    lookup = RulesLookupService(retriever=FakeLookupRetriever([]), answer_generator=generator)

    result = await lookup.lookup(
        owner_user_id=42,
        adapter_key="fate",
        query="stress",
        linked_rules_pack_refs=[{"source_type": "media_item", "source_id": 42}],
        mode="answer",
    )

    assert result.answer is None  # nosec B101
    assert result.answer_status == "no_evidence"  # nosec B101
    assert result.answer_citation_ids == []  # nosec B101
    assert generator.calls == []  # nosec B101


@pytest.mark.asyncio
async def test_answer_mode_calls_chat_service_with_grounded_prompt() -> None:
    calls: list[dict[str, Any]] = []

    async def fake_chat(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {"choices": [{"message": {"content": '{"answer": "Use the stress track.", "citation_ids": ["media:42:chunk:7"]}'}}]}

    generator = ChatRulesAnswerGenerator(chat_call=fake_chat)

    await generator.generate(query="How does stress work?", evidence=[_item()], options=RulesAnswerOptions())

    assert len(calls) == 1  # nosec B101
    call = calls[0]
    assert call["messages"][0]["role"] == "user"  # nosec B101
    assert "How does stress work?" in call["messages"][0]["content"]  # nosec B101
    assert "media:42:chunk:7" in call["messages"][0]["content"]  # nosec B101
    assert "Spend a stress box" in call["messages"][0]["content"]  # nosec B101
    assert "only from the provided rules evidence" in call["system_message"]  # nosec B101
    assert call["stream"] is False  # nosec B101


@pytest.mark.asyncio
async def test_answer_mode_extracts_openai_content() -> None:
    async def fake_chat(**kwargs: Any) -> dict[str, Any]:
        return {"choices": [{"message": {"content": '{"answer": "Stress absorbs harm.", "citation_ids": ["media:42:chunk:7"]}'}}]}

    result = await ChatRulesAnswerGenerator(chat_call=fake_chat).generate(
        query="stress",
        evidence=[_item()],
        options=RulesAnswerOptions(),
    )

    assert result.answer == "Stress absorbs harm."  # nosec B101
    assert result.answer_status == "answered"  # nosec B101
    assert result.citation_ids == ["media:42:chunk:7"]  # nosec B101


@pytest.mark.asyncio
async def test_answer_mode_filters_unknown_citation_ids() -> None:
    async def fake_chat(**kwargs: Any) -> str:
        return '{"answer": "Stress absorbs harm.", "citation_ids": ["media:42:chunk:7", "unknown"]}'

    result = await ChatRulesAnswerGenerator(chat_call=fake_chat).generate(
        query="stress",
        evidence=[_item()],
        options=RulesAnswerOptions(),
    )

    assert result.citation_ids == ["media:42:chunk:7"]  # nosec B101


@pytest.mark.asyncio
async def test_answer_mode_rejects_malformed_json_without_citations() -> None:
    async def fake_chat(**kwargs: Any) -> str:
        return "Stress absorbs harm."

    result = await ChatRulesAnswerGenerator(chat_call=fake_chat).generate(
        query="stress",
        evidence=[_item()],
        options=RulesAnswerOptions(),
    )

    assert result.answer is None  # nosec B101
    assert result.answer_status == "generation_error"  # nosec B101
    assert result.citation_ids == []  # nosec B101


@pytest.mark.asyncio
async def test_answer_mode_returns_generation_error_on_provider_failure() -> None:
    async def fake_chat(**kwargs: Any) -> str:
        raise ChatProviderError("provider failed", provider="openai")

    result = await ChatRulesAnswerGenerator(chat_call=fake_chat).generate(
        query="stress",
        evidence=[_item()],
        options=RulesAnswerOptions(provider="openai"),
    )

    assert result.answer is None  # nosec B101
    assert result.answer_status == "generation_error"  # nosec B101
    assert result.citation_ids == []  # nosec B101


@pytest.mark.asyncio
async def test_answer_mode_uses_request_provider_and_model() -> None:
    calls: list[dict[str, Any]] = []

    async def fake_chat(**kwargs: Any) -> str:
        calls.append(kwargs)
        return "Plain answer"

    await ChatRulesAnswerGenerator(chat_call=fake_chat).generate(
        query="stress",
        evidence=[_item()],
        options=RulesAnswerOptions(provider="openai", model="gpt-test", temperature=0.4, max_tokens=321),
    )

    assert calls[0]["api_provider"] == "openai"  # nosec B101
    assert calls[0]["model"] == "gpt-test"  # nosec B101
    assert calls[0]["temperature"] == 0.4  # nosec B101
    assert calls[0]["max_tokens"] == 321  # nosec B101


@pytest.mark.asyncio
async def test_answer_mode_uses_default_temperature_and_token_bounds() -> None:
    calls: list[dict[str, Any]] = []

    async def fake_chat(**kwargs: Any) -> str:
        calls.append(kwargs)
        return "Plain answer"

    await ChatRulesAnswerGenerator(chat_call=fake_chat).generate(
        query="stress",
        evidence=[_item()],
        options=RulesAnswerOptions(),
    )

    assert calls[0]["temperature"] == 0.2  # nosec B101
    assert calls[0]["max_tokens"] == 600  # nosec B101


@pytest.mark.asyncio
async def test_answer_mode_wires_generator_result_into_lookup() -> None:
    generator = FakeAnswerGenerator(
        RulesAnswerResult(
            answer="Use the stress track.",
            answer_status="answered",
            citation_ids=["media:42:chunk:7", "unknown"],
        )
    )
    lookup = RulesLookupService(retriever=FakeLookupRetriever([_item()]), answer_generator=generator)

    result = await lookup.lookup(
        owner_user_id=42,
        adapter_key="fate",
        query="stress",
        linked_rules_pack_refs=[{"source_type": "media_item", "source_id": 42}],
        mode="answer",
        answer_options=RulesAnswerOptions(provider="openai", model="gpt-test"),
    )

    assert result.answer == "Use the stress track."  # nosec B101
    assert result.answer_status == "answered"  # nosec B101
    assert result.answer_citation_ids == ["media:42:chunk:7"]  # nosec B101
    assert generator.calls[0]["options"].provider == "openai"  # nosec B101
    assert generator.calls[0]["options"].model == "gpt-test"  # nosec B101


@pytest.mark.asyncio
async def test_answer_mode_returns_generation_error_on_unexpected_generator_failure() -> None:
    class FailingAnswerGenerator:
        async def generate(self, **kwargs: Any) -> RulesAnswerResult:
            raise RuntimeError("provider adapter crashed")

    lookup = RulesLookupService(retriever=FakeLookupRetriever([_item()]), answer_generator=FailingAnswerGenerator())

    result = await lookup.lookup(
        owner_user_id=42,
        adapter_key="fate",
        query="stress",
        linked_rules_pack_refs=[{"source_type": "media_item", "source_id": 42}],
        mode="answer",
    )

    assert result.answer is None  # nosec B101
    assert result.answer_status == "generation_error"  # nosec B101
    assert result.answer_citation_ids == []  # nosec B101
