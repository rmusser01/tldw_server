from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Protocol

from loguru import logger

from tldw_Server_API.app.core.Chat.Chat_Deps import ChatAPIError, ChatProviderError
from tldw_Server_API.app.core.RPG.rules.content_packs import RuleLookupItem
from tldw_Server_API.app.core.Workflows.adapters._common import extract_openai_content

ChatCall = Callable[..., Awaitable[Any]]

_SYSTEM_MESSAGE = (
    "You are answering a tabletop RPG rules question. Answer only from the provided rules evidence. "
    "If the evidence is insufficient, say so. Return JSON with keys `answer` and `citation_ids`; "
    "`citation_ids` must only contain snippet IDs shown in the evidence."
)


@dataclass(frozen=True, slots=True)
class RulesAnswerOptions:
    provider: str | None = None
    model: str | None = None
    temperature: float = 0.2
    max_tokens: int = 600


@dataclass(frozen=True, slots=True)
class RulesAnswerResult:
    answer: str | None
    answer_status: str
    citation_ids: list[str]


class RulesAnswerGenerator(Protocol):
    async def generate(
        self,
        *,
        query: str,
        evidence: list[RuleLookupItem],
        options: RulesAnswerOptions,
    ) -> RulesAnswerResult:
        ...


class ChatRulesAnswerGenerator:
    def __init__(self, *, chat_call: ChatCall | None = None) -> None:
        self._chat_call = chat_call or _default_chat_call

    async def generate(
        self,
        *,
        query: str,
        evidence: list[RuleLookupItem],
        options: RulesAnswerOptions,
    ) -> RulesAnswerResult:
        evidence_items = [item for item in evidence if item.origin == "user_provided" and item.text.strip()]
        if not evidence_items:
            return RulesAnswerResult(answer=None, answer_status="no_evidence", citation_ids=[])

        try:
            response = await self._chat_call(
                messages=[{"role": "user", "content": _user_prompt(query=query, evidence=evidence_items)}],
                system_message=_SYSTEM_MESSAGE,
                api_provider=options.provider,
                model=options.model,
                temperature=options.temperature,
                max_tokens=options.max_tokens,
                stream=False,
            )
        except (ChatAPIError, ChatProviderError) as exc:
            logger.warning("RPG rules answer generation failed: {}", type(exc).__name__)
            return RulesAnswerResult(answer=None, answer_status="generation_error", citation_ids=[])

        return _answer_result_from_response(response, evidence_items)


async def _default_chat_call(**kwargs: Any) -> Any:
    from tldw_Server_API.app.core.Chat.chat_service import perform_chat_api_call_async

    return await perform_chat_api_call_async(**kwargs)


def _user_prompt(*, query: str, evidence: list[RuleLookupItem]) -> str:
    evidence_blocks = []
    for item in evidence:
        citation = item.citation
        evidence_blocks.append(
            "\n".join(
                [
                    f"Snippet ID: {citation.snippet_id}",
                    f"Source: {citation.source_title}",
                    f"Text: {item.text.strip()}",
                ]
            )
        )
    evidence_text = "\n\n".join(evidence_blocks)
    return (
        f"Question: {query.strip()}\n\n"
        "Rules evidence:\n"
        f"{evidence_text}\n\n"
        "Return JSON only, for example: "
        '{"answer": "brief grounded answer", "citation_ids": ["media:1:chunk:2"]}'
    )


def _answer_result_from_response(response: Any, evidence: list[RuleLookupItem]) -> RulesAnswerResult:
    text = str(extract_openai_content(response) or "").strip()
    if not text:
        return RulesAnswerResult(answer=None, answer_status="generation_error", citation_ids=[])

    allowed_ids = [item.citation.snippet_id for item in evidence if item.citation.snippet_id]
    parsed = _loads_json_object(text)
    if parsed is not None:
        answer = str(parsed.get("answer") or "").strip()
        citation_ids = _filtered_citation_ids(parsed.get("citation_ids"), allowed_ids)
        if not answer:
            return RulesAnswerResult(answer=None, answer_status="generation_error", citation_ids=[])
        return RulesAnswerResult(answer=answer, answer_status="answered", citation_ids=citation_ids)

    logger.warning("RPG rules answer generation returned non-JSON content")
    return RulesAnswerResult(answer=None, answer_status="generation_error", citation_ids=[])


def _loads_json_object(text: str) -> dict[str, Any] | None:
    candidate = _strip_json_fence(text)
    try:
        loaded = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    if isinstance(loaded, dict):
        return loaded
    return None


def _strip_json_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if len(lines) >= 2 and lines[0].startswith("```") and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return stripped


def _filtered_citation_ids(value: Any, allowed_ids: list[str]) -> list[str]:
    allowed = set(allowed_ids)
    filtered: list[str] = []
    if not isinstance(value, list):
        return filtered
    for item in value:
        text = str(item or "").strip()
        if text in allowed and text not in filtered:
            filtered.append(text)
    return filtered
