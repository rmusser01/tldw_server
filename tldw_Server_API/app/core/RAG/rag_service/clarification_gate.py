from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Literal, Optional


@dataclass
class ClarificationDecision:
    required: bool
    question: str | None
    reason: str
    confidence: float
    detector: Literal["heuristic", "llm", "hybrid"]


_AMBIGUOUS_PATTERNS = [
    re.compile(r"\b(this|that|it|they|those)\b", re.IGNORECASE),
    re.compile(r"^(fix|improve|update|debug)\b", re.IGNORECASE),
]


def _heuristic_decision(
    query: str,
    chat_history: list[dict[str, str]] | None,
) -> ClarificationDecision | None:
    q = (query or "").strip()
    if not q:
        return ClarificationDecision(False, None, "empty_query", 1.0, "heuristic")

    ambiguous_reference = _AMBIGUOUS_PATTERNS[0].search(q)
    # A named subject earlier in the question can supply the pronoun's context
    # ("Project Juniper ... who owns it?"). Question words alone cannot.
    prefix = q[: ambiguous_reference.start()] if ambiguous_reference else ""
    named_subject = any(
        not any(
            word.lower()
            in {"can", "could", "would", "should", "you", "what", "when", "how", "this", "that", "tell", "me", "about"}
            for word in match.group().split()
        )
        for match in re.finditer(r"\b[A-Z][\w'-]+(?:\s+[A-Z][\w'-]+)+\b", prefix)
    )
    if any(p.search(q) for p in _AMBIGUOUS_PATTERNS) and not chat_history and not named_subject:
        return ClarificationDecision(
            True,
            "Could you clarify what specific item or context you want me to focus on?",
            "ambiguous_reference_without_context",
            0.92,
            "heuristic",
        )

    if len(q.split()) >= 6 and not re.search(r"\b(this|that|it)\b", q, flags=re.IGNORECASE):
        return ClarificationDecision(False, None, "specific_enough", 0.85, "heuristic")

    return None


async def assess_query_for_clarification(
    query: str,
    chat_history: list[dict[str, str]] | None = None,
    *,
    timeout_sec: float = 1.5,
    llm_call: Optional[Callable[[str, list[dict[str, str]]], Awaitable[dict[str, Any]]]] = None,
) -> ClarificationDecision:
    heuristic = _heuristic_decision(query, chat_history)
    if heuristic is not None:
        return heuristic

    if llm_call is None:
        return ClarificationDecision(False, None, "fail_open", 0.5, "hybrid")

    try:
        payload = await asyncio.wait_for(llm_call(query, chat_history or []), timeout=timeout_sec)
        needs = bool(payload.get("needs_clarification", False))
        return ClarificationDecision(
            required=needs,
            question=(payload.get("clarifying_question") or None),
            reason=str(payload.get("reason", "llm_decision")),
            confidence=float(payload.get("confidence", 0.6)),
            detector="llm",
        )
    except TimeoutError:
        return ClarificationDecision(False, None, "llm_timeout_fallback", 0.5, "hybrid")
    except Exception:
        return ClarificationDecision(False, None, "fail_open", 0.5, "hybrid")
