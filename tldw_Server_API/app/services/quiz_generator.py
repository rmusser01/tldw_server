from __future__ import annotations

import asyncio
import contextlib
import json
import time
from collections.abc import Sequence
from typing import Any

from loguru import logger

from tldw_Server_API.app.api.v1.schemas.chat_request_schemas import DEFAULT_LLM_PROVIDER
from tldw_Server_API.app.core.Chat.Chat_Deps import ChatConfigurationError
from tldw_Server_API.app.core.Chat.chat_helpers import extract_response_content
from tldw_Server_API.app.core.Chat.chat_service import resolve_provider_api_key
from tldw_Server_API.app.core.config import load_and_log_configs
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.LLM_Calls.adapter_registry import get_registry
from tldw_Server_API.app.core.LLM_Calls.provider_metadata import provider_requires_api_key
from tldw_Server_API.app.core.testing import is_test_mode
from tldw_Server_API.app.services.quiz_source_resolver import resolve_quiz_sources

DEFAULT_QUESTION_TYPES = ["multiple_choice", "true_false", "fill_blank"]
SUPPORTED_GENERATED_QUESTION_TYPES = [
    "multiple_choice",
    "multi_select",
    "matching",
    "true_false",
    "fill_blank",
]
MAX_CONTENT_CHARS = 15000


class QuizProvenanceValidationError(ValueError):
    """Raised when generated quiz questions fail strict source provenance validation."""


QUIZ_GENERATION_PROMPT = """You are a quiz generator. Based on the following content, generate {num_questions} quiz questions.


Content:
{content}

Requirements:
- Difficulty: {difficulty}
- Question types to include: {question_types}
{focus_instruction}
{source_contract}

Return a JSON object in this exact format:
{{
  "questions": [
    {{
      "question_type": "multiple_choice" | "true_false" | "fill_blank",
      "question_text": "The question text",
      "options": ["A", "B", "C", "D"],
      "correct_answer": 0 | 1 | 2 | 3 | "true" | "false" | "the answer",
      "explanation": "Brief explanation of why this is correct",
      "hint": "Optional short hint shown on request",
      "hint_penalty_points": 0,
      "source_citations": [
        {{
          "source_type": "media" | "note" | "flashcard_deck" | "flashcard_card",
          "source_id": "Canonical source identifier",
          "label": "Optional citation label",
          "quote": "Supporting excerpt",
          "chunk_id": "Optional source chunk id",
          "timestamp_seconds": 0
        }}
      ],
      "points": 1
    }}
  ]
}}

Important:
- For multiple_choice: options must be array of 4 strings, correct_answer is 0-based index (0-3)
- For true_false: correct_answer must be exactly "true" or "false"
- For fill_blank: question_text should contain ___ where answer goes, correct_answer is the word/phrase
- hint_penalty_points must be a non-negative integer
- source_citations must include source_type and source_id and reference only provided sources
- Vary question difficulty according to the specified level
- Make questions test understanding, not just memorization
- Return ONLY valid JSON, no other text
"""


def _resolve_model(provider: str, model: str | None, app_config: dict[str, Any]) -> str | None:
    if model:
        return model
    key = f"{provider.replace('-', '_').replace('.', '_')}_api"
    return (app_config.get(key) or {}).get("model")


def _get_adapter(provider: str):
    registry = get_registry()
    adapter = registry.get_adapter(provider)
    if adapter is None:
        raise ChatConfigurationError(provider=provider, message="LLM adapter unavailable.")
    return adapter


def _normalize_question_type(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    aliases = {
        "multiple choice": "multiple_choice",
        "multiple-choice": "multiple_choice",
        "multi select": "multi_select",
        "multi-select": "multi_select",
        "true/false": "true_false",
        "true-false": "true_false",
        "fill in the blank": "fill_blank",
        "fill-in-the-blank": "fill_blank",
    }
    return aliases.get(text, text)


def _coerce_question_types(question_types: Sequence[Any] | None) -> list[str]:
    if not question_types:
        return list(DEFAULT_QUESTION_TYPES)
    normalized: list[str] = []
    for item in question_types:
        raw = getattr(item, "value", item)
        q_type = _normalize_question_type(raw)
        if q_type in DEFAULT_QUESTION_TYPES and q_type not in normalized:
            normalized.append(q_type)
    return normalized or list(DEFAULT_QUESTION_TYPES)


def _coerce_options(raw: Any, expected_count: int | None = None) -> list[str]:
    if isinstance(raw, list):
        options = [str(opt).strip() for opt in raw if str(opt).strip()]
    elif isinstance(raw, str):
        if "\n" in raw:
            options = [part.strip() for part in raw.splitlines() if part.strip()]
        elif "|" in raw:
            options = [part.strip() for part in raw.split("|") if part.strip()]
        elif ";" in raw:
            options = [part.strip() for part in raw.split(";") if part.strip()]
        else:
            options = []
    else:
        options = []
    if expected_count is not None:
        if len(options) != expected_count:
            raise ValueError(f"Expected {expected_count} options, got {len(options)}")
        return options
    if len(options) > 4:
        options = options[:4]
    return options


def _normalize_mc_answer(raw: Any, options: list[str]) -> int:
    if raw is None:
        return 0
    if isinstance(raw, (int, float)):
        idx = int(raw)
        if 0 <= idx < len(options):
            return idx
        return 0
    text = str(raw).strip()
    if text.isdigit():
        idx = int(text)
        if 0 <= idx < len(options):
            return idx
        return 0
    if len(text) == 1 and text.isalpha():
        idx = ord(text.upper()) - ord("A")
        if 0 <= idx < len(options):
            return idx
        return 0
    if options:
        for idx, option in enumerate(options):
            if option.strip().lower() == text.lower():
                return idx
    return 0


def _normalize_tf_answer(raw: Any) -> str:
    text = str(raw).strip().lower()
    return "true" if text in {"true", "1", "yes", "y"} else "false"


def _plan_value(item: Any, key: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        value = item.get(key, default)
    else:
        value = getattr(item, key, default)
    return getattr(value, "value", value)


def _plan_has_value(item: Any, key: str) -> bool:
    if isinstance(item, dict):
        return key in item and item.get(key) is not None
    return getattr(item, key, None) is not None


def _plan_int(item: Any, key: str, default: int) -> int:
    value = _plan_value(item, key, default)
    if value is None:
        value = default
    return int(value)


def _coerce_generation_plan(
    *,
    num_questions: int,
    question_types: Sequence[Any] | None = None,
    question_plan: Sequence[Any] | None = None,
) -> list[dict[str, Any]]:
    if question_plan:
        plan: list[dict[str, Any]] = []
        seen_types: set[str] = set()
        for item in question_plan:
            q_type = _normalize_question_type(_plan_value(item, "question_type"))
            if q_type not in SUPPORTED_GENERATED_QUESTION_TYPES:
                raise ValueError(f"Unsupported generated question type: {q_type}")
            if q_type in seen_types:
                raise ValueError("question_plan cannot contain duplicate question_type rows")
            seen_types.add(q_type)
            count = _plan_int(item, "count", 0)
            if count <= 0:
                raise ValueError("question_plan count must be positive")
            row: dict[str, Any] = {"question_type": q_type, "count": count}
            if q_type in {"multiple_choice", "multi_select"}:
                if _plan_has_value(item, "pair_count"):
                    raise ValueError("pair_count is only valid for matching questions")
                option_count = _plan_int(item, "option_count", 4)
                if not 2 <= option_count <= 6:
                    raise ValueError("option_count must be between 2 and 6")
                row["option_count"] = option_count
            elif q_type == "matching":
                if _plan_has_value(item, "option_count"):
                    raise ValueError("option_count is not valid for matching questions")
                pair_count = _plan_int(item, "pair_count", 4)
                if not 2 <= pair_count <= 6:
                    raise ValueError("pair_count must be between 2 and 6")
                row["pair_count"] = pair_count
            elif _plan_has_value(item, "option_count") or _plan_has_value(item, "pair_count"):
                raise ValueError("option_count and pair_count are not valid for this question_type")
            plan.append(row)
        if sum(item["count"] for item in plan) != int(num_questions):
            raise ValueError("question_plan counts must sum to num_questions")
        return plan

    types = _coerce_question_types(question_types)
    base, extra = divmod(max(0, int(num_questions)), len(types))
    return [
        {"question_type": q_type, "count": base + (1 if index < extra else 0)}
        for index, q_type in enumerate(types)
        if base or index < extra
    ]


def _normalize_planned_mc_answer(raw: Any, options: list[str]) -> int:
    if raw is None or isinstance(raw, bool):
        raise ValueError("multiple_choice correct_answer must be an option index or letter")
    if isinstance(raw, int):
        idx = raw
    else:
        text = str(raw).strip()
        if text.isdigit():
            idx = int(text)
        elif len(text) == 1 and text.isalpha():
            idx = ord(text.upper()) - ord("A")
        else:
            for option_idx, option in enumerate(options):
                if option.strip().lower() == text.lower():
                    return option_idx
            raise ValueError("multiple_choice correct_answer did not match any option")
    if 0 <= idx < len(options):
        return idx
    raise ValueError("multiple_choice correct_answer index is out of range")


def _normalize_planned_multi_select_answer(raw: Any, options: list[str]) -> list[int]:
    if not isinstance(raw, list) or not raw:
        raise ValueError("multi_select correct_answer must be a non-empty index array")
    indices: list[int] = []
    for item in raw:
        if isinstance(item, bool) or not isinstance(item, int):
            raise ValueError("multi_select correct_answer indices must be integers")
        if item < 0 or item >= len(options):
            raise ValueError("multi_select correct_answer index is out of range")
        indices.append(item)
    if len(set(indices)) != len(indices):
        raise ValueError("multi_select correct_answer indices must be unique")
    return sorted(indices)


def _normalize_planned_matching_answer(raw: Any, options: list[str]) -> dict[str, str]:
    if not isinstance(raw, dict):
        raise ValueError("matching correct_answer must map each option to an answer")
    if set(raw) != set(options):
        raise ValueError("matching correct_answer must include exactly the left-side options")
    normalized = {option: str(raw[option]).strip() for option in options}
    if any(not value for value in normalized.values()):
        raise ValueError("matching correct_answer values must be non-empty")
    if len(set(normalized.values())) != len(normalized):
        raise ValueError("matching correct_answer values must be unique")
    return normalized


def _normalize_planned_question(
    raw: Any,
    plan_item: Any,
    *,
    default_source_type: str = "media",
    default_source_id: str = "generated",
) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("Generated question must be an object")
    q_type = _normalize_question_type(raw.get("question_type"))
    expected_type = _normalize_question_type(_plan_value(plan_item, "question_type"))
    if q_type != expected_type:
        raise ValueError(f"Expected {expected_type} question, got {q_type}")

    question_text = str(raw.get("question_text") or raw.get("question") or "").strip()
    if not question_text:
        raise ValueError("question_text is required")

    options: list[str] | None = None
    correct_answer: int | str | list[int] | dict[str, str]
    if q_type == "multiple_choice":
        option_count = int(_plan_value(plan_item, "option_count", 4) or 4)
        options = _coerce_options(raw.get("options"), expected_count=option_count)
        correct_answer = _normalize_planned_mc_answer(raw.get("correct_answer"), options)
    elif q_type == "multi_select":
        option_count = int(_plan_value(plan_item, "option_count", 4) or 4)
        options = _coerce_options(raw.get("options"), expected_count=option_count)
        correct_answer = _normalize_planned_multi_select_answer(raw.get("correct_answer"), options)
    elif q_type == "matching":
        pair_count = int(_plan_value(plan_item, "pair_count", 4) or 4)
        options = _coerce_options(raw.get("options"), expected_count=pair_count)
        correct_answer = _normalize_planned_matching_answer(raw.get("correct_answer"), options)
    elif q_type == "true_false":
        correct_answer = raw.get("correct_answer")
        if correct_answer not in {"true", "false"}:
            raise ValueError('true_false correct_answer must be exactly "true" or "false"')
    elif q_type == "fill_blank":
        if "___" not in question_text:
            raise ValueError("fill_blank question_text must contain ___")
        correct_answer = str(raw.get("correct_answer") or "").strip()
        if not correct_answer:
            raise ValueError("fill_blank correct_answer is required")
    else:
        raise ValueError(f"Unsupported generated question type: {q_type}")

    try:
        points_val = int(raw.get("points", 1))
    except (TypeError, ValueError):
        points_val = 1
    try:
        hint_penalty_points = max(0, int(raw.get("hint_penalty_points", 0)))
    except (TypeError, ValueError):
        hint_penalty_points = 0

    return {
        "question_type": q_type,
        "question_text": question_text,
        "options": options,
        "correct_answer": correct_answer,
        "explanation": str(raw.get("explanation") or "").strip() or None,
        "hint": str(raw.get("hint") or "").strip() or None,
        "hint_penalty_points": hint_penalty_points,
        "source_citations": _coerce_source_citations(
            raw.get("source_citations"),
            default_source_type=default_source_type,
            default_source_id=default_source_id,
        ),
        "points": points_val if points_val >= 0 else 1,
    }


def _format_question_plan_instructions(plan: Sequence[dict[str, Any]]) -> str:
    rows: list[str] = []
    for item in plan:
        q_type = item["question_type"]
        if q_type in {"multiple_choice", "multi_select"}:
            rows.append(
                f"- {q_type}: {item['count']} question(s), exactly {item['option_count']} options"
            )
        elif q_type == "matching":
            rows.append(f"- {q_type}: {item['count']} question(s), exactly {item['pair_count']} pairs")
        else:
            rows.append(f"- {q_type}: {item['count']} question(s)")
    return "\n".join(
        [
            "Planned question requirements:",
            *rows,
            "",
            "Planned output shapes:",
            '- multiple_choice: {"question_type": "multiple_choice", '
            '"options": ["A", "..."], "correct_answer": 0}',
            '- multi_select: {"question_type": "multi_select", '
            '"options": ["A", "..."], "correct_answer": [0, 2]}',
            '- matching: {"question_type": "matching", "options": ["CPU", "RAM"], '
            '"correct_answer": {"CPU": "Processor", "RAM": "Memory"}}',
            '- true_false: {"question_type": "true_false", "correct_answer": "true" | "false"}',
            '- fill_blank: {"question_type": "fill_blank", '
            '"question_text": "The ___ executes instructions.", "correct_answer": "CPU"}',
        ]
    )


def _remove_legacy_shape_hints(prompt: str) -> str:
    replacements = {
        '"question_type": "multiple_choice" | "true_false" | "fill_blank"': (
            '"question_type": "multiple_choice" | "multi_select" | "matching" | '
            '"true_false" | "fill_blank"'
        ),
        '"options": ["A", "B", "C", "D"]': '"options": ["A", "..."]',
        '"correct_answer": 0 | 1 | 2 | 3 | "true" | "false" | "the answer"': (
            '"correct_answer": 0 | [0, 2] | {"left": "right"} | '
            '"true" | "false" | "the answer"'
        ),
        "- For multiple_choice: options must be array of 4 strings, correct_answer is 0-based index (0-3)": (
            "- For multiple_choice: options count must match the planned option_count, "
            "correct_answer is a 0-based index"
        ),
    }
    for old, new in replacements.items():
        prompt = prompt.replace(old, new)
    return prompt


def _format_quiz_generation_prompt(
    *,
    num_questions: int,
    content: str,
    difficulty: str,
    question_types: Sequence[Any] | None,
    focus_instruction: str,
    source_contract: str,
    question_plan: Sequence[Any] | None = None,
) -> str:
    plan = _coerce_generation_plan(
        num_questions=num_questions,
        question_types=question_types,
        question_plan=question_plan,
    )
    prompt = QUIZ_GENERATION_PROMPT.format(
        num_questions=num_questions,
        content=content,
        difficulty=difficulty,
        question_types=", ".join(item["question_type"] for item in plan),
        focus_instruction=focus_instruction,
        source_contract=source_contract,
    )
    if not question_plan:
        return prompt
    prompt = _remove_legacy_shape_hints(prompt)
    return f"{prompt}\n\n{_format_question_plan_instructions(plan)}"


def _coerce_source_citations(
    raw: Any,
    default_source_type: str,
    default_source_id: str,
) -> list[dict[str, Any]] | None:
    entries: list[dict[str, Any]] = []
    if isinstance(raw, list):
        candidates = raw
    elif isinstance(raw, dict):
        candidates = [raw]
    elif isinstance(raw, str) and raw.strip():
        candidates = [{"quote": raw.strip()}]
    else:
        candidates = []

    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            continue
        label = str(candidate.get("label") or "").strip() or None
        quote = str(candidate.get("quote") or candidate.get("excerpt") or "").strip() or None
        chunk_id = str(candidate.get("chunk_id") or candidate.get("chunkId") or "").strip() or None
        source_url = str(candidate.get("source_url") or candidate.get("url") or "").strip() or None
        timestamp_raw = candidate.get("timestamp_seconds", candidate.get("timestamp"))
        timestamp_seconds: float | None = None
        if isinstance(timestamp_raw, (int, float)):
            timestamp_seconds = float(max(0, timestamp_raw))

        source_type = str(candidate.get("source_type") or default_source_type).strip()
        source_id = str(candidate.get("source_id") or default_source_id).strip()
        if not source_type or not source_id:
            continue

        media_ref: int | None = None
        media_id_raw = candidate.get("media_id")
        if isinstance(media_id_raw, (int, float)):
            media_id_candidate = int(media_id_raw)
            if media_id_candidate > 0:
                media_ref = media_id_candidate
        elif source_type == "media":
            with contextlib.suppress(TypeError, ValueError):
                parsed_media_id = int(source_id)
                if parsed_media_id > 0:
                    media_ref = parsed_media_id

        citation: dict[str, Any] = {
            "source_type": source_type,
            "source_id": source_id,
        }
        if media_ref is not None:
            citation["media_id"] = media_ref
        if label:
            citation["label"] = label
        elif quote:
            citation["label"] = f"Source {index + 1}"
        if quote:
            citation["quote"] = quote
        if chunk_id:
            citation["chunk_id"] = chunk_id
        if timestamp_seconds is not None:
            citation["timestamp_seconds"] = timestamp_seconds
        if source_url:
            citation["source_url"] = source_url
        entries.append(citation)

    return entries or None


def _extract_json_payload(raw: Any) -> Any:
    if isinstance(raw, (dict, list)):
        return raw
    if raw is None:
        raise ValueError("LLM response was empty")
    text = str(raw).strip()
    if not text:
        raise ValueError("LLM response was empty")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for open_char, close_char in (("{", "}"), ("[", "]")):
        start_idx = text.find(open_char)
        end_idx = text.rfind(close_char)
        if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
            continue
        snippet = text[start_idx:end_idx + 1]
        try:
            return json.loads(snippet)
        except json.JSONDecodeError:
            continue
    raise ValueError("Failed to parse quiz JSON from LLM response")


def _normalize_questions(
    raw_questions: Sequence[Any],
    default_source_type: str,
    default_source_id: str,
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for raw in raw_questions:
        if not isinstance(raw, dict):
            continue
        q_type = _normalize_question_type(raw.get("question_type"))
        if q_type not in DEFAULT_QUESTION_TYPES:
            continue
        question_text = str(raw.get("question_text") or raw.get("question") or "").strip()
        if not question_text:
            continue
        points = raw.get("points", 1)
        try:
            points_val = int(points)
        except (TypeError, ValueError):
            points_val = 1
        hint_penalty_raw = raw.get("hint_penalty_points", 0)
        try:
            hint_penalty_points = max(0, int(hint_penalty_raw))
        except (TypeError, ValueError):
            hint_penalty_points = 0
        hint = str(raw.get("hint") or "").strip() or None
        source_citations = _coerce_source_citations(
            raw.get("source_citations"),
            default_source_type=default_source_type,
            default_source_id=default_source_id,
        )

        options: list[str] | None = None
        correct_answer: int | str
        if q_type == "multiple_choice":
            options = _coerce_options(raw.get("options"))
            correct_answer = _normalize_mc_answer(raw.get("correct_answer"), options)
        elif q_type == "true_false":
            correct_answer = _normalize_tf_answer(raw.get("correct_answer"))
        else:
            correct_answer = str(raw.get("correct_answer") or "").strip()

        normalized.append(
            {
                "question_type": q_type,
                "question_text": question_text,
                "options": options,
                "correct_answer": correct_answer,
                "explanation": str(raw.get("explanation") or "").strip() or None,
                "hint": hint,
                "hint_penalty_points": hint_penalty_points,
                "source_citations": source_citations,
                "points": points_val if points_val >= 0 else 1,
            }
        )
    return normalized


def _normalize_planned_questions(
    raw_questions: Sequence[Any],
    plan: Sequence[dict[str, Any]],
    default_source_type: str,
    default_source_id: str,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {item["question_type"]: [] for item in plan}
    first_error: tuple[int, str, str] | None = None
    expected_total = sum(int(item["count"]) for item in plan)

    for index, raw in enumerate(raw_questions, start=1):
        if not isinstance(raw, dict):
            continue
        q_type = _normalize_question_type(raw.get("question_type"))
        plan_item = next((item for item in plan if item["question_type"] == q_type), None)
        if plan_item is None:
            continue
        try:
            grouped[q_type].append(
                _normalize_planned_question(
                    raw,
                    plan_item,
                    default_source_type=default_source_type,
                    default_source_id=default_source_id,
                )
            )
        except ValueError as exc:
            if first_error is None:
                first_error = (index, str(q_type), str(exc))
            continue

    for item in plan:
        q_type = item["question_type"]
        expected = int(item["count"])
        got = len(grouped[q_type])
        if got != expected:
            if first_error is not None:
                error_index, error_type, error_detail = first_error
                raise ValueError(f"Question {error_index} {error_type} invalid: {error_detail}")
            raise ValueError(f"Generated {q_type} count mismatch: expected {expected}, got {got}")

    got_total = sum(len(items) for items in grouped.values())
    if got_total != expected_total or len(raw_questions) != expected_total:
        if first_error is not None:
            error_index, error_type, error_detail = first_error
            raise ValueError(f"Question {error_index} {error_type} invalid: {error_detail}")
        raise ValueError(f"Generated question plan mismatch: expected {expected_total}, got {len(raw_questions)}")

    return [question for item in plan for question in grouped[item["question_type"]]]


def _normalize_sources(sources: Sequence[Any]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for item in sources:
        if isinstance(item, dict):
            source_type = str(item.get("source_type") or "").strip()
            source_id = str(item.get("source_id") or "").strip()
        else:
            source_type = str(getattr(item, "source_type", "") or "").strip()
            source_id = str(getattr(item, "source_id", "") or "").strip()
        if not source_type or not source_id:
            raise ValueError("Each source must include non-empty source_type and source_id")
        normalized.append({"source_type": source_type, "source_id": source_id})
    if not normalized:
        raise ValueError("At least one source is required")
    return normalized


def _build_content_from_evidence(evidence_items: Sequence[dict[str, Any]]) -> str:
    blocks: list[str] = []
    remaining = MAX_CONTENT_CHARS

    for item in evidence_items:
        source_type = str(item.get("source_type") or "").strip()
        source_id = str(item.get("source_id") or "").strip()
        text = str(item.get("text") or "").strip()
        if not source_type or not source_id or not text:
            continue
        chunk_id = str(item.get("chunk_id") or "").strip()
        label = str(item.get("label") or "").strip()
        header = f"Source: {source_type}:{source_id}"
        if chunk_id:
            header += f" (chunk: {chunk_id})"
        if label:
            header += f" [{label}]"
        block = f"{header}\n{text}"
        if len(block) > remaining:
            block = block[:remaining]
        blocks.append(block)
        remaining -= len(block) + 2
        if remaining <= 0:
            break

    content = "\n\n".join(blocks).strip()
    if not content:
        raise ValueError("Resolved sources contained no usable content")
    return content


def _validate_strict_provenance(questions: Sequence[dict[str, Any]], selected_sources: Sequence[dict[str, str]]) -> None:
    allowed_sources = {(s["source_type"], s["source_id"]) for s in selected_sources}
    if not allowed_sources:
        raise QuizProvenanceValidationError("No selected sources available for provenance validation")

    for index, question in enumerate(questions):
        citations = question.get("source_citations")
        if not isinstance(citations, list) or not citations:
            raise QuizProvenanceValidationError(
                f"Question {index + 1} is missing required source_citations"
            )

        for citation in citations:
            if not isinstance(citation, dict):
                raise QuizProvenanceValidationError(
                    f"Question {index + 1} has source citations that do not map to selected sources"
                )
            source_type = str(citation.get("source_type") or "").strip()
            source_id = str(citation.get("source_id") or "").strip()
            if (source_type, source_id) not in allowed_sources:
                raise QuizProvenanceValidationError(
                    f"Question {index + 1} has source citations that do not map to selected sources"
                )


def _build_source_contract(selected_sources: Sequence[dict[str, str]]) -> str:
    source_refs = ", ".join(f"{s['source_type']}:{s['source_id']}" for s in selected_sources)
    return f"- Allowed sources for source_citations.source_type/source_id: {source_refs}"


def _truncate_quiz_evidence(text: str, limit: int = 120) -> str:
    """Collapse evidence text to a stable, citation-friendly excerpt."""
    normalized = " ".join(str(text or "").split()).strip()
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[: limit - 1].rstrip()}…"


def _build_test_mode_questions(
    *,
    evidence: Sequence[dict[str, Any]],
    normalized_sources: Sequence[dict[str, str]],
    num_questions: int,
    question_types: Sequence[Any] | None,
    question_plan: Sequence[Any] | None = None,
) -> list[dict[str, Any]]:
    """Build deterministic quiz questions that preserve evidence provenance in test mode."""
    if question_plan:
        plan = _coerce_generation_plan(
            num_questions=num_questions,
            question_types=question_types,
            question_plan=question_plan,
        )
        planned_types = [
            (item["question_type"], copy_index, item)
            for item in plan
            for copy_index in range(int(item["count"]))
        ]
        total_questions = len(planned_types)
    else:
        normalized_types = _coerce_question_types(question_types)
        total_questions = max(1, num_questions)
        planned_types = [
            (normalized_types[index % len(normalized_types)], index, {})
            for index in range(total_questions)
        ]
    questions: list[dict[str, Any]] = []

    for index in range(total_questions):
        source = normalized_sources[index % len(normalized_sources)]
        evidence_item = evidence[index % len(evidence)] if evidence else {}
        citation_source_type = str(evidence_item.get("source_type") or source["source_type"]).strip()
        citation_source_id = str(evidence_item.get("source_id") or source["source_id"]).strip()
        excerpt = _truncate_quiz_evidence(
            str(
                evidence_item.get("text")
                or f"Study point from {citation_source_type}:{citation_source_id}."
            )
        )
        citation = {
            "source_type": citation_source_type,
            "source_id": citation_source_id,
            "label": f"Source {index + 1}",
            "quote": excerpt,
        }
        question_type, copy_index, plan_item = planned_types[index]

        if question_type == "multiple_choice":
            option_count = int(plan_item.get("option_count", 4) or 4)
            questions.append(
                {
                    "question_type": "multiple_choice",
                    "question_text": (
                        f"Which statement is directly supported by "
                        f"{citation_source_type}:{citation_source_id}?"
                    ),
                    "options": [
                        excerpt,
                        "A conflicting claim with no evidence.",
                        "An empty workspace selection.",
                        "A discarded draft artifact.",
                    ][:option_count] + [f"Unused distractor {option_idx}" for option_idx in range(5, option_count + 1)],
                    "correct_answer": 0,
                    "explanation": "The first option quotes the selected source evidence.",
                    "hint": "Look for the excerpt copied from the selected source.",
                    "hint_penalty_points": 0,
                    "source_citations": [citation],
                    "points": 1,
                }
            )
            continue

        if question_type == "multi_select":
            option_count = int(plan_item.get("option_count", 4) or 4)
            questions.append(
                {
                    "question_type": "multi_select",
                    "question_text": (
                        f"Which statements are supported by {citation_source_type}:{citation_source_id}?"
                    ),
                    "options": [
                        excerpt,
                        f"{citation_source_type}:{citation_source_id} is one selected source.",
                        "A claim from an unselected source.",
                        "A statement with no citation.",
                    ][:option_count] + [f"Unused distractor {option_idx}" for option_idx in range(5, option_count + 1)],
                    "correct_answer": [0, 1],
                    "explanation": "The first two options are grounded in the selected source.",
                    "hint": "Choose only options tied to the citation.",
                    "hint_penalty_points": 0,
                    "source_citations": [citation],
                    "points": 1,
                }
            )
            continue

        if question_type == "matching":
            pair_count = int(plan_item.get("pair_count", 4) or 4)
            options = [f"Term {copy_index + 1}.{pair_index + 1}" for pair_index in range(pair_count)]
            questions.append(
                {
                    "question_type": "matching",
                    "question_text": f"Match each term supported by {citation_source_type}:{citation_source_id}.",
                    "options": options,
                    "correct_answer": {
                        option: f"Match {copy_index + 1}.{pair_index + 1}"
                        for pair_index, option in enumerate(options)
                    },
                    "explanation": "Deterministic matching placeholder for planned test-mode coverage.",
                    "hint": "Pair each term with the same numbered match.",
                    "hint_penalty_points": 0,
                    "source_citations": [citation],
                    "points": 1,
                }
            )
            continue

        if question_type == "true_false":
            questions.append(
                {
                    "question_type": "true_false",
                    "question_text": f"True or false: {excerpt}",
                    "options": None,
                    "correct_answer": "true",
                    "explanation": "The statement is taken directly from the selected source evidence.",
                    "hint": "This test-mode prompt quotes the source text verbatim.",
                    "hint_penalty_points": 0,
                    "source_citations": [citation],
                    "points": 1,
                }
            )
            continue

        questions.append(
            {
                "question_type": "fill_blank",
                "question_text": f"Fill in the blank: ___ {excerpt}",
                "options": None,
                "correct_answer": "Review",
                "explanation": "Deterministic fill-in placeholder for test-mode coverage.",
                "hint": "The missing word is a generic study cue.",
                "hint_penalty_points": 0,
                "source_citations": [citation],
                "points": 1,
            }
        )

    return questions


async def _call_quiz_generation_llm(
    *,
    prompt: str,
    model: str | None = None,
    api_provider: str | None = None,
    max_tokens: int = 2000,
) -> Any:
    provider = (api_provider or DEFAULT_LLM_PROVIDER or "openai").strip().lower()
    api_key, _debug = resolve_provider_api_key(provider, prefer_module_keys_in_tests=True)
    if provider_requires_api_key(provider) and not api_key:
        raise ValueError(f"Provider '{provider}' requires an API key.")

    messages_payload = [{"role": "user", "content": prompt}]
    response_format = {"type": "json_object"}

    def _call_llm():
        adapter = _get_adapter(provider)
        app_config = load_and_log_configs() or {}
        model_to_use = _resolve_model(provider, model, app_config)
        if model_to_use is None:
            raise ChatConfigurationError(provider=provider, message="Model is required for provider.")
        return adapter.chat(
            {
                "messages": messages_payload,
                "api_key": api_key,
                "model": model_to_use,
                "temperature": 0.3,
                "max_tokens": max_tokens,
                "response_format": response_format,
                "app_config": app_config,
            }
        )

    start = time.time()
    raw_response = await asyncio.get_running_loop().run_in_executor(None, _call_llm)
    logger.info("Quiz generation LLM call completed in {:.1f}ms", (time.time() - start) * 1000.0)
    return raw_response


_ORIGINAL_CALL_QUIZ_GENERATION_LLM = _call_quiz_generation_llm


def _should_use_deterministic_test_mode() -> bool:
    """Keep deterministic test-mode behavior unless a test explicitly patches the LLM call."""
    return is_test_mode() and _call_quiz_generation_llm is _ORIGINAL_CALL_QUIZ_GENERATION_LLM


def _resolve_primary_media_id(normalized_sources: Sequence[dict[str, str]]) -> int | None:
    for source in normalized_sources:
        if source["source_type"] != "media":
            continue
        with contextlib.suppress(TypeError, ValueError):
            media_candidate = int(source["source_id"])
            if media_candidate > 0:
                return media_candidate
    return None


def _resolve_quiz_title_from_media(media_db: MediaDatabase, primary_media_id: int | None) -> str:
    if primary_media_id is None:
        return "Mixed Sources"

    media = media_db.get_media_by_id(primary_media_id, include_deleted=False, include_trash=False)
    if not media:
        return "Mixed Sources"
    return str(media.get("title") or "").strip() or f"Media #{primary_media_id}"


def _is_remediation_source_set(normalized_sources: Sequence[dict[str, str]]) -> bool:
    if not normalized_sources:
        return False
    return all(
        source.get("source_type") in {"quiz_attempt", "quiz_attempt_question"}
        for source in normalized_sources
    )


def _resolve_generated_quiz_metadata(
    *,
    media_db: MediaDatabase,
    normalized_sources: Sequence[dict[str, str]],
    primary_media_id: int | None,
) -> tuple[str, str]:
    if _is_remediation_source_set(normalized_sources):
        return ("Remediation", "Auto-generated remediation quiz from missed questions")

    return (
        _resolve_quiz_title_from_media(media_db, primary_media_id),
        "Auto-generated quiz from selected sources",
    )


def _persist_generated_quiz(
    *,
    db: CharactersRAGDB,
    normalized_sources: list[dict[str, str]],
    questions: list[dict[str, Any]],
    quiz_title: str,
    quiz_description: str,
    primary_media_id: int | None,
    workspace_id: str | None,
    workspace_tag: str | None,
) -> dict[str, Any]:
    quiz_id = db.create_quiz(
        name=f"Quiz: {quiz_title}" if quiz_title else "Quiz: Mixed Sources",
        description=quiz_description,
        workspace_id=workspace_id,
        workspace_tag=workspace_tag,
        media_id=primary_media_id,
        source_bundle_json=normalized_sources,
    )
    for idx, question in enumerate(questions):
        db.create_question(
            quiz_id=quiz_id,
            question_type=question["question_type"],
            question_text=question["question_text"],
            correct_answer=question["correct_answer"],
            options=question.get("options"),
            explanation=question.get("explanation"),
            hint=question.get("hint"),
            hint_penalty_points=question.get("hint_penalty_points", 0),
            source_citations=question.get("source_citations"),
            points=question.get("points", 1),
            order_index=idx,
        )

    quiz = db.get_quiz(quiz_id)
    if not quiz:
        raise ValueError("Failed to load generated quiz")
    questions_payload = db.list_questions(quiz_id, include_answers=True, limit=None, offset=0)
    return {
        "quiz": quiz,
        "questions": questions_payload.get("items", []),
    }


async def generate_quiz_from_sources(
    *,
    db: CharactersRAGDB,
    media_db: MediaDatabase,
    sources: Sequence[Any],
    num_questions: int = 10,
    question_types: list[Any] | None = None,
    difficulty: str = "mixed",
    focus_topics: list[str] | None = None,
    question_plan: Sequence[Any] | None = None,
    model: str | None = None,
    api_provider: str | None = None,
    workspace_id: str | None = None,
    workspace_tag: str | None = None,
) -> dict[str, Any]:
    """Generate a quiz from mixed sources (media, notes, flashcard decks/cards)."""
    normalized_sources = _normalize_sources(sources)
    evidence = await asyncio.to_thread(
        resolve_quiz_sources,
        normalized_sources,
        db=db,
        media_db=media_db,
    )

    plan = _coerce_generation_plan(
        num_questions=num_questions,
        question_types=question_types,
        question_plan=question_plan,
    )
    normalized_types = [item["question_type"] for item in plan]
    focus_instruction = ""
    if focus_topics:
        focus_instruction = f"- Focus on these topics: {', '.join(t for t in focus_topics if t)}"
    source_contract = _build_source_contract(normalized_sources)
    primary_media_id = _resolve_primary_media_id(normalized_sources)
    quiz_title, quiz_description = await asyncio.to_thread(
        _resolve_generated_quiz_metadata,
        media_db=media_db,
        normalized_sources=normalized_sources,
        primary_media_id=primary_media_id,
    )

    if _should_use_deterministic_test_mode():
        questions = _build_test_mode_questions(
            evidence=evidence,
            normalized_sources=normalized_sources,
            num_questions=num_questions,
            question_types=normalized_types,
            question_plan=plan if question_plan else None,
        )
        _validate_strict_provenance(questions, normalized_sources)
        return await asyncio.to_thread(
            _persist_generated_quiz,
            db=db,
            normalized_sources=normalized_sources,
            questions=questions,
            quiz_title=quiz_title,
            quiz_description=quiz_description,
            primary_media_id=primary_media_id,
            workspace_id=workspace_id,
            workspace_tag=workspace_tag,
        )

    content = _build_content_from_evidence(evidence)

    prompt = _format_quiz_generation_prompt(
        num_questions=num_questions,
        content=content,
        difficulty=difficulty,
        question_types=normalized_types,
        focus_instruction=focus_instruction,
        source_contract=source_contract,
        question_plan=plan if question_plan else None,
    )

    llm_kwargs: dict[str, Any] = {
        "prompt": prompt,
        "model": model,
        "max_tokens": min(8000, max(2000, num_questions * 220)),
    }
    if api_provider:
        llm_kwargs["api_provider"] = api_provider
    raw_response = await _call_quiz_generation_llm(**llm_kwargs)
    content_text = extract_response_content(raw_response)
    payload = _extract_json_payload(content_text if content_text is not None else raw_response)
    raw_questions = payload.get("questions") if isinstance(payload, dict) else payload
    if not isinstance(raw_questions, list):
        raise ValueError("LLM response did not include a questions list")

    default_source = normalized_sources[0]
    if question_plan:
        questions = _normalize_planned_questions(
            raw_questions,
            plan,
            default_source_type=default_source["source_type"],
            default_source_id=default_source["source_id"],
        )
    else:
        questions = _normalize_questions(
            raw_questions,
            default_source_type=default_source["source_type"],
            default_source_id=default_source["source_id"],
        )
        if num_questions and len(questions) > num_questions:
            questions = questions[:num_questions]
    if not questions:
        raise ValueError("No valid questions generated")
    _validate_strict_provenance(questions, normalized_sources)

    return await asyncio.to_thread(
        _persist_generated_quiz,
        db=db,
        normalized_sources=normalized_sources,
        questions=questions,
        quiz_title=quiz_title,
        quiz_description=quiz_description,
        primary_media_id=primary_media_id,
        workspace_id=workspace_id,
        workspace_tag=workspace_tag,
    )


async def generate_quiz_from_media(
    *,
    db: CharactersRAGDB,
    media_db: MediaDatabase,
    media_id: int,
    num_questions: int = 10,
    question_types: list[Any] | None = None,
    difficulty: str = "mixed",
    focus_topics: list[str] | None = None,
    model: str | None = None,
    api_provider: str | None = None,
    workspace_id: str | None = None,
    workspace_tag: str | None = None,
) -> dict[str, Any]:
    """Backward-compatible wrapper for legacy media-only generation requests."""
    return await generate_quiz_from_sources(
        db=db,
        media_db=media_db,
        sources=[{"source_type": "media", "source_id": str(media_id)}],
        num_questions=num_questions,
        question_types=question_types,
        difficulty=difficulty,
        focus_topics=focus_topics,
        model=model,
        api_provider=api_provider,
        workspace_id=workspace_id,
        workspace_tag=workspace_tag,
    )
