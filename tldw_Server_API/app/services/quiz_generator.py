from __future__ import annotations

import asyncio
import contextlib
import json
import time
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from loguru import logger

from tldw_Server_API.app.api.v1.schemas.chat_request_schemas import DEFAULT_LLM_PROVIDER
from tldw_Server_API.app.core.Chat.Chat_Deps import ChatConfigurationError
from tldw_Server_API.app.core.Chat.chat_helpers import extract_response_content
from tldw_Server_API.app.core.Chat.chat_service import resolve_provider_api_key
from tldw_Server_API.app.core.config import load_and_log_configs
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.exceptions import BadRequestError
from tldw_Server_API.app.core.LLM_Calls.adapter_registry import get_registry
from tldw_Server_API.app.core.LLM_Calls.provider_metadata import provider_requires_api_key
from tldw_Server_API.app.core.testing import is_test_mode
from tldw_Server_API.app.services.quiz_source_resolver import resolve_quiz_sources

if TYPE_CHECKING:
    from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase

DEFAULT_QUESTION_TYPES = ["multiple_choice", "true_false", "fill_blank"]
SUPPORTED_GENERATED_QUESTION_TYPES = [
    "multiple_choice",
    "multi_select",
    "matching",
    "true_false",
    "fill_blank",
]
DEFAULT_GENERATION_PROFILE = "standard_recall"
BEST_OF_FIVE_TAG = "best_of_five"
ASSERTION_REASONING_TAG = "assertion_reasoning"
ASSERTION_REASONING_OPTIONS = (
    "Both the assertion and reason are true, and the reason correctly explains the assertion.",
    "Both the assertion and reason are true, but the reason does not explain the assertion.",
    "The assertion is true, but the reason is false.",
    "The assertion is false, but the reason is true.",
    "Both the assertion and reason are false.",
)
MAX_CONTENT_CHARS = 15000
MAX_EMQ_GROUP_ID_LENGTH = 128
MAX_EMQ_GROUP_PROMPT_LENGTH = 2000
MAX_EMQ_OPTIONS = 10
MAX_ASSERTION_REASONING_TEXT_LENGTH = 2000

_QUIZ_GENERATION_PROFILES: list[dict[str, Any]] = [
    {
        "id": "standard_recall",
        "label": "Standard Recall",
        "description": "Balanced source-grounded recall and application questions.",
        "status": "available",
        "default_num_questions": 10,
        "default_difficulty": "mixed",
        "default_question_types": DEFAULT_QUESTION_TYPES,
        "allowed_question_types": SUPPORTED_GENERATED_QUESTION_TYPES,
        "prompt_instruction": "Use concise recall and application questions across the selected question types.",
    },
    {
        "id": "mixed_assessment",
        "label": "Mixed Assessment",
        "description": "A broader mix of recall, interpretation, and applied understanding.",
        "status": "available",
        "default_num_questions": 10,
        "default_difficulty": "mixed",
        "default_question_types": DEFAULT_QUESTION_TYPES,
        "allowed_question_types": SUPPORTED_GENERATED_QUESTION_TYPES,
        "prompt_instruction": "Mix recall, interpretation, and applied understanding while preserving citations.",
    },
    {
        "id": "best_of_five",
        "label": "Best of Five",
        "description": "Single-best-answer questions with five plausible options.",
        "status": "available",
        "default_num_questions": 5,
        "default_difficulty": "mixed",
        "default_question_types": ["multiple_choice"],
        "allowed_question_types": ["multiple_choice"],
        "prompt_instruction": (
            "Best of Five: every question must be multiple_choice with exactly five answer options, "
            "one best answer, and plausible distractors."
        ),
    },
    {
        "id": "emq",
        "label": "EMQ",
        "description": "Extended matching questions with shared option banks.",
        "status": "available",
        "default_num_questions": 5,
        "default_difficulty": "mixed",
        "default_question_types": ["multiple_choice"],
        "allowed_question_types": ["multiple_choice"],
        "prompt_instruction": (
            "Create each group with one shared option bank and at least two stems; repeat the same "
            "group_id, group_prompt, and options on every multiple_choice stem."
        ),
    },
    {
        "id": "assertion_reasoning",
        "label": "Assertion / Reasoning",
        "description": "Assertion and reason pairs with concise evidence-backed rationales.",
        "status": "available",
        "default_num_questions": 5,
        "default_difficulty": "mixed",
        "default_question_types": ["multiple_choice"],
        "allowed_question_types": ["multiple_choice"],
        "prompt_instruction": (
            "Provide separate assertion and reason fields, then classify each pair using exactly one "
            "canonical outcome: A. Both the assertion and reason are true, and the reason correctly "
            "explains the assertion. B. Both the assertion and reason are true, but the reason does not "
            "explain the assertion. C. The assertion is true, but the reason is false. D. The assertion "
            "is false, but the reason is true. E. Both the assertion and reason are false. Include a "
            "concise evidence-backed rationale. Do not provide hidden chain-of-thought."
        ),
    },
    {
        "id": "osce_scenario",
        "label": "OSCE Scenario",
        "description": "Scenario practice with checklist and rubric feedback.",
        "status": "planned",
        "default_num_questions": 3,
        "default_difficulty": "mixed",
        "default_question_types": ["fill_blank"],
        "allowed_question_types": ["fill_blank"],
        "prompt_instruction": "",
    },
]
_PROFILE_BY_ID = {profile["id"]: profile for profile in _QUIZ_GENERATION_PROFILES}


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
      "assertion": "Optional assertion for assertion_reasoning",
      "reason": "Optional reason for assertion_reasoning",
      "group_id": "Optional EMQ group identifier",
      "group_prompt": "Optional shared EMQ group prompt",
      "options": ["A", "B", "C", "D", "E if required by the profile"],
      "correct_answer": 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | "true" | "false" | "the answer",
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
      "tags": ["optional topic or difficulty tag"],
      "points": 1
    }}
  ]
}}

Important:
- For multiple_choice: options must be an array of answer strings, correct_answer is the 0-based index
- For Best of Five: multiple_choice options must be exactly 5 strings
- For EMQ: create at least two stems per group; repeat one nonempty group_id, group_prompt, and shared bank of 2-10 options on every stem; correct_answer is a 0-based index into that bank
- For Assertion / Reasoning: use multiple_choice, provide separate assertion and reason fields, use the canonical A-E outcomes from the profile instruction, and include a concise evidence-backed explanation. Do not provide hidden chain-of-thought, reasoning_steps, or chain_of_thought fields
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


def _normalize_generation_profile(value: Any) -> str:
    raw = getattr(value, "value", value)
    text = str(raw or DEFAULT_GENERATION_PROFILE).strip().lower().replace("-", "_")
    aliases = {
        "standard": "standard_recall",
        "recall": "standard_recall",
        "mixed": "mixed_assessment",
        "bof": "best_of_five",
        "best of five": "best_of_five",
        "best-of-five": "best_of_five",
        "assertion reasoning": "assertion_reasoning",
        "assertion/reasoning": "assertion_reasoning",
        "osce": "osce_scenario",
    }
    profile_id = aliases.get(text, text)
    profile = _PROFILE_BY_ID.get(profile_id)
    if profile is None:
        raise BadRequestError(f"Unknown quiz generation profile: {raw}")
    if profile["status"] != "available":
        raise BadRequestError(f"Quiz generation profile '{profile_id}' is not available yet")
    return profile_id


def get_quiz_generation_profiles() -> list[dict[str, Any]]:
    return [
        {key: value for key, value in profile.items() if key not in {"allowed_question_types", "prompt_instruction"}}
        for profile in _QUIZ_GENERATION_PROFILES
    ]


def _build_generation_profile_instruction(generation_profile: Any) -> str:
    profile_id = _normalize_generation_profile(generation_profile)
    profile = _PROFILE_BY_ID[profile_id]
    return f"- Generation profile: {profile['label']}. {profile['prompt_instruction']}"


def _coerce_question_types(
    question_types: Sequence[Any] | None,
    *,
    generation_profile: Any = DEFAULT_GENERATION_PROFILE,
) -> list[str]:
    profile = _PROFILE_BY_ID[_normalize_generation_profile(generation_profile)]
    defaults = list(profile["default_question_types"])
    allowed_types = set(profile["allowed_question_types"])
    if not question_types:
        return defaults
    normalized: list[str] = []
    for item in question_types:
        raw = getattr(item, "value", item)
        q_type = _normalize_question_type(raw)
        if q_type in SUPPORTED_GENERATED_QUESTION_TYPES and q_type in allowed_types and q_type not in normalized:
            normalized.append(q_type)
    return normalized or defaults


def _coerce_options(
    raw: Any,
    expected_count: int | None = None,
    *,
    max_options: int | None = 4,
) -> list[str]:
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
    if max_options is not None and len(options) > max_options:
        options = options[:max_options]
    return options


def _coerce_question_tags(raw: Any, *, generation_profile: Any) -> list[str] | None:
    tags: list[str] = []
    seen: set[str] = set()
    profile_id = _normalize_generation_profile(generation_profile)

    if isinstance(raw, list):
        candidates = raw
    elif isinstance(raw, str) and raw.strip():
        candidates = [part.strip() for part in raw.replace(";", ",").split(",")]
    else:
        candidates = []

    for candidate in candidates:
        tag = str(candidate).strip()
        if not tag:
            continue
        normalized = tag.lower().replace("-", "_").replace(" ", "_")
        reserved_normalized = "_".join(
            tag.lower().replace("-", " ").replace("_", " ").replace("/", " ").split()
        )
        if reserved_normalized in {BEST_OF_FIVE_TAG, "bof"}:
            if profile_id != "best_of_five":
                continue
            tag = BEST_OF_FIVE_TAG
            normalized = BEST_OF_FIVE_TAG
        elif reserved_normalized == ASSERTION_REASONING_TAG:
            if profile_id != ASSERTION_REASONING_TAG:
                continue
            tag = ASSERTION_REASONING_TAG
            normalized = ASSERTION_REASONING_TAG
        if normalized in seen:
            continue
        seen.add(normalized)
        tags.append(tag)

    if profile_id == "best_of_five" and BEST_OF_FIVE_TAG not in seen:
        tags.append(BEST_OF_FIVE_TAG)
    elif profile_id == ASSERTION_REASONING_TAG and ASSERTION_REASONING_TAG not in seen:
        tags.append(ASSERTION_REASONING_TAG)

    return tags or None


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
    if options and len(text) == 1 and "A" <= text.upper() <= chr(ord("A") + len(options) - 1):
        idx = ord(text.upper()) - ord("A")
        if 0 <= idx < len(options):
            return idx
        return 0
    if options:
        for idx, option in enumerate(options):
            if option.strip().lower() == text.lower():
                return idx
    return 0


def _normalize_emq_mc_answer(raw: Any, options: list[str]) -> int:
    if isinstance(raw, bool):
        raise ValueError("EMQ correct_answer must be a valid option index, letter, or exact label")
    if isinstance(raw, int):
        if 0 <= raw < len(options):
            return raw
        raise ValueError("EMQ correct_answer index is out of range")
    if not isinstance(raw, str):
        raise ValueError("EMQ correct_answer must be a valid option index, letter, or exact label")

    text = raw.strip()
    candidates: set[int] = set()
    if text.isdigit():
        index = int(text)
        if 0 <= index < len(options):
            candidates.add(index)
    if len(text) == 1 and "A" <= text.upper() <= chr(ord("A") + len(options) - 1):
        candidates.add(ord(text.upper()) - ord("A"))
    candidates.update(index for index, option in enumerate(options) if option == text)

    if len(candidates) == 1:
        return candidates.pop()
    if candidates:
        raise ValueError("EMQ correct_answer is ambiguous")
    raise ValueError("EMQ correct_answer must be a valid option index, letter, or exact label")


def _normalize_assertion_reasoning_answer(raw: Any) -> int:
    if isinstance(raw, bool):
        raise ValueError("Assertion / Reasoning correct_answer must be an integer, A-E letter, or exact label")
    if isinstance(raw, int):
        if 0 <= raw < len(ASSERTION_REASONING_OPTIONS):
            return raw
        raise ValueError("Assertion / Reasoning correct_answer index is out of range")
    if not isinstance(raw, str):
        raise ValueError("Assertion / Reasoning correct_answer must be an integer, A-E letter, or exact label")

    text = raw.strip()
    if len(text) == 1 and "A" <= text.upper() <= "E":
        return ord(text.upper()) - ord("A")

    normalized = text.casefold()
    for index, option in enumerate(ASSERTION_REASONING_OPTIONS):
        if option.casefold() == normalized:
            return index
    raise ValueError("Assertion / Reasoning correct_answer must be an integer, A-E letter, or exact label")


def _require_assertion_reasoning_text(raw: Any, field: str) -> str:
    if not isinstance(raw, str):
        raise ValueError(f"Assertion / Reasoning {field} must contain 1-2000 characters")
    text = raw.strip()
    if not text or len(text) > MAX_ASSERTION_REASONING_TEXT_LENGTH:
        raise ValueError(f"Assertion / Reasoning {field} must contain 1-2000 characters")
    return text


def _validate_assertion_reasoning_questions(questions: Sequence[dict[str, Any]]) -> None:
    if not questions:
        raise ValueError("Assertion / Reasoning generation must include at least one question")

    assertion_prefix = "**Assertion:** "
    reason_separator = "\n\n**Reason:** "
    for question in questions:
        if question.get("question_type") != "multiple_choice":
            raise ValueError("Assertion / Reasoning questions must use the multiple_choice question type")

        question_text = question.get("question_text")
        if not isinstance(question_text, str) or not question_text.startswith(assertion_prefix):
            raise ValueError("Assertion / Reasoning question_text must contain labeled assertion and reason text")
        assertion, separator, reason = question_text[len(assertion_prefix) :].partition(reason_separator)
        if not separator:
            raise ValueError("Assertion / Reasoning question_text must contain labeled assertion and reason text")
        _require_assertion_reasoning_text(assertion, "assertion")
        _require_assertion_reasoning_text(reason, "reason")
        _require_assertion_reasoning_text(question.get("explanation"), "explanation")

        if question.get("options") != list(ASSERTION_REASONING_OPTIONS):
            raise ValueError("Assertion / Reasoning options must use the canonical A-E scale")
        correct_answer = question.get("correct_answer")
        if (
            isinstance(correct_answer, bool)
            or not isinstance(correct_answer, int)
            or not 0 <= correct_answer < len(ASSERTION_REASONING_OPTIONS)
        ):
            raise ValueError("Assertion / Reasoning correct_answer must be a zero-based integer from 0 to 4")
        if question.get("group_id") is not None or question.get("group_prompt") is not None:
            raise ValueError("Assertion / Reasoning group_id and group_prompt must be null")

        tags = question.get("tags")
        normalized_tags = _coerce_question_tags(tags, generation_profile=ASSERTION_REASONING_TAG)
        if tags != normalized_tags:
            raise ValueError("Assertion / Reasoning questions must include exactly one canonical subtype tag")


def _validate_emq_groups(questions: Sequence[dict[str, Any]]) -> None:
    groups: dict[str, dict[str, Any]] = {}
    normalized_answers: list[tuple[dict[str, Any], int]] = []
    for question in questions:
        if question.get("question_type") != "multiple_choice":
            raise ValueError("EMQ questions must use the multiple_choice question type")

        group_id = str(question.get("group_id") or "").strip()
        if not group_id or len(group_id) > MAX_EMQ_GROUP_ID_LENGTH:
            raise ValueError(f"EMQ group_id must contain 1-{MAX_EMQ_GROUP_ID_LENGTH} characters")

        group_prompt = str(question.get("group_prompt") or "").strip()
        if not group_prompt or len(group_prompt) > MAX_EMQ_GROUP_PROMPT_LENGTH:
            raise ValueError(
                f"EMQ group_prompt must contain 1-{MAX_EMQ_GROUP_PROMPT_LENGTH} characters"
            )

        explanation = str(question.get("explanation") or "").strip()
        if not explanation:
            raise ValueError("Each EMQ stem must include a nonempty explanation")

        options = question.get("options")
        if not isinstance(options, list) or not 2 <= len(options) <= MAX_EMQ_OPTIONS:
            raise ValueError(f"EMQ options must contain 2-{MAX_EMQ_OPTIONS} entries")

        normalized_answers.append(
            (question, _normalize_emq_mc_answer(question.get("correct_answer"), options))
        )
        group = groups.setdefault(
            group_id,
            {"group_prompt": group_prompt, "options": options, "count": 0},
        )
        if group["group_prompt"] != group_prompt:
            raise ValueError("Every stem in an EMQ group must use the same group_prompt")
        if group["options"] != options:
            raise ValueError("Every stem in an EMQ group must use the same option bank")
        group["count"] += 1

    if not groups:
        raise ValueError("EMQ generation must include at least one group")
    if any(group["count"] < 2 for group in groups.values()):
        raise ValueError("Every EMQ group must include at least two stems")
    for question, correct_answer in normalized_answers:
        question["correct_answer"] = correct_answer


def _normalize_tf_answer(raw: Any) -> str:
    text = str(raw).strip().lower()
    return "true" if text in {"true", "1", "yes", "y"} else "false"


def _plan_value(item: Any, key: str, default: Any = None) -> Any:
    """Read a plan field from either a dict or a Pydantic model."""
    if isinstance(item, dict):
        value = item.get(key, default)
    else:
        value = getattr(item, key, default)
    return getattr(value, "value", value)


def _plan_has_value(item: Any, key: str) -> bool:
    """Return true when a plan field was explicitly provided."""
    if isinstance(item, dict):
        return key in item and item.get(key) is not None
    return getattr(item, key, None) is not None


def _plan_int(item: Any, key: str, default: int) -> int:
    """Read a plan integer field with the service-level default applied."""
    value = _plan_value(item, key, default)
    if value is None:
        value = default
    return int(value)


def _coerce_generation_plan(
    *,
    num_questions: int,
    question_types: Sequence[Any] | None = None,
    question_plan: Sequence[Any] | None = None,
    generation_profile: Any = DEFAULT_GENERATION_PROFILE,
) -> list[dict[str, Any]]:
    """Normalize legacy question types or a structured question plan."""
    profile_id = _normalize_generation_profile(generation_profile)
    allowed_types = set(_PROFILE_BY_ID[profile_id]["allowed_question_types"])
    if question_plan:
        if profile_id not in {"standard_recall", "mixed_assessment"}:
            raise ValueError(
                "question_plan is only supported for standard_recall and mixed_assessment profiles"
            )
        plan: list[dict[str, Any]] = []
        seen_types: set[str] = set()
        for item in question_plan:
            q_type = _normalize_question_type(_plan_value(item, "question_type"))
            if q_type not in SUPPORTED_GENERATED_QUESTION_TYPES:
                raise ValueError(f"Unsupported generated question type: {q_type}")
            if q_type not in allowed_types:
                raise ValueError(
                    f"Question type '{q_type}' is not allowed for generation profile '{profile_id}'"
                )
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
                option_count = 5 if profile_id == "best_of_five" else _plan_int(item, "option_count", 4)
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

    types = _coerce_question_types(question_types, generation_profile=profile_id)
    base, extra = divmod(max(0, int(num_questions)), len(types))
    return [
        {"question_type": q_type, "count": base + (1 if index < extra else 0)}
        for index, q_type in enumerate(types)
        if base or index < extra
    ]


def _normalize_planned_mc_answer(raw: Any, options: list[str]) -> int:
    """Normalize a planned multiple-choice answer to a zero-based option index."""
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
    """Normalize planned multi-select answers to sorted zero-based option indices."""
    if not isinstance(raw, list) or not raw:
        raise ValueError("multi_select correct_answer must be a non-empty index array")
    indices: list[int] = []
    for item in raw:
        if item is None or isinstance(item, bool):
            raise ValueError("multi_select correct_answer entries must be indices or letters")
        if isinstance(item, int):
            idx = item
        else:
            text = str(item).strip()
            if text.isdigit():
                idx = int(text)
            elif len(text) == 1 and text.isalpha():
                idx = ord(text.upper()) - ord("A")
            else:
                raise ValueError("multi_select correct_answer entries must be indices or letters")
        if idx < 0 or idx >= len(options):
            raise ValueError("multi_select correct_answer index is out of range")
        indices.append(idx)
    if len(set(indices)) != len(indices):
        raise ValueError("multi_select correct_answer indices must be unique")
    return sorted(indices)


def _normalize_planned_matching_answer(raw: Any, options: list[str]) -> dict[str, str]:
    """Normalize planned matching answers using the canonical option labels."""
    if not isinstance(raw, dict):
        raise ValueError("matching correct_answer must map each option to an answer")
    option_by_key = {option.lower(): option for option in options}
    if len(option_by_key) != len(options):
        raise ValueError("matching options must be unique case-insensitively")
    normalized: dict[str, str] = {}
    for raw_key, raw_value in raw.items():
        canonical_key = option_by_key.get(str(raw_key).strip().lower())
        if canonical_key is None or canonical_key in normalized:
            raise ValueError("matching correct_answer must include exactly the left-side options")
        normalized[canonical_key] = str(raw_value).strip()
    if set(normalized) != set(options):
        raise ValueError("matching correct_answer must include exactly the left-side options")
    if any(not value for value in normalized.values()):
        raise ValueError("matching correct_answer values must be non-empty")
    if len(set(normalized.values())) != len(normalized):
        raise ValueError("matching correct_answer values must be unique")
    return {option: normalized[option] for option in options}


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
    """Render extra prompt instructions for exact planned question counts."""
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
    """Replace legacy fixed-shape prompt hints with planned-generation hints."""
    replacements = {
        '"question_type": "multiple_choice" | "true_false" | "fill_blank"': (
            '"question_type": "multiple_choice" | "multi_select" | "matching" | '
            '"true_false" | "fill_blank"'
        ),
        '"options": ["A", "B", "C", "D"]': '"options": ["A", "..."]',
        '"correct_answer": 0 | 1 | 2 | 3 | "true" | "false" | "the answer"': (
            '"correct_answer": 0 | [0, 2] | {{"left": "right"}} | '
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
    generation_profile: Any = DEFAULT_GENERATION_PROFILE,
) -> str:
    """Render the quiz generation prompt, including structured plan instructions."""
    plan = _coerce_generation_plan(
        num_questions=num_questions,
        question_types=question_types,
        question_plan=question_plan,
        generation_profile=generation_profile,
    )
    template = QUIZ_GENERATION_PROMPT
    if question_plan:
        template = _remove_legacy_shape_hints(template)
    prompt = template.format(
        num_questions=num_questions,
        content=content,
        difficulty=difficulty,
        question_types=", ".join(item["question_type"] for item in plan),
        focus_instruction=focus_instruction,
        source_contract=source_contract,
    )
    if not question_plan:
        return prompt
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
    generation_profile: Any = DEFAULT_GENERATION_PROFILE,
) -> list[dict[str, Any]]:
    profile_id = _normalize_generation_profile(generation_profile)
    is_emq = profile_id == "emq"
    is_assertion_reasoning = profile_id == ASSERTION_REASONING_TAG
    mc_option_count = None if is_emq else 5 if profile_id == "best_of_five" else 4
    normalized: list[dict[str, Any]] = []
    for raw in raw_questions:
        if not isinstance(raw, dict):
            if is_emq or is_assertion_reasoning:
                raise ValueError(f"Each {profile_id} question must be a JSON object")
            continue
        q_type = _normalize_question_type(raw.get("question_type"))
        if is_assertion_reasoning and q_type != "multiple_choice":
            raise ValueError("Assertion / Reasoning questions must use the multiple_choice question type")
        if q_type not in DEFAULT_QUESTION_TYPES and not is_emq:
            continue
        if is_assertion_reasoning:
            assertion = _require_assertion_reasoning_text(raw.get("assertion"), "assertion")
            reason = _require_assertion_reasoning_text(raw.get("reason"), "reason")
            question_text = f"**Assertion:** {assertion}\n\n**Reason:** {reason}"
            explanation = _require_assertion_reasoning_text(
                raw.get("explanation"),
                "explanation",
            )
        else:
            question_text = str(raw.get("question_text") or raw.get("question") or "").strip()
            if not question_text:
                if is_emq:
                    raise ValueError("Each EMQ stem must include question_text")
                continue
            explanation = str(raw.get("explanation") or "").strip() or None
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
        tags = _coerce_question_tags(raw.get("tags"), generation_profile=profile_id)

        options: list[str] | None = None
        correct_answer: Any
        if q_type == "multiple_choice":
            if is_assertion_reasoning:
                options = list(ASSERTION_REASONING_OPTIONS)
                correct_answer = _normalize_assertion_reasoning_answer(raw.get("correct_answer"))
            else:
                options = _coerce_options(raw.get("options"), max_options=mc_option_count)
                if profile_id == "best_of_five" and len(options) != 5:
                    raise ValueError("Best-of-Five questions must include exactly 5 options")
                correct_answer = (
                    raw.get("correct_answer")
                    if is_emq
                    else _normalize_mc_answer(raw.get("correct_answer"), options)
                )
        elif q_type == "true_false":
            correct_answer = _normalize_tf_answer(raw.get("correct_answer"))
        elif q_type == "fill_blank":
            correct_answer = str(raw.get("correct_answer") or "").strip()
        else:
            correct_answer = raw.get("correct_answer")

        question_payload = {
            "question_type": q_type,
            "question_text": question_text,
            "group_id": (str(raw.get("group_id") or "").strip() or None) if is_emq else None,
            "group_prompt": (str(raw.get("group_prompt") or "").strip() or None) if is_emq else None,
            "options": options,
            "correct_answer": correct_answer,
            "explanation": explanation,
            "hint": hint,
            "hint_penalty_points": hint_penalty_points,
            "source_citations": source_citations,
            "points": points_val if points_val >= 0 else 1,
        }
        if tags:
            question_payload["tags"] = tags
        normalized.append(question_payload)
    if is_assertion_reasoning:
        _validate_assertion_reasoning_questions(normalized)
    if is_emq:
        _validate_emq_groups(normalized)
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


def _limit_questions_by_profile(
    questions: Sequence[dict[str, Any]],
    *,
    num_questions: int,
    generation_profile: Any,
) -> list[dict[str, Any]]:
    limited = list(questions)
    if not num_questions or len(limited) <= num_questions:
        return limited
    if _normalize_generation_profile(generation_profile) != "emq":
        return limited[:num_questions]

    selected_group_ids = {
        str(question.get("group_id") or "").strip()
        for question in limited[:num_questions]
    }
    return [
        question
        for question in limited
        if str(question.get("group_id") or "").strip() in selected_group_ids
    ]


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
    generation_profile: Any = DEFAULT_GENERATION_PROFILE,
) -> list[dict[str, Any]]:
    """Build deterministic quiz questions that preserve evidence provenance in test mode."""
    profile_id = _normalize_generation_profile(generation_profile)
    if question_plan:
        plan = _coerce_generation_plan(
            num_questions=num_questions,
            question_types=question_types,
            question_plan=question_plan,
            generation_profile=profile_id,
        )
        planned_types = [
            (item["question_type"], copy_index, item)
            for item in plan
            for copy_index in range(int(item["count"]))
        ]
        total_questions = len(planned_types)
    else:
        normalized_types = _coerce_question_types(
            question_types,
            generation_profile=profile_id,
        )
        total_questions = max(2 if profile_id == "emq" else 1, num_questions)
        planned_types = [
            (normalized_types[index % len(normalized_types)], index, {})
            for index in range(total_questions)
        ]
    questions: list[dict[str, Any]] = []
    emq_options = [
        "Supported by the selected source evidence.",
        "Contradicted by the selected source evidence.",
        "Not addressed by the selected source evidence.",
        "Requires evidence from a different source.",
    ]
    emq_group_prompt = "Choose the option that best characterizes the evidence for each stem."

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
            if profile_id == ASSERTION_REASONING_TAG:
                options = list(ASSERTION_REASONING_OPTIONS)
                question_text = (
                    f"**Assertion:** {excerpt}\n\n**Reason:** The selected source directly supports this assertion."
                )
                explanation = (
                    "Both statements are true and the reason explains the assertion because the "
                    "citation quotes the selected source evidence."
                )
            elif profile_id == "emq":
                options = list(emq_options)
                question_text = (
                    f"Stem {index + 1}: how is this claim characterized by "
                    f"{citation_source_type}:{citation_source_id}?"
                )
                explanation = (
                    f"Stem {index + 1} is supported because its citation quotes the selected source evidence."
                )
            else:
                option_count = (
                    5
                    if profile_id == "best_of_five"
                    else int(plan_item.get("option_count", 4) or 4)
                )
                options = [
                    excerpt,
                    "A conflicting claim with no evidence.",
                    "An empty workspace selection.",
                    "A discarded draft artifact.",
                ]
                if option_count >= 5:
                    options.append("A plausible but unsupported alternate answer.")
                if option_count > len(options):
                    options.extend(
                        f"Unused distractor {option_idx}"
                        for option_idx in range(len(options) + 1, option_count + 1)
                    )
                options = options[:option_count]
                question_text = (
                    f"Which statement is the best answer supported by "
                    f"{citation_source_type}:{citation_source_id}?"
                )
                explanation = "The first option is the best answer because it quotes the selected source evidence."
            question_payload = {
                "question_type": "multiple_choice",
                "question_text": question_text,
                "group_id": "emq-test-1" if profile_id == "emq" else None,
                "group_prompt": emq_group_prompt if profile_id == "emq" else None,
                "options": options,
                "correct_answer": 0,
                "explanation": explanation,
                "hint": "Look for the excerpt copied from the selected source.",
                "hint_penalty_points": 0,
                "source_citations": [citation],
                "points": 1,
            }
            tags = _coerce_question_tags(None, generation_profile=profile_id)
            if tags:
                question_payload["tags"] = tags
            questions.append(question_payload)
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
            tags=question.get("tags"),
            group_id=question.get("group_id"),
            group_prompt=question.get("group_prompt"),
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
    generation_profile: Any = DEFAULT_GENERATION_PROFILE,
    difficulty: str = "mixed",
    focus_topics: list[str] | None = None,
    question_plan: Sequence[Any] | None = None,
    model: str | None = None,
    api_provider: str | None = None,
    workspace_id: str | None = None,
    workspace_tag: str | None = None,
) -> dict[str, Any]:
    """Generate a quiz from mixed sources (media, notes, flashcard decks/cards)."""
    normalized_profile = _normalize_generation_profile(generation_profile)
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
        generation_profile=normalized_profile,
    )
    normalized_types = [item["question_type"] for item in plan]
    focus_instructions = [_build_generation_profile_instruction(normalized_profile)]
    if focus_topics:
        focus_instructions.append(f"- Focus on these topics: {', '.join(t for t in focus_topics if t)}")
    focus_instruction = "\n".join(focus_instructions)
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
            generation_profile=normalized_profile,
        )
        questions = _limit_questions_by_profile(
            questions,
            num_questions=num_questions,
            generation_profile=normalized_profile,
        )
        if normalized_profile == "emq":
            _validate_emq_groups(questions)
        _validate_strict_provenance(questions, normalized_sources)
        if normalized_profile == ASSERTION_REASONING_TAG:
            _validate_assertion_reasoning_questions(questions)
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

    prompt_question_count = max(2, num_questions) if normalized_profile == "emq" else num_questions
    prompt = _format_quiz_generation_prompt(
        num_questions=prompt_question_count,
        content=content,
        difficulty=difficulty,
        question_types=normalized_types,
        focus_instruction=focus_instruction,
        source_contract=source_contract,
        question_plan=plan if question_plan else None,
        generation_profile=normalized_profile,
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
    if question_plan and normalized_profile in {"standard_recall", "mixed_assessment"}:
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
            generation_profile=normalized_profile,
        )
        questions = _limit_questions_by_profile(
            questions,
            num_questions=num_questions,
            generation_profile=normalized_profile,
        )
    if not questions:
        raise ValueError("No valid questions generated")
    if normalized_profile == "emq":
        _validate_emq_groups(questions)
    _validate_strict_provenance(questions, normalized_sources)
    if normalized_profile == ASSERTION_REASONING_TAG:
        _validate_assertion_reasoning_questions(questions)

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
    generation_profile: Any = DEFAULT_GENERATION_PROFILE,
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
        generation_profile=generation_profile,
        difficulty=difficulty,
        focus_topics=focus_topics,
        model=model,
        api_provider=api_provider,
        workspace_id=workspace_id,
        workspace_tag=workspace_tag,
    )
