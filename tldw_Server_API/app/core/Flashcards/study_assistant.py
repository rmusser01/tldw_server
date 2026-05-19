"""Context assembly and prompt helpers for flashcard and quiz study assistance."""

from __future__ import annotations

import json
import re
from typing import Any, Mapping

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, ConflictError, InputError
from tldw_Server_API.app.core.StudyPacks.provenance import FlashcardProvenanceStore

try:
    from tldw_Server_API.app.core.Chat.chat_service import perform_chat_api_call_async
except ImportError:  # pragma: no cover - fallback for limited test imports
    async def perform_chat_api_call_async(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise ImportError("chat_service_unavailable")

from tldw_Server_API.app.core.Workflows.adapters._common import extract_openai_content

STUDY_ASSISTANT_ACTIONS = ("explain", "mnemonic", "follow_up", "fact_check", "freeform")
DEFAULT_MAX_HISTORY_MESSAGES = 8
DEFAULT_MAX_FIELD_CHARS = 1200


def normalize_study_assistant_action(action: str) -> str:
    """Normalize and validate a study-assistant action name."""
    normalized = str(action or "").strip().lower()
    if normalized not in STUDY_ASSISTANT_ACTIONS:
        raise InputError(  # noqa: TRY003
            f"Invalid study assistant action '{action}'. Allowed: {', '.join(STUDY_ASSISTANT_ACTIONS)}"
        )
    return normalized


def _truncate_text(value: Any, max_chars: int) -> str | None:
    if value is None:
        return None
    text = str(value)
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    if max_chars <= 3:
        return text[:max_chars]
    return text[: max_chars - 3] + "..."


def _normalize_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item is not None]
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return [stripped]
        if isinstance(parsed, list):
            return [str(item) for item in parsed if item is not None]
    return [str(value)]


def _normalize_history(messages: list[dict[str, Any]], *, max_history_messages: int, max_field_chars: int) -> list[dict[str, Any]]:
    recent = messages[-max(0, int(max_history_messages)):] if max_history_messages >= 0 else list(messages)
    normalized: list[dict[str, Any]] = []
    for item in recent:
        normalized.append(
            {
                "id": item.get("id"),
                "thread_id": item.get("thread_id"),
                "role": item.get("role"),
                "action_type": item.get("action_type"),
                "input_modality": item.get("input_modality"),
                "content": _truncate_text(item.get("content"), max_field_chars),
                "structured_payload": item.get("structured_payload") or {},
                "context_snapshot": item.get("context_snapshot") or {},
                "provider": item.get("provider"),
                "model": item.get("model"),
                "created_at": item.get("created_at"),
                "client_id": item.get("client_id"),
            }
        )
    return normalized


def _normalize_thread(thread: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(thread)
    normalized["deleted"] = bool(normalized.get("deleted"))
    return normalized


def build_flashcard_assistant_context(
    db: CharactersRAGDB,
    flashcard_uuid: str,
    *,
    max_history_messages: int = DEFAULT_MAX_HISTORY_MESSAGES,
    max_field_chars: int = DEFAULT_MAX_FIELD_CHARS,
) -> dict[str, Any]:
    """Build server-side assistant context for a single flashcard."""
    flashcard = db.get_flashcard(flashcard_uuid)
    if not flashcard:
        raise ConflictError("Flashcard not found", entity="flashcards", identifier=flashcard_uuid)  # noqa: TRY003

    thread = db.get_or_create_study_assistant_thread(
        context_type="flashcard",
        flashcard_uuid=flashcard_uuid,
    )
    history = db.list_study_assistant_messages(thread["id"])
    provenance_store = FlashcardProvenanceStore(db)
    provenance = provenance_store.read_flashcard_provenance(flashcard_uuid)
    study_pack_summary = provenance_store.get_study_pack_summary(flashcard_uuid)

    existing_context = {
        "context_type": "flashcard",
        "thread": _normalize_thread(thread),
        "flashcard": {
            "uuid": str(flashcard.get("uuid")),
            "deck_id": flashcard.get("deck_id"),
            "deck_name": flashcard.get("deck_name"),
            "front": _truncate_text(flashcard.get("front"), max_field_chars),
            "back": _truncate_text(flashcard.get("back"), max_field_chars),
            "notes": _truncate_text(flashcard.get("notes"), max_field_chars),
            "extra": _truncate_text(flashcard.get("extra"), max_field_chars),
            "tags": _normalize_string_list(flashcard.get("tags_json")),
            "source_ref_type": flashcard.get("source_ref_type"),
            "source_ref_id": flashcard.get("source_ref_id"),
            "model_type": flashcard.get("model_type"),
            "reverse": bool(flashcard.get("reverse")),
            "version": flashcard.get("version"),
        },
        "history": _normalize_history(
            history,
            max_history_messages=max_history_messages,
            max_field_chars=max_field_chars,
        ),
        "available_actions": list(STUDY_ASSISTANT_ACTIONS),
    }
    return {
        **existing_context,
        "study_pack": study_pack_summary,
        "citations": provenance["citations"],
        "primary_citation": provenance["primary_citation"],
        "deep_dive_target": provenance["deep_dive_target"],
    }


def build_quiz_attempt_question_context(
    db: CharactersRAGDB,
    quiz_attempt_id: int,
    question_id: int,
    *,
    max_history_messages: int = DEFAULT_MAX_HISTORY_MESSAGES,
    max_field_chars: int = DEFAULT_MAX_FIELD_CHARS,
) -> dict[str, Any]:
    """Build server-side assistant context for one quiz-attempt question."""
    attempt = db.get_attempt(int(quiz_attempt_id), include_questions=True, include_answers=True)
    if not attempt:
        raise ConflictError("Quiz attempt not found", entity="quiz_attempts", identifier=quiz_attempt_id)  # noqa: TRY003

    questions = attempt.get("questions") or []
    question = next((item for item in questions if int(item.get("id")) == int(question_id)), None)
    if not question:
        raise ConflictError("Quiz question not found in attempt", entity="quiz_questions", identifier=question_id)  # noqa: TRY003

    answers = attempt.get("answers") or []
    answer = next((item for item in answers if int(item.get("question_id")) == int(question_id)), None)
    thread = db.get_or_create_study_assistant_thread(
        context_type="quiz_attempt_question",
        quiz_attempt_id=int(quiz_attempt_id),
        question_id=int(question_id),
    )
    history = db.list_study_assistant_messages(thread["id"])

    return {
        "context_type": "quiz_attempt_question",
        "thread": _normalize_thread(thread),
        "attempt": {
            "id": int(attempt.get("id")),
            "quiz_id": attempt.get("quiz_id"),
            "score": attempt.get("score"),
            "total_possible": attempt.get("total_possible"),
            "completed_at": attempt.get("completed_at"),
        },
        "question": {
            "id": int(question.get("id")),
            "question_type": question.get("question_type"),
            "question_text": _truncate_text(question.get("question_text"), max_field_chars),
            "options": question.get("options") or [],
            "correct_answer": question.get("correct_answer"),
            "explanation": _truncate_text(question.get("explanation"), max_field_chars),
            "source_citations": question.get("source_citations") or [],
            "user_answer": answer.get("user_answer") if answer else None,
            "is_correct": bool(answer.get("is_correct")) if answer else None,
            "points_awarded": answer.get("points_awarded") if answer else None,
            "hint_used": bool(answer.get("hint_used")) if answer else False,
        },
        "history": _normalize_history(
            history,
            max_history_messages=max_history_messages,
            max_field_chars=max_field_chars,
        ),
        "available_actions": list(STUDY_ASSISTANT_ACTIONS),
    }


def normalize_fact_check_payload(raw_payload: Mapping[str, Any] | None, *, assistant_text: str = "") -> dict[str, Any]:
    """Normalize fact-check structured payloads to the required response contract."""
    payload = dict(raw_payload or {})
    verdict = str(payload.get("verdict") or "partially_correct").strip().lower()
    if verdict not in {"correct", "partially_correct", "incorrect"}:
        verdict = "partially_correct"

    corrections = _normalize_string_list(payload.get("corrections"))
    missing_points = _normalize_string_list(payload.get("missing_points"))
    next_prompt = str(payload.get("next_prompt") or "").strip()
    if not next_prompt:
        if verdict == "correct":
            next_prompt = "What related detail would you like to reinforce next?"
        elif missing_points:
            next_prompt = f"Want to review this next: {missing_points[0]}"
        elif corrections:
            next_prompt = f"Want to practice this correction next: {corrections[0]}"
        elif assistant_text:
            next_prompt = f"Want to turn this into a quick recall prompt: {assistant_text[:80]}"
        else:
            next_prompt = "What part would you like to review next?"

    return {
        "verdict": verdict,
        "corrections": corrections,
        "missing_points": missing_points,
        "next_prompt": next_prompt,
    }


def build_study_assistant_prompt_package(
    *,
    action: str,
    context: Mapping[str, Any],
    user_message: str | None = None,
) -> dict[str, str]:
    """Build a narrow action-specific prompt package for downstream LLM calls."""
    normalized_action = normalize_study_assistant_action(action)
    context_type = str(context.get("context_type") or "flashcard")
    if context_type == "quiz_attempt_question":
        target = context.get("question") or {}
        anchor_text = target.get("question_text") or ""
    else:
        target = context.get("flashcard") or {}
        anchor_text = target.get("front") or ""

    action_instruction = {
        "explain": "Explain the material clearly and stay anchored to the provided study context only.",
        "mnemonic": "Offer one memorable mnemonic tied directly to the provided study context only.",
        "follow_up": "Answer the follow-up question using only the provided study context and thread history.",
        "fact_check": "Fact-check the learner explanation against the provided study context and return structured corrections.",
        "freeform": "Answer the learner message using only the provided study context and keep the response concise.",
    }[normalized_action]

    return {
        "action": normalized_action,
        "system_prompt": (
            "You are a focused study assistant. Stay strictly within the provided flashcard or quiz-question context. "
            "Do not broaden to unrelated material or invent external facts. "
            f"{action_instruction}"
        ),
        "user_prompt": (
            f"Context type: {context_type}\n"
            f"Anchor: {anchor_text}\n"
            f"Learner message: {(user_message or '').strip() or '[none provided]'}"
        ),
    }


def _extract_json_object(text: str) -> dict[str, Any] | None:
    stripped = str(text or "").strip()
    if not stripped:
        return None
    candidates = [stripped]
    match = re.search(r"\{[\s\S]*\}", stripped)
    if match:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _render_fact_check_text(payload: Mapping[str, Any]) -> str:
    verdict = str(payload.get("verdict") or "partially_correct").replace("_", " ")
    corrections = payload.get("corrections") or []
    missing_points = payload.get("missing_points") or []
    parts = [f"Verdict: {verdict}."]
    if corrections:
        parts.append("Corrections: " + " ".join(str(item) for item in corrections))
    if missing_points:
        parts.append("Missing points: " + " ".join(str(item) for item in missing_points))
    next_prompt = str(payload.get("next_prompt") or "").strip()
    if next_prompt:
        parts.append(next_prompt)
    return " ".join(part for part in parts if part).strip()


async def generate_study_assistant_reply(
    *,
    action: str,
    context: Mapping[str, Any],
    message: str | None = None,
    provider: str | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """Generate an assistant reply for the provided card/question context."""
    prompt_package = build_study_assistant_prompt_package(
        action=action,
        context=context,
        user_message=message,
    )
    normalized_action = str(prompt_package["action"])
    system_prompt = prompt_package["system_prompt"]
    if normalized_action == "fact_check":
        system_prompt += (
            " Return a JSON object with keys: verdict, corrections, missing_points, next_prompt, response_text."
        )

    response = await perform_chat_api_call_async(
        messages=[{"role": "user", "content": prompt_package["user_prompt"]}],
        api_provider=provider,
        model=model,
        system_message=system_prompt,
        max_tokens=1000,
        temperature=0.3,
    )
    response_text = (extract_openai_content(response) or "").strip()
    resolved_provider = provider or "default"
    resolved_model = model

    if normalized_action == "fact_check":
        parsed = _extract_json_object(response_text) or {}
        assistant_text = str(parsed.get("response_text") or "").strip()
        structured_payload = normalize_fact_check_payload(parsed, assistant_text=assistant_text or response_text)
        if not assistant_text:
            assistant_text = _render_fact_check_text(structured_payload)
    else:
        structured_payload = {}
        assistant_text = response_text or "I couldn't generate a study response."

    return {
        "assistant_text": assistant_text.strip(),
        "structured_payload": structured_payload,
        "provider": resolved_provider,
        "model": resolved_model,
    }
