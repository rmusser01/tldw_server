# flashcards.py
# REST endpoints for Flashcards/Decks backed by ChaChaNotes DB (schema v5)

import json
import os
import re
from typing import Any, Optional

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from fastapi.responses import Response, StreamingResponse
from loguru import logger
from pydantic import BaseModel

from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_owner, get_chacha_db_for_user
from tldw_Server_API.app.api.v1.API_Deps.jobs_deps import get_job_manager
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import check_rate_limit, get_auth_principal
from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.api.v1.schemas.chat_request_schemas import DEFAULT_LLM_PROVIDER
from tldw_Server_API.app.api.v1.schemas.flashcards import (
    Deck,
    DeckCreate,
    DeckUpdate,
    Flashcard,
    FlashcardAnalyticsSummaryResponse,
    FlashcardAssetMetadata,
    FlashcardBulkUpdateError,
    FlashcardBulkUpdateItem,
    FlashcardBulkUpdateResponse,
    FlashcardBulkUpdateResult,
    FlashcardCreate,
    FlashcardGenerateRequest,
    FlashcardGenerateResponse,
    FlashcardListResponse,
    FlashcardNextReviewResponse,
    FlashcardTemplate,
    FlashcardTemplateCreate,
    FlashcardTemplateListResponse,
    FlashcardTemplateUpdate,
    FlashcardResetSchedulingRequest,
    FlashcardReviewRequest,
    FlashcardReviewResponse,
    FlashcardReviewSessionSummary,
    FlashcardTagSuggestionsResponse,
    FlashcardsImportRequest,
    FlashcardTagsUpdate,
    FlashcardUpdate,
    StructuredQaImportPreviewRequest,
    StructuredQaImportPreviewResponse,
    StudyAssistantContextResponse,
    StudyAssistantRespondRequest,
    StudyAssistantRespondResponse,
)
from tldw_Server_API.app.api.v1.schemas.study_packs import (
    StudyPackCreateJobRequest,
    StudyPackJobAcceptedResponse,
    StudyPackJobStatusResponse,
    StudyPackSummaryResponse,
)
from tldw_Server_API.app.core.AuthNZ.permissions import FLASHCARDS_ADMIN
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.config import loaded_config_data
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    CharactersRAGDBError,
    ConflictError,
    InputError,
)
from tldw_Server_API.app.core.Flashcards.apkg_exporter import export_apkg_from_rows
from tldw_Server_API.app.core.Flashcards.apkg_importer import (
    APKGImportError,
    import_rows_from_apkg_bytes,
)
from tldw_Server_API.app.core.Flashcards.asset_refs import (
    build_flashcard_asset_markdown,
    build_flashcard_asset_reference,
    extract_flashcard_asset_uuids,
)
from tldw_Server_API.app.core.Flashcards.scheduler_fsrs import (
    FsrsSettingsError,
    build_fsrs_next_interval_previews,
)
from tldw_Server_API.app.core.Flashcards.scheduler_sm2 import (
    SchedulerSettingsError,
    build_next_interval_previews,
    get_default_scheduler_settings_envelope,
)
from tldw_Server_API.app.core.Flashcards.structured_qa_import import parse_structured_qa_preview
from tldw_Server_API.app.core.Flashcards.study_assistant import (
    build_flashcard_assistant_context,
    generate_study_assistant_reply,
)
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.StudyPacks.jobs import (
    STUDY_PACKS_DOMAIN,
    STUDY_PACKS_JOB_TYPE,
    build_study_pack_job_payload,
    extract_study_pack_source_items,
    study_pack_jobs_queue,
)
from tldw_Server_API.app.core.StudySuggestions.jobs import (
    STUDY_SUGGESTIONS_DOMAIN,
    STUDY_SUGGESTIONS_REFRESH_JOB_TYPE,
    build_study_suggestions_job_payload,
    study_suggestions_jobs_queue,
)
from tldw_Server_API.app.core.testing import is_test_mode
from tldw_Server_API.app.core.Utils.image_validation import (
    get_max_flashcard_asset_bytes,
    validate_uploaded_image_bytes,
)
from tldw_Server_API.app.core.Workflows.adapters.content import run_flashcard_generate_adapter

router = APIRouter(prefix="/flashcards", tags=["flashcards"])
_FLASHCARDS_INT_PARSE_EXCEPTIONS: tuple[type[BaseException], ...] = (
    TypeError,
    ValueError,
)


def _ensure_workspace_exists(db: CharactersRAGDB, workspace_id: Optional[str]) -> None:
    if workspace_id is None:
        return
    if db.get_workspace(workspace_id) is None:
        raise HTTPException(status_code=404, detail=f"Workspace '{workspace_id}' not found")


def _coerce_scheduler_type(value: Any) -> str:
    return "fsrs" if str(value or "").strip().lower() == "fsrs" else "sm2_plus"


def _parse_scheduler_settings_envelope(raw: Any) -> dict[str, Any]:
    if isinstance(raw, str):
        if not raw.strip():
            return {}
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    if isinstance(raw, dict):
        return raw
    return {}


def _truncate_test_mode_flashcard_text(text: str, limit: int = 220) -> str:
    normalized = " ".join(str(text or "").split()).strip()
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[: limit - 1].rstrip()}…"


def _build_test_mode_flashcards(payload: FlashcardGenerateRequest) -> list[dict[str, Any]]:
    normalized_text = _truncate_test_mode_flashcard_text(payload.text) or "Workspace study aid coverage."
    card_type = str(payload.card_type or "basic").strip().lower() or "basic"
    if card_type not in ("basic", "basic_reverse", "cloze"):
        card_type = "basic"

    tags = [str(topic).strip() for topic in (payload.focus_topics or []) if str(topic).strip()]
    if not tags:
        tags = ["workspace", "study"]

    cards: list[dict[str, Any]] = []
    for index in range(max(1, int(payload.num_cards or 1))):
        if card_type == "cloze":
            front = f"{{{{c1::Study point {index + 1}}}}}: {normalized_text}"
            back = f"Study point {index + 1}"
        else:
            front = f"What study point {index + 1} should you remember?"
            back = normalized_text

        cards.append(
            {
                "front": front,
                "back": back,
                "tags": tags,
                "model_type": card_type,
                "notes": "Deterministic test-mode flashcard.",
            }
        )
    return cards


def _should_return_test_mode_flashcards(error: object) -> bool:
    normalized = str(error or "").lower()
    if not normalized:
        return False
    return (
        "llm provider is required" in normalized
        or "requires an api key" in normalized
        or "didn't provide an api key" in normalized
    )


def _attach_scheduler_preview(card: dict[str, Any], deck: dict[str, Any] | None) -> dict[str, Any]:
    scheduler_type = _coerce_scheduler_type(deck.get("scheduler_type") if deck else None)
    card["scheduler_type"] = scheduler_type
    raw_settings = deck.get("scheduler_settings_json") if deck else None
    queue_state = str(card.get("queue_state") or "").strip().lower()

    if scheduler_type == "fsrs" and queue_state == "review":
        envelope = _parse_scheduler_settings_envelope(raw_settings)
        card["next_intervals"] = build_fsrs_next_interval_previews(card, envelope.get("fsrs"))
    else:
        card["next_intervals"] = build_next_interval_previews(card, raw_settings)
    return card


def _build_review_scope_key(*, review_mode: str, deck_id: int | None, tag_filter: str | None = None) -> str:
    scope_parts = [str(review_mode or "due").strip().lower() or "due"]
    scope_parts.append(f"deck:{deck_id}" if deck_id is not None else "global")
    if tag_filter:
        scope_parts.append(f"tag:{tag_filter}")
    return ":".join(scope_parts)
_FLASHCARDS_NONCRITICAL_EXCEPTIONS: tuple[type[BaseException], ...] = (
    AttributeError,
    CharactersRAGDBError,
    LookupError,
    OSError,
    RuntimeError,
    TypeError,
    UnicodeError,
    ValueError,
    json.JSONDecodeError,
)
_ADMIN_CLAIM_PERMISSIONS = frozenset({"*", "system.configure"})
_STUDY_PACK_JOB_STATUS_MAP = {
    "queued": "queued",
    "processing": "running",
    "completed": "completed",
    "failed": "failed",
    "cancelled": "cancelled",
}


class ReviewSessionEndRequest(BaseModel):
    review_session_id: int


def _int_env(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
        return max(1, value)
    except _FLASHCARDS_INT_PARSE_EXCEPTIONS:
        return default


def _config_default_llm_provider() -> str | None:
    """Return the configured server default LLM provider when present."""
    cfg = loaded_config_data

    for section in ("llm_api_settings", "API"):
        try:
            section_data = cfg.get(section)
        except (AttributeError, TypeError, KeyError):
            section_data = None
        if not isinstance(section_data, dict):
            continue
        default_api = section_data.get("default_api")
        if isinstance(default_api, str):
            candidate = default_api.strip()
            if candidate:
                return candidate
    return None


def _resolve_flashcard_generation_provider(request_provider: str | None) -> str:
    """Resolve provider for flashcard generation using the server's default contract."""
    provider = str(request_provider or "").strip()
    if provider:
        return provider.lower()

    cfg_default = _config_default_llm_provider()
    if cfg_default:
        return cfg_default.strip().lower()

    env_default = str(os.getenv("DEFAULT_LLM_PROVIDER") or "").strip()
    if env_default:
        return env_default.lower()

    if is_test_mode():
        return "local-llm"

    fallback = str(DEFAULT_LLM_PROVIDER or "openai").strip().lower()
    return fallback or "openai"


def _is_admin_principal(principal: AuthPrincipal) -> bool:
    perms = {
        str(permission).strip().lower()
        for permission in (principal.permissions or [])
        if str(permission).strip()
    }
    roles = {
        str(role).strip().lower()
        for role in (principal.roles or [])
        if str(role).strip()
    }
    return bool("admin" in roles or perms & _ADMIN_CLAIM_PERMISSIONS or FLASHCARDS_ADMIN.lower() in perms)


def _serialize_study_pack(pack: dict[str, Any]) -> dict[str, Any]:
    return StudyPackSummaryResponse.model_validate(pack).model_dump(mode="json")


def _serialize_study_pack_job(job: dict[str, Any]) -> dict[str, Any]:
    raw_status = str(job.get("status") or "").strip().lower()
    return {
        "id": int(job["id"]),
        "status": _STUDY_PACK_JOB_STATUS_MAP.get(raw_status, "queued"),
        "domain": str(job.get("domain") or ""),
        "queue": str(job.get("queue") or ""),
        "job_type": str(job.get("job_type") or ""),
    }


def _ensure_study_pack_job_access(job: dict[str, Any], *, current_user: User, principal: AuthPrincipal) -> None:
    if _is_admin_principal(principal):
        return
    owner_user_id = str(job.get("owner_user_id") or "").strip()
    if owner_user_id != str(current_user.id):
        raise HTTPException(status_code=404, detail="Study-pack job not found")


def _study_pack_from_job_result(db: CharactersRAGDB, job: dict[str, Any]) -> dict[str, Any] | None:
    result = job.get("result") or {}
    pack_id = result.get("pack_id") if isinstance(result, dict) else None
    if pack_id is None:
        return None
    pack = db.get_study_pack(int(pack_id))
    return _serialize_study_pack(pack) if pack else None


def _public_study_pack_job_error(job: dict[str, Any]) -> str | None:
    raw_status = str(job.get("status") or "").strip().lower()
    if raw_status == "cancelled":
        reason = str(job.get("cancellation_reason") or "").strip()
        return reason or "Study pack generation was cancelled."
    if raw_status != "failed":
        return None

    raw_error = str(job.get("last_error") or job.get("error_message") or "").strip()
    if raw_error:
        logger.warning("Study-pack job {} failed: {}", int(job["id"]), raw_error)
    return "Study pack generation failed."


def _enqueue_study_suggestions_refresh(
    *,
    jm: JobManager,
    current_user: User,
    anchor_type: str,
    anchor_id: int,
) -> None:
    try:
        jm.create_job(
            domain=STUDY_SUGGESTIONS_DOMAIN,
            queue=study_suggestions_jobs_queue(),
            job_type=STUDY_SUGGESTIONS_REFRESH_JOB_TYPE,
            payload=build_study_suggestions_job_payload(
                job_type=STUDY_SUGGESTIONS_REFRESH_JOB_TYPE,
                anchor_type=anchor_type,
                anchor_id=anchor_id,
            ),
            owner_user_id=str(current_user.id),
            priority=5,
            max_retries=1,
        )
    except _FLASHCARDS_NONCRITICAL_EXCEPTIONS as exc:
        logger.warning("Study-suggestions refresh enqueue skipped for {}:{}: {}", anchor_type, anchor_id, exc)


async def _study_pack_db_for_job(
    job: dict[str, Any],
    *,
    request_db: CharactersRAGDB,
    current_user: User,
    principal: AuthPrincipal,
) -> CharactersRAGDB:
    if not _is_admin_principal(principal):
        return request_db

    owner_user_id = str(job.get("owner_user_id") or "").strip()
    if not owner_user_id or owner_user_id == str(current_user.id):
        return request_db

    return await get_chacha_db_for_owner(int(owner_user_id))


def _validate_bulk_flashcard_field_lengths(data: dict[str, Any]) -> None:
    """Reject oversize text fields before bulk flashcard insertion."""
    max_field_length = _int_env("FLASHCARDS_IMPORT_MAX_FIELD_LENGTH", 8192)
    field_labels = {
        "front": "Front",
        "back": "Back",
        "notes": "Notes",
        "extra": "Extra",
    }
    for field_name, label in field_labels.items():
        value = str(data.get(field_name) or "")
        if len(value.encode("utf-8")) > max_field_length:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Flashcard field too long",
                    "invalid_fields": [field_name],
                    "message": f"Field too long: {label} (> {max_field_length} bytes)",
                },
            )


def _fetch_flashcard_or_404(card_uuid: str, db: CharactersRAGDB) -> dict:
    card = db.get_flashcard(card_uuid)
    if not card:
        raise HTTPException(status_code=404, detail="Flashcard not found")
    return card


def _fetch_flashcard_template_or_404(template_id: int, db: CharactersRAGDB) -> FlashcardTemplate:
    """Return a validated flashcard template or raise a 404 when it does not exist."""

    template = db.get_flashcard_template(template_id)
    if not template:
        raise HTTPException(status_code=404, detail="Flashcard template not found")
    return FlashcardTemplate.model_validate(template)


def _build_assistant_context_snapshot(context: dict[str, Any]) -> dict[str, Any]:
    return {
        "context_type": context.get("context_type"),
        "flashcard": context.get("flashcard"),
    }


def _default_study_assistant_message(action: str, context: dict[str, Any]) -> str:
    front = str((context.get("flashcard") or {}).get("front") or "this card").strip()
    return {
        "explain": f"Explain this card: {front}",
        "mnemonic": f"Give me a mnemonic for this card: {front}",
        "follow_up": f"I have a follow-up question about this card: {front}",
        "fact_check": f"Fact-check my explanation of this card: {front}",
        "freeform": f"Help me study this card: {front}",
    }.get(action, f"Help me study this card: {front}")


def _normalize_flashcard_model_fields(
    front: Optional[str],
    model_type: Optional[str],
    is_cloze: Optional[bool],
    reverse: Optional[bool],
) -> tuple[str, bool, bool]:
    if model_type is not None:
        effective_model = str(model_type).lower()
    elif is_cloze is True:
        effective_model = "cloze"
    elif reverse is True:
        effective_model = "basic_reverse"
    else:
        effective_model = "basic"

    if effective_model not in ("basic", "basic_reverse", "cloze"):
        raise HTTPException(status_code=400, detail="Invalid model_type")

    eff_is_cloze = effective_model == "cloze"
    eff_reverse = effective_model == "basic_reverse"

    if eff_is_cloze and not re.search(r"\{\{c\d+::", front or ""):
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Invalid cloze",
                "invalid_fields": ["front"],
                "message": "Front must contain one or more {{cN::...}} patterns",
            },
        )
    return effective_model, eff_is_cloze, eff_reverse


def _normalize_flashcard_update_model_fields(
    current: dict[str, Any],
    data: dict[str, Any],
) -> None:
    if any(k in data for k in ("model_type", "reverse", "is_cloze")):
        current_model = current.get("model_type") or "basic"
        incoming_model = data.get("model_type")
        incoming_reverse = data.get("reverse")
        incoming_is_cloze = data.get("is_cloze")

        if incoming_model is not None:
            effective_model = str(incoming_model).lower()
        elif incoming_is_cloze is True:
            effective_model = "cloze"
        elif incoming_is_cloze is False:
            effective_model = "basic_reverse" if incoming_reverse is True else "basic"
        else:
            if incoming_reverse is True:
                effective_model = "basic_reverse" if current_model != "cloze" else "cloze"
            elif incoming_reverse is False:
                effective_model = "basic" if current_model == "basic_reverse" else current_model
            else:
                effective_model = current_model

        if effective_model not in ("basic", "basic_reverse", "cloze"):
            raise HTTPException(status_code=400, detail="Invalid model_type")

        data["model_type"] = effective_model
        data["is_cloze"] = (effective_model == "cloze")
        data["reverse"] = (effective_model == "basic_reverse")


def _prepare_flashcard_update(
    card_uuid: str,
    payload: FlashcardUpdate,
    db: CharactersRAGDB,
) -> tuple[dict[str, Any], int | None, list[str] | None]:
    data = payload.model_dump(exclude_unset=True)
    expected_version = data.pop("expected_version", None)
    tags = data.pop("tags", None)
    current = db.get_flashcard(card_uuid)
    if not current:
        raise HTTPException(status_code=404, detail="Flashcard not found")

    if "deck_id" in data:
        deck_id = data.get("deck_id")
        if deck_id is not None:
            deck = db.get_deck(int(deck_id))
            if not deck or bool(deck.get("deleted")):
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "Deck not found",
                        "invalid_deck_ids": [int(deck_id)],
                        "message": "Fix or remove invalid deck_id and retry",
                    },
                )
            data["deck_id"] = int(deck_id)

    _normalize_flashcard_update_model_fields(current, data)

    effective_model = data.get("model_type") or current.get("model_type")
    if effective_model == "cloze":
        front_text = data.get("front") if "front" in data else (current.get("front") or "")
        if not re.search(r"\{\{c\d+::", front_text):
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Invalid cloze",
                    "invalid_fields": ["front"],
                    "message": "Front must contain one or more {{cN::...}} patterns",
                },
            )

    return data, expected_version, tags


def _coerce_bulk_update_error(
    code: str,
    detail: str | dict[str, Any] | None,
) -> FlashcardBulkUpdateError:
    if isinstance(detail, dict):
        return FlashcardBulkUpdateError(
            code=code,
            message=str(detail.get("message") or detail.get("error") or code.replace("_", " ")),
            invalid_fields=[str(field) for field in detail.get("invalid_fields", [])],
            invalid_deck_ids=[int(deck_id) for deck_id in detail.get("invalid_deck_ids", [])],
        )
    return FlashcardBulkUpdateError(
        code=code,
        message=str(detail or code.replace("_", " ")),
    )


def _collect_flashcard_asset_refs_by_field(values: dict[str, Any]) -> dict[str, list[str]]:
    refs_by_field: dict[str, list[str]] = {}
    for field_name in ("front", "back", "extra", "notes"):
        refs = extract_flashcard_asset_uuids(values.get(field_name))
        if refs:
            refs_by_field[field_name] = refs
    return refs_by_field


def _validate_flashcard_asset_refs(
    refs_by_field: dict[str, list[str]],
    db: CharactersRAGDB,
    *,
    card_uuid: str | None = None,
) -> None:
    invalid_fields: list[str] = []
    problems: list[str] = []
    seen: set[str] = set()

    for field_name, asset_uuids in refs_by_field.items():
        for asset_uuid in asset_uuids:
            if asset_uuid in seen:
                continue
            seen.add(asset_uuid)
            asset = db.get_flashcard_asset(asset_uuid)
            if not asset:
                invalid_fields.append(field_name)
                problems.append(f"Unknown asset: {asset_uuid}")
                continue
            attached_card_uuid = asset.get("card_uuid")
            if attached_card_uuid and attached_card_uuid != card_uuid:
                invalid_fields.append(field_name)
                problems.append(f"Asset already attached to another card: {asset_uuid}")

    if problems:
        deduped_fields = list(dict.fromkeys(invalid_fields))
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Invalid flashcard asset reference",
                "invalid_fields": deduped_fields,
                "message": "; ".join(problems),
            },
        )


def _reconcile_flashcard_assets_or_500(card_uuid: str, db: CharactersRAGDB) -> None:
    card = db.get_flashcard(card_uuid)
    if not card:
        raise HTTPException(status_code=404, detail="Flashcard not found")
    try:
        db.reconcile_flashcard_asset_refs(
            card_uuid,
            front=card.get("front"),
            back=card.get("back"),
            extra=card.get("extra"),
            notes=card.get("notes"),
        )
    except (ConflictError, InputError) as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Invalid flashcard asset reference",
                "message": str(exc),
            },
        ) from exc


def _require_flashcards_admin(principal: AuthPrincipal) -> None:
    """Raise 403 if the principal lacks flashcards admin permission."""
    perms = {
        str(permission).strip().lower()
        for permission in (principal.permissions or [])
        if str(permission).strip()
    }
    roles = {
        str(role).strip().lower()
        for role in (principal.roles or [])
        if str(role).strip()
    }
    if FLASHCARDS_ADMIN.lower() not in perms and "admin" not in roles and not (perms & _ADMIN_CLAIM_PERMISSIONS):
        raise HTTPException(
            status_code=403,
            detail="Admin flashcards permission required for override",
        )


def _get_flashcards_apkg_max_media_bytes() -> int:
    """Resolve the total APKG media cap for flashcard import/export."""
    raw_bytes = os.getenv("FLASHCARDS_APKG_MAX_MEDIA_BYTES")
    if raw_bytes is not None:
        try:
            return max(1, int(raw_bytes))
        except _FLASHCARDS_INT_PARSE_EXCEPTIONS:
            logger.warning(
                "Invalid FLASHCARDS_APKG_MAX_MEDIA_BYTES={!r}; falling back to derived default.",
                raw_bytes,
            )
    return max(get_max_flashcard_asset_bytes() * 10, get_max_flashcard_asset_bytes())


@router.post("/decks", response_model=Deck)
def create_deck(payload: DeckCreate, db: CharactersRAGDB = Depends(get_chacha_db_for_user)):
    try:
        _ensure_workspace_exists(db, payload.workspace_id)
        review_prompt_side = payload.review_prompt_side if "review_prompt_side" in payload.model_fields_set else None
        deck_id = db.add_deck(
            payload.name,
            payload.description,
            payload.scheduler_settings.model_dump() if payload.scheduler_settings else None,
            scheduler_type=payload.scheduler_type,
            workspace_id=payload.workspace_id,
            review_prompt_side=review_prompt_side,
        )
        # Return the exact deck row by id
        deck = db.get_deck(deck_id)
        if deck:
            return deck
        # Fallback: minimal shape if retrieval failed (should not happen)
        return {
            "id": deck_id,
            "name": payload.name,
            "description": payload.description,
            "workspace_id": payload.workspace_id,
            "review_prompt_side": payload.review_prompt_side,
            "created_at": None,
            "last_modified": None,
            "deleted": False,
            "client_id": "",
            "version": 1,
            "scheduler_type": payload.scheduler_type,
            "scheduler_settings_json": None,
            "scheduler_settings": (
                payload.scheduler_settings.model_dump()
                if payload.scheduler_settings
                else get_default_scheduler_settings_envelope()
            ),
        }
    except SchedulerSettingsError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except ConflictError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    except CharactersRAGDBError as e:
        logger.error(f"Failed to create deck: {e}")
        raise HTTPException(status_code=500, detail="Failed to create deck") from e


@router.get("/decks", response_model=list[Deck])
def list_decks(
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    include_deleted: bool = False,
    workspace_id: Optional[str] = None,
    include_workspace_items: bool = False,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    try:
        return db.list_decks(
            limit=limit,
            offset=offset,
            include_deleted=include_deleted,
            workspace_id=workspace_id,
            include_workspace_items=include_workspace_items,
        )
    except CharactersRAGDBError as e:
        logger.error(f"Failed to list decks: {e}")
        raise HTTPException(status_code=500, detail="Failed to list decks") from e


@router.patch("/decks/{deck_id}", response_model=Deck)
def update_deck(
    deck_id: int,
    payload: DeckUpdate,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
):
    data = payload.model_dump(exclude_unset=True)
    expected_version = data.pop("expected_version", None)
    scheduler_settings = data.pop("scheduler_settings", None)
    scheduler_type = data.pop("scheduler_type", None)
    review_prompt_side = data.pop("review_prompt_side", None)
    workspace_id = data.pop("workspace_id", ...)
    try:
        if workspace_id is not ...:
            _ensure_workspace_exists(db, workspace_id)
        ok = db.update_deck(
            deck_id,
            name=data.get("name"),
            description=data.get("description"),
            review_prompt_side=review_prompt_side,
            scheduler_settings=scheduler_settings,
            scheduler_type=scheduler_type,
            workspace_id=workspace_id,
            expected_version=expected_version,
        )
        if not ok:
            raise HTTPException(status_code=404, detail="Deck not found or not updated")
        deck = db.get_deck(deck_id)
        if not deck:
            raise HTTPException(status_code=404, detail="Deck not found")
        return deck
    except SchedulerSettingsError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except ConflictError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    except CharactersRAGDBError as e:
        logger.error(f"Failed to update deck: {e}")
        raise HTTPException(status_code=500, detail="Failed to update deck") from e


@router.post("/assets", response_model=FlashcardAssetMetadata)
async def upload_flashcard_asset(
    file: UploadFile = File(...),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
):
    file_bytes = await file.read()
    is_valid, error_message, width, height = validate_uploaded_image_bytes(
        file_bytes,
        file.content_type or "",
    )
    if not is_valid:
        raise HTTPException(status_code=400, detail=error_message or "Invalid image upload")

    original_filename = file.filename or "image"
    alt_text = os.path.splitext(original_filename)[0] or "image"
    try:
        asset_uuid = db.add_flashcard_asset(
            image_bytes=file_bytes,
            mime_type=file.content_type or "application/octet-stream",
            original_filename=original_filename,
            width=width,
            height=height,
        )
    except CharactersRAGDBError as exc:
        logger.error(f"Failed to store flashcard asset: {exc}")
        raise HTTPException(status_code=500, detail="Failed to store flashcard asset") from exc

    reference = build_flashcard_asset_reference(asset_uuid)
    return FlashcardAssetMetadata(
        asset_uuid=asset_uuid,
        reference=reference,
        markdown_snippet=build_flashcard_asset_markdown(asset_uuid, alt_text),
        mime_type=file.content_type or "application/octet-stream",
        byte_size=len(file_bytes),
        width=width,
        height=height,
        original_filename=original_filename,
    )


@router.get("/assets/{asset_uuid}/content")
def get_flashcard_asset_content(
    asset_uuid: str,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
):
    asset = db.get_flashcard_asset(asset_uuid)
    if not asset:
        raise HTTPException(status_code=404, detail="Flashcard asset not found")
    content = db.get_flashcard_asset_content(asset_uuid)
    if content is None:
        raise HTTPException(status_code=404, detail="Flashcard asset content not found")
    return Response(content=content, media_type=str(asset.get("mime_type") or "application/octet-stream"))


@router.post("", response_model=Flashcard)
def create_flashcard(payload: FlashcardCreate, db: CharactersRAGDB = Depends(get_chacha_db_for_user)):
    try:
        data = payload.model_dump()
        refs_by_field = _collect_flashcard_asset_refs_by_field(data)
        _validate_flashcard_asset_refs(refs_by_field, db)
        # Convert tags list to tags_json string
        tags = data.pop("tags", None)
        if tags is not None:
            data["tags_json"] = json.dumps(tags)
        # Validate deck_id if provided
        deck_id = data.get("deck_id")
        if deck_id is not None:
            deck = db.get_deck(int(deck_id))
            if not deck or bool(deck.get("deleted")):
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "Deck not found",
                        "invalid_deck_ids": [int(deck_id)],
                        "message": "Fix or remove invalid deck_id and retry",
                    },
                )
        # Cloze validation if requested
        effective_model, eff_is_cloze, eff_reverse = _normalize_flashcard_model_fields(
            data.get("front"),
            data.get("model_type"),
            data.get("is_cloze"),
            data.get("reverse"),
        )
        data["model_type"] = effective_model
        data["is_cloze"] = eff_is_cloze
        data["reverse"] = eff_reverse
        card_uuid = db.add_flashcard(data)
        # Link keyword tags if provided
        if tags:
            try:
                db.set_flashcard_tags(card_uuid, tags)
            except _FLASHCARDS_NONCRITICAL_EXCEPTIONS as _e:
                logger.warning(f"Failed to link tags->keywords on create: {_e}")
        # Return the created flashcard via direct fetch
        card = db.get_flashcard(card_uuid)
        if not card:
            raise HTTPException(status_code=500, detail="Failed to fetch created flashcard")
        if refs_by_field:
            _reconcile_flashcard_assets_or_500(card_uuid, db)
            card = db.get_flashcard(card_uuid)
            if not card:
                raise HTTPException(status_code=500, detail="Failed to fetch created flashcard")
        return card
    except CharactersRAGDBError as e:
        logger.error(f"Failed to create flashcard: {e}")
        raise HTTPException(status_code=500, detail="Failed to create flashcard") from e


@router.post("/bulk", response_model=FlashcardListResponse)
def create_flashcards_bulk(payload: list[FlashcardCreate], db: CharactersRAGDB = Depends(get_chacha_db_for_user)):
    try:
        card_dicts = []
        raw_list = []
        tag_lists: list[Optional[list[str]]] = []
        deck_ids_to_validate: set[int] = set()
        for item in payload:
            data = item.model_dump()
            _validate_bulk_flashcard_field_lengths(data)
            refs_by_field = _collect_flashcard_asset_refs_by_field(data)
            _validate_flashcard_asset_refs(refs_by_field, db)
            tags = data.pop("tags", None)
            if tags is not None:
                data["tags_json"] = json.dumps(tags)
            tag_lists.append(tags)
            if data.get("deck_id") is not None:
                try:
                    deck_ids_to_validate.add(int(data.get("deck_id")))
                except _FLASHCARDS_INT_PARSE_EXCEPTIONS:
                    raise HTTPException(status_code=400, detail="Invalid deck_id type") from None
            effective_model, eff_is_cloze, eff_reverse = _normalize_flashcard_model_fields(
                data.get("front"),
                data.get("model_type"),
                data.get("is_cloze"),
                data.get("reverse"),
            )
            data["model_type"] = effective_model
            data["is_cloze"] = eff_is_cloze
            data["reverse"] = eff_reverse
            raw_list.append(data)
        # Validate all referenced deck IDs before inserting; collect all invalids
        invalid_deck_ids: list[int] = []
        for did in deck_ids_to_validate:
            d = db.get_deck(did)
            if not d or bool(d.get("deleted")):
                invalid_deck_ids.append(did)
        if invalid_deck_ids:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "One or more decks not found",
                    "invalid_deck_ids": sorted(invalid_deck_ids),
                    "message": "Fix or remove invalid deck_id values and retry",
                },
            )
        uuids = db.add_flashcards_bulk(raw_list)
        # Link keyword tags for each created card
        for u, tags in zip(uuids, tag_lists):
            if tags:
                try:
                    db.set_flashcard_tags(u, tags)
                except _FLASHCARDS_NONCRITICAL_EXCEPTIONS as _e:
                    logger.warning(f"Failed to link tags for {u}: {_e}")
            _reconcile_flashcard_assets_or_500(u, db)
        # Fetch created cards precisely by uuid (order-preserving)
        try:
            items = db.get_flashcards_by_uuids(uuids)
            index = {c["uuid"]: c for c in items}
            for u in uuids:
                if u in index:
                    card_dicts.append(index[u])
        except _FLASHCARDS_NONCRITICAL_EXCEPTIONS:
            for u in uuids:
                c = db.get_flashcard(u)
                if c:
                    card_dicts.append(c)
        return {"items": card_dicts, "count": len(card_dicts)}
    except CharactersRAGDBError as e:
        logger.error(f"Failed bulk create flashcards: {e}")
        raise HTTPException(status_code=500, detail="Failed to create flashcards") from e


@router.patch("/bulk", response_model=FlashcardBulkUpdateResponse)
def update_flashcards_bulk(
    payload: list[FlashcardBulkUpdateItem],
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
):
    try:
        results: list[FlashcardBulkUpdateResult] = []
        for item in payload:
            try:
                update_payload = FlashcardUpdate(
                    **item.model_dump(exclude={"uuid"}, exclude_unset=True)
                )
                data, expected_version, tags = _prepare_flashcard_update(item.uuid, update_payload, db)
                current = db.get_flashcard(item.uuid)
                if not current:
                    raise HTTPException(status_code=404, detail="Flashcard not found")
                refs_by_field = _collect_flashcard_asset_refs_by_field(
                    {
                        key: data.get(key) if key in data else current.get(key)
                        for key in ("front", "back", "extra", "notes")
                    }
                )
                _validate_flashcard_asset_refs(refs_by_field, db, card_uuid=item.uuid)
                ok = db.update_flashcard(item.uuid, data, expected_version, tags=tags)
                if not ok:
                    results.append(
                        FlashcardBulkUpdateResult(
                            uuid=item.uuid,
                            status="not_found",
                            error=_coerce_bulk_update_error("not_found", "Flashcard not found or not updated"),
                        )
                    )
                    continue
                card = db.get_flashcard(item.uuid)
                if not card:
                    results.append(
                        FlashcardBulkUpdateResult(
                            uuid=item.uuid,
                            status="not_found",
                            error=_coerce_bulk_update_error("not_found", "Flashcard not found"),
                        )
                    )
                    continue
                if refs_by_field:
                    _reconcile_flashcard_assets_or_500(item.uuid, db)
                    card = db.get_flashcard(item.uuid)
                    if not card:
                        results.append(
                            FlashcardBulkUpdateResult(
                                uuid=item.uuid,
                                status="not_found",
                                error=_coerce_bulk_update_error("not_found", "Flashcard not found"),
                            )
                        )
                        continue
                results.append(
                    FlashcardBulkUpdateResult(
                        uuid=item.uuid,
                        status="updated",
                        flashcard=card,
                    )
                )
            except HTTPException as exc:
                if exc.status_code == 404:
                    status = "not_found"
                    code = "not_found"
                elif exc.status_code == 400:
                    status = "validation_error"
                    code = "validation_error"
                else:
                    raise
                results.append(
                    FlashcardBulkUpdateResult(
                        uuid=item.uuid,
                        status=status,
                        error=_coerce_bulk_update_error(code, exc.detail),
                    )
                )
            except ConflictError as exc:
                results.append(
                    FlashcardBulkUpdateResult(
                        uuid=item.uuid,
                        status="conflict",
                        error=_coerce_bulk_update_error("conflict", str(exc)),
                    )
                )
        return FlashcardBulkUpdateResponse(results=results)
    except CharactersRAGDBError as e:
        logger.error(f"Failed bulk update flashcards: {e}")
        raise HTTPException(status_code=500, detail="Failed to update flashcards") from e


@router.get("", response_model=FlashcardListResponse)
def list_flashcards(
    deck_id: Optional[int] = None,
    workspace_id: Optional[str] = None,
    include_workspace_items: bool = False,
    tag: Optional[str] = None,
    due_status: Optional[str] = Query('all', pattern="^(new|learning|due|all)$"),
    q: Optional[str] = None,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    order_by: Optional[str] = Query('due_at', pattern="^(due_at|created_at)$"),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user)
):
    try:
        items = db.list_flashcards(
            deck_id=deck_id,
            workspace_id=workspace_id,
            include_workspace_items=include_workspace_items,
            tag=tag,
            due_status=due_status or 'all',
            q=q,
            include_deleted=False,
            limit=limit,
            offset=offset,
            order_by=order_by or 'due_at',
        )
        total = db.count_flashcards(
            deck_id=deck_id,
            workspace_id=workspace_id,
            include_workspace_items=include_workspace_items,
            tag=tag,
            due_status=due_status or 'all',
            q=q,
            include_deleted=False,
        )
        return {"items": items, "count": len(items), "total": int(total)}
    except CharactersRAGDBError as e:
        logger.error(f"Failed to list flashcards: {e}")
        raise HTTPException(status_code=500, detail="Failed to list flashcards") from e


@router.get(
    "/tags",
    response_model=FlashcardTagSuggestionsResponse,
    dependencies=[Depends(check_rate_limit)],
)
def list_flashcard_tag_suggestions(
    q: Optional[str] = None,
    limit: int = Query(50, ge=1, le=100),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
) -> FlashcardTagSuggestionsResponse:
    """List global flashcard tag suggestions.

    Args:
        q: Optional trimmed substring filter applied case-insensitively to tag names.
        limit: Maximum number of suggestions to return.
        db: Flashcards database dependency for the current user.

    Returns:
        A typed response containing matching tag suggestions and the number returned.
    """
    try:
        items = db.list_flashcard_tag_suggestions(q=q, limit=limit)
        return FlashcardTagSuggestionsResponse(items=items, count=len(items))
    except CharactersRAGDBError as e:
        logger.error(f"Failed to list flashcard tag suggestions: {e}")
        raise HTTPException(status_code=500, detail="Failed to list flashcard tag suggestions") from e


@router.get("/analytics/summary", response_model=FlashcardAnalyticsSummaryResponse)
def get_flashcard_analytics_summary(
    deck_id: Optional[int] = Query(None, ge=1),
    workspace_id: Optional[str] = None,
    include_workspace_items: bool = False,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
):
    try:
        return db.get_flashcard_analytics_summary(
            deck_id=deck_id,
            workspace_id=workspace_id,
            include_workspace_items=include_workspace_items,
        )
    except CharactersRAGDBError as e:
        logger.error(f"Failed to get flashcard analytics summary: {e}")
        raise HTTPException(status_code=500, detail="Failed to get flashcard analytics summary") from e

## (export endpoint moved earlier)

## Note: /export endpoint is defined above to avoid path shadowing by /{card_uuid}

@router.get("/id/{card_uuid}", response_model=Flashcard)
def get_flashcard(card_uuid: str, db: CharactersRAGDB = Depends(get_chacha_db_for_user)):
    try:
        return _fetch_flashcard_or_404(card_uuid, db)
    except CharactersRAGDBError as e:
        logger.error(f"Failed to get flashcard: {e}")
        raise HTTPException(status_code=500, detail="Failed to get flashcard") from e


@router.patch("/{card_uuid}", response_model=Flashcard)
def update_flashcard(card_uuid: str, payload: FlashcardUpdate, db: CharactersRAGDB = Depends(get_chacha_db_for_user)):
    try:
        data, expected_version, tags = _prepare_flashcard_update(card_uuid, payload, db)
        current = db.get_flashcard(card_uuid)
        if not current:
            raise HTTPException(status_code=404, detail="Flashcard not found")
        refs_by_field = _collect_flashcard_asset_refs_by_field(
            {
                key: data.get(key) if key in data else current.get(key)
                for key in ("front", "back", "extra", "notes")
            }
        )
        _validate_flashcard_asset_refs(refs_by_field, db, card_uuid=card_uuid)
        ok = db.update_flashcard(card_uuid, data, expected_version, tags=tags)
        if not ok:
            raise HTTPException(status_code=404, detail="Flashcard not found or not updated")
        if refs_by_field:
            _reconcile_flashcard_assets_or_500(card_uuid, db)
        card = db.get_flashcard(card_uuid)
        if not card:
            raise HTTPException(status_code=404, detail="Flashcard not found")
        return card
    except ConflictError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    except CharactersRAGDBError as e:
        logger.error(f"Failed to update flashcard: {e}")
        raise HTTPException(status_code=500, detail="Failed to update flashcard") from e


@router.delete("/{card_uuid}")
def delete_flashcard(card_uuid: str, expected_version: int = Query(..., ge=1), db: CharactersRAGDB = Depends(get_chacha_db_for_user)):
    try:
        ok = db.soft_delete_flashcard(card_uuid, expected_version)
        if not ok:
            raise HTTPException(status_code=404, detail="Flashcard not found or already deleted")
        return {"deleted": True}
    except ConflictError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    except CharactersRAGDBError as e:
        logger.error(f"Failed to delete flashcard: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete flashcard") from e


@router.post("/{card_uuid}/reset-scheduling", response_model=Flashcard)
def reset_flashcard_scheduling(
    card_uuid: str,
    payload: FlashcardResetSchedulingRequest,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
):
    try:
        ok = db.reset_flashcard_scheduling(
            card_uuid,
            expected_version=payload.expected_version,
        )
        if not ok:
            raise HTTPException(status_code=404, detail="Flashcard not found or not updated")
        card = db.get_flashcard(card_uuid)
        if not card:
            raise HTTPException(status_code=404, detail="Flashcard not found")
        return card
    except ConflictError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    except CharactersRAGDBError as e:
        logger.error(f"Failed to reset flashcard scheduling: {e}")
        raise HTTPException(status_code=500, detail="Failed to reset flashcard scheduling") from e


@router.put("/{card_uuid}/tags", response_model=Flashcard)
def set_flashcard_tags(card_uuid: str, payload: FlashcardTagsUpdate, db: CharactersRAGDB = Depends(get_chacha_db_for_user)):
    try:
        db.set_flashcard_tags(card_uuid, payload.tags)
        card = db.get_flashcard(card_uuid)
        return card
    except CharactersRAGDBError as e:
        logger.error(f"Failed to set flashcard tags: {e}")
        raise HTTPException(status_code=500, detail="Failed to set flashcard tags") from e


@router.get("/{card_uuid}/tags")
def get_flashcard_tags(card_uuid: str, db: CharactersRAGDB = Depends(get_chacha_db_for_user)):
    try:
        kws = db.get_keywords_for_flashcard(card_uuid)
        return {"items": kws, "count": len(kws)}
    except CharactersRAGDBError as e:
        logger.error(f"Failed to get flashcard tags: {e}")
        raise HTTPException(status_code=500, detail="Failed to get flashcard tags") from e


@router.post(
    "/import/structured/preview",
    response_model=StructuredQaImportPreviewResponse,
)
def preview_structured_qa_import(
    payload: StructuredQaImportPreviewRequest,
    max_lines: Optional[int] = Query(
        None,
        ge=1,
        description="Admin override: max preview lines to parse (cannot exceed env cap)",
    ),
    max_line_length: Optional[int] = Query(
        None,
        ge=1,
        description="Admin override: max preview line length in bytes (cannot exceed env cap)",
    ),
    max_field_length: Optional[int] = Query(
        None,
        ge=1,
        description="Admin override: max preview field length in bytes (cannot exceed env cap)",
    ),
    principal: AuthPrincipal = Depends(get_auth_principal),
):
    """Preview structured Q&A text without creating flashcards."""
    env_max_lines = _int_env("FLASHCARDS_IMPORT_MAX_LINES", 10000)
    env_max_line_length = _int_env("FLASHCARDS_IMPORT_MAX_LINE_LENGTH", 32768)
    env_max_field_length = _int_env("FLASHCARDS_IMPORT_MAX_FIELD_LENGTH", 8192)

    if any(p is not None for p in (max_lines, max_line_length, max_field_length)):
        _require_flashcards_admin(principal)

    effective_max_lines = min(env_max_lines, max_lines) if max_lines else env_max_lines
    effective_max_line_length = (
        min(env_max_line_length, max_line_length)
        if max_line_length
        else env_max_line_length
    )
    effective_max_field_length = (
        min(env_max_field_length, max_field_length)
        if max_field_length
        else env_max_field_length
    )

    result = parse_structured_qa_preview(
        payload.content,
        max_lines=effective_max_lines,
        max_line_length=effective_max_line_length,
        max_field_length=effective_max_field_length,
    )
    return {
        "drafts": [draft.__dict__ for draft in result.drafts],
        "errors": [error.__dict__ for error in result.errors],
        "detected_format": result.detected_format,
        "skipped_blocks": result.skipped_blocks,
    }


@router.post("/import")
def import_flashcards(
    payload: FlashcardsImportRequest,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    max_lines: Optional[int] = Query(None, ge=1, description="Admin override: max lines to import (cannot exceed env cap)"),
    max_line_length: Optional[int] = Query(None, ge=1, description="Admin override: max line length in bytes (cannot exceed env cap)"),
    max_field_length: Optional[int] = Query(None, ge=1, description="Admin override: max field length in bytes (cannot exceed env cap)"),
    principal: AuthPrincipal = Depends(get_auth_principal),
):
    try:
        delimiter = payload.delimiter or '\t'
        raw_lines = list((payload.content or '').splitlines())
        header_map: dict[str, int] = {}
        lines = raw_lines
        errors: list[dict] = []
        # Abuse caps
        # Configurable caps via environment
        ENV_MAX_LINES = _int_env('FLASHCARDS_IMPORT_MAX_LINES', 10000)
        ENV_MAX_LINE_LENGTH = _int_env('FLASHCARDS_IMPORT_MAX_LINE_LENGTH', 32768)
        ENV_MAX_FIELD_LENGTH = _int_env('FLASHCARDS_IMPORT_MAX_FIELD_LENGTH', 8192)
        # Effective caps: query param can only lower the env caps, and requires admin to use
        if any(p is not None for p in (max_lines, max_line_length, max_field_length)):
            _require_flashcards_admin(principal)
        MAX_LINES = min(ENV_MAX_LINES, max_lines) if max_lines else ENV_MAX_LINES
        MAX_LINE_LENGTH = min(ENV_MAX_LINE_LENGTH, max_line_length) if max_line_length else ENV_MAX_LINE_LENGTH
        MAX_FIELD_LENGTH = min(ENV_MAX_FIELD_LENGTH, max_field_length) if max_field_length else ENV_MAX_FIELD_LENGTH
        if payload.has_header and raw_lines:
            header = raw_lines[0]
            lines = raw_lines[1:]
            cols = [c.strip() for c in header.split(delimiter)]
            # Build header map (lower-cased keys)
            for idx, name in enumerate(cols):
                lname = name.strip().lower()
                header_map[lname] = idx
                # Also support variants without underscores (e.g., ModelType -> modeltype)
                header_map[lname.replace('_', '')] = idx
        # Build or cache decks by name
        decks_cache: dict[str, int] = {}
        # Preload existing decks once
        existing_decks = {
            d.get('name'): d.get('id')
            for d in db.list_decks(limit=10000, offset=0, include_deleted=False, include_workspace_items=True)
        }
        default_deck_name = 'Default'
        # Ensure default deck exists in cache
        if default_deck_name not in existing_decks:
            did = db.add_deck(default_deck_name, description=None)
            existing_decks[default_deck_name] = did
        decks_cache.update(existing_decks)
        created: list[dict] = []
        processed = 0
        for i, raw in enumerate(lines, start=1):
            if processed >= MAX_LINES:
                errors.append({'line': None, 'error': f'Maximum import line limit reached ({MAX_LINES})'})
                break
            if not raw.strip():
                continue
            # Enforce line length in bytes
            if len(raw.encode('utf-8')) > MAX_LINE_LENGTH:
                errors.append({'line': (i + 1 if payload.has_header else i), 'error': f'Line too long (>{MAX_LINE_LENGTH} bytes)'})
                continue
            parts = raw.split(delimiter)
            # Default mapping: Deck, Front, Back, Tags, Notes
            deck_name = front = back = tags_s = notes = extra = model_type = deck_desc = None
            reverse_flag = None
            is_cloze_flag = None
            def get_col(*names: str, _parts=parts) -> Optional[str]:
                for nm in names:
                    idx = header_map.get(nm)
                    if idx is not None and idx < len(_parts):
                        return _parts[idx].strip()
                return None

            if header_map:
                deck_name = get_col('deck', 'deckname', 'deck_name') or ''
                deck_desc = get_col('deckdescription', 'deck_description', 'deckdesc', 'deck desc')
                front = get_col('front', 'question') or ''
                back = get_col('back', 'answer') or ''
                tags_s = get_col('tags') or ''
                notes = get_col('notes', 'note') or ''
                extra = get_col('extra', 'back extra', 'back_extra') or None
                model_type = (get_col('model_type', 'modeltype', 'model', 'type') or '').lower() or None
                rev_val = (get_col('reverse', 'reversed') or '').lower()
                if rev_val in ('1', 'true', 'yes', 'y', 'on'):
                    reverse_flag = True
                elif rev_val in ('0', 'false', 'no', 'n', 'off'):
                    reverse_flag = False
                ic_val = (get_col('iscloze', 'is_cloze') or '').lower()
                if ic_val in ('1', 'true', 'yes', 'y', 'on'):
                    is_cloze_flag = True
                elif ic_val in ('0', 'false', 'no', 'n', 'off'):
                    is_cloze_flag = False
            else:
                # Expect 5 columns: Deck, Front, Back, Tags, Notes
                if len(parts) < 5:
                    parts = parts + [''] * (5 - len(parts))
                deck_name, front, back, tags_s, notes = [p.strip() for p in parts[:5]]
                extra = None
                model_type = None
                reverse_flag = None
                deck_desc = None
                is_cloze_flag = None

            # Basic validation: require front text (don't import empty rows)
            if not (front and front.strip()):
                errors.append({
                    'line': (i + 1 if payload.has_header else i),
                    'error': 'Missing required field: Front'
                })
                continue
            # Field length caps
            field_lengths = {
                'Deck': deck_name or '',
                'Front': front or '',
                'Back': back or '',
                'Tags': tags_s or '',
                'Notes': notes or '',
                'Extra': extra or '',
                'DeckDescription': deck_desc or '',
            }
            # Enforce field caps using UTF-8 byte length
            too_long = next(((k, v) for k, v in field_lengths.items() if len(v.encode('utf-8')) > MAX_FIELD_LENGTH), None)
            if too_long:
                errors.append({
                    'line': (i + 1 if payload.has_header else i),
                    'error': f'Field too long: {too_long[0]} (> {MAX_FIELD_LENGTH} bytes)'
                })
                continue

            # If header explicitly contains a deck column, enforce non-empty deck
            deck_keys = {'deck', 'deckname', 'deck_name'}
            if header_map and any(k in header_map for k in deck_keys) and not deck_name:
                errors.append({
                    'line': (i + 1 if payload.has_header else i),
                    'error': 'Missing required field: Deck'
                })
                continue

            # Stricter cloze rule: if cloze model or flag, require cN pattern in front
            effective_is_cloze = ((model_type or '') == 'cloze') or (is_cloze_flag is True)
            if effective_is_cloze and not re.search(r"\{\{c\d+::", front):
                errors.append({
                    'line': (i + 1 if payload.has_header else i),
                    'error': 'Invalid cloze: Front must contain one or more {{cN::...}} patterns'
                })
                continue
            # Resolve deck; default to 'Default' when none provided
            deck_id = None
            eff_deck_name = deck_name or default_deck_name
            if eff_deck_name in decks_cache:
                deck_id = decks_cache[eff_deck_name]
            else:
                if eff_deck_name in existing_decks:
                    deck_id = existing_decks[eff_deck_name]
                else:
                    deck_id = db.add_deck(eff_deck_name, description=deck_desc)
                    existing_decks[eff_deck_name] = deck_id
                decks_cache[eff_deck_name] = deck_id
            # Parse tags from space-delimited
            tags_list = [t for t in (tags_s or '').replace(',', ' ').split() if t]
            # Determine effective model type
            if ((model_type or '').lower() == 'cloze') or (is_cloze_flag is True):
                eff_model_type = 'cloze'
            elif (model_type or '') in ('basic', 'basic_reverse'):
                eff_model_type = model_type  # type: ignore[assignment]
            elif reverse_flag is True:
                eff_model_type = 'basic_reverse'
            else:
                eff_model_type = 'basic'

            # Create card
            data: dict = {
                'deck_id': deck_id,
                'front': front,
                'back': back,
                'notes': notes,
                'tags_json': json.dumps(tags_list) if tags_list else None,
                'model_type': eff_model_type,
                'reverse': eff_model_type == 'basic_reverse',
            }
            if extra:
                data['extra'] = extra
            if eff_model_type == 'cloze':
                data['is_cloze'] = True
            uuid = db.add_flashcard(data)
            created.append({'uuid': uuid, 'deck_id': deck_id})
            # Also ensure keyword links reflect tags
            if tags_list:
                db.set_flashcard_tags(uuid, tags_list)
            processed += 1
        return {'imported': len(created), 'items': created, 'errors': errors}
    except HTTPException:
        raise
    except CharactersRAGDBError as e:
        logger.error(f"Failed to import flashcards: {e}")
        raise HTTPException(status_code=500, detail="Failed to import flashcards") from e
    except _FLASHCARDS_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"TSV import failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to import TSV flashcards") from e


@router.post("/import/json")
async def import_flashcards_json(
    file: UploadFile = File(...),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    max_items: Optional[int] = Query(None, ge=1, description="Admin override: max JSON items to import (cannot exceed env cap)"),
    max_field_length: Optional[int] = Query(None, ge=1, description="Admin override: max field length in bytes (cannot exceed env cap)"),
    principal: AuthPrincipal = Depends(get_auth_principal),
):
    try:
        raw = await file.read()
        # Caps via env
        ENV_MAX_ITEMS = _int_env('FLASHCARDS_IMPORT_MAX_LINES', 10000)
        ENV_MAX_FIELD_LENGTH = _int_env('FLASHCARDS_IMPORT_MAX_FIELD_LENGTH', 8192)
        if any(p is not None for p in (max_items, max_field_length)):
            _require_flashcards_admin(principal)
        MAX_ITEMS = min(ENV_MAX_ITEMS, max_items) if max_items else ENV_MAX_ITEMS
        MAX_FIELD_LENGTH = min(ENV_MAX_FIELD_LENGTH, max_field_length) if max_field_length else ENV_MAX_FIELD_LENGTH

        # Parse JSON: accept array or object with 'items' key
        import json as _json
        text = raw.decode('utf-8')
        try:
            data = _json.loads(text)
        except _json.JSONDecodeError:
            # Try JSON Lines: one JSON object per line
            items = []
            for ln in text.splitlines():
                if not ln.strip():
                    continue
                try:
                    items.append(_json.loads(ln))
                except _json.JSONDecodeError:
                    raise HTTPException(status_code=400, detail="Invalid JSON/JSONL upload: failed to parse a line") from None
            data = {'items': items}

        items = data.get('items') if isinstance(data, dict) and 'items' in data else data
        if not isinstance(items, list):
            raise HTTPException(status_code=400, detail="JSON content must be a list of objects or {'items': [...]} ")

        # Deck cache
        existing_decks = {
            d.get('name'): d.get('id')
            for d in db.list_decks(limit=10000, offset=0, include_deleted=False, include_workspace_items=True)
        }
        decks_cache: dict[str, int] = dict(existing_decks)
        default_deck_name = 'Default'
        if default_deck_name not in decks_cache:
            did = db.add_deck(default_deck_name, description=None)
            decks_cache[default_deck_name] = did
            existing_decks[default_deck_name] = did

        errors: list[dict] = []
        created: list[dict] = []
        processed = 0
        for idx, obj in enumerate(items, start=1):
            if processed >= MAX_ITEMS:
                errors.append({'index': idx, 'error': f'Maximum import item limit reached ({MAX_ITEMS})'})
                break
            if not isinstance(obj, dict):
                errors.append({'index': idx, 'error': 'Item must be a JSON object'})
                continue
            deck_name = (obj.get('deck') or obj.get('deck_name') or '').strip()
            deck_desc = (obj.get('deck_description') or obj.get('deckdesc') or obj.get('deckDescription') or None)
            front = (obj.get('front') or obj.get('question') or '').strip()
            back = (obj.get('back') or obj.get('answer') or '').strip()
            notes = (obj.get('notes') or obj.get('note') or '')
            extra = obj.get('extra')
            model_type = (obj.get('model_type') or obj.get('model') or obj.get('type') or '').lower() or None
            reverse_flag = obj.get('reverse')
            is_cloze_flag = obj.get('is_cloze')
            tags_val = obj.get('tags')
            if isinstance(tags_val, list):
                tags_list = [str(t) for t in tags_val]
            elif isinstance(tags_val, str):
                tags_list = [t for t in tags_val.replace(',', ' ').split() if t]
            else:
                tags_list = []

            # Validation
            if not front:
                errors.append({'index': idx, 'error': 'Missing required field: Front'})
                continue
            eff_deck = default_deck_name if deck_name == '' else deck_name
            # Field caps
            fields = {
                'Deck': eff_deck,
                'Front': front,
                'Back': back,
                'Notes': notes or '',
                'Extra': extra or '',
                'DeckDescription': deck_desc or ''
            }
            too_long = next(((k, v) for k, v in fields.items() if len((v or '').encode('utf-8')) > MAX_FIELD_LENGTH), None)
            if too_long:
                errors.append({'index': idx, 'error': f'Field too long: {too_long[0]} (> {MAX_FIELD_LENGTH} bytes)'} )
                continue
            # Cloze validation
            effective_is_cloze = ((model_type or '') == 'cloze') or (is_cloze_flag is True)
            if effective_is_cloze and not re.search(r"\{\{c\d+::", front):
                errors.append({'index': idx, 'error': 'Invalid cloze: Front must contain one or more {{cN::...}} patterns'})
                continue

            # Resolve/create deck
            if eff_deck in decks_cache:
                deck_id = decks_cache[eff_deck]
            else:
                deck_id = db.add_deck(eff_deck, description=deck_desc)
                decks_cache[eff_deck] = deck_id
                existing_decks[eff_deck] = deck_id

            # Determine effective model type
            if effective_is_cloze:
                eff_model_type = 'cloze'
            elif (model_type or '') in ('basic', 'basic_reverse', 'cloze'):
                eff_model_type = model_type  # type: ignore[assignment]
            elif reverse_flag is True:
                eff_model_type = 'basic_reverse'
            else:
                eff_model_type = 'basic'

            # Build card
            data: dict = {
                'deck_id': deck_id,
                'front': front,
                'back': back,
                'notes': notes,
                'tags_json': _json.dumps(tags_list) if tags_list else None,
                'model_type': eff_model_type,
                'reverse': eff_model_type == 'basic_reverse',
            }
            if extra:
                data['extra'] = extra
            if effective_is_cloze:
                data['is_cloze'] = True

            uuid = db.add_flashcard(data)
            created.append({'uuid': uuid, 'deck_id': deck_id})
            if tags_list:
                db.set_flashcard_tags(uuid, tags_list)
            processed += 1

        return {'imported': len(created), 'items': created, 'errors': errors}
    except HTTPException:
        raise
    except CharactersRAGDBError as e:
        logger.error(f"Failed to import JSON flashcards: {e}")
        raise HTTPException(status_code=500, detail="Failed to import flashcards") from e
    except _FLASHCARDS_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"JSON import failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to import JSON flashcards") from e


@router.post("/import/apkg")
async def import_flashcards_apkg(
    file: UploadFile = File(...),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    max_items: Optional[int] = Query(
        None,
        ge=1,
        description="Admin override: max APKG notes to import (cannot exceed env cap)",
    ),
    max_field_length: Optional[int] = Query(
        None,
        ge=1,
        description="Admin override: max APKG field length in bytes (cannot exceed env cap)",
    ),
    principal: AuthPrincipal = Depends(get_auth_principal),
):
    try:
        raw = await file.read()

        env_max_items = _int_env("FLASHCARDS_IMPORT_MAX_LINES", 10000)
        env_max_field_length = _int_env("FLASHCARDS_IMPORT_MAX_FIELD_LENGTH", 8192)
        apkg_max_media_bytes = _get_flashcards_apkg_max_media_bytes()
        if any(p is not None for p in (max_items, max_field_length)):
            _require_flashcards_admin(principal)
        effective_max_items = min(env_max_items, max_items) if max_items else env_max_items
        effective_max_field_length = (
            min(env_max_field_length, max_field_length)
            if max_field_length
            else env_max_field_length
        )

        def asset_importer(content: bytes, mime_type: str, original_filename: str) -> str:
            is_valid, error_message, width, height = validate_uploaded_image_bytes(content, mime_type)
            if not is_valid:
                raise APKGImportError(error_message or "Invalid APKG image media")
            try:
                return db.add_flashcard_asset(
                    image_bytes=content,
                    mime_type=mime_type,
                    original_filename=original_filename,
                    width=width,
                    height=height,
                )
            except CharactersRAGDBError as exc:
                logger.error(f"Failed to store APKG flashcard asset: {exc}")
                raise APKGImportError("Failed to store APKG flashcard media") from exc

        rows, errors = import_rows_from_apkg_bytes(
            raw,
            max_notes=effective_max_items,
            max_field_length=effective_max_field_length,
            max_total_media_bytes=apkg_max_media_bytes,
            asset_importer=asset_importer,
        )

        existing_decks = {
            d.get("name"): d.get("id")
            for d in db.list_decks(limit=10000, offset=0, include_deleted=False, include_workspace_items=True)
        }
        decks_cache: dict[str, int] = dict(existing_decks)
        if "Default" not in decks_cache:
            did = db.add_deck("Default", description=None)
            decks_cache["Default"] = did
            existing_decks["Default"] = did

        created: list[dict[str, Any]] = []
        for idx, row in enumerate(rows, start=1):
            deck_name = str(row.get("deck_name") or "Default").strip() or "Default"
            deck_id = decks_cache.get(deck_name)
            if deck_id is None:
                deck_id = db.add_deck(deck_name, description=None)
                decks_cache[deck_name] = deck_id
                existing_decks[deck_name] = deck_id

            tags_value = row.get("tags")
            if isinstance(tags_value, list):
                tags_list = [str(tag).strip() for tag in tags_value if str(tag).strip()]
            else:
                tags_list = []

            model_type = str(row.get("model_type") or "basic").lower()
            if model_type not in ("basic", "basic_reverse", "cloze"):
                errors.append(
                    {
                        "index": idx,
                        "error": f"Unsupported model_type in APKG note: {model_type}",
                    }
                )
                continue

            is_cloze = bool(row.get("is_cloze")) or model_type == "cloze"
            reverse = bool(row.get("reverse")) or model_type == "basic_reverse"
            data: dict[str, Any] = {
                "deck_id": deck_id,
                "front": str(row.get("front") or ""),
                "back": str(row.get("back") or ""),
                "notes": row.get("notes"),
                "extra": row.get("extra"),
                "tags_json": json.dumps(tags_list) if tags_list else None,
                "model_type": model_type,
                "reverse": reverse,
                "is_cloze": is_cloze,
                "ef": float(row.get("ef") or 2.5),
                "interval_days": int(row.get("interval_days") or 0),
                "repetitions": int(row.get("repetitions") or 0),
                "lapses": int(row.get("lapses") or 0),
                "due_at": row.get("due_at"),
            }
            uuid = db.add_flashcard(data)
            if _collect_flashcard_asset_refs_by_field(data):
                _reconcile_flashcard_assets_or_500(uuid, db)
            if tags_list:
                db.set_flashcard_tags(uuid, tags_list)
            created.append({"uuid": uuid, "deck_id": deck_id})

        return {"imported": len(created), "items": created, "errors": errors}
    except APKGImportError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except HTTPException:
        raise
    except CharactersRAGDBError as e:
        logger.error(f"Failed to import APKG flashcards: {e}")
        raise HTTPException(status_code=500, detail="Failed to import flashcards") from e
    except _FLASHCARDS_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"APKG import failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to import APKG flashcards") from e


@router.post("/review", response_model=FlashcardReviewResponse)
def review_flashcard(payload: FlashcardReviewRequest, db: CharactersRAGDB = Depends(get_chacha_db_for_user)):
    try:
        card = db.get_flashcard(payload.card_uuid)
        if not card:
            raise HTTPException(status_code=404, detail=f"Flashcard not found ({payload.card_uuid})")
        deck_id = int(card["deck_id"]) if card.get("deck_id") is not None else None
        session = db.get_or_create_flashcard_review_session(
            deck_id=deck_id,
            review_mode="due",
            tag_filter=None,
            scope_key=_build_review_scope_key(review_mode="due", deck_id=deck_id),
        )
        updated = db.review_flashcard(
            payload.card_uuid,
            payload.rating,
            payload.answer_time_ms,
            review_session_id=session["id"],
        )
        return updated
    except (SchedulerSettingsError, FsrsSettingsError) as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except ConflictError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except CharactersRAGDBError as e:
        logger.error(f"Failed to review flashcard: {e}")
        raise HTTPException(status_code=500, detail="Failed to review flashcard") from e


@router.get("/review-sessions", response_model=list[FlashcardReviewSessionSummary])
def list_review_sessions(
    deck_id: Optional[int] = Query(None, ge=1),
    scope_key: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = Query(20, ge=1, le=200),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
):
    try:
        return db.list_flashcard_review_sessions(
            deck_id=deck_id,
            scope_key=scope_key,
            status=status,
            limit=limit,
        )
    except InputError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except CharactersRAGDBError as exc:
        logger.error(f"Failed to list flashcard review sessions: {exc}")
        raise HTTPException(status_code=500, detail="Failed to list flashcard review sessions") from exc


@router.post("/review-sessions/end", response_model=FlashcardReviewSessionSummary)
def end_review_session(
    payload: ReviewSessionEndRequest,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    current_user: User = Depends(get_request_user),
    jm: JobManager = Depends(get_job_manager),
):
    try:
        session = db.mark_flashcard_review_session_completed(int(payload.review_session_id))
        _enqueue_study_suggestions_refresh(
            jm=jm,
            current_user=current_user,
            anchor_type="flashcard_review_session",
            anchor_id=int(session["id"]),
        )
        return session
    except ConflictError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except CharactersRAGDBError as exc:
        logger.error(f"Failed to complete flashcard review session: {exc}")
        raise HTTPException(status_code=500, detail="Failed to complete flashcard review session") from exc


@router.get("/review/next", response_model=FlashcardNextReviewResponse)
def get_next_review_card(
    deck_id: Optional[int] = Query(None, ge=1),
    workspace_id: Optional[str] = None,
    include_workspace_items: bool = False,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
):
    try:
        card, selection_reason = db.get_next_review_card(
            deck_id=deck_id,
            workspace_id=workspace_id,
            include_workspace_items=include_workspace_items,
        )
        if not card:
            return {"card": None, "selection_reason": "none"}
        deck = db.get_deck(int(card["deck_id"])) if card.get("deck_id") is not None else None
        card = _attach_scheduler_preview(card, deck)
        return {"card": card, "selection_reason": selection_reason}
    except (SchedulerSettingsError, FsrsSettingsError) as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except CharactersRAGDBError as e:
        logger.error(f"Failed to fetch next review card: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch next review card") from e


@router.post("/study-packs/jobs", response_model=StudyPackJobAcceptedResponse, status_code=202)
def create_study_pack_job(
    payload: StudyPackCreateJobRequest,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    current_user: User = Depends(get_request_user),
    jm: JobManager = Depends(get_job_manager),
):
    try:
        _ensure_workspace_exists(db, payload.workspace_id)
        job = jm.create_job(
            domain=STUDY_PACKS_DOMAIN,
            queue=study_pack_jobs_queue(),
            job_type=STUDY_PACKS_JOB_TYPE,
            payload=build_study_pack_job_payload(payload),
            owner_user_id=str(current_user.id),
            priority=5,
            max_retries=2,
        )
        return {"job": _serialize_study_pack_job(job)}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/study-packs/jobs/{job_id}", response_model=StudyPackJobStatusResponse)
async def get_study_pack_job_status(
    job_id: int,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    current_user: User = Depends(get_request_user),
    principal: AuthPrincipal = Depends(get_auth_principal),
    jm: JobManager = Depends(get_job_manager),
):
    job = jm.get_job(job_id)
    if not job or str(job.get("domain") or "").strip().lower() != STUDY_PACKS_DOMAIN:
        raise HTTPException(status_code=404, detail="Study-pack job not found")
    _ensure_study_pack_job_access(job, current_user=current_user, principal=principal)

    study_pack = None
    raw_status = str(job.get("status") or "").strip().lower()
    if raw_status == "completed":
        study_pack_db = await _study_pack_db_for_job(
            job,
            request_db=db,
            current_user=current_user,
            principal=principal,
        )
        study_pack = _study_pack_from_job_result(study_pack_db, job)
    error = _public_study_pack_job_error(job)
    return {
        "job": _serialize_study_pack_job(job),
        "study_pack": study_pack,
        "error": error,
    }


@router.get("/study-packs/{pack_id}", response_model=StudyPackSummaryResponse)
def get_study_pack_detail(
    pack_id: int,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
):
    pack = db.get_study_pack(pack_id)
    if not pack:
        raise HTTPException(status_code=404, detail="Study pack not found")
    return _serialize_study_pack(pack)


@router.post("/study-packs/{pack_id}/regenerate", response_model=StudyPackJobAcceptedResponse, status_code=202)
def regenerate_study_pack(
    pack_id: int,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    current_user: User = Depends(get_request_user),
    jm: JobManager = Depends(get_job_manager),
):
    pack = db.get_study_pack(pack_id)
    if not pack:
        raise HTTPException(status_code=404, detail="Study pack not found")

    source_items = extract_study_pack_source_items(pack.get("source_bundle_json"))
    if not source_items:
        raise HTTPException(status_code=400, detail="Study pack source bundle is empty")

    request = StudyPackCreateJobRequest.model_validate(
        {
            "title": pack.get("title"),
            "workspace_id": pack.get("workspace_id"),
            "deck_mode": "new",
            "source_items": source_items,
        }
    )
    job = jm.create_job(
        domain=STUDY_PACKS_DOMAIN,
        queue=study_pack_jobs_queue(),
        job_type=STUDY_PACKS_JOB_TYPE,
        payload=build_study_pack_job_payload(
            request,
            regenerate_from_pack_id=pack_id,
            expected_version=int(pack["version"]) if pack.get("version") is not None else None,
        ),
        owner_user_id=str(current_user.id),
        priority=5,
        max_retries=2,
    )
    return {"job": _serialize_study_pack_job(job)}


@router.get("/{card_uuid}/assistant", response_model=StudyAssistantContextResponse)
def get_flashcard_assistant(
    card_uuid: str,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
):
    try:
        _fetch_flashcard_or_404(card_uuid, db)
        context = build_flashcard_assistant_context(db, card_uuid)
        return {
            "thread": context["thread"],
            "messages": context["history"],
            "context_snapshot": _build_assistant_context_snapshot(context),
            "available_actions": context["available_actions"],
            "citations": context.get("citations") or [],
            "primary_citation": context.get("primary_citation"),
            "deep_dive_target": context.get("deep_dive_target"),
            "study_pack": context.get("study_pack"),
        }
    except HTTPException:
        raise
    except ConflictError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except CharactersRAGDBError as exc:
        logger.error(f"Failed to fetch flashcard assistant context: {exc}")
        raise HTTPException(status_code=500, detail="Failed to fetch study assistant context") from exc


@router.post("/{card_uuid}/assistant/respond", response_model=StudyAssistantRespondResponse)
async def respond_flashcard_assistant(
    card_uuid: str,
    payload: StudyAssistantRespondRequest,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
):
    try:
        _fetch_flashcard_or_404(card_uuid, db)
        context = build_flashcard_assistant_context(db, card_uuid)
        thread = context["thread"]
        if payload.expected_thread_version is not None and int(thread["version"]) != int(payload.expected_thread_version):
            raise HTTPException(status_code=409, detail="Study assistant thread version mismatch")

        user_content = str(payload.message or "").strip() or _default_study_assistant_message(payload.action, context)
        reply = await generate_study_assistant_reply(
            action=payload.action,
            context=context,
            message=user_content,
            provider=payload.provider,
            model=payload.model,
        )
        context_snapshot = _build_assistant_context_snapshot(context)
        user_message = db.append_study_assistant_message(
            thread_id=int(thread["id"]),
            role="user",
            action_type=payload.action,
            input_modality=payload.input_modality,
            content=user_content,
            structured_payload={"action": payload.action},
            context_snapshot=context_snapshot,
            provider=payload.provider,
            model=payload.model,
            expected_thread_version=payload.expected_thread_version,
        )
        assistant_message = db.append_study_assistant_message(
            thread_id=int(thread["id"]),
            role="assistant",
            action_type=payload.action,
            input_modality="text",
            content=str(reply.get("assistant_text") or "").strip(),
            structured_payload=reply.get("structured_payload") or {},
            context_snapshot=context_snapshot,
            provider=str(reply.get("provider") or payload.provider or "default"),
            model=reply.get("model") or payload.model,
        )
        updated_thread = db.get_study_assistant_thread(int(thread["id"]))
        if not updated_thread:
            raise HTTPException(status_code=404, detail="Study assistant thread not found after update")
        return {
            "thread": updated_thread,
            "user_message": user_message,
            "assistant_message": assistant_message,
            "structured_payload": reply.get("structured_payload") or {},
            "context_snapshot": context_snapshot,
        }
    except HTTPException:
        raise
    except ConflictError as exc:
        raise HTTPException(status_code=409, detail="Study assistant thread version mismatch") from exc
    except CharactersRAGDBError as exc:
        logger.error(f"Failed to respond with flashcard assistant: {exc}")
        raise HTTPException(status_code=500, detail="Failed to generate study assistant response") from exc
    except _FLASHCARDS_NONCRITICAL_EXCEPTIONS as exc:
        logger.error(f"Unexpected flashcard assistant failure: {exc}")
        raise HTTPException(status_code=500, detail="Failed to generate study assistant response") from exc


@router.post("/generate", response_model=FlashcardGenerateResponse)
async def generate_flashcards(payload: FlashcardGenerateRequest):
    """Generate flashcards from free text using the workflows flashcard_generate adapter."""
    try:
        provider = _resolve_flashcard_generation_provider(payload.provider)
        result = await run_flashcard_generate_adapter(
            {
                "text": payload.text,
                "num_cards": payload.num_cards,
                "card_type": payload.card_type,
                "difficulty": payload.difficulty,
                "focus_topics": payload.focus_topics,
                "provider": provider,
                "model": payload.model,
            },
            {},
        )

        if isinstance(result, dict) and result.get("__status__") == "cancelled":
            raise HTTPException(status_code=499, detail="Generation cancelled")

        error = result.get("error") if isinstance(result, dict) else None
        if error:
            if is_test_mode() and _should_return_test_mode_flashcards(error):
                generated_cards = _build_test_mode_flashcards(payload)
                return {
                    "flashcards": generated_cards,
                    "count": len(generated_cards),
                }
            raise HTTPException(status_code=400, detail=str(error))

        raw_flashcards = result.get("flashcards") if isinstance(result, dict) else []
        generated_cards: list[dict] = []
        for raw in raw_flashcards or []:
            if not isinstance(raw, dict):
                continue
            front = str(raw.get("front") or "").strip()
            back = str(raw.get("back") or "").strip()
            if not front or not back:
                continue

            tags_value = raw.get("tags")
            if isinstance(tags_value, list):
                tags = [str(tag).strip() for tag in tags_value if str(tag).strip()]
            elif isinstance(tags_value, str):
                tags = [token for token in tags_value.replace(",", " ").split() if token]
            else:
                tags = []

            model_type = str(raw.get("model_type") or payload.card_type).lower()
            if model_type not in ("basic", "basic_reverse", "cloze"):
                model_type = payload.card_type

            card = {
                "front": front,
                "back": back,
                "tags": tags,
                "model_type": model_type,
            }
            notes = raw.get("notes")
            if isinstance(notes, str) and notes.strip():
                card["notes"] = notes
            extra = raw.get("extra")
            if isinstance(extra, str) and extra.strip():
                card["extra"] = extra
            generated_cards.append(card)

        return {
            "flashcards": generated_cards,
            "count": len(generated_cards),
        }
    except HTTPException:
        raise
    except _FLASHCARDS_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Flashcard generation failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate flashcards") from e


@router.get("/export")
def export_flashcards(
    deck_id: Optional[int] = None,
    workspace_id: Optional[str] = None,
    include_workspace_items: bool = False,
    tag: Optional[str] = None,
    q: Optional[str] = None,
    export_format: Optional[str] = Query("csv", alias="format", pattern="^(csv|apkg|json)$"),
    include_reverse: Optional[bool] = False,
    delimiter: Optional[str] = Query('\t', description="CSV/TSV delimiter; default tab"),
    include_header: Optional[bool] = Query(False, description="Include header row for CSV/TSV"),
    extended_header: Optional[bool] = Query(False, description="Include Extra and Reverse columns"),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user)
):
    try:
        items = db.list_flashcards(
            deck_id=deck_id,
            workspace_id=workspace_id,
            include_workspace_items=include_workspace_items,
            tag=tag,
            q=q,
            due_status='all',
            include_deleted=False,
            limit=100000,
            offset=0,
        )
        if export_format == 'json':
            def _parse_tags(raw) -> list:
                if isinstance(raw, str):
                    try:
                        parsed = json.loads(raw)
                        if isinstance(parsed, list):
                            return [str(t).strip() for t in parsed if str(t).strip()]
                        return []
                    except (json.JSONDecodeError, TypeError):
                        return []
                if isinstance(raw, list):
                    return [str(t).strip() for t in raw if str(t).strip()]
                return []

            def _stream_json():
                yield "[\n"
                first = True
                for item in items:
                    row = {
                        "front": item.get("front", ""),
                        "back": item.get("back", ""),
                        "notes": item.get("notes", ""),
                        "tags": _parse_tags(item.get("tags_json")),
                        "deck": item.get("deck_name", ""),
                        "model_type": item.get("model_type", "basic"),
                        "extra": item.get("extra", ""),
                        "reverse": bool(item.get("reverse")),
                        "is_cloze": bool(item.get("is_cloze")),
                    }
                    chunk = json.dumps(row, ensure_ascii=False)
                    if not first:
                        yield ",\n"
                    first = False
                    yield "  " + chunk
                yield "\n]\n"

            return StreamingResponse(
                _stream_json(),
                media_type="application/json; charset=utf-8",
                headers={"Content-Disposition": "attachment; filename=flashcards.json"},
            )
        if export_format == 'apkg':
            apkg_max_media_bytes = _get_flashcards_apkg_max_media_bytes()

            def asset_loader(asset_uuid: str) -> dict[str, Any]:
                asset = db.get_flashcard_asset(asset_uuid)
                if not asset:
                    raise ValueError(f"Managed flashcard asset not found: {asset_uuid}")
                content = db.get_flashcard_asset_content(asset_uuid)
                if content is None:
                    raise ValueError(f"Managed flashcard asset content missing: {asset_uuid}")
                return {
                    "content": content,
                    "mime_type": asset.get("mime_type"),
                    "original_filename": asset.get("original_filename"),
                }

            apkg = export_apkg_from_rows(
                items,
                include_reverse=include_reverse,
                asset_loader=asset_loader,
                max_total_media_bytes=apkg_max_media_bytes,
            )
            return StreamingResponse(iter([apkg]), media_type="application/apkg",
                                     headers={"Content-Disposition": "attachment; filename=flashcards.apkg"})
        # default csv/tsv
        dlm = delimiter or '\t'
        data = db.export_flashcards_csv(
            deck_id=deck_id,
            workspace_id=workspace_id,
            include_workspace_items=include_workspace_items,
            tag=tag,
            q=q,
            delimiter=dlm,
            include_header=bool(include_header),
            extended_header=bool(extended_header),
        )
        media_type = "text/tab-separated-values; charset=utf-8" if dlm == '\t' else "text/csv; charset=utf-8"
        filename = "flashcards.tsv" if dlm == '\t' else "flashcards.csv"
        return StreamingResponse(iter([data]), media_type=media_type,
                                 headers={"Content-Disposition": f"attachment; filename={filename}"})
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except CharactersRAGDBError as e:
        logger.error(f"Failed to export flashcards: {e}")
        raise HTTPException(status_code=500, detail="Failed to export flashcards") from e


@router.post("/templates", response_model=FlashcardTemplate)
def create_flashcard_template(
    payload: FlashcardTemplateCreate,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
) -> FlashcardTemplate:
    """Create a reusable flashcard authoring template for the current user."""

    try:
        template_id = db.add_flashcard_template(**payload.model_dump())
        return _fetch_flashcard_template_or_404(template_id, db)
    except InputError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except ConflictError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    except CharactersRAGDBError as e:
        logger.bind(template_name=payload.name, model_type=payload.model_type).error(
            f"Failed to create flashcard template: {e}"
        )
        raise HTTPException(status_code=500, detail="Failed to create flashcard template") from e


@router.get("/templates", response_model=FlashcardTemplateListResponse)
def list_flashcard_templates(
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
) -> FlashcardTemplateListResponse:
    """List flashcard authoring templates for the current user."""

    try:
        items = db.list_flashcard_templates(limit=limit, offset=offset)
        return FlashcardTemplateListResponse(
            items=items,
            count=len(items),
            total=db.count_flashcard_templates(),
        )
    except CharactersRAGDBError as e:
        logger.bind(limit=limit, offset=offset).error(
            f"Failed to list flashcard templates: {e}"
        )
        raise HTTPException(status_code=500, detail="Failed to list flashcard templates") from e


@router.get("/templates/{template_id}", response_model=FlashcardTemplate)
def get_flashcard_template(
    template_id: int,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
) -> FlashcardTemplate:
    """Fetch a single flashcard authoring template by ID."""

    try:
        return _fetch_flashcard_template_or_404(template_id, db)
    except CharactersRAGDBError as e:
        logger.bind(template_id=template_id).error(
            f"Failed to get flashcard template: {e}"
        )
        raise HTTPException(status_code=500, detail="Failed to get flashcard template") from e


@router.patch("/templates/{template_id}", response_model=FlashcardTemplate)
def update_flashcard_template(
    template_id: int,
    payload: FlashcardTemplateUpdate,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
) -> FlashcardTemplate:
    """Update a flashcard authoring template using optimistic locking when provided."""

    data: dict[str, Any] = {}
    expected_version: Optional[int] = None
    try:
        data = payload.model_dump(exclude_unset=True)
        expected_version = data.pop("expected_version", None)
        ok = db.update_flashcard_template(template_id, data, expected_version)
        if not ok:
            raise HTTPException(status_code=404, detail="Flashcard template not found or not updated")
        return _fetch_flashcard_template_or_404(template_id, db)
    except InputError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except ConflictError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    except CharactersRAGDBError as e:
        logger.bind(
            template_id=template_id,
            expected_version=expected_version,
            updated_fields=sorted(data.keys()),
        ).error(f"Failed to update flashcard template: {e}")
        raise HTTPException(status_code=500, detail="Failed to update flashcard template") from e


@router.delete("/templates/{template_id}")
def delete_flashcard_template(
    template_id: int,
    expected_version: int = Query(..., ge=1),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
) -> dict[str, bool]:
    """Soft-delete a flashcard authoring template using optimistic locking."""

    try:
        ok = db.soft_delete_flashcard_template(template_id, expected_version)
        if not ok:
            raise HTTPException(status_code=404, detail="Flashcard template not found or already deleted")
        return {"deleted": True}
    except ConflictError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    except CharactersRAGDBError as e:
        logger.bind(template_id=template_id, expected_version=expected_version).error(
            f"Failed to delete flashcard template: {e}"
        )
        raise HTTPException(status_code=500, detail="Failed to delete flashcard template") from e


@router.get("/{card_uuid}", response_model=Flashcard)
def get_flashcard_alias(card_uuid: str, db: CharactersRAGDB = Depends(get_chacha_db_for_user)):
    try:
        return _fetch_flashcard_or_404(card_uuid, db)
    except CharactersRAGDBError as e:
        logger.error(f"Failed to get flashcard: {e}")
        raise HTTPException(status_code=500, detail="Failed to get flashcard") from e
