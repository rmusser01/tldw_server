"""Shared API routes for reading and refreshing study-suggestion snapshots."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from loguru import logger

from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal
from tldw_Server_API.app.api.v1.schemas.study_suggestions import (
    SuggestionActionRequest,
    SuggestionActionResponse,
    SuggestionJobAcceptedResponse,
    SuggestionRefreshRequest,
    SuggestionSnapshotResponse,
    SuggestionStatusResponse,
)
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, ConflictError
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.StudySuggestions.actions import (
    build_flashcard_generation_payload,
    build_follow_up_flashcard_deck_name,
    build_follow_up_flashcard_source_text,
    build_selection_fingerprint,
    canonicalize_follow_up_action,
    finalize_generation_link,
    find_generation_link_by_fingerprint,
    is_pending_generation_target_id,
    normalize_selected_topics,
    release_generation_link_reservation,
    reserve_generation_link,
    resolve_lineage_equivalent_topic_selection,
    resolve_selected_topic_identity_groups,
    resolve_selected_topic_labels,
    resolve_selected_topic_normalization_version,
    resolve_selected_topic_semantic_keys,
    soft_delete_deck,
)
from tldw_Server_API.app.core.StudySuggestions import snapshot_service
from tldw_Server_API.app.core.StudySuggestions.jobs import (
    STUDY_SUGGESTIONS_DOMAIN,
    STUDY_SUGGESTIONS_REFRESH_JOB_TYPE,
    build_study_suggestions_job_payload,
    study_suggestions_jobs_queue,
)
from tldw_Server_API.app.core.Workflows.adapters.content import run_flashcard_generate_adapter
from tldw_Server_API.app.services.quiz_generator import generate_quiz_from_sources


router = APIRouter(prefix="/study-suggestions", tags=["study-suggestions"])

_job_manager_cache: dict[str, JobManager] = {}
_job_manager_lock = threading.Lock()


def get_job_manager() -> JobManager:
    """Return a cached JobManager keyed by JOBS_DB_URL or JOBS_DB_PATH."""

    import os

    db_url = (os.getenv("JOBS_DB_URL") or "").strip()
    db_path = (os.getenv("JOBS_DB_PATH") or "").strip()
    cache_key = f"url:{db_url}" if db_url else f"path:{db_path or 'default'}"
    with _job_manager_lock:
        cached = _job_manager_cache.get(cache_key)
        if cached is not None:
            return cached
        if db_url:
            backend = "postgres" if db_url.startswith("postgres") else None
            job_manager = JobManager(backend=backend, db_url=db_url)
        elif db_path:
            job_manager = JobManager(db_path=Path(db_path))
        else:
            job_manager = JobManager()
        _job_manager_cache[cache_key] = job_manager
        return job_manager


def _serialize_job(job: dict[str, object]) -> dict[str, object]:
    return {
        "id": int(job["id"]),
        "status": str(job.get("status") or "queued"),
    }


def _is_generation_link_target_live(note_db: CharactersRAGDB, generation_link: dict[str, Any]) -> bool:
    """Return whether a persisted generation link still points at a launchable target."""

    target_service = str(generation_link.get("target_service") or "").strip().lower()
    target_type = str(generation_link.get("target_type") or "").strip().lower()
    target_id = str(generation_link.get("target_id") or "").strip()
    if not target_id or is_pending_generation_target_id(target_id):
        return False
    if target_service == "quiz" and target_type == "quiz":
        try:
            quiz_id = int(target_id)
        except (TypeError, ValueError):
            return False
        return note_db.get_quiz(quiz_id) is not None
    if target_service == "flashcards" and target_type == "deck":
        try:
            deck_id = int(target_id)
        except (TypeError, ValueError):
            return False
        deck = note_db.get_deck(deck_id)
        return bool(deck) and not bool(deck.get("deleted"))
    return False


def _iter_refreshed_ancestor_snapshots(
    note_db: CharactersRAGDB,
    snapshot_row: dict[str, Any],
) -> Iterator[dict[str, Any]]:
    """Yield refreshed ancestor snapshots from newest parent to oldest reachable ancestor."""

    seen_ids: set[int] = set()
    parent_id = snapshot_row.get("refreshed_from_snapshot_id")
    while parent_id is not None:
        try:
            ancestor_snapshot = note_db.get_suggestion_snapshot(int(parent_id))
        except (TypeError, ValueError):
            break
        if not ancestor_snapshot:
            break
        ancestor_id = int(ancestor_snapshot["id"])
        if ancestor_id in seen_ids:
            break
        seen_ids.add(ancestor_id)
        yield ancestor_snapshot
        parent_id = ancestor_snapshot.get("refreshed_from_snapshot_id")


def _find_live_generation_link_in_refreshed_lineage(
    note_db: CharactersRAGDB,
    snapshot_row: dict[str, Any],
    *,
    target_service: str,
    target_type: str,
    requested_identity_groups: list[frozenset[str]],
    action_kind: str,
    generator_version: str | None,
) -> dict[str, Any] | None:
    """Find a reusable live generation link in refreshed ancestor snapshots."""

    if not requested_identity_groups:
        return None

    for ancestor_snapshot in _iter_refreshed_ancestor_snapshots(note_db, snapshot_row):
        lineage_selection = resolve_lineage_equivalent_topic_selection(
            ancestor_snapshot,
            requested_identity_groups=requested_identity_groups,
        )
        if lineage_selection is None:
            continue
        selected_topics, normalization_version = lineage_selection
        current_fingerprint, legacy_fingerprint = _build_selection_fingerprint_variants(
            snapshot_id=int(ancestor_snapshot["id"]),
            target_service=target_service,
            target_type=target_type,
            selected_topics=selected_topics,
            action_kind=action_kind,
            generator_version=generator_version,
            normalization_version=normalization_version,
        )
        candidates = _list_generation_link_candidates(
            note_db,
            snapshot_id=int(ancestor_snapshot["id"]),
            target_service=target_service,
            target_type=target_type,
            selection_fingerprint=current_fingerprint,
            legacy_selection_fingerprint=legacy_fingerprint,
        )
        for existing_link, _matched_fingerprint in candidates:
            if _is_generation_link_target_live(note_db, existing_link):
                return existing_link
    return None


def _build_selection_fingerprint_variants(
    *,
    snapshot_id: int,
    target_service: str,
    target_type: str,
    selected_topics: list[str],
    action_kind: str,
    generator_version: str | None,
    normalization_version: str,
) -> tuple[str, str | None]:
    """Build current and legacy-compatible selection fingerprints for one action."""

    selection_fingerprint = build_selection_fingerprint(
        snapshot_id=snapshot_id,
        target_service=target_service,
        target_type=target_type,
        selected_topics=selected_topics,
        action_kind=action_kind,
        generator_version=generator_version,
        normalization_version=normalization_version,
    )
    legacy_selection_fingerprint = build_selection_fingerprint(
        snapshot_id=snapshot_id,
        target_service=target_service,
        target_type=target_type,
        selected_topics=selected_topics,
        action_kind=action_kind,
        generator_version=generator_version,
        normalization_version=normalization_version,
        include_normalization_version=False,
    )
    return selection_fingerprint, (
        legacy_selection_fingerprint if legacy_selection_fingerprint != selection_fingerprint else None
    )


def _list_generation_link_candidates(
    note_db: CharactersRAGDB,
    *,
    snapshot_id: int,
    target_service: str,
    target_type: str,
    selection_fingerprint: str,
    legacy_selection_fingerprint: str | None,
) -> list[tuple[dict[str, Any], str]]:
    """Return matching generation links for current and legacy fingerprints."""

    candidates: list[tuple[dict[str, Any], str]] = []
    seen_fingerprints: set[str] = set()
    for fingerprint in (selection_fingerprint, legacy_selection_fingerprint):
        if not fingerprint or fingerprint in seen_fingerprints:
            continue
        seen_fingerprints.add(fingerprint)
        existing_link = find_generation_link_by_fingerprint(
            note_db,
            snapshot_id=snapshot_id,
            target_service=target_service,
            target_type=target_type,
            selection_fingerprint=fingerprint,
        )
        if existing_link:
            candidates.append((existing_link, fingerprint))
    return candidates


def _resolve_quiz_sources(
    note_db: CharactersRAGDB,
    snapshot_row: dict[str, Any],
) -> tuple[list[dict[str, str]], str | None, str | None]:
    """Synchronous: resolve quiz generation sources from a snapshot anchor."""

    anchor_type = str(snapshot_row.get("anchor_type") or "").strip().lower()
    anchor_id = int(snapshot_row["anchor_id"])
    workspace_id: str | None = None
    workspace_tag: str | None = None
    if anchor_type == "quiz_attempt":
        attempt = note_db.get_attempt(anchor_id, include_questions=True, include_answers=True)
        if not attempt:
            raise HTTPException(status_code=404, detail="Quiz attempt not found")
        quiz = note_db.get_quiz(int(attempt["quiz_id"]))
        if not quiz:
            raise HTTPException(status_code=404, detail="Quiz not found")
        sources = list(quiz.get("source_bundle_json") or [])
        if not sources and quiz.get("media_id") is not None:
            sources = [{"source_type": "media", "source_id": str(quiz["media_id"])}]
        workspace_id = quiz.get("workspace_id")
        workspace_tag = quiz.get("workspace_tag")
    elif anchor_type == "flashcard_review_session":
        session = note_db.get_flashcard_review_session(anchor_id)
        if not session:
            raise HTTPException(status_code=404, detail="Flashcard review session not found")
        deck_id = session.get("deck_id")
        if deck_id is None:
            raise HTTPException(status_code=400, detail="Flashcard review session has no deck lineage for quiz follow-up")
        deck = note_db.get_deck(int(deck_id))
        workspace_id = str(deck.get("workspace_id") or "").strip() or None if deck else None
        tag_filter = str(session.get("tag_filter") or "").strip() or None
        if tag_filter:
            cards = note_db.list_flashcards(
                deck_id=int(deck_id),
                tag=tag_filter,
                due_status="all",
                limit=25,
            )
            sources = [
                {"source_type": "flashcard_card", "source_id": str(card["uuid"])}
                for card in cards
                if str(card.get("uuid") or "").strip()
            ]
            if not sources:
                sources = [{"source_type": "flashcard_deck", "source_id": str(int(deck_id))}]
        else:
            sources = [{"source_type": "flashcard_deck", "source_id": str(int(deck_id))}]
    else:
        raise HTTPException(status_code=400, detail="Quiz follow-up actions require quiz-attempt or flashcard-session lineage")
    if not sources:
        raise HTTPException(status_code=400, detail="Quiz follow-up actions require reusable source lineage")
    return sources, workspace_id, workspace_tag


def _persist_flashcard_deck(
    note_db: CharactersRAGDB,
    *,
    snapshot_row: dict[str, Any],
    selected_topics: list[str],
    raw_flashcards: list[Any],
) -> str:
    """Synchronous: create deck and insert generated flashcards."""

    # Retry deck creation on name conflict (TOCTOU race under concurrent workers).
    max_retries = 3
    deck_id = None
    for attempt in range(max_retries):
        deck_name = build_follow_up_flashcard_deck_name(
            note_db,
            snapshot_id=int(snapshot_row["id"]),
            selected_topics=selected_topics,
        )
        try:
            deck_id = note_db.add_deck(
                deck_name,
                description=f"Generated from study suggestion snapshot {int(snapshot_row['id'])}.",
            )
            break
        except ConflictError:
            if attempt == max_retries - 1:
                raise
            continue
    if deck_id is None:
        raise ConflictError(  # noqa: TRY003
            "Failed to create deck after retries",
            entity="decks",
            identifier=f"snapshot:{int(snapshot_row['id'])}",
        )
    flashcard_payloads = build_flashcard_generation_payload(
        deck_id=int(deck_id),
        selected_topics=selected_topics,
        raw_flashcards=raw_flashcards,
    )
    try:
        note_db.add_flashcards_bulk(flashcard_payloads)
    except Exception as exc:
        with contextlib.suppress(Exception):
            soft_delete_deck(note_db, deck_id=int(deck_id))
        logger.warning("Flashcard follow-up generation cleanup deleted deck {} after insert failure: {}", deck_id, exc)
        raise
    return str(deck_id)


async def _dispatch_follow_up_action(
    *,
    note_db: CharactersRAGDB,
    snapshot_row: dict[str, Any],
    request_body: dict[str, Any],
    media_db: Any = None,
) -> dict[str, str]:
    target_service = str(request_body["target_service"]).strip().lower()
    target_type = str(request_body["target_type"]).strip().lower()
    selected_topics = list(request_body.get("selected_topics") or [])

    if target_service == "quiz":
        sources, workspace_id, workspace_tag = await asyncio.to_thread(
            _resolve_quiz_sources, note_db, snapshot_row,
        )
        result = await generate_quiz_from_sources(
            db=note_db,
            media_db=media_db,
            sources=sources,
            num_questions=5,
            question_types=None,
            difficulty="mixed",
            focus_topics=selected_topics,
            model=None,
            api_provider=None,
            workspace_id=workspace_id,
            workspace_tag=workspace_tag,
        )
        quiz_row = result.get("quiz") if isinstance(result, dict) else None
        target_id = str((quiz_row or {}).get("id") or "").strip()
        if not target_id:
            raise HTTPException(status_code=500, detail="Quiz follow-up generation did not return a quiz id")
        return {
            "target_service": "quiz",
            "target_type": target_type,
            "target_id": target_id,
        }

    if target_service == "flashcards":
        source_text = await asyncio.to_thread(
            build_follow_up_flashcard_source_text,
            note_db,
            snapshot_row=snapshot_row,
            selected_topics=selected_topics,
        )
        result = await run_flashcard_generate_adapter(
            {
                "text": source_text,
                "num_cards": 10,
                "card_type": "basic",
                "difficulty": "mixed",
                "focus_topics": selected_topics,
                "provider": None,
                "model": None,
            },
            {},
        )
        if isinstance(result, dict) and result.get("__status__") == "cancelled":
            raise HTTPException(status_code=499, detail="Generation cancelled")
        if isinstance(result, dict) and result.get("error"):
            raise HTTPException(status_code=400, detail=str(result["error"]))

        raw_flashcards = result.get("flashcards") if isinstance(result, dict) else []
        has_valid_flashcards = any(
            str(raw_card.get("front") or "").strip() and str(raw_card.get("back") or "").strip()
            for raw_card in raw_flashcards or []
            if isinstance(raw_card, dict)
        )
        if not has_valid_flashcards:
            raise HTTPException(status_code=500, detail="Flashcard follow-up generation returned no valid cards")
        deck_id = await asyncio.to_thread(
            _persist_flashcard_deck,
            note_db,
            snapshot_row=snapshot_row,
            selected_topics=selected_topics,
            raw_flashcards=raw_flashcards or [],
        )
        return {
            "target_service": "flashcards",
            "target_type": target_type,
            "target_id": deck_id,
        }

    raise HTTPException(status_code=400, detail="Unsupported study suggestion action target")


@router.get("/anchors/{anchor_type}/{anchor_id}/status", response_model=SuggestionStatusResponse)
def get_suggestion_status(
    anchor_type: str,
    anchor_id: int,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    current_user: User = Depends(get_request_user),
    jm: JobManager = Depends(get_job_manager),
) -> SuggestionStatusResponse:
    status_payload = snapshot_service.get_anchor_status(
        note_db=db,
        job_manager=jm,
        anchor_type=anchor_type,
        anchor_id=anchor_id,
        owner_user_id=str(current_user.id),
    )
    return SuggestionStatusResponse.model_validate(status_payload)


@router.get("/snapshots/{snapshot_id}", response_model=SuggestionSnapshotResponse)
def get_suggestion_snapshot(
    snapshot_id: int,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> SuggestionSnapshotResponse:
    snapshot_row = db.get_suggestion_snapshot(snapshot_id)
    if not snapshot_row:
        raise HTTPException(status_code=404, detail="Suggestion snapshot not found")

    try:
        live_evidence = snapshot_service.load_live_evidence_for_snapshot(
            snapshot_row,
            note_db=db,
            principal=principal,
        )
    except PermissionError:
        live_evidence = {
            "source_available": False,
            "reason": "unavailable",
        }

    return SuggestionSnapshotResponse.model_validate(
        {
            "snapshot": snapshot_service.serialize_snapshot(snapshot_row),
            "live_evidence": live_evidence,
        }
    )


@router.post(
    "/snapshots/{snapshot_id}/refresh",
    response_model=SuggestionJobAcceptedResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def refresh_suggestion_snapshot(
    snapshot_id: int,
    payload: SuggestionRefreshRequest,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    current_user: User = Depends(get_request_user),
    jm: JobManager = Depends(get_job_manager),
) -> SuggestionJobAcceptedResponse:
    _ = payload
    snapshot_row = db.get_suggestion_snapshot(snapshot_id)
    if not snapshot_row:
        raise HTTPException(status_code=404, detail="Suggestion snapshot not found")

    try:
        job = jm.create_job(
            domain=STUDY_SUGGESTIONS_DOMAIN,
            queue=study_suggestions_jobs_queue(),
            job_type=STUDY_SUGGESTIONS_REFRESH_JOB_TYPE,
            payload=build_study_suggestions_job_payload(
                job_type=STUDY_SUGGESTIONS_REFRESH_JOB_TYPE,
                anchor_type=str(snapshot_row["anchor_type"]),
                anchor_id=int(snapshot_row["anchor_id"]),
                snapshot_id=int(snapshot_row["id"]),
            ),
            owner_user_id=str(current_user.id),
            priority=5,
            max_retries=1,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return SuggestionJobAcceptedResponse.model_validate({"job": _serialize_job(job)})


def _prepare_action(
    db: CharactersRAGDB,
    snapshot_id: int,
    payload: SuggestionActionRequest,
) -> tuple[dict[str, Any], dict[str, str], list[str], str, bool, str | None]:
    """Synchronous pre-dispatch phase: validate, deduplicate, and reserve."""

    snapshot_row = db.get_suggestion_snapshot(snapshot_id)
    if not snapshot_row:
        raise HTTPException(status_code=404, detail="Suggestion snapshot not found")
    try:
        action_contract = canonicalize_follow_up_action(
            target_service=payload.target_service,
            target_type=payload.target_type,
            action_kind=payload.action_kind,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    selected_topics = resolve_selected_topic_labels(
        snapshot_row,
        selected_topic_ids=payload.selected_topic_ids,
        selected_topic_edits=payload.selected_topic_edits,
        manual_topic_labels=payload.manual_topic_labels,
        has_explicit_selection=payload.has_explicit_selection,
    )
    semantic_selected_topics = resolve_selected_topic_semantic_keys(
        snapshot_row,
        selected_topic_ids=payload.selected_topic_ids,
        manual_topic_labels=payload.manual_topic_labels,
        has_explicit_selection=payload.has_explicit_selection,
    )
    normalized_manual_topic_labels = normalize_selected_topics(payload.manual_topic_labels or [])
    normalization_version = resolve_selected_topic_normalization_version(
        snapshot_row,
        selected_topic_ids=payload.selected_topic_ids,
        has_explicit_selection=payload.has_explicit_selection,
    )
    requested_identity_groups = resolve_selected_topic_identity_groups(
        snapshot_row,
        selected_topic_ids=payload.selected_topic_ids,
        has_explicit_selection=payload.has_explicit_selection,
    )
    selection_fingerprint, legacy_selection_fingerprint = _build_selection_fingerprint_variants(
        snapshot_id=snapshot_id,
        target_service=action_contract["target_service"],
        target_type=action_contract["target_type"],
        selected_topics=semantic_selected_topics,
        action_kind=action_contract["action_kind"],
        generator_version=payload.generator_version,
        normalization_version=normalization_version,
    )
    pending_reservation_created = False

    existing_candidates = _list_generation_link_candidates(
        db,
        snapshot_id=snapshot_id,
        target_service=action_contract["target_service"],
        target_type=action_contract["target_type"],
        selection_fingerprint=selection_fingerprint,
        legacy_selection_fingerprint=legacy_selection_fingerprint,
    )
    pending_existing = next(
        (
            (existing_link, matched_fingerprint)
            for existing_link, matched_fingerprint in existing_candidates
            if is_pending_generation_target_id(existing_link.get("target_id"))
        ),
        None,
    )
    live_existing = next(
        (
            (existing_link, matched_fingerprint)
            for existing_link, matched_fingerprint in existing_candidates
            if not is_pending_generation_target_id(existing_link.get("target_id"))
            and _is_generation_link_target_live(db, existing_link)
        ),
        None,
    )
    stale_existing = [
        (existing_link, matched_fingerprint)
        for existing_link, matched_fingerprint in existing_candidates
        if not is_pending_generation_target_id(existing_link.get("target_id"))
        and not _is_generation_link_target_live(db, existing_link)
    ]
    if pending_existing and not payload.force_regenerate:
        raise HTTPException(status_code=409, detail="Study suggestion action already in progress")
    if live_existing and not payload.force_regenerate:
        existing_link, _matched_existing_fingerprint = live_existing
        raise _EarlyReturn(
            SuggestionActionResponse.model_validate(
                {
                    "disposition": "opened_existing",
                    "snapshot_id": snapshot_id,
                    "selection_fingerprint": selection_fingerprint,
                    "target_service": action_contract["target_service"],
                    "target_type": action_contract["target_type"],
                    "target_id": str(existing_link["target_id"]),
                }
            )
        )

    ancestor_link = _find_live_generation_link_in_refreshed_lineage(
        db,
        snapshot_row,
        target_service=action_contract["target_service"],
        target_type=action_contract["target_type"],
        requested_identity_groups=[] if normalized_manual_topic_labels else requested_identity_groups,
        action_kind=action_contract["action_kind"],
        generator_version=payload.generator_version,
    )
    if ancestor_link and not payload.force_regenerate:
        raise _EarlyReturn(
            SuggestionActionResponse.model_validate(
                {
                    "disposition": "opened_existing",
                    "snapshot_id": snapshot_id,
                    "selection_fingerprint": selection_fingerprint,
                    "target_service": action_contract["target_service"],
                    "target_type": action_contract["target_type"],
                    "target_id": str(ancestor_link["target_id"]),
                }
            )
    )

    if not payload.force_regenerate:
        for _existing_link, matched_existing_fingerprint in stale_existing:
            try:
                db.soft_delete_suggestion_generation_link(
                    snapshot_id=snapshot_id,
                    target_service=action_contract["target_service"],
                    target_type=action_contract["target_type"],
                    selection_fingerprint=matched_existing_fingerprint,
                )
            except Exception as exc:  # pragma: no cover - defensive cleanup path
                logger.warning(
                    "Failed to retire stale study suggestion generation link for snapshot_id={} fingerprint={}: {}",
                    snapshot_id,
                    matched_existing_fingerprint,
                    exc,
                )
        try:
            reserve_generation_link(
                db,
                snapshot_id=snapshot_id,
                target_service=action_contract["target_service"],
                target_type=action_contract["target_type"],
                selection_fingerprint=selection_fingerprint,
            )
            pending_reservation_created = True
        except Exception:
            existing_candidates = _list_generation_link_candidates(
                db,
                snapshot_id=snapshot_id,
                target_service=action_contract["target_service"],
                target_type=action_contract["target_type"],
                selection_fingerprint=selection_fingerprint,
                legacy_selection_fingerprint=legacy_selection_fingerprint,
            )
            pending_existing = next(
                (
                    (existing_link, matched_fingerprint)
                    for existing_link, matched_fingerprint in existing_candidates
                    if is_pending_generation_target_id(existing_link.get("target_id"))
                ),
                None,
            )
            live_existing = next(
                (
                    (existing_link, matched_fingerprint)
                    for existing_link, matched_fingerprint in existing_candidates
                    if not is_pending_generation_target_id(existing_link.get("target_id"))
                    and _is_generation_link_target_live(db, existing_link)
                ),
                None,
            )
            if pending_existing:
                raise HTTPException(status_code=409, detail="Study suggestion action already in progress")
            if live_existing and not payload.force_regenerate:
                existing_link, _matched_existing_fingerprint = live_existing
                raise _EarlyReturn(
                    SuggestionActionResponse.model_validate(
                        {
                            "disposition": "opened_existing",
                            "snapshot_id": snapshot_id,
                            "selection_fingerprint": selection_fingerprint,
                            "target_service": action_contract["target_service"],
                            "target_type": action_contract["target_type"],
                            "target_id": str(existing_link["target_id"]),
                        }
                    )
                )
            raise

    retired_selection_fingerprint = (
        next(
            (
                matched_fingerprint
                for _existing_link, matched_fingerprint in existing_candidates
                if matched_fingerprint != selection_fingerprint
            ),
            None,
        )
        if existing_candidates
        else None
    )
    return (
        snapshot_row,
        action_contract,
        selected_topics,
        selection_fingerprint,
        pending_reservation_created,
        retired_selection_fingerprint,
    )


def _finalize_action(
    db: CharactersRAGDB,
    *,
    snapshot_id: int,
    generated: dict[str, str],
    selection_fingerprint: str,
    pending_reservation_created: bool,
    force_regenerate: bool,
    retired_selection_fingerprint: str | None,
) -> None:
    """Synchronous post-dispatch phase: persist the generation link."""

    if retired_selection_fingerprint:
        db.soft_delete_suggestion_generation_link(
            snapshot_id=snapshot_id,
            target_service=str(generated["target_service"]),
            target_type=str(generated["target_type"]),
            selection_fingerprint=retired_selection_fingerprint,
        )
    if pending_reservation_created:
        finalize_generation_link(
            db,
            snapshot_id=snapshot_id,
            target_service=str(generated["target_service"]),
            target_type=str(generated["target_type"]),
            selection_fingerprint=selection_fingerprint,
            final_target_id=str(generated["target_id"]),
        )
    elif force_regenerate:
        db.replace_suggestion_generation_link(
            snapshot_id=snapshot_id,
            target_service=str(generated["target_service"]),
            target_type=str(generated["target_type"]),
            target_id=str(generated["target_id"]),
            selection_fingerprint=selection_fingerprint,
        )
    else:
        db.create_suggestion_generation_link(
            snapshot_id=snapshot_id,
            target_service=str(generated["target_service"]),
            target_type=str(generated["target_type"]),
            target_id=str(generated["target_id"]),
            selection_fingerprint=selection_fingerprint,
        )


class _EarlyReturn(Exception):
    """Internal signal to return an existing-link response from the sync prepare phase."""

    def __init__(self, response: SuggestionActionResponse) -> None:
        self.response = response


@router.post("/snapshots/{snapshot_id}/actions", response_model=SuggestionActionResponse)
async def trigger_suggestion_action(
    snapshot_id: int,
    payload: SuggestionActionRequest,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    media_db: Any = Depends(get_media_db_for_user),
) -> SuggestionActionResponse:
    try:
        (
            snapshot_row,
            action_contract,
            selected_topics,
            selection_fingerprint,
            pending_reservation_created,
            retired_selection_fingerprint,
        ) = (
            await asyncio.to_thread(_prepare_action, db, snapshot_id, payload)
        )
    except _EarlyReturn as early:
        return early.response

    action_payload = payload.model_dump(mode="json")
    action_payload.update(action_contract)
    action_payload["selected_topics"] = selected_topics
    try:
        generated = await _dispatch_follow_up_action(
            note_db=db,
            snapshot_row=snapshot_row,
            request_body=action_payload,
            media_db=media_db,
        )
        await asyncio.to_thread(
            _finalize_action,
            db,
            snapshot_id=snapshot_id,
            generated=generated,
            selection_fingerprint=selection_fingerprint,
            pending_reservation_created=pending_reservation_created,
            force_regenerate=payload.force_regenerate,
            retired_selection_fingerprint=retired_selection_fingerprint,
        )
    except HTTPException:
        if pending_reservation_created:
            with contextlib.suppress(Exception):
                await asyncio.to_thread(
                    release_generation_link_reservation,
                    db,
                    snapshot_id=snapshot_id,
                    target_service=action_contract["target_service"],
                    target_type=action_contract["target_type"],
                    selection_fingerprint=selection_fingerprint,
                )
        raise
    except Exception:
        if pending_reservation_created:
            with contextlib.suppress(Exception):
                await asyncio.to_thread(
                    release_generation_link_reservation,
                    db,
                    snapshot_id=snapshot_id,
                    target_service=action_contract["target_service"],
                    target_type=action_contract["target_type"],
                    selection_fingerprint=selection_fingerprint,
                )
        raise
    return SuggestionActionResponse.model_validate(
        {
            "disposition": "generated",
            "snapshot_id": snapshot_id,
            "selection_fingerprint": selection_fingerprint,
            "target_service": generated["target_service"],
            "target_type": generated["target_type"],
            "target_id": generated["target_id"],
        }
    )
