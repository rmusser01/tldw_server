"""Canonical manual-link writes, including semantic-edge conversion."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.DB_Management.chacha.note_link_store import NotesLink
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    CharactersRAGDBError,
    ConflictError,
    InputError,
)
from tldw_Server_API.app.core.Sync.v2.notes_link_coordinator import (
    NotesLinkPreflightError,
    resolve_notes_link_dataset_authority,
)

from .graph_cache import GraphCache
from .graph_service import NoteGraphService
from .semantic_projector import SemanticGraphProjector, SemanticProjectionError

ProjectorFactory = Callable[..., SemanticGraphProjector]
AuditEmitter = Callable[..., Awaitable[None]]
CoordinatorResolver = Callable[..., Any]


def _dataset_key(*, user_id: str, dataset_id: str | None) -> str:
    """Resolve the canonical Sync dataset or the owner's legacy namespace."""

    authority = resolve_notes_link_dataset_authority(
        user_id=user_id,
        dataset_id=dataset_id,
    )
    return f"legacy:{user_id}" if authority is None else authority[1].dataset_id


def _has_existing_manual_link(
    db: CharactersRAGDB,
    *,
    source_note_id: str,
    target_note_id: str,
) -> bool:
    """Confirm a live authoritative manual link for an unordered note pair."""

    expected_pair = {source_note_id, target_note_id}
    return any(
        link.type == "manual"
        and not link.deleted
        and {link.source_note_id, link.target_note_id} == expected_pair
        for link in db.notes_link_store.list_for_notes([source_note_id, target_note_id])
    )


async def create_manual_note_link(
    *,
    owner_user_id: str,
    db: CharactersRAGDB,
    source_note_id: str,
    target_note_id: str,
    directed: bool,
    weight: float,
    label: str | None,
    properties: dict[str, Any],
    dataset_id: str | None,
    idempotency_key: str | None,
    semantic_generation_id: str | None,
    graph_cache: GraphCache,
    projector_factory: ProjectorFactory,
    audit_emitter: AuditEmitter,
    coordinator_resolver: CoordinatorResolver,
) -> NotesLink | dict[str, Any]:
    """Validate an optional semantic conversion and write its canonical manual link."""

    dataset_key: str | None = None
    if semantic_generation_id is not None:
        dataset_key = _dataset_key(user_id=owner_user_id, dataset_id=dataset_id)
        graph_service = NoteGraphService(
            user_id=owner_user_id,
            dataset_id=dataset_key,
            db=db,
            cache=graph_cache,
        )
        projector = projector_factory(
            owner_user_id=owner_user_id,
            dataset_id=dataset_key,
            db=db,
            graph_service=graph_service,
        )
        await projector.validate_conversion(
            source_note_id=source_note_id,
            target_note_id=target_note_id,
            generation_id=semantic_generation_id,
        )
        directed = False
        weight = 1.0

    try:
        coordinator = coordinator_resolver(
            user_id=owner_user_id,
            note_db=db,
            dataset_id=dataset_id,
        )
        if coordinator is not None:
            result: NotesLink | dict[str, Any] = coordinator.create(
                source_note_id=source_note_id,
                target_note_id=target_note_id,
                directed=directed,
                weight=weight,
                label=label,
                properties=properties,
                idempotency_key=idempotency_key,
            )
        else:
            metadata = dict(properties)
            if label is not None:
                metadata["label"] = label
            result = db.create_manual_note_edge(
                user_id=owner_user_id,
                from_note_id=source_note_id,
                to_note_id=target_note_id,
                directed=directed,
                weight=weight,
                metadata=metadata,
                created_by=f"user:{owner_user_id}",
            )
            stored_edge_id = str(result.get("edge_id") or "")
            if stored_edge_id and hasattr(db, "notes_link_store"):
                stored = db.notes_link_store.get(stored_edge_id)
                if stored is not None:
                    result = stored
    except (ConflictError, NotesLinkPreflightError) as exc:
        if semantic_generation_id is None:
            raise
        try:
            manual_link_exists = _has_existing_manual_link(
                db,
                source_note_id=source_note_id,
                target_note_id=target_note_id,
            )
        except (CharactersRAGDBError, InputError):
            manual_link_exists = False
        if manual_link_exists:
            raise SemanticProjectionError(
                "notes_semantic_conversion_manual_link_exists"
            ) from exc
        raise

    if semantic_generation_id is not None and dataset_key is not None:
        try:
            await audit_emitter(
                actor_user_id=owner_user_id,
                source_note_id=source_note_id,
                target_note_id=target_note_id,
                generation_id=semantic_generation_id,
                result="created",
                dataset_id=dataset_key,
            )
        except Exception:  # noqa: BLE001 - the link is already authoritative.
            logger.bind(
                operation="notes_semantic.manual_conversion",
                actor_user_id=owner_user_id,
                dataset_id=dataset_key,
                source_note_id=source_note_id,
                target_note_id=target_note_id,
                generation_id=semantic_generation_id,
            ).opt(exception=True).warning(
                "Notes semantic conversion audit emission failed"
            )

    return result


__all__ = ["create_manual_note_link"]
