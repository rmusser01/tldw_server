"""Shared active-Sync capture seam for non-Notes mutation surfaces."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Literal
from uuid import UUID

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    ConflictError,
    InputError,
)
from tldw_Server_API.app.core.Sync.v2 import server_origin
from tldw_Server_API.app.core.Sync.v2.errors import SyncStoreError
from tldw_Server_API.app.core.Sync.v2.notes_organization_coordinator import (
    NotesOrganizationCoordinator,
    PlannedNotesMutation,
)
from tldw_Server_API.app.core.Sync.v2.server_origin import server_origin_stable_key
from tldw_Server_API.app.core.Sync.v2.server_origin_batch import (
    ServerOriginMutationStep,
)

_MAX_SERVER_ROUTING_METADATA_BYTES = 1_024


def active_coordinator(
    db: CharactersRAGDB,
    *,
    user_id: int | str,
) -> NotesOrganizationCoordinator | None:
    """Return a ready owner-bound coordinator, or ``None`` when Sync is inactive."""

    owner = str(user_id or "").strip()
    if not owner:
        raise SyncStoreError("An authenticated owner is required for Notes mutation capture")
    service = server_origin.get_active_server_origin_sync_service_for_user(owner)
    if service is None:
        return None
    coordinator = NotesOrganizationCoordinator(service=service, note_db=db, user_id=owner)
    coordinator.require_ready()
    return coordinator


def stable_note_id(source: str, key: str) -> str:
    """Derive an opaque stable note id for a capture request."""

    digest = hashlib.sha256(f"{source}:notes.note:{key}".encode()).hexdigest()
    return str(UUID(digest[:32], version=4))


def compound_note_id(request_key: str) -> str:
    """Derive the stable note identity for one compound Notes API request."""

    digest = hashlib.sha256(f"notes.note:{request_key}".encode()).hexdigest()
    return str(UUID(digest[:32], version=4))


def compound_note_request_fingerprint(
    coordinator: NotesOrganizationCoordinator,
    *,
    operation: str,
    note_id: str,
    note_fields: Mapping[str, object],
    keywords: Sequence[str] | None,
    folder_paths: Sequence[str] | None,
    expected_version: int | None = None,
) -> str:
    """Hash the immutable inputs for one compound Notes API request."""

    return coordinator.request_fingerprint(
        operation,
        {
            "note_id": note_id,
            "note_fields": dict(note_fields),
            "keywords": list(keywords) if keywords is not None else None,
            "folder_paths": list(folder_paths) if folder_paths is not None else None,
            "expected_version": expected_version,
        },
    )


def plan_compound_note(
    coordinator: NotesOrganizationCoordinator,
    *,
    note_id: str,
    note_payload: Mapping[str, object],
    keywords: Sequence[str] | None,
    folder_paths: Sequence[str] | None,
    request_key: str,
    request_fingerprint: str,
    response_status: int | None = None,
) -> PlannedNotesMutation:
    """Build one request-bound note-and-organization mutation plan."""

    plan = coordinator.plan_note_with_organization(
        note_step=ServerOriginMutationStep(
            domain="notes.note",
            operation="upsert",
            object_id=note_id,
            payload=dict(note_payload),
            stable_key=server_origin_stable_key(
                source="notes-api",
                domain="notes.note",
                operation="upsert",
                idempotency_key=request_key,
            ),
        ),
        keywords=keywords,
        folder_paths=folder_paths,
    )
    plan = coordinator.bind_request(plan, request_fingerprint)
    if response_status is not None:
        plan = coordinator.bind_response_status(plan, response_status)
    return plan


def capture_plan(
    coordinator: NotesOrganizationCoordinator,
    plan: PlannedNotesMutation,
    *,
    source: str,
    key: str,
) -> object:
    """Append a complete plan before loading its materialized product result."""

    if plan.steps:
        result = coordinator.capture(steps=plan.steps, source=source, idempotency_key=key)
        if not result.fully_applied:
            raise SyncStoreError("Notes organization projection is incomplete")
    return plan.load_result()


def capture_note_upsert(
    coordinator: NotesOrganizationCoordinator,
    *,
    note_id: str,
    title: str,
    content: str,
    source: str,
    keywords: Sequence[str] | None = None,
    folder_paths: Sequence[str] | None = None,
    conversation_id: object = None,
    message_id: object = None,
    expected_version: int | None = None,
    key: str | None = None,
    request_fingerprint: str | None = None,
    server_routing_metadata: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Capture a note and its requested organization state as one batch."""

    fields: dict[str, object] = {
        "note_id": note_id,
        "title": title,
        "content": content,
        "conversation_id": conversation_id,
        "message_id": message_id,
        "keywords": list(keywords) if keywords is not None else None,
        "folder_paths": list(folder_paths) if folder_paths is not None else None,
        "expected_version": expected_version,
    }
    fingerprint = request_fingerprint or coordinator.request_fingerprint("note.upsert", fields)
    capture_key = key or fingerprint
    replay = coordinator.replay_request_plan(
        source=source,
        idempotency_key=capture_key,
        request_fingerprint=fingerprint,
        result_domain="notes.note",
    )
    if replay is not None:
        result = capture_plan(coordinator, replay, source=source, key=capture_key)
        if not isinstance(result, dict):
            raise SyncStoreError("Note projection did not return a note")
        return result

    existing = coordinator.note_db.get_note_by_id(note_id)
    if expected_version is not None:
        if existing is None or int(existing.get("version") or 0) != int(expected_version):
            raise ConflictError(
                f"Note ID {note_id} update failed: version mismatch.",
                entity="notes",
                entity_id=note_id,
            )
    routing_metadata = dict(server_routing_metadata or {})
    if any(not isinstance(key, str) for key in routing_metadata):
        raise InputError("Server routing metadata keys must be strings")
    try:
        encoded_routing_metadata = json.dumps(
            routing_metadata,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise InputError("Server routing metadata must be JSON serializable") from exc
    if len(encoded_routing_metadata) > _MAX_SERVER_ROUTING_METADATA_BYTES:
        raise InputError("Server routing metadata exceeds the 1024-byte limit")
    plan = coordinator.plan_note_with_organization(
        note_step=ServerOriginMutationStep(
            domain="notes.note",
            operation="upsert",
            object_id=note_id,
            payload={
                "title": title,
                "content": content,
                "conversation_id": conversation_id,
                "message_id": message_id,
            },
            routing_metadata=routing_metadata,
            stable_key=capture_key,
        ),
        keywords=keywords,
        folder_paths=folder_paths,
    )
    plan = coordinator.bind_request(plan, fingerprint)
    result = capture_plan(coordinator, plan, source=source, key=capture_key)
    if not isinstance(result, dict):
        raise SyncStoreError("Note projection did not return a note")
    return result


def replace_keywords(
    coordinator: NotesOrganizationCoordinator,
    *,
    subject_type: Literal["note", "conversation"],
    subject_id: str,
    keywords: Sequence[str],
    source: str,
    key: str | None = None,
) -> None:
    """Capture replacement of a note or conversation keyword set."""

    desired: list[tuple[str, str]] = []
    seen: set[str] = set()
    for value in keywords:
        normalized = str(value or "").strip()
        folded = normalized.casefold()
        if normalized and folded not in seen:
            seen.add(folded)
            desired.append((folded, normalized))
    fields: dict[str, object] = {
        "subject_type": subject_type,
        "subject_id": subject_id,
        "keywords": [value for _, value in desired],
    }
    fingerprint = coordinator.request_fingerprint("keywords.replace", fields)
    capture_key = key or fingerprint
    replay = coordinator.replay_request_plan(
        source=source,
        idempotency_key=capture_key,
        request_fingerprint=fingerprint,
        result_domain=None,
    )
    if replay is not None:
        capture_plan(coordinator, replay, source=source, key=capture_key)
        return

    current_rows = (
        coordinator.note_db.get_keywords_for_note(subject_id)
        if subject_type == "note"
        else coordinator.note_db.get_keywords_for_conversation(subject_id)
    )
    current = {str(row["keyword"]).casefold(): str(row["sync_id"]) for row in current_rows}
    desired_ids: dict[str, str] = {}
    steps: list[ServerOriginMutationStep] = []
    for folded, value in desired:
        row = coordinator.note_db.get_keyword_by_text(value)
        if row is None:
            keyword_plan = coordinator.plan_keyword_create(
                value,
                idempotency_key=f"{capture_key}:keyword:{folded}",
            )
            steps.extend(keyword_plan.steps)
            desired_ids[folded] = keyword_plan.steps[0].object_id
        else:
            desired_ids[folded] = str(row["sync_id"])
    for folded in sorted(set(current) - set(desired_ids)):
        steps.extend(
            coordinator.plan_relationship(
                "notes.keyword_link",
                {
                    "subject_type": subject_type,
                    "subject_id": subject_id,
                    "keyword_sync_id": current[folded],
                },
                False,
            ).steps
        )
    for folded, _ in desired:
        if folded in current:
            continue
        steps.extend(
            coordinator.plan_relationship(
                "notes.keyword_link",
                {
                    "subject_type": subject_type,
                    "subject_id": subject_id,
                    "keyword_sync_id": desired_ids[folded],
                },
                True,
            ).steps
        )
    plan = coordinator.bind_request(
        PlannedNotesMutation(steps=tuple(steps), load_result=lambda: None),
        fingerprint,
    )
    capture_plan(coordinator, plan, source=source, key=capture_key)


def capture_note_tombstone(
    coordinator: NotesOrganizationCoordinator,
    *,
    note_id: str,
    expected_version: int,
    source: str,
    key: str | None = None,
) -> None:
    """Capture a version-checked note tombstone."""

    fields = {"note_id": note_id, "expected_version": expected_version}
    fingerprint = coordinator.request_fingerprint("note.tombstone", fields)
    capture_key = key or fingerprint
    replay = coordinator.replay_request_plan(
        source=source,
        idempotency_key=capture_key,
        request_fingerprint=fingerprint,
        result_domain=None,
    )
    if replay is not None:
        capture_plan(coordinator, replay, source=source, key=capture_key)
        return
    note = coordinator.note_db.get_note_by_id(note_id)
    if note is None or int(note.get("version") or 0) != int(expected_version):
        raise ConflictError(
            f"Note ID {note_id} delete failed: version mismatch.",
            entity="notes",
            entity_id=note_id,
        )
    plan = coordinator.bind_request(
        PlannedNotesMutation(
            steps=(
                ServerOriginMutationStep(
                    domain="notes.note",
                    operation="tombstone",
                    object_id=note_id,
                    payload={},
                    stable_key=capture_key,
                ),
            ),
            load_result=lambda: None,
        ),
        fingerprint,
    )
    capture_plan(coordinator, plan, source=source, key=capture_key)
