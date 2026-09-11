"""OpenWebUI folder mirroring helpers for Chatbooks imports."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from tldw_Server_API.app.core.DB_Management.chacha.organization_sync_store import (
    NotesOrganizationSyncStore,
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDBError,
    ConflictError,
    InputError,
)
from tldw_Server_API.app.core.Notes.organization_capture import (
    active_coordinator,
    capture_plan,
    stable_note_id,
)
from tldw_Server_API.app.core.Sync.v2.notes_organization_coordinator import (
    NotesOrganizationCoordinator,
    PlannedNotesMutation,
)
from tldw_Server_API.app.core.Sync.v2.server_origin_batch import ServerOriginMutationStep

if TYPE_CHECKING:
    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB


OPENWEBUI_COLLECTION_ROOT = "OpenWebUI"
UNFILED_COLLECTION = "Unfiled"

_CONTROL_CHARS_RE = re.compile(r"[\x00-\x1f\x7f]+")
_WHITESPACE_RE = re.compile(r"\s+")
_MIRROR_RESPONSE_KEY = "openwebui_folder_mirror_response"


@dataclass(frozen=True)
class OpenWebUIFolderMirrorResult:
    """Result of mirroring one OpenWebUI folder path into keyword collections."""

    final_collection_id: int | None
    collection_ids: list[int] = field(default_factory=list)
    keyword_id: int | None = None
    created_collections: int = 0
    reused_collections: int = 0
    collection_keyword_linked: bool = False
    conversation_keyword_linked: bool = False
    warnings: list[str] = field(default_factory=list)


def sanitize_openwebui_folder_segment(value: Any) -> str:
    """Return a safe, readable keyword-collection segment for OpenWebUI names."""
    text = "" if value is None else str(value)
    text = _CONTROL_CHARS_RE.sub(" ", text)
    text = text.replace("/", "_").replace("\\", "_")
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text or "Untitled"


def build_openwebui_namespace_segments(source_user_label: str, source_user_id: str) -> list[str]:
    """Return the root namespace used for one selected OpenWebUI source user."""
    user_label = sanitize_openwebui_folder_segment(source_user_label or "OpenWebUI user")
    user_id = sanitize_openwebui_folder_segment(source_user_id or "unknown")
    return [OPENWEBUI_COLLECTION_ROOT, f"{user_label} ({user_id})"]


def mirror_openwebui_folder_for_conversation(
    db: CharactersRAGDB,
    *,
    conversation_id: str,
    namespace_segments: list[str],
    source_path_segments: list[str],
    source_folder_id: str | None = None,
    metadata: dict[str, Any] | None = None,
    owner_user_id: int | str,
) -> OpenWebUIFolderMirrorResult:
    """Mirror one OpenWebUI source path and link the imported conversation to it."""
    warnings: list[str] = []
    path_segments = _sanitize_path_segments(
        list(namespace_segments or []),
        context="namespace",
        warnings=warnings,
    )
    source_segments = _sanitize_path_segments(
        list(source_path_segments or [UNFILED_COLLECTION]),
        context="source path",
        warnings=warnings,
    )
    if not source_segments:
        source_segments = [UNFILED_COLLECTION]

    full_path = path_segments + source_segments
    coordinator = active_coordinator(db, user_id=owner_user_id)
    if coordinator is not None:
        return _mirror_openwebui_folder_with_sync(
            db,
            coordinator=coordinator,
            conversation_id=conversation_id,
            full_path=full_path,
            source_folder_id=source_folder_id,
            metadata=metadata or {},
            warnings=warnings,
        )

    collection_ids: list[int] = []
    created_collections = 0
    reused_collections = 0
    parent_id: int | None = None
    for index, segment in enumerate(full_path):
        collection_id, created, disambiguated_name = _ensure_collection_segment(
            db,
            segment,
            parent_id=parent_id,
            path_segments=full_path[: index + 1],
            source_folder_id=source_folder_id,
            metadata=metadata or {},
        )
        if disambiguated_name is not None:
            warnings.append(
                "OpenWebUI folder segment "
                f"'{segment}' was disambiguated as '{disambiguated_name}' due to a global collection name collision."
            )
        collection_ids.append(collection_id)
        parent_id = collection_id
        if created:
            created_collections += 1
        else:
            reused_collections += 1

    final_collection_id = collection_ids[-1] if collection_ids else None
    if final_collection_id is None:
        return OpenWebUIFolderMirrorResult(
            final_collection_id=None,
            warnings=warnings,
        )

    keyword_text = _keyword_text_for_path(collection_ids, full_path, source_folder_id, metadata or {})
    keyword_id, _keyword_created = _get_or_create_keyword_id(db, keyword_text)
    collection_keyword_linked = db.link_collection_to_keyword(final_collection_id, keyword_id)
    conversation_keyword_linked = db.link_conversation_to_keyword(conversation_id, keyword_id)

    return OpenWebUIFolderMirrorResult(
        final_collection_id=final_collection_id,
        collection_ids=collection_ids,
        keyword_id=keyword_id,
        created_collections=created_collections,
        reused_collections=reused_collections,
        collection_keyword_linked=collection_keyword_linked,
        conversation_keyword_linked=conversation_keyword_linked,
        warnings=warnings,
    )


def _mirror_openwebui_folder_with_sync(
    db: CharactersRAGDB,
    *,
    coordinator: NotesOrganizationCoordinator,
    conversation_id: str,
    full_path: list[str],
    source_folder_id: str | None,
    metadata: dict[str, Any],
    warnings: list[str],
) -> OpenWebUIFolderMirrorResult:
    """Plan the complete folder mirror before one durable append."""

    fingerprint = coordinator.request_fingerprint(
        "openwebui.folder.mirror",
        {
            "conversation_id": conversation_id,
            "full_path": full_path,
            "source_folder_id": source_folder_id,
            "source_user_id": metadata.get("source_user_id"),
        },
    )
    key = fingerprint
    replay = coordinator.replay_request_plan(
        source="openwebui-import",
        idempotency_key=key,
        request_fingerprint=fingerprint,
        result_domain=None,
    )
    if replay is not None:
        capture_plan(coordinator, replay, source="openwebui-import", key=key)
        return _load_sync_mirror_result(db, replay.steps, warnings=warnings)

    steps: list[ServerOriginMutationStep] = []
    collection_sync_ids: list[str] = []
    created_collections = 0
    reused_collections = 0
    parent_id: int | None = None
    parent_sync_id: str | None = None
    parent_is_planned = False

    for index, segment in enumerate(full_path):
        expected_parent_sync_id = parent_sync_id
        name = segment
        existing = db.get_keyword_collection_by_name(name)
        same_parent = (
            not parent_is_planned
            and existing is not None
            and _same_parent(existing.get("parent_id"), parent_id)
        )
        if not same_parent and existing is not None:
            base_hash = _short_hash(
                {
                    "segment": segment,
                    "parent_sync_id": parent_sync_id,
                    "path": full_path[: index + 1],
                    "source_folder_id": source_folder_id,
                    "source_user_id": metadata.get("source_user_id"),
                }
            )
            for suffix in (base_hash, *[f"{base_hash}-{counter}" for counter in range(1, 100)]):
                candidate = f"{segment} (owui-{suffix})"
                candidate_existing = db.get_keyword_collection_by_name(candidate)
                if (
                    not parent_is_planned
                    and candidate_existing is not None
                    and _same_parent(candidate_existing.get("parent_id"), parent_id)
                ):
                    existing = candidate_existing
                    same_parent = True
                    name = candidate
                    break
                if candidate_existing is None:
                    existing = None
                    name = candidate
                    break
            else:
                raise ConflictError(
                    f"Unable to disambiguate OpenWebUI folder collection '{segment}'.",
                    entity="keyword_collections",
                    entity_id=segment,
                )
            warnings.append(
                "OpenWebUI folder segment "
                f"'{segment}' was disambiguated as '{name}' due to a global collection name collision."
            )

        if same_parent and existing is not None:
            parent_id = int(existing["id"])
            parent_sync_id = str(existing["sync_id"])
            parent_is_planned = False
            reused_collections += 1
            steps.append(
                ServerOriginMutationStep(
                    domain="notes.keyword_collection",
                    operation="upsert",
                    object_id=parent_sync_id,
                    payload={
                        "name": str(existing["name"]),
                        "parent_sync_id": expected_parent_sync_id,
                    },
                    parent_id=expected_parent_sync_id,
                )
            )
        else:
            segment_key = _short_hash(
                {"request": key, "path": full_path[: index + 1], "name": name},
                length=32,
            )
            sync_id = stable_note_id("openwebui-collection", segment_key)
            steps.append(
                ServerOriginMutationStep(
                    domain="notes.keyword_collection",
                    operation="upsert",
                    object_id=sync_id,
                    payload={"name": name, "parent_sync_id": parent_sync_id},
                    parent_id=parent_sync_id,
                )
            )
            parent_id = None
            parent_sync_id = sync_id
            parent_is_planned = True
            created_collections += 1
        collection_sync_ids.append(parent_sync_id)

    if parent_sync_id is None:
        return OpenWebUIFolderMirrorResult(final_collection_id=None, warnings=warnings)

    keyword_text = _keyword_text_for_path([], full_path, source_folder_id, metadata)
    keyword = db.get_keyword_by_text(keyword_text)
    if keyword is None:
        keyword_sync_id = stable_note_id("openwebui-keyword", key)
        steps.append(
            ServerOriginMutationStep(
                domain="notes.keyword",
                operation="upsert",
                object_id=keyword_sync_id,
                payload={"keyword": keyword_text},
            )
        )
    else:
        keyword_sync_id = str(keyword["sync_id"])
    steps.extend(
        coordinator.plan_relationship(
            "notes.keyword_collection_link",
            {
                "collection_sync_id": parent_sync_id,
                "keyword_sync_id": keyword_sync_id,
            },
            True,
        ).steps
    )
    steps.extend(
        coordinator.plan_relationship(
            "notes.keyword_link",
            {
                "subject_type": "conversation",
                "subject_id": conversation_id,
                "keyword_sync_id": keyword_sync_id,
            },
            True,
        ).steps
    )
    response_metadata = {
        "created_collections": created_collections,
        "reused_collections": reused_collections,
        "collection_keyword_linked": True,
        "conversation_keyword_linked": True,
        "warnings": list(warnings),
    }
    steps[0] = replace(
        steps[0],
        routing_metadata={
            **dict(steps[0].routing_metadata),
            _MIRROR_RESPONSE_KEY: response_metadata,
        },
    )
    plan = coordinator.bind_request(
        PlannedNotesMutation(steps=tuple(steps), load_result=lambda: None),
        fingerprint,
    )
    capture_plan(coordinator, plan, source="openwebui-import", key=key)
    return _load_sync_mirror_result(db, plan.steps, warnings=warnings)


def _load_sync_mirror_result(
    db: CharactersRAGDB,
    steps: tuple[ServerOriginMutationStep, ...],
    *,
    warnings: list[str],
) -> OpenWebUIFolderMirrorResult:
    organization_store = NotesOrganizationSyncStore(db)
    collection_rows = []
    for step in steps:
        if step.domain != "notes.keyword_collection":
            continue
        resource = organization_store.get_resource("notes.keyword_collection", step.object_id)
        collection_rows.append(
            db.get_keyword_collection_by_id(resource.local_id) if resource is not None else None
        )
    if not collection_rows or any(row is None for row in collection_rows):
        raise CharactersRAGDBError("OpenWebUI collection projection is incomplete.")
    keyword_step = next((step for step in steps if step.domain == "notes.keyword"), None)
    keyword_resource = (
        organization_store.get_resource("notes.keyword", keyword_step.object_id)
        if keyword_step is not None
        else None
    )
    keyword = (
        db.get_keyword_by_id(keyword_resource.local_id)
        if keyword_resource is not None
        else None
    )
    if keyword is None:
        link = next(step for step in steps if step.domain == "notes.keyword_collection_link")
        keyword = next(
            (
                row
                for row in db.get_keywords_for_collection(int(collection_rows[-1]["id"]))
                if str(row["sync_id"]) == str(link.payload["keyword_sync_id"])
            ),
            None,
        )
    if keyword is None:
        raise CharactersRAGDBError("OpenWebUI keyword projection is incomplete.")
    collection_ids = [int(row["id"]) for row in collection_rows if row is not None]
    response_metadata = next(
        (
            step.routing_metadata.get(_MIRROR_RESPONSE_KEY)
            for step in steps
            if isinstance(step.routing_metadata.get(_MIRROR_RESPONSE_KEY), dict)
        ),
        {},
    )
    return OpenWebUIFolderMirrorResult(
        final_collection_id=collection_ids[-1],
        collection_ids=collection_ids,
        keyword_id=int(keyword["id"]),
        created_collections=int(response_metadata.get("created_collections", len(collection_ids))),
        reused_collections=int(response_metadata.get("reused_collections", 0)),
        collection_keyword_linked=bool(response_metadata.get("collection_keyword_linked", True)),
        conversation_keyword_linked=bool(response_metadata.get("conversation_keyword_linked", True)),
        warnings=list(response_metadata.get("warnings", warnings)),
    )


def _sanitize_path_segments(
    segments: list[Any],
    *,
    context: str,
    warnings: list[str],
) -> list[str]:
    sanitized: list[str] = []
    for raw_segment in segments:
        raw_text = "" if raw_segment is None else str(raw_segment)
        segment = sanitize_openwebui_folder_segment(raw_text)
        if segment != raw_text.strip():
            warnings.append(
                f"OpenWebUI {context} segment was sanitized from {raw_text!r} to {segment!r}."
            )
        sanitized.append(segment)
    return sanitized


def _ensure_collection_segment(
    db: CharactersRAGDB,
    segment: str,
    *,
    parent_id: int | None,
    path_segments: list[str],
    source_folder_id: str | None,
    metadata: dict[str, Any],
) -> tuple[int, bool, str | None]:
    existing = db.get_keyword_collection_by_name(segment)
    if existing is not None and _same_parent(existing.get("parent_id"), parent_id):
        return int(existing["id"]), False, None
    if existing is None:
        return _create_collection(db, segment, parent_id), True, None

    base_hash = _short_hash(
        {
            "segment": segment,
            "parent_id": parent_id,
            "path": path_segments,
            "source_folder_id": source_folder_id,
            "source_user_id": metadata.get("source_user_id"),
        }
    )
    for suffix in (base_hash, *[f"{base_hash}-{counter}" for counter in range(1, 100)]):
        candidate = f"{segment} (owui-{suffix})"
        candidate_existing = db.get_keyword_collection_by_name(candidate)
        if candidate_existing is not None and _same_parent(candidate_existing.get("parent_id"), parent_id):
            return int(candidate_existing["id"]), False, candidate
        if candidate_existing is None:
            return _create_collection(db, candidate, parent_id), True, candidate

    raise ConflictError(
        f"Unable to disambiguate OpenWebUI folder collection '{segment}'.",
        entity="keyword_collections",
        entity_id=segment,
    )


def _create_collection(
    db: CharactersRAGDB,
    name: str,
    parent_id: int | None,
) -> int:
    try:
        collection_id = db.add_keyword_collection(name, parent_id=parent_id)
    except ConflictError:
        existing = db.get_keyword_collection_by_name(name)
        if existing is not None and _same_parent(existing.get("parent_id"), parent_id):
            return int(existing["id"])
        raise
    if collection_id is None:
        raise CharactersRAGDBError(f"Keyword collection '{name}' could not be created.")
    return int(collection_id)


def _get_or_create_keyword_id(db: CharactersRAGDB, keyword_text: str) -> tuple[int, bool]:
    existing = db.get_keyword_by_text(keyword_text)
    if existing is not None:
        return int(existing["id"]), False
    try:
        keyword_id = db.add_keyword(keyword_text)
    except ConflictError:
        existing = db.get_keyword_by_text(keyword_text)
        if existing is not None:
            return int(existing["id"]), False
        raise
    except InputError:
        raise
    if keyword_id is None:
        raise CharactersRAGDBError(f"Keyword '{keyword_text}' could not be created.")
    return int(keyword_id), True


def _keyword_text_for_path(
    collection_ids: list[int],
    path_segments: list[str],
    source_folder_id: str | None,
    metadata: dict[str, Any],
) -> str:
    path_label = " / ".join(path_segments)
    stable_hash = _short_hash(
        {
            "collection_ids": collection_ids,
            "path": path_segments,
            "source_folder_id": source_folder_id,
            "source_user_id": metadata.get("source_user_id"),
        },
        length=12,
    )
    return f"OpenWebUI folder: {path_label} [{stable_hash}]"


def _same_parent(existing_parent_id: Any, expected_parent_id: int | None) -> bool:
    if existing_parent_id in (None, ""):
        return expected_parent_id is None
    try:
        return int(existing_parent_id) == expected_parent_id
    except (TypeError, ValueError):
        return False


def _short_hash(value: Any, *, length: int = 8) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]
