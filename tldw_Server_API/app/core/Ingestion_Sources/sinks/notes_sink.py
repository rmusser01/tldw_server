from __future__ import annotations

import hashlib
from pathlib import PurePosixPath
from typing import Any
from uuid import UUID

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import ConflictError
from tldw_Server_API.app.core.Notes.organization_capture import capture_note_tombstone
from tldw_Server_API.app.core.Sync.v2.errors import SyncStoreError
from tldw_Server_API.app.core.Sync.v2.notes_organization_coordinator import (
    NotesOrganizationCoordinator,
)
from tldw_Server_API.app.core.Sync.v2.server_origin import (
    get_active_server_origin_sync_service_for_user,
)
from tldw_Server_API.app.core.Sync.v2.server_origin_batch import (
    ServerOriginMutationStep,
)

_INGESTION_EXPECTED_VERSION_KEY = "notes_ingestion_expected_product_version"


def _title_from_text(relative_path: str, text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            title = stripped.lstrip("#").strip()
            if title:
                return title
        return stripped

    fallback = PurePosixPath(relative_path).stem
    return fallback or "synced-note"


def _expected_version(binding: dict[str, Any]) -> int:
    raw = (
        binding.get("current_version")
        or binding.get("version")
        or binding.get("expected_version")
        or 1
    )
    return int(raw)


def _folder_paths_from_relative_path(relative_path: str) -> list[str]:
    """Return folder ancestors from shallowest to deepest for a repo-relative file path.

    Example:
        ``docs/api/alpha.md`` -> ``["docs", "docs/api"]``

    The helper normalizes separators to POSIX form, trims leading/trailing separators,
    and excludes the file name itself from the returned folder path list.
    """
    normalized = str(relative_path or "").strip().replace("\\", "/").strip("/")
    if not normalized:
        return []
    path = PurePosixPath(normalized)
    folder_paths = [
        str(parent)
        for parent in reversed(path.parents)
        if str(parent) not in {"", "."}
    ]
    return folder_paths


def _active_organization_coordinator(notes_db) -> NotesOrganizationCoordinator | None:
    user_id = str(getattr(notes_db, "client_id", "") or "").strip()
    if not user_id:
        return None
    service = get_active_server_origin_sync_service_for_user(user_id)
    if service is None:
        return None
    return NotesOrganizationCoordinator(
        service=service,
        note_db=notes_db,
        user_id=user_id,
    )


def _source_folder_request_key(*parts: object) -> str:
    encoded = ":".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _capture_source_folder_steps(
    coordinator: NotesOrganizationCoordinator,
    *,
    steps,
    request_key: str,
) -> None:
    if not steps:
        return
    result = coordinator.capture(
        steps=steps,
        source="notes-ingestion",
        idempotency_key=request_key,
    )
    if not result.fully_applied:
        raise SyncStoreError("Notes ingestion folder projection is incomplete")


def _capture_ingestion_note(
    coordinator: NotesOrganizationCoordinator,
    *,
    note_id: str,
    title: str,
    content: str,
    request_key: str,
    expected_version: int | None = None,
) -> None:
    request_fingerprint = coordinator.request_fingerprint(
        "ingestion.note.upsert",
        {
            "note_id": note_id,
            "title": title,
            "content": content,
            "expected_version": expected_version,
        },
    )
    replay = coordinator.replay_request_plan(
        source="notes-ingestion",
        idempotency_key=request_key,
        request_fingerprint=request_fingerprint,
        result_domain="notes.note",
    )
    if replay is not None:
        _capture_source_folder_steps(
            coordinator,
            steps=replay.steps,
            request_key=request_key,
        )
        return
    existing = coordinator.note_db.get_note_by_id(note_id)
    if expected_version is not None and existing is None:
        raise ConflictError(
            f"Note ID {note_id} update failed: note is missing.",
            entity="notes",
            entity_id=note_id,
        )
    existing = existing or {}
    current_version = existing.get("version")
    if (
        expected_version is not None
        and current_version is not None
        and int(current_version) != int(expected_version)
    ):
        raise ConflictError(
            (
                f"Note ID {note_id} update failed: version mismatch "
                f"(db has {current_version}, client expected {expected_version})."
            ),
            entity="notes",
            entity_id=note_id,
        )
    plan = coordinator.plan_note_with_organization(
        note_step=ServerOriginMutationStep(
            domain="notes.note",
            operation="upsert",
            object_id=note_id,
            payload={
                "title": title,
                "content": content,
                "conversation_id": existing.get("conversation_id"),
                "message_id": existing.get("message_id"),
            },
            routing_metadata={
                _INGESTION_EXPECTED_VERSION_KEY: (
                    int(expected_version) if expected_version is not None else 0
                )
            },
            stable_key=request_key,
        ),
        keywords=None,
        folder_paths=None,
    )
    plan = coordinator.bind_request(plan, request_fingerprint)
    _capture_source_folder_steps(
        coordinator,
        steps=plan.steps,
        request_key=request_key,
    )


def _sync_source_folders_with_coordinator(
    coordinator: NotesOrganizationCoordinator,
    *,
    note_id: str,
    source_id: int,
    folder_paths: list[str],
) -> None:
    coordinator.require_ready()
    for path in folder_paths:
        if coordinator.note_db.get_note_folder_by_path(path) is not None:
            continue
        plan = coordinator.plan_folder_path(
            path,
            idempotency_key=_source_folder_request_key("folder", source_id, path.casefold()),
        )
        prefixes = [
            "/".join(path.split("/")[: index + 1])
            for index in range(len(path.split("/")))
        ]
        missing_steps = tuple(
            step
            for prefix, step in zip(prefixes, plan.steps)
            if coordinator.note_db.get_note_folder_by_path(prefix) is None
        )
        _capture_source_folder_steps(
            coordinator,
            steps=missing_steps,
            request_key=_source_folder_request_key("folder-plan", source_id, path.casefold()),
        )

    desired_rows = [
        coordinator.note_db.get_note_folder_by_path(path) for path in folder_paths
    ]
    if any(row is None for row in desired_rows):
        raise SyncStoreError("Notes ingestion folder projection is incomplete")
    desired_folder_ids = {int(row["id"]) for row in desired_rows if row is not None}
    with coordinator.note_db.transaction() as conn:
        current_rows = conn.execute(
            "SELECT folder_id FROM note_folder_source_memberships "
            "WHERE note_id = ? AND source_id = ?",
            (note_id, source_id),
        ).fetchall()
    current_folder_ids = {int(row["folder_id"]) for row in current_rows}
    note = coordinator.note_db.get_note_by_id(note_id)
    if note is None:
        raise SyncStoreError("Notes ingestion note projection is missing")
    note_version = int(note["version"])

    changes = [
        (folder_id, False)
        for folder_id in sorted(current_folder_ids - desired_folder_ids)
    ]
    changes.extend(
        (folder_id, True)
        for folder_id in sorted(desired_folder_ids - current_folder_ids)
    )
    for folder_id, present in changes:
        request_key = _source_folder_request_key(
            "source-folder",
            source_id,
            note_id,
            folder_id,
            present,
            note_version,
        )
        plan = coordinator.plan_source_folder_change(
            note_id=note_id,
            source_id=source_id,
            folder_id=folder_id,
            present=present,
            idempotency_key=request_key,
        )
        if plan.steps:
            _capture_source_folder_steps(
                coordinator,
                steps=plan.steps,
                request_key=request_key,
            )
        else:
            if plan.source_transition is None:
                raise SyncStoreError("Notes ingestion source transition is missing")
            coordinator.apply_source_folder_provenance_only(
                note_id=note_id,
                source_id=source_id,
                folder_id=folder_id,
                present=present,
                transition=plan.source_transition,
            )


def apply_notes_change(
    notes_db,
    *,
    binding: dict[str, Any] | None,
    change: dict[str, Any],
    policy: str,
) -> dict[str, Any]:
    sync_status = None if not binding else binding.get("sync_status")
    if binding and sync_status == "conflict_detached":
        return {"action": "skipped_detached", "sync_status": "conflict_detached"}

    event_type = str(change.get("event_type") or "").strip().lower()
    if event_type == "deleted":
        if binding and policy == "canonical":
            note_id = str(binding["note_id"])
            expected_version = _expected_version(binding)
            coordinator = _active_organization_coordinator(notes_db)
            if coordinator is not None:
                capture_note_tombstone(
                    coordinator,
                    note_id=note_id,
                    expected_version=expected_version,
                    source="notes-ingestion",
                    key=_source_folder_request_key(
                        "note-delete", note_id, expected_version
                    ),
                )
            else:
                notes_db.soft_delete_note(note_id, expected_version)
            return {
                "action": "archived",
                "note_id": note_id,
                "sync_status": "archived_upstream_removed",
            }
        return {"action": "ignored_delete", "sync_status": sync_status}

    text = change.get("text")
    if text is None:
        raise ValueError("Change text is required for notes create/update events.")
    body = str(text)
    relative_path = str(change.get("relative_path") or "").strip()
    title = _title_from_text(relative_path, body)
    source_id = change.get("source_id")
    folder_paths = _folder_paths_from_relative_path(relative_path)
    coordinator = (
        _active_organization_coordinator(notes_db)
        if source_id is not None
        else None
    )
    if coordinator is not None:
        coordinator.require_ready()

    if binding:
        note_id = str(binding["note_id"])
        if coordinator is not None:
            expected_version = _expected_version(binding)
            _capture_ingestion_note(
                coordinator,
                note_id=note_id,
                title=title,
                content=body,
                request_key=_source_folder_request_key(
                    "note",
                    source_id,
                    note_id,
                    title,
                    body,
                    expected_version,
                ),
                expected_version=expected_version,
            )
        else:
            notes_db.update_note(
                note_id,
                {"title": title, "content": body},
                expected_version=_expected_version(binding),
            )
        if source_id is not None and coordinator is not None:
            _sync_source_folders_with_coordinator(
                coordinator,
                note_id=note_id,
                source_id=int(source_id),
                folder_paths=folder_paths,
            )
        elif source_id is not None and hasattr(notes_db, "sync_note_source_folders"):
            notes_db.sync_note_source_folders(note_id, int(source_id), folder_paths)
        return {"action": "updated", "note_id": note_id, "sync_status": "sync_managed"}

    if coordinator is not None:
        identity_key = _source_folder_request_key(
            "note-id",
            source_id,
            relative_path.casefold(),
        )
        note_id = str(UUID(identity_key[:32], version=4))
        _capture_ingestion_note(
            coordinator,
            note_id=note_id,
            title=title,
            content=body,
            request_key=_source_folder_request_key(
                "note",
                source_id,
                note_id,
                title,
                body,
            ),
        )
    else:
        note_id = notes_db.add_note(title=title, content=body)
    if source_id is not None and coordinator is not None:
        _sync_source_folders_with_coordinator(
            coordinator,
            note_id=note_id,
            source_id=int(source_id),
            folder_paths=folder_paths,
        )
    elif source_id is not None and hasattr(notes_db, "sync_note_source_folders"):
        notes_db.sync_note_source_folders(note_id, int(source_id), folder_paths)
    return {"action": "created", "note_id": note_id, "sync_status": "sync_managed"}
