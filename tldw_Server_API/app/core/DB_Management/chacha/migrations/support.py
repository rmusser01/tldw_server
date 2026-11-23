"""Helpers shared by the schema-migration steps, moved out of CharactersRAGDB (TASK-13950.22)."""

from __future__ import annotations

import hashlib
import json
import math
import uuid
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import SchemaError
from tldw_Server_API.app.core.Sync_Contracts.models import validate_notes_note_upsert_payload
from tldw_Server_API.app.core.Sync_Contracts.notes_link import (
    NOTES_LINK_LABEL_MAX_CHARS,
    NOTES_LINK_WEIGHT_MAX,
    NotesLinkValidationError,
    validate_notes_link_object_id,
    validate_notes_link_properties,
)
from tldw_Server_API.app.core.Sync_Contracts.notes_moodboard_studio_contract import (
    canonical_json_bytes as canonical_moodboard_studio_json_bytes,
)


def _web_clipper_v56_rekey_plan(rows: list[Any]) -> list[tuple[str, str, str]]:
    """Validate legacy clip mappings and return ``(clip, old, new)`` rekeys."""
    plan: list[tuple[str, str, str]] = []
    targets: set[str] = set()
    for row in rows:
        clip_id = str(row["clip_id"])
        note_id = str(row["note_id"])
        if row["existing_note_id"] is None:
            raise SchemaError(f"WebClipper mapping '{clip_id}' references a missing note.")  # noqa: TRY003
        if clip_id != note_id:
            raise SchemaError(f"WebClipper v55 mapping '{clip_id}' has inconsistent identities.")  # noqa: TRY003
        try:
            parsed = uuid.UUID(note_id)
        except ValueError:
            digest = hashlib.sha256(
                f"web-clipper-migration:notes.note:{clip_id}".encode()
            ).hexdigest()
            replacement = str(uuid.UUID(digest[:32], version=4))
            if replacement in targets:
                raise SchemaError("WebClipper note rekey collision detected.") from None  # noqa: TRY003
            targets.add(replacement)
            plan.append((clip_id, note_id, replacement))
            continue
        if parsed.version != 4 or str(parsed) != note_id:
            raise SchemaError(  # noqa: TRY003
                f"WebClipper note '{note_id}' is a noncanonical UUID; explicit migration is required."
            )
    return plan


def _notes_link_v58_transform_row(
    row: Mapping[str, Any],
    *,
    source_owner: object,
    target_owner: object,
) -> dict[str, Any]:
    """Validate one legacy edge and return its canonical v58 values."""

    try:
        edge_id = validate_notes_link_object_id(str(row["edge_id"]))
        source_note_id = validate_notes_link_object_id(str(row["from_note_id"]))
        target_note_id = validate_notes_link_object_id(str(row["to_note_id"]))
    except (KeyError, NotesLinkValidationError) as exc:
        raise SchemaError(f"Notes link v58 requires canonical UUIDv4 identity: {exc}") from exc  # noqa: TRY003

    owner = str(row.get("user_id") or "")
    if not owner:
        raise SchemaError("Notes link v58 requires a non-empty owner.")  # noqa: TRY003
    if source_owner is None or target_owner is None:
        raise SchemaError("Notes link v58 requires both note endpoints to exist.")  # noqa: TRY003
    if str(source_owner) != owner or str(target_owner) != owner:
        raise SchemaError("Notes link v58 requires both endpoints to have the same owner.")  # noqa: TRY003
    if source_note_id == target_note_id:
        raise SchemaError("Notes link v58 endpoints must differ.")  # noqa: TRY003
    if row.get("type") != "manual":
        raise SchemaError("Notes link v58 supports only manual explicit links.")  # noqa: TRY003

    directed_value = row.get("directed")
    if isinstance(directed_value, bool):
        directed = directed_value
    elif isinstance(directed_value, int) and directed_value in (0, 1):
        directed = bool(directed_value)
    else:
        raise SchemaError("Notes link v58 directed must be a strict boolean.")  # noqa: TRY003
    if not directed and source_note_id > target_note_id:
        raise SchemaError("Notes link v58 undirected endpoints must use canonical order.")  # noqa: TRY003

    raw_weight = row.get("weight")
    weight_value = 1.0 if raw_weight is None else raw_weight
    if isinstance(weight_value, bool):
        raise SchemaError("Notes link v58 weight is invalid.")  # noqa: TRY003
    try:
        weight = float(weight_value)
    except (TypeError, ValueError) as exc:
        raise SchemaError("Notes link v58 weight is invalid.") from exc  # noqa: TRY003
    if not math.isfinite(weight) or not 0 <= weight <= NOTES_LINK_WEIGHT_MAX:
        raise SchemaError("Notes link v58 weight is outside the supported range.")  # noqa: TRY003

    raw_metadata = row.get("metadata")
    if raw_metadata is None:
        properties: object = {}
    elif isinstance(raw_metadata, str):
        try:
            properties = json.loads(raw_metadata)
        except json.JSONDecodeError as exc:
            raise SchemaError("Notes link v58 metadata must be valid JSON.") from exc  # noqa: TRY003
    elif isinstance(raw_metadata, Mapping):
        properties = dict(raw_metadata)
    else:
        raise SchemaError("Notes link v58 metadata must be a JSON object.")  # noqa: TRY003
    if not isinstance(properties, dict):
        raise SchemaError("Notes link v58 metadata must be a JSON object.")  # noqa: TRY003

    raw_label = properties.get("label")
    if raw_label is not None and not isinstance(raw_label, str):
        raise SchemaError("Notes link v58 metadata.label must be a string or null.")  # noqa: TRY003
    label = raw_label
    if label is not None:
        if len(label) > NOTES_LINK_LABEL_MAX_CHARS:
            raise SchemaError("Notes link v58 label exceeds the supported bound.")  # noqa: TRY003
        properties = dict(properties)
        properties.pop("label")
    try:
        canonical_properties = validate_notes_link_properties(properties)
    except NotesLinkValidationError as exc:
        raise SchemaError(f"Notes link v58 properties are invalid: {exc}") from exc  # noqa: TRY003

    created_at = row.get("created_at")
    created_by = str(row.get("created_by") or "")
    if created_at is None or not created_by:
        raise SchemaError("Notes link v58 requires stable creation provenance.")  # noqa: TRY003
    return {
        "edge_id": edge_id,
        "user_id": owner,
        "from_note_id": source_note_id,
        "to_note_id": target_note_id,
        "type": "manual",
        "directed": directed,
        "weight": weight,
        "label": label,
        "properties": canonical_properties,
        "created_at": created_at,
        "last_modified": created_at,
        "created_by": created_by,
        "version": 1,
        "deleted": False,
        "deleted_at": None,
    }


def _validate_notes_link_v58_unique(rows: Sequence[Mapping[str, Any]]) -> None:
    """Fail closed when legacy rows collide under canonical logical identity."""

    identities: set[tuple[str, str, bool, str, str]] = set()
    edge_ids: set[str] = set()
    for row in rows:
        edge_id = str(row["edge_id"])
        identity = (
            str(row["user_id"]),
            str(row["type"]),
            bool(row["directed"]),
            str(row["from_note_id"]),
            str(row["to_note_id"]),
        )
        if edge_id in edge_ids or identity in identities:
            raise SchemaError("Notes link v58 found a duplicate logical edge.")  # noqa: TRY003
        edge_ids.add(edge_id)
        identities.add(identity)


def _note_task_v60_migration_checkpoint(_stage: str) -> None:
    """No-op seam used to prove rollback at each migration boundary."""


def _legacy_sync_timestamp_v61(value: object) -> str:
    text = str(value or "").strip()
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except (TypeError, ValueError, OverflowError) as exc:
        raise SchemaError("Notes moodboard/Studio v61 migration found an invalid timestamp.") from exc  # noqa: TRY003
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    utc = parsed.astimezone(timezone.utc)
    normalized = utc.isoformat(timespec="microseconds").replace("+00:00", "Z")
    return normalized.replace(".000000Z", "Z")


def _notes_studio_envelope_size_v61(
    payload: Mapping[str, Any],
    *,
    revision: int,
    deleted: bool,
    canonical_hash: str,
) -> int:
    return len(
        canonical_moodboard_studio_json_bytes(
            {
                "domain": "notes.studio_document",
                "schema_version": 1,
                "operation": "tombstone" if deleted else "upsert",
                "object_id": payload["note_id"],
                "parent_id": payload["note_id"],
                "object_revision": revision,
                "payload_hash": canonical_hash,
                "payload": payload,
            }
        )
    )


def _legacy_note_hash_v61(ctx, note: Mapping[str, Any]) -> str:
    payload = validate_notes_note_upsert_payload(
        {
            "title": note.get("title"),
            "content": note.get("content"),
            "conversation_id": note.get("conversation_id"),
            "message_id": note.get("message_id"),
        }
    )
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _migration_v65_checkpoint(ctx, stage: str) -> None:
    """No-op failpoint used to verify atomic v65 migrations."""


def _native_fork_schema_is_installed(ctx, conn: Any) -> bool:
    """Recognize the published H2 catalog without replacing durable receipts.

        H2 previously used the same version numbers as dev's Persona migration.
        A complete native catalog is retained; a partial catalog needs explicit
        repair instead of silently marking missing storage as migrated.
        """
    from tldw_Server_API.app.core.DB_Management.chacha.native_fork_schema import NATIVE_CHAT_TABLES

    present_tables = {
        table for table in NATIVE_CHAT_TABLES if ctx.backend.table_exists(table, connection=conn)
    }
    required_columns = {
        "workspaces": {"native_chat_admission_closed"},
        "conversations": {
            "required_projection_version", "native_creation_operation_kind",
            "native_creation_operation_id", "native_bundle_json",
        },
    }
    present_columns = {
        table: required & {column["name"] for column in ctx.backend.get_table_info(table, connection=conn)}
        for table, required in required_columns.items()
    }
    if not present_tables and not any(present_columns.values()):
        return False
    if present_tables != set(NATIVE_CHAT_TABLES) or present_columns != required_columns:
        raise SchemaError("Incomplete native fork schema; manual repair is required.")  # noqa: TRY003
    return True
