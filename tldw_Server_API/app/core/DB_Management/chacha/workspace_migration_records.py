"""Validate and present receipt-only workspace migration records."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any


def serialize_workspace_migration_json(value: Any, field_name: str, *, default: Any) -> str:
    """Validate a serialized workspace migration field."""
    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import InputError

    if value is None:
        value = default
    if isinstance(value, str):
        try:
            json.loads(value)
        except json.JSONDecodeError as exc:
            raise InputError(f"{field_name} must be valid JSON when provided as a string.") from exc  # noqa: TRY003
        return value
    try:
        return json.dumps(value, ensure_ascii=True)
    except TypeError as exc:
        raise InputError(f"{field_name} must be JSON serializable.") from exc  # noqa: TRY003


def parse_workspace_migration_json(value: Any, *, fallback: Any) -> Any:
    """Read optional JSON using the existing fallback contract."""
    if value is None:
        return fallback
    if isinstance(value, (dict, list)):
        return value
    if not isinstance(value, str) or not value.strip():
        return fallback
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return fallback


def workspace_migration_session_from_row(row: Mapping[str, Any], *, chunks: list[dict[str, Any]]) -> dict[str, Any]:
    """Keep receipt-only sessions, including older finalized rows, non-destructive."""
    session = dict(row)
    session["_chunks"] = chunks
    # Hashes and byte counts do not store or read back the original content.
    session["client_delete_eligible"] = False
    recovery = parse_workspace_migration_json(session.get("recovery_manifest_json"), fallback={})
    if isinstance(recovery, dict) and recovery:
        recovery.update(client_delete_eligible=False, can_delete_legacy_storage=False, server_readback_verified=False)
        if recovery.get("verification_status") == "verified":
            recovery["verification_status"] = "content_not_migrated"
        session["recovery_manifest_json"] = serialize_workspace_migration_json(
            recovery, "recovery_manifest", default={}
        )
    return session
