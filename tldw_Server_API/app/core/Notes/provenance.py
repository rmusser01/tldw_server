"""Notes response and local transaction boundaries for independent evidence."""

from collections.abc import Callable
from typing import Any

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, ConflictError
from tldw_Server_API.app.core.Sync.v2.notes_provenance_contract import (
    read_notes_provenance,
    retain_notes_provenance,
    strip_notes_provenance,
)


def provenance_response(
    note: dict[str, Any], record: dict[str, Any] | None, *, supported: bool = True, portable: bool = False
) -> dict[str, Any]:
    """Project a retained child without treating editable markers as authority."""
    marker = read_notes_provenance(str(note.get("content") or ""))
    state = (
        "unsupported" if not supported else "absent" if record is None else "deleted" if record["deleted"] else "active"
    )
    payload = record["payload"] if state == "active" else None
    note.update(
        knowledge_provenance_state=state,
        knowledge_provenance_version=record["version"] if record else 0 if supported else None,
        knowledge_provenance_hash=record["object_hash"] if record else None,
        knowledge_provenance=payload,
        knowledge_provenance_reconciliation=(
            "canonical_wins" if record and marker is not None and marker != payload else None
        ),
    )
    if portable:
        if state == "active":
            note["content"] = retain_notes_provenance(note["content"], payload)
        elif state == "deleted":
            note["content"] = strip_notes_provenance(note["content"])
    return note


def save_local_note(
    db: CharactersRAGDB,
    *,
    note_id: str | None,
    fields: dict[str, Any],
    expected_note_version: int = 0,
    provenance: dict[str, Any] | None = None,
    expected_provenance_version: int | None = None,
    receipt_key: str | None = None,
    request_fingerprint: str | None = None,
    load_result: Callable[[str], dict[str, Any]] | None = None,
    restore: bool = False,
) -> str | dict[str, Any]:
    """Commit the core note and an explicit child replacement in one transaction."""
    if provenance is None and load_result is None:
        if expected_note_version == 0:
            return db.add_note(note_id=note_id, **fields)
        if not db.update_note(note_id=note_id, update_data=fields, expected_version=expected_note_version):
            raise ConflictError("Note version mismatch")
        return note_id
    with db.transaction() as conn:
        if receipt_key is not None:
            if request_fingerprint is None or load_result is None:
                raise ValueError("A local receipt requires a fingerprint and acknowledgment loader")
            replay = db.note_provenance_store.claim_receipt(receipt_key, request_fingerprint, conn)
            if replay is not None:
                return replay
        if expected_note_version == 0:
            note_id = db.add_note(note_id=note_id, **fields, conn=conn)
        elif not db.update_note(note_id, fields, expected_note_version, conn=conn):
            raise ConflictError("Note version mismatch")
        if provenance is not None:
            db.note_provenance_store.put(note_id, provenance, expected_provenance_version, conn=conn, restore=restore)
        if load_result is not None:
            response = load_result(note_id)
            if receipt_key is not None:
                db.note_provenance_store.complete_receipt(receipt_key, request_fingerprint, response, conn)
            return response
    return note_id
