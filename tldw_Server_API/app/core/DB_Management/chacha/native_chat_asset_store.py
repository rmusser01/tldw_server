"""Generation-bound native asset candidate reservations; no byte I/O."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from tldw_Server_API.app.core.DB_Management.backends.base import BackendType

if TYPE_CHECKING:
    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB


@dataclass(frozen=True)
class NativeCandidate:
    """Receipt for one immutable byte candidate in the owning transaction."""

    candidate_id: str
    generation: int
    state: str


@dataclass(frozen=True)
class NativeReclaimCandidate:
    """Exact descriptor for a candidate fenced from further adoption or writes."""

    candidate_id: str
    generation: int
    state: str
    namespace_id: str
    upload_id: str
    expected_hash: str
    size_bytes: int
    storage_key: str | None


class NativeChatAssetStore:
    """Lock the operation before accepting an attempt's candidate identity."""

    def __init__(self, db: CharactersRAGDB) -> None:
        self._db = db

    def _require_live_attempt(
        self, owner: Any, kind: str, operation_id: str, request_digest: str,
        generation: int, now: datetime, *, conn: Any,
    ) -> datetime:
        """Lock and validate the one generation allowed to write candidate state."""
        if type(generation) is not int or generation < 1:
            raise ValueError("invalid_candidate_identity")
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("candidate_now_must_be_aware")
        now = now.astimezone(timezone.utc)
        operation = self._db.native_forks._row(owner, kind, operation_id, conn=conn)
        if operation is None:
            raise ValueError("operation_not_recorded")
        self._db.native_forks._check_identity(operation, owner, request_digest)
        lease_text = operation["lease_expires_at"]
        if (operation["state"] != "preparing" or operation["attempt_generation"] != generation
                or not isinstance(lease_text, str)):
            raise ValueError("stale_attempt")
        try:
            expiry = datetime.fromisoformat(lease_text)
        except ValueError as exc:
            raise ValueError("operation_lease_invalid") from exc
        if expiry.tzinfo is None or expiry.utcoffset() is None or expiry.astimezone(timezone.utc) <= now:
            raise ValueError("stale_attempt")
        self._db.native_forks.lock_open_workspace(owner, conn=conn)
        return now

    def _candidate_row(self, client_id: str, candidate_id: str, *, conn: Any) -> Any:
        query = (
            "SELECT * FROM native_chat_asset_candidates WHERE client_id = ? AND candidate_id = ? FOR UPDATE"
            if self._db.backend_type == BackendType.POSTGRESQL else
            "SELECT * FROM native_chat_asset_candidates WHERE client_id = ? AND candidate_id = ?"
        )
        return conn.execute(query, (client_id, candidate_id)).fetchone()

    def reserve_candidate(
        self, owner: Any, kind: str, operation_id: str, request_digest: str,
        generation: int, candidate_id: str, namespace_id: str, upload_id: str,
        content_hash: str, size_bytes: int, mime_type: str, representation: str,
        *, now: datetime, conn: Any,
    ) -> NativeCandidate:
        """Reject stale workers before recording a candidate for external preparation."""
        if (not isinstance(generation, int) or isinstance(generation, bool) or generation < 1
                or any(not isinstance(value, str) or not 1 <= len(value) <= 256
                       for value in (candidate_id, namespace_id, upload_id))
                or not isinstance(content_hash, str)
                or re.fullmatch(r"[0-9a-f]{64}", content_hash) is None
                or not isinstance(size_bytes, int) or isinstance(size_bytes, bool) or size_bytes < 0
                or not isinstance(mime_type, str) or not 1 <= len(mime_type) <= 255
                or not isinstance(representation, str)
                or re.fullmatch(r"[a-z][a-z0-9_]{0,63}", representation) is None):
            raise ValueError("invalid_candidate_identity")
        now = self._require_live_attempt(owner, kind, operation_id, request_digest, generation, now, conn=conn)
        timestamp = now.isoformat()
        conn.execute(
            "INSERT INTO native_chat_asset_candidates "
            "(client_id, candidate_id, operation_kind, operation_id, owner_key, attempt_generation, "
            "storage_namespace_id, upload_id, expected_hash, expected_size_bytes, mime_type, "
            "representation, state, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'reserved', ?, ?) "
            "ON CONFLICT (client_id, candidate_id) DO NOTHING",
            (owner.client_id, candidate_id, kind, operation_id, owner.owner_key, generation,
             namespace_id, upload_id, content_hash, size_bytes, mime_type, representation,
             timestamp, timestamp),
        )
        row = self._candidate_row(owner.client_id, candidate_id, conn=conn)
        if row is None:
            raise RuntimeError("candidate_reservation_missing")
        expected = {
            "operation_kind": kind, "operation_id": operation_id, "owner_key": owner.owner_key,
            "attempt_generation": generation, "storage_namespace_id": namespace_id,
            "upload_id": upload_id, "expected_hash": content_hash, "expected_size_bytes": size_bytes,
            "mime_type": mime_type, "representation": representation,
        }
        if any(row[key] != value for key, value in expected.items()):
            raise ValueError("candidate_id_conflict")
        return NativeCandidate(candidate_id, generation, row["state"])

    def mark_candidate_prepared(
        self, owner: Any, kind: str, operation_id: str, request_digest: str,
        generation: int, candidate_id: str, actual_hash: str, actual_size: int,
        actual_mime: str, storage_key: str, *, now: datetime, conn: Any,
    ) -> NativeCandidate:
        """Record a caller-verified descriptor only for a live exact candidate."""
        if not isinstance(storage_key, str) or not 1 <= len(storage_key) <= 1024:
            raise ValueError("invalid_storage_key")
        now = self._require_live_attempt(owner, kind, operation_id, request_digest, generation, now, conn=conn)
        row = self._candidate_row(owner.client_id, candidate_id, conn=conn)
        if row is None or any((row["owner_key"] != owner.owner_key,
                               row["operation_kind"] != kind,
                               row["operation_id"] != operation_id,
                               row["attempt_generation"] != generation)):
            raise ValueError("candidate_not_found")
        if (row["expected_hash"] != actual_hash or row["expected_size_bytes"] != actual_size
                or row["mime_type"] != actual_mime):
            raise ValueError("candidate_content_mismatch")
        if row["state"] == "prepared":
            if row["storage_key"] != storage_key:
                raise ValueError("candidate_id_conflict")
            return NativeCandidate(candidate_id, generation, "prepared")
        if row["state"] != "reserved":
            raise ValueError("candidate_state_conflict")
        conn.execute(
            "UPDATE native_chat_asset_candidates SET state = 'prepared', storage_key = ?, updated_at = ? "
            "WHERE client_id = ? AND candidate_id = ? AND state = 'reserved'",
            (storage_key, now.isoformat(), owner.client_id, candidate_id),
        )
        return NativeCandidate(candidate_id, generation, "prepared")

    def _locked_candidate_for_operation(
        self, owner: Any, kind: str, operation_id: str, request_digest: str,
        candidate_id: str, *, conn: Any,
    ) -> tuple[dict[str, Any], Any]:
        operation = self._db.native_forks._row(owner, kind, operation_id, conn=conn)
        if operation is None:
            raise ValueError("operation_not_recorded")
        self._db.native_forks._check_identity(operation, owner, request_digest)
        candidate = self._candidate_row(owner.client_id, candidate_id, conn=conn)
        if candidate is None or any((candidate["owner_key"] != owner.owner_key,
                                     candidate["operation_kind"] != kind,
                                     candidate["operation_id"] != operation_id)):
            raise ValueError("candidate_not_found")
        return operation, candidate

    def _require_unclaimed(self, owner: Any, candidate_id: str, *, conn: Any) -> None:
        lock = " FOR UPDATE" if self._db.backend_type == BackendType.POSTGRESQL else ""
        claim = conn.execute(
            "SELECT claim_id FROM native_chat_asset_claims "
            "WHERE client_id = ? AND candidate_id = ? AND state = 'live' LIMIT 1" + lock,  # nosec B608 - fixed backend-only lock suffix.
            (owner.client_id, candidate_id),
        ).fetchone()
        if claim is not None:
            raise ValueError("candidate_still_claimed")

    def claim_reclamation(
        self, owner: Any, kind: str, operation_id: str, request_digest: str,
        candidate_id: str, *, now: datetime, conn: Any,
    ) -> NativeReclaimCandidate | None:
        """Fence an abandoned unclaimed candidate before external writer drain."""
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("reclamation_now_must_be_aware")
        now = now.astimezone(timezone.utc)
        operation, candidate = self._locked_candidate_for_operation(
            owner, kind, operation_id, request_digest, candidate_id, conn=conn,
        )
        if candidate["state"] == "discarded":
            return None
        self._require_unclaimed(owner, candidate_id, conn=conn)
        if candidate["state"] == "adopted" and operation["state"] in {"preparing", "committed"}:
            return None
        if operation["state"] == "preparing":
            lease_text = operation["lease_expires_at"]
            if not isinstance(lease_text, str):
                raise ValueError("operation_lease_invalid")
            try:
                expiry = datetime.fromisoformat(lease_text)
            except ValueError as exc:
                raise ValueError("operation_lease_invalid") from exc
            if expiry.tzinfo is None or expiry.utcoffset() is None:
                raise ValueError("operation_lease_invalid")
            if expiry.astimezone(timezone.utc) > now and candidate["state"] != "reclaiming":
                return None
        if candidate["state"] not in {"reserved", "prepared", "adopted", "reclaiming"}:
            raise ValueError("candidate_state_conflict")
        if candidate["state"] != "reclaiming":
            conn.execute(
                "UPDATE native_chat_asset_candidates SET state = 'reclaiming', updated_at = ? "
                "WHERE client_id = ? AND candidate_id = ? AND state = ?",
                (now.isoformat(), owner.client_id, candidate_id, candidate["state"]),
            )
        return NativeReclaimCandidate(
            candidate_id, candidate["attempt_generation"], "reclaiming",
            candidate["storage_namespace_id"], candidate["upload_id"],
            candidate["expected_hash"], candidate["expected_size_bytes"], candidate["storage_key"],
        )

    def finish_reclamation(
        self, owner: Any, kind: str, operation_id: str, request_digest: str,
        candidate_id: str, generation: int, *, conn: Any,
    ) -> bool:
        """Finalize only after exact byte removal and any quota release confirmation."""
        _, candidate = self._locked_candidate_for_operation(
            owner, kind, operation_id, request_digest, candidate_id, conn=conn,
        )
        if candidate["attempt_generation"] != generation:
            raise ValueError("stale_reclamation")
        if candidate["state"] == "discarded":
            return False
        if candidate["state"] != "reclaiming":
            raise ValueError("candidate_not_reclaiming")
        self._require_unclaimed(owner, candidate_id, conn=conn)
        quota = conn.execute(
            "SELECT owner_key, size_bytes, state FROM native_chat_quota_intents "
            "WHERE client_id = ? AND candidate_id = ? AND intent_kind = 'reserve'",
            (owner.client_id, candidate_id),
        ).fetchone()
        if quota is not None:
            if quota["owner_key"] != owner.owner_key or quota["size_bytes"] != candidate["expected_size_bytes"]:
                raise ValueError("quota_intent_mismatch")
            released = conn.execute(
                "SELECT owner_key, size_bytes, state FROM native_chat_quota_intents "
                "WHERE client_id = ? AND candidate_id = ? AND intent_kind = 'release'",
                (owner.client_id, candidate_id),
            ).fetchone()
            if released is None or released["state"] != "confirmed":
                raise ValueError("quota_release_unconfirmed")
            if released["owner_key"] != owner.owner_key or released["size_bytes"] != candidate["expected_size_bytes"]:
                raise ValueError("quota_intent_mismatch")
        updated = conn.execute(
            "UPDATE native_chat_asset_candidates SET state = 'discarded', updated_at = ? "
            "WHERE client_id = ? AND candidate_id = ? AND state = 'reclaiming' AND attempt_generation = ?",
            (self._db._get_current_utc_timestamp_iso(), owner.client_id, candidate_id, generation),
        )
        return updated.rowcount == 1
