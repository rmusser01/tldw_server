"""Generation-bound native asset candidate reservations; no byte I/O."""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from tldw_Server_API.app.core.DB_Management.backends.base import BackendType

if TYPE_CHECKING:
    from tldw_Server_API.app.api.v1.schemas.native_fork_schemas import NativeAssetContextUpdateV1, NativeAssetContextV1
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


class NativeQuotaReleasePendingError(ValueError):
    """Removed bytes retain their accounting fence until original authority settles."""


class NativeChatAssetStore:
    """Lock the operation before accepting an attempt's candidate identity."""

    def __init__(self, db: CharactersRAGDB) -> None:
        self._db = db

    def _require_live_attempt(
        self, owner: Any, kind: str, operation_id: str, request_digest: str,
        generation: int, now: datetime, *, conn: Any,
    ) -> datetime:
        """Lock and validate the one generation allowed to write candidate state."""
        if owner.client_id != self._db.client_id:
            raise ValueError("operation_owner_mismatch")
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
        if self._db.backend_type == BackendType.POSTGRESQL:
            keys = {
                int.from_bytes(
                    hashlib.sha256(f"native_candidate_path\0{owner.client_id}\0{kind}\0{value}".encode()).digest()[:8],
                    "big",
                    signed=True,
                )
                for kind, value in (("namespace", namespace_id), ("upload", upload_id))
            }
            for key in sorted(keys):
                conn.execute("SELECT pg_advisory_xact_lock(?)", (key,)).fetchone()
        self._require_exclusive_paths(owner.client_id, candidate_id, namespace_id, upload_id, conn=conn)
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

    def _require_exclusive_paths(
        self,
        client_id: str,
        candidate_id: str,
        namespace_id: str,
        upload_id: str,
        *,
        conn: Any,
    ) -> None:
        """Retain path ownership even after discard so old workers cannot alias new bytes."""
        other = conn.execute(
            "SELECT 1 FROM native_chat_asset_candidates WHERE client_id = ? AND candidate_id <> ? "
            "AND (storage_namespace_id = ? OR upload_id = ?) LIMIT 1",
            (client_id, candidate_id, namespace_id, upload_id),
        ).fetchone()
        if other is not None:
            raise ValueError("candidate_storage_identity_conflict")

    def read_candidate_for_write(
        self,
        owner: Any,
        kind: str,
        operation_id: str,
        request_digest: str,
        generation: int,
        candidate_id: str,
        *,
        now: datetime,
        conn: Any,
    ) -> dict[str, Any]:
        """Detach one reserved descriptor after the guarded writer's short admission."""
        self._require_live_attempt(owner, kind, operation_id, request_digest, generation, now, conn=conn)
        row = self._candidate_row(owner.client_id, candidate_id, conn=conn)
        if row is None or any(
            (
                row["owner_key"] != owner.owner_key,
                row["operation_kind"] != kind,
                row["operation_id"] != operation_id,
                row["attempt_generation"] != generation,
            )
        ):
            raise ValueError("candidate_not_found")
        if row["state"] != "reserved":
            raise ValueError("candidate_state_conflict")
        self._require_exclusive_paths(
            owner.client_id, candidate_id, row["storage_namespace_id"], row["upload_id"], conn=conn
        )
        self._require_confirmed_quota(owner, row, conn=conn)
        self._require_live_attempt(
            owner, kind, operation_id, request_digest, generation, datetime.now(timezone.utc), conn=conn
        )
        return dict(row)

    def _require_confirmed_quota(self, owner: Any, row: Any, *, conn: Any) -> None:
        """Require the original confirmed reservation before writes or adoption."""
        candidate_id = row["candidate_id"]
        quota = conn.execute(
            "SELECT owner_key,size_bytes,state FROM native_chat_quota_intents "
            "WHERE client_id = ? AND candidate_id = ? AND intent_kind = 'reserve'",
            (owner.client_id, candidate_id),
        ).fetchone()
        if quota is None or quota["state"] != "confirmed":
            raise ValueError("quota_reservation_unconfirmed")
        if quota["owner_key"] != owner.owner_key or quota["size_bytes"] != row["expected_size_bytes"]:
            raise ValueError("quota_intent_mismatch")
        if (
            conn.execute(
                "SELECT 1 FROM native_chat_quota_intents WHERE client_id = ? AND candidate_id = ? AND intent_kind = 'release'",
                (owner.client_id, candidate_id),
            ).fetchone()
            is not None
        ):
            raise ValueError("candidate_release_started")

    @staticmethod
    def _quota_identity(owner: Any, quota_owner: Any) -> str:
        """Freeze authenticated account, native scope and applicable pool attribution."""
        import json
        from dataclasses import asdict

        from tldw_Server_API.app.core.DB_Management.storage_quota_reservation_store import NativeQuotaOwner
        if (not isinstance(quota_owner, NativeQuotaOwner)
                or str(quota_owner.user_id) != owner.client_id
                or quota_owner.owner_key != owner.owner_key
                or quota_owner.scope_type != owner.scope.kind
                or quota_owner.workspace_id != owner.scope.workspace_id):
            raise ValueError('quota_intent_mismatch')
        return json.dumps(asdict(quota_owner), sort_keys=True, separators=(',', ':'))

    def record_quota_reservation_intent(
        self, owner: Any, kind: str, operation_id: str, request_digest: str,
        candidate_id: str, generation: int, quota_owner: Any, *, conn: Any,
    ) -> int:
        """Commit original identity before any external quota allocation."""
        identity = self._quota_identity(owner, quota_owner)
        self._require_live_attempt(owner, kind, operation_id, request_digest, generation, datetime.now(timezone.utc), conn=conn)
        _, candidate = self._locked_candidate_for_operation(owner, kind, operation_id, request_digest, candidate_id, conn=conn)
        if candidate['attempt_generation'] != generation or candidate['state'] != 'reserved':
            raise ValueError('stale_candidate')
        if conn.execute("SELECT 1 FROM native_chat_quota_intents WHERE client_id = ? AND candidate_id = ? AND intent_kind = 'release'", (owner.client_id, candidate_id)).fetchone():
            raise ValueError('candidate_release_started')
        stamp = self._db._get_current_utc_timestamp_iso()
        conn.execute(
            "INSERT INTO native_chat_quota_intents (client_id,candidate_id,intent_kind,owner_key,size_bytes,state,created_at,updated_at,quota_owner_json) "
            "VALUES (?,?,'reserve',?,?,'pending',?,?,?) ON CONFLICT (client_id,candidate_id,intent_kind) DO NOTHING",
            (owner.client_id, candidate_id, owner.owner_key, candidate['expected_size_bytes'], stamp, stamp, identity),
        )
        row = conn.execute("SELECT owner_key,size_bytes,quota_owner_json FROM native_chat_quota_intents WHERE client_id = ? AND candidate_id = ? AND intent_kind = 'reserve'", (owner.client_id, candidate_id)).fetchone()
        if (row is None or row['owner_key'] != owner.owner_key or row['size_bytes'] != candidate['expected_size_bytes']
                or row['quota_owner_json'] != identity):
            raise ValueError('quota_intent_mismatch')
        self._require_live_attempt(owner, kind, operation_id, request_digest, generation, datetime.now(timezone.utc), conn=conn)
        return candidate['expected_size_bytes']

    def confirm_quota_reservation(
        self, owner: Any, kind: str, operation_id: str, request_digest: str,
        candidate_id: str, generation: int, quota_owner: Any, reservation: Any, *, conn: Any,
    ) -> None:
        """Recheck the original live writer after the external response before allowing bytes."""
        size = self.record_quota_reservation_intent(owner, kind, operation_id, request_digest, candidate_id, generation, quota_owner, conn=conn)
        if reservation is None or reservation.candidate_key != candidate_id or reservation.owner != quota_owner or reservation.size_bytes != size or reservation.state != 'active':
            raise ValueError('quota_reservation_unconfirmed')
        conn.execute("UPDATE native_chat_quota_intents SET state = 'confirmed', updated_at = ? WHERE client_id = ? AND candidate_id = ? AND intent_kind = 'reserve'", (self._db._get_current_utc_timestamp_iso(), owner.client_id, candidate_id))

    def read_quota_release_intent(
        self, owner: Any, kind: str, operation_id: str, request_digest: str, candidate_id: str, *, conn: Any,
    ) -> tuple[Any, int, int]:
        """Read only the immutable release identity produced by guarded byte removal."""
        import json

        from tldw_Server_API.app.core.DB_Management.storage_quota_reservation_store import NativeQuotaOwner
        _, candidate = self._locked_candidate_for_operation(owner, kind, operation_id, request_digest, candidate_id, conn=conn)
        if candidate['state'] not in {'reclaiming', 'discarded'}:
            raise ValueError('candidate_not_reclaiming')
        self._require_unclaimed(owner, candidate_id, conn=conn)
        row = conn.execute("SELECT owner_key,size_bytes,quota_owner_json FROM native_chat_quota_intents WHERE client_id = ? AND candidate_id = ? AND intent_kind = 'release'", (owner.client_id, candidate_id)).fetchone()
        if row is None or row['owner_key'] != owner.owner_key or row['size_bytes'] != candidate['expected_size_bytes'] or row['quota_owner_json'] is None:
            raise ValueError('quota_intent_mismatch')
        quota_owner = NativeQuotaOwner(**json.loads(row['quota_owner_json']))
        if self._quota_identity(owner, quota_owner) != row['quota_owner_json']:
            raise ValueError('quota_intent_mismatch')
        return quota_owner, row['size_bytes'], candidate['attempt_generation']

    def confirm_quota_release(
        self, owner: Any, kind: str, operation_id: str, request_digest: str, candidate_id: str,
        reservation: Any, *, conn: Any,
    ) -> int:
        """Confirm original released authority, keeping unknown outcomes pending."""
        quota_owner, size, generation = self.read_quota_release_intent(owner, kind, operation_id, request_digest, candidate_id, conn=conn)
        if reservation is None or reservation.candidate_key != candidate_id or reservation.owner != quota_owner or reservation.size_bytes != size or reservation.state != 'released':
            raise NativeQuotaReleasePendingError('quota_release_unconfirmed')
        conn.execute("UPDATE native_chat_quota_intents SET state = 'confirmed', updated_at = ? WHERE client_id = ? AND candidate_id = ? AND intent_kind = 'release'", (self._db._get_current_utc_timestamp_iso(), owner.client_id, candidate_id))
        return generation

    def record_release_after_removal(
        self,
        owner: Any,
        kind: str,
        operation_id: str,
        request_digest: str,
        candidate_id: str,
        generation: int,
        *,
        conn: Any,
    ) -> None:
        """Record a pending authority reconciliation only after guarded physical removal."""
        _, row = self._locked_candidate_for_operation(
            owner, kind, operation_id, request_digest, candidate_id, conn=conn
        )
        if row["attempt_generation"] != generation or row["state"] != "reclaiming":
            raise ValueError("stale_reclamation")
        self._require_unclaimed(owner, candidate_id, conn=conn)
        reserve = conn.execute(
            "SELECT owner_key,size_bytes,quota_owner_json FROM native_chat_quota_intents "
            "WHERE client_id = ? AND candidate_id = ? AND intent_kind = 'reserve'",
            (owner.client_id, candidate_id),
        ).fetchone()
        if reserve is None:
            return
        if reserve["owner_key"] != owner.owner_key or reserve["size_bytes"] != row["expected_size_bytes"]:
            raise ValueError("quota_intent_mismatch")
        stamp = self._db._get_current_utc_timestamp_iso()
        conn.execute(
            "INSERT INTO native_chat_quota_intents "
            "(client_id,candidate_id,intent_kind,owner_key,size_bytes,state,created_at,updated_at,quota_owner_json) "
            "VALUES (?,?,'release',?,?,'pending',?,?,?) ON CONFLICT (client_id,candidate_id,intent_kind) DO NOTHING",
            (owner.client_id, candidate_id, owner.owner_key, row["expected_size_bytes"], stamp, stamp, reserve['quota_owner_json']),
        )
        release = conn.execute(
            "SELECT owner_key,size_bytes,quota_owner_json FROM native_chat_quota_intents "
            "WHERE client_id = ? AND candidate_id = ? AND intent_kind = 'release'",
            (owner.client_id, candidate_id),
        ).fetchone()
        if (
            release is None
            or release["owner_key"] != owner.owner_key
            or release["size_bytes"] != row["expected_size_bytes"]
            or release['quota_owner_json'] != reserve['quota_owner_json']
        ):
            raise ValueError("quota_intent_mismatch")

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

    def adopt_fork_candidate(
        self, owner: Any, operation_id: str, request_digest: str, generation: int,
        candidate_id: str, conversation_id: str, *, now: datetime, conn: Any,
    ) -> dict[str, Any]:
        """Install one exact live child claim in the caller's native fork transaction.

        Graph/reference/receipt failures must roll back this same transaction.
        This private transition performs no byte or external quota work and grants no readiness.
        """
        from tldw_Server_API.app.core.Sync.v2.blob_store import LocalSyncBlobStore

        self._db.native_forks._require_active_transaction(conn)
        if not isinstance(conversation_id, str) or not 1 <= len(conversation_id) <= 256:
            raise ValueError("invalid_native_child_identity")
        kind = "native_fork_v1"
        now = self._require_live_attempt(owner, kind, operation_id, request_digest, generation, now, conn=conn)
        lock = " FOR UPDATE" if self._db.backend_type == BackendType.POSTGRESQL else ""
        child = conn.execute(
            "SELECT client_id,scope_type,workspace_id,deleted,"
            "required_projection_version,native_creation_operation_kind,native_creation_operation_id "
            "FROM conversations WHERE id = ? AND client_id = ?" + lock,  # nosec B608 - fixed backend-only lock suffix.
            (conversation_id, owner.client_id),
        ).fetchone()
        if child is None or any((
            child["scope_type"] != owner.scope.kind, child["workspace_id"] != owner.scope.workspace_id,
            bool(child["deleted"]),
            child["required_projection_version"] != "native-fork-v1",
            child["native_creation_operation_kind"] != kind,
            child["native_creation_operation_id"] != operation_id,
        )):
            raise ValueError("native_child_adoption_conflict")
        row = self._candidate_row(owner.client_id, candidate_id, conn=conn)
        if row is None or any((
            row["owner_key"] != owner.owner_key, row["operation_kind"] != kind,
            row["operation_id"] != operation_id, row["attempt_generation"] != generation,
        )):
            raise ValueError("candidate_not_found")
        if row["state"] not in {"prepared", "adopted"}:
            raise ValueError("candidate_state_conflict")
        if row["representation"] not in {
            "embedded_image_v1", "generated_image_revision_v1", "document_original_v1", "document_text_v1",
        } or row["expected_size_bytes"] < 1:
            raise ValueError("native_asset_representation_unsupported")
        canonical_key = LocalSyncBlobStore.namespace_storage_key(row["storage_namespace_id"], "sha256:" + row["expected_hash"])
        if row["storage_key"] != canonical_key:
            raise ValueError("candidate_storage_identity_conflict")
        self._require_exclusive_paths(owner.client_id, candidate_id, row["storage_namespace_id"], row["upload_id"], conn=conn)
        self._require_confirmed_quota(owner, row, conn=conn)
        claims = conn.execute(
            "SELECT * FROM native_chat_asset_claims WHERE client_id = ? AND candidate_id = ? "
            "ORDER BY claim_id" + lock,  # nosec B608 - fixed backend-only lock suffix.
            (owner.client_id, candidate_id),
        ).fetchall()
        expected = {
            "client_id": owner.client_id, "owner_key": owner.owner_key, "conversation_id": conversation_id,
            "candidate_id": candidate_id, "asset_revision": 1, "content_hash": row["expected_hash"],
            "size_bytes": row["expected_size_bytes"], "mime_type": row["mime_type"],
            "representation": row["representation"], "state": "live",
        }
        self._require_live_attempt(owner, kind, operation_id, request_digest, generation, datetime.now(timezone.utc), conn=conn)
        if row["state"] == "adopted":
            if len(claims) != 1 or any(claims[0][key] != value for key, value in expected.items()):
                raise ValueError("candidate_claim_conflict")
            return dict(claims[0])
        if claims:
            raise ValueError("candidate_claim_conflict")
        claim_id, asset_id = self._db._generate_uuid(), self._db._generate_uuid()
        stamp = now.isoformat()
        conn.execute(
            "INSERT INTO native_chat_asset_claims "
            "(client_id,claim_id,owner_key,conversation_id,candidate_id,asset_id,asset_revision,"
            "content_hash,size_bytes,mime_type,representation,state,created_at,updated_at) "
            "VALUES (?,?,?,?,?,?,1,?,?,?,?, 'live',?,?)",
            (owner.client_id, claim_id, owner.owner_key, conversation_id, candidate_id, asset_id,
             row["expected_hash"], row["expected_size_bytes"], row["mime_type"], row["representation"], stamp, stamp),
        )
        changed = conn.execute(
            "UPDATE native_chat_asset_candidates SET state = 'adopted',updated_at = ? "
            "WHERE client_id = ? AND candidate_id = ? AND state = 'prepared' AND attempt_generation = ?",
            (stamp, owner.client_id, candidate_id, generation),
        )
        if changed.rowcount != 1:
            raise ValueError("candidate_state_conflict")
        return dict(conn.execute(
            "SELECT * FROM native_chat_asset_claims WHERE client_id = ? AND claim_id = ?",
            (owner.client_id, claim_id),
        ).fetchone())

    def read_committed_asset(
        self, owner: Any, conversation_id: str, asset_id: str, revision: str, *, conn: Any,
    ) -> dict[str, Any]:
        """Detach an exact committed fork claim; this grants no bundle readiness."""
        self._db.native_forks._require_active_transaction(conn)
        if owner.client_id != self._db.client_id:
            raise ValueError("operation_owner_mismatch")
        if any(not isinstance(value, str) or not 1 <= len(value) <= bound
               for value, bound in ((conversation_id, 256), (asset_id, 256), (revision, 128))):
            raise ValueError("invalid_native_asset_identity")
        matches = conn.execute(
            "SELECT a.claim_id,a.candidate_id,a.asset_revision,c.operation_kind,c.operation_id "
            "FROM native_chat_asset_claims a JOIN native_chat_asset_candidates c "
            "ON c.client_id = a.client_id AND c.candidate_id = a.candidate_id "
            "WHERE a.client_id = ? AND a.owner_key = ? AND a.conversation_id = ? AND a.asset_id = ? "
            "AND a.asset_revision = ?",
            (owner.client_id, owner.owner_key, conversation_id, asset_id, revision),
        ).fetchall()
        if len(matches) != 1 or str(matches[0]["asset_revision"]) != revision:
            raise ValueError("native_asset_not_found")
        initial = matches[0]
        if initial["operation_kind"] != "native_fork_v1":
            raise ValueError("native_asset_read_unsupported")
        operation = self._db.native_forks._row(owner, "native_fork_v1", initial["operation_id"], conn=conn)
        if operation is None:
            raise ValueError("operation_not_recorded")
        self._db.native_forks._check_identity(operation, owner, operation["request_digest"])
        if operation["state"] != "committed" or operation["child_conversation_id"] != conversation_id:
            raise ValueError("native_asset_operation_uncommitted")
        self._db.native_forks.lock_open_workspace(owner, conn=conn)
        lock = " FOR UPDATE" if self._db.backend_type == BackendType.POSTGRESQL else ""
        child = conn.execute(
            "SELECT scope_type,workspace_id,deleted,required_projection_version,"
            "native_creation_operation_kind,native_creation_operation_id "
            "FROM conversations WHERE client_id = ? AND id = ?" + lock,  # nosec B608 - fixed backend-only lock suffix.
            (owner.client_id, conversation_id),
        ).fetchone()
        if child is None or any((
            child["scope_type"] != owner.scope.kind, child["workspace_id"] != owner.scope.workspace_id,
            bool(child["deleted"]), child["required_projection_version"] != "native-fork-v1",
            child["native_creation_operation_kind"] != "native_fork_v1",
            child["native_creation_operation_id"] != initial["operation_id"],
        )):
            raise ValueError("native_asset_conversation_unavailable")
        candidate = self._candidate_row(owner.client_id, initial["candidate_id"], conn=conn)
        claim = conn.execute(
            "SELECT * FROM native_chat_asset_claims WHERE client_id = ? AND claim_id = ?" + lock,  # nosec B608 - fixed backend-only lock suffix.
            (owner.client_id, initial["claim_id"]),
        ).fetchone()
        return self._committed_claim_descriptor(
            owner, conversation_id, asset_id, revision, operation, candidate, claim, conn=conn,
        )

    def _committed_claim_descriptor(
        self,
        owner: Any,
        conversation_id: str,
        asset_id: str,
        revision: str,
        operation: Any,
        candidate: Any,
        claim: Any,
        *,
        conn: Any,
    ) -> dict[str, Any]:
        """Validate already-read provenance without acquiring an operation lock."""
        from tldw_Server_API.app.core.Sync.v2.blob_store import LocalSyncBlobStore

        if (
            candidate is None
            or claim is None
            or any(
                (
                    candidate["state"] != "adopted",
                    candidate["owner_key"] != owner.owner_key,
                    candidate["attempt_generation"] != operation["attempt_generation"],
                    candidate["operation_kind"] != "native_fork_v1",
                    candidate["operation_id"] != operation["operation_id"],
                    claim["state"] != "live",
                    claim["owner_key"] != owner.owner_key,
                    claim["conversation_id"] != conversation_id,
                    claim["asset_id"] != asset_id,
                    str(claim["asset_revision"]) != revision,
                    claim["candidate_id"] != candidate["candidate_id"],
                    claim["content_hash"] != candidate["expected_hash"],
                    claim["size_bytes"] != candidate["expected_size_bytes"],
                    claim["mime_type"] != candidate["mime_type"],
                    claim["representation"] != candidate["representation"],
                    claim["size_bytes"] < 1,
                    claim["representation"]
                    not in {
                        "embedded_image_v1",
                        "generated_image_revision_v1",
                        "document_original_v1",
                        "document_text_v1",
                    },
                )
            )
        ):
            raise ValueError("native_asset_claim_conflict")
        self._require_confirmed_quota(owner, candidate, conn=conn)
        self._require_exclusive_paths(
            owner.client_id, candidate["candidate_id"], candidate["storage_namespace_id"], candidate["upload_id"], conn=conn
        )
        key = LocalSyncBlobStore.namespace_storage_key(
            candidate["storage_namespace_id"], "sha256:" + candidate["expected_hash"]
        )
        if candidate["storage_key"] != key:
            raise ValueError("candidate_storage_identity_conflict")
        return {
            **{
                field: claim[field]
                for field in (
                    "claim_id",
                    "client_id",
                    "owner_key",
                    "conversation_id",
                    "candidate_id",
                    "asset_id",
                    "asset_revision",
                    "content_hash",
                    "size_bytes",
                    "mime_type",
                    "representation",
                )
            },
            **{
                field: candidate[field]
                for field in ("storage_namespace_id", "upload_id", "storage_key", "attempt_generation")
            },
        }


    def read_native_asset_context(self, owner: Any, conversation_id: str, *, conn: Any) -> NativeAssetContextV1:
        """Read private fork-backed context metadata; never read bytes or publish readiness."""
        import json

        from tldw_Server_API.app.api.v1.schemas.native_fork_schemas import (
            NativeAssetContextV1,
            NativeAssetManifestEntryV1,
        )

        self._db.native_forks._require_active_transaction(conn)
        if owner.client_id != self._db.client_id:
            raise ValueError("operation_owner_mismatch")
        if not isinstance(conversation_id, str) or not 1 <= len(conversation_id) <= 256:
            raise ValueError("invalid_native_asset_identity")
        self._db.native_forks.lock_open_workspace(owner, conn=conn)
        lock = " FOR UPDATE" if self._db.backend_type == BackendType.POSTGRESQL else ""
        child = conn.execute(
            "SELECT scope_type,workspace_id,deleted,required_projection_version,"
            "native_creation_operation_kind,native_creation_operation_id,native_bundle_json "
            "FROM conversations WHERE client_id = ? AND id = ?" + lock,  # nosec B608 - fixed backend-only lock suffix.
            (owner.client_id, conversation_id),
        ).fetchone()
        if child is None or any(
            (
                child["scope_type"] != owner.scope.kind,
                child["workspace_id"] != owner.scope.workspace_id,
                bool(child["deleted"]),
                child["required_projection_version"] != "native-fork-v1",
                child["native_creation_operation_kind"] != "native_fork_v1",
            )
        ):
            raise ValueError("native_asset_context_unsupported")
        try:
            bundle = json.loads(child["native_bundle_json"])
            if (
                not isinstance(bundle, dict)
                or type(bundle.get("version")) is not int
                or bundle["version"] != 1
                or bundle.get("state") != "ready"
                or bundle.get("required_projection_version") != "native-fork-v1"
                or type(bundle.get("settings_revision")) is not int
                or bundle["settings_revision"] < 1
            ):
                raise ValueError("invalid_native_bundle")
            context = NativeAssetContextV1(asset_manifest_revision=bundle["asset_manifest_revision"], asset_manifest=())
        except (ValueError, TypeError, KeyError) as exc:
            raise ValueError("native_asset_context_invalid") from exc
        # Deletion/publication waits on this conversation; no operation lock is taken in reverse order.
        operation = conn.execute(
            "SELECT * FROM native_chat_operations WHERE client_id = ? AND operation_kind = 'native_fork_v1' "
            "AND operation_id = ?",
            (owner.client_id, child["native_creation_operation_id"]),
        ).fetchone()
        if operation is None:
            raise ValueError("native_asset_operation_uncommitted")
        self._db.native_forks._check_identity(operation, owner, operation["request_digest"])
        if (
            operation["state"] not in {"committed", "gone"}
            or operation["child_conversation_id"] != conversation_id
            or operation["projection_version"] != child["required_projection_version"]
        ):
            raise ValueError("native_asset_operation_uncommitted")
        references = conn.execute(
            "SELECT * FROM native_chat_asset_references WHERE client_id = ? AND conversation_id = ? "
            "ORDER BY position,reference_id" + lock,  # nosec B608 - fixed backend-only lock suffix.
            (owner.client_id, conversation_id),
        ).fetchall()
        positions = set()
        for reference in references:
            if (
                reference["owner_key"] != owner.owner_key
                or reference["role"] is None
                or type(reference["position"]) is not int
                or reference["position"] < 0
                or reference["position"] in positions
                or type(reference["revision"]) is not int
                or reference["revision"] < 1
                or reference["context_enabled"] not in (False, True, 0, 1)
            ):
                raise ValueError("native_asset_reference_invalid")
            positions.add(reference["position"])
        # Lock candidates before claims in deterministic order, matching existing reclamation.
        initial_claims = conn.execute(
            "SELECT * FROM native_chat_asset_claims WHERE client_id = ? AND conversation_id = ?",
            (owner.client_id, conversation_id),
        ).fetchall()
        initial_by_id = {claim["claim_id"]: claim for claim in initial_claims}
        if any(reference["claim_id"] not in initial_by_id for reference in references):
            raise ValueError("native_asset_claim_conflict")
        candidate_ids = {initial_by_id[reference["claim_id"]]["candidate_id"] for reference in references}
        if None in candidate_ids:
            raise ValueError("native_asset_claim_conflict")
        candidates = {
            candidate_id: self._candidate_row(owner.client_id, candidate_id, conn=conn)
            for candidate_id in sorted(candidate_ids)
        }
        claims = conn.execute(
            "SELECT * FROM native_chat_asset_claims WHERE client_id = ? AND conversation_id = ? "
            "ORDER BY claim_id" + lock,  # nosec B608 - fixed backend-only lock suffix.
            (owner.client_id, conversation_id),
        ).fetchall()
        by_id = {claim["claim_id"]: claim for claim in claims}
        identities = Counter((claim["asset_id"], claim["asset_revision"]) for claim in claims)
        entries = []
        for reference in references:
            claim = by_id.get(reference["claim_id"])
            if claim is None or claim["candidate_id"] not in candidates:
                raise ValueError("native_asset_claim_conflict")
            if type(claim["asset_revision"]) is not int or claim["asset_revision"] < 1:
                raise ValueError("native_asset_claim_conflict")
            if identities[(claim["asset_id"], claim["asset_revision"])] != 1:
                raise ValueError("native_asset_claim_ambiguous")
            descriptor = self._committed_claim_descriptor(
                owner,
                conversation_id,
                claim["asset_id"],
                str(claim["asset_revision"]),
                operation,
                candidates[claim["candidate_id"]],
                claim,
                conn=conn,
            )
            entries.append(
                NativeAssetManifestEntryV1(
                    reference_id=reference["reference_id"],
                    reference_revision=str(reference["revision"]),
                    asset_id=descriptor["asset_id"],
                    revision=str(descriptor["asset_revision"]),
                    representation=descriptor["representation"],
                    role=reference["role"],
                    context_enabled=bool(reference["context_enabled"]),
                    hash=descriptor["content_hash"],
                    fidelity_disposition="retained",
                )
            )
        return NativeAssetContextV1(asset_manifest_revision=context.asset_manifest_revision, asset_manifest=tuple(entries))

    def update_native_asset_context(
        self,
        owner: Any,
        conversation_id: str,
        request: NativeAssetContextUpdateV1,
        *,
        conn: Any,
    ) -> NativeAssetContextV1:
        """Remove one reviewed reference atomically; replacement authority is unqualified."""
        import json

        from tldw_Server_API.app.api.v1.schemas.native_fork_schemas import NativeAssetContextUpdateV1

        self._db.native_forks._require_active_transaction(conn)
        if not isinstance(request, NativeAssetContextUpdateV1):
            raise ValueError("invalid_native_asset_context_request")
        request = NativeAssetContextUpdateV1.model_validate(request.model_dump(mode="python"))
        if request.action != "remove_from_context":
            raise ValueError("native_asset_context_action_unsupported")
        # Contain tail failures even when the caller catches them inside its larger transaction.
        conn.execute("SAVEPOINT native_asset_context_remove")
        try:
            current = self.read_native_asset_context(owner, conversation_id, conn=conn)
            target = next(
                (entry for entry in current.asset_manifest if entry.reference_id == request.target_reference_id), None
            )
            if (
                current.asset_manifest_revision != request.expected_asset_manifest_revision
                or target is None
                or target.reference_revision != request.expected_reference_revision
                or target.revision != request.expected_asset_revision
            ):
                raise ValueError("stale_native_asset_context")
            result = current
            if target.context_enabled:
                stamp = self._db._get_current_utc_timestamp_iso()
                changed = conn.execute(
                    "UPDATE native_chat_asset_references SET context_enabled = ?,revision = revision + 1,updated_at = ? "
                    "WHERE client_id = ? AND conversation_id = ? AND reference_id = ? AND revision = ? AND context_enabled = ?",
                    (
                        False,
                        stamp,
                        owner.client_id,
                        conversation_id,
                        target.reference_id,
                        int(target.reference_revision),
                        True,
                    ),
                )
                if changed.rowcount != 1:
                    raise ValueError("stale_native_asset_context")
                original = conn.execute(
                    "SELECT native_bundle_json FROM conversations WHERE client_id = ? AND id = ?",
                    (owner.client_id, conversation_id),
                ).fetchone()["native_bundle_json"]
                bundle = json.loads(original)
                bundle["asset_manifest_revision"] = self._db._generate_uuid()
                changed = conn.execute(
                    "UPDATE conversations SET native_bundle_json = ? WHERE client_id = ? AND id = ? "
                    "AND native_bundle_json = ? AND deleted = ?",
                    (
                        json.dumps(bundle, sort_keys=True, separators=(",", ":")),
                        owner.client_id,
                        conversation_id,
                        original,
                        False,
                    ),
                )
                if changed.rowcount != 1:
                    raise ValueError("stale_native_asset_context")
                self._db.message_store._advance_history_version(conn, conversation_id)
                result = self.read_native_asset_context(owner, conversation_id, conn=conn)
            conn.execute("RELEASE SAVEPOINT native_asset_context_remove")
            return result
        except BaseException:
            conn.execute("ROLLBACK TO SAVEPOINT native_asset_context_remove")
            conn.execute("RELEASE SAVEPOINT native_asset_context_remove")
            raise

    def _locked_candidate_for_operation(
        self, owner: Any, kind: str, operation_id: str, request_digest: str,
        candidate_id: str, *, conn: Any,
    ) -> tuple[dict[str, Any], Any]:
        if owner.client_id != self._db.client_id:
            raise ValueError("operation_owner_mismatch")
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
                raise NativeQuotaReleasePendingError("quota_release_unconfirmed")
            if released["owner_key"] != owner.owner_key or released["size_bytes"] != candidate["expected_size_bytes"]:
                raise ValueError("quota_intent_mismatch")
        updated = conn.execute(
            "UPDATE native_chat_asset_candidates SET state = 'discarded', updated_at = ? "
            "WHERE client_id = ? AND candidate_id = ? AND state = 'reclaiming' AND attempt_generation = ?",
            (self._db._get_current_utc_timestamp_iso(), owner.client_id, candidate_id, generation),
        )
        return updated.rowcount == 1
