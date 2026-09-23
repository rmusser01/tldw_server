"""Owner-bound durable operation receipts for native chat forks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

from tldw_Server_API.app.core.DB_Management.backends.base import BackendType

if TYPE_CHECKING:
    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB


class NativeForkStoreError(ValueError):
    """A stable native-fork storage failure that callers can inspect by code."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class NativeOperationResult:
    """Minimal durable result; terminal keys never become new reservations."""

    state: str
    child_conversation_id: str | None = None


@dataclass(frozen=True)
class NativeAttempt:
    """Generation-fenced preparation lease for one accepted operation."""

    generation: int
    lease_expires_at: str


class NativeForkStore:
    """Keep operation identity in the owning ChaCha transaction."""

    def __init__(self, db: CharactersRAGDB) -> None:
        self._db = db

    def _row(self, owner: Any, kind: str, operation_id: str, *, conn: Any) -> dict[str, Any] | None:
        """Lock and read an operation only within the caller's owned transaction."""
        query = (
            "SELECT * FROM native_chat_operations WHERE client_id = ? AND operation_kind = ? "
            "AND operation_id = ? FOR UPDATE"
            if self._db.backend_type == BackendType.POSTGRESQL else
            "SELECT * FROM native_chat_operations WHERE client_id = ? AND operation_kind = ? "
            "AND operation_id = ?"
        )
        result = conn.execute(
            query,
            (owner.client_id, kind, operation_id),
        ).fetchone()
        return dict(result) if result else None

    def lock_open_workspace(self, owner: Any, *, conn: Any) -> None:
        """After the operation lock, serialize native admission with workspace closure."""
        if owner.scope.kind == "global":
            return
        workspace_lock = " FOR UPDATE" if self._db.backend_type == BackendType.POSTGRESQL else ""
        workspace = conn.execute(
            "SELECT id FROM workspaces WHERE id = ? AND client_id = ? AND deleted = ? "
            "AND system_operation_state IS NULL AND native_chat_admission_closed = ?" + workspace_lock,  # nosec B608 - fixed backend-only lock suffix.
            (owner.scope.workspace_id, owner.client_id, False, False),
        ).fetchone()
        if workspace is None:
            raise NativeForkStoreError("workspace_native_unavailable")

    @staticmethod
    def _check_identity(row: dict[str, Any], owner: Any, digest: str, canonical_json: str | None = None) -> None:
        """Reject replay of a key with different owner, scope, or request bytes."""
        scope = owner.scope
        if (row["owner_key"] != owner.owner_key or row["scope_type"] != scope.kind
                or row["workspace_id"] != scope.workspace_id):
            raise NativeForkStoreError("operation_owner_mismatch")
        if row["request_digest"] != digest or (
            canonical_json is not None and row["canonical_request_json"] is not None
            and row["canonical_request_json"] != canonical_json
        ):
            raise NativeForkStoreError("operation_id_conflict")

    @staticmethod
    def _result(row: dict[str, Any] | None) -> NativeOperationResult:
        """Convert a stored operation row to its stable public receipt shape."""
        if row is None:
            return NativeOperationResult("not_recorded")
        return NativeOperationResult(row["state"], row["child_conversation_id"])

    def reserve_operation(
        self, owner: Any, kind: str, operation_id: str, request_digest: str,
        canonical_request: Any, *, conn: Any,
    ) -> NativeOperationResult:
        """Reserve one immutable owner/key/request identity; replay returns its receipt."""
        if kind not in {"native_fork_v1", "native_asset_retention_v1"}:
            raise NativeForkStoreError("unsupported_operation_kind")
        canonical_json = json.dumps(canonical_request, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
        row = self._row(owner, kind, operation_id, conn=conn)
        if row is not None:
            self._check_identity(row, owner, request_digest, canonical_json)
            return self.resolve_operation(owner, kind, operation_id, request_digest, conn=conn)
        now = self._db._get_current_utc_timestamp_iso()
        inserted = conn.execute(
            "INSERT INTO native_chat_operations "
            "(client_id, operation_kind, operation_id, owner_key, scope_type, workspace_id, "
            "request_digest, canonical_request_json, projection_version, state, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'native-fork-v1', 'preparing', ?, ?) "
            "ON CONFLICT (client_id, operation_kind, operation_id) DO NOTHING",
            (owner.client_id, kind, operation_id, owner.owner_key, owner.scope.kind,
             owner.scope.workspace_id, request_digest, canonical_json, now, now),
        )
        row = self._row(owner, kind, operation_id, conn=conn)
        if row is None:
            raise NativeForkStoreError("operation_reservation_missing")
        self._check_identity(row, owner, request_digest, canonical_json)
        if inserted.rowcount == 1:
            try:
                self.lock_open_workspace(owner, conn=conn)
            except ValueError:
                conn.execute(
                    "DELETE FROM native_chat_operations WHERE client_id = ? AND operation_kind = ? AND operation_id = ?",
                    (owner.client_id, kind, operation_id),
                )
                raise
        return self.resolve_operation(owner, kind, operation_id, request_digest, conn=conn)

    def resolve_operation(
        self, owner: Any, kind: str, operation_id: str, request_digest: str, *, conn: Any,
    ) -> NativeOperationResult:
        """Resolve accepted keys before looking at any mutable source conversation."""
        row = self._row(owner, kind, operation_id, conn=conn)
        if row is None:
            return NativeOperationResult("not_recorded")
        self._check_identity(row, owner, request_digest)
        child_id = row["child_conversation_id"]
        if row["state"] == "committed" and not child_id:
            raise NativeForkStoreError("operation_receipt_incomplete")
        if row["state"] == "committed" and child_id:
            child = conn.execute(
                "SELECT deleted, workspace_id, required_projection_version, "
                "native_creation_operation_kind, native_creation_operation_id "
                "FROM conversations WHERE id = ? AND client_id = ?",
                (child_id, owner.client_id),
            ).fetchone()
            if child is None or child["deleted"]:
                conn.execute(
                    "UPDATE native_chat_operations SET state = 'gone', result_json = NULL, "
                    "terminal_reason = 'child_deleted', updated_at = ? "
                    "WHERE client_id = ? AND operation_kind = ? AND operation_id = ?",
                    (self._db._get_current_utc_timestamp_iso(), owner.client_id, kind, operation_id),
                )
                row["state"] = "gone"
                row["result_json"] = None
            else:
                if (child["workspace_id"] != row["workspace_id"]
                        or child["required_projection_version"] != row["projection_version"]):
                    raise NativeForkStoreError("operation_child_scope_mismatch")
                if kind == "native_fork_v1" and (
                    child["native_creation_operation_kind"] != kind
                    or child["native_creation_operation_id"] != operation_id
                ):
                    raise NativeForkStoreError("operation_child_binding_mismatch")
                if row["workspace_id"] is not None:
                    workspace = conn.execute(
                        "SELECT deleted, native_chat_admission_closed, system_operation_state "
                        "FROM workspaces WHERE id = ? AND client_id = ?",
                        (row["workspace_id"], owner.client_id),
                    ).fetchone()
                    if (workspace is None or workspace["deleted"]
                            or workspace["native_chat_admission_closed"]
                            or workspace["system_operation_state"] is not None):
                        raise NativeForkStoreError("workspace_native_unavailable")
        return self._result(row)

    def claim_attempt(
        self, owner: Any, kind: str, operation_id: str, request_digest: str,
        *, now: datetime, conn: Any,
    ) -> NativeAttempt | NativeOperationResult:
        """Assign a generation only after the prior lease and candidates are settled."""
        if now.tzinfo is None or now.utcoffset() is None:
            raise NativeForkStoreError("attempt_now_must_be_aware")
        now = now.astimezone(timezone.utc)
        row = self._row(owner, kind, operation_id, conn=conn)
        if row is None:
            return NativeOperationResult("not_recorded")
        self._check_identity(row, owner, request_digest)
        if row["state"] != "preparing":
            return self.resolve_operation(owner, kind, operation_id, request_digest, conn=conn)
        lease_text = row["lease_expires_at"]
        if lease_text is not None:
            try:
                expiry = datetime.fromisoformat(lease_text)
            except ValueError as exc:
                raise NativeForkStoreError("operation_lease_invalid") from exc
            if expiry.tzinfo is None or expiry.utcoffset() is None:
                raise NativeForkStoreError("operation_lease_invalid")
            if expiry.astimezone(timezone.utc) > now:
                return NativeOperationResult("pending")
        self.lock_open_workspace(owner, conn=conn)
        candidate = conn.execute(
            "SELECT 1 FROM native_chat_asset_candidates WHERE client_id = ? "
            "AND operation_kind = ? AND operation_id = ? AND state <> 'discarded' LIMIT 1",
            (owner.client_id, kind, operation_id),
        ).fetchone()
        if candidate is not None:
            return NativeOperationResult("reconciliation_required")
        generation = row["attempt_generation"] + 1
        lease_expires_at = (now + timedelta(seconds=300)).isoformat()
        updated = conn.execute(
            "UPDATE native_chat_operations SET attempt_generation = ?, lease_expires_at = ?, "
            "updated_at = ? WHERE client_id = ? AND operation_kind = ? AND operation_id = ? "
            "AND state = 'preparing' AND attempt_generation = ?",
            (generation, lease_expires_at, now.isoformat(), owner.client_id, kind, operation_id,
             row["attempt_generation"]),
        )
        if updated.rowcount != 1:
            return NativeOperationResult("pending")
        return NativeAttempt(generation, lease_expires_at)

    def mark_child_gone(self, client_id: str, child_id: str, *, conn: Any) -> None:
        """Burn a child's creation receipt before deleting the child row."""
        query = (
            "SELECT operation_kind, operation_id FROM native_chat_operations "
            "WHERE client_id = ? AND child_conversation_id = ? AND state = 'committed' "
            "ORDER BY operation_kind, operation_id FOR UPDATE"
            if self._db.backend_type == BackendType.POSTGRESQL else
            "SELECT operation_kind, operation_id FROM native_chat_operations "
            "WHERE client_id = ? AND child_conversation_id = ? AND state = 'committed' "
            "ORDER BY operation_kind, operation_id"
        )
        rows = conn.execute(
            query,
            (client_id, child_id),
        ).fetchall()
        for row in rows:
            conn.execute(
                "UPDATE native_chat_operations SET state = 'gone', result_json = NULL, "
                "terminal_reason = 'child_deleted', updated_at = ? "
                "WHERE client_id = ? AND operation_kind = ? AND operation_id = ?",
                (self._db._get_current_utc_timestamp_iso(), client_id,
                 row["operation_kind"], row["operation_id"]),
            )
