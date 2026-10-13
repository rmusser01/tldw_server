"""Owner-bound durable operation receipts for native chat forks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

from tldw_Server_API.app.core.Chat.native_fork_schemas import (
    NativeAssetRetentionCommittedV1,
    NativeAssetRetentionRequestV1,
    NativeForkCommittedV1,
    NativeForkRequestV1,
)
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
    """Durable outcome; committed operations carry an immutable validated receipt."""

    state: str
    child_conversation_id: str | None = None
    receipt: NativeForkCommittedV1 | NativeAssetRetentionCommittedV1 | None = None


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
            if self._db.backend_type == BackendType.POSTGRESQL
            else "SELECT * FROM native_chat_operations WHERE client_id = ? AND operation_kind = ? "
            "AND operation_id = ?"
        )
        result = conn.execute(
            query,
            (owner.client_id, kind, operation_id),
        ).fetchone()
        return dict(result) if result else None

    def _require_active_transaction(self, conn: Any) -> None:
        """Require the actual owning transaction before atomic native installation."""
        if self._db.backend_type == BackendType.POSTGRESQL:
            state = self._db._connection_state()
            active = (
                getattr(conn, "_db", None) is self._db
                and getattr(conn, "_connection", None) is getattr(state, "conn", None)
                and getattr(state, "tx_depth", 0) > 0
            )
        else:
            active = conn is self._db.get_connection() and conn.in_transaction
        if not active:
            raise NativeForkStoreError("active_transaction_required")

    def lock_open_workspace(self, owner: Any, *, conn: Any) -> None:
        """After the operation lock, serialize native admission with workspace closure."""
        if owner.scope.kind == "global":
            return
        workspace_lock = " FOR UPDATE" if self._db.backend_type == BackendType.POSTGRESQL else ""
        workspace = conn.execute(
            "SELECT id FROM workspaces WHERE id = ? AND client_id = ? AND deleted = ? "
            "AND system_operation_state IS NULL AND native_chat_admission_closed = ?"
            + workspace_lock,  # nosec B608 - fixed backend-only lock suffix.
            (owner.scope.workspace_id, owner.client_id, False, False),
        ).fetchone()
        if workspace is None:
            raise NativeForkStoreError("workspace_native_unavailable")

    @staticmethod
    def _check_identity(row: dict[str, Any], owner: Any, digest: str, canonical_json: str | None = None) -> None:
        """Reject replay of a key with different owner, scope, or request bytes."""
        scope = owner.scope
        if (
            row["owner_key"] != owner.owner_key
            or row["scope_type"] != scope.kind
            or row["workspace_id"] != scope.workspace_id
        ):
            raise NativeForkStoreError("operation_owner_mismatch")
        if row["request_digest"] != digest or (
            canonical_json is not None
            and row["canonical_request_json"] is not None
            and row["canonical_request_json"] != canonical_json
        ):
            raise NativeForkStoreError("operation_id_conflict")

    @staticmethod
    def _result(row: dict[str, Any] | None) -> NativeOperationResult:
        """Convert a stored operation row to its stable public receipt shape."""
        if row is None:
            return NativeOperationResult("not_recorded")
        receipt = None
        if row["state"] == "committed" and row["operation_kind"] == "native_fork_v1":
            from tldw_Server_API.app.core.Chat.native_fork_projection import native_fork_request_digest

            try:
                request = NativeForkRequestV1.model_validate_json(row["canonical_request_json"])
                receipt = NativeForkCommittedV1.model_validate_json(row["result_json"])
                source_ids = {member.id for member in request.input.selection.messages}
                if (
                    native_fork_request_digest(request) != row["request_digest"]
                    or request.request_digest != row["request_digest"]
                    or request.operation_id != row["operation_id"]
                    or request.owner_key != row["owner_key"]
                    or request.scope.kind != row["scope_type"]
                    or request.scope.workspace_id != row["workspace_id"]
                    or request.source_conversation_id != row["source_conversation_id"]
                    or receipt.operation_id != row["operation_id"]
                    or receipt.owner_key != row["owner_key"]
                    or receipt.projection_version != row["projection_version"]
                    or receipt.child_id != row["child_conversation_id"]
                    or receipt.child_id == request.source_conversation_id
                    or set(receipt.message_map) != source_ids
                    or source_ids.intersection(receipt.message_map.values())
                ):
                    raise ValueError("receipt_binding_mismatch")
            except (ValueError, TypeError) as exc:
                raise NativeForkStoreError("operation_receipt_invalid") from exc
        elif row["state"] == "committed" and row["operation_kind"] == "native_asset_retention_v1":
            from tldw_Server_API.app.core.Chat.native_fork_projection import native_asset_retention_request_digest

            try:
                request = NativeAssetRetentionRequestV1.model_validate_json(row["canonical_request_json"])
                receipt = NativeAssetRetentionCommittedV1.model_validate_json(row["result_json"])
                if (
                    native_asset_retention_request_digest(request) != row["request_digest"]
                    or request.request_digest != row["request_digest"]
                    or request.operation_id != row["operation_id"]
                    or request.owner_key != row["owner_key"]
                    or request.scope.kind != row["scope_type"]
                    or request.scope.workspace_id != row["workspace_id"]
                    or request.source_conversation_id != row["source_conversation_id"]
                    or request.source_conversation_id != row["child_conversation_id"]
                    or receipt.operation_id != row["operation_id"]
                    or receipt.owner_key != row["owner_key"]
                    or receipt.projection_version != row["projection_version"]
                    or set(receipt.asset_map) != {asset.reference_id for asset in request.asset_manifest}
                ):
                    raise ValueError("receipt_binding_mismatch")
                descriptors = {}
                for asset in request.asset_manifest:
                    retained = receipt.asset_map[asset.reference_id]
                    identity = (retained.asset_id, retained.revision)
                    descriptor = (asset.hash, asset.representation, asset.fidelity_disposition)
                    if identity in descriptors and descriptors[identity] != descriptor:
                        raise ValueError("receipt_asset_alias_mismatch")
                    descriptors[identity] = descriptor
            except (ValueError, TypeError) as exc:
                raise NativeForkStoreError("operation_receipt_invalid") from exc
        return NativeOperationResult(row["state"], row["child_conversation_id"], receipt)

    def read_retention_source_revision(self, owner: Any, conversation_id: str, *, conn: Any) -> str:
        """Capture an ordinary source precondition, never native manifest readiness."""
        self._require_active_transaction(conn)
        if owner.client_id != self._db.client_id:
            raise NativeForkStoreError("operation_owner_mismatch")
        if not isinstance(conversation_id, str) or not 1 <= len(conversation_id) <= 256:
            raise NativeForkStoreError("invalid_native_asset_identity")
        if owner.owner_key is None:
            raise NativeForkStoreError("operation_owner_mismatch")
        namespace = self._db.message_store._history_owner_key(owner.client_id, owner.owner_key)
        self.lock_open_workspace(owner, conn=conn)
        lock = " FOR UPDATE" if self._db.backend_type == BackendType.POSTGRESQL else ""
        query = (
            "SELECT source_incarnation_id,scope_type,workspace_id,version,history_version,required_projection_version,"  # nosec B608 - fixed lock suffix; all values bound.
            "native_bundle_json,native_creation_operation_kind,native_creation_operation_id "
            "FROM conversations WHERE client_id = ? AND id = ? AND deleted = ?" + lock
        )
        params = (owner.client_id, conversation_id, False)
        row = conn.execute(query, params).fetchone()
        if row is None:
            raise NativeForkStoreError("native_retention_source_unavailable")
        before = dict(row)
        if not isinstance(before["source_incarnation_id"], str) or not before["source_incarnation_id"]:
            raise NativeForkStoreError("native_retention_source_unavailable")
        if (
            before["scope_type"] != owner.scope.kind
            or before["workspace_id"] != owner.scope.workspace_id
            or any(
                before[field] is not None
                for field in (
                    "required_projection_version",
                    "native_bundle_json",
                    "native_creation_operation_kind",
                    "native_creation_operation_id",
                )
            )
        ):
            raise NativeForkStoreError("native_retention_source_unsupported")
        snapshot, _ = self._db.message_store._read_history_snapshot(
            conversation_id,
            owner_client_id=owner.client_id,
            owner_key=namespace,
            projection_id=None,
            conn=conn,
        )
        after = conn.execute(query, params).fetchone()
        if after is None or dict(after) != before:
            raise NativeForkStoreError("stale_retention_source")
        if (
            conn.execute(
                "SELECT 1 FROM native_chat_asset_claims WHERE client_id = ? AND conversation_id = ? "
                "UNION ALL SELECT 1 FROM native_chat_asset_references WHERE client_id = ? AND conversation_id = ? LIMIT 1",
                (owner.client_id, conversation_id, owner.client_id, conversation_id),
            ).fetchone()
            is not None
        ):
            raise NativeForkStoreError("native_retention_source_unsupported")
        return self._db.message_store._history_digest(
            [
                "tldw-ordinary-retention-source",
                1,
                owner.client_id,
                namespace,
                [owner.scope.kind, owner.scope.workspace_id],
                conversation_id,
                before["source_incarnation_id"],
                [snapshot.fences.conversation, snapshot.fences.history, snapshot.fences.settings],
                snapshot.source_digest,
                snapshot.storage_context_digest,
            ]
        )

    def reserve_retention_operation(
        self,
        owner: Any,
        request: NativeAssetRetentionRequestV1,
        *,
        conn: Any,
    ) -> NativeOperationResult:
        """Reserve frozen metadata; lookup recovery precedes source re-admission."""
        from tldw_Server_API.app.core.Chat.native_fork_projection import native_asset_retention_request_digest

        self._require_active_transaction(conn)
        if not isinstance(request, NativeAssetRetentionRequestV1):
            raise NativeForkStoreError("invalid_retention_request")
        request = NativeAssetRetentionRequestV1.model_validate(request.model_dump(mode="json"))
        if owner.client_id != self._db.client_id or request.owner_key != owner.owner_key or request.scope != owner.scope:
            raise NativeForkStoreError("operation_owner_mismatch")
        if native_asset_retention_request_digest(request) != request.request_digest:
            raise NativeForkStoreError("operation_id_conflict")
        canonical = json.dumps(
            request.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        )
        kind = "native_asset_retention_v1"
        conn.execute("SAVEPOINT native_retention_reservation")
        try:
            row = self._row(owner, kind, request.operation_id, conn=conn)
            if row is None:
                now = self._db._get_current_utc_timestamp_iso()
                conn.execute(
                    "INSERT INTO native_chat_operations "
                    "(client_id,operation_kind,operation_id,owner_key,scope_type,workspace_id,request_digest,"
                    "canonical_request_json,projection_version,state,source_conversation_id,created_at,updated_at) "
                    "VALUES (?,?,?,?,?,?,?,?,'native-fork-v1','preparing',?,?,?) "
                    "ON CONFLICT (client_id,operation_kind,operation_id) DO NOTHING",
                    (
                        owner.client_id,
                        kind,
                        request.operation_id,
                        owner.owner_key,
                        owner.scope.kind,
                        owner.scope.workspace_id,
                        request.request_digest,
                        canonical,
                        request.source_conversation_id,
                        now,
                        now,
                    ),
                )
                row = self._row(owner, kind, request.operation_id, conn=conn)
            if row is None:
                raise NativeForkStoreError("operation_reservation_missing")
            self._check_identity(row, owner, request.request_digest, canonical)
            result = self.resolve_operation(owner, kind, request.operation_id, request.request_digest, conn=conn)
            if result.state == "preparing":
                if (
                    row["canonical_request_json"] != canonical
                    or row["source_conversation_id"] != request.source_conversation_id
                    or row["projection_version"] != "native-fork-v1"
                ):
                    raise NativeForkStoreError("retention_source_binding_mismatch")
                current = self.read_retention_source_revision(owner, request.source_conversation_id, conn=conn)
                if current != request.expected_asset_manifest_revision:
                    raise NativeForkStoreError("stale_retention_source")
            conn.execute("RELEASE SAVEPOINT native_retention_reservation")
            return result
        except BaseException:
            conn.execute("ROLLBACK TO SAVEPOINT native_retention_reservation")
            conn.execute("RELEASE SAVEPOINT native_retention_reservation")
            raise

    def reserve_operation(
        self,
        owner: Any,
        kind: str,
        operation_id: str,
        request_digest: str,
        canonical_request: Any,
        *,
        conn: Any,
    ) -> NativeOperationResult:
        """Reserve one immutable owner/key/request identity; replay returns its receipt."""
        if kind not in {"native_fork_v1", "native_asset_retention_v1"}:
            raise NativeForkStoreError("unsupported_operation_kind")
        canonical_json = json.dumps(
            canonical_request, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        )
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
            (
                owner.client_id,
                kind,
                operation_id,
                owner.owner_key,
                owner.scope.kind,
                owner.scope.workspace_id,
                request_digest,
                canonical_json,
                now,
                now,
            ),
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
        self,
        owner: Any,
        kind: str,
        operation_id: str,
        request_digest: str,
        *,
        conn: Any,
    ) -> NativeOperationResult:
        """Resolve accepted keys before looking at any mutable source conversation."""
        row = self._row(owner, kind, operation_id, conn=conn)
        if row is None:
            return NativeOperationResult("not_recorded")
        self._check_identity(row, owner, request_digest)
        if row["state"] == "committed" and kind == "native_asset_retention_v1":
            # Refuse a substituted retention target before lifecycle mutation.
            self._result(row)
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
                if (
                    child["workspace_id"] != row["workspace_id"]
                    or child["required_projection_version"] != row["projection_version"]
                ):
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
                    if (
                        workspace is None
                        or workspace["deleted"]
                        or workspace["native_chat_admission_closed"]
                        or workspace["system_operation_state"] is not None
                    ):
                        raise NativeForkStoreError("workspace_native_unavailable")
        return self._result(row)

    def claim_retention_attempt(
        self,
        owner: Any,
        request: NativeAssetRetentionRequestV1,
        *,
        conn: Any,
    ) -> NativeAttempt | NativeOperationResult:
        """Grant preparation only after original-source admission; publish no assets."""
        self._require_active_transaction(conn)
        conn.execute("SAVEPOINT native_retention_attempt")
        try:
            result = self.reserve_retention_operation(owner, request, conn=conn)
            if result.state == "preparing":
                # Lock waits and source reads must not consume the new lease's lifetime.
                now = datetime.now(timezone.utc)
                result = self.claim_attempt(
                    owner,
                    "native_asset_retention_v1",
                    request.operation_id,
                    request.request_digest,
                    now=now,
                    conn=conn,
                )
            conn.execute("RELEASE SAVEPOINT native_retention_attempt")
            return result
        except BaseException:
            conn.execute("ROLLBACK TO SAVEPOINT native_retention_attempt")
            conn.execute("RELEASE SAVEPOINT native_retention_attempt")
            raise

    def claim_attempt(
        self,
        owner: Any,
        kind: str,
        operation_id: str,
        request_digest: str,
        *,
        now: datetime,
        conn: Any,
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
        if type(row["attempt_generation"]) is not int or row["attempt_generation"] < 0:
            raise NativeForkStoreError("operation_generation_invalid")
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
            (
                generation,
                lease_expires_at,
                now.isoformat(),
                owner.client_id,
                kind,
                operation_id,
                row["attempt_generation"],
            ),
        )
        if updated.rowcount != 1:
            return NativeOperationResult("pending")
        return NativeAttempt(generation, lease_expires_at)

    def commit_fork(
        self,
        owner: Any,
        request: NativeForkRequestV1,
        attempt: NativeAttempt,
        *,
        now: datetime,
        conn: Any,
    ) -> NativeOperationResult:
        """Insert an asset-free neutral retained child in the caller's transaction.

        Resolve original receipts before mutable-source or attempt checks. Assets
        and accepted character bindings require their later qualified adapters;
        this private subset grants no public native-fork capability.
        """
        from tldw_Server_API.app.core.Chat.history_selection import _wire_json
        from tldw_Server_API.app.core.Chat.native_fork_projection import (
            native_fork_request_digest,
            project_native_fork_context,
            project_native_fork_messages,
        )

        self._require_active_transaction(conn)
        request = NativeForkRequestV1.model_validate(request.model_dump(mode="json"))
        if (
            request.owner_key != owner.owner_key
            or request.scope != owner.scope
            or owner.client_id != self._db.client_id
        ):
            raise NativeForkStoreError("operation_owner_mismatch")
        if native_fork_request_digest(request) != request.request_digest:
            raise NativeForkStoreError("operation_id_conflict")
        canonical = json.dumps(
            request.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        )
        row = self._row(owner, "native_fork_v1", request.operation_id, conn=conn)
        if row is None:
            return NativeOperationResult("not_recorded")
        self._check_identity(row, owner, request.request_digest, canonical)
        result = self.resolve_operation(
            owner, "native_fork_v1", request.operation_id, request.request_digest, conn=conn
        )
        if result.state != "preparing":
            return result
        if now.tzinfo is None or now.utcoffset() is None:
            raise NativeForkStoreError("attempt_now_must_be_aware")
        try:
            expiry = datetime.fromisoformat(row["lease_expires_at"])
        except (TypeError, ValueError) as exc:
            raise NativeForkStoreError("operation_lease_invalid") from exc
        if expiry.tzinfo is None or expiry.utcoffset() is None:
            raise NativeForkStoreError("operation_lease_invalid")
        if (
            type(attempt.generation) is not int
            or attempt.generation < 1
            or attempt.generation != row["attempt_generation"]
            or attempt.lease_expires_at != row["lease_expires_at"]
            or expiry <= now.astimezone(timezone.utc)
        ):
            raise NativeForkStoreError("stale_attempt")
        self.lock_open_workspace(owner, conn=conn)
        candidate = conn.execute(
            "SELECT 1 FROM native_chat_asset_candidates WHERE client_id = ? "
            "AND operation_kind = 'native_fork_v1' AND operation_id = ? AND state <> 'discarded' LIMIT 1",
            (owner.client_id, request.operation_id),
        ).fetchone()
        if candidate is not None:
            raise NativeForkStoreError("reconciliation_required")
        if request.asset_manifest:
            raise NativeForkStoreError("unsupported_native_fork_assets")
        selection = request.input.selection
        view = {
            "view_session_id": "native-fork-commit-v1",
            "owner_key": owner.owner_key,
            "conversation_id": request.source_conversation_id,
            "interpretation": selection.interpretation.model_dump(mode="json"),
            "cursor": selection.cursor.model_dump(mode="json"),
            "selection_revision": selection.selection_revision,
        }
        state, _, rows = self._db.message_store.read_native_fork_source(
            view,
            owner_client_id=owner.client_id,
            owner_key=owner.owner_key,
            scope_type=owner.scope.kind,
            workspace_id=owner.scope.workspace_id,
            conn=conn,
        )
        source = conn.execute(
            "SELECT required_projection_version, native_bundle_json, native_creation_operation_kind, "
            "native_creation_operation_id FROM conversations WHERE id = ? AND client_id = ?",
            (request.source_conversation_id, owner.client_id),
        ).fetchone()
        if source is None or any(
            source[key] is not None
            for key in (
                "required_projection_version",
                "native_bundle_json",
                "native_creation_operation_kind",
                "native_creation_operation_id",
            )
        ):
            raise NativeForkStoreError("unsupported_native_fork_source_bundle")
        carrier = conn.execute(
            "SELECT 1 FROM native_chat_asset_claims WHERE client_id = ? AND conversation_id = ? "
            "UNION ALL SELECT 1 FROM native_chat_asset_references WHERE client_id = ? AND conversation_id = ? LIMIT 1",
            (owner.client_id, request.source_conversation_id, owner.client_id, request.source_conversation_id),
        ).fetchone()
        if carrier is not None or any(item.get("images") or item.get("assets") for item in rows):
            raise NativeForkStoreError("unsupported_native_fork_assets")
        projected = project_native_fork_context(state, rows, ())
        if projected.binding is not None:
            raise NativeForkStoreError("unsupported_native_fork_character_binding")
        messages = project_native_fork_messages(rows, context=projected)
        if projected.retained_context_digest != request.retained_context_digest or [
            (item["id"], item["revision"]) for item in messages
        ] != [(member.id, member.revision) for member in selection.messages]:
            raise NativeForkStoreError("stale_retained_source")
        child_id = self._db._generate_uuid()
        message_map = {item["id"]: self._db._generate_uuid() for item in messages}
        call_map = {call["id"]: self._db._generate_uuid() for item in messages for call in item["tool_calls"]}
        inserted = self._db.add_conversation(
            {
                "id": child_id,
                "root_id": child_id,
                "title": request.child_title,
                "character_id": None,
                "client_id": owner.client_id,
                "scope_type": owner.scope.kind,
                "workspace_id": owner.scope.workspace_id,
            },
            conn=conn,
        )
        if inserted != child_id:
            raise NativeForkStoreError("child_write_failed")
        parent = None
        for item in messages:
            mid = message_map[item["id"]]
            extra = _wire_json(item["extra_metadata"])
            calls = _wire_json(item["tool_calls"])
            for call in calls:
                call["id"] = call_map[call["id"]]
            if "tool_call_id" in extra:
                extra["tool_call_id"] = call_map[extra["tool_call_id"]]
            text = item["message"]
            if not text and item["role"] == "tool":
                text = "[empty tool result]"
                extra["content_placeholder_reason"] = "empty_tool_result"
            if (
                self._db.add_message(
                    {
                        "id": mid,
                        "conversation_id": child_id,
                        "sender": item["role"],
                        "content": text or "[tool_call]",
                        "parent_message_id": parent,
                        "client_id": owner.client_id,
                    },
                    conn=conn,
                )
                != mid
            ):
                raise NativeForkStoreError("message_write_failed")
            if not text:
                # The public writer requires text; closed replay may truthfully retain an empty assistant.
                if (
                    conn.execute(
                        "UPDATE messages SET content = '' WHERE id = ? AND conversation_id = ?", (mid, child_id)
                    ).rowcount
                    != 1
                ):
                    raise NativeForkStoreError("message_write_failed")
            if (calls or extra) and not self._db.add_message_metadata(
                mid, tool_calls=calls or None, extra=extra or None, conn=conn
            ):
                raise NativeForkStoreError("metadata_write_failed")
            authority = {
                "version": 1,
                "interpretation": {"kind": "parent_graph_v1"},
                "settled": True,
                "native_fork": {
                    "operation_id": request.operation_id,
                    "owner_key": owner.owner_key,
                    "conversation_id": child_id,
                },
            }
            if (
                conn.execute(
                    "UPDATE messages SET history_admission_json = ? WHERE id = ? AND conversation_id = ?",
                    (json.dumps(authority, sort_keys=True, separators=(",", ":")), mid, child_id),
                ).rowcount
                != 1
            ):
                raise NativeForkStoreError("provenance_write_failed")
            parent = mid
        settings = json.loads(projected.settings_json)
        if "pinnedMessageIds" in settings:
            settings["pinnedMessageIds"] = [message_map[mid] for mid in settings["pinnedMessageIds"]]
        if not self._db.upsert_conversation_settings(child_id, settings, conn=conn, expected_settings_version=0):
            raise NativeForkStoreError("settings_write_failed")
        bundle = {
            "version": 1,
            "state": "ready",
            "required_projection_version": "native-fork-v1",
            "settings_revision": 1,
            "asset_manifest_revision": "1",
        }
        if (
            conn.execute(
                "UPDATE conversations SET required_projection_version = 'native-fork-v1', native_creation_operation_kind = 'native_fork_v1', "
                "native_creation_operation_id = ?, native_bundle_json = ? WHERE id = ? AND client_id = ?",
                (
                    request.operation_id,
                    json.dumps(bundle, sort_keys=True, separators=(",", ":")),
                    child_id,
                    owner.client_id,
                ),
            ).rowcount
            != 1
        ):
            raise NativeForkStoreError("child_protection_write_failed")
        receipt = NativeForkCommittedV1(
            state="committed",
            operation_id=request.operation_id,
            owner_key=owner.owner_key,
            child_id=child_id,
            message_map=message_map,
        )
        if (
            conn.execute(
                "UPDATE native_chat_operations SET state = 'committed', source_conversation_id = ?, child_conversation_id = ?, "
                "result_json = ?, updated_at = ? WHERE client_id = ? AND operation_kind = 'native_fork_v1' "
                "AND operation_id = ? AND state = 'preparing' AND attempt_generation = ? AND lease_expires_at = ?",
                (
                    request.source_conversation_id,
                    child_id,
                    receipt.model_dump_json(),
                    self._db._get_current_utc_timestamp_iso(),
                    owner.client_id,
                    request.operation_id,
                    attempt.generation,
                    attempt.lease_expires_at,
                ),
            ).rowcount
            != 1
        ):
            raise NativeForkStoreError("operation_commit_conflict")
        return NativeOperationResult("committed", child_id, receipt)

    def mark_child_gone(self, client_id: str, child_id: str, *, conn: Any) -> None:
        """Burn a child's creation receipt before deleting the child row."""
        query = (
            "SELECT operation_kind, operation_id FROM native_chat_operations "
            "WHERE client_id = ? AND child_conversation_id = ? AND state = 'committed' "
            "ORDER BY operation_kind, operation_id FOR UPDATE"
            if self._db.backend_type == BackendType.POSTGRESQL
            else "SELECT operation_kind, operation_id FROM native_chat_operations "
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
                (self._db._get_current_utc_timestamp_iso(), client_id, row["operation_kind"], row["operation_id"]),
            )
