"""Owner-bound SQL seam for Notes graph suggestion persistence."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import sqlite3
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Literal, TypeVar

from tldw_Server_API.app.core.Notes_Graph.suggestion_content import content_fingerprint

from ..ChaChaNotes_DB import BackendConnectionWrapper, BackendType
from .note_graph_suggestion_models import (
    NoteGraphSuggestion,
    NoteGraphSuggestionKind,
    NoteGraphSuggestionOperationKind,
    NoteGraphSuggestionRejectionSet,
    NoteGraphSuggestionRun,
    NoteGraphSuggestionRunState,
    NoteGraphSuggestionState,
)

if TYPE_CHECKING:
    from ..ChaChaNotes_DB import CharactersRAGDB


SuggestionConnection = sqlite3.Connection | BackendConnectionWrapper
SuggestionReadT = TypeVar("SuggestionReadT")


class NotesGraphSourceTooLargeError(ValueError):
    """Raised when a source note exceeds the non-truncating analysis limit."""


class NotesGraphFTSNotReadyError(RuntimeError):
    """Raised when Notes FTS structures are unavailable or structurally incomplete."""


class NotesGraphDatasetScopeError(RuntimeError):
    """Raised when an owner is not authorized for the requested Notes dataset."""


@dataclass(frozen=True, slots=True)
class RunAdmissionResult:
    """Durable admission or replay outcome without Jobs-side details."""

    disposition: Literal["created", "in_progress", "terminal_replay"]
    run: NoteGraphSuggestionRun
    continuation: str | None = None
    replay_envelope: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class SuggestionPage:
    """One stable keyset page of visible suggestions."""

    items: tuple[NoteGraphSuggestion, ...]
    next_cursor: str | None


@dataclass(frozen=True, slots=True)
class MutationResult:
    """Receipt-backed suggestion mutation outcome."""

    disposition: Literal["completed", "terminal_replay", "in_progress"]
    envelope: dict[str, Any]
    rejection_set: Any | None = None
    suggestion: NoteGraphSuggestion | None = None
    continuation: str | None = None


@dataclass(frozen=True, slots=True)
class SuggestionNoteRecord:
    """A bounded owner-scoped note payload returned after a SQL byte predicate."""

    note_id: str
    title: str
    content: str
    version: int
    last_modified: str


class NoteGraphSuggestionStore:
    """Own the database boundary for future suggestion reads and transitions."""

    def __init__(self, db: CharactersRAGDB) -> None:
        self._db = db

    @property
    def owner_user_id(self) -> str:
        """Return the ChaChaNotes owner bound to this store instance."""
        return str(self._db.client_id)

    @property
    def is_postgres(self) -> bool:
        """Return whether this store is backed by PostgreSQL."""
        return self._db.backend_type == BackendType.POSTGRESQL

    def _scope(self, dataset_id: str) -> str:
        dataset = str(dataset_id).strip()
        if not dataset:
            raise NotesGraphDatasetScopeError("notes_graph_dataset_scope_invalid")
        return dataset

    def _deleted_value(self) -> bool | int:
        return False if self.is_postgres else 0

    def _db_datetime(self, value: datetime) -> datetime | str:
        aware = self._aware_utc(value)
        return aware if self.is_postgres else aware.isoformat()

    @staticmethod
    def _aware_utc(value: datetime) -> datetime:
        if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("notes_graph_timestamp_invalid")
        return value.astimezone(timezone.utc)

    @staticmethod
    def _iso(value: object | None) -> str | None:
        if value is None:
            return None
        if isinstance(value, datetime):
            parsed = value
        else:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc).isoformat()

    def _set_dataset_scope(self, conn: SuggestionConnection, dataset_id: str) -> None:
        if self.is_postgres:
            conn.execute("SELECT set_config('app.current_dataset_id', ?, true)", (dataset_id,))
        self._require_dataset_scope(conn, dataset_id)

    def _with_dataset_scope(
        self,
        dataset_id: str,
        fn: Callable[[SuggestionConnection], SuggestionReadT],
    ) -> SuggestionReadT:
        if not self.is_postgres:
            with self._db.transaction() as conn:
                self._set_dataset_scope(conn, dataset_id)
                return fn(conn)
        with self._db.transaction() as conn:
            self._set_dataset_scope(conn, dataset_id)
            return fn(conn)

    def _require_dataset_scope(self, conn: SuggestionConnection, dataset_id: str) -> None:
        row = conn.execute(
            "SELECT 1 FROM note_task_scope_authority "
            "WHERE owner_user_id = ? AND dataset_id = ?",
            (self.owner_user_id, dataset_id),
        ).fetchone()
        if row is None:
            raise NotesGraphDatasetScopeError("notes_graph_dataset_scope_invalid")

    def _source_byte_expression(self) -> str:
        if self.is_postgres:
            return "octet_length(COALESCE(n.title, '')) + octet_length(COALESCE(n.content, ''))"
        return "length(CAST(COALESCE(n.title, '') AS BLOB)) + length(CAST(COALESCE(n.content, '') AS BLOB))"

    @staticmethod
    def idempotency_key_digest(idempotency_key: str) -> str:
        """Hash one bounded key so raw idempotency material is never persisted."""

        if (
            not isinstance(idempotency_key, str)
            or not idempotency_key.strip()
            or len(idempotency_key.encode("utf-8")) > 256
        ):
            raise ValueError("notes_graph_idempotency_key_invalid")
        material = b"notes-graph-idempotency-key-v1\0" + idempotency_key.encode("utf-8")
        return f"sha256:{hashlib.sha256(material).hexdigest()}"

    @staticmethod
    def canonical_request_fingerprint(operation_kind: str, fields: dict[str, Any]) -> str:
        """Hash bounded canonical non-content request fields with an explicit version."""

        forbidden_keys = {
            "content",
            "credential",
            "credentials",
            "excerpt",
            "note_text",
            "password",
            "prompt",
            "secret",
            "text",
            "token",
        }

        def validate(value: Any, path: tuple[str, ...] = ()) -> None:
            if isinstance(value, dict):
                for key, child in value.items():
                    normalized = str(key).lower()
                    if normalized in forbidden_keys or normalized.endswith(("_secret", "_credential", "_token")):
                        raise ValueError("notes_graph_request_contains_sensitive_field")
                    validate(child, (*path, normalized))
            elif isinstance(value, (list, tuple)):
                for child in value:
                    validate(child, path)
            elif value is not None and not isinstance(value, (str, int, float, bool)):
                raise ValueError("notes_graph_request_fingerprint_invalid")

        if not isinstance(operation_kind, str) or not operation_kind.strip() or not isinstance(fields, dict):
            raise ValueError("notes_graph_request_fingerprint_invalid")
        validate(fields)
        encoded = json.dumps(
            {"v": 1, "operation": operation_kind, "fields": fields},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        if len(encoded) > 4096:
            raise ValueError("notes_graph_request_fingerprint_too_large")
        digest = hashlib.sha256(b"notes-graph-request-v1\0" + encoded).hexdigest()
        return f"sha256:{digest}"

    @staticmethod
    def _decode_envelope(value: object | None) -> dict[str, Any] | None:
        if value is None:
            return None
        decoded = json.loads(str(value))
        if not isinstance(decoded, dict):
            raise RuntimeError("notes_graph_receipt_envelope_invalid")
        return decoded

    @staticmethod
    def _encode_envelope(value: dict[str, Any]) -> str:
        allowed_keys = {
            "accepted_resource_identity",
            "cleared_count",
            "count",
            "error_code",
            "guidance_key",
            "invalid_item_count",
            "operation_id",
            "rejection_set_revision",
            "related_note_count",
            "revision",
            "run_id",
            "source_note_id",
            "state",
            "suggestion_count",
            "suggestion_id",
            "tag_count",
        }
        if any(key not in allowed_keys for key in value) or any(
            isinstance(item, str) and len(item.encode("utf-8")) > 256
            or item is not None and not isinstance(item, (str, int, bool))
            for item in value.values()
        ):
            raise ValueError("notes_graph_replay_envelope_invalid")
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        if len(encoded.encode("ascii")) > 4096:
            raise ValueError("notes_graph_replay_envelope_too_large")
        return encoded

    def _run_from_row(self, row: Any) -> NoteGraphSuggestionRun:
        return NoteGraphSuggestionRun(
            id=str(row["id"]),
            owner_user_id=str(row["owner_user_id"]),
            dataset_id=str(row["dataset_id"]),
            source_note_id=str(row["source_note_id"]),
            source_fingerprint=str(row["source_fingerprint"]),
            state=NoteGraphSuggestionRunState(str(row["state"])),
            revision=int(row["revision"]),
            created_at=self._iso(row["created_at"]) or "",
            expires_at=self._iso(row["expires_at"]) or "",
            admission_receipt_id=row["admission_receipt_id"],
            provider=row["provider"],
            model=row["model"],
            capability_revision=row["capability_revision"],
            prompt_contract_version=row["prompt_contract_version"],
            job_id=row["job_id"],
            expected_completion_token=row["expected_completion_token"],
            result_digest=row["result_digest"],
            suggestion_count=int(row["suggestion_count"]),
            related_note_count=int(row["related_note_count"]),
            tag_count=int(row["tag_count"]),
            invalid_item_count=int(row["invalid_item_count"]),
            error_code=row["error_code"],
            guidance_key=row["guidance_key"],
            started_at=self._iso(row["started_at"]),
            completed_at=self._iso(row["completed_at"]),
        )

    def _load_run(self, conn: SuggestionConnection, dataset_id: str, run_id: str) -> NoteGraphSuggestionRun:
        row = conn.execute(
            "SELECT * FROM note_graph_suggestion_runs WHERE owner_user_id=? AND dataset_id=? AND id=?",
            (self.owner_user_id, dataset_id, run_id),
        ).fetchone()
        if row is None:
            raise RuntimeError("notes_graph_run_not_found")
        return self._run_from_row(row)

    def admit_run(
        self,
        *,
        dataset_id: str,
        source_note_id: str,
        source_fingerprint: str,
        provider: str,
        model: str,
        capability_revision: str,
        prompt_contract_version: str,
        idempotency_key: str,
        now: datetime,
    ) -> RunAdmissionResult:
        """Create or replay one durable unbound run without calling Jobs."""

        dataset = self._scope(dataset_id)
        now_utc = self._aware_utc(now)
        key_digest = self.idempotency_key_digest(idempotency_key)
        request_fingerprint = self.canonical_request_fingerprint(
            "run_admit",
            {
                "source_note_id": source_note_id,
                "source_fingerprint": source_fingerprint,
                "provider": provider,
                "model": model,
                "capability_revision": capability_revision,
                "prompt_contract_version": prompt_contract_version,
            },
        )

        def mutate(conn: SuggestionConnection) -> RunAdmissionResult:
            receipt = conn.execute(
                "SELECT * FROM note_graph_suggestion_operation_receipts "
                "WHERE owner_user_id=? AND dataset_id=? AND operation_kind='run_admit' "
                "AND resource_identity=? AND idempotency_key_digest=?",
                (self.owner_user_id, dataset, source_note_id, key_digest),
            ).fetchone()
            if receipt is not None:
                if not secrets.compare_digest(str(receipt["request_fingerprint"]), request_fingerprint):
                    raise RuntimeError("notes_graph_suggestion_idempotency_mismatch")
                run_row = conn.execute(
                    "SELECT * FROM note_graph_suggestion_runs WHERE owner_user_id=? AND dataset_id=? "
                    "AND admission_receipt_id=?",
                    (self.owner_user_id, dataset, receipt["id"]),
                ).fetchone()
                if run_row is None:
                    raise RuntimeError("notes_graph_receipt_run_missing")
                run = self._run_from_row(run_row)
                if str(receipt["state"]) in {"completed", "failed"}:
                    return RunAdmissionResult(
                        disposition="terminal_replay",
                        run=run,
                        replay_envelope=self._decode_envelope(receipt["replay_envelope"]),
                    )
                return RunAdmissionResult(
                    disposition="in_progress",
                    run=run,
                    continuation="resume_run_admission",
                )

            if self._current_note_fingerprint(conn, source_note_id) != source_fingerprint:
                raise RuntimeError("notes_graph_fingerprint_stale")

            active = conn.execute(
                "SELECT id FROM note_graph_suggestion_runs WHERE owner_user_id=? AND dataset_id=? "
                "AND source_note_id=? AND source_fingerprint=? AND provider=? AND model=? "
                "AND prompt_contract_version=? "
                "AND state IN ('admitting','queued','running','cancelling','publishing') LIMIT 1",
                (
                    self.owner_user_id,
                    dataset,
                    source_note_id,
                    source_fingerprint,
                    provider,
                    model,
                    prompt_contract_version,
                ),
            ).fetchone()
            if active is not None:
                raise RuntimeError("notes_graph_active_run_conflict")

            receipt_id = str(uuid.uuid4())
            run_id = str(uuid.uuid4())
            conn.execute(
                "INSERT INTO note_graph_suggestion_operation_receipts("
                "id,operation_kind,owner_user_id,dataset_id,source_note_id,resource_identity,"
                "idempotency_key_digest,request_fingerprint,state,created_at,expires_at"
                ") VALUES (?,'run_admit',?,?,?,?,?,?, 'in_progress',?,?)",
                (
                    receipt_id,
                    self.owner_user_id,
                    dataset,
                    source_note_id,
                    source_note_id,
                    key_digest,
                    request_fingerprint,
                    self._db_datetime(now_utc),
                    self._db_datetime(now_utc + timedelta(days=90)),
                ),
            )
            conn.execute(
                "INSERT INTO note_graph_suggestion_runs("
                "id,owner_user_id,dataset_id,source_note_id,source_fingerprint,admission_receipt_id,"
                "provider,model,capability_revision,prompt_contract_version,state,revision,created_at,expires_at"
                ") VALUES (?,?,?,?,?,?,?,?,?,?,'admitting',1,?,?)",
                (
                    run_id,
                    self.owner_user_id,
                    dataset,
                    source_note_id,
                    source_fingerprint,
                    receipt_id,
                    provider,
                    model,
                    capability_revision,
                    prompt_contract_version,
                    self._db_datetime(now_utc),
                    self._db_datetime(now_utc + timedelta(days=30)),
                ),
            )
            return RunAdmissionResult(
                disposition="created",
                run=self._load_run(conn, dataset, run_id),
                continuation="resume_run_admission",
            )

        return self._with_dataset_scope(dataset, mutate)

    def bind_admitted_run(
        self,
        *,
        dataset_id: str,
        run_id: str,
        expected_state: str,
        expected_revision: int,
        job_id: str,
        completion_token: str,
        replay_envelope: dict[str, Any],
        now: datetime,
    ) -> NoteGraphSuggestionRun:
        """Bind one Jobs identity and terminalize its admission receipt atomically."""

        dataset = self._scope(dataset_id)
        now_utc = self._aware_utc(now)
        envelope = self._encode_envelope(replay_envelope)

        def mutate(conn: SuggestionConnection) -> NoteGraphSuggestionRun:
            updated = conn.execute(
                "UPDATE note_graph_suggestion_runs SET job_id=?,expected_completion_token=?,"
                "state='queued',revision=revision+1 WHERE owner_user_id=? AND dataset_id=? AND id=? "
                "AND state=? AND revision=?",
                (
                    job_id,
                    completion_token,
                    self.owner_user_id,
                    dataset,
                    run_id,
                    expected_state,
                    expected_revision,
                ),
            )
            if updated.rowcount != 1:
                raise RuntimeError("notes_graph_run_conflict")
            run = self._load_run(conn, dataset, run_id)
            receipt = conn.execute(
                "UPDATE note_graph_suggestion_operation_receipts SET state='completed',http_status=202,"
                "replay_envelope=?,completed_at=?,expires_at=? WHERE owner_user_id=? AND dataset_id=? "
                "AND id=? AND state='in_progress' AND request_fingerprint=(SELECT request_fingerprint "
                "FROM note_graph_suggestion_operation_receipts WHERE owner_user_id=? AND dataset_id=? AND id=?)",
                (
                    envelope,
                    self._db_datetime(now_utc),
                    self._db_datetime(now_utc + timedelta(days=90)),
                    self.owner_user_id,
                    dataset,
                    run.admission_receipt_id,
                    self.owner_user_id,
                    dataset,
                    run.admission_receipt_id,
                ),
            )
            if receipt.rowcount != 1:
                raise RuntimeError("notes_graph_receipt_conflict")
            return run

        return self._with_dataset_scope(dataset, mutate)

    def transition_run(
        self,
        *,
        dataset_id: str,
        run_id: str,
        expected_state: str,
        expected_revision: int,
        new_state: str,
        now: datetime,
        error_code: str | None = None,
    ) -> NoteGraphSuggestionRun:
        """Apply one owner/dataset/state/revision fenced run transition."""

        dataset = self._scope(dataset_id)
        now_utc = self._aware_utc(now)
        if new_state not in {state.value for state in NoteGraphSuggestionRunState}:
            raise ValueError("notes_graph_run_state_invalid")
        terminal = new_state in {"succeeded", "failed", "cancelled", "stale"}

        def mutate(conn: SuggestionConnection) -> NoteGraphSuggestionRun:
            if terminal:
                query = (
                    "UPDATE note_graph_suggestion_runs SET state=?,revision=revision+1,error_code=?,"
                    "completed_at=?,expires_at=? WHERE owner_user_id=? AND dataset_id=? AND id=? "
                    "AND state=? AND revision=?"
                )
                values: tuple[object, ...] = (
                    new_state,
                    error_code,
                    self._db_datetime(now_utc),
                    self._db_datetime(now_utc + timedelta(days=90 if new_state == "succeeded" else 30)),
                    self.owner_user_id,
                    dataset,
                    run_id,
                    expected_state,
                    expected_revision,
                )
            else:
                started_sql = "started_at=?," if new_state == "running" else ""
                query = (
                    "UPDATE note_graph_suggestion_runs SET state=?,revision=revision+1,error_code=?,"
                    f"{started_sql}expires_at=expires_at WHERE owner_user_id=? AND dataset_id=? "  # nosec B608
                    "AND id=? AND state=? AND revision=?"
                )
                started_values: tuple[object, ...] = (
                    (self._db_datetime(now_utc),) if new_state == "running" else ()
                )
                values = (
                    new_state,
                    error_code,
                    *started_values,
                    self.owner_user_id,
                    dataset,
                    run_id,
                    expected_state,
                    expected_revision,
                )
            cursor = conn.execute(query, values)
            if cursor.rowcount != 1:
                raise RuntimeError("notes_graph_run_conflict")
            return self._load_run(conn, dataset, run_id)

        return self._with_dataset_scope(dataset, mutate)

    def _suggestion_from_row(self, row: Any) -> NoteGraphSuggestion:
        return NoteGraphSuggestion(
            id=str(row["id"]),
            run_id=str(row["run_id"]),
            owner_user_id=str(row["owner_user_id"]),
            dataset_id=str(row["dataset_id"]),
            kind=NoteGraphSuggestionKind(str(row["kind"])),
            source_note_id=str(row["source_note_id"]),
            source_fingerprint=str(row["source_fingerprint"]),
            state=NoteGraphSuggestionState(str(row["state"])),
            revision=int(row["revision"]),
            created_at=self._iso(row["created_at"]) or "",
            updated_at=self._iso(row["updated_at"]) or "",
            target_note_id=row["target_note_id"],
            target_fingerprint=row["target_fingerprint"],
            normalized_tag=row["normalized_tag"],
            display_tag=row["display_tag"],
            keyword_sync_id=row["keyword_sync_id"],
            match_strength=row["match_strength"],
            rationale=row["rationale"],
            decision_reason=row["decision_reason"],
            accepted_resource_identity=row["accepted_resource_identity"],
            decision_at=self._iso(row["decision_at"]),
            acceptance_lease_token=row["acceptance_lease_token"],
            acceptance_lease_expires_at=self._iso(row["acceptance_lease_expires_at"]),
            decision_receipt_id=row["decision_receipt_id"],
            expires_at=self._iso(row["expires_at"]),
        )

    def _load_suggestion(
        self,
        conn: SuggestionConnection,
        dataset_id: str,
        suggestion_id: str,
    ) -> NoteGraphSuggestion:
        row = conn.execute(
            "SELECT * FROM note_graph_suggestions WHERE owner_user_id=? AND dataset_id=? AND id=?",
            (self.owner_user_id, dataset_id, suggestion_id),
        ).fetchone()
        if row is None:
            raise RuntimeError("notes_graph_suggestion_not_found")
        return self._suggestion_from_row(row)

    def _current_note_fingerprint(
        self,
        conn: SuggestionConnection,
        note_id: str,
    ) -> str | None:
        row = conn.execute(
            "SELECT title,content FROM notes WHERE client_id=? AND id=? AND deleted=?",
            (self.owner_user_id, note_id, self._deleted_value()),
        ).fetchone()
        if row is None:
            return None
        return content_fingerprint(row["title"], row["content"])

    def stage_suggestions(
        self,
        *,
        dataset_id: str,
        run_id: str,
        expected_state: str,
        expected_revision: int,
        result_digest: str,
        candidates: tuple[dict[str, Any], ...],
        invalid_item_count: int,
        now: datetime,
    ) -> NoteGraphSuggestionRun:
        """Persist a complete hidden result set and fence the run into publishing."""

        dataset = self._scope(dataset_id)
        now_utc = self._aware_utc(now)
        if not result_digest.startswith("sha256:") or len(result_digest) != 71:
            raise ValueError("notes_graph_result_digest_invalid")
        if not 0 <= invalid_item_count <= 10_000 or len(candidates) > 100:
            raise ValueError("notes_graph_stage_bounds_invalid")

        def mutate(conn: SuggestionConnection) -> NoteGraphSuggestionRun:
            run = self._load_run(conn, dataset, run_id)
            if run.state.value != expected_state or run.revision != expected_revision:
                raise RuntimeError("notes_graph_run_conflict")
            related_count = 0
            tag_count = 0
            for candidate in candidates:
                candidate_id = str(candidate.get("id") or "")
                kind = str(candidate.get("kind") or "")
                if not candidate_id or kind not in {"related_note", "tag"}:
                    raise ValueError("notes_graph_staged_candidate_invalid")
                evidence = candidate.get("evidence", ())
                if not isinstance(evidence, (tuple, list)) or len(evidence) > 8:
                    raise ValueError("notes_graph_evidence_invalid")
                if kind == "related_note":
                    target_note_id = str(candidate.get("target_note_id") or "")
                    target_fingerprint = str(candidate.get("target_fingerprint") or "")
                    if not target_note_id or not target_fingerprint:
                        raise ValueError("notes_graph_staged_candidate_invalid")
                    normalized_tag = display_tag = keyword_sync_id = None
                    related_count += 1
                else:
                    target_note_id = target_fingerprint = None
                    normalized_tag = str(candidate.get("normalized_tag") or "").strip().casefold()
                    display_tag = str(candidate.get("display_tag") or "").strip()
                    keyword_sync_id = candidate.get("keyword_sync_id")
                    if not normalized_tag or not display_tag:
                        raise ValueError("notes_graph_staged_candidate_invalid")
                    tag_count += 1
                rationale = candidate.get("rationale")
                if (
                    candidate.get("match_strength") not in {"strong", "possible"}
                    or rationale is not None
                    and len(str(rationale)) > 240
                ):
                    raise ValueError("notes_graph_staged_candidate_invalid")
                conn.execute(
                    "INSERT INTO note_graph_suggestions("
                    "id,run_id,owner_user_id,dataset_id,kind,source_note_id,source_fingerprint,"
                    "target_note_id,target_fingerprint,normalized_tag,display_tag,keyword_sync_id,"
                    "match_strength,rationale,state,revision,created_at,updated_at"
                    ") VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,'staged',1,?,?)",
                    (
                        candidate_id,
                        run_id,
                        self.owner_user_id,
                        dataset,
                        kind,
                        run.source_note_id,
                        run.source_fingerprint,
                        target_note_id,
                        target_fingerprint,
                        normalized_tag,
                        display_tag,
                        keyword_sync_id,
                        candidate.get("match_strength"),
                        rationale,
                        self._db_datetime(now_utc),
                        self._db_datetime(now_utc),
                    ),
                )
                for reference in evidence:
                    allowed = {
                        "side",
                        "ordinal",
                        "note_id",
                        "field",
                        "content_fingerprint",
                        "start_offset",
                        "end_offset",
                    }
                    if not isinstance(reference, dict) or set(reference) != allowed:
                        raise ValueError("notes_graph_evidence_invalid")
                    start = reference["start_offset"]
                    end = reference["end_offset"]
                    expected_note_id = (
                        run.source_note_id if reference["side"] == "source" else target_note_id
                    )
                    expected_fingerprint = (
                        run.source_fingerprint
                        if reference["side"] == "source"
                        else target_fingerprint
                    )
                    if (
                        reference["side"] not in {"source", "target"}
                        or expected_note_id is None
                        or reference["note_id"] != expected_note_id
                        or reference["content_fingerprint"] != expected_fingerprint
                        or reference["field"] not in {"title", "content"}
                        or isinstance(reference["ordinal"], bool)
                        or not isinstance(reference["ordinal"], int)
                        or reference["ordinal"] < 0
                        or isinstance(start, bool)
                        or isinstance(end, bool)
                        or not isinstance(start, int)
                        or not isinstance(end, int)
                        or start < 0
                        or end <= start
                    ):
                        raise ValueError("notes_graph_evidence_invalid")
                    conn.execute(
                        "INSERT INTO note_graph_suggestion_evidence("
                        "suggestion_id,owner_user_id,dataset_id,side,ordinal,note_id,field,"
                        "content_fingerprint,start_offset,end_offset) VALUES (?,?,?,?,?,?,?,?,?,?)",
                        (
                            candidate_id,
                            self.owner_user_id,
                            dataset,
                            reference["side"],
                            reference["ordinal"],
                            reference["note_id"],
                            reference["field"],
                            reference["content_fingerprint"],
                            start,
                            end,
                        ),
                    )
            cursor = conn.execute(
                "UPDATE note_graph_suggestion_runs SET state='publishing',revision=revision+1,"
                "result_digest=?,suggestion_count=?,related_note_count=?,tag_count=?,invalid_item_count=? "
                "WHERE owner_user_id=? AND dataset_id=? AND id=? AND state=? AND revision=?",
                (
                    result_digest,
                    len(candidates),
                    related_count,
                    tag_count,
                    invalid_item_count,
                    self.owner_user_id,
                    dataset,
                    run_id,
                    expected_state,
                    expected_revision,
                ),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("notes_graph_run_conflict")
            return self._load_run(conn, dataset, run_id)

        return self._with_dataset_scope(dataset, mutate)

    def _has_current_link(
        self,
        conn: SuggestionConnection,
        source_note_id: str,
        target_note_id: str,
    ) -> bool:
        row = conn.execute(
            "SELECT 1 FROM note_edges WHERE user_id=? AND deleted=? AND "
            "((from_note_id=? AND to_note_id=?) OR (from_note_id=? AND to_note_id=?)) LIMIT 1",
            (
                self.owner_user_id,
                self._deleted_value(),
                source_note_id,
                target_note_id,
                target_note_id,
                source_note_id,
            ),
        ).fetchone()
        if row is not None:
            return True
        query = (
            "SELECT 1 FROM note_wikilink_edges WHERE "
            "((source_note_id=? AND target_note_id=?) OR (source_note_id=? AND target_note_id=?))"
        )
        params: tuple[object, ...] = (
            source_note_id,
            target_note_id,
            target_note_id,
            source_note_id,
        )
        if self.is_postgres:
            query += " AND owner_user_id=?"
            params += (self.owner_user_id,)
        return conn.execute(f"{query} LIMIT 1", params).fetchone() is not None  # nosec B608

    def _is_suppressed(self, conn: SuggestionConnection, row: Any) -> bool:
        if row["kind"] == "related_note":
            return conn.execute(
                "SELECT 1 FROM note_graph_suggestions WHERE owner_user_id=? AND dataset_id=? "
                "AND id<>? AND kind='related_note' AND state='rejected' AND source_note_id=? "
                "AND source_fingerprint=? AND target_note_id=? AND target_fingerprint=? LIMIT 1",
                (
                    self.owner_user_id,
                    row["dataset_id"],
                    row["id"],
                    row["source_note_id"],
                    row["source_fingerprint"],
                    row["target_note_id"],
                    row["target_fingerprint"],
                ),
            ).fetchone() is not None
        return conn.execute(
            "SELECT 1 FROM note_graph_suggestions WHERE owner_user_id=? AND dataset_id=? "
            "AND id<>? AND kind='tag' AND state='rejected' AND source_note_id=? "
            "AND source_fingerprint=? AND normalized_tag=? LIMIT 1",
            (
                self.owner_user_id,
                row["dataset_id"],
                row["id"],
                row["source_note_id"],
                row["source_fingerprint"],
                row["normalized_tag"],
            ),
        ).fetchone() is not None

    def _tag_identity_is_current(self, conn: SuggestionConnection, row: Any) -> bool:
        sync_id = row["keyword_sync_id"]
        if sync_id is None:
            return True
        keyword_table = self._db._map_table_for_backend("keywords")
        keyword = conn.execute(
            f"SELECT keyword FROM {keyword_table} WHERE client_id=? AND sync_id=? AND deleted=?",  # nosec B608
            (self.owner_user_id, sync_id, self._deleted_value()),
        ).fetchone()
        return keyword is not None and str(keyword["keyword"]).strip().casefold() == row["normalized_tag"]

    def _supersede_pending(self, conn: SuggestionConnection, row: Any, now_utc: datetime) -> None:
        if row["kind"] == "related_note":
            predicate = "target_note_id=? AND target_fingerprint=?"
            identity: tuple[object, ...] = (row["target_note_id"], row["target_fingerprint"])
        else:
            predicate = "normalized_tag=?"
            identity = (row["normalized_tag"],)
        pending = conn.execute(
            "SELECT id,revision FROM note_graph_suggestions "
            "WHERE owner_user_id=? AND dataset_id=? AND id<>? AND source_note_id=? "
            "AND source_fingerprint=? AND kind=? AND state='pending' AND "
            f"{predicate} ORDER BY id",  # nosec B608
            (
                self.owner_user_id,
                row["dataset_id"],
                row["id"],
                row["source_note_id"],
                row["source_fingerprint"],
                row["kind"],
                *identity,
            ),
        ).fetchall()
        for prior in pending:
            updated = conn.execute(
                "UPDATE note_graph_suggestions SET state='stale',revision=revision+1,"
                "decision_reason='superseded_by_run',updated_at=?,expires_at=? "
                "WHERE owner_user_id=? AND dataset_id=? AND id=? AND state='pending' AND revision=?",
                (
                    self._db_datetime(now_utc),
                    self._db_datetime(now_utc + timedelta(days=30)),
                    self.owner_user_id,
                    row["dataset_id"],
                    prior["id"],
                    prior["revision"],
                ),
            )
            if updated.rowcount != 1:
                raise RuntimeError("notes_graph_suggestion_conflict")

    def _delete_staged_for_run(
        self,
        conn: SuggestionConnection,
        *,
        dataset_id: str,
        run_id: str,
    ) -> None:
        staged = conn.execute(
            "SELECT id,revision FROM note_graph_suggestions WHERE owner_user_id=? "
            "AND dataset_id=? AND run_id=? AND state='staged' ORDER BY id",
            (self.owner_user_id, dataset_id, run_id),
        ).fetchall()
        for row in staged:
            deleted = conn.execute(
                "DELETE FROM note_graph_suggestions WHERE owner_user_id=? AND dataset_id=? "
                "AND id=? AND state='staged' AND revision=?",
                (self.owner_user_id, dataset_id, row["id"], row["revision"]),
            )
            if deleted.rowcount != 1:
                raise RuntimeError("notes_graph_suggestion_conflict")

    def activate_staged_run(
        self,
        *,
        dataset_id: str,
        run_id: str,
        expected_state: str,
        expected_revision: int,
        observed_job_id: str | None,
        observed_completion_token: str | None,
        observed_result_digest: str | None,
        now: datetime,
    ) -> NoteGraphSuggestionRun:
        """Validate a terminal receipt projection and atomically expose a complete staged set."""

        dataset = self._scope(dataset_id)
        now_utc = self._aware_utc(now)

        def mutate(conn: SuggestionConnection) -> NoteGraphSuggestionRun:
            run = self._load_run(conn, dataset, run_id)
            if (
                run.state == NoteGraphSuggestionRunState.STALE
                and run.job_id == observed_job_id
                and run.expected_completion_token == observed_completion_token
                and run.result_digest == observed_result_digest
            ):
                return run
            if (
                run.state.value != expected_state
                or run.revision != expected_revision
                or run.job_id != observed_job_id
                or run.expected_completion_token != observed_completion_token
                or run.result_digest != observed_result_digest
            ):
                raise RuntimeError("notes_graph_publication_receipt_mismatch")
            staged = conn.execute(
                "SELECT * FROM note_graph_suggestions WHERE owner_user_id=? AND dataset_id=? "
                "AND run_id=? AND state='staged' ORDER BY id",
                (self.owner_user_id, dataset, run_id),
            ).fetchall()
            source_current = self._current_note_fingerprint(conn, run.source_note_id)
            freshness_failed = source_current != run.source_fingerprint
            if not freshness_failed:
                for row in staged:
                    if row["kind"] == "related_note" and self._current_note_fingerprint(
                        conn, str(row["target_note_id"])
                    ) != row["target_fingerprint"]:
                        freshness_failed = True
                        break
                    if row["kind"] == "tag" and not self._tag_identity_is_current(conn, row):
                        freshness_failed = True
                        break
            if freshness_failed:
                self._delete_staged_for_run(
                    conn,
                    dataset_id=dataset,
                    run_id=run_id,
                )
                cursor = conn.execute(
                    "UPDATE note_graph_suggestion_runs SET state='stale',revision=revision+1,"
                    "error_code='notes_graph_fingerprint_stale',completed_at=?,expires_at=? "
                    "WHERE owner_user_id=? AND dataset_id=? AND id=? AND state=? AND revision=?",
                    (
                        self._db_datetime(now_utc),
                        self._db_datetime(now_utc + timedelta(days=30)),
                        self.owner_user_id,
                        dataset,
                        run_id,
                        expected_state,
                        expected_revision,
                    ),
                )
                if cursor.rowcount != 1:
                    raise RuntimeError("notes_graph_run_conflict")
                return self._load_run(conn, dataset, run_id)

            activated_ids: list[str] = []
            for row in staged:
                filtered = self._is_suppressed(conn, row)
                if row["kind"] == "related_note":
                    filtered = filtered or self._has_current_link(
                        conn, str(row["source_note_id"]), str(row["target_note_id"])
                    )
                if filtered:
                    deleted = conn.execute(
                        "DELETE FROM note_graph_suggestions WHERE owner_user_id=? AND dataset_id=? "
                        "AND id=? AND state='staged' AND revision=?",
                        (self.owner_user_id, dataset, row["id"], row["revision"]),
                    )
                    if deleted.rowcount != 1:
                        raise RuntimeError("notes_graph_suggestion_conflict")
                    continue
                self._supersede_pending(conn, row, now_utc)
                cursor = conn.execute(
                    "UPDATE note_graph_suggestions SET state='pending',updated_at=? "
                    "WHERE owner_user_id=? AND dataset_id=? AND id=? AND state='staged' AND revision=?",
                    (
                        self._db_datetime(now_utc),
                        self.owner_user_id,
                        dataset,
                        row["id"],
                        row["revision"],
                    ),
                )
                if cursor.rowcount != 1:
                    raise RuntimeError("notes_graph_suggestion_conflict")
                activated_ids.append(str(row["id"]))
            cursor = conn.execute(
                "UPDATE note_graph_suggestion_runs SET state='succeeded',revision=revision+1,"
                "suggestion_count=?,related_note_count=(SELECT COUNT(*) FROM note_graph_suggestions "
                "WHERE owner_user_id=? AND dataset_id=? AND run_id=? AND kind='related_note' AND state='pending'),"
                "tag_count=(SELECT COUNT(*) FROM note_graph_suggestions WHERE owner_user_id=? "
                "AND dataset_id=? AND run_id=? AND kind='tag' AND state='pending'),completed_at=?,expires_at=? "
                "WHERE owner_user_id=? AND dataset_id=? AND id=? AND state=? AND revision=?",
                (
                    len(activated_ids),
                    self.owner_user_id,
                    dataset,
                    run_id,
                    self.owner_user_id,
                    dataset,
                    run_id,
                    self._db_datetime(now_utc),
                    self._db_datetime(now_utc + timedelta(days=90)),
                    self.owner_user_id,
                    dataset,
                    run_id,
                    expected_state,
                    expected_revision,
                ),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("notes_graph_run_conflict")
            return self._load_run(conn, dataset, run_id)

        return self._with_dataset_scope(dataset, mutate)

    @staticmethod
    def _cursor_segment(value: bytes) -> str:
        return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")

    @staticmethod
    def _decode_cursor_segment(value: str) -> bytes:
        decoded = base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))
        if NoteGraphSuggestionStore._cursor_segment(decoded) != value:
            raise ValueError("notes_graph_cursor_invalid")
        return decoded

    def _encode_cursor(self, payload: dict[str, Any], secret: bytes) -> str:
        raw = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
        if len(raw) > 1024 or not isinstance(secret, bytes) or len(secret) < 16:
            raise ValueError("notes_graph_cursor_invalid")
        signature = hmac.digest(secret, raw, "sha256")
        return f"{self._cursor_segment(raw)}.{self._cursor_segment(signature)}"

    def _decode_cursor(self, raw_cursor: str, expected: dict[str, Any], secret: bytes) -> tuple[str, str]:
        if not isinstance(raw_cursor, str) or len(raw_cursor.encode("ascii", "ignore")) > 2048:
            raise ValueError("notes_graph_cursor_invalid")
        try:
            payload_segment, signature_segment = raw_cursor.split(".")
            raw = self._decode_cursor_segment(payload_segment)
            signature = self._decode_cursor_segment(signature_segment)
            if not hmac.compare_digest(signature, hmac.digest(secret, raw, "sha256")):
                raise ValueError("notes_graph_cursor_invalid")
            payload = json.loads(raw)
        except (UnicodeError, ValueError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError("notes_graph_cursor_invalid") from exc
        if not isinstance(payload, dict) or any(payload.get(key) != value for key, value in expected.items()):
            raise ValueError("notes_graph_cursor_invalid")
        after_time = payload.get("after_time")
        after_id = payload.get("after_id")
        if not isinstance(after_time, str) or not isinstance(after_id, str):
            raise ValueError("notes_graph_cursor_invalid")
        return after_time, after_id

    def list_suggestions(
        self,
        *,
        dataset_id: str,
        source_note_id: str,
        source_fingerprint: str,
        states: tuple[str, ...],
        limit: int,
        cursor: str | None,
        cursor_secret: bytes,
    ) -> SuggestionPage:
        """List only suggestions from succeeded runs using a signed keyset cursor."""

        dataset = self._scope(dataset_id)
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
            raise ValueError("notes_graph_page_limit_invalid")
        allowed_states = {"pending", "accepting", "accepted", "rejected", "stale"}
        if not states or any(state not in allowed_states for state in states):
            raise ValueError("notes_graph_state_filter_invalid")
        state_values = tuple(sorted(set(states)))
        binding = {
            "v": 1,
            "owner": self.owner_user_id,
            "dataset": dataset,
            "source": source_note_id,
            "fingerprint": source_fingerprint,
            "states": list(state_values),
        }
        after = self._decode_cursor(cursor, binding, cursor_secret) if cursor is not None else None

        def read(conn: SuggestionConnection) -> SuggestionPage:
            placeholders = ",".join("?" for _ in state_values)
            query = (
                "SELECT suggestion.* FROM note_graph_suggestions suggestion "
                "JOIN note_graph_suggestion_runs run ON run.owner_user_id=suggestion.owner_user_id "
                "AND run.dataset_id=suggestion.dataset_id AND run.id=suggestion.run_id "
                "WHERE suggestion.owner_user_id=? AND suggestion.dataset_id=? "
                "AND suggestion.source_note_id=? AND suggestion.source_fingerprint=? "
                "AND run.state='succeeded' "
                f"AND suggestion.state IN ({placeholders}) "  # nosec B608
            )
            params: list[object] = [
                self.owner_user_id,
                dataset,
                source_note_id,
                source_fingerprint,
                *state_values,
            ]
            if after is not None:
                query += "AND (suggestion.updated_at < ? OR (suggestion.updated_at = ? AND suggestion.id > ?)) "
                params.extend((after[0], after[0], after[1]))
            query += "ORDER BY suggestion.updated_at DESC,suggestion.id ASC LIMIT ?"
            params.append(limit + 1)
            rows = conn.execute(query, tuple(params)).fetchall()
            visible = rows[:limit]
            next_cursor = None
            if len(rows) > limit and visible:
                last = visible[-1]
                next_cursor = self._encode_cursor(
                    {
                        **binding,
                        "after_time": self._iso(last["updated_at"]),
                        "after_id": str(last["id"]),
                    },
                    cursor_secret,
                )
            return SuggestionPage(
                items=tuple(self._suggestion_from_row(row) for row in visible),
                next_cursor=next_cursor,
            )

        return self._with_dataset_scope(dataset, read)

    def _admit_receipt(
        self,
        conn: SuggestionConnection,
        *,
        dataset_id: str,
        operation_kind: str,
        source_note_id: str,
        resource_identity: str,
        idempotency_key: str,
        request_fields: dict[str, Any],
        now_utc: datetime,
        receipt_id: str | None = None,
    ) -> tuple[str, Any]:
        key_digest = self.idempotency_key_digest(idempotency_key)
        request_fingerprint = self.canonical_request_fingerprint(operation_kind, request_fields)
        existing = conn.execute(
            "SELECT * FROM note_graph_suggestion_operation_receipts WHERE owner_user_id=? "
            "AND dataset_id=? AND operation_kind=? AND resource_identity=? "
            "AND idempotency_key_digest=?",
            (
                self.owner_user_id,
                dataset_id,
                operation_kind,
                resource_identity,
                key_digest,
            ),
        ).fetchone()
        if existing is not None:
            if not secrets.compare_digest(str(existing["request_fingerprint"]), request_fingerprint):
                raise RuntimeError("notes_graph_suggestion_idempotency_mismatch")
            disposition = "terminal_replay" if existing["state"] in {"completed", "failed"} else "in_progress"
            return disposition, existing
        new_id = receipt_id or str(uuid.uuid4())
        conn.execute(
            "INSERT INTO note_graph_suggestion_operation_receipts("
            "id,operation_kind,owner_user_id,dataset_id,source_note_id,resource_identity,"
            "idempotency_key_digest,request_fingerprint,state,created_at,expires_at"
            ") VALUES (?,?,?,?,?,?,?,?, 'in_progress',?,?)",
            (
                new_id,
                operation_kind,
                self.owner_user_id,
                dataset_id,
                source_note_id,
                resource_identity,
                key_digest,
                request_fingerprint,
                self._db_datetime(now_utc),
                self._db_datetime(now_utc + timedelta(days=90)),
            ),
        )
        return "created", conn.execute(
            "SELECT * FROM note_graph_suggestion_operation_receipts WHERE owner_user_id=? "
            "AND dataset_id=? AND id=?",
            (self.owner_user_id, dataset_id, new_id),
        ).fetchone()

    def _complete_receipt(
        self,
        conn: SuggestionConnection,
        *,
        dataset_id: str,
        receipt: Any,
        envelope: dict[str, Any],
        http_status: int,
        now_utc: datetime,
    ) -> None:
        cursor = conn.execute(
            "UPDATE note_graph_suggestion_operation_receipts SET state='completed',http_status=?,"
            "replay_envelope=?,completed_at=?,expires_at=? WHERE owner_user_id=? AND dataset_id=? "
            "AND id=? AND state='in_progress' AND request_fingerprint=?",
            (
                http_status,
                self._encode_envelope(envelope),
                self._db_datetime(now_utc),
                self._db_datetime(now_utc + timedelta(days=90)),
                self.owner_user_id,
                dataset_id,
                receipt["id"],
                receipt["request_fingerprint"],
            ),
        )
        if cursor.rowcount != 1:
            raise RuntimeError("notes_graph_receipt_conflict")

    def _rejection_set_from_row(self, row: Any) -> NoteGraphSuggestionRejectionSet:
        return NoteGraphSuggestionRejectionSet(
            owner_user_id=str(row["owner_user_id"]),
            dataset_id=str(row["dataset_id"]),
            source_note_id=str(row["source_note_id"]),
            source_fingerprint=str(row["source_fingerprint"]),
            revision=int(row["revision"]),
            rejection_count=int(row["rejection_count"]),
            updated_at=self._iso(row["updated_at"]) or "",
        )

    def _load_rejection_set(
        self,
        conn: SuggestionConnection,
        dataset_id: str,
        source_note_id: str,
        source_fingerprint: str,
    ) -> NoteGraphSuggestionRejectionSet | None:
        row = conn.execute(
            "SELECT * FROM note_graph_suggestion_rejection_sets WHERE owner_user_id=? "
            "AND dataset_id=? AND source_note_id=? AND source_fingerprint=?",
            (self.owner_user_id, dataset_id, source_note_id, source_fingerprint),
        ).fetchone()
        return self._rejection_set_from_row(row) if row is not None else None

    def _require_current_suggestion_fingerprints(
        self,
        conn: SuggestionConnection,
        suggestion: NoteGraphSuggestion,
        expected_source_fingerprint: str,
        expected_target_fingerprint: str | None,
    ) -> None:
        source_current = self._current_note_fingerprint(conn, suggestion.source_note_id)
        target_current = (
            self._current_note_fingerprint(conn, suggestion.target_note_id)
            if suggestion.target_note_id is not None
            else None
        )
        if (
            suggestion.source_fingerprint != expected_source_fingerprint
            or source_current != expected_source_fingerprint
            or suggestion.target_fingerprint != expected_target_fingerprint
            or target_current != expected_target_fingerprint
        ):
            raise RuntimeError("notes_graph_fingerprint_stale")

    def reject_suggestion(
        self,
        *,
        dataset_id: str,
        suggestion_id: str,
        expected_revision: int,
        expected_source_fingerprint: str,
        expected_target_fingerprint: str | None,
        idempotency_key: str,
        now: datetime,
    ) -> MutationResult:
        """Reject and terminalize its receipt in one fenced transaction."""

        dataset = self._scope(dataset_id)
        now_utc = self._aware_utc(now)

        def mutate(conn: SuggestionConnection) -> MutationResult:
            suggestion = self._load_suggestion(conn, dataset, suggestion_id)
            disposition, receipt = self._admit_receipt(
                conn,
                dataset_id=dataset,
                operation_kind=NoteGraphSuggestionOperationKind.SUGGESTION_REJECT.value,
                source_note_id=suggestion.source_note_id,
                resource_identity=suggestion_id,
                idempotency_key=idempotency_key,
                request_fields={
                    "suggestion_id": suggestion_id,
                    "expected_revision": expected_revision,
                    "source_fingerprint": expected_source_fingerprint,
                    "target_fingerprint": expected_target_fingerprint,
                },
                now_utc=now_utc,
            )
            if disposition == "terminal_replay":
                envelope = self._decode_envelope(receipt["replay_envelope"]) or {}
                rejection_set = self._load_rejection_set(
                    conn,
                    dataset,
                    suggestion.source_note_id,
                    suggestion.source_fingerprint,
                )
                return MutationResult("terminal_replay", envelope, rejection_set=rejection_set)
            if disposition == "in_progress":
                raise RuntimeError("notes_graph_receipt_conflict")
            if suggestion.state.value != "pending" or suggestion.revision != expected_revision:
                raise RuntimeError("notes_graph_suggestion_conflict")
            self._require_current_suggestion_fingerprints(
                conn,
                suggestion,
                expected_source_fingerprint,
                expected_target_fingerprint,
            )
            cursor = conn.execute(
                "UPDATE note_graph_suggestions SET state='rejected',revision=revision+1,"
                "rationale=NULL,decision_reason='user_rejected',decision_at=?,decision_receipt_id=?,"
                "updated_at=?,expires_at=? WHERE owner_user_id=? AND dataset_id=? AND id=? "
                "AND state='pending' AND revision=?",
                (
                    self._db_datetime(now_utc),
                    receipt["id"],
                    self._db_datetime(now_utc),
                    self._db_datetime(now_utc + timedelta(days=30)),
                    self.owner_user_id,
                    dataset,
                    suggestion_id,
                    expected_revision,
                ),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("notes_graph_suggestion_conflict")
            conn.execute(
                "DELETE FROM note_graph_suggestion_evidence WHERE owner_user_id=? AND dataset_id=? "
                "AND suggestion_id=?",
                (self.owner_user_id, dataset, suggestion_id),
            )
            rejection_set = self._load_rejection_set(
                conn,
                dataset,
                suggestion.source_note_id,
                suggestion.source_fingerprint,
            )
            if rejection_set is None:
                conn.execute(
                    "INSERT INTO note_graph_suggestion_rejection_sets("
                    "owner_user_id,dataset_id,source_note_id,source_fingerprint,revision,rejection_count,updated_at"
                    ") VALUES (?,?,?,?,1,1,?)",
                    (
                        self.owner_user_id,
                        dataset,
                        suggestion.source_note_id,
                        suggestion.source_fingerprint,
                        self._db_datetime(now_utc),
                    ),
                )
            else:
                updated = conn.execute(
                    "UPDATE note_graph_suggestion_rejection_sets SET revision=revision+1,"
                    "rejection_count=rejection_count+1,updated_at=? WHERE owner_user_id=? "
                    "AND dataset_id=? AND source_note_id=? AND source_fingerprint=? AND revision=?",
                    (
                        self._db_datetime(now_utc),
                        self.owner_user_id,
                        dataset,
                        suggestion.source_note_id,
                        suggestion.source_fingerprint,
                        rejection_set.revision,
                    ),
                )
                if updated.rowcount != 1:
                    raise RuntimeError("notes_graph_rejection_set_conflict")
            rejection_set = self._load_rejection_set(
                conn,
                dataset,
                suggestion.source_note_id,
                suggestion.source_fingerprint,
            )
            envelope = {"suggestion_id": suggestion_id, "state": "rejected", "revision": expected_revision + 1}
            self._complete_receipt(
                conn,
                dataset_id=dataset,
                receipt=receipt,
                envelope=envelope,
                http_status=200,
                now_utc=now_utc,
            )
            return MutationResult("completed", envelope, rejection_set=rejection_set)

        return self._with_dataset_scope(dataset, mutate)

    def reset_rejections(
        self,
        *,
        dataset_id: str,
        source_note_id: str,
        source_fingerprint: str,
        expected_revision: int,
        idempotency_key: str,
        now: datetime,
    ) -> MutationResult:
        """CAS-reset only the exact fingerprint rejection set and preserve its row."""

        dataset = self._scope(dataset_id)
        now_utc = self._aware_utc(now)

        def mutate(conn: SuggestionConnection) -> MutationResult:
            disposition, receipt = self._admit_receipt(
                conn,
                dataset_id=dataset,
                operation_kind=NoteGraphSuggestionOperationKind.REJECTIONS_RESET.value,
                source_note_id=source_note_id,
                resource_identity=source_note_id,
                idempotency_key=idempotency_key,
                request_fields={
                    "source_note_id": source_note_id,
                    "source_fingerprint": source_fingerprint,
                    "expected_revision": expected_revision,
                },
                now_utc=now_utc,
            )
            if disposition == "terminal_replay":
                rejection_set = self._load_rejection_set(
                    conn, dataset, source_note_id, source_fingerprint
                )
                return MutationResult(
                    "terminal_replay",
                    self._decode_envelope(receipt["replay_envelope"]) or {},
                    rejection_set=rejection_set,
                )
            if disposition == "in_progress":
                raise RuntimeError("notes_graph_receipt_conflict")
            if self._current_note_fingerprint(conn, source_note_id) != source_fingerprint:
                raise RuntimeError("notes_graph_fingerprint_stale")
            rejection_set = self._load_rejection_set(
                conn, dataset, source_note_id, source_fingerprint
            )
            if rejection_set is None or rejection_set.revision != expected_revision:
                raise RuntimeError("notes_graph_rejection_set_conflict")
            rejected_rows = conn.execute(
                "SELECT id,revision FROM note_graph_suggestions WHERE owner_user_id=? "
                "AND dataset_id=? AND source_note_id=? AND source_fingerprint=? "
                "AND state='rejected' ORDER BY id",
                (self.owner_user_id, dataset, source_note_id, source_fingerprint),
            ).fetchall()
            cleared = 0
            for rejected in rejected_rows:
                changed = conn.execute(
                    "UPDATE note_graph_suggestions SET state='stale',revision=revision+1,"
                    "decision_reason='rejections_reset',updated_at=?,expires_at=? "
                    "WHERE owner_user_id=? AND dataset_id=? AND id=? AND state='rejected' AND revision=?",
                    (
                        self._db_datetime(now_utc),
                        self._db_datetime(now_utc + timedelta(days=30)),
                        self.owner_user_id,
                        dataset,
                        rejected["id"],
                        rejected["revision"],
                    ),
                )
                if changed.rowcount != 1:
                    raise RuntimeError("notes_graph_suggestion_conflict")
                cleared += 1
            updated = conn.execute(
                "UPDATE note_graph_suggestion_rejection_sets SET revision=revision+1,"
                "rejection_count=0,updated_at=? WHERE owner_user_id=? AND dataset_id=? "
                "AND source_note_id=? AND source_fingerprint=? AND revision=?",
                (
                    self._db_datetime(now_utc),
                    self.owner_user_id,
                    dataset,
                    source_note_id,
                    source_fingerprint,
                    expected_revision,
                ),
            )
            if updated.rowcount != 1:
                raise RuntimeError("notes_graph_rejection_set_conflict")
            rejection_set = self._load_rejection_set(
                conn, dataset, source_note_id, source_fingerprint
            )
            envelope = {
                "source_note_id": source_note_id,
                "cleared_count": int(cleared or 0),
                "rejection_set_revision": expected_revision + 1,
            }
            self._complete_receipt(
                conn,
                dataset_id=dataset,
                receipt=receipt,
                envelope=envelope,
                http_status=200,
                now_utc=now_utc,
            )
            return MutationResult("completed", envelope, rejection_set=rejection_set)

        return self._with_dataset_scope(dataset, mutate)

    def claim_acceptance(
        self,
        *,
        dataset_id: str,
        suggestion_id: str,
        expected_revision: int,
        expected_source_fingerprint: str,
        expected_target_fingerprint: str | None,
        idempotency_key: str,
        now: datetime,
    ) -> MutationResult:
        """Claim a pending decision with a durable five-minute lease."""

        dataset = self._scope(dataset_id)
        now_utc = self._aware_utc(now)

        def mutate(conn: SuggestionConnection) -> MutationResult:
            suggestion = self._load_suggestion(conn, dataset, suggestion_id)
            disposition, receipt = self._admit_receipt(
                conn,
                dataset_id=dataset,
                operation_kind=NoteGraphSuggestionOperationKind.SUGGESTION_ACCEPT.value,
                source_note_id=suggestion.source_note_id,
                resource_identity=suggestion_id,
                idempotency_key=idempotency_key,
                request_fields={
                    "suggestion_id": suggestion_id,
                    "expected_revision": expected_revision,
                    "source_fingerprint": expected_source_fingerprint,
                    "target_fingerprint": expected_target_fingerprint,
                },
                now_utc=now_utc,
            )
            if disposition == "terminal_replay":
                return MutationResult(
                    "terminal_replay",
                    self._decode_envelope(receipt["replay_envelope"]) or {},
                    suggestion=suggestion,
                )
            if disposition == "in_progress":
                return MutationResult(
                    "in_progress",
                    {},
                    suggestion=suggestion,
                    continuation="resume_suggestion_acceptance",
                )
            if suggestion.state != NoteGraphSuggestionState.PENDING or suggestion.revision != expected_revision:
                raise RuntimeError("notes_graph_suggestion_conflict")
            self._require_current_suggestion_fingerprints(
                conn,
                suggestion,
                expected_source_fingerprint,
                expected_target_fingerprint,
            )
            lease_token = secrets.token_urlsafe(32)
            updated = conn.execute(
                "UPDATE note_graph_suggestions SET state='accepting',revision=revision+1,"
                "acceptance_lease_token=?,acceptance_lease_expires_at=?,decision_receipt_id=?,"
                "updated_at=? WHERE owner_user_id=? AND dataset_id=? AND id=? AND state='pending' "
                "AND revision=?",
                (
                    lease_token,
                    self._db_datetime(now_utc + timedelta(minutes=5)),
                    receipt["id"],
                    self._db_datetime(now_utc),
                    self.owner_user_id,
                    dataset,
                    suggestion_id,
                    expected_revision,
                ),
            )
            if updated.rowcount != 1:
                raise RuntimeError("notes_graph_suggestion_conflict")
            claimed = self._load_suggestion(conn, dataset, suggestion_id)
            return MutationResult(
                "completed",
                {"suggestion_id": suggestion_id, "state": "accepting", "revision": claimed.revision},
                suggestion=claimed,
                continuation="continue_suggestion_acceptance",
            )

        return self._with_dataset_scope(dataset, mutate)

    def reclaim_expired_acceptance(
        self,
        *,
        dataset_id: str,
        suggestion_id: str,
        expected_state: str,
        expected_revision: int,
        expected_lease_token: str,
        now: datetime,
    ) -> NoteGraphSuggestion:
        """Advance the fence and replace one expired acceptance lease."""

        dataset = self._scope(dataset_id)
        now_utc = self._aware_utc(now)

        def mutate(conn: SuggestionConnection) -> NoteGraphSuggestion:
            lease_token = secrets.token_urlsafe(32)
            updated = conn.execute(
                "UPDATE note_graph_suggestions SET revision=revision+1,acceptance_lease_token=?,"
                "acceptance_lease_expires_at=?,updated_at=? WHERE owner_user_id=? AND dataset_id=? "
                "AND id=? AND state=? AND revision=? AND acceptance_lease_token=? "
                "AND acceptance_lease_expires_at<=?",
                (
                    lease_token,
                    self._db_datetime(now_utc + timedelta(minutes=5)),
                    self._db_datetime(now_utc),
                    self.owner_user_id,
                    dataset,
                    suggestion_id,
                    expected_state,
                    expected_revision,
                    expected_lease_token,
                    self._db_datetime(now_utc),
                ),
            )
            if updated.rowcount != 1:
                raise RuntimeError("notes_graph_suggestion_conflict")
            return self._load_suggestion(conn, dataset, suggestion_id)

        return self._with_dataset_scope(dataset, mutate)

    def release_acceptance(
        self,
        *,
        dataset_id: str,
        suggestion_id: str,
        expected_state: str,
        expected_revision: int,
        expected_lease_token: str,
        now: datetime,
    ) -> NoteGraphSuggestion:
        """Return an accepting row to pending only under its current fence."""

        dataset = self._scope(dataset_id)
        now_utc = self._aware_utc(now)

        def mutate(conn: SuggestionConnection) -> NoteGraphSuggestion:
            updated = conn.execute(
                "UPDATE note_graph_suggestions SET state='pending',revision=revision+1,"
                "acceptance_lease_token=NULL,acceptance_lease_expires_at=NULL,decision_receipt_id=NULL,"
                "updated_at=? WHERE owner_user_id=? AND dataset_id=? AND id=? AND state=? "
                "AND revision=? AND acceptance_lease_token=?",
                (
                    self._db_datetime(now_utc),
                    self.owner_user_id,
                    dataset,
                    suggestion_id,
                    expected_state,
                    expected_revision,
                    expected_lease_token,
                ),
            )
            if updated.rowcount != 1:
                raise RuntimeError("notes_graph_suggestion_conflict")
            return self._load_suggestion(conn, dataset, suggestion_id)

        return self._with_dataset_scope(dataset, mutate)

    def cancellation_operation_id(self, *, dataset_id: str, run_id: str, run_revision: int) -> str:
        """Return the stable operation UUID for one run cancellation fence."""

        material = "\0".join(
            ("notes-graph-run-cancel-v1", self.owner_user_id, self._scope(dataset_id), run_id, str(run_revision))
        )
        return str(uuid.uuid5(uuid.NAMESPACE_URL, material))

    def _persist_cancellation_receipt(
        self,
        conn: SuggestionConnection,
        *,
        dataset_id: str,
        run: NoteGraphSuggestionRun,
        now_utc: datetime,
    ) -> None:
        operation_id = self.cancellation_operation_id(
            dataset_id=dataset_id,
            run_id=run.id,
            run_revision=run.revision,
        )
        request_fields = {
            "run_id": run.id,
            "run_revision": run.revision,
            "reason": "notes_graph_source_changed",
        }
        conn.execute(
            "INSERT INTO note_graph_suggestion_operation_receipts("
            "id,operation_kind,owner_user_id,dataset_id,source_note_id,resource_identity,"
            "idempotency_key_digest,request_fingerprint,state,created_at,expires_at"
            ") VALUES (?,?,?,?,?,?,?,?, 'in_progress',?,?)",
            (
                operation_id,
                NoteGraphSuggestionOperationKind.RUN_CANCEL.value,
                self.owner_user_id,
                dataset_id,
                run.source_note_id,
                run.id,
                self.idempotency_key_digest(operation_id),
                self.canonical_request_fingerprint("run_cancel", request_fields),
                self._db_datetime(now_utc),
                self._db_datetime(now_utc + timedelta(days=90)),
            ),
        )

    def _invalidate_suggestion_row(
        self,
        conn: SuggestionConnection,
        *,
        dataset_id: str,
        suggestion: NoteGraphSuggestion,
        reason: str,
        now_utc: datetime,
    ) -> None:
        updated = conn.execute(
            "UPDATE note_graph_suggestions SET state='stale',revision=revision+1,decision_reason=?,"
            "acceptance_lease_token=NULL,acceptance_lease_expires_at=NULL,updated_at=?,expires_at=? "
            "WHERE owner_user_id=? AND dataset_id=? AND id=? AND state=? AND revision=?",
            (
                reason,
                self._db_datetime(now_utc),
                self._db_datetime(now_utc + timedelta(days=30)),
                self.owner_user_id,
                dataset_id,
                suggestion.id,
                suggestion.state.value,
                suggestion.revision,
            ),
        )
        if updated.rowcount != 1:
            raise RuntimeError("notes_graph_suggestion_conflict")

    def invalidate_for_note_change(
        self,
        *,
        note_id: str,
        conn: SuggestionConnection,
    ) -> None:
        """Invalidate source and target state inside the caller's note transaction."""

        now_utc = datetime.now(timezone.utc)
        datasets = conn.execute(
            "SELECT dataset_id FROM note_task_scope_authority WHERE owner_user_id=? ORDER BY dataset_id",
            (self.owner_user_id,),
        ).fetchall()
        for dataset_row in datasets:
            dataset = str(dataset_row["dataset_id"])
            self._set_dataset_scope(conn, dataset)
            run_rows = conn.execute(
                "SELECT * FROM note_graph_suggestion_runs WHERE owner_user_id=? AND dataset_id=? "
                "AND source_note_id=? AND state IN ('admitting','queued','running','publishing') "
                "ORDER BY id",
                (self.owner_user_id, dataset, note_id),
            ).fetchall()
            for row in run_rows:
                run = self._run_from_row(row)
                if run.state == NoteGraphSuggestionRunState.PUBLISHING:
                    self._delete_staged_for_run(
                        conn,
                        dataset_id=dataset,
                        run_id=run.id,
                    )
                bound_admitting = (
                    run.state == NoteGraphSuggestionRunState.ADMITTING and run.job_id is not None
                )
                if run.state == NoteGraphSuggestionRunState.RUNNING or bound_admitting:
                    next_state = "cancelling"
                    completion_sql = ""
                    transition_values: tuple[object, ...] = (
                        next_state,
                        "notes_graph_source_changed",
                        self._db_datetime(now_utc + timedelta(days=30)),
                    )
                else:
                    next_state = "stale"
                    completion_sql = "completed_at=?,"
                    transition_values = (
                        next_state,
                        "notes_graph_source_changed",
                        self._db_datetime(now_utc),
                        self._db_datetime(now_utc + timedelta(days=30)),
                    )
                updated = conn.execute(
                    "UPDATE note_graph_suggestion_runs SET state=?,revision=revision+1,error_code=?,"
                    f"{completion_sql}expires_at=? "  # nosec B608
                    "WHERE owner_user_id=? AND dataset_id=? AND id=? AND state=? AND revision=?",
                    (
                        *transition_values,
                        self.owner_user_id,
                        dataset,
                        run.id,
                        run.state.value,
                        run.revision,
                    ),
                )
                if updated.rowcount != 1:
                    raise RuntimeError("notes_graph_run_conflict")
                if run.state in {NoteGraphSuggestionRunState.QUEUED, NoteGraphSuggestionRunState.RUNNING} or bound_admitting:
                    self._persist_cancellation_receipt(
                        conn,
                        dataset_id=dataset,
                        run=run,
                        now_utc=now_utc,
                    )

            staged_targets = conn.execute(
                "SELECT DISTINCT run_id FROM note_graph_suggestions WHERE owner_user_id=? "
                "AND dataset_id=? AND target_note_id=? AND state='staged' ORDER BY run_id",
                (self.owner_user_id, dataset, note_id),
            ).fetchall()
            for target_row in staged_targets:
                run = self._load_run(conn, dataset, str(target_row["run_id"]))
                if run.state != NoteGraphSuggestionRunState.PUBLISHING:
                    raise RuntimeError("notes_graph_run_conflict")
                self._delete_staged_for_run(
                    conn,
                    dataset_id=dataset,
                    run_id=run.id,
                )
                updated = conn.execute(
                    "UPDATE note_graph_suggestion_runs SET state='stale',revision=revision+1,"
                    "error_code='notes_graph_target_changed',completed_at=?,expires_at=? "
                    "WHERE owner_user_id=? AND dataset_id=? AND id=? AND state='publishing' AND revision=?",
                    (
                        self._db_datetime(now_utc),
                        self._db_datetime(now_utc + timedelta(days=30)),
                        self.owner_user_id,
                        dataset,
                        run.id,
                        run.revision,
                    ),
                )
                if updated.rowcount != 1:
                    raise RuntimeError("notes_graph_run_conflict")

            source_rows = conn.execute(
                "SELECT * FROM note_graph_suggestions WHERE owner_user_id=? AND dataset_id=? "
                "AND source_note_id=? AND state IN ('pending','accepting','rejected') ORDER BY id",
                (self.owner_user_id, dataset, note_id),
            ).fetchall()
            for row in source_rows:
                self._invalidate_suggestion_row(
                    conn,
                    dataset_id=dataset,
                    suggestion=self._suggestion_from_row(row),
                    reason="source_changed",
                    now_utc=now_utc,
                )
            target_rows = conn.execute(
                "SELECT * FROM note_graph_suggestions WHERE owner_user_id=? AND dataset_id=? "
                "AND target_note_id=? AND state IN ('pending','accepting','rejected') ORDER BY id",
                (self.owner_user_id, dataset, note_id),
            ).fetchall()
            for row in target_rows:
                self._invalidate_suggestion_row(
                    conn,
                    dataset_id=dataset,
                    suggestion=self._suggestion_from_row(row),
                    reason="target_changed",
                    now_utc=now_utc,
                )

    def cleanup_retention(
        self,
        *,
        dataset_id: str,
        now: datetime,
        limit: int,
    ) -> dict[str, int]:
        """Delete bounded expired detail without aging out current review state."""

        if not 1 <= limit <= 1000:
            raise ValueError("notes_graph_cleanup_limit_invalid")
        dataset = self._scope(dataset_id)
        now_utc = self._aware_utc(now)
        cutoff_30 = self._db_datetime(now_utc - timedelta(days=30))
        now_value = self._db_datetime(now_utc)

        def delete_fenced(
            conn: SuggestionConnection,
            table: str,
            rows: list[Any],
            fence_columns: tuple[str, ...],
        ) -> int:
            deleted_count = 0
            fence_sql = " AND ".join(f"{column}=?" for column in fence_columns)
            for row in rows:
                deleted = conn.execute(
                    f"DELETE FROM {table} WHERE owner_user_id=? AND dataset_id=? AND id=? "  # nosec B608
                    f"AND {fence_sql}",  # nosec B608
                    (
                        self.owner_user_id,
                        dataset,
                        row["id"],
                        *(row[column] for column in fence_columns),
                    ),
                )
                if deleted.rowcount != 1:
                    raise RuntimeError("notes_graph_cleanup_conflict")
                deleted_count += 1
            return deleted_count

        def mutate(conn: SuggestionConnection) -> dict[str, int]:
            remaining = limit
            counts = {"suggestions": 0, "receipts": 0, "runs": 0, "rejection_sets": 0}
            removable = list(
                conn.execute(
                    "SELECT id,state,revision,expires_at FROM note_graph_suggestions "
                    "WHERE owner_user_id=? AND dataset_id=? AND state IN ('stale','accepted') "
                    "AND expires_at<=? ORDER BY expires_at,id LIMIT ?",
                    (self.owner_user_id, dataset, now_value, remaining),
                ).fetchall()
            )
            if len(removable) < remaining:
                rejected_rows = conn.execute(
                    "SELECT id,state,revision,expires_at,source_note_id,source_fingerprint "
                    "FROM note_graph_suggestions WHERE owner_user_id=? AND dataset_id=? "
                    "AND state='rejected' AND expires_at<=? ORDER BY expires_at,id LIMIT ?",
                    (self.owner_user_id, dataset, now_value, remaining - len(removable)),
                ).fetchall()
                removable.extend(
                    row
                    for row in rejected_rows
                    if self._current_note_fingerprint(conn, str(row["source_note_id"]))
                    != str(row["source_fingerprint"])
                )
            counts["suggestions"] = delete_fenced(
                conn,
                "note_graph_suggestions",
                removable,
                ("state", "revision", "expires_at"),
            )
            remaining -= counts["suggestions"]

            if remaining:
                receipt_rows = conn.execute(
                    "SELECT receipt.id,receipt.state,receipt.request_fingerprint,receipt.expires_at "
                    "FROM note_graph_suggestion_operation_receipts receipt "
                    "WHERE receipt.owner_user_id=? AND receipt.dataset_id=? "
                    "AND receipt.state IN ('completed','failed') AND receipt.expires_at<=? "
                    "AND NOT EXISTS (SELECT 1 FROM note_graph_suggestion_runs run WHERE "
                    "run.owner_user_id=receipt.owner_user_id AND run.dataset_id=receipt.dataset_id "
                    "AND run.admission_receipt_id=receipt.id) "
                    "AND NOT EXISTS (SELECT 1 FROM note_graph_suggestions suggestion WHERE "
                    "suggestion.owner_user_id=receipt.owner_user_id AND suggestion.dataset_id=receipt.dataset_id "
                    "AND suggestion.decision_receipt_id=receipt.id) ORDER BY receipt.expires_at,receipt.id LIMIT ?",
                    (self.owner_user_id, dataset, now_value, remaining),
                ).fetchall()
                counts["receipts"] = delete_fenced(
                    conn,
                    "note_graph_suggestion_operation_receipts",
                    list(receipt_rows),
                    ("state", "request_fingerprint", "expires_at"),
                )
                remaining -= counts["receipts"]

            if remaining:
                run_rows = conn.execute(
                    "SELECT run.id,run.state,run.revision,run.expires_at "
                    "FROM note_graph_suggestion_runs run WHERE run.owner_user_id=? "
                    "AND run.dataset_id=? AND run.state IN ('failed','cancelled','stale','succeeded') "
                    "AND run.expires_at<=? AND NOT EXISTS (SELECT 1 FROM note_graph_suggestions suggestion "
                    "WHERE suggestion.owner_user_id=run.owner_user_id AND suggestion.dataset_id=run.dataset_id "
                    "AND suggestion.run_id=run.id) ORDER BY run.expires_at,run.id LIMIT ?",
                    (self.owner_user_id, dataset, now_value, remaining),
                ).fetchall()
                counts["runs"] = delete_fenced(
                    conn,
                    "note_graph_suggestion_runs",
                    list(run_rows),
                    ("state", "revision", "expires_at"),
                )
                remaining -= counts["runs"]

            if remaining:
                rejection_rows = conn.execute(
                    "SELECT * FROM note_graph_suggestion_rejection_sets WHERE owner_user_id=? "
                    "AND dataset_id=? AND updated_at<=? ORDER BY updated_at,source_note_id LIMIT ?",
                    (self.owner_user_id, dataset, cutoff_30, remaining),
                ).fetchall()
                obsolete = [
                    row
                    for row in rejection_rows
                    if self._current_note_fingerprint(conn, str(row["source_note_id"]))
                    != str(row["source_fingerprint"])
                ]
                for row in obsolete:
                    deleted = conn.execute(
                        "DELETE FROM note_graph_suggestion_rejection_sets WHERE owner_user_id=? "
                        "AND dataset_id=? AND source_note_id=? AND source_fingerprint=? AND revision=?",
                        (
                            self.owner_user_id,
                            dataset,
                            row["source_note_id"],
                            row["source_fingerprint"],
                            row["revision"],
                        ),
                    ).rowcount
                    counts["rejection_sets"] += int(deleted or 0)
            return counts

        return self._with_dataset_scope(dataset, mutate)

    @staticmethod
    def _normalized_sql(value: str) -> str:
        return " ".join(value.lower().split()).rstrip(";")

    def _ensure_fts_ready(self, conn: SuggestionConnection) -> None:
        if self.is_postgres:
            row = conn.execute(
                "SELECT data_type, udt_name FROM information_schema.columns "
                "WHERE table_schema = current_schema() AND table_name = 'notes' "
                "AND column_name = 'notes_fts_tsv'"
            ).fetchone()
            if row is None or row["data_type"] != "tsvector" or row["udt_name"] != "tsvector":
                raise NotesGraphFTSNotReadyError("notes_graph_fts_not_ready")
            trigger = conn.execute(
                "SELECT trigger_row.tgtype, function_row.proname, function_row.prosrc "
                "FROM pg_trigger trigger_row "
                "JOIN pg_proc function_row ON function_row.oid = trigger_row.tgfoid "
                "WHERE trigger_row.tgrelid = 'notes'::regclass "
                "AND trigger_row.tgname = 'update_notes_fts_tsv_trigger' "
                "AND NOT trigger_row.tgisinternal"
            ).fetchone()
            expected_function = self._normalized_sql(
                """
                BEGIN
                    NEW."notes_fts_tsv" := to_tsvector('english', coalesce(NEW."title", '') || ' ' || coalesce(NEW."content", ''));
                    RETURN NEW;
                END;
                """
            )
            if (
                trigger is None
                or int(trigger["tgtype"]) != 23
                or str(trigger["proname"]) != "update_notes_fts_tsv_function"
                or self._normalized_sql(str(trigger["prosrc"])) != expected_function
            ):
                raise NotesGraphFTSNotReadyError("notes_graph_fts_not_ready")
            index = conn.execute(
                "SELECT access_method.amname, index_row.indisvalid, "
                "array_agg(attribute_row.attname ORDER BY key_column.ordinality) AS columns "
                "FROM pg_index index_row "
                "JOIN pg_class index_relation ON index_relation.oid = index_row.indexrelid "
                "JOIN pg_class table_relation ON table_relation.oid = index_row.indrelid "
                "JOIN pg_am access_method ON access_method.oid = index_relation.relam "
                "JOIN unnest(index_row.indkey) WITH ORDINALITY AS key_column(attnum, ordinality) ON true "
                "JOIN pg_attribute attribute_row ON attribute_row.attrelid = table_relation.oid "
                "AND attribute_row.attnum = key_column.attnum "
                "WHERE table_relation.oid = 'notes'::regclass "
                "AND index_relation.relname = 'idx_notes_notes_fts_tsv' "
                "GROUP BY access_method.amname, index_row.indisvalid"
            ).fetchone()
            if (
                index is None
                or str(index["amname"]) != "gin"
                or not bool(index["indisvalid"])
                or tuple(index["columns"]) != ("notes_fts_tsv",)
            ):
                raise NotesGraphFTSNotReadyError("notes_graph_fts_not_ready")
            return
        rows = conn.execute(
            "SELECT type, name, tbl_name, sql FROM sqlite_master WHERE type IN ('table', 'trigger') "
            "AND name IN ('notes_fts', 'notes_ai', 'notes_au', 'notes_ad')"
        ).fetchall()
        by_name = {str(row["name"]): row for row in rows}
        table_definition, trigger_definitions = self._db._notes_fts_sqlite_contract()
        expected_trigger_definitions = {
            definition.split()[2]: self._normalized_sql(definition)
            for definition in trigger_definitions
        }
        table = by_name.get("notes_fts")
        if (
            table is None
            or str(table["type"]) != "table"
            or self._normalized_sql(str(table["sql"] or ""))
            != self._normalized_sql(table_definition)
        ):
            raise NotesGraphFTSNotReadyError("notes_graph_fts_not_ready")
        if set(by_name) != {"notes_fts", *expected_trigger_definitions}:
            raise NotesGraphFTSNotReadyError("notes_graph_fts_not_ready")
        for trigger_name, expected_definition in expected_trigger_definitions.items():
            trigger = by_name.get(trigger_name)
            if (
                trigger is None
                or str(trigger["type"]) != "trigger"
                or str(trigger["tbl_name"]) != "notes"
                or self._normalized_sql(str(trigger["sql"] or "")) != expected_definition
            ):
                raise NotesGraphFTSNotReadyError("notes_graph_fts_not_ready")
        columns = conn.execute("PRAGMA table_info(notes_fts)").fetchall()
        if tuple(str(column["name"]) for column in columns) != ("title", "content"):
            raise NotesGraphFTSNotReadyError("notes_graph_fts_not_ready")

    def load_source_note(self, *, dataset_id: str, note_id: str) -> SuggestionNoteRecord:
        """Load one active owned source only after its SQL byte-limit predicate passes."""

        dataset = self._scope(dataset_id)
        max_bytes = 1_000_000

        def read(conn: SuggestionConnection) -> SuggestionNoteRecord:
            byte_expression = self._source_byte_expression()
            row = conn.execute(
                "SELECT n.id, n.title, n.content, n.version, n.last_modified FROM notes n "
                f"WHERE n.id = ? AND n.client_id = ? AND n.deleted = ? AND ({byte_expression}) <= ?",  # nosec B608
                (note_id, self.owner_user_id, self._deleted_value(), max_bytes),
            ).fetchone()
            if row is not None:
                return SuggestionNoteRecord(
                    note_id=str(row["id"]),
                    title=str(row["title"] or ""),
                    content=str(row["content"] or ""),
                    version=int(row["version"]),
                    last_modified=str(row["last_modified"]),
                )
            size = conn.execute(
                f"SELECT ({byte_expression}) AS byte_count FROM notes n "  # nosec B608
                "WHERE n.id = ? AND n.client_id = ? AND n.deleted = ?",
                (note_id, self.owner_user_id, self._deleted_value()),
            ).fetchone()
            if size is not None and int(size["byte_count"]) > max_bytes:
                raise NotesGraphSourceTooLargeError("notes_graph_source_too_large")
            raise ValueError("Notes graph source is unavailable")

        return self._with_dataset_scope(dataset, read)

    def fetch_ranked_candidates(
        self,
        *,
        dataset_id: str,
        source_note_id: str,
        terms: tuple[str, ...],
        source_fingerprint: str,
        limit: int,
    ) -> tuple[tuple[SuggestionNoteRecord, ...], int, int]:
        """Return at most 60 FTS-ranked byte-safe candidates and an oversized aggregate."""

        del source_fingerprint
        dataset = self._scope(dataset_id)
        if not terms or not 1 <= len(terms) <= 24 or not 1 <= limit <= 60:
            return (), 0, 0

        def read(conn: SuggestionConnection) -> tuple[tuple[SuggestionNoteRecord, ...], int, int]:
            self._ensure_fts_ready(conn)
            byte_expression = self._source_byte_expression()
            direct_exclusion = (
                "NOT EXISTS (SELECT 1 FROM note_edges edge WHERE edge.user_id = ? "
                "AND edge.deleted = ? AND ((edge.from_note_id = ? AND edge.to_note_id = n.id) "
                "OR (edge.to_note_id = ? AND edge.from_note_id = n.id))) "
                "AND NOT EXISTS (SELECT 1 FROM note_wikilink_edges edge WHERE "
                "((edge.source_note_id = ? AND edge.target_note_id = n.id) "
                "OR (edge.target_note_id = ? AND edge.source_note_id = n.id))"
            )
            direct_params: tuple[object, ...] = (
                self.owner_user_id,
                self._deleted_value(),
                source_note_id,
                source_note_id,
                source_note_id,
                source_note_id,
            )
            if self.is_postgres:
                direct_exclusion += " AND edge.owner_user_id = ?"
                direct_params += (self.owner_user_id,)
            direct_exclusion += ")"
            if self.is_postgres:
                tsquery = " | ".join(terms)
                ranked_sql = (
                    "WITH ranked AS MATERIALIZED ("
                    "SELECT n.id AS note_id, ts_rank(n.notes_fts_tsv, to_tsquery('english', ?)) AS rank_value "
                    "FROM notes n WHERE n.client_id = ? AND n.deleted = ? AND n.id <> ? "
                    "AND n.notes_fts_tsv @@ to_tsquery('english', ?) "
                    "ORDER BY rank_value DESC, n.id ASC LIMIT ?"
                    ") SELECT ranked.note_id, "
                    f"({byte_expression}) AS byte_count, "  # nosec B608
                    f"CASE WHEN {direct_exclusion} THEN 0 ELSE 1 END AS direct_connected "  # nosec B608
                    "FROM ranked JOIN notes n ON n.id = ranked.note_id "
                    "ORDER BY ranked.rank_value DESC, ranked.note_id ASC"
                )
                ranked_rows = conn.execute(
                    ranked_sql,
                    (tsquery, self.owner_user_id, self._deleted_value(), source_note_id, tsquery, limit, *direct_params),
                ).fetchall()
            else:
                fts_query = " OR ".join(f'"{term}"' for term in terms)
                ranked_sql = (
                    "WITH ranked AS MATERIALIZED ("
                    "SELECT n.id AS note_id, bm25(notes_fts) AS rank_value "
                    "FROM notes_fts JOIN notes n ON notes_fts.rowid = n.rowid "
                    "WHERE notes_fts MATCH ? AND n.client_id = ? AND n.deleted = ? AND n.id <> ? "
                    "ORDER BY rank_value ASC, n.id ASC LIMIT ?"
                    ") SELECT ranked.note_id, "
                    f"({byte_expression}) AS byte_count, "  # nosec B608
                    f"CASE WHEN {direct_exclusion} THEN 0 ELSE 1 END AS direct_connected "  # nosec B608
                    "FROM ranked JOIN notes n ON n.id = ranked.note_id "
                    "ORDER BY ranked.rank_value ASC, ranked.note_id ASC"
                )
                ranked_rows = conn.execute(
                    ranked_sql,
                    (fts_query, self.owner_user_id, self._deleted_value(), source_note_id, limit, *direct_params),
                ).fetchall()

            oversized_count = sum(
                int(row["byte_count"]) > 250_000 and not bool(row["direct_connected"])
                for row in ranked_rows
            )
            eligible_ids = tuple(
                str(row["note_id"])
                for row in ranked_rows
                if int(row["byte_count"]) <= 250_000 and not bool(row["direct_connected"])
            )
            if not eligible_ids:
                return (), oversized_count, len(ranked_rows)
            placeholders = ", ".join("?" for _ in eligible_ids)
            payload_rows = conn.execute(
                "SELECT n.id, n.title, n.content, n.version, n.last_modified FROM notes n "
                f"WHERE n.id IN ({placeholders}) AND n.client_id = ? AND n.deleted = ? "  # nosec B608
                f"AND ({byte_expression}) <= ?",  # nosec B608
                (*eligible_ids, self.owner_user_id, self._deleted_value(), 250_000),
            ).fetchall()
            records_by_id = {
                str(row["id"]): SuggestionNoteRecord(
                    note_id=str(row["id"]),
                    title=str(row["title"] or ""),
                    content=str(row["content"] or ""),
                    version=int(row["version"]),
                    last_modified=str(row["last_modified"]),
                )
                for row in payload_rows
            }
            return (
                tuple(records_by_id[note_id] for note_id in eligible_ids if note_id in records_by_id),
                oversized_count,
                len(ranked_rows),
            )

        return self._with_dataset_scope(dataset, read)

    def list_rejected_candidate_fingerprints(
        self,
        *,
        dataset_id: str,
        source_note_id: str,
        source_fingerprint: str,
    ) -> frozenset[tuple[str, str]]:
        """Return only exact current-fingerprint relationship rejections for one scope."""

        dataset = self._scope(dataset_id)

        def read(conn: SuggestionConnection) -> frozenset[tuple[str, str]]:
            rows = conn.execute(
                "SELECT target_note_id, target_fingerprint FROM note_graph_suggestions "
                "WHERE owner_user_id = ? AND dataset_id = ? AND source_note_id = ? "
                "AND source_fingerprint = ? AND kind = 'related_note' AND state = 'rejected'",
                (self.owner_user_id, dataset, source_note_id, source_fingerprint),
            ).fetchall()
            return frozenset((str(row["target_note_id"]), str(row["target_fingerprint"])) for row in rows)

        return self._with_dataset_scope(dataset, read)

    def list_tag_catalog(self, *, dataset_id: str, terms: tuple[str, ...], limit: int) -> tuple[str, ...]:
        """Return at most 100 active owner tags with parameterized overlap predicates."""

        dataset = self._scope(dataset_id)
        if not terms or not 1 <= limit <= 100:
            return ()
        predicates = " OR ".join("LOWER(k.keyword) LIKE ?" for _ in terms)
        keyword_table = self._db._map_table_for_backend("keywords")
        params: tuple[object, ...] = (
            self.owner_user_id,
            self._deleted_value(),
            *(f"%{term}%" for term in terms),
            limit,
        )

        def read(conn: SuggestionConnection) -> tuple[str, ...]:
            rows = conn.execute(
                f"SELECT k.keyword FROM {keyword_table} k WHERE k.client_id = ? AND k.deleted = ? "  # nosec B608
                f"AND ({predicates}) ORDER BY LOWER(k.keyword) ASC LIMIT ?",  # nosec B608
                params,
            ).fetchall()
            return tuple(str(row["keyword"]) for row in rows)

        return self._with_dataset_scope(dataset, read)

    def is_projection_fresh(self, *, dataset_id: str, note_id: str) -> bool:
        """Report whether the source wikilink projection matches its current note version."""

        dataset = self._scope(dataset_id)

        def read(conn: SuggestionConnection) -> bool:
            query = (
                "SELECT state.source_version, n.version FROM note_graph_note_state state "
                "JOIN notes n ON n.id = state.note_id WHERE state.note_id = ? AND n.client_id = ?"
            )
            params: tuple[object, ...] = (note_id, self.owner_user_id)
            if self.is_postgres:
                query += " AND state.owner_user_id = ?"
                params += (self.owner_user_id,)
            row = conn.execute(query, params).fetchone()
            return row is not None and int(row["source_version"]) == int(row["version"])

        return self._with_dataset_scope(dataset, read)
