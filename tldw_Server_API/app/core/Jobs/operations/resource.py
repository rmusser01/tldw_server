"""Jobs-owned resource admission and head-before-row lifecycle transactions."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from .. import resource_contracts as contracts
from ..migrations import SLIDES_ARCHIVE_EXACT_FIELDS, normalize_slides_archive_projection
from ..resource_policy import resource_requirement_for
from .contracts import AdmissionRejectionReason, AdmissionResult, CreateJobCommand


@dataclass(frozen=True)
class ResourceCreation:
    """Complete immutable metadata for INSERT, never an unbound-row UPDATE."""

    binding: contracts.ResourceBinding
    settings_json: str
    settings_fingerprint: str

    def values(self) -> tuple[object, ...]:
        """Return values in RESOURCE_METADATA_FIELDS order."""
        return (
            self.binding.scope.resource,
            self.binding.authority_uuid,
            self.binding.generation,
            self.settings_json,
            self.settings_fingerprint,
        )


class ResourceAdmissionStore:
    """Small shared SQL store with fixed backend syntax and parameterized values."""

    def __init__(self, executor: Any, *, postgres: bool = False) -> None:
        self.executor = executor
        self.postgres = postgres
        self.marker = "%s" if postgres else "?"

    def execute(self, statement: str, values: tuple = ()) -> Any:
        """Execute a fixed internal statement; callers never supply SQL fragments."""
        return self.executor.execute(statement, values)

    def timestamp(self, now: datetime) -> datetime | str:
        """Match each backend's persisted UTC timestamp representation."""
        now = now.replace(tzinfo=timezone.utc) if now.tzinfo is None else now.astimezone(timezone.utc)
        return now if self.postgres else now.replace(tzinfo=None).strftime("%Y-%m-%d %H:%M:%S")

    def scope_values(self, scope: contracts.ResourceScope) -> tuple[str, str, str]:
        """Return the single shared identity order used by all head operations."""
        return scope.domain, scope.owner_user_id, scope.resource

    @property
    def scope_where(self) -> str:
        """Render only trusted fixed column names and backend placeholders."""
        p = self.marker
        return f"domain={p} AND owner_user_id={p} AND resource={p}"

    def lock_head(self, scope: contracts.ResourceScope) -> dict[str, Any]:
        """Implemented by backend modules: caller already owns the transaction."""
        raise NotImplementedError

    def resolve_job(self, scope: contracts.ResourceScope, job_uuid: str, *, normalize: bool = True) -> dict[str, Any]:
        """Lock exact live/archive targets after the head, rejecting ambiguity."""
        p = self.marker
        fields = ",".join(("id", *SLIDES_ARCHIVE_EXACT_FIELDS, *contracts.RESOURCE_METADATA_FIELDS))
        lock = " FOR UPDATE" if self.postgres else ""
        values = (job_uuid, scope.domain, scope.owner_user_id)
        live = self.execute(
            f"SELECT {fields}, NULL AS payload_compressed, NULL AS result_compressed, "  # nosec B608 - fixed projection/syntax; values bound.
            f"0 AS archived FROM jobs WHERE uuid={p} AND domain={p} AND owner_user_id={p}{lock}",
            values,
        ).fetchall()
        archived = self.execute(
            f"SELECT {fields}, payload_compressed, result_compressed, 1 AS archived "  # nosec B608 - fixed projection/syntax; values bound.
            f"FROM jobs_archive WHERE uuid={p} AND domain={p} AND owner_user_id={p}{lock}",
            values,
        ).fetchall()
        rows = [*live, *archived]
        if len(rows) != 1:
            raise RuntimeError("resource Job target is missing or ambiguous")
        return normalize_slides_archive_projection(rows[0]) if normalize else dict(rows[0])

    def active_jobs(self, scope: contracts.ResourceScope, command: CreateJobCommand) -> list[dict[str, Any]]:
        """Detect bound or legacy active candidates across all queues/methods."""
        p = self.marker
        lock = " FOR UPDATE" if self.postgres else ""
        alias_prefix = command.job_type.replace("!", "!!").replace("%", "!%").replace("_", "!_") + ".v%"
        rows = self.execute(
            f"SELECT * FROM jobs WHERE domain={p} AND owner_user_id={p} "  # nosec B608 - backend placeholders only; values bound.
            f"AND (resource={p} OR job_type={p} OR job_type LIKE {p} ESCAPE '!') "
            f"AND status IN ('queued','processing') ORDER BY id{lock}",
            (*self.scope_values(scope), command.job_type, alias_prefix),
        ).fetchall()
        active = []
        for row in rows:
            try:
                required = resource_requirement_for(row["domain"], row["job_type"])
            except ValueError:
                # Reserved but unsupported aliases cannot hide a legacy active Job.
                required = scope.resource
            if row["resource"] == scope.resource or required == scope.resource:
                active.append(dict(row))
        return active

    def has_bound_history(self, scope: contracts.ResourceScope) -> bool:
        """A missing head must not mint a new authority over retained bound Jobs."""
        row = self.execute(
            f"SELECT 1 FROM jobs WHERE {self.scope_where} "  # nosec B608 - fixed predicate/placeholders; values bound.
            f"UNION ALL SELECT 1 FROM jobs_archive WHERE {self.scope_where} LIMIT 1",
            (*self.scope_values(scope), *self.scope_values(scope)),
        ).fetchone()
        return row is not None

    def receipt(self, scope: contracts.ResourceScope, digest: str, now: datetime) -> dict[str, Any] | None:
        """Read only retained keys under the resource lock."""
        p = self.marker
        row = self.execute(
            f"SELECT * FROM job_resource_request_receipts WHERE {self.scope_where} "  # nosec B608 - fixed predicate/placeholders; values bound.
            f"AND request_key_digest={p} AND expires_at>{p}",
            (*self.scope_values(scope), digest, self.timestamp(now)),
        ).fetchone()
        return dict(row) if row else None

    def attach_receipt(self, request: contracts.ResourceAdmissionRequest, job: dict[str, Any], now: datetime) -> bool:
        """Attach a bounded 24-hour alias; existing retained aliases were read first."""
        if request.request_key is None:
            return True
        p = self.marker
        scope = self.scope_values(request.scope)
        row = self.execute(
            f"SELECT COUNT(*) AS c FROM job_resource_request_receipts WHERE {self.scope_where} "  # nosec B608 - fixed predicate/placeholders; values bound.
            f"AND job_uuid={p} AND expires_at>{p}",
            (*scope, job["uuid"], self.timestamp(now)),
        ).fetchone()
        count = row["c"] if hasattr(row, "keys") else row[0]
        if count >= contracts.RESOURCE_REQUEST_MAX_ALIASES:
            return False
        digest = hashlib.sha256(request.request_key.encode("ascii")).hexdigest()
        self.execute(
            f"DELETE FROM job_resource_request_receipts WHERE {self.scope_where} "  # nosec B608 - fixed predicate/placeholders; values bound.
            f"AND request_key_digest={p} AND expires_at<={p}",
            (*scope, digest, self.timestamp(now)),
        )
        self.execute(
            "INSERT INTO job_resource_request_receipts "  # nosec B608 - fixed columns/placeholders; values bound.
            "(domain,owner_user_id,resource,request_key_digest,settings_json,settings_fingerprint,job_uuid,expires_at) "
            f"VALUES ({','.join([p] * 8)})",
            (
                *scope,
                digest,
                job["resource_settings_json"],
                job["resource_settings_fingerprint"],
                job["uuid"],
                self.timestamp(now + timedelta(seconds=contracts.RESOURCE_REQUEST_RETENTION_SECONDS)),
            ),
        )
        return True

    def attach_created(self, request: contracts.ResourceAdmissionRequest, job: dict[str, Any], now: datetime) -> None:
        """Publish the new head and alias in the existing insert/event transaction."""
        p = self.marker
        self.execute(
            f"UPDATE job_resource_heads SET generation={p},active_job_uuid={p} WHERE {self.scope_where}",  # nosec B608 - fixed predicate/placeholders; values bound.
            (job["resource_generation"], job["uuid"], *self.scope_values(request.scope)),
        )
        if not self.attach_receipt(request, job, now):
            raise RuntimeError("resource alias limit for newly created Job")


def _validate_target(
    job: dict[str, Any], head: dict[str, Any], command: CreateJobCommand, request: contracts.ResourceAdmissionRequest
) -> None:
    """Fail closed on mismatched identity, authority, generation or settings encoding."""
    try:
        binding = contracts.ResourceBinding(request.scope, job["resource_authority_uuid"], job["resource_generation"])
        settings = json.loads(job["resource_settings_json"])
        valid = (
            job["domain"] == request.scope.domain
            and job["owner_user_id"] == request.scope.owner_user_id
            and job["resource"] == request.scope.resource
            and job["job_type"] == command.job_type
            and binding.authority_uuid == head["authority_uuid"]
            and binding.generation <= head["generation"]
            and job["resource_settings_json"] == contracts.canonical_settings_json(settings)
            and job["resource_settings_fingerprint"] == contracts.settings_fingerprint(settings)
        )
    except (KeyError, TypeError, ValueError):
        valid = False
    if not valid:
        raise RuntimeError("resource Job authority is invalid")


def _conflict_reference(job: dict[str, Any]) -> dict[str, Any]:
    """Project an already validated target without payload, result or raw errors."""
    return {field: job[field] for field in ("id", "uuid", "domain", "owner_user_id", "job_type", "resource")}


def resource_scope_for(row: dict[str, Any]) -> contracts.ResourceScope | None:
    """Classify opt-out rows, rejecting partial or unbound registered state."""
    required = resource_requirement_for(row["domain"], row["job_type"])
    metadata = [row.get(field) for field in contracts.RESOURCE_METADATA_FIELDS]
    if required is None and all(value is None for value in metadata):
        return None
    if required is None or any(value is None for value in metadata) or row["resource"] != required:
        raise ValueError("resource Job binding is invalid")
    return contracts.ResourceScope(row["domain"], row["owner_user_id"], row["resource"])


class ResourceMutation:
    """Freeze a mutation's exact targets under head-before-row locks.

    The caller owns the transaction and must use ``where``/``params`` for broad
    SQL writes. PostgreSQL discovery is deliberately nonlocking; identities and
    predicates are re-read after every head lock. No missing head is repaired.
    """

    def __init__(
        self,
        store: ResourceAdmissionStore,
        *,
        where: str,
        params: tuple = (),
        archived: bool = False,
        reactivating: bool = False,
        pruning: bool = False,
        read_only: bool = False,
        batch: bool = False,
        expected_uuid: str | None = None,
        clock: Callable[[], datetime] | None = None,
        limit: int | None = None,
        expired_recovery_limit: int | None = None,
    ) -> None:
        """Describe one closed internal mutation; transaction ownership stays outside."""
        self.store = store
        self.table = "jobs_archive" if archived else "jobs"
        self.original_where = where
        self.original_params = params
        self.reactivating = reactivating
        self.pruning = pruning
        self.read_only = read_only
        self.batch = batch
        self.expected_uuid = expected_uuid
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.limit = limit
        self.expired_recovery_limit = expired_recovery_limit
        self.rows: dict[int, dict[str, Any]] = {}
        self.heads: dict[contracts.ResourceScope, dict[str, Any]] = {}
        self.bound: dict[int, contracts.ResourceScope] = {}
        self.where = "1=0"
        self.params: tuple = ()
        self.finished = False

    def _select(
        self, where: str, params: tuple, *, lock: bool = False, limit: int | None = None
    ) -> list[dict[str, Any]]:
        """Read a fixed internal table and caller-built parameterized predicate."""
        suffix = " FOR UPDATE" if lock and self.store.postgres else ""
        order = "id"
        if self.expired_recovery_limit is not None:
            order = "leased_until ASC NULLS FIRST,id" if self.store.postgres else "leased_until ASC,id"
            if lock:
                limit = self.expired_recovery_limit
                if self.store.postgres:
                    suffix += " SKIP LOCKED"
        limit_sql = ""
        if limit is not None:
            limit_sql = " LIMIT %s" if self.store.postgres else " LIMIT ?"
            params = (*params, int(limit))
        return [
            dict(row)
            for row in self.store.execute(
                f"SELECT * FROM {self.table} WHERE {where} ORDER BY {order}{limit_sql}{suffix}",  # nosec B608 - internal table/predicate; values bound.
                params,
            ).fetchall()
        ]

    def __enter__(self) -> ResourceMutation:
        """Discover, lock heads, then lock/revalidate the exact candidate population."""
        preliminary = self._select(self.original_where, self.original_params, limit=self.limit)
        if self.expected_uuid is not None:
            preliminary = [row for row in preliminary if str(row["uuid"] or "") == str(self.expected_uuid)]
        candidates = {}
        for row in preliminary:
            scope = resource_scope_for(row)
            if scope is not None:
                candidates.setdefault(scope, row)
        if candidates and self.table == "jobs_archive" and len({row["id"] for row in preliminary}) != len(preliminary):
            raise RuntimeError("resource Job target is missing or ambiguous")
        if (
            not candidates
            and not self.batch
            and not self.pruning
            and self.limit is None
            and self.expired_recovery_limit is None
        ):
            # Legacy conditional UPDATE/CTE operations own their row-lock point.
            # Freeze UUIDs without moving that point or admitting unseen rows.
            self.rows = {row["id"]: row for row in preliminary}
            self._freeze_rows(preliminary)
            return self
        p = self.store.marker
        for scope in sorted(candidates, key=self.store.scope_values):
            suffix = " FOR UPDATE" if self.store.postgres else ""
            row = self.store.execute(
                f"SELECT * FROM job_resource_heads WHERE {self.store.scope_where}{suffix}",  # nosec B608 - fixed predicate/placeholders; values bound.
                self.store.scope_values(scope),
            ).fetchone()
            if row is None:
                raise RuntimeError("resource head is missing")
            self.heads[scope] = dict(row)
        for scope, head in self.heads.items():
            candidate = candidates[scope]
            command = CreateJobCommand(scope.domain, candidate["queue"], candidate["job_type"], {}, scope.owner_user_id)
            active = self.store.active_jobs(scope, command)
            if head["active_job_uuid"] is None:
                if active:
                    raise RuntimeError("resource has ambiguous or unbound active Jobs")
            else:
                target = self.store.resolve_job(scope, head["active_job_uuid"], normalize=False)
                _validate_target(
                    target,
                    head,
                    command,
                    contracts.ResourceAdmissionRequest(scope, json.loads(target["resource_settings_json"])),
                )
                if (
                    len(active) != 1
                    or active[0]["uuid"] != head["active_job_uuid"]
                    or target["status"] not in ("queued", "processing")
                    or target.get("archived")
                ):
                    raise RuntimeError("resource active authority is invalid")
        if not preliminary:
            return self
        ids = [row["id"] for row in preliminary]
        identity_where, identity_params = self._ids_predicate(ids)
        locked = self._select(
            f"({self.original_where}) AND {identity_where}", (*self.original_params, *identity_params), lock=True
        )
        original = {row["id"]: row for row in preliminary}
        reservations: dict[contracts.ResourceScope, str] = {}
        for row in locked:
            if row["uuid"] != original[row["id"]]["uuid"]:
                continue
            scope = resource_scope_for(row)
            if scope is not None:
                if scope not in self.heads:
                    raise RuntimeError("resource identity changed during locking")
                head = self.heads[scope]
                settings = json.loads(row["resource_settings_json"])
                command = CreateJobCommand(row["domain"], row["queue"], row["job_type"], {}, row["owner_user_id"])
                request = contracts.ResourceAdmissionRequest(scope, settings)
                _validate_target(row, head, command, request)
                self.store.resolve_job(scope, row["uuid"], normalize=False)
                active = head["active_job_uuid"]
                if row["status"] in ("queued", "processing") and active != row["uuid"]:
                    raise RuntimeError("resource active authority is invalid")
                if self.reactivating and reservations.get(scope, active) not in (None, row["uuid"]):
                    continue
                if self.reactivating:
                    # A batch can reactivate only one logical Job in each slot.
                    reservations[scope] = row["uuid"]
                if self.pruning:
                    self.store.execute(
                        f"DELETE FROM job_resource_request_receipts WHERE {self.store.scope_where} AND expires_at<={p}",  # nosec B608 - fixed predicate/placeholders; values bound.
                        (*self.store.scope_values(scope), self.store.timestamp(self.clock())),
                    )
                    receipt = self.store.execute(
                        f"SELECT 1 FROM job_resource_request_receipts WHERE {self.store.scope_where} "  # nosec B608 - fixed predicate/placeholders; values bound.
                        f"AND job_uuid={p} AND expires_at>{p} LIMIT 1",
                        (*self.store.scope_values(scope), row["uuid"], self.store.timestamp(self.clock())),
                    ).fetchone()
                    if receipt:
                        # Retain the exact live target, even with archive disabled.
                        continue
                self.bound[row["id"]] = scope
            self.rows[row["id"]] = row
        self._freeze_rows()
        return self

    def _freeze_rows(self, rows: list[dict[str, Any]] | None = None) -> None:
        """Fence all broad writes to discovered numeric IDs AND logical UUIDs."""
        rows = list(self.rows.values()) if rows is None else rows
        if not rows:
            return
        targets = json.dumps([{"id": row["id"], "uuid": row["uuid"]} for row in rows])
        predicates = {
            (False, "jobs"): "EXISTS (SELECT 1 FROM json_each(?) AS resource_targets "
            "WHERE jobs.id=json_extract(resource_targets.value,'$.id') "
            "AND jobs.uuid IS json_extract(resource_targets.value,'$.uuid'))",
            (False, "jobs_archive"): "EXISTS (SELECT 1 FROM json_each(?) AS resource_targets "
            "WHERE jobs_archive.id=json_extract(resource_targets.value,'$.id') "
            "AND jobs_archive.uuid IS json_extract(resource_targets.value,'$.uuid'))",
            (
                True,
                "jobs",
            ): "EXISTS (SELECT 1 FROM jsonb_to_recordset(%s::jsonb) AS resource_targets(id bigint,uuid text) "
            "WHERE jobs.id=resource_targets.id AND jobs.uuid IS NOT DISTINCT FROM resource_targets.uuid)",
            (
                True,
                "jobs_archive",
            ): "EXISTS (SELECT 1 FROM jsonb_to_recordset(%s::jsonb) AS resource_targets(id bigint,uuid text) "
            "WHERE jobs_archive.id=resource_targets.id AND jobs_archive.uuid IS NOT DISTINCT FROM resource_targets.uuid)",
        }
        self.where = f"({self.original_where}) AND {predicates[self.store.postgres, self.table]}"
        self.params = (*self.original_params, targets)

    def _ids_predicate(self, ids: list[int]) -> tuple[str, tuple]:
        """Bound large maintenance populations without SQLite's variable limit."""
        if self.store.postgres:
            return "id = ANY(%s)", (ids,)
        return "id IN (SELECT value FROM json_each(?))", (json.dumps(ids),)

    def worker_allowed(
        self, job_id: int, worker_id: str | None, lease_id: str | None, expected_uuid: str | None = None
    ) -> bool:
        """Resource workers need actual live attempt authority, even ack-off.

        Calls without any worker identity remain intentional Jobs admin overrides.
        They are not a resource worker API or a cancellation promise.
        """
        row = self.rows.get(job_id)
        if row is None or job_id not in self.bound:
            return True
        if worker_id is None and lease_id is None:
            return True
        if row["status"] != "processing":
            # Token replay of a terminal operation is harmless; a queued Job
            # has no acquired attempt, even when admin overrides are enabled.
            return row["status"] in ("completed", "failed", "cancelled", "quarantined")
        until = row.get("leased_until")
        try:
            until = until if isinstance(until, datetime) else datetime.fromisoformat(str(until).replace("Z", "+00:00"))
            until = until.replace(tzinfo=timezone.utc) if until.tzinfo is None else until.astimezone(timezone.utc)
        except ValueError:
            return False
        now = self.clock()
        now = now.replace(tzinfo=timezone.utc) if now.tzinfo is None else now.astimezone(timezone.utc)
        return bool(
            worker_id
            and lease_id
            and row["worker_id"] == worker_id
            and row["lease_id"] == lease_id
            and until > now
            and (expected_uuid is None or row["uuid"] == expected_uuid)
        )

    def item_allowed(self, item: dict[str, Any]) -> bool:
        """Fence batch items before any payload, counter or event writes."""
        job_id = int(item["job_id"])
        row = self.rows.get(job_id)
        expected_uuid = item.get("expected_uuid")
        return bool(
            row
            and (expected_uuid is None or row["uuid"] == expected_uuid)
            and self.worker_allowed(job_id, item.get("worker_id"), item.get("lease_id"), expected_uuid)
        )

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        """Close only successful transitions, before the caller can commit."""
        if exc_type is not None or self.finished:
            return
        self.finish()

    def finish(self) -> None:
        """Close head transitions before an existing explicit commit, once only."""
        if self.finished or self.read_only:
            return
        p = self.store.marker
        for job_id, scope in self.bound.items():
            before = self.rows[job_id]
            current = self.store.execute(
                (
                    "SELECT * FROM jobs WHERE id=%s AND uuid=%s"
                    if self.store.postgres
                    else "SELECT * FROM jobs WHERE id=? AND uuid=?"
                ),
                (job_id, before["uuid"]),
            ).fetchone()
            if (
                self.batch
                and current
                and before["status"] == "processing"
                and current["status"] in ("completed", "failed")
            ):
                if os.getenv("JOBS_COUNTERS_ENABLED", "").lower() in ("1", "true", "yes", "on"):
                    if self.store.postgres:
                        self.store.execute(
                            "UPDATE job_counters SET processing_count=GREATEST(processing_count-1,0), updated_at=NOW() "
                            "WHERE domain=%s AND queue=%s AND job_type=%s",
                            (before["domain"], before["queue"], before["job_type"]),
                        )
                    else:
                        self.store.execute(
                            "UPDATE job_counters SET processing_count=MAX(processing_count-1,0), updated_at=DATETIME('now') "
                            "WHERE domain=? AND queue=? AND job_type=?",
                            (before["domain"], before["queue"], before["job_type"]),
                        )
                if os.getenv("JOBS_EVENTS_OUTBOX", "").lower() in ("1", "true", "yes", "on"):
                    self.store.execute(
                        "INSERT INTO job_events(job_id,domain,queue,job_type,event_type,attrs_json,owner_user_id,request_id,trace_id) "  # nosec B608 - backend placeholders only; values bound.
                        f"VALUES ({','.join([p] * 9)})",
                        (
                            job_id,
                            before["domain"],
                            before["queue"],
                            before["job_type"],
                            "job." + current["status"],
                            "{}",
                            before["owner_user_id"],
                            before.get("request_id"),
                            before.get("trace_id"),
                        ),
                    )
            if current and current["status"] in ("queued", "processing"):
                head = self.heads[scope]
                if head["active_job_uuid"] not in (None, before["uuid"]):
                    raise RuntimeError("resource active conflict")
                if head["active_job_uuid"] is None:
                    self.store.execute(
                        f"UPDATE job_resource_heads SET active_job_uuid={p} WHERE {self.store.scope_where}",  # nosec B608 - fixed predicate/placeholders; values bound.
                        (before["uuid"], *self.store.scope_values(scope)),
                    )
                head["active_job_uuid"] = before["uuid"]
            elif not self.table == "jobs_archive":
                self.store.execute(
                    f"UPDATE job_resource_heads SET active_job_uuid=NULL WHERE {self.store.scope_where} AND active_job_uuid={p}",  # nosec B608 - fixed predicate/placeholders; values bound.
                    (*self.store.scope_values(scope), before["uuid"]),
                )
                if self.heads[scope]["active_job_uuid"] == before["uuid"]:
                    self.heads[scope]["active_job_uuid"] = None
        self.finished = True


@contextmanager
def resource_mutation(executor: Any, *, postgres: bool = False, **kwargs: Any) -> Iterator[ResourceMutation]:
    """Compose a lifecycle guard with an already-owned backend transaction."""
    payload_replacement = kwargs.pop("payload_replacement", False)
    try:
        with ResourceMutation(ResourceAdmissionStore(executor, postgres=postgres), **kwargs) as mutation:
            yield mutation
    except (ValueError, RuntimeError) as exc:
        if payload_replacement:
            raise ValueError("resource_payload_conflict") from exc
        raise


def lookup_resource_admission(
    store: ResourceAdmissionStore,
    *,
    command: CreateJobCommand,
    request: contracts.ResourceAdmissionRequest,
    now: datetime,
) -> contracts.ResourceAdmission | ResourceCreation:
    """Resolve retained replay/active reuse before any creation-only rejection."""
    head = store.lock_head(request.scope)
    # Validate even an empty head, without allocating a logical generation yet.
    contracts.ResourceBinding(request.scope, head["authority_uuid"], head["generation"] + 1)
    encoded = contracts.canonical_settings_json(request.canonical_settings)
    fingerprint = contracts.settings_fingerprint(request.canonical_settings)
    if request.request_key is not None:
        digest = hashlib.sha256(request.request_key.encode("ascii")).hexdigest()
        receipt = store.receipt(request.scope, digest, now)
        if receipt is not None:
            job = store.resolve_job(request.scope, receipt["job_uuid"])
            _validate_target(job, head, command, request)
            if (receipt["settings_json"], receipt["settings_fingerprint"]) != (
                job["resource_settings_json"],
                job["resource_settings_fingerprint"],
            ):
                raise RuntimeError("resource receipt authority is invalid")
            equal = (receipt["settings_json"], receipt["settings_fingerprint"]) == (encoded, fingerprint)
            return contracts.ResourceAdmission(
                "request_replayed" if equal else "request_conflict", job if equal else _conflict_reference(job), None
            )

    if head["generation"] == 0 and store.has_bound_history(request.scope):
        raise RuntimeError("resource head is missing for retained bound Jobs")
    active = store.active_jobs(request.scope, command)
    target = head["active_job_uuid"]
    if target is not None:
        job = store.resolve_job(request.scope, target)
        _validate_target(job, head, command, request)
        if (
            len(active) != 1
            or active[0]["uuid"] != target
            or job.get("archived")
            or job["status"] not in ("queued", "processing")
        ):
            raise RuntimeError("resource active authority is invalid")
        if (job["resource_settings_json"], job["resource_settings_fingerprint"]) != (encoded, fingerprint):
            return contracts.ResourceAdmission("active_conflict", _conflict_reference(job), None)
        if not store.attach_receipt(request, job, now):
            return contracts.ResourceAdmission(
                "rejected",
                None,
                AdmissionResult.rejected(
                    AdmissionRejectionReason.QUOTA_EXCEEDED,
                    message="Resource request alias quota exceeded",
                ),
            )
        return contracts.ResourceAdmission("active_reused", job, None)
    if active:
        raise RuntimeError("resource has ambiguous or unbound active Jobs")
    return ResourceCreation(
        contracts.ResourceBinding(request.scope, head["authority_uuid"], head["generation"] + 1), encoded, fingerprint
    )
