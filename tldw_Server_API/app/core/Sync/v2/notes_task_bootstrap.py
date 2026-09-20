"""Bounded, resumable private bootstrap for canonical Notes tasks."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.exceptions import NotesTaskBootstrapInterrupted

from .errors import SyncStoreError
from .models import SyncDataset, SyncEnvelope
from .server_origin_batch import (
    ServerOriginMutationStep,
    capture_server_origin_mutation_batch,
)
from .service import SyncV2Service

_SOURCE = "notes-task-bootstrap"
_EMPTY_FINGERPRINT = hashlib.sha256(b"notes.task.bootstrap.v1").hexdigest()


class _SourceInvalid(RuntimeError):
    """Internal marker for an unreadable or noncanonical task source."""


class _SourceChanged(RuntimeError):
    """Internal marker for source progress that no longer matches its digest."""


class NotesTaskBootstrapper:
    """Capture at most one source-verified task page per invocation."""

    PAGE_LIMIT = 500

    def __init__(
        self,
        note_db: CharactersRAGDB,
        *,
        page_limit: int = PAGE_LIMIT,
        after_page: Callable[[int], None] | None = None,
    ) -> None:
        """Configure one bounded bootstrap worker over a canonical task store."""

        if isinstance(page_limit, bool) or not 1 <= page_limit <= self.PAGE_LIMIT:
            raise ValueError("Notes task bootstrap page limit must be 1..500")
        self._db = note_db
        self._tasks = note_db.task_store
        self._page_limit = page_limit
        self._after_page = after_page

    def bootstrap(
        self,
        *,
        service: SyncV2Service,
        dataset: SyncDataset,
    ) -> SyncDataset:
        """Resume one bounded page and return the durable readiness state."""

        owner = dataset.owner_user_id
        current = service.store.get_dataset(
            dataset.dataset_id,
            owner_user_id=owner,
        )
        if current is None:
            raise SyncStoreError("Sync dataset was not found or is not accessible")
        if self._tasks.resolve_task_compatibility_dataset_id(
            owner_user_id=owner
        ) != current.dataset_id:
            raise SyncStoreError("notes_task_readiness_source_scope_invalid")

        current = self._ensure_bootstrapping(service, current)
        state = _readiness(current)
        if state["state"] == "ready":
            return current
        if state["state"] != "bootstrapping":
            raise SyncStoreError("notes_task_sync_not_ready")

        bootstrap_id = _bootstrap_id(owner, current.dataset_id)
        try:
            source = self._source_summary(owner, current.dataset_id)
            prefix_fingerprint = _verify_stored_progress(
                service=service,
                dataset_id=current.dataset_id,
                bootstrap_id=bootstrap_id,
                state=state,
                source=source,
            )
            after_task_id = _optional_string(state.get("source_cursor"))
            page = self._tasks.page_tasks_for_sync_bootstrap(
                owner_user_id=owner,
                dataset_id=current.dataset_id,
                after_task_id=after_task_id,
                limit=self._page_limit,
            )
            running_count = int(state["source_count"])
            running_fingerprint = prefix_fingerprint
            for row in page:
                self._capture_row(
                    service=service,
                    dataset=current,
                    bootstrap_id=bootstrap_id,
                    row=row,
                )
                running_count += 1
                running_fingerprint = _task_bootstrap_fingerprint(
                    running_fingerprint,
                    str(row["id"]),
                    str(row["canonical_hash"]),
                )

            if page and self._after_page is not None:
                self._after_page(1)
            if page:
                current = service.store.transition_notes_task_readiness(
                    current.dataset_id,
                    owner_user_id=owner,
                    expected_state="bootstrapping",
                    state="bootstrapping",
                    source_dataset_id=current.dataset_id,
                    source_cursor=str(page[-1]["id"]),
                    source_count=running_count,
                    source_fingerprint=running_fingerprint,
                )
            if running_count < source.count:
                return current
            verified = self._source_summary(owner, current.dataset_id)
            if verified != source:
                verified_state = _readiness(current)
                running_fingerprint = _verify_stored_progress(
                    service=service,
                    dataset_id=current.dataset_id,
                    bootstrap_id=bootstrap_id,
                    state=verified_state,
                    source=verified,
                )
                running_count = int(verified_state["source_count"])
                source = verified
                if running_count < source.count:
                    return current
            if (
                running_count != source.count
                or running_fingerprint != source.fingerprint
                or not _sync_bootstrap_matches_source(
                    service,
                    current.dataset_id,
                    bootstrap_id=bootstrap_id,
                    source=source,
                )
            ):
                raise _SourceChanged
            final_state = _readiness(current)
            current = service.store.transition_notes_task_readiness(
                current.dataset_id,
                owner_user_id=owner,
                expected_state="bootstrapping",
                state="verifying",
                source_dataset_id=current.dataset_id,
                source_cursor=source.cursor,
                source_count=source.count,
                source_fingerprint=source.fingerprint,
                captured_source_rebase=(
                    source.fingerprint != final_state.get("source_fingerprint")
                    and source.cursor == final_state.get("source_cursor")
                    and source.count == final_state.get("source_count")
                ),
            )
            return service.store.transition_notes_task_readiness(
                current.dataset_id,
                owner_user_id=owner,
                expected_state="verifying",
                state="ready",
                source_dataset_id=current.dataset_id,
                source_cursor=source.cursor,
                source_count=source.count,
                source_fingerprint=source.fingerprint,
            )
        except NotesTaskBootstrapInterrupted:
            raise
        except _SourceChanged:
            return _block(service, current, reason="notes_task_source_changed")
        except Exception as exc:  # noqa: BLE001 - source errors become bounded state.
            if isinstance(exc, SyncStoreError) and str(exc) not in {
                "notes_task_source_invalid",
                "notes_task_source_changed",
            }:
                raise
            return _block(service, current, reason="notes_task_source_invalid")

    @property
    def note_db(self) -> CharactersRAGDB:
        """Return the product database whose task graph is being bootstrapped."""

        return self._db

    def _ensure_bootstrapping(
        self,
        service: SyncV2Service,
        dataset: SyncDataset,
    ) -> SyncDataset:
        """Enter or resume the task bootstrapping phase for one dataset."""

        state = dataset.metadata.get("notes_task_v1")
        if isinstance(state, Mapping) and state.get("state") in {
            "bootstrapping",
            "ready",
        }:
            return dataset
        if (
            isinstance(state, Mapping)
            and state.get("state") == "blocked"
            and state.get("resume_phase") == "bootstrapping"
        ):
            return service.store.transition_notes_task_readiness(
                dataset.dataset_id,
                owner_user_id=dataset.owner_user_id,
                expected_state="blocked",
                state="bootstrapping",
                source_dataset_id=dataset.dataset_id,
                source_cursor=_optional_string(state.get("source_cursor")),
                source_count=int(state["source_count"]),
                source_fingerprint=(
                    str(state["source_fingerprint"])
                    if state.get("source_fingerprint") is not None
                    else None
                ),
            )
        if isinstance(state, Mapping) and state.get("state") == "enrolling":
            return service.store.transition_notes_task_readiness(
                dataset.dataset_id,
                owner_user_id=dataset.owner_user_id,
                expected_state="enrolling",
                state="bootstrapping",
                source_dataset_id=dataset.dataset_id,
                source_cursor=None,
                source_count=0,
                source_fingerprint=_EMPTY_FINGERPRINT,
            )
        if state is not None:
            raise SyncStoreError("notes_task_sync_not_ready")
        dataset = service.store.transition_notes_task_readiness(
            dataset.dataset_id,
            owner_user_id=dataset.owner_user_id,
            expected_state="not_enrolled",
            state="enrolling",
            source_dataset_id=dataset.dataset_id,
            source_cursor=None,
            source_count=0,
            source_fingerprint=None,
        )
        dataset = service.store.transition_notes_task_activity_readiness(
            dataset.dataset_id,
            owner_user_id=dataset.owner_user_id,
            expected_state="not_enrolled",
            state="enrolling",
            source_dataset_id=dataset.dataset_id,
            source_cursor=None,
            source_count=0,
            source_fingerprint=None,
            task_activity_capture_enabled=True,
        )
        return service.store.transition_notes_task_readiness(
            dataset.dataset_id,
            owner_user_id=dataset.owner_user_id,
            expected_state="enrolling",
            state="bootstrapping",
            source_dataset_id=dataset.dataset_id,
            source_cursor=None,
            source_count=0,
            source_fingerprint=_EMPTY_FINGERPRINT,
        )

    def _source_summary(self, owner: str, dataset_id: str) -> _SourceSummary:
        """Read the full canonical source identity using bounded keyset pages."""

        rows: list[tuple[str, str, int, str, str]] = []
        cursor: str | None = None
        fingerprint = _EMPTY_FINGERPRINT
        try:
            while True:
                page = self._tasks.page_tasks_for_sync_bootstrap(
                    owner_user_id=owner,
                    dataset_id=dataset_id,
                    after_task_id=cursor,
                    limit=self.PAGE_LIMIT,
                )
                for row in page:
                    task_id = str(row["id"])
                    canonical_hash = str(row["canonical_hash"])
                    rows.append(
                        (
                            task_id,
                            canonical_hash,
                            int(row["canonical_revision"]),
                            "tombstone" if bool(row.get("deleted")) else "upsert",
                            str(row["note_id"]),
                        )
                    )
                    fingerprint = _task_bootstrap_fingerprint(
                        fingerprint,
                        task_id,
                        canonical_hash,
                    )
                    cursor = task_id
                if len(page) < self.PAGE_LIMIT:
                    break
        except Exception as exc:  # noqa: BLE001 - product details stay private.
            raise _SourceInvalid from exc
        return _SourceSummary(tuple(rows), cursor, len(rows), fingerprint)

    def _capture_row(
        self,
        *,
        service: SyncV2Service,
        dataset: SyncDataset,
        bootstrap_id: str,
        row: Mapping[str, object],
    ) -> None:
        """Capture one source-verified task row through the server-origin path."""

        task_id = str(row["id"])
        canonical_hash = str(row["canonical_hash"])
        current_head = service.store.get_current_head(
            dataset.dataset_id,
            "notes.task",
            task_id,
        )
        if current_head is not None and _task_head_matches_row(current_head, row):
            return
        payload = row.get("sync_payload")
        if not isinstance(payload, Mapping):
            raise _SourceInvalid
        operation = "tombstone" if bool(row.get("deleted")) else "upsert"
        step = ServerOriginMutationStep(
            domain="notes.task",
            operation=operation,
            object_id=task_id,
            parent_id=str(row["note_id"]),
            payload=dict(payload),
            routing_metadata=_task_bootstrap_routing(bootstrap_id),
            client_envelope_id=_task_bootstrap_envelope_id(
                bootstrap_id,
                task_id,
                canonical_hash,
            ),
            object_revision=int(row["canonical_revision"]),
        )

        def source_matches(envelope: SyncEnvelope) -> bool:
            """Return whether the product row still matches the planned envelope."""

            try:
                current = self._tasks.get_task(
                    owner_user_id=dataset.owner_user_id,
                    dataset_id=dataset.dataset_id,
                    task_id=task_id,
                    include_deleted=True,
                )
                if current is None:
                    return False
                verified = self._tasks._sync_bootstrap_task_row(
                    current,
                    dataset.owner_user_id,
                )
            except Exception:  # noqa: BLE001 - verifier returns only a boolean.
                return False
            return bool(
                envelope.client_envelope_id == step.client_envelope_id
                and envelope.object_id == task_id
                and envelope.parent_id == row["note_id"]
                and envelope.operation == operation
                and envelope.object_revision == row["canonical_revision"]
                and envelope.payload_hash == canonical_hash
                and dict(envelope.payload) == dict(verified["sync_payload"])
            )

        capture_server_origin_mutation_batch(
            service=service,
            user_id=dataset.owner_user_id,
            steps=[step],
            source=_SOURCE,
            idempotency_key=f"{bootstrap_id}:{task_id}:{canonical_hash}",
            trusted_notes_task_bootstrap_id=bootstrap_id,
            bootstrap_step_verifier=source_matches,
        )


@dataclass(frozen=True, slots=True)
class _SourceSummary:
    """Immutable source identity with prefix lookup."""

    rows: tuple[tuple[str, str, int, str, str], ...]
    cursor: str | None
    count: int
    fingerprint: str


def _verify_stored_progress(
    *,
    service: SyncV2Service,
    dataset_id: str,
    bootstrap_id: str,
    state: Mapping[str, object],
    source: _SourceSummary,
) -> str:
    """Reject stored progress that is not an exact prefix of the source."""

    count = int(state["source_count"])
    cursor = _optional_string(state.get("source_cursor"))
    fingerprint = str(state.get("source_fingerprint") or _EMPTY_FINGERPRINT)
    if count == 0:
        if cursor is not None or fingerprint != _EMPTY_FINGERPRINT:
            raise _SourceChanged
        return _EMPTY_FINGERPRINT
    if count > len(source.rows) or source.rows[count - 1][0] != cursor:
        raise _SourceChanged
    expected = _EMPTY_FINGERPRINT
    for task_id, canonical_hash, *_ in source.rows[:count]:
        expected = _task_bootstrap_fingerprint(expected, task_id, canonical_hash)
    if expected == fingerprint:
        return expected
    if not (
        _stored_bootstrap_prefix_is_authentic(
            service=service,
            dataset_id=dataset_id,
            bootstrap_id=bootstrap_id,
            count=count,
            cursor=cursor,
            fingerprint=fingerprint,
        )
        and _sync_heads_match_source_rows(
            service,
            dataset_id,
            source.rows[:count],
        )
    ):
        raise _SourceChanged
    return expected


def _stored_bootstrap_prefix_is_authentic(
    *,
    service: SyncV2Service,
    dataset_id: str,
    bootstrap_id: str,
    count: int,
    cursor: str | None,
    fingerprint: str,
) -> bool:
    """Authenticate stored progress against immutable source-verified envelopes."""

    found: dict[str, SyncEnvelope] = {}
    after_cursor = 0
    while True:
        page = service.store.list_envelopes_after(
            dataset_id,
            after_cursor,
            limit=500,
            domains=["notes.task"],
            status="accepted",
        )
        for envelope in page:
            if envelope.server_cursor is None:
                return False
            after_cursor = envelope.server_cursor
            if envelope.routing_metadata.get("bootstrap_id") != bootstrap_id:
                continue
            if envelope.object_id in found:
                return False
            found[envelope.object_id] = envelope
        if len(page) < 500:
            break
    prefix = [found[task_id] for task_id in sorted(found)[:count]]
    if (
        len(prefix) != count
        or not prefix
        or prefix[-1].object_id != cursor
    ):
        return False
    expected = _EMPTY_FINGERPRINT
    for envelope in prefix:
        canonical_hash = envelope.payload_hash
        if (
            envelope.apply_status != "applied"
            or envelope.object_revision is None
            or canonical_hash is None
            or envelope.client_envelope_id
            != _task_bootstrap_envelope_id(
                bootstrap_id,
                envelope.object_id,
                canonical_hash,
            )
        ):
            return False
        expected = _task_bootstrap_fingerprint(
            expected,
            envelope.object_id,
            canonical_hash,
        )
    return expected == fingerprint


def _sync_bootstrap_matches_source(
    service: SyncV2Service,
    dataset_id: str,
    *,
    bootstrap_id: str,
    source: _SourceSummary,
) -> bool:
    """Return whether accepted applied envelopes exactly cover the source."""

    del bootstrap_id
    return _sync_heads_match_source_rows(service, dataset_id, source.rows)


def _sync_heads_match_source_rows(
    service: SyncV2Service,
    dataset_id: str,
    rows: tuple[tuple[str, str, int, str, str], ...],
) -> bool:
    """Verify that current applied task heads exactly cover canonical source rows."""

    expected = {row[0]: row for row in rows}
    found: set[str] = set()
    offset = 0
    while True:
        page = service.store.list_current_heads(
            dataset_id,
            "notes.task",
            limit=500,
            offset=offset,
        )
        for envelope in page:
            source_row = expected.get(envelope.object_id)
            if source_row is None or not _task_head_matches_source(
                envelope,
                source_row,
            ):
                return False
            found.add(envelope.object_id)
        if len(page) < 500:
            break
        offset += len(page)
    return found == set(expected)


def _task_head_matches_row(
    envelope: SyncEnvelope,
    row: Mapping[str, object],
) -> bool:
    """Return whether one current head is the exact canonical product row."""

    payload = row.get("sync_payload")
    return bool(
        isinstance(payload, Mapping)
        and dict(envelope.payload) == dict(payload)
        and _task_head_matches_source(
            envelope,
            (
                str(row["id"]),
                str(row["canonical_hash"]),
                int(row["canonical_revision"]),
                "tombstone" if bool(row.get("deleted")) else "upsert",
                str(row["note_id"]),
            ),
        )
    )


def _task_head_matches_source(
    envelope: SyncEnvelope,
    source: tuple[str, str, int, str, str],
) -> bool:
    """Return whether one applied current head matches a source identity tuple."""

    task_id, canonical_hash, canonical_revision, operation, note_id = source
    return bool(
        envelope.status == "accepted"
        and envelope.apply_status == "applied"
        and envelope.object_id == task_id
        and envelope.payload_hash == canonical_hash
        and envelope.object_revision == canonical_revision
        and envelope.operation == operation
        and envelope.parent_id == note_id
    )


def _block(
    service: SyncV2Service,
    dataset: SyncDataset,
    *,
    reason: str,
) -> SyncDataset:
    """Persist a bounded blocked state while retaining verified progress."""

    state = _readiness(dataset)
    return service.store.transition_notes_task_readiness(
        dataset.dataset_id,
        owner_user_id=dataset.owner_user_id,
        expected_state="bootstrapping",
        state="blocked",
        source_dataset_id=dataset.dataset_id,
        source_cursor=_optional_string(state.get("source_cursor")),
        source_count=int(state["source_count"]),
        source_fingerprint=(
            str(state["source_fingerprint"])
            if state.get("source_fingerprint") is not None
            else None
        ),
        reason_code=reason,
    )


def _readiness(dataset: SyncDataset) -> Mapping[str, object]:
    """Return the strict task readiness record for a dataset."""

    state = dataset.metadata.get("notes_task_v1")
    if not isinstance(state, Mapping):
        raise SyncStoreError("notes_task_readiness_state_invalid")
    return state


def _optional_string(value: object) -> str | None:
    """Return a string value or ``None`` without coercion."""

    return value if isinstance(value, str) else None


def _bootstrap_id(owner_user_id: str, dataset_id: str) -> str:
    """Derive the stable private bootstrap identity for one owner dataset."""

    digest = hashlib.sha256(
        f"notes.task.bootstrap.v1:{owner_user_id}:{dataset_id}".encode()
    ).hexdigest()
    return f"notes-task-bootstrap-{digest[:32]}"


def _task_bootstrap_envelope_id(
    bootstrap_id: str,
    task_id: str,
    canonical_hash: str,
) -> str:
    """Derive the stable envelope identity for one canonical task version."""

    digest = hashlib.sha256(
        json.dumps(
            [bootstrap_id, task_id, canonical_hash],
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return f"notes-task-bootstrap-{digest[:32]}"


def _task_bootstrap_routing(bootstrap_id: str) -> dict[str, object]:
    """Build the minimal trusted routing metadata for task bootstrap."""

    return {"bootstrap_capture": True, "bootstrap_id": bootstrap_id}


def _task_bootstrap_fingerprint(
    previous_fingerprint: str | None,
    task_id: str,
    canonical_hash: str,
) -> str:
    """Extend the ordered source fingerprint with one canonical task row."""

    seed = previous_fingerprint or _EMPTY_FINGERPRINT
    return hashlib.sha256(
        json.dumps(
            [seed, task_id, canonical_hash],
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


__all__ = ["NotesTaskBootstrapInterrupted", "NotesTaskBootstrapper"]
