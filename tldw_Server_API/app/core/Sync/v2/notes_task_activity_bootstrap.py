"""Bounded, resumable bootstrap for legacy Notes task activity."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from tldw_Server_API.app.core.DB_Management.chacha.task_store import TaskStore
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.exceptions import (
    NotesTaskActivitySourceChanged,
    NotesTaskActivitySourceInvalid,
    NotesTaskBootstrapInterrupted,
    NotesTaskContractError,
)

from .errors import SyncStoreError
from .models import SyncDataset, SyncEnvelope
from .notes_task_contract import (
    NotesTaskActivityV1,
    convert_legacy_task_event,
    notes_task_activity_object_hash,
    parse_notes_task_activity_v1,
)
from .server_origin_batch import (
    ServerOriginMutationStep,
    capture_server_origin_mutation_batch,
    is_trusted_notes_task_coordinator_envelope,
)
from .service import SyncV2Service

_SOURCE = "notes-task-activity-bootstrap"
_EMPTY_FINGERPRINT = hashlib.sha256(b"notes.task_activity.bootstrap.v1").hexdigest()


class NotesTaskActivityBootstrapper:
    """Capture at most one owner-scoped legacy activity page per invocation."""

    PAGE_LIMIT = 1_000

    def __init__(
        self,
        note_db: CharactersRAGDB,
        *,
        page_limit: int = PAGE_LIMIT,
        after_page: Callable[[int], None] | None = None,
    ) -> None:
        if isinstance(page_limit, bool) or not 1 <= page_limit <= self.PAGE_LIMIT:
            raise ValueError("Notes task activity bootstrap page limit must be 1..1000")
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
        """Resume one bounded source page and return durable activity readiness."""

        owner = dataset.owner_user_id
        current = service.store.get_dataset(dataset.dataset_id, owner_user_id=owner)
        if current is None:
            raise SyncStoreError("Sync dataset was not found or is not accessible")
        if self._tasks.resolve_task_compatibility_dataset_id(
            owner_user_id=owner
        ) != current.dataset_id:
            raise SyncStoreError("notes_task_activity_readiness_source_scope_invalid")
        task_state = current.metadata.get("notes_task_v1")
        if not isinstance(task_state, Mapping) or task_state.get("state") != "ready":
            raise SyncStoreError("notes_task_sync_not_ready")
        if current.metadata.get("task_activity_capture_enabled") is not True:
            raise SyncStoreError("notes_task_activity_sync_not_ready")

        current = self._ensure_bootstrapping(service, current)
        state = _readiness(current)
        if state["state"] == "ready":
            return current
        if state["state"] != "bootstrapping":
            raise SyncStoreError("notes_task_activity_sync_not_ready")

        bootstrap_id = _bootstrap_id(owner, current.dataset_id)
        try:
            self._verify_resume_boundary(
                service=service,
                owner=owner,
                dataset_id=current.dataset_id,
                bootstrap_id=bootstrap_id,
                state=state,
            )
            after_created_at, after_activity_id = _split_cursor(
                _optional_string(state.get("source_cursor"))
            )
            page = self._tasks.page_legacy_events_for_sync_bootstrap(
                owner_user_id=owner,
                dataset_id=current.dataset_id,
                after_created_at=after_created_at,
                after_activity_id=after_activity_id,
                limit=self._page_limit,
            )
            running_count = int(state["source_count"])
            running_fingerprint = str(
                state.get("source_fingerprint") or _EMPTY_FINGERPRINT
            )
            for row in page:
                source_row = self._source_row(
                    service=service,
                    owner=owner,
                    dataset_id=current.dataset_id,
                    bootstrap_id=bootstrap_id,
                    row=row,
                )
                self._capture_row(
                    service=service,
                    dataset=current,
                    bootstrap_id=bootstrap_id,
                    source=source_row,
                )
                running_count += 1
                running_fingerprint = _activity_bootstrap_fingerprint(
                    running_fingerprint,
                    source_row.cursor,
                    source_row.canonical_hash,
                )

            if page and self._after_page is not None:
                self._after_page(1)
            if page:
                current = service.store.transition_notes_task_activity_readiness(
                    current.dataset_id,
                    owner_user_id=owner,
                    expected_state="bootstrapping",
                    state="bootstrapping",
                    source_dataset_id=current.dataset_id,
                    source_cursor=_row_cursor(page[-1]),
                    source_count=running_count,
                    source_fingerprint=running_fingerprint,
                )
            if len(page) == self._page_limit and not all(
                row.get("sync_server_cursor") is not None for row in page
            ):
                return current
            source = self._source_summary(
                service=service,
                owner=owner,
                dataset_id=current.dataset_id,
                bootstrap_id=bootstrap_id,
            )
            if running_count < source.count:
                return current
            if (
                running_count != source.count
                or running_fingerprint != source.fingerprint
                or _optional_string(
                    _readiness(current).get("source_cursor")
                ) != source.cursor
                or not self._sync_heads_match_source(
                    service=service,
                    owner=owner,
                    dataset_id=current.dataset_id,
                    bootstrap_id=bootstrap_id,
                    source=source,
                )
            ):
                raise NotesTaskActivitySourceChanged
            current = service.store.transition_notes_task_activity_readiness(
                current.dataset_id,
                owner_user_id=owner,
                expected_state="bootstrapping",
                state="verifying",
                source_dataset_id=current.dataset_id,
                source_cursor=source.cursor,
                source_count=source.count,
                source_fingerprint=source.fingerprint,
            )
            return service.store.transition_notes_task_activity_readiness(
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
        except NotesTaskActivitySourceChanged:
            return _block(service, current, reason="notes_task_activity_source_changed")
        except Exception as exc:  # noqa: BLE001 - details become bounded readiness.
            if isinstance(exc, SyncStoreError) and str(exc) not in {
                "notes_task_activity_source_invalid",
                "notes_task_activity_source_changed",
            }:
                raise
            return _block(service, current, reason="notes_task_activity_source_invalid")

    @property
    def note_db(self) -> CharactersRAGDB:
        """Return the product database whose activity log is being bootstrapped."""

        return self._db

    def _ensure_bootstrapping(
        self,
        service: SyncV2Service,
        dataset: SyncDataset,
    ) -> SyncDataset:
        """Enter or resume activity bootstrapping without enrolling the domain."""

        state = _readiness(dataset)
        if state.get("state") in {"bootstrapping", "ready"}:
            return dataset
        if state.get("state") == "blocked" and state.get("resume_phase") == "bootstrapping":
            return service.store.transition_notes_task_activity_readiness(
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
        if state.get("state") != "enrolling":
            raise SyncStoreError("notes_task_activity_sync_not_ready")
        return service.store.transition_notes_task_activity_readiness(
            dataset.dataset_id,
            owner_user_id=dataset.owner_user_id,
            expected_state="enrolling",
            state="bootstrapping",
            source_dataset_id=dataset.dataset_id,
            source_cursor=None,
            source_count=0,
            source_fingerprint=_EMPTY_FINGERPRINT,
        )

    def _verify_resume_boundary(
        self,
        *,
        service: SyncV2Service,
        owner: str,
        dataset_id: str,
        bootstrap_id: str,
        state: Mapping[str, object],
    ) -> None:
        """Verify the durable progress boundary without rescanning its prefix."""

        count = int(state["source_count"])
        cursor = _optional_string(state.get("source_cursor"))
        fingerprint = str(state.get("source_fingerprint") or _EMPTY_FINGERPRINT)
        if count == 0:
            if cursor is not None or fingerprint != _EMPTY_FINGERPRINT:
                raise NotesTaskActivitySourceChanged
            return
        if cursor is None or fingerprint == _EMPTY_FINGERPRINT:
            raise NotesTaskActivitySourceChanged
        _created_at, activity_id = _split_cursor(cursor)
        if activity_id is None:
            raise NotesTaskActivitySourceChanged
        row = self._tasks.get_sync_task_activity(
            owner_user_id=owner,
            dataset_id=dataset_id,
            activity_id=activity_id,
        )
        if row is None or _row_cursor(row) != cursor:
            raise NotesTaskActivitySourceChanged
        try:
            boundary = self._source_row(
                service=service,
                owner=owner,
                dataset_id=dataset_id,
                bootstrap_id=bootstrap_id,
                row=row,
            )
        except NotesTaskActivitySourceInvalid as exc:
            raise NotesTaskActivitySourceChanged from exc
        if boundary.cursor != cursor:
            raise NotesTaskActivitySourceChanged

    def _source_summary(
        self,
        *,
        service: SyncV2Service,
        owner: str,
        dataset_id: str,
        bootstrap_id: str,
    ) -> _SourceSummary:
        """Stream the complete source identity with page-bounded memory."""

        after_created_at: str | None = None
        after_activity_id: str | None = None
        cursor: str | None = None
        count = 0
        fingerprint = _EMPTY_FINGERPRINT
        while True:
            try:
                page = self._tasks.page_legacy_events_for_sync_bootstrap(
                    owner_user_id=owner,
                    dataset_id=dataset_id,
                    after_created_at=after_created_at,
                    after_activity_id=after_activity_id,
                    limit=self.PAGE_LIMIT,
                )
                for row in page:
                    item = self._source_row(
                        service=service,
                        owner=owner,
                        dataset_id=dataset_id,
                        bootstrap_id=bootstrap_id,
                        row=row,
                    )
                    count += 1
                    cursor = item.cursor
                    fingerprint = _activity_bootstrap_fingerprint(
                        fingerprint,
                        item.cursor,
                        item.canonical_hash,
                    )
                    after_created_at, after_activity_id = _split_cursor(item.cursor)
            except NotesTaskActivitySourceChanged:
                raise
            except Exception as exc:  # noqa: BLE001 - source details remain private.
                raise NotesTaskActivitySourceInvalid from exc
            if len(page) < self.PAGE_LIMIT:
                break
        return _SourceSummary(cursor, count, fingerprint)

    def _source_row(
        self,
        *,
        service: SyncV2Service,
        owner: str,
        dataset_id: str,
        bootstrap_id: str,
        row: Mapping[str, object],
    ) -> _SourceRow:
        """Return one verified pending or already-adopted source row."""

        cursor = _row_cursor(row)
        if row.get("sync_server_cursor") is None:
            payload = legacy_task_event_to_activity(row, owner_user_id=owner)
            canonical_hash = notes_task_activity_object_hash(
                payload,
                revision=1,
                deleted=False,
            )
        else:
            envelope = service.store.get_envelope_by_server_cursor(
                int(row["sync_server_cursor"])
            )
            selected_dataset = service.store.get_dataset(
                dataset_id,
                owner_user_id=owner,
            )
            bootstrap_capture = bool(
                envelope is not None
                and envelope.routing_metadata.get("bootstrap_id") == bootstrap_id
            )
            trusted_capture = bool(
                envelope is not None
                and (
                    bootstrap_capture
                    or (
                        selected_dataset is not None
                        and is_trusted_notes_task_coordinator_envelope(
                            service=service,
                            dataset=selected_dataset,
                            envelope=envelope,
                        )
                    )
                )
            )
            if (
                envelope is None
                or envelope.dataset_id != dataset_id
                or envelope.domain != "notes.task_activity"
                or envelope.object_id != row.get("id")
                or envelope.parent_id != row.get("note_id")
                or envelope.operation != "upsert"
                or envelope.object_revision != 1
                or not trusted_capture
                or envelope.apply_status != "applied"
            ):
                raise NotesTaskActivitySourceChanged
            payload = _parse_bootstrap_payload(envelope, owner)
            canonical_hash = notes_task_activity_object_hash(
                payload,
                revision=1,
                deleted=False,
            )
            if (
                envelope.payload_hash != canonical_hash
                or row.get("sync_object_hash") != canonical_hash
                or (
                    bootstrap_capture
                    and envelope.created_at_client != row.get("created_at")
                )
                or not self._tasks.verify_sync_task_activity_postcondition(
                    owner_user_id=owner,
                    dataset_id=dataset_id,
                    payload=payload,
                    sync_revision=1,
                    sync_object_hash=canonical_hash,
                    sync_server_cursor=int(row["sync_server_cursor"]),
                )
            ):
                raise NotesTaskActivitySourceChanged
        return _SourceRow(
            activity_id=payload.activity_id,
            note_id=payload.note_id,
            cursor=cursor,
            canonical_hash=canonical_hash,
            payload=payload,
        )

    def _sync_heads_match_source(
        self,
        *,
        service: SyncV2Service,
        owner: str,
        dataset_id: str,
        bootstrap_id: str,
        source: _SourceSummary,
    ) -> bool:
        """Rescan and match every source row to its exact applied current head."""

        after_created_at: str | None = None
        after_activity_id: str | None = None
        cursor: str | None = None
        count = 0
        fingerprint = _EMPTY_FINGERPRINT
        while True:
            page = self._tasks.page_legacy_events_for_sync_bootstrap(
                owner_user_id=owner,
                dataset_id=dataset_id,
                after_created_at=after_created_at,
                after_activity_id=after_activity_id,
                limit=self.PAGE_LIMIT,
            )
            for row in page:
                item = self._source_row(
                    service=service,
                    owner=owner,
                    dataset_id=dataset_id,
                    bootstrap_id=bootstrap_id,
                    row=row,
                )
                head = service.store.get_current_head(
                    dataset_id,
                    "notes.task_activity",
                    item.activity_id,
                )
                if head is None or not _activity_head_matches_source(head, item):
                    return False
                count += 1
                cursor = item.cursor
                fingerprint = _activity_bootstrap_fingerprint(
                    fingerprint,
                    item.cursor,
                    item.canonical_hash,
                )
                after_created_at, after_activity_id = _split_cursor(item.cursor)
            if len(page) < self.PAGE_LIMIT:
                break
        bootstrap_head_count = 0
        coordinator_head_count = 0
        selected_dataset = service.store.get_dataset(
            dataset_id,
            owner_user_id=owner,
        )
        if selected_dataset is None:
            return False
        offset = 0
        while True:
            heads = service.store.list_current_heads(
                dataset_id,
                "notes.task_activity",
                limit=500,
                offset=offset,
            )
            for head in heads:
                if head.routing_metadata.get("bootstrap_id") == bootstrap_id:
                    bootstrap_head_count += 1
                    continue
                if not is_trusted_notes_task_coordinator_envelope(
                    service=service,
                    dataset=selected_dataset,
                    envelope=head,
                ) or not _coordinator_activity_head_matches_product(
                    tasks=self._tasks,
                    owner=owner,
                    dataset_id=dataset_id,
                    envelope=head,
                ):
                    return False
                coordinator_head_count += 1
            if len(heads) < 500:
                break
            offset += len(heads)
        return bool(
            bootstrap_head_count == count
            and bootstrap_head_count + coordinator_head_count
            == _count_current_activity_heads(service, dataset_id)
            and count == source.count
            and cursor == source.cursor
            and fingerprint == source.fingerprint
        )

    def _capture_row(
        self,
        *,
        service: SyncV2Service,
        dataset: SyncDataset,
        bootstrap_id: str,
        source: _SourceRow,
    ) -> None:
        """Capture and adopt one source-verified legacy event."""

        current_head = service.store.get_current_head(
            dataset.dataset_id,
            "notes.task_activity",
            source.activity_id,
        )
        if current_head is not None and _activity_head_matches_source(
            current_head,
            source,
        ):
            return

        step = ServerOriginMutationStep(
            domain="notes.task_activity",
            operation="upsert",
            object_id=source.activity_id,
            parent_id=source.note_id,
            payload=source.payload.model_dump(mode="json"),
            routing_metadata=_activity_bootstrap_routing(bootstrap_id),
            client_envelope_id=_activity_bootstrap_envelope_id(
                bootstrap_id,
                source.activity_id,
                source.canonical_hash,
            ),
            object_revision=1,
            created_at_client=source.payload.client_occurred_at,
        )

        def source_matches(envelope: SyncEnvelope) -> bool:
            """Verify source identity immediately before product adoption."""

            current = self._tasks.get_sync_task_activity(
                owner_user_id=dataset.owner_user_id,
                dataset_id=dataset.dataset_id,
                activity_id=source.activity_id,
            )
            if current is None:
                return False
            if current.get("sync_server_cursor") is not None:
                return self._tasks.verify_sync_task_activity_postcondition(
                    owner_user_id=dataset.owner_user_id,
                    dataset_id=dataset.dataset_id,
                    payload=source.payload,
                    sync_revision=1,
                    sync_object_hash=source.canonical_hash,
                    sync_server_cursor=int(current["sync_server_cursor"]),
                )
            try:
                converted = legacy_task_event_to_activity(
                    current,
                    owner_user_id=dataset.owner_user_id,
                    resolved_task_note_id=(
                        source.note_id if current.get("task_id") is not None else None
                    ),
                )
            except NotesTaskContractError:
                return False
            return bool(
                envelope.client_envelope_id == step.client_envelope_id
                and envelope.object_id == source.activity_id
                and envelope.parent_id == source.note_id
                and envelope.payload_hash == source.canonical_hash
                and dict(envelope.payload) == converted.model_dump(mode="json")
            )

        capture_server_origin_mutation_batch(
            service=service,
            user_id=dataset.owner_user_id,
            steps=[step],
            source=_SOURCE,
            idempotency_key=(
                f"{bootstrap_id}:{source.activity_id}:{source.canonical_hash}"
            ),
            trusted_notes_task_activity_bootstrap_id=bootstrap_id,
            bootstrap_step_verifier=source_matches,
        )


@dataclass(frozen=True, slots=True)
class _SourceRow:
    """One canonical legacy/adopted activity source row."""

    activity_id: str
    note_id: str
    cursor: str
    canonical_hash: str
    payload: NotesTaskActivityV1


@dataclass(frozen=True, slots=True)
class _SourceSummary:
    """Constant-memory identity summary of the complete ordered source."""

    cursor: str | None
    count: int
    fingerprint: str


def _activity_head_matches_source(
    envelope: SyncEnvelope,
    source: _SourceRow,
) -> bool:
    """Return whether an applied immutable head exactly matches one source event."""

    return bool(
        envelope.status == "accepted"
        and envelope.apply_status == "applied"
        and envelope.operation == "upsert"
        and envelope.object_id == source.activity_id
        and envelope.parent_id == source.note_id
        and envelope.object_revision == 1
        and envelope.payload_hash == source.canonical_hash
        and envelope.created_at_client == source.payload.client_occurred_at
        and dict(envelope.payload) == source.payload.model_dump(mode="json")
    )


def _coordinator_activity_head_matches_product(
    *,
    tasks: TaskStore,
    owner: str,
    dataset_id: str,
    envelope: SyncEnvelope,
) -> bool:
    """Verify one captured coordinator activity against its exact product row."""

    if envelope.server_cursor is None:
        return False
    try:
        payload = _parse_bootstrap_payload(envelope, owner)
        canonical_hash = notes_task_activity_object_hash(
            payload,
            revision=1,
            deleted=False,
        )
        return bool(
            _activity_head_matches_source(
                envelope,
                _SourceRow(
                    activity_id=payload.activity_id,
                    note_id=payload.note_id,
                    cursor=f"{payload.client_occurred_at}|{payload.activity_id}",
                    canonical_hash=canonical_hash,
                    payload=payload,
                ),
            )
            and tasks.verify_sync_task_activity_postcondition(
                owner_user_id=owner,
                dataset_id=dataset_id,
                payload=payload,
                sync_revision=1,
                sync_object_hash=canonical_hash,
                sync_server_cursor=envelope.server_cursor,
            )
        )
    except Exception:  # noqa: BLE001 - readiness verification is total and fail-closed.
        return False


def _count_current_activity_heads(service: SyncV2Service, dataset_id: str) -> int:
    """Count current activity heads with bounded pages for exact set accounting."""

    count = 0
    offset = 0
    while True:
        page = service.store.list_current_heads(
            dataset_id,
            "notes.task_activity",
            limit=500,
            offset=offset,
        )
        count += len(page)
        if len(page) < 500:
            return count
        offset += len(page)


def legacy_task_event_to_activity(
    row: Mapping[str, object],
    *,
    owner_user_id: str,
    resolved_task_note_id: str | None = None,
) -> NotesTaskActivityV1:
    """Convert one exact decoded task-event row through the strict contract."""

    resolved = (
        resolved_task_note_id
        if resolved_task_note_id is not None
        else (
            str(row["resolved_task_note_id"])
            if row.get("resolved_task_note_id") is not None
            else None
        )
    )
    source = {
        "id": row.get("id"),
        "task_id": row.get("task_id"),
        "note_id": row.get("note_id"),
        "event_type": row.get("event_type"),
        "actor_type": row.get("actor_type"),
        "actor_id": row.get("actor_id"),
        "tool_name": row.get("tool_name"),
        "policy_mode": row.get("policy_mode"),
        "approval_id": row.get("approval_id"),
        "old_value": row.get("old_value_json", row.get("old_value")),
        "new_value": row.get("new_value_json", row.get("new_value")),
        "created_at": row.get("created_at"),
        "client_id": row.get("client_id"),
    }
    return convert_legacy_task_event(
        source,
        owner_user_id=owner_user_id,
        resolved_task_note_id=resolved,
    )


def _parse_bootstrap_payload(envelope: SyncEnvelope, owner: str) -> NotesTaskActivityV1:
    """Re-validate one stored trusted-bootstrap activity payload."""

    return parse_notes_task_activity_v1(
        envelope.payload,
        owner_user_id=owner,
        bound_actor_type=str(envelope.payload.get("actor_type")),
        bound_actor_id=envelope.payload.get("actor_id"),
        authenticated_device_id=None,
        trusted_server_origin=True,
    )


def _block(
    service: SyncV2Service,
    dataset: SyncDataset,
    *,
    reason: str,
) -> SyncDataset:
    """Persist bounded blocked activity readiness while retaining progress."""

    state = _readiness(dataset)
    return service.store.transition_notes_task_activity_readiness(
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
    """Return the strict internal activity readiness mapping."""

    state = dataset.metadata.get("notes_task_activity_v1")
    if not isinstance(state, Mapping):
        raise SyncStoreError("notes_task_activity_readiness_state_invalid")
    return state


def _row_cursor(row: Mapping[str, object]) -> str:
    """Return the canonical created-at/activity-id keyset cursor."""

    created_at = row.get("created_at")
    activity_id = row.get("id")
    if not isinstance(created_at, str) or not isinstance(activity_id, str):
        raise NotesTaskActivitySourceInvalid
    return f"{created_at}|{activity_id}"


def _split_cursor(value: str | None) -> tuple[str | None, str | None]:
    """Split one canonical activity cursor into its keyset components."""

    if value is None:
        return None, None
    try:
        created_at, activity_id = value.rsplit("|", 1)
    except ValueError as exc:
        raise NotesTaskActivitySourceChanged from exc
    return created_at, activity_id


def _optional_string(value: object) -> str | None:
    """Return a string value or None without coercion."""

    return value if isinstance(value, str) else None


def _bootstrap_id(owner_user_id: str, dataset_id: str) -> str:
    """Return the deterministic trusted-bootstrap identifier for a dataset."""

    digest = hashlib.sha256(
        f"notes.task_activity.bootstrap.v1:{owner_user_id}:{dataset_id}".encode()
    ).hexdigest()
    return f"notes-task-activity-bootstrap-{digest[:32]}"


def _activity_bootstrap_envelope_id(
    bootstrap_id: str,
    activity_id: str,
    canonical_hash: str,
) -> str:
    """Return the deterministic envelope identity for one source activity."""

    digest = hashlib.sha256(
        json.dumps(
            [bootstrap_id, activity_id, canonical_hash],
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return f"notes-task-activity-bootstrap-{digest[:32]}"


def _activity_bootstrap_routing(bootstrap_id: str) -> dict[str, object]:
    """Return trusted routing metadata for one activity bootstrap."""

    return {"bootstrap_capture": True, "bootstrap_id": bootstrap_id}


def _activity_bootstrap_fingerprint(
    previous_fingerprint: str | None,
    source_cursor: str,
    canonical_hash: str,
) -> str:
    """Extend the ordered activity source fingerprint by one row."""

    seed = previous_fingerprint or _EMPTY_FINGERPRINT
    return hashlib.sha256(
        json.dumps(
            [seed, source_cursor, canonical_hash],
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


__all__ = [
    "NotesTaskActivityBootstrapper",
    "NotesTaskBootstrapInterrupted",
    "legacy_task_event_to_activity",
]
