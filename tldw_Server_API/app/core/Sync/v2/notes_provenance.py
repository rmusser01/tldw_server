"""Canonical Knowledge history capture, pair projection, and bounded enrollment.

API callers use ``capture_note_with_provenance`` or insert
``provenance_step`` immediately after their core note step. Expected versions
are independent; zero means absent. Retrying the same request key reuses the
complete durable manifest. Omitted provenance uses the ordinary core path.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

from .errors import SyncStoreError
from .materializers.base import MaterializationResult
from .materializers.notes import NotesMaterializer
from .models import SyncDataset, SyncEnvelope, SyncEnvelopeCreate, SyncObjectState, validate_notes_note_upsert_payload
from .mutation_group_validation import (
    materialization_group_view,
    mutation_group_plan_hash,
    validate_stored_mutation_group,
)
from .notes_provenance_contract import (
    notes_provenance_object_hash,
    read_notes_provenance,
    validate_notes_provenance_payload,
)

if TYPE_CHECKING:
    from .server_origin_batch import ServerOriginBatchResult, ServerOriginMutationStep
    from .service import SyncV2Service
    from .store import SyncV2Store

READINESS_KEY = "notes_provenance_v1"


class NotesProvenanceCheckpointError(SyncStoreError):
    """Abort the entire held Sync transaction after a product pair committed."""


@dataclass(slots=True)
class NotesProvenanceMaterializer:
    """Apply independent evidence using the shared core conflict/checkpoint rules."""

    note_db: CharactersRAGDB
    domain: str = "notes.provenance"

    def apply(self, envelope: SyncEnvelope, *, store: SyncV2Store) -> MaterializationResult:
        return project_notes_provenance(envelope, store=store, note_db=self.note_db)


def provenance_pair(envelope: SyncEnvelope, store: SyncV2Store) -> list[SyncEnvelope]:
    """Resolve the same adjacent pair from either member, including filtered repair."""
    if not envelope.mutation_group_id:
        return [envelope]
    group = store.list_mutation_group(envelope.dataset_id, envelope.mutation_group_id)
    validate_stored_mutation_group(group, dataset_id=envelope.dataset_id, mutation_group_id=envelope.mutation_group_id)
    group = materialization_group_view(group)
    index = next(i for i, item in enumerate(group) if item.server_cursor == envelope.server_cursor)
    left = index if envelope.domain == "notes.note" else index - 1
    if left >= 0 and left + 1 < len(group):
        core, child = group[left : left + 2]
        if core.domain == "notes.note" and child.domain == "notes.provenance" and core.object_id == child.object_id:
            return [core, child]
    return [group[index]]


def is_notes_provenance_bootstrap(envelope: SyncEnvelope | SyncEnvelopeCreate, owner_user_id: str) -> bool:
    """Recognize immutable server-authored capture; client routing is insufficient."""
    routing = envelope.routing_metadata
    return (
        envelope.device_id == "server-origin"
        and envelope.mutation_group_id is not None
        and routing.get("source") == "notes-provenance-bootstrap"
        and routing.get("origin") == "server"
        and routing.get("server_device_id") == "server-origin"
        and routing.get("server_owner_user_id") == owner_user_id
    )


def notes_provenance_bootstrap_matches_source(
    note_db: CharactersRAGDB, envelope: SyncEnvelope | SyncEnvelopeCreate
) -> bool:
    """Verify retained product versions instead of replaying old bootstrap bytes."""
    owner = str(note_db.owner_user_id)
    if not is_notes_provenance_bootstrap(envelope, owner):
        return False
    note = note_db.get_note_by_id(envelope.object_id, include_deleted=True)
    if (
        note is None
        or str(note.get("client_id")) != owner
        or note["version"] != envelope.routing_metadata.get("notes_provenance_parent_version")
    ):
        return False
    if envelope.domain == "notes.note":
        payload = validate_notes_note_upsert_payload(
            {key: note.get(key) for key in ("title", "content", "conversation_id", "message_id")}
        )
        return (
            envelope.object_revision == note["version"]
            and envelope.payload == payload
            and (envelope.operation == "tombstone") == bool(note["deleted"])
        )
    record = note_db.note_provenance_store.get(envelope.object_id, include_deleted=True)
    return (
        record is not None
        and envelope.object_revision == record["version"]
        and envelope.payload_hash == record["object_hash"]
        and envelope.payload == record["payload"]
        and (envelope.operation == "tombstone") == record["deleted"]
    )


def project_notes_provenance(
    envelope: SyncEnvelope, *, store: SyncV2Store, note_db: CharactersRAGDB
) -> MaterializationResult:
    """Commit both product records before atomically checkpointing both Sync states."""
    core = NotesMaterializer(note_db)
    pair = provenance_pair(envelope, store)
    dataset = store.get_dataset(envelope.dataset_id)
    if (
        dataset is None
        or dataset.owner_user_id != str(note_db.owner_user_id)
        or dataset.encryption_policy != "server_trusted_v1"
    ):
        raise SyncStoreError("notes_provenance_owner_or_encryption_invalid")
    pending = []
    for member in pair:
        state = store.get_object_state(member.dataset_id, member.domain, member.object_id)
        if core._is_already_materialized(member, state):
            continue
        conflict = core._detect_conflict(member, state)
        if conflict is not None:
            store.mark_envelope_apply_status(
                member.server_cursor,
                apply_status="conflict",
                apply_error_code="whole_object_conflict",
                apply_error_message=conflict.message,
            )
            return conflict
        pending.append((member, core._next_object_revision(member, state)))
    try:
        with note_db.transaction() as conn:
            for member, revision in pending:
                if is_notes_provenance_bootstrap(member, str(note_db.owner_user_id)):
                    if not notes_provenance_bootstrap_matches_source(note_db, member):
                        raise SyncStoreError("notes_provenance_bootstrap_source_changed")
                    continue
                if member.domain == "notes.note":
                    core.project_product(member, object_revision=revision, conn=conn)
                else:
                    # Applied parent state is authoritative at this projection position;
                    # a later accepted delete may already be the durable current head.
                    parent = next((item for item in pair if item.domain == "notes.note"), None)
                    parent_state = store.get_object_state(member.dataset_id, "notes.note", member.object_id)
                    if parent is None and (
                        parent_state is None or (parent_state.deleted and member.operation != "tombstone")
                    ):
                        raise SyncStoreError("notes_provenance_parent_missing")
                    note_db.note_provenance_store.apply_sync(
                        member.object_id,
                        member.payload,
                        revision,
                        member.payload_hash,
                        deleted=member.operation == "tombstone",
                        conn=conn,
                        restore=member.routing_metadata.get("restore_intent") is True,
                    )
    except Exception:  # noqa: BLE001 - product rollback is replayable, never expose source bytes.
        for member, _ in pending:
            store.mark_envelope_apply_status(
                member.server_cursor,
                apply_status="failed",
                apply_error_code="notes_provenance_projection_failed",
                apply_error_message="Notes provenance product projection failed",
            )
        return MaterializationResult(status="failed", error_code="notes_provenance_projection_failed")
    try:
        for member, revision in pending:
            store.upsert_object_state(
                SyncObjectState(
                    dataset_id=member.dataset_id,
                    domain=member.domain,
                    object_id=member.object_id,
                    object_revision=revision,
                    object_hash=member.payload_hash,
                    latest_server_cursor=member.server_cursor,
                    deleted=member.operation == "tombstone",
                )
            )
        for member in pair:
            store.mark_envelope_apply_status(member.server_cursor, apply_status="applied")
    except Exception as exc:  # noqa: BLE001 - rollback every checkpoint, retain committed product pair.
        raise NotesProvenanceCheckpointError("notes_provenance_checkpoint_failed") from exc
    return MaterializationResult(status="applied")


def provenance_step(
    *,
    service: SyncV2Service,
    dataset: SyncDataset,
    note_id: str,
    payload: Mapping[str, object],
    expected_version: int,
    restore: bool = False,
) -> ServerOriginMutationStep:
    """Build an exact independent replacement/retained restore for a compound plan."""
    from .server_origin_batch import ServerOriginMutationStep

    if type(expected_version) is not int or expected_version < 0:
        raise SyncStoreError("notes_provenance_expected_version_invalid")
    value = validate_notes_provenance_payload(payload)
    head = service.store.get_current_head(dataset.dataset_id, "notes.provenance", note_id)
    if (head.object_revision if head else 0) != expected_version:
        raise SyncStoreError("notes_provenance_version_conflict")
    return ServerOriginMutationStep(
        domain="notes.provenance",
        operation="upsert",
        object_id=note_id,
        parent_id=note_id,
        payload=value,
        object_revision=expected_version + 1,
        base_object_revision=expected_version or None,
        base_object_hash=head.payload_hash if head else None,
        routing_metadata={"restore_intent": True} if restore else {},
    )


def capture_note_with_provenance(
    *,
    service: SyncV2Service,
    note_db: CharactersRAGDB,
    user_id: str,
    note_id: str,
    note_payload: Mapping[str, object],
    expected_note_version: int,
    provenance: Mapping[str, object],
    expected_provenance_version: int,
    idempotency_key: str,
    source: str,
    restore: bool = False,
) -> ServerOriginBatchResult:
    """Save a complete pair with independent expected versions and durable retry identity."""
    from .server_origin_batch import (
        ServerOriginMutationStep,
        capture_server_origin_mutation_batch,
        load_server_origin_mutation_batch_manifest,
    )

    dataset = ensure_notes_provenance_ready(service=service, note_db=note_db, user_id=user_id)
    payload = validate_notes_note_upsert_payload(note_payload)
    value = validate_notes_provenance_payload(provenance)
    fingerprint = hashlib.sha256(
        json.dumps(
            [note_id, payload, value, expected_note_version, expected_provenance_version, restore],
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    existing = load_server_origin_mutation_batch_manifest(
        service=service, dataset_id=dataset.dataset_id, source=source, idempotency_key=idempotency_key
    )
    if existing is not None:
        if any(step.routing_metadata.get("notes_provenance_request") != fingerprint for step in existing):
            raise SyncStoreError("notes_provenance_idempotency_conflict")
        return capture_server_origin_mutation_batch(
            service=service, user_id=user_id, steps=existing, source=source, idempotency_key=idempotency_key
        )
    if type(expected_note_version) is not int or expected_note_version < 0:
        raise SyncStoreError("notes_note_expected_version_invalid")
    parent = service.store.get_current_head(dataset.dataset_id, "notes.note", note_id)
    if (parent.object_revision if parent else 0) != expected_note_version:
        raise SyncStoreError("notes_note_version_conflict")
    steps = [
        ServerOriginMutationStep(
            domain="notes.note",
            operation="upsert",
            object_id=note_id,
            payload=payload,
            object_revision=expected_note_version + 1,
            base_object_revision=expected_note_version or None,
            base_object_hash=parent.payload_hash if parent else None,
        ),
        provenance_step(
            service=service,
            dataset=dataset,
            note_id=note_id,
            payload=value,
            expected_version=expected_provenance_version,
            restore=restore,
        ),
    ]
    steps = [
        replace(step, routing_metadata={**step.routing_metadata, "notes_provenance_request": fingerprint})
        for step in steps
    ]
    return capture_server_origin_mutation_batch(
        service=service, user_id=user_id, steps=steps, source=source, idempotency_key=idempotency_key
    )


def expand_provenance_tombstone_steps(
    steps: Sequence[ServerOriginMutationStep],
    *,
    store: SyncV2Store,
    dataset_id: str,
    existing: Sequence[SyncEnvelope] = (),
) -> tuple[ServerOriginMutationStep, ...]:
    """Expand all server note deletes; canonical stored children govern retries."""
    from .server_origin_batch import ServerOriginMutationStep

    result = []
    for index, step in enumerate(steps):
        result.append(step)
        if step.domain != "notes.note" or step.operation != "tombstone":
            continue
        following = steps[index + 1] if index + 1 < len(steps) else None
        if following and following.domain == "notes.provenance" and following.object_id == step.object_id:
            continue
        child = next(
            (item for item in existing if item.domain == "notes.provenance" and item.object_id == step.object_id), None
        )
        if child is None:
            child = store.get_current_head(dataset_id, "notes.provenance", step.object_id)
            if child is None or child.operation == "tombstone":
                continue
        result.append(
            ServerOriginMutationStep(
                domain="notes.provenance",
                operation="tombstone",
                object_id=step.object_id,
                parent_id=step.object_id,
                payload=child.payload,
            )
        )
    return tuple(result)


def singleton_delete_group_id(envelope: SyncEnvelopeCreate) -> str:
    """Stable private group identity without changing the original client envelope."""
    identity = (envelope.dataset_id, envelope.device_id, envelope.client_envelope_id)
    return "notes-delete:" + hashlib.sha256(repr(identity).encode()).hexdigest()


def expand_provenance_tombstone(envelope: SyncEnvelopeCreate, *, store: SyncV2Store) -> list[SyncEnvelopeCreate] | None:
    """Create a complete two-member plan for a singleton old-client parent delete."""
    if (
        envelope.domain != "notes.note"
        or envelope.operation != "tombstone"
        or envelope.status != "accepted"
        or envelope.mutation_group_id is not None
    ):
        return None
    child = store.get_current_head(envelope.dataset_id, "notes.provenance", envelope.object_id)
    if child is None or child.operation == "tombstone":
        return None
    group_id = singleton_delete_group_id(envelope)
    parent = replace(
        envelope, mutation_group_id=group_id, mutation_step=0, mutation_step_count=2, mutation_plan_hash="0" * 64
    )
    child_delete = SyncEnvelopeCreate(
        dataset_id=envelope.dataset_id,
        client_envelope_id=group_id + ":provenance",
        domain="notes.provenance",
        operation="tombstone",
        object_id=envelope.object_id,
        parent_id=envelope.object_id,
        device_id="server-origin",
        object_revision=child.object_revision + 1,
        base_server_cursor=child.server_cursor,
        base_object_revision=child.object_revision,
        base_object_hash=child.payload_hash,
        payload=child.payload,
        payload_hash=notes_provenance_object_hash(child.payload, deleted=True),
        deleted=True,
        mutation_group_id=group_id,
        mutation_step=1,
        mutation_step_count=2,
        mutation_plan_hash="0" * 64,
        routing_metadata={"source": "notes-parent-delete", "origin": "server"},
    )
    plan = [parent, child_delete]
    digest = mutation_group_plan_hash(plan)
    return [replace(item, mutation_plan_hash=digest) for item in plan]


def ensure_notes_provenance_ready(*, service: SyncV2Service, note_db: CharactersRAGDB, user_id: str) -> SyncDataset:
    """Enroll an old default profile in bounded 200-note pages and two-step groups.

    Existing canonical heads, especially independent tombstones, always win.
    Source-only heads retain product revisions. Retryable interruptions retain
    their bootstrap ID and durable manifests; malformed/drifting sources fail
    closed without inventing a parent base or replaying product revisions.
    """
    from .server_origin import canonical_payload_hash
    from .server_origin_batch import (
        ServerOriginMutationStep,
        capture_server_origin_mutation_batch,
        load_server_origin_mutation_batch_manifest,
    )

    if str(note_db.owner_user_id) != str(user_id):
        raise SyncStoreError("notes_provenance_owner_mismatch")
    dataset = next(
        (
            item
            for item in service.store.list_datasets_for_user(user_id)
            if item.metadata.get("default_personal") is True and item.metadata.get("client_family") == "chatbook"
        ),
        None,
    )
    if dataset is None or dataset.encryption_policy != "server_trusted_v1":
        raise SyncStoreError("notes_provenance_encryption_unsupported")
    dataset = service.store.db.begin_notes_provenance_bootstrap(
        dataset.dataset_id, owner_user_id=user_id, bootstrap_id=service.id_factory("notes-provenance-bootstrap")
    )
    metadata = dataset.metadata[READINESS_KEY]
    if metadata["state"] == "ready":
        return dataset
    bootstrap_id = metadata["bootstrap_id"]
    captured_count = 0
    expected_count = 0
    source_hash = metadata.get("source_hash")

    def source_note(note_id: str) -> dict:
        note = note_db.get_note_by_id(note_id, include_deleted=True)
        if note is None or str(note.get("client_id")) != str(user_id):
            raise SyncStoreError("notes_provenance_bootstrap_source_invalid")
        return note

    def core_payload(note: Mapping[str, object]) -> dict:
        return validate_notes_note_upsert_payload(
            {key: note.get(key) for key in ("title", "content", "conversation_id", "message_id")}
        )

    def verified(envelope: SyncEnvelope | SyncEnvelopeCreate) -> bool:
        return notes_provenance_bootstrap_matches_source(note_db, envelope)

    def sources(*, prepare_markers: bool = False):
        after_note_id = None
        while True:
            page = note_db.note_provenance_store.list_parent_notes(after_note_id=after_note_id)
            if not page:
                return
            for note in page:
                note_id = note["id"]
                canonical = service.store.get_current_head(dataset.dataset_id, "notes.provenance", note_id)
                record = note_db.note_provenance_store.get(note_id, include_deleted=True)
                if prepare_markers and canonical is None and record is None and not note["deleted"]:
                    marker = read_notes_provenance(note["content"])
                    if marker is not None:
                        with note_db.transaction() as conn:
                            record = note_db.note_provenance_store.put(note_id, marker, expected_version=0, conn=conn)
                            # put holds the parent row lock until this exact-version check commits.
                            if source_note(note_id)["version"] != note["version"]:
                                raise SyncStoreError("notes_provenance_bootstrap_source_changed")
                yield note, record
            after_note_id = page[-1]["id"]

    def summary(*, prepare_markers: bool = False, verify_heads: bool = False):
        digest = hashlib.sha256()
        count = 0
        for note, record in sources(prepare_markers=prepare_markers):
            if verify_heads:
                parent = service.store.get_current_head(dataset.dataset_id, "notes.note", note["id"])
                child = service.store.get_current_head(dataset.dataset_id, "notes.provenance", note["id"])
                if (
                    parent is None
                    or parent.apply_status != "applied"
                    or parent.object_revision != note["version"]
                    or (
                        record is not None
                        and (
                            child is None
                            or child.apply_status != "applied"
                            or child.payload_hash != record["object_hash"]
                        )
                    )
                ):
                    raise SyncStoreError("notes_provenance_bootstrap_source_changed")
            digest.update(json.dumps([note, record], sort_keys=True, separators=(",", ":"), default=str).encode())
            count += 1
        return count, digest.hexdigest()

    try:
        expected_count, digest = summary(prepare_markers=True)
        if source_hash is not None and source_hash != digest:
            raise SyncStoreError("notes_provenance_bootstrap_source_changed")
        source_hash = digest
        dataset = service.store.db.transition_notes_provenance_bootstrap(
            dataset.dataset_id,
            bootstrap_id=bootstrap_id,
            expected_state="initializing",
            state="initializing",
            captured_count=0,
            expected_count=expected_count,
            source_hash=source_hash,
        )
        for note, record in sources():
            note_id = note["id"]
            key = f"{bootstrap_id}:{note_id}"
            steps = load_server_origin_mutation_batch_manifest(
                service=service, dataset_id=dataset.dataset_id, source="notes-provenance-bootstrap", idempotency_key=key
            )
            if steps is None:
                parent = service.store.get_current_head(dataset.dataset_id, "notes.note", note_id)
                canonical = service.store.get_current_head(dataset.dataset_id, "notes.provenance", note_id)
                routing = {"bootstrap_id": bootstrap_id, "notes_provenance_parent_version": note["version"]}
                steps = []
                if parent is None:
                    steps.append(
                        ServerOriginMutationStep(
                            domain="notes.note",
                            operation="tombstone" if note["deleted"] else "upsert",
                            object_id=note_id,
                            payload=core_payload(note),
                            object_revision=note["version"],
                            routing_metadata=routing,
                        )
                    )
                elif (
                    parent.object_revision != note["version"]
                    or parent.operation != ("tombstone" if note["deleted"] else "upsert")
                    or (
                        parent.operation == "upsert"
                        and parent.payload_hash != canonical_payload_hash(core_payload(note))[0]
                    )
                ):
                    raise SyncStoreError("notes_provenance_bootstrap_parent_changed")
                if canonical is None and record is not None:
                    steps.append(
                        ServerOriginMutationStep(
                            domain="notes.provenance",
                            operation="tombstone" if record["deleted"] else "upsert",
                            object_id=note_id,
                            parent_id=note_id,
                            payload=record["payload"],
                            object_revision=record["version"],
                            routing_metadata=routing,
                        )
                    )
                elif canonical is not None and (
                    record is None
                    or canonical.object_revision != record["version"]
                    or canonical.payload_hash != record["object_hash"]
                ):
                    raise SyncStoreError("notes_provenance_bootstrap_source_changed")
            if steps:
                capture_server_origin_mutation_batch(
                    service=service,
                    user_id=user_id,
                    steps=steps,
                    source="notes-provenance-bootstrap",
                    idempotency_key=key,
                    trusted_notes_provenance_bootstrap_id=bootstrap_id,
                    bootstrap_step_verifier=verified,
                )
            captured_count += 1

        def ready() -> bool:
            return summary(verify_heads=True) == (expected_count, source_hash)

        return service.store.db.transition_notes_provenance_bootstrap(
            dataset.dataset_id,
            bootstrap_id=bootstrap_id,
            expected_state="initializing",
            state="ready",
            captured_count=captured_count,
            expected_count=expected_count,
            source_hash=source_hash,
            ready_verifier=ready,
        )
    except Exception as exc:  # noqa: BLE001 - persist safe readiness, never source text.
        from .server_origin_batch import SyncServerOriginBatchMaterializationError

        retryable = isinstance(exc, SyncServerOriginBatchMaterializationError) and exc.retryable
        service.store.db.transition_notes_provenance_bootstrap(
            dataset.dataset_id,
            bootstrap_id=bootstrap_id,
            expected_state="initializing",
            state="initializing" if retryable else "failed",
            captured_count=min(captured_count, expected_count),
            expected_count=expected_count,
            source_hash=source_hash,
            error_code="notes_provenance_bootstrap_incomplete",
        )
        raise SyncStoreError("notes_provenance_sync_not_ready") from exc
