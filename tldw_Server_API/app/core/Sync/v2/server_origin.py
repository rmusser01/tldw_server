from __future__ import annotations

"""Server-origin Sync v2 capture helpers for normal Notes/Chat API writes."""

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TypeVar
from uuid import uuid4

from loguru import logger

from .adapters import (
    AdapterAccepted,
    AdapterConflict,
    AdapterDeferred,
    AdapterRejected,
    SyncAdapterContext,
)
from .errors import SyncMaterializationPredecessorError, SyncStoreError
from .models import (
    CLIENT_PRIVATE_SERVER_FRONTEND_LIMITATION_CODE,
    CLIENT_PRIVATE_SERVER_FRONTEND_LIMITATION_MESSAGE,
    DEFAULT_M1_ENCRYPTION_POLICY,
    SyncConflict,
    SyncDataset,
    SyncDomain,
    SyncEnvelope,
    SyncEnvelopeCreate,
    SyncObjectState,
    SyncOperation,
    server_frontend_mutation_enabled_for_policy,
)
from .personal_context_ongoing_contract import PersonalContextAuthorityMetadata
from .service import SyncV2Service
from .store import SyncV2Store

SERVER_ORIGIN_DEVICE_ID = "server-origin"
_APPLIED_CAPTURE_ADAPTER_VERSION = 1
_APPLIED_CAPTURE_OPERATIONS: frozenset[SyncOperation] = frozenset({"upsert", "append"})
# Recorded on a conflict this module dismisses itself, so the dismissal can be told
# apart from a reviewer's.
STRANDED_TOMBSTONE_RESOLUTION_NOTE = "server_origin_tombstone_of_object_without_sync_history"
# A dataset holds at most one such conflict at a time (it blocks the next append), so
# this only bounds the loop against a resolution that does not take effect.
_MAX_STRANDED_TOMBSTONE_DISMISSALS = 8

_WriteResult = TypeVar("_WriteResult")


def insert_personal_context_authority(
    service: SyncV2Service,
    *,
    envelope: SyncEnvelopeCreate,
    authority: PersonalContextAuthorityMetadata,
    sync_store: SyncV2Store | None = None,
) -> SyncEnvelope:
    """Insert one internal-only already-canonical Personal Context egress row."""

    if authority.role != "home_authority":
        raise SyncStoreError("Personal Context authority role is required")
    store = service.store if sync_store is None else sync_store
    stored = store.insert_envelope(
        replace(
            envelope,
            device_id=SERVER_ORIGIN_DEVICE_ID,
            status="accepted",
            apply_status="pending",
            routing_metadata={
                **envelope.routing_metadata,
                "personal_context_authority": authority.model_dump(mode="json"),
            },
        )
    )
    if stored.server_cursor is None:
        raise SyncStoreError("Personal Context authority receipt is unavailable")
    return stored


class SyncServerOriginMaterializationError(SyncStoreError):
    """Raised when an accepted server-origin envelope is not applied cleanly."""

    def __init__(self, envelope: SyncEnvelope) -> None:
        super().__init__("sync_server_origin_materialization_failed")
        self.envelope = envelope


class SyncServerOriginIdempotencyConflictError(SyncStoreError):
    """Raised when an idempotency key is reused for a different mutation."""

    def __init__(self, envelope: SyncEnvelope) -> None:
        super().__init__("sync_server_origin_idempotency_conflict")
        self.envelope = envelope


class SyncServerOriginMutationNotSupportedError(SyncStoreError):
    """Raised when a dataset policy cannot support trusted server-origin writes."""

    def __init__(self, dataset: SyncDataset, domain: SyncDomain) -> None:
        super().__init__(CLIENT_PRIVATE_SERVER_FRONTEND_LIMITATION_MESSAGE)
        self.dataset = dataset
        self.domain = domain
        self.error_code = CLIENT_PRIVATE_SERVER_FRONTEND_LIMITATION_CODE


class SyncServerOriginRestoreConflictError(SyncStoreError):
    """Raised when a server-origin restore does not target the current tombstone."""

    def __init__(self, object_id: str) -> None:
        super().__init__("Note restore requires the current deleted note version.")
        self.object_id = object_id
        self.error_code = "sync_server_origin_restore_conflict"


@dataclass(frozen=True, slots=True)
class ServerOriginCaptureResult:
    """Result of accepting and materializing a server-origin mutation."""

    dataset: SyncDataset
    envelope: SyncEnvelope


def capture_server_origin_mutation(
    service: SyncV2Service,
    *,
    user_id: str,
    domain: SyncDomain,
    operation: SyncOperation,
    object_id: str,
    payload: dict[str, object],
    source: str,
    parent_id: str | None = None,
    stable_key: str | None = None,
    routing_metadata: Mapping[str, object] | None = None,
) -> ServerOriginCaptureResult:
    """Append and materialize one trusted server-origin Sync v2 mutation.

    A dataset that refuses the append because of the conflict an earlier delete of
    an object without Sync history left behind (see
    ``delete_unenrolled_server_origin_objects``) is unblocked first, and the append
    is tried once more. Every other unresolved conflict still refuses it.
    """

    dataset = _active_default_personal_dataset(service, user_id)
    if domain not in dataset.domains:
        raise SyncStoreError(f"Sync domain is not enrolled for this dataset: {domain}")
    if not server_frontend_mutation_enabled_for_policy(dataset.encryption_policy):
        raise SyncServerOriginMutationNotSupportedError(dataset, domain)

    def append() -> ServerOriginCaptureResult:
        return _append_server_origin_mutation(
            service,
            dataset,
            user_id=user_id,
            domain=domain,
            operation=operation,
            object_id=object_id,
            payload=payload,
            source=source,
            parent_id=parent_id,
            stable_key=stable_key,
            routing_metadata=routing_metadata,
        )

    try:
        return append()
    except SyncMaterializationPredecessorError:
        # The refusal comes from the envelope insert, so nothing was appended.
        if not _unblock_stranded_dataset(service, dataset, user_id=user_id):
            raise
    return append()


def _append_server_origin_mutation(
    service: SyncV2Service,
    dataset: SyncDataset,
    *,
    user_id: str,
    domain: SyncDomain,
    operation: SyncOperation,
    object_id: str,
    payload: dict[str, object],
    source: str,
    parent_id: str | None,
    stable_key: str | None,
    routing_metadata: Mapping[str, object] | None,
) -> ServerOriginCaptureResult:
    """Append one server-origin envelope to ``dataset`` and require its projection."""

    payload_hash, payload_size = canonical_payload_hash(payload)
    if stable_key:
        existing = service.store.list_envelopes_for_entity(
            dataset.dataset_id,
            domain,
            stable_key=stable_key,
            limit=1,
        )
        if existing:
            accepted = existing[0]
            if (
                accepted.operation != operation
                or accepted.object_id != object_id
                or accepted.parent_id != parent_id
                or not _payload_matches_idempotent_replay(accepted, payload, payload_hash)
            ):
                raise SyncServerOriginIdempotencyConflictError(accepted)
            accepted = _require_capture_applied(service, accepted)
            return ServerOriginCaptureResult(dataset=dataset, envelope=accepted)

    state = service.store.get_object_state(dataset.dataset_id, domain, object_id)
    object_revision = 1 if state is None else state.object_revision + 1
    now = service.clock() or None
    client_envelope_id = (
        stable_server_origin_envelope_id(dataset.dataset_id, domain, stable_key)
        if stable_key
        else f"server-origin-{uuid4().hex}"
    )
    canonical_routing_metadata = dict(routing_metadata or {})
    canonical_routing_metadata.update(
        {
            "source": source,
            "origin": "server",
            "server_device_id": SERVER_ORIGIN_DEVICE_ID,
            "server_owner_user_id": user_id,
        }
    )
    envelope = SyncEnvelopeCreate(
        dataset_id=dataset.dataset_id,
        client_envelope_id=client_envelope_id,
        domain=domain,
        operation=operation,
        object_id=object_id,
        device_id=SERVER_ORIGIN_DEVICE_ID,
        client_sequence=None,
        base_server_cursor=state.latest_server_cursor if state is not None else None,
        base_object_revision=state.object_revision if state is not None else None,
        base_object_hash=state.object_hash if state is not None else None,
        object_revision=object_revision,
        parent_id=parent_id,
        schema_version=1,
        payload=dict(payload),
        payload_hash=payload_hash,
        payload_size_bytes=payload_size,
        created_at_client=now,
        deleted=operation == "tombstone",
        encryption_metadata={"policy": DEFAULT_M1_ENCRYPTION_POLICY},
        routing_metadata=canonical_routing_metadata,
        stable_key=stable_key,
    )

    outcome = service._evaluate_envelope(dataset, envelope)
    if isinstance(outcome, AdapterRejected):
        raise SyncStoreError(outcome.message)
    if isinstance(outcome, AdapterDeferred):
        raise SyncStoreError(outcome.message)
    if isinstance(outcome, AdapterConflict):
        raise SyncStoreError(outcome.message or "Sync server-origin mutation conflicted")
    if not isinstance(outcome, AdapterAccepted):
        raise SyncStoreError("Sync server-origin mutation was not accepted")

    inserted = service.store.insert_envelope(envelope)
    inserted = _require_capture_applied(service, inserted)
    return ServerOriginCaptureResult(dataset=dataset, envelope=inserted)


def _require_capture_applied(
    service: SyncV2Service,
    envelope: SyncEnvelope,
) -> SyncEnvelope:
    """Retry replayable capture debt and return only a durably applied envelope."""

    if envelope.apply_status in {"conflict", "superseded"}:
        raise SyncServerOriginMaterializationError(envelope)
    if envelope.apply_status != "applied":
        service._materialize_envelope(envelope)
        envelope = service._envelope_snapshot(envelope)
    if envelope.apply_status != "applied":
        raise SyncServerOriginMaterializationError(envelope)
    return envelope


@dataclass(frozen=True, slots=True)
class AppliedServerOriginObject:
    """One new Sync object whose server projection the caller has already committed."""

    domain: SyncDomain
    operation: SyncOperation
    object_id: str
    payload: dict[str, object]
    parent_id: str | None = None


def capture_applied_server_origin_write(
    service: SyncV2Service,
    *,
    user_id: str,
    source: str,
    claims: Sequence[tuple[SyncDomain, str]],
    write: Callable[[], _WriteResult],
    describe: Callable[[_WriteResult], Sequence[AppliedServerOriginObject]],
) -> _WriteResult:
    """Run a product write inside the Sync projection fence and publish what it created.

    ``capture_server_origin_mutation`` appends an envelope and lets a
    materializer write the projection. That cannot carry a write that must
    validate and commit under the product owner's own transaction, such as a
    versioned chat history admission. Here the order is reversed: the caller's
    ``write`` commits the projection, then each object it created is recorded as
    an accepted envelope that is already applied, with matching object state.
    No materializer runs, so the caller's rows are never rewritten.

    Everything on the Sync side happens in one transaction that holds the
    dataset projection fence, so no other envelope can be accepted or projected
    for the dataset in between. A ``write`` that raises records nothing. The
    product commit and the Sync commit are still two commits: if the second is
    lost, the object exists on the server without an envelope. ``write`` must
    therefore be idempotent, because the retried request is what publishes it.

    Args:
        service: The owner's active Sync service.
        user_id: The authenticated owner.
        source: Routing label for the envelopes, such as ``server_api``.
        claims: The ``(domain, object_id)`` pairs known before the write: what it
            may create, and anything else ``describe`` may report. Their domains
            are the ones fenced. A claimed id whose Sync history is not yet
            applied refuses before the write.
        write: The idempotent product write. It runs at most once per call.
        describe: Maps the write's result to the objects to publish, parents
            first. Only a first-revision create (``upsert`` or ``append``) can be
            published. An object that already has Sync history is skipped, so a
            replayed write publishes nothing twice.

    Returns:
        Whatever ``write`` returned.

    Raises:
        SyncServerOriginMutationNotSupportedError: The dataset policy does not
            allow server-origin writes. Raised before the write.
        SyncMaterializationPredecessorError: An unresolved projection conflict
            blocks new history in the dataset and a claimed id is not published
            yet. Raised before the write.
        SyncServerOriginMaterializationError: A claimed id has Sync history that
            is not applied. Raised before the write.
        SyncStoreError: The dataset, domain or fence is unavailable (before the
            write), or the committed write could not be recorded. Every failure
            after the write is raised as this error and leaves the Sync log
            unchanged. An error raised by ``write`` itself propagates as is.
    """

    domains = list(dict.fromkeys(domain for domain, _object_id in claims))
    if not domains:
        raise SyncStoreError("Sync applied capture requires at least one claimed object")
    dataset = _active_default_personal_dataset(service, user_id)
    if not server_frontend_mutation_enabled_for_policy(dataset.encryption_policy):
        raise SyncServerOriginMutationNotSupportedError(dataset, domains[0])
    for domain in domains:
        if domain not in dataset.domains:
            raise SyncStoreError(f"Sync domain is not enrolled for this dataset: {domain}")
        if not service.adapters.has_domain(domain) or not service.adapters.supports_version(
            domain, _APPLIED_CAPTURE_ADAPTER_VERSION
        ):
            raise SyncStoreError(f"Adapter version is not supported for {domain}")

    # Only a failure of the product write itself is the caller's to interpret. A failure
    # before it means nothing was written; a failure after it, including the Sync commit,
    # means the write stands unpublished. Both are Sync failures and are raised as such.
    phase = "fence"
    try:
        with service.store.projection_fence(dataset.dataset_id, domains) as guarded:
            _require_claims_publishable(service, guarded, dataset, claims, user_id=user_id)
            phase = "write"
            result = write()
            phase = "record"
            context = SyncAdapterContext(
                get_head=lambda domain, object_id: guarded.get_current_head(dataset.dataset_id, domain, object_id),
                list_heads=lambda domain: service._list_current_heads_for_adapter(
                    dataset.dataset_id, domain, store=guarded
                ),
            )
            for item in describe(result):
                if item.operation not in _APPLIED_CAPTURE_OPERATIONS:
                    raise SyncStoreError("Sync applied capture records only newly created objects")
                if (
                    guarded.get_current_head(dataset.dataset_id, item.domain, item.object_id) is not None
                    or guarded.get_object_state(dataset.dataset_id, item.domain, item.object_id) is not None
                ):
                    continue
                _record_applied_object(
                    service, guarded, dataset, item, user_id=user_id, source=source, context=context
                )
    except Exception as exc:
        if phase == "write":
            raise
        if phase == "record":
            logger.warning(
                "Sync capture failed after a server-origin write committed; retrying the request publishes it. "
                "dataset={} claims={} error={}",
                dataset.dataset_id,
                list(claims),
                type(exc).__name__,
            )
        if isinstance(exc, SyncStoreError):
            raise
        raise SyncStoreError(
            "Sync could not record the applied server-origin write"
            if phase == "record"
            else "Sync projection fence is unavailable"
        ) from exc
    return result


def _require_claims_publishable(
    service: SyncV2Service,
    guarded: SyncV2Store,
    dataset: SyncDataset,
    claims: Sequence[tuple[SyncDomain, str]],
    *,
    user_id: str,
) -> None:
    """Refuse, before the product write, a claim the dataset could not accept afterwards."""

    def current_heads() -> list[SyncEnvelope | None]:
        return [guarded.get_current_head(dataset.dataset_id, domain, object_id) for domain, object_id in claims]

    heads = current_heads()
    # A pure replay publishes nothing, so it keeps working whatever blocks the dataset.
    if all(head is not None and head.apply_status == "applied" for head in heads):
        return
    # A dataset blocked by a projection conflict accepts no new history. The conflict a
    # delete of an object without Sync history left behind is dismissed here; any other
    # one refuses the write.
    blocker, dismissed = _dismiss_stranded_tombstones(service, guarded, dataset, user_id=user_id)
    if dismissed:
        heads = current_heads()
    for head in heads:
        if head is not None and head.apply_status != "applied":
            raise SyncServerOriginMaterializationError(head)
    if blocker is not None and any(head is None for head in heads):
        raise SyncMaterializationPredecessorError(
            apply_status="conflict",
            conflict_id=blocker.conflict_id,
            domain=blocker.domain,
            entity_id=blocker.entity_id,
            server_sequence=blocker.server_sequence,
        )


def delete_unenrolled_server_origin_objects(
    service: SyncV2Service,
    *,
    user_id: str,
    objects: Sequence[tuple[SyncDomain, str]],
    delete: Callable[[Sequence[tuple[SyncDomain, str]]], None],
) -> list[tuple[SyncDomain, str]]:
    """Delete, outside the log, the objects the dataset has never held, and return the rest.

    An object is enrolled when the dataset holds Sync history for it: a current
    head or object state. A chat that predates the profile, a row written by a
    path that publishes nothing, and a row whose capture was lost have neither.
    Deleting such a row must not append a tombstone:

    * no device received the row through this dataset, so no device can apply
      the tombstone;
    * the ``chat.message`` materializer rejects a tombstone without a base as a
      projection conflict, and an unresolved projection conflict refuses every
      later append in the dataset.

    So ``delete`` is called with those objects and removes them from the owner's
    database, the same write the route makes when no profile is active. Nothing
    is recorded in the log, and no row is published only to be deleted.

    The check and ``delete`` run inside the dataset projection fence, so no
    envelope for one of these ids can be accepted or projected in between. An
    object whose history is accepted but not applied yet is not "never seen" (a
    device may already have pulled it); it is returned with the enrolled ones and
    the tombstone append then settles or refuses it.

    A request that names both kinds is refused as a whole when the dataset cannot
    take the tombstones: nothing is deleted directly if an unresolved conflict
    blocks the dataset and an enrolled object is among ``objects``.

    Args:
        service: The owner's active Sync service.
        user_id: The authenticated owner.
        objects: The ``(domain, object_id)`` pairs the route is about to delete.
        delete: Removes the given objects from the owner's database. Called at
            most once, with the objects that have no Sync history, in order.

    Returns:
        The objects that have Sync history, in the given order. The caller
        tombstones them through ``capture_server_origin_mutation``.

    Raises:
        SyncServerOriginMutationNotSupportedError: The dataset policy does not
            allow server-origin writes. Raised before any delete.
        SyncMaterializationPredecessorError: An enrolled object needs a tombstone
            and an unresolved projection conflict blocks the dataset. Raised
            before any delete.
        SyncStoreError: The dataset, a domain or the fence is unavailable. Raised
            before any delete. An error raised by ``delete`` itself propagates as is.
            A fence that fails to commit after ``delete`` returned is logged and not
            raised: the rows are deleted and the fence recorded nothing.
    """

    targets = list(objects)
    domains = list(dict.fromkeys(domain for domain, _object_id in targets))
    if not domains:
        return []
    dataset = _active_default_personal_dataset(service, user_id)
    if not server_frontend_mutation_enabled_for_policy(dataset.encryption_policy):
        raise SyncServerOriginMutationNotSupportedError(dataset, domains[0])
    for domain in domains:
        if domain not in dataset.domains:
            raise SyncStoreError(f"Sync domain is not enrolled for this dataset: {domain}")

    enrolled: list[tuple[SyncDomain, str]] = []
    unenrolled: list[tuple[SyncDomain, str]] = []
    phase = "fence"
    try:
        with service.store.projection_fence(dataset.dataset_id, domains) as guarded:
            # A delete that ran before this rule existed may have left its tombstone as the
            # head of one of these objects. Dismissing it first lets the retry go through.
            blocker, _dismissed = _dismiss_stranded_tombstones(service, guarded, dataset, user_id=user_id)
            for domain, object_id in targets:
                has_history = (
                    guarded.get_current_head(dataset.dataset_id, domain, object_id) is not None
                    or guarded.get_object_state(dataset.dataset_id, domain, object_id) is not None
                )
                (enrolled if has_history else unenrolled).append((domain, object_id))
            if enrolled and blocker is not None:
                raise SyncMaterializationPredecessorError(
                    apply_status="conflict",
                    conflict_id=blocker.conflict_id,
                    domain=blocker.domain,
                    entity_id=blocker.entity_id,
                    server_sequence=blocker.server_sequence,
                )
            if unenrolled:
                phase = "delete"
                delete(unenrolled)
                phase = "deleted"
    except Exception as exc:
        if phase == "delete":
            raise
        if phase == "deleted":
            # The rows are deleted in the owner's database. The Sync transaction held only
            # the fence and, at most, a dismissal that the next write repeats, so its failure
            # to commit changes nothing the caller relies on.
            logger.warning(
                "Sync fence did not commit after a direct delete of objects without Sync history. "
                "dataset={} objects={} error={}",
                dataset.dataset_id,
                unenrolled,
                type(exc).__name__,
            )
            return enrolled
        if isinstance(exc, SyncStoreError):
            raise
        raise SyncStoreError("Sync projection fence is unavailable") from exc
    return enrolled


def _unblock_stranded_dataset(service: SyncV2Service, dataset: SyncDataset, *, user_id: str) -> bool:
    """Take the dataset fence and dismiss a stranded tombstone; report whether one was dismissed.

    This runs while a write is being refused because the dataset is blocked, so it
    never raises: if it cannot help, the caller reports the original refusal.
    """

    try:
        with service.store.conflict_resolution_guard(dataset.dataset_id) as guarded:
            _blocker, dismissed = _dismiss_stranded_tombstones(service, guarded, dataset, user_id=user_id)
    except Exception as exc:  # noqa: BLE001 - best effort; the caller re-raises the refusal it already has.
        logger.warning(
            "Sync could not check a blocked dataset for a stranded server-origin tombstone. dataset={} error={}",
            dataset.dataset_id,
            type(exc).__name__,
        )
        return False
    return dismissed


def _dismiss_stranded_tombstones(
    service: SyncV2Service,
    guarded: SyncV2Store,
    dataset: SyncDataset,
    *,
    user_id: str,
) -> tuple[SyncConflict | None, bool]:
    """Unblock a dataset that an earlier delete of an object without Sync history blocked.

    Before ``delete_unenrolled_server_origin_objects`` existed, such a delete
    appended a tombstone the ``chat.message`` materializer rejected, and the
    unresolved conflict refused every later append in the dataset. That conflict
    is resolved here with ``skip``, which marks the tombstone superseded and
    restores the object's head. This loses nothing:

    * the tombstone was never delivered: a pull withholds conflicted and
      superseded envelopes and stops at the blocking one;
    * nothing was projected from it, and the row it named is untouched;
    * it is the server's own write, and its request answered 503, so no client
      was told the delete happened.

    Any other conflict, including a device's tombstone of an unknown message, is
    left for review.

    Args:
        service: The owner's active Sync service.
        guarded: A store that holds the dataset fence. The dismissal commits or
            rolls back with the caller's transaction, and a dismissal that fails
            part-way is rolled back on its own.
        dataset: The dataset server-origin writes are routed to.
        user_id: The authenticated owner.

    Returns:
        The conflict that still blocks the dataset, if any, and whether a stranded
        tombstone was dismissed.
    """

    dismissed = False
    blocker = guarded.get_unresolved_materialization_conflict(dataset.dataset_id)
    for _attempt in range(_MAX_STRANDED_TOMBSTONE_DISMISSALS):
        if blocker is None or not _is_stranded_tombstone(guarded, dataset, blocker):
            break
        try:
            with guarded.conflict_resolution_savepoint():
                service.resolve_conflict(
                    user_id=user_id,
                    dataset_id=dataset.dataset_id,
                    conflict_id=blocker.conflict_id,
                    action="skip",
                    notes=STRANDED_TOMBSTONE_RESOLUTION_NOTE,
                    _conflict=blocker,
                    _store=guarded,
                )
        except SyncStoreError as exc:
            logger.warning(
                "Sync could not dismiss a stranded server-origin tombstone. dataset={} conflict={} error={}",
                dataset.dataset_id,
                blocker.conflict_id,
                type(exc).__name__,
            )
            break
        dismissed = True
        logger.warning(
            "Sync dismissed a server-origin tombstone of an object without Sync history. "
            "dataset={} conflict={} domain={} object={}",
            dataset.dataset_id,
            blocker.conflict_id,
            blocker.domain,
            blocker.object_id,
        )
        blocker = guarded.get_unresolved_materialization_conflict(dataset.dataset_id)
    return blocker, dismissed


def _is_stranded_tombstone(store: SyncV2Store, dataset: SyncDataset, conflict: SyncConflict) -> bool:
    """Return whether a conflict is the server's own tombstone of an object without Sync state."""

    if (
        conflict.domain != "chat.message"
        or conflict.conflict_type != "message_base_conflict"
        or conflict.metadata.get("reason") != "missing_server_message"
        or conflict.server_sequence is None
    ):
        return False
    source = store.get_envelope_by_server_cursor(conflict.server_sequence)
    return (
        source is not None
        and source.dataset_id == dataset.dataset_id
        and source.client_envelope_id == conflict.local_envelope_id
        and source.domain == conflict.domain
        and source.object_id == conflict.object_id
        and source.operation == "tombstone"
        and source.device_id == SERVER_ORIGIN_DEVICE_ID
        and source.status == "accepted"
        and source.apply_status == "conflict"
        and source.base_server_cursor is None
        and source.base_object_revision is None
        and source.base_object_hash is None
        and store.get_object_state(dataset.dataset_id, source.domain, source.object_id) is None
    )


def _record_applied_object(
    service: SyncV2Service,
    guarded: SyncV2Store,
    dataset: SyncDataset,
    item: AppliedServerOriginObject,
    *,
    user_id: str,
    source: str,
    context: SyncAdapterContext,
) -> SyncEnvelope:
    """Append one first-revision envelope and mark it applied in the guarded transaction."""

    payload_hash, payload_size = canonical_payload_hash(item.payload)
    envelope = SyncEnvelopeCreate(
        dataset_id=dataset.dataset_id,
        client_envelope_id=applied_server_origin_envelope_id(
            dataset.dataset_id,
            source=source,
            domain=item.domain,
            operation=item.operation,
            object_id=item.object_id,
        ),
        domain=item.domain,
        operation=item.operation,
        object_id=item.object_id,
        device_id=SERVER_ORIGIN_DEVICE_ID,
        client_sequence=None,
        object_revision=1,
        parent_id=item.parent_id,
        schema_version=_APPLIED_CAPTURE_ADAPTER_VERSION,
        payload=dict(item.payload),
        payload_hash=payload_hash,
        payload_size_bytes=payload_size,
        created_at_client=service.clock() or None,
        deleted=False,
        encryption_metadata={"policy": DEFAULT_M1_ENCRYPTION_POLICY},
        routing_metadata={
            "source": source,
            "origin": "server",
            "server_device_id": SERVER_ORIGIN_DEVICE_ID,
            "server_owner_user_id": user_id,
        },
        stable_key=_applied_server_origin_stable_key(
            source=source,
            domain=item.domain,
            operation=item.operation,
            object_id=item.object_id,
        ),
    )
    outcome = service._evaluate_envelope(dataset, envelope, context=context)
    if isinstance(outcome, (AdapterRejected, AdapterDeferred)):
        raise SyncStoreError(outcome.message)
    if isinstance(outcome, AdapterConflict):
        raise SyncStoreError(outcome.message or "Sync server-origin mutation conflicted")
    if not isinstance(outcome, AdapterAccepted):
        raise SyncStoreError("Sync server-origin mutation was not accepted")

    inserted = guarded.insert_envelope(envelope)
    if inserted.server_cursor is None:
        raise SyncStoreError("Stored Sync envelope is missing a server cursor")
    guarded.upsert_object_state(
        SyncObjectState(
            dataset_id=dataset.dataset_id,
            domain=item.domain,
            object_id=item.object_id,
            object_revision=1,
            object_hash=payload_hash,
            latest_server_cursor=inserted.server_cursor,
            deleted=False,
        )
    )
    return guarded.mark_envelope_apply_status(inserted.server_cursor, apply_status="applied")


def _applied_server_origin_stable_key(
    *,
    source: str,
    domain: SyncDomain,
    operation: SyncOperation,
    object_id: str,
) -> str:
    """Key an applied capture by object id, apart from ``Idempotency-Key`` stable keys."""

    digest = hashlib.sha256(object_id.encode("utf-8")).hexdigest()
    return f"{source}:{domain}:{operation}:applied:{digest}"


def applied_server_origin_envelope_id(
    dataset_id: str,
    *,
    source: str,
    domain: SyncDomain,
    operation: SyncOperation,
    object_id: str,
) -> str:
    """Return the deterministic envelope id of an object's applied capture."""

    return stable_server_origin_envelope_id(
        dataset_id,
        domain,
        _applied_server_origin_stable_key(source=source, domain=domain, operation=operation, object_id=object_id),
    )


def capture_server_origin_note_restore(
    service: SyncV2Service,
    *,
    user_id: str,
    object_id: str,
    note: Mapping[str, object],
    expected_version: int,
    source: str,
) -> ServerOriginCaptureResult:
    """Validate and capture a restore of the current Notes tombstone."""

    current_version = note.get("version")
    try:
        version_matches = int(current_version) == int(expected_version)
    except (TypeError, ValueError):
        version_matches = False
    if not bool(note.get("deleted")) or not version_matches:
        raise SyncServerOriginRestoreConflictError(object_id)

    return capture_server_origin_mutation(
        service,
        user_id=user_id,
        domain="notes.note",
        operation="upsert",
        object_id=object_id,
        payload={
            "title": str(note.get("title") or ""),
            "content": str(note.get("content") or ""),
            "conversation_id": note.get("conversation_id"),
            "message_id": note.get("message_id"),
        },
        source=source,
        routing_metadata={"restore_intent": True},
    )


def server_origin_stable_key(
    *,
    source: str,
    domain: SyncDomain,
    operation: SyncOperation,
    idempotency_key: str | None,
) -> str | None:
    """Return a privacy-preserving stable key for API idempotency."""

    if idempotency_key is None:
        return None
    normalized = idempotency_key.strip()
    if not normalized:
        return None
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return f"{source}:{domain}:{operation}:{digest}"


def server_origin_object_id(domain: SyncDomain, idempotency_key: str | None) -> str | None:
    """Return a deterministic object id for active-Sync create retries."""

    if idempotency_key is None:
        return None
    normalized = idempotency_key.strip()
    if not normalized:
        return None
    digest = hashlib.sha256(f"{domain}:{normalized}".encode()).hexdigest()
    return f"{domain.replace('.', '-')}-{digest[:32]}"


def stable_server_origin_envelope_id(
    dataset_id: str,
    domain: SyncDomain,
    stable_key: str | None,
) -> str:
    """Return a deterministic client envelope id for active-Sync API retries."""

    digest = hashlib.sha256(f"{dataset_id}:{domain}:{stable_key}".encode()).hexdigest()
    return f"server-origin-{digest[:32]}"


def get_active_server_origin_sync_service_for_user(user_id: str) -> SyncV2Service | None:
    """Return the active personal service, preserving lookup failures."""

    from .factory import sync_v2_service_for_user, sync_v2_storage_exists_for_user

    if not sync_v2_storage_exists_for_user(user_id):
        return None
    service = sync_v2_service_for_user(user_id)
    for dataset in service.store.list_datasets_for_user(user_id):
        if (
            dataset.scope_type == "personal"
            and dataset.metadata.get("default_personal") is True
            and dataset.metadata.get("client_family") == "chatbook"
        ):
            return service
    return None


def canonical_payload_hash(payload: dict[str, object]) -> tuple[str, int]:
    """Return the canonical server-trusted payload hash and encoded size."""

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}", len(encoded)


def _payload_matches_idempotent_replay(
    accepted: SyncEnvelope,
    payload: dict[str, object],
    payload_hash: str,
) -> bool:
    if accepted.payload_hash == payload_hash:
        return True
    if accepted.domain != "chat.message" or accepted.operation != "append":
        return False

    accepted_payload = dict(accepted.payload)
    replay_payload = dict(payload)
    accepted_payload.pop("timestamp", None)
    replay_payload.pop("timestamp", None)
    accepted_without_timestamp, _ = canonical_payload_hash(accepted_payload)
    replay_without_timestamp, _ = canonical_payload_hash(replay_payload)
    return accepted_without_timestamp == replay_without_timestamp


def _active_default_personal_dataset(service: SyncV2Service, user_id: str) -> SyncDataset:
    for dataset in service.store.list_datasets_for_user(user_id):
        if (
            dataset.scope_type == "personal"
            and dataset.metadata.get("default_personal") is True
            and dataset.metadata.get("client_family") == "chatbook"
        ):
            return dataset
    raise SyncStoreError("Sync default personal dataset was not found or is not accessible")


__all__ = [
    "SERVER_ORIGIN_DEVICE_ID",
    "STRANDED_TOMBSTONE_RESOLUTION_NOTE",
    "AppliedServerOriginObject",
    "ServerOriginCaptureResult",
    "CLIENT_PRIVATE_SERVER_FRONTEND_LIMITATION_CODE",
    "SyncServerOriginMaterializationError",
    "SyncServerOriginMutationNotSupportedError",
    "SyncServerOriginRestoreConflictError",
    "applied_server_origin_envelope_id",
    "canonical_payload_hash",
    "insert_personal_context_authority",
    "capture_applied_server_origin_write",
    "capture_server_origin_note_restore",
    "capture_server_origin_mutation",
    "delete_unenrolled_server_origin_objects",
    "get_active_server_origin_sync_service_for_user",
]
