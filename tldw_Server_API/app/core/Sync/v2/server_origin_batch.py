from __future__ import annotations

"""Durable, ordered server-origin Sync mutation-group coordination."""

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace

from .adapters import (
    AdapterAccepted,
    AdapterConflict,
    AdapterDeferred,
    AdapterRejected,
    SyncAdapterContext,
    SyncHead,
)
from .attachment_refs_v2 import attachment_ref_v2_object_hash
from .errors import (
    SyncIdempotencyConflictError,
    SyncMaterializationPredecessorError,
    SyncStoreError,
)
from .materializers.guarded_product_mutation import (
    GUARD_REQUIRED_ROUTING_KEY,
    GuardedProductMutation,
    GuardedProductMutationIdentityError,
    has_guard_required_routing_key,
)
from .models import (
    DEFAULT_M1_ENCRYPTION_POLICY,
    NOTES_ORGANIZATION_DOMAINS,
    SyncDataset,
    SyncDomain,
    SyncEnvelope,
    SyncEnvelopeCreate,
    SyncOperation,
    server_frontend_mutation_enabled_for_policy,
)
from .mutation_group_validation import (
    SYNC_MUTATION_GROUP_MAX_SIZE,
    StoredMutationGroupValidationError,
    materialization_group_view,
    mutation_group_plan_hash,
    validate_stored_mutation_group,
)
from .notes_task_contract import (
    notes_task_activity_object_hash,
    notes_task_object_hash,
    parse_notes_task_activity_v1,
    parse_notes_task_v1,
)
from .notes_task_readiness import notes_task_capture_is_active
from .server_origin import (
    SERVER_ORIGIN_DEVICE_ID,
    SyncServerOriginMutationNotSupportedError,
    canonical_payload_hash,
)
from .service import SyncV2Service
from .store import SyncV2Store


@dataclass(frozen=True, slots=True)
class ServerOriginMutationStep:
    """One primitive mutation in an ordered server-origin plan."""

    domain: SyncDomain
    operation: SyncOperation
    object_id: str
    payload: Mapping[str, object]
    parent_id: str | None = None
    routing_metadata: Mapping[str, object] = field(default_factory=dict)
    stable_key: str | None = None
    created_at_client: str | None = None
    schema_version: int = 1
    adapter_version: int = 1
    client_envelope_id: str | None = None
    object_revision: int | None = None
    base_object_revision: int | None = None
    base_object_hash: str | None = None


@dataclass(frozen=True, slots=True)
class ServerOriginBatchResult:
    """Stored group and whether every product projection is applied."""

    dataset: SyncDataset
    envelopes: tuple[SyncEnvelope, ...]
    fully_applied: bool


class SyncServerOriginBatchIdempotencyConflictError(SyncIdempotencyConflictError):
    """Raised when a stable group identity is reused for a different plan."""

    error_code = "sync_server_origin_batch_idempotency_conflict"

    def __init__(self, mutation_group_id: str) -> None:
        super().__init__(self.error_code)
        self.mutation_group_id = mutation_group_id


class SyncServerOriginBatchMaterializationError(SyncStoreError):
    """Raised when a durable group has an incomplete product projection."""

    error_code = "sync_server_origin_batch_materialization_failed"

    def __init__(self, result: ServerOriginBatchResult, *, retryable: bool) -> None:
        super().__init__(self.error_code)
        self.result = result
        self.retryable = retryable


class SyncServerOriginBatchAppendError(SyncStoreError):
    """Raised when the complete preflighted group cannot be appended atomically."""

    error_code = "sync_server_origin_batch_append_failed"

    def __init__(self, mutation_group_id: str) -> None:
        super().__init__(self.error_code)
        self.mutation_group_id = mutation_group_id


def capture_server_origin_mutation_batch(
    *,
    service: SyncV2Service,
    user_id: str,
    steps: Sequence[ServerOriginMutationStep],
    source: str,
    idempotency_key: str,
    trusted_notes_organization_bootstrap_id: str | None = None,
    trusted_notes_link_bootstrap_id: str | None = None,
    trusted_notes_attachment_bootstrap_id: str | None = None,
    trusted_notes_task_bootstrap_id: str | None = None,
    trusted_notes_task_activity_bootstrap_id: str | None = None,
    trusted_notes_task_coordinator: bool = False,
    bootstrap_relationship_verifier: Callable[[SyncDomain, str, Mapping[str, object]], bool]
    | None = None,
    bootstrap_relationship_absence_verifier: Callable[
        [SyncDomain, str, Mapping[str, object]], bool
    ]
    | None = None,
    bootstrap_step_verifier: Callable[[SyncEnvelope], bool] | None = None,
    guarded_mutation: GuardedProductMutation | None = None,
    guarded_mutations: Sequence[GuardedProductMutation] = (),
) -> ServerOriginBatchResult:
    """Preflight, atomically append, and ordered-materialize one complete plan."""

    plan = tuple(steps)
    if not plan:
        raise SyncStoreError("Sync server-origin mutation batch must contain at least one step")
    if len(plan) > SYNC_MUTATION_GROUP_MAX_SIZE:
        raise SyncStoreError("Sync server-origin mutation batch exceeds the 1000-step limit")
    normalized_key = idempotency_key.strip()
    if not normalized_key:
        raise SyncStoreError("Sync server-origin mutation batch requires an idempotency key")
    trusted_contexts = tuple(
        item
        for item in (
            trusted_notes_organization_bootstrap_id,
            trusted_notes_link_bootstrap_id,
            trusted_notes_attachment_bootstrap_id,
            trusted_notes_task_bootstrap_id,
            trusted_notes_task_activity_bootstrap_id,
        )
        if item is not None
    )
    if len(trusted_contexts) > 1:
        raise SyncStoreError("Only one trusted bootstrap context may be supplied")
    if trusted_notes_task_coordinator and trusted_contexts:
        raise SyncStoreError("Task coordinator capture cannot use a bootstrap context")
    if trusted_notes_task_coordinator and not {
        step.domain for step in plan
    }.issubset({"notes.task", "notes.task_activity", "notes.note"}):
        raise SyncStoreError("Task coordinator capture contains an unsupported domain")
    trusted_bootstrap_id = (
        trusted_notes_organization_bootstrap_id
        or trusted_notes_link_bootstrap_id
        or trusted_notes_attachment_bootstrap_id
        or trusted_notes_task_bootstrap_id
        or trusted_notes_task_activity_bootstrap_id
    )
    if trusted_bootstrap_id is not None and bootstrap_step_verifier is None:
        raise SyncStoreError("Sync bootstrap capture requires a source-step verifier")

    dataset = _active_default_personal_dataset(service, user_id)
    _require_batch_write_ready(
        dataset,
        {step.domain for step in plan},
        trusted_bootstrap_id=trusted_bootstrap_id,
        notes_task_bootstrap=trusted_notes_task_bootstrap_id is not None,
        notes_task_activity_bootstrap=(
            trusted_notes_task_activity_bootstrap_id is not None
        ),
        notes_task_coordinator=trusted_notes_task_coordinator,
    )
    if not server_frontend_mutation_enabled_for_policy(dataset.encryption_policy):
        raise SyncServerOriginMutationNotSupportedError(dataset, plan[0].domain)

    canonical_steps = tuple(
        _canonical_step(step, source=source, user_id=user_id) for step in plan
    )
    guards = _normalize_guarded_mutations(
        guarded_mutation=guarded_mutation,
        guarded_mutations=guarded_mutations,
    )
    canonical_steps = _bind_guard_requirements(canonical_steps, ())
    guarded_canonical_steps = _bind_guard_requirements(canonical_steps, guards)
    mutation_group_id = _mutation_group_id(
        dataset.dataset_id,
        source=source,
        idempotency_key=normalized_key,
    )

    existing = service.store.list_mutation_group(dataset.dataset_id, mutation_group_id)
    if existing:
        expected_steps = (
            guarded_canonical_steps
            if any(
                envelope.routing_metadata.get(GUARD_REQUIRED_ROUTING_KEY) is True
                for envelope in existing
            )
            else canonical_steps
        )
        try:
            _validate_stored_group(
                existing,
                dataset_id=dataset.dataset_id,
                mutation_group_id=mutation_group_id,
                expected_steps=expected_steps,
            )
        except SyncIdempotencyConflictError as exc:
            raise SyncServerOriginBatchIdempotencyConflictError(mutation_group_id) from exc
        return _materialize_group(
            service=service,
            dataset=dataset,
            envelopes=existing,
            bootstrap_id=trusted_bootstrap_id,
            bootstrap_step_verifier=bootstrap_step_verifier,
            materialize_verified_bootstrap=(
                trusted_notes_attachment_bootstrap_id is not None
                or trusted_notes_task_activity_bootstrap_id is not None
            ),
            notes_task_bootstrap=trusted_notes_task_bootstrap_id is not None,
            notes_task_activity_bootstrap=(
                trusted_notes_task_activity_bootstrap_id is not None
            ),
            guarded_mutations=guards,
        )

    canonical_steps = guarded_canonical_steps
    mutation_plan_hash = _mutation_plan_hash(canonical_steps)
    envelopes = _evaluate_plan(
        service=service,
        dataset=dataset,
        canonical_steps=canonical_steps,
        mutation_group_id=mutation_group_id,
        mutation_plan_hash=mutation_plan_hash,
        bootstrap_id=trusted_bootstrap_id,
        notes_link_bootstrap=trusted_notes_link_bootstrap_id is not None,
        notes_attachment_bootstrap=(
            trusted_notes_attachment_bootstrap_id is not None
        ),
        notes_task_bootstrap=trusted_notes_task_bootstrap_id is not None,
        notes_task_activity_bootstrap=(
            trusted_notes_task_activity_bootstrap_id is not None
        ),
        notes_task_coordinator=trusted_notes_task_coordinator,
        bootstrap_relationship_verifier=bootstrap_relationship_verifier,
        bootstrap_relationship_absence_verifier=(
            bootstrap_relationship_absence_verifier
        ),
    )
    try:
        if trusted_bootstrap_id is None:
            if trusted_notes_task_coordinator:
                inserted = service.store.insert_envelopes_atomic(
                    envelopes,
                    trusted_notes_task_coordinator=True,
                )
            else:
                inserted = service.store.insert_envelopes_atomic(envelopes)
        else:
            inserted = service.store.insert_envelopes_atomic(
                envelopes,
                trusted_notes_organization_bootstrap_id=trusted_bootstrap_id,
                trusted_notes_task_bootstrap_id=(
                    trusted_notes_task_bootstrap_id
                    or trusted_notes_task_activity_bootstrap_id
                ),
            )
    except SyncIdempotencyConflictError as exc:
        raise SyncServerOriginBatchIdempotencyConflictError(mutation_group_id) from exc
    except SyncStoreError as exc:
        raise SyncServerOriginBatchAppendError(mutation_group_id) from exc
    return _materialize_group(
        service=service,
        dataset=dataset,
        envelopes=inserted,
        bootstrap_id=trusted_bootstrap_id,
        bootstrap_step_verifier=bootstrap_step_verifier,
        materialize_verified_bootstrap=(
            trusted_notes_attachment_bootstrap_id is not None
            or trusted_notes_task_activity_bootstrap_id is not None
        ),
        notes_task_bootstrap=trusted_notes_task_bootstrap_id is not None,
        notes_task_activity_bootstrap=(
            trusted_notes_task_activity_bootstrap_id is not None
        ),
        guarded_mutations=guards,
    )


def load_server_origin_mutation_batch_manifest(
    *,
    service: SyncV2Service,
    dataset_id: str,
    source: str,
    idempotency_key: str,
) -> tuple[ServerOriginMutationStep, ...] | None:
    """Load one validated durable group as its authoritative step manifest."""

    normalized_key = idempotency_key.strip()
    if not normalized_key:
        raise SyncStoreError("Sync server-origin mutation batch requires an idempotency key")
    mutation_group_id = _mutation_group_id(
        dataset_id,
        source=source,
        idempotency_key=normalized_key,
    )
    existing = service.store.list_mutation_group(dataset_id, mutation_group_id)
    if not existing:
        return None
    _validate_stored_group(
        existing,
        dataset_id=dataset_id,
        mutation_group_id=mutation_group_id,
    )
    return tuple(_canonical_step_from_envelope(envelope) for envelope in existing)


def server_origin_mutation_batch_group_id(
    *,
    dataset_id: str,
    source: str,
    idempotency_key: str,
) -> str:
    """Resolve the durable group identity for one server-origin idempotency key."""

    normalized_key = idempotency_key.strip()
    if not normalized_key:
        raise SyncStoreError("Sync server-origin mutation batch requires an idempotency key")
    return _mutation_group_id(
        dataset_id,
        source=source,
        idempotency_key=normalized_key,
    )


def resume_server_origin_mutation_group(
    *,
    service: SyncV2Service,
    dataset_id: str,
    mutation_group_id: str,
    guarded_mutations: Sequence[GuardedProductMutation] = (),
) -> ServerOriginBatchResult:
    """Resume at the first non-applied step without skipping a blocked step."""

    dataset = service.store.get_dataset(dataset_id)
    if dataset is None or dataset.archived_at is not None:
        raise SyncStoreError("Sync dataset was not found or is not accessible")
    envelopes = service.store.list_mutation_group(dataset_id, mutation_group_id)
    if not envelopes:
        raise SyncStoreError("Sync server-origin mutation group was not found")
    _validate_stored_group(
        envelopes,
        dataset_id=dataset_id,
        mutation_group_id=mutation_group_id,
    )
    notes_task_coordinator = _is_trusted_notes_task_coordinator_group(
        dataset,
        envelopes,
    )
    _require_batch_write_ready(
        dataset,
        {envelope.domain for envelope in envelopes},
        notes_task_coordinator=notes_task_coordinator,
    )
    guards = _normalize_guarded_mutations(guarded_mutations=guarded_mutations)
    return _materialize_group(
        service=service,
        dataset=dataset,
        envelopes=envelopes,
        guarded_mutations=guards,
    )


def materialize_accepted_mutation_group(
    *,
    service: SyncV2Service,
    dataset: SyncDataset,
    envelopes: Sequence[SyncEnvelope],
    guarded_mutations: Sequence[GuardedProductMutation] = (),
) -> ServerOriginBatchResult:
    """Materialize one already-validated accepted group in dependency order."""

    if not envelopes or envelopes[0].mutation_group_id is None:
        raise SyncStoreError("Sync accepted mutation group is missing group identity")
    _validate_stored_group(
        envelopes,
        dataset_id=dataset.dataset_id,
        mutation_group_id=envelopes[0].mutation_group_id,
    )
    guards = _normalize_guarded_mutations(guarded_mutations=guarded_mutations)
    return _materialize_group(
        service=service,
        dataset=dataset,
        envelopes=envelopes,
        guarded_mutations=guards,
    )


def _is_trusted_notes_task_coordinator_group(
    dataset: SyncDataset,
    envelopes: Sequence[SyncEnvelope],
) -> bool:
    """Recognize only a validated, closed server-origin task mutation group."""

    if not envelopes or any(
        envelope.device_id != SERVER_ORIGIN_DEVICE_ID
        or envelope.routing_metadata.get("origin") != "server"
        or envelope.routing_metadata.get("server_device_id")
        != SERVER_ORIGIN_DEVICE_ID
        or envelope.routing_metadata.get("server_owner_user_id")
        != dataset.owner_user_id
        for envelope in envelopes
    ):
        return False
    try:
        from .notes_task_coordinator import _validate_task_mutation_plan

        _validate_task_mutation_plan(
            tuple(_canonical_step_from_envelope(envelope) for envelope in envelopes)
        )
    except (ImportError, SyncStoreError):
        return False
    return True


def is_trusted_notes_task_coordinator_envelope(
    *,
    service: SyncV2Service,
    dataset: SyncDataset,
    envelope: SyncEnvelope,
) -> bool:
    """Recognize one envelope only through its complete validated coordinator group."""

    group_id = envelope.mutation_group_id
    if group_id is None:
        return False
    group = service.store.list_mutation_group(dataset.dataset_id, group_id)
    try:
        _validate_stored_group(
            group,
            dataset_id=dataset.dataset_id,
            mutation_group_id=group_id,
        )
    except (SyncIdempotencyConflictError, SyncStoreError):
        return False
    return bool(
        any(item.server_cursor == envelope.server_cursor for item in group)
        and _is_trusted_notes_task_coordinator_group(dataset, group)
    )


def _evaluate_plan(
    *,
    service: SyncV2Service,
    dataset: SyncDataset,
    canonical_steps: Sequence[ServerOriginMutationStep],
    mutation_group_id: str,
    mutation_plan_hash: str,
    bootstrap_id: str | None = None,
    notes_link_bootstrap: bool = False,
    notes_attachment_bootstrap: bool = False,
    notes_task_bootstrap: bool = False,
    notes_task_activity_bootstrap: bool = False,
    notes_task_coordinator: bool = False,
    bootstrap_relationship_verifier: Callable[[SyncDomain, str, Mapping[str, object]], bool]
    | None = None,
    bootstrap_relationship_absence_verifier: Callable[
        [SyncDomain, str, Mapping[str, object]], bool
    ]
    | None = None,
) -> list[SyncEnvelopeCreate]:
    overlay: dict[tuple[SyncDomain, str], SyncEnvelopeCreate] = {}
    stored: dict[tuple[SyncDomain, str], SyncEnvelope | None] = {}

    def get_head(domain: SyncDomain, object_id: str) -> SyncHead | None:
        key = (domain, object_id)
        if key in overlay:
            return overlay[key]
        if key not in stored:
            stored[key] = service.store.get_current_head(
                dataset.dataset_id, domain, object_id
            )
        return stored[key]

    def list_heads(domain: SyncDomain) -> Sequence[SyncHead]:
        planned = {
            object_id: head
            for (head_domain, object_id), head in overlay.items()
            if head_domain == domain
        }
        stored_heads = {
            head.object_id: head
            for head in service._list_current_heads_for_adapter(
                dataset.dataset_id, domain
            )
        }
        stored_heads.update(planned)
        return tuple(stored_heads.values())

    envelopes: list[SyncEnvelopeCreate] = []
    step_count = len(canonical_steps)
    for index, step in enumerate(canonical_steps):
        prior_head = get_head(step.domain, step.object_id)
        task_readiness = dataset.metadata.get("notes_task_v1")
        trusted_prebootstrap_task_capture = bool(
            notes_task_coordinator
            and step.domain == "notes.task"
            and prior_head is None
            and step.base_object_revision is not None
            and step.base_object_hash is not None
            and isinstance(task_readiness, Mapping)
            and task_readiness.get("state") in {"enrolling", "bootstrapping"}
        )
        if step.base_object_revision is not None:
            if prior_head is None:
                if not trusted_prebootstrap_task_capture:
                    raise SyncStoreError("notes_task_prebootstrap_base_invalid")
            elif (
                step.base_object_revision != prior_head.object_revision
                or step.base_object_hash != prior_head.payload_hash
            ):
                raise SyncStoreError("notes_task_coordinator_base_conflict")
        object_revision = (
            _next_object_revision(prior_head)
            if step.object_revision is None
            else step.object_revision
        )
        payload_hash, payload_size = canonical_payload_hash(dict(step.payload))
        if notes_attachment_bootstrap and step.domain == "attachment.ref":
            payload_hash = attachment_ref_v2_object_hash(
                step.operation,
                step.payload,
                object_revision=object_revision,
            )
        if (notes_task_bootstrap or notes_task_coordinator) and step.domain == "notes.task":
            task_payload = parse_notes_task_v1(
                step.payload,
                owner_user_id=dataset.owner_user_id,
            )
            payload_hash = notes_task_object_hash(
                task_payload,
                revision=object_revision,
                deleted=step.operation == "tombstone",
            )
        if (
            notes_task_activity_bootstrap or notes_task_coordinator
        ) and step.domain == "notes.task_activity":
            if step.operation != "upsert" or object_revision != 1:
                raise SyncStoreError("notes_task_activity_bootstrap_lineage_invalid")
            activity_payload = parse_notes_task_activity_v1(
                step.payload,
                owner_user_id=dataset.owner_user_id,
                bound_actor_type=str(step.payload.get("actor_type")),
                bound_actor_id=step.payload.get("actor_id"),
                authenticated_device_id=None,
                trusted_server_origin=True,
            )
            payload_hash = notes_task_activity_object_hash(
                activity_payload,
                revision=1,
                deleted=False,
            )
        base_server_cursor = prior_head.server_cursor if prior_head is not None else None
        if isinstance(prior_head, SyncEnvelopeCreate) and base_server_cursor is None:
            # Cursor zero is outside stored history and marks an in-plan virtual head;
            # revision and hash carry the actual optimistic lineage until append.
            base_server_cursor = 0
        envelope = SyncEnvelopeCreate(
            dataset_id=dataset.dataset_id,
            client_envelope_id=(
                step.client_envelope_id or _envelope_id(mutation_group_id, index)
            ),
            domain=step.domain,
            operation=step.operation,
            object_id=step.object_id,
            device_id=SERVER_ORIGIN_DEVICE_ID,
            base_server_cursor=base_server_cursor,
            base_object_revision=(
                prior_head.object_revision
                if prior_head is not None
                else step.base_object_revision
            ),
            base_object_hash=(
                prior_head.payload_hash if prior_head is not None else step.base_object_hash
            ),
            base_version=(
                (
                    prior_head.entity_version
                    if prior_head.entity_version is not None
                    else prior_head.object_revision
                )
                if step.domain == "notes.link" and prior_head is not None
                else None
            ),
            object_revision=object_revision,
            entity_version=(
                object_revision
                if step.domain in {"notes.link", "notes.task", "notes.task_activity"}
                else None
            ),
            parent_id=step.parent_id,
            schema_version=step.schema_version,
            adapter_version=step.adapter_version,
            payload=dict(step.payload),
            payload_hash=payload_hash,
            payload_size_bytes=payload_size,
            created_at_client=step.created_at_client or service.clock() or None,
            deleted=step.operation == "tombstone",
            encryption_metadata={"policy": DEFAULT_M1_ENCRYPTION_POLICY},
            routing_metadata=dict(step.routing_metadata),
            stable_key=step.stable_key,
            mutation_group_id=mutation_group_id,
            mutation_step=index,
            mutation_step_count=step_count,
            mutation_plan_hash=mutation_plan_hash,
        )
        if service._payload_exceeds_size_limit(envelope):
            raise SyncStoreError("Sync envelope payload exceeds the server size limit")
        history = service.store.list_envelopes_for_entity(
            dataset.dataset_id,
            step.domain,
            entity_id=step.object_id,
            stable_key=step.stable_key,
            limit=100,
        )
        planned_prior = overlay.get((step.domain, step.object_id))
        context = SyncAdapterContext(
            prior_envelopes=(*history, *((planned_prior,) if planned_prior else ())),
            get_head=get_head,
            list_heads=list_heads,
            get_authorized_note=lambda note_id: get_head("notes.note", note_id),
            get_authorized_task=lambda task_id: get_head("notes.task", task_id),
            trusted_server_origin=(
                bootstrap_id is not None or notes_task_coordinator
            ),
            trusted_notes_task_prebootstrap_capture=(
                trusted_prebootstrap_task_capture
            ),
            organization_group_state=("initializing" if bootstrap_id is not None else None),
            organization_bootstrap_id=(
                bootstrap_id
                if not notes_link_bootstrap and not notes_attachment_bootstrap
                else None
            ),
            notes_link_bootstrap_id=(bootstrap_id if notes_link_bootstrap else None),
            attachment_ref_bootstrap_id=(
                bootstrap_id if notes_attachment_bootstrap else None
            ),
            notes_task_bootstrap_id=(bootstrap_id if notes_task_bootstrap else None),
            notes_task_activity_bootstrap_id=(
                bootstrap_id if notes_task_activity_bootstrap else None
            ),
            authenticated_actor_type=(
                str(step.payload.get("actor_type"))
                if step.domain == "notes.task_activity"
                else None
            ),
            authenticated_actor_id=(
                step.payload.get("actor_id")
                if step.domain == "notes.task_activity"
                else None
            ),
            bootstrap_relationship_verifier=bootstrap_relationship_verifier,
            bootstrap_relationship_absence_verifier=(
                bootstrap_relationship_absence_verifier
            ),
        )
        outcome = service._evaluate_envelope(dataset, envelope, context=context)
        if isinstance(outcome, AdapterRejected | AdapterDeferred):
            raise SyncStoreError(outcome.message)
        if isinstance(outcome, AdapterConflict):
            raise SyncStoreError(outcome.message or "Sync server-origin mutation conflicted")
        if not isinstance(outcome, AdapterAccepted):
            raise SyncStoreError("Sync server-origin mutation was not accepted")
        overlay[(step.domain, step.object_id)] = envelope
        envelopes.append(envelope)
    stored_plan_hash = _materialization_plan_hash(envelopes)
    return [replace(envelope, mutation_plan_hash=stored_plan_hash) for envelope in envelopes]


def _materialize_group(
    *,
    service: SyncV2Service,
    dataset: SyncDataset,
    envelopes: Sequence[SyncEnvelope],
    bootstrap_id: str | None = None,
    bootstrap_step_verifier: Callable[[SyncEnvelope], bool] | None = None,
    materialize_verified_bootstrap: bool = False,
    notes_task_bootstrap: bool = False,
    notes_task_activity_bootstrap: bool = False,
    guarded_mutations: Sequence[GuardedProductMutation] = (),
) -> ServerOriginBatchResult:
    group = list(envelopes)
    _validate_stored_group(
        group,
        dataset_id=dataset.dataset_id,
        mutation_group_id=group[0].mutation_group_id or "",
    )
    guard_by_identity = _require_group_guards(dataset, group, guarded_mutations)
    trusted_notes_task_coordinator = (
        bootstrap_id is None
        and _is_trusted_notes_task_coordinator_group(dataset, group)
    )
    try:
        with service.store.materialization_guard(
            group,
            require_predecessors=bootstrap_id is None,
            trusted_notes_task_bootstrap_id=(
                bootstrap_id
                if notes_task_bootstrap or notes_task_activity_bootstrap
                else None
            ),
            trusted_notes_task_coordinator=trusted_notes_task_coordinator,
        ) as guarded_store:
            group = guarded_store.list_mutation_group(
                dataset.dataset_id,
                group[0].mutation_group_id or "",
            )
            _validate_apply_status_vector(group)
            result, retryable = _materialize_group_guarded(
                service=service,
                store=guarded_store,
                dataset=dataset,
                group=group,
                bootstrap_id=bootstrap_id,
                bootstrap_step_verifier=bootstrap_step_verifier,
                materialize_verified_bootstrap=materialize_verified_bootstrap,
                notes_task_bootstrap=notes_task_bootstrap,
                guard_by_identity=guard_by_identity,
            )
    except SyncIdempotencyConflictError:
        raise
    except SyncMaterializationPredecessorError as exc:
        raise _materialization_error(dataset, group, retryable=exc.retryable) from exc
    except Exception as exc:  # noqa: BLE001 - storage/commit failures are retryable.
        raise _materialization_error(dataset, group, retryable=True) from exc
    if retryable is not None:
        raise SyncServerOriginBatchMaterializationError(result, retryable=retryable)
    return result


def _normalize_guarded_mutations(
    *,
    guarded_mutation: GuardedProductMutation | None = None,
    guarded_mutations: Sequence[GuardedProductMutation] = (),
) -> tuple[GuardedProductMutation, ...]:
    guards = (*guarded_mutations, *((guarded_mutation,) if guarded_mutation else ()))
    identities: set[tuple[SyncDomain, str]] = set()
    for guard in guards:
        if not isinstance(guard, GuardedProductMutation):
            raise TypeError("Guarded product mutations must be process-local capabilities")
        identity = (guard.expected_domain, guard.expected_object_id)
        if identity in identities:
            raise GuardedProductMutationIdentityError(
                "Guarded product mutation identity is duplicate"
            )
        identities.add(identity)
    return guards


def _bind_guard_requirements(
    steps: Sequence[ServerOriginMutationStep],
    guards: Sequence[GuardedProductMutation],
) -> tuple[ServerOriginMutationStep, ...]:
    stripped = tuple(
        replace(
            step,
            routing_metadata={
                key: value
                for key, value in step.routing_metadata.items()
                if key != GUARD_REQUIRED_ROUTING_KEY
            },
        )
        for step in steps
    )
    if not guards:
        return stripped

    eligible: dict[tuple[SyncDomain, str], list[int]] = {}
    for index, step in enumerate(stripped):
        if GuardedProductMutation.supports_domain(step.domain):
            eligible.setdefault((step.domain, step.object_id), []).append(index)
    supplied = {
        (guard.expected_domain, guard.expected_object_id): guard for guard in guards
    }
    missing = set(eligible).difference(supplied)
    if missing:
        raise GuardedProductMutationIdentityError(
            "Guarded product mutation plan is missing a required guard"
        )
    for identity in supplied:
        if len(eligible.get(identity, ())) != 1:
            raise GuardedProductMutationIdentityError(
                "Guarded product mutation identity must match exactly one plan step"
            )

    bound = list(stripped)
    for indexes in eligible.values():
        index = indexes[0]
        step = bound[index]
        bound[index] = replace(
            step,
            routing_metadata={
                **dict(step.routing_metadata),
                GUARD_REQUIRED_ROUTING_KEY: True,
            },
        )
    return tuple(bound)


def _require_group_guards(
    dataset: SyncDataset,
    group: Sequence[SyncEnvelope],
    guards: Sequence[GuardedProductMutation],
) -> dict[tuple[SyncDomain, str], GuardedProductMutation]:
    required: dict[tuple[SyncDomain, str], SyncEnvelope] = {}
    for envelope in group:
        marker = envelope.routing_metadata.get(GUARD_REQUIRED_ROUTING_KEY)
        if has_guard_required_routing_key(envelope.routing_metadata) and marker is not True:
            raise SyncIdempotencyConflictError(
                "Sync stored mutation group has an invalid guard-required marker"
            )
        if marker is True:
            if not GuardedProductMutation.supports_domain(
                envelope.domain
            ) or not _has_guarded_server_origin_provenance(dataset, envelope):
                raise SyncIdempotencyConflictError(
                    "Sync guard-required envelope failed integrity validation"
                )
            identity = (envelope.domain, envelope.object_id)
            if identity in required:
                raise GuardedProductMutationIdentityError(
                    "Guard-required identity must match exactly one envelope"
                )
            required[identity] = envelope

    supplied = {
        (guard.expected_domain, guard.expected_object_id): guard for guard in guards
    }
    if not required:
        for identity in supplied:
            if sum(
                envelope.domain == identity[0] and envelope.object_id == identity[1]
                for envelope in group
            ) != 1:
                raise GuardedProductMutationIdentityError(
                    "Fresh guarded mutation must match exactly one unmarked envelope"
                )
        return supplied
    if set(required).difference(supplied):
        raise GuardedProductMutationIdentityError(
            "Guard-required Sync envelope requires a fresh matching guard"
        )
    if set(supplied).difference(required):
        raise GuardedProductMutationIdentityError(
            "Fresh guarded mutation does not match a guard-required envelope"
        )
    return supplied


def _has_guarded_server_origin_provenance(
    dataset: SyncDataset,
    envelope: SyncEnvelope,
) -> bool:
    routing = envelope.routing_metadata
    source = routing.get("source")
    return (
        envelope.device_id == SERVER_ORIGIN_DEVICE_ID
        and routing.get("origin") == "server"
        and routing.get("server_device_id") == SERVER_ORIGIN_DEVICE_ID
        and routing.get("server_owner_user_id") == dataset.owner_user_id
        and isinstance(source, str)
        and bool(source)
    )


def _materialize_group_guarded(
    *,
    service: SyncV2Service,
    store: SyncV2Store,
    dataset: SyncDataset,
    group: list[SyncEnvelope],
    bootstrap_id: str | None,
    bootstrap_step_verifier: Callable[[SyncEnvelope], bool] | None,
    materialize_verified_bootstrap: bool,
    notes_task_bootstrap: bool,
    guard_by_identity: Mapping[
        tuple[SyncDomain, str],
        GuardedProductMutation,
    ],
) -> tuple[ServerOriginBatchResult, bool | None]:
    """Project a complete group while the caller retains all object locks."""

    for index, envelope in enumerate(materialization_group_view(group)):
        envelope_guard = guard_by_identity.get((envelope.domain, envelope.object_id))
        if envelope.apply_status in {"applied", "superseded"} and envelope_guard is None:
            continue
        if envelope.apply_status == "conflict":
            return _materialization_result(dataset, group), False
        if bootstrap_id is not None and bootstrap_step_verifier is not None:
            if envelope.server_cursor is None or not bootstrap_step_verifier(envelope):
                return _materialization_result(dataset, group), True
            if materialize_verified_bootstrap:
                materialization = service._materialize_envelope(
                    envelope,
                    store=store,
                )
                if materialization.status != "applied":
                    group = store.list_mutation_group(
                        dataset.dataset_id,
                        envelope.mutation_group_id or "",
                    )
                    return _materialization_result(dataset, group), True
            else:
                store.mark_bootstrap_envelope_verified(
                    envelope.server_cursor,
                    bootstrap_id=bootstrap_id,
                    notes_task_bootstrap=notes_task_bootstrap,
                )
            group = store.list_mutation_group(
                dataset.dataset_id,
                envelope.mutation_group_id or "",
            )
            continue
        materialization = service._materialize_envelope(
            envelope,
            store=store,
            guarded_mutation=envelope_guard,
        )
        group = store.list_mutation_group(
            dataset.dataset_id,
            envelope.mutation_group_id or "",
        )
        current = group[index]
        if materialization.status == "skipped" and current.apply_status != "applied":
            if current.server_cursor is None:
                raise SyncIdempotencyConflictError(
                    "Sync stored mutation group step has no server cursor"
                )
            store.mark_envelope_apply_status(
                current.server_cursor,
                apply_status="failed",
                apply_error_code="sync_projection_materializer_missing",
                apply_error_message="Projection materializer is not registered",
            )
            group = store.list_mutation_group(
                dataset.dataset_id,
                current.mutation_group_id or "",
            )
            return _materialization_result(dataset, group), True
        if materialization.status == "conflict" or current.apply_status == "conflict":
            return _materialization_result(dataset, group), False
        if materialization.status == "failed" or current.apply_status == "failed":
            return _materialization_result(dataset, group), True
        if current.apply_status != "applied":
            if current.server_cursor is None:
                raise SyncIdempotencyConflictError(
                    "Sync stored mutation group step has no server cursor"
                )
            store.mark_envelope_apply_status(
                current.server_cursor,
                apply_status="failed",
                apply_error_code="sync_projection_status_missing",
                apply_error_message="Projection did not record applied status",
            )
            group = store.list_mutation_group(
                dataset.dataset_id,
                current.mutation_group_id or "",
            )
            return _materialization_result(dataset, group), True
    return ServerOriginBatchResult(
        dataset=dataset,
        envelopes=tuple(group),
        fully_applied=all(
            envelope.apply_status in {"applied", "superseded"} for envelope in group
        ),
    ), None


def _materialization_result(
    dataset: SyncDataset,
    envelopes: Sequence[SyncEnvelope],
) -> ServerOriginBatchResult:
    return ServerOriginBatchResult(
        dataset=dataset,
        envelopes=tuple(envelopes),
        fully_applied=False,
    )


def _validate_apply_status_vector(envelopes: Sequence[SyncEnvelope]) -> None:
    seen_non_applied = False
    for envelope in envelopes:
        if envelope.apply_status in {"applied", "superseded"}:
            if seen_non_applied:
                raise SyncIdempotencyConflictError(
                    "Sync stored mutation group contains a non-prefix applied step"
                )
            continue
        seen_non_applied = True


def _materialization_error(
    dataset: SyncDataset,
    envelopes: Sequence[SyncEnvelope],
    *,
    retryable: bool,
) -> SyncServerOriginBatchMaterializationError:
    return SyncServerOriginBatchMaterializationError(
        _materialization_result(dataset, envelopes),
        retryable=retryable,
    )


def _validate_stored_group(
    envelopes: Sequence[SyncEnvelope],
    *,
    dataset_id: str,
    mutation_group_id: str,
    expected_steps: Sequence[ServerOriginMutationStep] | None = None,
) -> None:
    try:
        validate_stored_mutation_group(
            envelopes,
            dataset_id=dataset_id,
            mutation_group_id=mutation_group_id,
        )
    except StoredMutationGroupValidationError as exc:
        if exc.error_code == "mutation_group_fingerprint_invalid":
            raise SyncIdempotencyConflictError(
                "Sync stored mutation group fingerprint does not match its plan hash"
            ) from exc
        raise SyncIdempotencyConflictError(
            "Sync stored mutation group shape is invalid"
        ) from exc
    stored_steps = tuple(
        _canonical_step_from_envelope(envelope) for envelope in envelopes
    )
    if expected_steps is not None and len(stored_steps) == len(expected_steps):
        stored_steps = tuple(
            replace(
                stored,
                client_envelope_id=(
                    stored.client_envelope_id
                    if expected.client_envelope_id is not None
                    else None
                ),
                object_revision=(
                    stored.object_revision
                    if expected.object_revision is not None
                    else None
                ),
                base_object_revision=(
                    stored.base_object_revision
                    if expected.base_object_revision is not None
                    else None
                ),
                base_object_hash=(
                    stored.base_object_hash
                    if expected.base_object_hash is not None
                    else None
                ),
            )
            for stored, expected in zip(stored_steps, expected_steps, strict=True)
        )
    expected_plan_matches = expected_steps is None or (
        len(stored_steps) == len(expected_steps)
        and _mutation_plan_hash(stored_steps) == _mutation_plan_hash(expected_steps)
    )
    if not expected_plan_matches:
        raise SyncIdempotencyConflictError(
            "Sync stored mutation group fingerprint does not match its plan hash"
        )


def _canonical_step(
    step: ServerOriginMutationStep,
    *,
    source: str,
    user_id: str,
) -> ServerOriginMutationStep:
    if step.object_revision is not None and (
        type(step.object_revision) is not int or step.object_revision < 1
    ):
        raise SyncStoreError("Sync server-origin object revision must be positive")
    base_values = (step.base_object_revision, step.base_object_hash)
    if any(value is not None for value in base_values) != all(
        value is not None for value in base_values
    ):
        raise SyncStoreError("Sync server-origin object base must be complete")
    if step.base_object_revision is not None and (
        type(step.base_object_revision) is not int
        or step.base_object_revision < 1
        or step.object_revision != step.base_object_revision + 1
    ):
        raise SyncStoreError("Sync server-origin object base revision is invalid")
    routing_metadata = dict(step.routing_metadata)
    if not (step.domain == "attachment.ref" and step.adapter_version == 2):
        routing_metadata.update(
            {
                "source": source,
                "origin": "server",
                "server_device_id": SERVER_ORIGIN_DEVICE_ID,
                "server_owner_user_id": user_id,
            }
        )
    return ServerOriginMutationStep(
        domain=step.domain,
        operation=step.operation,
        object_id=step.object_id,
        payload=dict(step.payload),
        parent_id=step.parent_id,
        routing_metadata=routing_metadata,
        stable_key=step.stable_key,
        created_at_client=step.created_at_client,
        schema_version=step.schema_version,
        adapter_version=step.adapter_version,
        client_envelope_id=step.client_envelope_id,
        object_revision=step.object_revision,
        base_object_revision=step.base_object_revision,
        base_object_hash=step.base_object_hash,
    )


def _canonical_step_from_envelope(envelope: SyncEnvelope) -> ServerOriginMutationStep:
    return ServerOriginMutationStep(
        domain=envelope.domain,
        operation=envelope.operation,
        object_id=envelope.object_id,
        payload=dict(envelope.payload),
        parent_id=envelope.parent_id,
        routing_metadata=dict(envelope.routing_metadata),
        stable_key=envelope.stable_key,
        created_at_client=envelope.created_at_client,
        schema_version=envelope.schema_version,
        adapter_version=envelope.adapter_version,
        client_envelope_id=envelope.client_envelope_id,
        object_revision=envelope.object_revision,
        base_object_revision=envelope.base_object_revision,
        base_object_hash=envelope.base_object_hash,
    )


def _mutation_plan_hash(steps: Sequence[ServerOriginMutationStep]) -> str:
    plan: list[dict[str, object]] = []
    for step in steps:
        encoded_step: dict[str, object] = {
            "domain": step.domain,
            "operation": step.operation,
            "object_id": step.object_id,
            "payload": step.payload,
            "parent_id": step.parent_id,
            "routing_metadata": step.routing_metadata,
            "stable_key": step.stable_key,
        }
        if step.schema_version != 1 or step.adapter_version != 1:
            encoded_step["schema_version"] = step.schema_version
            encoded_step["adapter_version"] = step.adapter_version
        if step.client_envelope_id is not None:
            encoded_step["client_envelope_id"] = step.client_envelope_id
        if step.object_revision is not None:
            encoded_step["object_revision"] = step.object_revision
        if step.base_object_revision is not None:
            encoded_step["base_object_revision"] = step.base_object_revision
            encoded_step["base_object_hash"] = step.base_object_hash
        plan.append(encoded_step)
    encoded = json.dumps(
        plan,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _materialization_plan_hash(envelopes: Sequence[SyncHead]) -> str:
    return mutation_group_plan_hash(envelopes)


def _mutation_group_id(
    dataset_id: str,
    *,
    source: str,
    idempotency_key: str,
) -> str:
    key_hash = hashlib.sha256(idempotency_key.encode("utf-8")).hexdigest()
    digest = hashlib.sha256(f"{dataset_id}:{source}:{key_hash}".encode()).hexdigest()
    return f"server-origin-group-{digest[:32]}"


def _envelope_id(mutation_group_id: str, step: int) -> str:
    digest = hashlib.sha256(f"{mutation_group_id}:{step}".encode()).hexdigest()
    return f"server-origin-{digest[:32]}"


def _next_object_revision(head: SyncHead | None) -> int:
    if head is None or head.object_revision is None:
        return 1
    return head.object_revision + 1


def _active_default_personal_dataset(
    service: SyncV2Service,
    user_id: str,
) -> SyncDataset:
    for dataset in service.store.list_datasets_for_user(user_id):
        if (
            dataset.scope_type == "personal"
            and dataset.metadata.get("default_personal") is True
            and dataset.metadata.get("client_family") == "chatbook"
        ):
            return dataset
    raise SyncStoreError("Sync default personal dataset was not found or is not accessible")


def _require_batch_write_ready(
    dataset: SyncDataset,
    domains: set[SyncDomain],
    *,
    trusted_bootstrap_id: str | None = None,
    notes_task_bootstrap: bool = False,
    notes_task_activity_bootstrap: bool = False,
    notes_task_coordinator: bool = False,
) -> None:
    missing_domains = domains.difference(dataset.domains)
    if notes_task_bootstrap:
        missing_domains = missing_domains.difference({"notes.task"})
    if notes_task_activity_bootstrap:
        missing_domains = missing_domains.difference({"notes.task_activity"})
    capture_active = notes_task_capture_is_active(dataset.metadata)
    if notes_task_coordinator and capture_active:
        missing_domains = missing_domains.difference(
            {"notes.task", "notes.task_activity"}
        )
    missing = sorted(missing_domains)
    if missing:
        raise SyncStoreError(
            "Sync domains are not enrolled for this dataset: " + ", ".join(missing)
        )
    if "notes.task" in domains:
        metadata = dataset.metadata.get("notes_task_v1")
        state = metadata.get("state") if isinstance(metadata, Mapping) else None
        if notes_task_coordinator and capture_active:
            pass
        elif not notes_task_bootstrap or trusted_bootstrap_id is None or state != "bootstrapping":
            raise SyncStoreError("notes_task_sync_not_ready")
    if "notes.task_activity" in domains:
        metadata = dataset.metadata.get("notes_task_activity_v1")
        state = metadata.get("state") if isinstance(metadata, Mapping) else None
        if notes_task_coordinator and capture_active:
            pass
        elif (
            not notes_task_activity_bootstrap
            or trusted_bootstrap_id is None
            or state != "bootstrapping"
        ):
            raise SyncStoreError("notes_task_activity_sync_not_ready")
    if "notes.link" in domains:
        metadata = dataset.metadata.get("notes_link_v1")
        state = metadata.get("state") if isinstance(metadata, Mapping) else None
        current_bootstrap_id = (
            metadata.get("bootstrap_id") if isinstance(metadata, Mapping) else None
        )
        if state != "ready" and (
            state != "initializing"
            or trusted_bootstrap_id is None
            or current_bootstrap_id != trusted_bootstrap_id
        ):
            raise SyncStoreError("notes_link_sync_not_ready")
    if "attachment.ref" in domains:
        metadata = dataset.metadata.get("notes_attachment_v2")
        state = metadata.get("state") if isinstance(metadata, Mapping) else None
        current_bootstrap_id = (
            metadata.get("bootstrap_id") if isinstance(metadata, Mapping) else None
        )
        if state != "ready" and (
            state != "initializing"
            or trusted_bootstrap_id is None
            or current_bootstrap_id != trusted_bootstrap_id
        ):
            raise SyncStoreError("notes_attachment_sync_not_ready")
    if domains.intersection(NOTES_ORGANIZATION_DOMAINS):
        metadata = dataset.metadata.get("notes_organization_v1")
        state = metadata.get("state") if isinstance(metadata, Mapping) else None
        current_bootstrap_id = (
            metadata.get("bootstrap_id") if isinstance(metadata, Mapping) else None
        )
        if state != "ready" and (
            state != "initializing"
            or trusted_bootstrap_id is None
            or current_bootstrap_id != trusted_bootstrap_id
        ):
            raise SyncStoreError("notes_organization_sync_not_ready")


__all__ = [
    "is_trusted_notes_task_coordinator_envelope",
    "materialize_accepted_mutation_group",
    "ServerOriginBatchResult",
    "ServerOriginMutationStep",
    "SyncServerOriginBatchAppendError",
    "SyncServerOriginBatchIdempotencyConflictError",
    "SyncServerOriginBatchMaterializationError",
    "capture_server_origin_mutation_batch",
    "resume_server_origin_mutation_group",
]
