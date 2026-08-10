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
from .errors import (
    SyncIdempotencyConflictError,
    SyncMaterializationPredecessorError,
    SyncStoreError,
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
    StoredMutationGroupValidationError,
    materialization_group_view,
    mutation_group_plan_hash,
    validate_stored_mutation_group,
)
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
    bootstrap_relationship_verifier: Callable[[SyncDomain, str, Mapping[str, object]], bool]
    | None = None,
    bootstrap_relationship_absence_verifier: Callable[
        [SyncDomain, str, Mapping[str, object]], bool
    ]
    | None = None,
    bootstrap_step_verifier: Callable[[SyncEnvelope], bool] | None = None,
) -> ServerOriginBatchResult:
    """Preflight, atomically append, and ordered-materialize one complete plan."""

    plan = tuple(steps)
    if not plan:
        raise SyncStoreError("Sync server-origin mutation batch must contain at least one step")
    normalized_key = idempotency_key.strip()
    if not normalized_key:
        raise SyncStoreError("Sync server-origin mutation batch requires an idempotency key")
    if (
        trusted_notes_organization_bootstrap_id is not None
        and bootstrap_step_verifier is None
    ):
        raise SyncStoreError("Sync bootstrap capture requires a source-step verifier")

    dataset = _active_default_personal_dataset(service, user_id)
    _require_batch_write_ready(
        dataset,
        {step.domain for step in plan},
        trusted_bootstrap_id=trusted_notes_organization_bootstrap_id,
    )
    if not server_frontend_mutation_enabled_for_policy(dataset.encryption_policy):
        raise SyncServerOriginMutationNotSupportedError(dataset, plan[0].domain)

    canonical_steps = tuple(
        _canonical_step(step, source=source, user_id=user_id) for step in plan
    )
    mutation_plan_hash = _mutation_plan_hash(canonical_steps)
    mutation_group_id = _mutation_group_id(
        dataset.dataset_id,
        source=source,
        idempotency_key=normalized_key,
    )

    existing = service.store.list_mutation_group(dataset.dataset_id, mutation_group_id)
    if existing:
        try:
            _validate_stored_group(
                existing,
                dataset_id=dataset.dataset_id,
                mutation_group_id=mutation_group_id,
                expected_steps=canonical_steps,
            )
        except SyncIdempotencyConflictError as exc:
            raise SyncServerOriginBatchIdempotencyConflictError(mutation_group_id) from exc
        return _materialize_group(
            service=service,
            dataset=dataset,
            envelopes=existing,
            bootstrap_id=trusted_notes_organization_bootstrap_id,
            bootstrap_step_verifier=bootstrap_step_verifier,
        )

    envelopes = _evaluate_plan(
        service=service,
        dataset=dataset,
        canonical_steps=canonical_steps,
        mutation_group_id=mutation_group_id,
        mutation_plan_hash=mutation_plan_hash,
        bootstrap_id=trusted_notes_organization_bootstrap_id,
        bootstrap_relationship_verifier=bootstrap_relationship_verifier,
        bootstrap_relationship_absence_verifier=(
            bootstrap_relationship_absence_verifier
        ),
    )
    try:
        if trusted_notes_organization_bootstrap_id is None:
            inserted = service.store.insert_envelopes_atomic(envelopes)
        else:
            inserted = service.store.insert_envelopes_atomic(
                envelopes,
                trusted_notes_organization_bootstrap_id=trusted_notes_organization_bootstrap_id,
            )
    except SyncIdempotencyConflictError as exc:
        raise SyncServerOriginBatchIdempotencyConflictError(mutation_group_id) from exc
    except SyncStoreError as exc:
        raise SyncServerOriginBatchAppendError(mutation_group_id) from exc
    return _materialize_group(
        service=service,
        dataset=dataset,
        envelopes=inserted,
        bootstrap_id=trusted_notes_organization_bootstrap_id,
        bootstrap_step_verifier=bootstrap_step_verifier,
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
) -> ServerOriginBatchResult:
    """Resume at the first non-applied step without skipping a blocked step."""

    dataset = service.store.get_dataset(dataset_id)
    if dataset is None or dataset.archived_at is not None:
        raise SyncStoreError("Sync dataset was not found or is not accessible")
    envelopes = service.store.list_mutation_group(dataset_id, mutation_group_id)
    if not envelopes:
        raise SyncStoreError("Sync server-origin mutation group was not found")
    _require_batch_write_ready(dataset, {envelope.domain for envelope in envelopes})
    _validate_stored_group(
        envelopes,
        dataset_id=dataset_id,
        mutation_group_id=mutation_group_id,
    )
    return _materialize_group(service=service, dataset=dataset, envelopes=envelopes)


def _evaluate_plan(
    *,
    service: SyncV2Service,
    dataset: SyncDataset,
    canonical_steps: Sequence[ServerOriginMutationStep],
    mutation_group_id: str,
    mutation_plan_hash: str,
    bootstrap_id: str | None = None,
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
        payload_hash, payload_size = canonical_payload_hash(dict(step.payload))
        object_revision = _next_object_revision(prior_head)
        base_server_cursor = prior_head.server_cursor if prior_head is not None else None
        if isinstance(prior_head, SyncEnvelopeCreate) and base_server_cursor is None:
            # Cursor zero is outside stored history and marks an in-plan virtual head;
            # revision and hash carry the actual optimistic lineage until append.
            base_server_cursor = 0
        envelope = SyncEnvelopeCreate(
            dataset_id=dataset.dataset_id,
            client_envelope_id=_envelope_id(mutation_group_id, index),
            domain=step.domain,
            operation=step.operation,
            object_id=step.object_id,
            device_id=SERVER_ORIGIN_DEVICE_ID,
            base_server_cursor=base_server_cursor,
            base_object_revision=(
                prior_head.object_revision if prior_head is not None else None
            ),
            base_object_hash=prior_head.payload_hash if prior_head is not None else None,
            object_revision=object_revision,
            parent_id=step.parent_id,
            schema_version=1,
            payload=dict(step.payload),
            payload_hash=payload_hash,
            payload_size_bytes=payload_size,
            created_at_client=service.clock() or None,
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
            trusted_server_origin=bootstrap_id is not None,
            organization_group_state=("initializing" if bootstrap_id is not None else None),
            organization_bootstrap_id=bootstrap_id,
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
) -> ServerOriginBatchResult:
    group = list(envelopes)
    _validate_stored_group(
        group,
        dataset_id=dataset.dataset_id,
        mutation_group_id=group[0].mutation_group_id or "",
    )
    try:
        with service.store.materialization_guard(
            group,
            require_predecessors=bootstrap_id is None,
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


def _materialize_group_guarded(
    *,
    service: SyncV2Service,
    store: SyncV2Store,
    dataset: SyncDataset,
    group: list[SyncEnvelope],
    bootstrap_id: str | None,
    bootstrap_step_verifier: Callable[[SyncEnvelope], bool] | None,
) -> tuple[ServerOriginBatchResult, bool | None]:
    """Project a complete group while the caller retains all object locks."""

    for index, envelope in enumerate(materialization_group_view(group)):
        if envelope.apply_status in {"applied", "superseded"}:
            continue
        if envelope.apply_status == "conflict":
            return _materialization_result(dataset, group), False
        if bootstrap_id is not None and bootstrap_step_verifier is not None:
            if envelope.server_cursor is None or not bootstrap_step_verifier(envelope):
                return _materialization_result(dataset, group), True
            store.mark_bootstrap_envelope_verified(
                envelope.server_cursor,
                bootstrap_id=bootstrap_id,
            )
            group = store.list_mutation_group(
                dataset.dataset_id,
                envelope.mutation_group_id or "",
            )
            continue
        materialization = service._materialize_envelope(envelope, store=store)
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
    expected_plan_matches = expected_steps is None or _mutation_plan_hash(
        tuple(_canonical_step_from_envelope(envelope) for envelope in envelopes)
    ) == _mutation_plan_hash(expected_steps)
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
    routing_metadata = dict(step.routing_metadata)
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
    )


def _mutation_plan_hash(steps: Sequence[ServerOriginMutationStep]) -> str:
    encoded = json.dumps(
        [
            {
                "domain": step.domain,
                "operation": step.operation,
                "object_id": step.object_id,
                "payload": step.payload,
                "parent_id": step.parent_id,
                "routing_metadata": step.routing_metadata,
                "stable_key": step.stable_key,
            }
            for step in steps
        ],
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
) -> None:
    missing = sorted(domains.difference(dataset.domains))
    if missing:
        raise SyncStoreError(
            "Sync domains are not enrolled for this dataset: " + ", ".join(missing)
        )
    if not domains.intersection(NOTES_ORGANIZATION_DOMAINS):
        return
    metadata = dataset.metadata.get("notes_organization_v1")
    state = metadata.get("state") if isinstance(metadata, Mapping) else None
    current_bootstrap_id = (
        metadata.get("bootstrap_id") if isinstance(metadata, Mapping) else None
    )
    if state == "ready":
        return
    if (
        state != "initializing"
        or trusted_bootstrap_id is None
        or current_bootstrap_id != trusted_bootstrap_id
    ):
        raise SyncStoreError("notes_organization_sync_not_ready")


__all__ = [
    "ServerOriginBatchResult",
    "ServerOriginMutationStep",
    "SyncServerOriginBatchAppendError",
    "SyncServerOriginBatchIdempotencyConflictError",
    "SyncServerOriginBatchMaterializationError",
    "capture_server_origin_mutation_batch",
    "resume_server_origin_mutation_group",
]
