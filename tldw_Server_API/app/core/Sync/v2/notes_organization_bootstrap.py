from __future__ import annotations

"""Durable bootstrap of existing ChaCha Notes organization state into Sync v2."""

import hashlib
import json
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from typing import Protocol

from tldw_Server_API.app.core.DB_Management.chacha.organization_sync_store import (
    NotesOrganizationSyncStore,
    OrganizationResource,
    OrganizationSnapshot,
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

from .errors import SyncStoreError
from .models import NOTES_ORGANIZATION_DOMAINS, SyncDataset, SyncDomain, SyncEnvelope
from .server_origin_batch import (
    ServerOriginMutationStep,
    SyncServerOriginBatchMaterializationError,
    capture_server_origin_mutation_batch,
    load_server_origin_mutation_batch_manifest,
)
from .service import SyncV2Service

_RESOURCE_DOMAINS: tuple[SyncDomain, ...] = (
    "notes.keyword",
    "notes.keyword_collection",
    "notes.folder",
)
_RELATIONSHIP_DOMAINS: tuple[SyncDomain, ...] = tuple(
    domain for domain in NOTES_ORGANIZATION_DOMAINS if domain not in _RESOURCE_DOMAINS
)
_SAFE_SOURCE_ERROR = "notes_organization_bootstrap_source_invalid"
_SAFE_CAPTURE_ERROR = "notes_organization_bootstrap_capture_failed"
_BOOTSTRAP_HISTORY_PAGE_SIZE = 200


class SyncDatasetBootstrapper(Protocol):
    """Injected dataset bootstrap boundary used by profile enrollment."""

    def bootstrap(
        self,
        *,
        service: SyncV2Service,
        user_id: str,
        dataset: SyncDataset,
    ) -> SyncDataset:
        """Capture existing product state and return the latest dataset state."""


class NotesOrganizationBootstrapInterrupted(RuntimeError):
    """Test/worker interruption that intentionally leaves durable initializing state."""


class _SourceInvalidError(SyncStoreError):
    pass


class NotesOrganizationBootstrapper:
    """Capture one user's existing Notes organization state without replaying it."""

    def __init__(
        self,
        note_db: CharactersRAGDB,
        *,
        batch_size: int = 200,
        after_group: Callable[[int], None] | None = None,
    ) -> None:
        if batch_size < 1 or batch_size > 1000:
            raise ValueError("Notes organization bootstrap batch_size must be 1..1000")
        self._projection = NotesOrganizationSyncStore(note_db)
        self._batch_size = batch_size
        self._after_group = after_group

    def bootstrap(
        self,
        *,
        service: SyncV2Service,
        user_id: str,
        dataset: SyncDataset,
    ) -> SyncDataset:
        """Resume the current bootstrap ID to ready, or record a safe failed state."""

        if dataset.owner_user_id != user_id:
            raise SyncStoreError("Sync dataset was not found or is not accessible")
        metadata = dataset.metadata.get("notes_organization_v1")
        if not isinstance(metadata, Mapping):
            raise SyncStoreError("notes_organization_sync_not_ready")
        state = metadata.get("state")
        bootstrap_id = metadata.get("bootstrap_id")
        if state == "ready":
            return dataset
        if state != "initializing" or not isinstance(bootstrap_id, str) or not bootstrap_id:
            raise SyncStoreError("notes_organization_sync_not_ready")

        expected_count = int(metadata.get("expected_count") or 0)
        captured_count = int(metadata.get("captured_count") or 0)
        try:
            snapshot, represented_resources, stale_groups = (
                self._drain_preexisting_heads(service, dataset)
            )
            removed_relationships = self._captured_relationship_removals(
                service,
                dataset,
                snapshot,
                bootstrap_id=bootstrap_id,
            )
            planned_omissions = (
                represented_resources if removed_relationships else set()
            )
            restore_resources = self._resource_restore_keys(
                service,
                dataset.dataset_id,
                snapshot,
                planned_omissions,
            )
            steps = self._plan(
                snapshot,
                bootstrap_id=bootstrap_id,
                removed_relationships=removed_relationships,
                represented_resources=planned_omissions,
                restore_resources=restore_resources,
            )
            expected_count = len(snapshot.resources) + len(snapshot.relationships)
            dataset = service.store.transition_notes_organization_bootstrap(
                dataset.dataset_id,
                bootstrap_id=bootstrap_id,
                expected_state="initializing",
                state="initializing",
                captured_count=min(captured_count, expected_count),
                expected_count=expected_count,
            )
            snapshot_hash = _snapshot_hash(snapshot)
            remaining_steps = list(steps)
            group_index = 0
            completed_groups = 0
            while True:
                idempotency_key = f"{bootstrap_id}:{snapshot_hash}:{group_index}"
                stored_manifest = load_server_origin_mutation_batch_manifest(
                    service=service,
                    dataset_id=dataset.dataset_id,
                    source="notes-organization-bootstrap",
                    idempotency_key=idempotency_key,
                )
                if stored_manifest is not None:
                    batch = list(stored_manifest)
                    remaining_steps = _without_manifest_steps(
                        remaining_steps,
                        stored_manifest,
                    )
                elif remaining_steps:
                    batch = remaining_steps[: self._batch_size]
                    del remaining_steps[: self._batch_size]
                else:
                    break
                capture_server_origin_mutation_batch(
                    service=service,
                    user_id=user_id,
                    steps=batch,
                    source="notes-organization-bootstrap",
                    idempotency_key=idempotency_key,
                    trusted_notes_organization_bootstrap_id=bootstrap_id,
                    bootstrap_relationship_verifier=self._relationship_matches_source,
                    bootstrap_relationship_absence_verifier=(
                        self._relationship_absent_from_source
                    ),
                    bootstrap_step_verifier=self._step_matches_source,
                )
                captured_count = min(expected_count, captured_count + len(batch))
                dataset = service.store.transition_notes_organization_bootstrap(
                    dataset.dataset_id,
                    bootstrap_id=bootstrap_id,
                    expected_state="initializing",
                    state="initializing",
                    captured_count=captured_count,
                    expected_count=expected_count,
                )
                group_index += 1
                completed_groups += 1
                if self._after_group is not None:
                    self._after_group(completed_groups)

            fresh = self._projection.snapshot()
            if _snapshot_hash(fresh) != snapshot_hash:
                raise _SourceInvalidError("Notes organization source changed during bootstrap")
            self._reconcile_stale_groups(
                service,
                dataset,
                bootstrap_id=bootstrap_id,
                snapshot=fresh,
                groups=stale_groups,
            )
            return service.store.transition_notes_organization_bootstrap(
                dataset.dataset_id,
                bootstrap_id=bootstrap_id,
                expected_state="initializing",
                state="ready",
                captured_count=expected_count,
                expected_count=expected_count,
                ready_verifier=lambda: self._ready_snapshot_matches(
                    service,
                    dataset.dataset_id,
                    snapshot_hash,
                ),
            )
        except NotesOrganizationBootstrapInterrupted:
            raise
        except SyncServerOriginBatchMaterializationError as exc:
            if exc.retryable:
                return self._pause_retryable(
                    service,
                    dataset,
                    bootstrap_id=bootstrap_id,
                    captured_count=captured_count,
                    expected_count=expected_count,
                )
            return self._fail(
                service,
                dataset,
                bootstrap_id=bootstrap_id,
                captured_count=captured_count,
                expected_count=expected_count,
                error_code=_SAFE_CAPTURE_ERROR,
            )
        except _SourceInvalidError:
            return self._fail(
                service,
                dataset,
                bootstrap_id=bootstrap_id,
                captured_count=captured_count,
                expected_count=expected_count,
                error_code=_SAFE_SOURCE_ERROR,
            )
        except Exception:  # noqa: BLE001 - every failure becomes safe durable state.
            return self._fail(
                service,
                dataset,
                bootstrap_id=bootstrap_id,
                captured_count=captured_count,
                expected_count=expected_count,
                error_code=_SAFE_CAPTURE_ERROR,
            )

    def _plan(
        self,
        snapshot: OrganizationSnapshot,
        *,
        bootstrap_id: str,
        removed_relationships: Sequence[SyncEnvelope],
        represented_resources: set[tuple[SyncDomain, str]] | None = None,
        restore_resources: set[tuple[SyncDomain, str]] | None = None,
    ) -> list[ServerOriginMutationStep]:
        ordered_resources = _parents_before_children(snapshot.resources)
        represented = represented_resources or set()
        restores = restore_resources or set()
        steps = [
            self._resource_step(
                resource,
                operation="upsert",
                restore_intent=(resource.domain, resource.sync_id) in restores,
            )
            for resource in ordered_resources
            if (resource.domain, resource.sync_id) not in represented
        ]
        steps.extend(
            ServerOriginMutationStep(
                domain=relationship.domain,
                operation="upsert",
                object_id=relationship.object_id,
                payload=dict(relationship.payload),
                routing_metadata={
                    "bootstrap_capture": True,
                    "bootstrap_id": bootstrap_id,
                },
                stable_key=f"{relationship.domain}:{relationship.object_id}",
            )
            for relationship in snapshot.relationships
        )
        steps.extend(
            ServerOriginMutationStep(
                domain=envelope.domain,
                operation="tombstone",
                object_id=envelope.object_id,
                payload=dict(envelope.payload),
                routing_metadata={
                    "bootstrap_removal": True,
                    "bootstrap_id": bootstrap_id,
                },
                stable_key=f"{envelope.domain}:{envelope.object_id}",
            )
            for envelope in removed_relationships
        )
        steps.extend(
            self._resource_step(resource, operation="tombstone")
            for resource in ordered_resources
            if resource.deleted and (resource.domain, resource.sync_id) not in represented
        )
        return steps

    @staticmethod
    def _resource_step(
        resource: OrganizationResource,
        *,
        operation: str,
        restore_intent: bool = False,
    ) -> ServerOriginMutationStep:
        if operation == "tombstone":
            payload: dict[str, object] = {}
        elif resource.domain == "notes.keyword":
            payload = {"keyword": resource.name}
        else:
            payload = {"name": resource.name, "parent_sync_id": resource.parent_sync_id}
        return ServerOriginMutationStep(
            domain=resource.domain,
            operation=operation,
            object_id=resource.sync_id,
            payload=payload,
            routing_metadata={"restore_intent": True} if restore_intent else {},
            stable_key=f"{resource.domain}:{resource.sync_id}",
        )

    def _relationship_matches_source(
        self,
        domain: SyncDomain,
        object_id: str,
        payload: Mapping[str, object],
    ) -> bool:
        return any(
            item.domain == domain
            and item.object_id == object_id
            and dict(item.payload) == dict(payload)
            for item in self._projection.snapshot().relationships
        )

    def _relationship_absent_from_source(
        self,
        domain: SyncDomain,
        object_id: str,
        payload: Mapping[str, object],
    ) -> bool:
        return not any(
            item.domain == domain
            and item.object_id == object_id
            and dict(item.payload) == dict(payload)
            for item in self._projection.snapshot().relationships
        )

    def _step_matches_source(self, envelope: SyncEnvelope) -> bool:
        return self._step_matches_snapshot(envelope, self._projection.snapshot())

    @staticmethod
    def _step_matches_snapshot(
        envelope: SyncEnvelope,
        snapshot: OrganizationSnapshot,
    ) -> bool:
        if envelope.domain not in _RESOURCE_DOMAINS:
            present = any(
                item.domain == envelope.domain
                and item.object_id == envelope.object_id
                and dict(item.payload) == dict(envelope.payload)
                for item in snapshot.relationships
            )
            return not present if envelope.operation == "tombstone" else present
        resource = next(
            (
                item
                for item in snapshot.resources
                if item.domain == envelope.domain and item.sync_id == envelope.object_id
            ),
            None,
        )
        if resource is None:
            return False
        if envelope.operation == "tombstone":
            return resource.deleted and not envelope.payload
        expected = (
            {"keyword": resource.name}
            if resource.domain == "notes.keyword"
            else {"name": resource.name, "parent_sync_id": resource.parent_sync_id}
        )
        return dict(envelope.payload) == expected

    @classmethod
    def _captured_relationship_removals(
        cls,
        service: SyncV2Service,
        dataset: SyncDataset,
        snapshot: OrganizationSnapshot,
        *,
        bootstrap_id: str,
    ) -> list[SyncEnvelope]:
        current = {
            (relationship.domain, relationship.object_id)
            for relationship in snapshot.relationships
        }
        captured: dict[tuple[SyncDomain, str], list[SyncEnvelope]] = {}
        for group in cls._bootstrap_mutation_groups(service, dataset.dataset_id):
            for envelope in group:
                if (
                    envelope.domain in _RELATIONSHIP_DOMAINS
                    and envelope.operation == "upsert"
                    and envelope.routing_metadata.get("bootstrap_capture") is True
                    and isinstance(
                        envelope.routing_metadata.get("bootstrap_id"), str
                    )
                    and bool(envelope.routing_metadata.get("bootstrap_id"))
                ):
                    captured.setdefault(
                        (envelope.domain, envelope.object_id), []
                    ).append(envelope)

        removed: list[SyncEnvelope] = []
        for key in sorted(captured):
            if key in current:
                continue
            head = service.store.get_current_head(dataset.dataset_id, *key)
            if head is None:
                continue
            candidates = captured[key]
            if (
                head.operation == "upsert"
                and head.routing_metadata.get("bootstrap_capture") is True
                and head.routing_metadata.get("source")
                == "notes-organization-bootstrap"
                and isinstance(head.routing_metadata.get("bootstrap_id"), str)
                and bool(head.routing_metadata.get("bootstrap_id"))
            ):
                template = next(
                    (
                        envelope
                        for envelope in reversed(candidates)
                        if envelope.client_envelope_id == head.client_envelope_id
                    ),
                    None,
                )
            elif (
                head.operation == "tombstone"
                and head.routing_metadata.get("bootstrap_removal") is True
                and head.routing_metadata.get("bootstrap_id") == bootstrap_id
            ):
                template = next(
                    (
                        envelope
                        for envelope in reversed(candidates)
                        if dict(envelope.payload) == dict(head.payload)
                    ),
                    None,
                )
            else:
                template = None
            if template is not None:
                removed.append(template)
        return removed

    def _ready_snapshot_matches(
        self,
        service: SyncV2Service,
        dataset_id: str,
        snapshot_hash: str,
    ) -> bool:
        fresh = self._projection.snapshot()
        return _snapshot_hash(fresh) == snapshot_hash and _heads_match_snapshot(
            service,
            dataset_id,
            fresh,
        )

    def _drain_preexisting_heads(
        self,
        service: SyncV2Service,
        dataset: SyncDataset,
    ) -> tuple[
        OrganizationSnapshot,
        set[tuple[SyncDomain, str]],
        list[list[SyncEnvelope]],
    ]:
        metadata = dataset.metadata.get("notes_organization_v1")
        bootstrap_id = metadata.get("bootstrap_id") if isinstance(metadata, Mapping) else None
        if not isinstance(bootstrap_id, str) or not bootstrap_id:
            raise SyncStoreError("notes_organization_sync_not_ready")
        for domain in NOTES_ORGANIZATION_DOMAINS:
            offset = 0
            while True:
                heads = service.store.list_current_heads(
                    dataset.dataset_id, domain, limit=200, offset=offset
                )
                if not heads:
                    break
                for envelope in heads:
                    if envelope.apply_status == "applied":
                        continue
                    if envelope.routing_metadata.get("source") == "notes-organization-bootstrap":
                        continue
                    result = service._materialize_envelope(envelope)
                    if result.status != "applied":
                        raise SyncStoreError("Notes organization pre-bootstrap projection failed")
                offset += len(heads)

        snapshot = self._projection.snapshot()
        groups = self._bootstrap_mutation_groups(service, dataset.dataset_id)
        stale_groups: list[list[SyncEnvelope]] = []
        for group in groups:
            if all(envelope.apply_status == "applied" for envelope in group):
                continue
            if not all(
                self._step_matches_snapshot(envelope, snapshot) for envelope in group
            ):
                stale_groups.append(group)
                continue
            # Repair the complete group in mutation-step order. A pending step that
            # is shadowed by an applied current head is audit-reconciled without
            # moving object state backward, even transiently.
            for envelope in group:
                if envelope.apply_status == "applied":
                    continue
                if envelope.server_cursor is None:
                    raise SyncStoreError("Stored bootstrap step has no server cursor")
                current = service.store.get_current_head(
                    dataset.dataset_id,
                    envelope.domain,
                    envelope.object_id,
                )
                if (
                    current is not None
                    and current.server_cursor is not None
                    and current.server_cursor > envelope.server_cursor
                    and current.apply_status == "applied"
                ):
                    service.store.reconcile_bootstrap_envelope_superseded(
                        envelope.server_cursor,
                        bootstrap_id=bootstrap_id,
                        superseded_by_cursor=current.server_cursor,
                    )
                    continue
                service.store.mark_bootstrap_envelope_verified(
                    envelope.server_cursor,
                    bootstrap_id=bootstrap_id,
                )
        groups = self._bootstrap_mutation_groups(service, dataset.dataset_id)
        return (
            snapshot,
            self._represented_resource_lineages(
                service,
                dataset.dataset_id,
                snapshot,
                groups,
            ),
            stale_groups,
        )

    @classmethod
    def _reconcile_stale_groups(
        cls,
        service: SyncV2Service,
        dataset: SyncDataset,
        *,
        bootstrap_id: str,
        snapshot: OrganizationSnapshot,
        groups: Sequence[Sequence[SyncEnvelope]],
    ) -> None:
        for group in groups:
            for envelope in group:
                if envelope.apply_status == "applied":
                    continue
                if envelope.server_cursor is None:
                    raise SyncStoreError("Stored bootstrap step has no server cursor")
                correction = service.store.get_current_head(
                    dataset.dataset_id,
                    envelope.domain,
                    envelope.object_id,
                )
                if (
                    correction is None
                    or correction.server_cursor is None
                    or correction.server_cursor <= envelope.server_cursor
                    or correction.apply_status != "applied"
                    or not cls._step_matches_snapshot(correction, snapshot)
                ):
                    raise _SourceInvalidError(
                        "Bootstrap correction is not durably source-verified"
                    )
                service.store.reconcile_bootstrap_envelope_superseded(
                    envelope.server_cursor,
                    bootstrap_id=bootstrap_id,
                    superseded_by_cursor=correction.server_cursor,
                )

    @staticmethod
    def _resource_restore_keys(
        service: SyncV2Service,
        dataset_id: str,
        snapshot: OrganizationSnapshot,
        represented_resources: set[tuple[SyncDomain, str]],
    ) -> set[tuple[SyncDomain, str]]:
        restores: set[tuple[SyncDomain, str]] = set()
        for resource in snapshot.resources:
            key = (resource.domain, resource.sync_id)
            if not resource.deleted or key in represented_resources:
                continue
            head = service.store.get_current_head(dataset_id, *key)
            if head is not None and head.operation == "tombstone":
                restores.add(key)
        return restores

    @classmethod
    def _represented_resource_lineages(
        cls,
        service: SyncV2Service,
        dataset_id: str,
        snapshot: OrganizationSnapshot,
        groups: Sequence[Sequence[SyncEnvelope]],
    ) -> set[tuple[SyncDomain, str]]:
        history = [envelope for group in groups for envelope in group]
        represented: set[tuple[SyncDomain, str]] = set()
        for resource in snapshot.resources:
            key = (resource.domain, resource.sync_id)
            head = service.store.get_current_head(dataset_id, *key)
            if (
                head is None
                or head.apply_status != "applied"
                or not cls._step_matches_snapshot(head, snapshot)
                or head.operation != ("tombstone" if resource.deleted else "upsert")
            ):
                continue
            if not resource.deleted:
                represented.add(key)
                continue
            if any(
                envelope.domain == resource.domain
                and envelope.object_id == resource.sync_id
                and envelope.operation == "upsert"
                and envelope.apply_status == "applied"
                and cls._step_matches_snapshot(envelope, snapshot)
                for envelope in history
            ):
                represented.add(key)
        return represented

    @staticmethod
    def _bootstrap_mutation_groups(
        service: SyncV2Service,
        dataset_id: str,
    ) -> list[list[SyncEnvelope]]:
        groups: dict[str, list[SyncEnvelope]] = {}
        cursor = 0
        while True:
            page = service.store.list_envelopes_after(
                dataset_id,
                cursor,
                limit=_BOOTSTRAP_HISTORY_PAGE_SIZE,
                domains=NOTES_ORGANIZATION_DOMAINS,
                status="accepted",
            )
            if not page:
                break
            next_cursor = max(envelope.server_cursor or 0 for envelope in page)
            if next_cursor <= cursor:
                raise SyncStoreError("Stored bootstrap history cursor did not advance")
            cursor = next_cursor
            for envelope in page:
                if (
                    envelope.routing_metadata.get("source")
                    != "notes-organization-bootstrap"
                ):
                    continue
                group_id = envelope.mutation_group_id
                if not isinstance(group_id, str) or not group_id:
                    raise SyncStoreError("Stored bootstrap step has no mutation group")
                groups.setdefault(group_id, []).append(envelope)

        ordered = sorted(
            groups.values(),
            key=lambda group: min(envelope.server_cursor or 0 for envelope in group),
        )
        for group in ordered:
            expected_count = group[0].mutation_step_count
            if (
                expected_count is None
                or len(group) != expected_count
                or [envelope.mutation_step for envelope in group]
                != list(range(expected_count))
            ):
                raise SyncStoreError("Stored bootstrap mutation group is incomplete")
        return ordered

    @staticmethod
    def _pause_retryable(
        service: SyncV2Service,
        dataset: SyncDataset,
        *,
        bootstrap_id: str,
        captured_count: int,
        expected_count: int,
    ) -> SyncDataset:
        return service.store.transition_notes_organization_bootstrap(
            dataset.dataset_id,
            bootstrap_id=bootstrap_id,
            expected_state="initializing",
            state="initializing",
            captured_count=min(captured_count, expected_count),
            expected_count=expected_count,
            error_code=_SAFE_CAPTURE_ERROR,
        )

    @staticmethod
    def _fail(
        service: SyncV2Service,
        dataset: SyncDataset,
        *,
        bootstrap_id: str,
        captured_count: int,
        expected_count: int,
        error_code: str,
    ) -> SyncDataset:
        return service.store.transition_notes_organization_bootstrap(
            dataset.dataset_id,
            bootstrap_id=bootstrap_id,
            expected_state="initializing",
            state="failed",
            captured_count=min(captured_count, expected_count),
            expected_count=expected_count,
            error_code=error_code,
        )


def _without_manifest_steps(
    planned: Sequence[ServerOriginMutationStep],
    stored: Sequence[ServerOriginMutationStep],
) -> list[ServerOriginMutationStep]:
    stored_counts = Counter(_step_semantic_key(step) for step in stored)
    remaining: list[ServerOriginMutationStep] = []
    for step in planned:
        key = _step_semantic_key(step)
        if stored_counts[key]:
            stored_counts[key] -= 1
        else:
            remaining.append(step)
    return remaining


def _step_semantic_key(step: ServerOriginMutationStep) -> str:
    return json.dumps(
        {
            "domain": step.domain,
            "operation": step.operation,
            "object_id": step.object_id,
            "payload": step.payload,
            "parent_id": step.parent_id,
            "stable_key": step.stable_key,
        },
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _heads_match_snapshot(
    service: SyncV2Service,
    dataset_id: str,
    snapshot: OrganizationSnapshot,
) -> bool:
    expected: dict[tuple[SyncDomain, str], tuple[str, dict[str, object]]] = {}
    for resource in snapshot.resources:
        operation = "tombstone" if resource.deleted else "upsert"
        if operation == "tombstone":
            payload: dict[str, object] = {}
        elif resource.domain == "notes.keyword":
            payload = {"keyword": resource.name}
        else:
            payload = {
                "name": resource.name,
                "parent_sync_id": resource.parent_sync_id,
            }
        expected[(resource.domain, resource.sync_id)] = (operation, payload)
    expected.update(
        {
            (relationship.domain, relationship.object_id): (
                "upsert",
                dict(relationship.payload),
            )
            for relationship in snapshot.relationships
        }
    )

    seen: set[tuple[SyncDomain, str]] = set()
    for domain in NOTES_ORGANIZATION_DOMAINS:
        offset = 0
        while True:
            heads = service.store.list_current_heads(
                dataset_id,
                domain,
                limit=200,
                offset=offset,
            )
            if not heads:
                break
            for envelope in heads:
                key = (envelope.domain, envelope.object_id)
                wanted = expected.get(key)
                if wanted is None:
                    if envelope.operation != "tombstone":
                        return False
                    continue
                if (
                    envelope.apply_status != "applied"
                    or envelope.operation != wanted[0]
                    or dict(envelope.payload) != wanted[1]
                ):
                    return False
                seen.add(key)
            offset += len(heads)
    return seen == set(expected)


def _parents_before_children(
    resources: Sequence[OrganizationResource],
) -> list[OrganizationResource]:
    result: list[OrganizationResource] = []
    for domain in _RESOURCE_DOMAINS:
        domain_resources = {item.sync_id: item for item in resources if item.domain == domain}
        result.extend(_parent_order(domain_resources))
    return result


def _parent_order(
    resources: Mapping[str, OrganizationResource],
) -> list[OrganizationResource]:
    ordered: list[OrganizationResource] = []
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(resource: OrganizationResource) -> None:
        if resource.sync_id in visited:
            return
        if resource.sync_id in visiting:
            raise _SourceInvalidError("Notes organization hierarchy contains a cycle")
        visiting.add(resource.sync_id)
        if resource.parent_sync_id is not None:
            parent = resources.get(resource.parent_sync_id)
            if parent is None:
                raise _SourceInvalidError("Notes organization hierarchy parent is missing")
            visit(parent)
        visiting.remove(resource.sync_id)
        visited.add(resource.sync_id)
        ordered.append(resource)

    for resource in sorted(resources.values(), key=lambda item: item.sync_id):
        visit(resource)
    return ordered


def _snapshot_hash(snapshot: OrganizationSnapshot) -> str:
    encoded = json.dumps(
        {
            "resources": [
                {
                    "domain": item.domain,
                    "sync_id": item.sync_id,
                    "name": item.name,
                    "parent_sync_id": item.parent_sync_id,
                    "deleted": item.deleted,
                }
                for item in snapshot.resources
            ],
            "relationships": [
                {
                    "domain": item.domain,
                    "object_id": item.object_id,
                    "payload": dict(item.payload),
                }
                for item in snapshot.relationships
            ],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "NotesOrganizationBootstrapInterrupted",
    "NotesOrganizationBootstrapper",
    "SyncDatasetBootstrapper",
]
