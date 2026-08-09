from __future__ import annotations

"""Durable bootstrap of existing ChaCha Notes organization state into Sync v2."""

import hashlib
import json
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
            self._drain_preexisting_heads(service, dataset)
            snapshot = self._projection.snapshot()
            removed_relationships = self._captured_relationship_removals(
                service,
                dataset,
                snapshot,
                bootstrap_id=bootstrap_id,
            )
            steps = self._plan(
                snapshot,
                bootstrap_id=bootstrap_id,
                removed_relationships=removed_relationships,
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
            for completed_groups, offset in enumerate(
                range(0, len(steps), self._batch_size), start=1
            ):
                batch = steps[offset : offset + self._batch_size]
                capture_server_origin_mutation_batch(
                    service=service,
                    user_id=user_id,
                    steps=batch,
                    source="notes-organization-bootstrap",
                    idempotency_key=(
                        f"{bootstrap_id}:{snapshot_hash}:{offset // self._batch_size}"
                    ),
                    trusted_notes_organization_bootstrap_id=bootstrap_id,
                    bootstrap_relationship_verifier=self._relationship_matches_source,
                    bootstrap_relationship_absence_verifier=(
                        self._relationship_absent_from_source
                    ),
                    bootstrap_step_verifier=self._step_matches_source,
                )
                captured_count = min(expected_count, offset + len(batch))
                dataset = service.store.transition_notes_organization_bootstrap(
                    dataset.dataset_id,
                    bootstrap_id=bootstrap_id,
                    expected_state="initializing",
                    state="initializing",
                    captured_count=captured_count,
                    expected_count=expected_count,
                )
                if self._after_group is not None:
                    self._after_group(completed_groups)

            fresh = self._projection.snapshot()
            if _snapshot_hash(fresh) != snapshot_hash:
                raise _SourceInvalidError("Notes organization source changed during bootstrap")
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
    ) -> list[ServerOriginMutationStep]:
        ordered_resources = _parents_before_children(snapshot.resources)
        steps = [self._resource_step(resource, operation="upsert") for resource in ordered_resources]
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
            if resource.deleted
        )
        return steps

    @staticmethod
    def _resource_step(
        resource: OrganizationResource,
        *,
        operation: str,
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
        snapshot = self._projection.snapshot()
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

    @staticmethod
    def _captured_relationship_removals(
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
        removed: list[SyncEnvelope] = []
        for domain in _RELATIONSHIP_DOMAINS:
            offset = 0
            while True:
                heads = service.store.list_current_heads(
                    dataset.dataset_id,
                    domain,
                    limit=200,
                    offset=offset,
                )
                if not heads:
                    break
                removed.extend(
                    envelope
                    for envelope in heads
                    if envelope.operation == "upsert"
                    and envelope.routing_metadata.get("source")
                    == "notes-organization-bootstrap"
                    and envelope.routing_metadata.get("bootstrap_capture") is True
                    and envelope.routing_metadata.get("bootstrap_id") == bootstrap_id
                    and (envelope.domain, envelope.object_id) not in current
                )
                offset += len(heads)
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
    ) -> None:
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
                        if envelope.server_cursor is None:
                            raise SyncStoreError(
                                "Stored bootstrap step has no server cursor"
                            )
                        # The step was structurally source-attested before its atomic
                        # append. Mark that historical capture verified; the fresh plan
                        # below exact-verifies and supersedes it when source has changed.
                        service.store.mark_bootstrap_envelope_verified(
                            envelope.server_cursor,
                            bootstrap_id=bootstrap_id,
                        )
                        continue
                    result = service._materialize_envelope(envelope)
                    if result.status != "applied":
                        raise SyncStoreError("Notes organization pre-bootstrap projection failed")
                offset += len(heads)

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
