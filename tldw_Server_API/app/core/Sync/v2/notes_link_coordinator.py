from __future__ import annotations

"""Owner-bound planning and capture for explicit Notes links."""

import hashlib
import json
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, replace

from tldw_Server_API.app.core.DB_Management.chacha.note_link_store import (
    NotesLink,
    NotesLinkStore,
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    ConflictError,
    InputError,
)

from .errors import SyncStoreError
from .materializers.guarded_product_mutation import GuardedProductMutation
from .models import SyncDataset, normalize_sync_timestamp
from .notes_link import validate_notes_link_properties
from .server_origin import (
    SERVER_ORIGIN_DEVICE_ID,
    get_active_server_origin_sync_service_for_user,
)
from .server_origin_batch import (
    ServerOriginMutationStep,
    SyncServerOriginBatchAppendError,
    SyncServerOriginBatchIdempotencyConflictError,
    SyncServerOriginBatchMaterializationError,
    capture_server_origin_mutation_batch,
    load_server_origin_mutation_batch_manifest,
    server_origin_mutation_batch_group_id,
)
from .service import SyncV2Service

_REQUEST_FINGERPRINT_KEY = "notes_link_request_fingerprint"
_SOURCE_CREATE = "notes.graph.link.create"
_SOURCE_UPDATE = "notes.graph.link.update"
_SOURCE_TOMBSTONE = "notes.graph.link.tombstone"
_SOURCE_RESTORE = "notes.graph.link.restore"


class NotesLinkDatasetConflictError(SyncStoreError):
    """A supplied dataset is not the canonical product authority."""

    error_code = "notes_link_dataset_conflict"

    def __init__(self) -> None:
        super().__init__(self.error_code)


class NotesLinkSyncInactiveDatasetError(SyncStoreError):
    """A dataset was supplied while Sync is inactive for the owner."""

    error_code = "notes_link_sync_inactive_dataset"

    def __init__(self) -> None:
        super().__init__(self.error_code)


class NotesLinkNotReadyError(SyncStoreError):
    """The canonical notes.link dataset exists but is not write-ready."""

    error_code = "notes_link_sync_not_ready"

    def __init__(self, state: str) -> None:
        super().__init__(self.error_code)
        self.state = state


class NotesLinkResourceNotFoundError(InputError):
    """An owner-scoped explicit link was not found."""

    error_code = "notes_link_not_found"

    def __init__(self) -> None:
        super().__init__(self.error_code)


class NotesLinkVersionConflictError(ConflictError):
    """The product link changed since the caller read it."""

    error_code = "notes_link_version_conflict"

    def __init__(self, edge_id: str) -> None:
        super().__init__(self.error_code, entity="note_edges", entity_id=edge_id)


class NotesLinkPreflightError(SyncStoreError):
    """A canonical link plan conflicted before durable append."""

    error_code = "notes_link_preflight_failed"

    def __init__(self) -> None:
        super().__init__(self.error_code)


@dataclass(slots=True)
class NotesLinkCoordinator:
    """Build and capture canonical link plans for one active Notes dataset."""

    service: SyncV2Service
    note_db: CharactersRAGDB
    user_id: str
    dataset: SyncDataset | None = None

    def __post_init__(self) -> None:
        if self.dataset is None:
            self.dataset = _active_default_personal_dataset(self.service, self.user_id)
        if self.dataset.owner_user_id != self.user_id:
            raise NotesLinkDatasetConflictError()

    @property
    def _links(self) -> NotesLinkStore:
        return self.note_db.notes_link_store

    @property
    def dataset_id(self) -> str:
        dataset = self.dataset
        if dataset is None:
            raise NotesLinkDatasetConflictError()
        return dataset.dataset_id

    def require_ready(self) -> None:
        """Require the separate notes.link enrollment and verified ready state."""

        dataset = self.dataset
        if dataset is None:
            raise NotesLinkDatasetConflictError()
        if not {"notes.note", "notes.link"}.issubset(dataset.domains):
            raise NotesLinkNotReadyError("absent")
        metadata = dataset.metadata.get("notes_link_v1")
        state = metadata.get("state") if isinstance(metadata, Mapping) else None
        if state != "ready":
            raise NotesLinkNotReadyError(state if isinstance(state, str) else "absent")

    def get(self, edge_id: str) -> NotesLink:
        """Load one owner-scoped link, including tombstones."""

        link = self._links.get(edge_id)
        if link is None:
            raise NotesLinkResourceNotFoundError()
        return link

    def create(
        self,
        *,
        source_note_id: str,
        target_note_id: str,
        directed: bool,
        weight: float,
        label: str | None,
        properties: Mapping[str, object],
        idempotency_key: str | None,
        guarded_mutation: GuardedProductMutation | None = None,
    ) -> NotesLink:
        """Create one explicit link through canonical server-origin capture."""

        self.require_ready()
        source_note_id, target_note_id = _canonical_endpoints(
            source_note_id,
            target_note_id,
            directed=directed,
        )
        canonical_properties = validate_notes_link_properties(dict(properties))
        normalized_key = _capture_key(idempotency_key)
        edge_id = _edge_id(self.dataset_id, _SOURCE_CREATE, normalized_key)
        fingerprint = _request_fingerprint(
            "create",
            {
                "source_note_id": source_note_id,
                "target_note_id": target_note_id,
                "directed": directed,
                "weight": weight,
                "label": label,
                "properties": canonical_properties,
            },
        )
        replay = self._replay(
            source=_SOURCE_CREATE,
            idempotency_key=normalized_key,
            request_fingerprint=fingerprint,
            edge_id=edge_id,
            guarded_mutation=guarded_mutation,
        )
        if replay is not None:
            return replay

        self._links.validate_public_endpoints(source_note_id, target_note_id)
        now = _clock(self.service)
        payload = {
            "source_note_id": source_note_id,
            "target_note_id": target_note_id,
            "type": "manual",
            "directed": directed,
            "weight": weight,
            "label": label,
            "properties": canonical_properties,
            "created_at": now,
            "last_modified": now,
            "created_by": SERVER_ORIGIN_DEVICE_ID,
        }
        return self._capture(
            source=_SOURCE_CREATE,
            idempotency_key=normalized_key,
            request_fingerprint=fingerprint,
            step=ServerOriginMutationStep(
                domain="notes.link",
                operation="upsert",
                object_id=edge_id,
                payload=payload,
                stable_key=f"notes-link:{edge_id}",
                created_at_client=now,
            ),
            guarded_mutation=guarded_mutation,
        )

    def update(
        self,
        *,
        edge_id: str,
        expected_version: int,
        weight: float,
        label: str | None,
        properties: Mapping[str, object],
        idempotency_key: str | None,
    ) -> NotesLink:
        """Update mutable presentation fields using an optimistic version."""

        return self._mutate_existing(
            operation="update",
            source=_SOURCE_UPDATE,
            edge_id=edge_id,
            expected_version=expected_version,
            idempotency_key=idempotency_key,
            weight=weight,
            label=label,
            properties=properties,
        )

    def tombstone(
        self,
        *,
        edge_id: str,
        expected_version: int,
        reason: str | None,
        idempotency_key: str | None,
    ) -> NotesLink:
        """Soft-delete one explicit link through canonical capture."""

        return self._mutate_existing(
            operation="tombstone",
            source=_SOURCE_TOMBSTONE,
            edge_id=edge_id,
            expected_version=expected_version,
            idempotency_key=idempotency_key,
            reason=reason,
        )

    def restore(
        self,
        *,
        edge_id: str,
        expected_version: int,
        idempotency_key: str | None,
    ) -> NotesLink:
        """Restore one tombstoned explicit link through canonical capture."""

        return self._mutate_existing(
            operation="restore",
            source=_SOURCE_RESTORE,
            edge_id=edge_id,
            expected_version=expected_version,
            idempotency_key=idempotency_key,
        )

    def _mutate_existing(
        self,
        *,
        operation: str,
        source: str,
        edge_id: str,
        expected_version: int,
        idempotency_key: str | None,
        weight: float | None = None,
        label: str | None = None,
        properties: Mapping[str, object] | None = None,
        reason: str | None = None,
    ) -> NotesLink:
        self.require_ready()
        normalized_key = _capture_key(idempotency_key)
        fingerprint_fields: dict[str, object] = {
            "edge_id": edge_id,
            "expected_version": expected_version,
        }
        if operation == "update":
            fingerprint_fields.update(
                weight=weight,
                label=label,
                properties=dict(properties or {}),
            )
        elif operation == "tombstone":
            fingerprint_fields["reason"] = reason
        fingerprint = _request_fingerprint(operation, fingerprint_fields)
        replay = self._replay(
            source=source,
            idempotency_key=normalized_key,
            request_fingerprint=fingerprint,
            edge_id=edge_id,
        )
        if replay is not None:
            return replay

        current = self.get(edge_id)
        if current.version != expected_version:
            raise NotesLinkVersionConflictError(edge_id)
        if operation == "restore" and not current.deleted:
            raise NotesLinkVersionConflictError(edge_id)
        if operation != "restore" and current.deleted:
            raise NotesLinkVersionConflictError(edge_id)
        now = _clock(self.service)
        payload: dict[str, object] = {
            "source_note_id": current.source_note_id,
            "target_note_id": current.target_note_id,
            "type": current.type,
            "directed": current.directed,
            "weight": current.weight if weight is None else weight,
            "label": current.label if operation != "update" else label,
            "properties": (
                dict(current.properties)
                if operation != "update"
                else validate_notes_link_properties(dict(properties or {}))
            ),
            "created_at": current.created_at,
            "last_modified": now,
            "created_by": current.created_by,
        }
        routing: dict[str, object] = {}
        sync_operation = "upsert"
        if operation == "tombstone":
            sync_operation = "tombstone"
            payload.update(deleted_at=now, reason=reason)
        elif operation == "restore":
            routing["restore_intent"] = True
        return self._capture(
            source=source,
            idempotency_key=normalized_key,
            request_fingerprint=fingerprint,
            step=ServerOriginMutationStep(
                domain="notes.link",
                operation=sync_operation,
                object_id=edge_id,
                payload=payload,
                routing_metadata=routing,
                stable_key=f"notes-link:{edge_id}:{operation}:{normalized_key}",
                created_at_client=now,
            ),
        )

    def _replay(
        self,
        *,
        source: str,
        idempotency_key: str,
        request_fingerprint: str,
        edge_id: str,
        guarded_mutation: GuardedProductMutation | None = None,
    ) -> NotesLink | None:
        manifest = load_server_origin_mutation_batch_manifest(
            service=self.service,
            dataset_id=self.dataset_id,
            source=source,
            idempotency_key=idempotency_key,
        )
        if manifest is None:
            return None
        if (
            len(manifest) != 1
            or manifest[0].domain != "notes.link"
            or manifest[0].object_id != edge_id
            or manifest[0].routing_metadata.get(_REQUEST_FINGERPRINT_KEY) != request_fingerprint
        ):
            raise SyncServerOriginBatchIdempotencyConflictError(
                server_origin_mutation_batch_group_id(
                    dataset_id=self.dataset_id,
                    source=source,
                    idempotency_key=idempotency_key,
                )
            )
        capture_server_origin_mutation_batch(
            service=self.service,
            user_id=self.user_id,
            steps=manifest,
            source=source,
            idempotency_key=idempotency_key,
            guarded_mutation=guarded_mutation,
        )
        return self.get(edge_id)

    def _capture(
        self,
        *,
        source: str,
        idempotency_key: str,
        request_fingerprint: str,
        step: ServerOriginMutationStep,
        guarded_mutation: GuardedProductMutation | None = None,
    ) -> NotesLink:
        bound = replace(
            step,
            routing_metadata={
                **dict(step.routing_metadata),
                _REQUEST_FINGERPRINT_KEY: request_fingerprint,
            },
        )
        try:
            capture_server_origin_mutation_batch(
                service=self.service,
                user_id=self.user_id,
                steps=(bound,),
                source=source,
                idempotency_key=idempotency_key,
                guarded_mutation=guarded_mutation,
            )
        except (
            SyncServerOriginBatchAppendError,
            SyncServerOriginBatchIdempotencyConflictError,
            SyncServerOriginBatchMaterializationError,
        ):
            raise
        except SyncStoreError as exc:
            raise NotesLinkPreflightError() from exc
        return self.get(step.object_id)


def resolve_notes_link_coordinator(
    *,
    user_id: str,
    note_db: CharactersRAGDB,
    dataset_id: str | None,
) -> NotesLinkCoordinator | None:
    """Resolve active canonical authority or preserve inactive legacy behavior."""

    authority = resolve_notes_link_dataset_authority(
        user_id=user_id,
        dataset_id=dataset_id,
    )
    if authority is None:
        return None
    service, dataset = authority
    coordinator = NotesLinkCoordinator(service, note_db, user_id, dataset)
    coordinator.require_ready()
    return coordinator


def resolve_notes_link_dataset_authority(
    *,
    user_id: str,
    dataset_id: str | None,
) -> tuple[SyncV2Service, SyncDataset] | None:
    """Authorize the canonical Notes dataset before any graph cache lookup."""

    service = get_active_server_origin_sync_service_for_user(user_id)
    normalized_dataset_id = None
    if dataset_id is not None:
        normalized_dataset_id = str(dataset_id).strip()
        if not normalized_dataset_id:
            raise NotesLinkDatasetConflictError()
    if service is None:
        if normalized_dataset_id is not None:
            raise NotesLinkSyncInactiveDatasetError()
        return None
    dataset = _active_default_personal_dataset(service, user_id)
    if normalized_dataset_id is not None and normalized_dataset_id != dataset.dataset_id:
        raise NotesLinkDatasetConflictError()
    return service, dataset


def _active_default_personal_dataset(
    service: SyncV2Service,
    user_id: str,
) -> SyncDataset:
    matches = [
        dataset
        for dataset in service.store.list_datasets_for_user(user_id)
        if dataset.scope_type == "personal"
        and dataset.metadata.get("default_personal") is True
        and dataset.metadata.get("client_family") == "chatbook"
        and dataset.archived_at is None
    ]
    if len(matches) != 1:
        raise NotesLinkDatasetConflictError()
    return matches[0]


def _canonical_endpoints(source: str, target: str, *, directed: bool) -> tuple[str, str]:
    if not directed and source > target:
        return target, source
    return source, target


def _capture_key(value: str | None) -> str:
    if value is None:
        return str(uuid.uuid4())
    normalized = str(value).strip()
    if not normalized:
        raise InputError("idempotency_key must not be blank")
    return normalized


def _edge_id(dataset_id: str, source: str, idempotency_key: str) -> str:
    digest = hashlib.sha256(f"{dataset_id}:{source}:{idempotency_key}".encode()).digest()[:16]
    return str(uuid.UUID(bytes=digest, version=4))


def _request_fingerprint(operation: str, fields: Mapping[str, object]) -> str:
    encoded = json.dumps(
        {"operation": operation, "fields": dict(fields)},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _clock(service: SyncV2Service) -> str:
    value = normalize_sync_timestamp(service.clock())
    if value is None:
        raise SyncStoreError("Sync clock did not produce a timestamp")
    return value


__all__ = [
    "NotesLinkCoordinator",
    "NotesLinkDatasetConflictError",
    "NotesLinkNotReadyError",
    "NotesLinkPreflightError",
    "NotesLinkResourceNotFoundError",
    "NotesLinkSyncInactiveDatasetError",
    "NotesLinkVersionConflictError",
    "resolve_notes_link_coordinator",
]
