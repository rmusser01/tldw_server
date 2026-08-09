"""Portable planning and capture for server-origin Notes organization writes."""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from tldw_Server_API.app.core.DB_Management.chacha.organization_sync_store import (
    NotesOrganizationSyncStore,
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    ConflictError,
    InputError,
)

from .errors import SyncStoreError
from .models import NOTES_ORGANIZATION_DOMAINS, SyncDataset, SyncDomain
from .notes_organization import new_organization_sync_id, organization_link_id
from .server_origin import SyncServerOriginMutationNotSupportedError
from .server_origin_batch import (
    ServerOriginBatchResult,
    ServerOriginMutationStep,
    SyncServerOriginBatchAppendError,
    SyncServerOriginBatchIdempotencyConflictError,
    SyncServerOriginBatchMaterializationError,
    capture_server_origin_mutation_batch,
    load_server_origin_mutation_batch_manifest,
    server_origin_mutation_batch_group_id,
)
from .service import SyncV2Service

_RESOURCE_TABLES: dict[SyncDomain, tuple[str, str]] = {
    "notes.keyword": ("keywords", "keyword"),
    "notes.keyword_collection": ("keyword_collections", "name"),
    "notes.folder": ("note_folders", "name"),
}
_REQUEST_FINGERPRINT_KEY = "notes_organization_request_fingerprint"


class NotesOrganizationDomainsIncompleteError(SyncStoreError):
    """An active personal dataset lacks part of the atomic domain group."""

    error_code = "notes_organization_sync_domains_incomplete"

    def __init__(self, missing_domains: Sequence[str]) -> None:
        super().__init__(self.error_code)
        self.missing_domains = tuple(sorted(missing_domains))


class NotesOrganizationNotReadyError(SyncStoreError):
    """The complete organization group exists but is not write-ready."""

    error_code = "notes_organization_sync_not_ready"

    def __init__(self, *, state: str, repair_error_code: str | None = None) -> None:
        super().__init__(self.error_code)
        self.state = state
        self.repair_error_code = repair_error_code


class NotesOrganizationPreflightError(SyncStoreError):
    """A canonical organization plan was rejected before durable append."""

    error_code = "notes_organization_sync_preflight_failed"

    def __init__(self) -> None:
        super().__init__(self.error_code)


class NotesOrganizationResourceNotFoundError(InputError):
    """An owner-scoped local organization resource does not exist or is deleted."""

    error_code = "notes_organization_resource_not_found"

    def __init__(self) -> None:
        super().__init__(self.error_code)


class NotesOrganizationVersionConflictError(ConflictError):
    """An organization resource failed its optimistic-version precondition."""

    error_code = "notes_organization_version_conflict"

    def __init__(self) -> None:
        super().__init__(self.error_code)


@dataclass(frozen=True)
class PlannedNotesMutation:
    """A read-only canonical plan and its post-materialization result loader."""

    steps: tuple[ServerOriginMutationStep, ...]
    load_result: Callable[[], object]


@dataclass(slots=True)
class NotesOrganizationCoordinator:
    """Plan and capture organization mutations for one authenticated user."""

    service: SyncV2Service
    note_db: CharactersRAGDB
    user_id: str

    def active_dataset(self) -> SyncDataset:
        """Return the active default personal dataset or fail closed."""

        for dataset in self.service.store.list_datasets_for_user(self.user_id):
            if (
                dataset.scope_type == "personal"
                and dataset.metadata.get("default_personal") is True
                and dataset.metadata.get("client_family") == "chatbook"
            ):
                return dataset
        raise SyncStoreError("Sync default personal dataset was not found")

    def require_ready(self) -> SyncDataset:
        """Require the complete six-domain group and verified ready state."""

        dataset = self.active_dataset()
        missing = sorted(set(NOTES_ORGANIZATION_DOMAINS).difference(dataset.domains))
        if missing:
            raise NotesOrganizationDomainsIncompleteError(missing)
        metadata = dataset.metadata.get("notes_organization_v1")
        state = metadata.get("state") if isinstance(metadata, Mapping) else None
        if state != "ready":
            repair_error = metadata.get("error_code") if isinstance(metadata, Mapping) else None
            raise NotesOrganizationNotReadyError(
                state=state if isinstance(state, str) else "absent",
                repair_error_code=(repair_error if isinstance(repair_error, str) else None),
            )
        return dataset

    def capture(
        self,
        *,
        steps: Sequence[ServerOriginMutationStep],
        source: str,
        idempotency_key: str,
    ) -> ServerOriginBatchResult:
        """Capture one planned Notes mutation through durable Sync authority."""

        self.require_ready()
        try:
            return capture_server_origin_mutation_batch(
                service=self.service,
                user_id=self.user_id,
                steps=tuple(steps),
                source=source,
                idempotency_key=idempotency_key,
            )
        except (
            NotesOrganizationDomainsIncompleteError,
            NotesOrganizationNotReadyError,
            SyncServerOriginBatchAppendError,
            SyncServerOriginBatchIdempotencyConflictError,
            SyncServerOriginBatchMaterializationError,
            SyncServerOriginMutationNotSupportedError,
        ):
            raise
        except SyncStoreError as exc:
            raise NotesOrganizationPreflightError() from exc

    @staticmethod
    def request_fingerprint(operation: str, fields: Mapping[str, object]) -> str:
        """Hash immutable normalized request identity without retaining raw fields."""

        encoded = json.dumps(
            {"operation": operation, "fields": dict(fields)},
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def bind_request(
        plan: PlannedNotesMutation,
        request_fingerprint: str,
    ) -> PlannedNotesMutation:
        """Bind a privacy-safe request fingerprint to every plan step."""

        return PlannedNotesMutation(
            steps=tuple(
                replace(
                    step,
                    routing_metadata={
                        **dict(step.routing_metadata),
                        _REQUEST_FINGERPRINT_KEY: request_fingerprint,
                    },
                )
                for step in plan.steps
            ),
            load_result=plan.load_result,
        )

    def replay_request_plan(
        self,
        *,
        source: str,
        idempotency_key: str | None,
        request_fingerprint: str,
        result_domain: SyncDomain | None,
        relationship_result: bool = False,
    ) -> PlannedNotesMutation | None:
        """Return an exact durable manifest before mutable projection checks."""

        normalized_key = str(idempotency_key or "").strip()
        if not normalized_key:
            return None
        dataset = self.require_ready()
        manifest = load_server_origin_mutation_batch_manifest(
            service=self.service,
            dataset_id=dataset.dataset_id,
            source=source,
            idempotency_key=normalized_key,
        )
        if manifest is None:
            return None
        if any(
            step.routing_metadata.get(_REQUEST_FINGERPRINT_KEY)
            != request_fingerprint
            for step in manifest
        ):
            raise SyncServerOriginBatchIdempotencyConflictError(
                server_origin_mutation_batch_group_id(
                    dataset_id=dataset.dataset_id,
                    source=source,
                    idempotency_key=normalized_key,
                )
            )
        if result_domain is None:
            def loader() -> object:
                return None
        else:
            result_step = next(
                (step for step in reversed(manifest) if step.domain == result_domain),
                None,
            )
            if result_step is None:
                raise SyncServerOriginBatchIdempotencyConflictError(
                    server_origin_mutation_batch_group_id(
                        dataset_id=dataset.dataset_id,
                        source=source,
                        idempotency_key=normalized_key,
                    )
                )
            if relationship_result:
                def loader() -> object:
                    return self._relationship_present(
                        result_domain, result_step.object_id
                    )
            else:
                def loader() -> object:
                    return self._load_resource_row(
                        result_domain, result_step.object_id
                    )
        return PlannedNotesMutation(steps=manifest, load_result=loader)

    def plan_keyword_create(
        self,
        keyword: str,
        *,
        idempotency_key: str | None = None,
    ) -> PlannedNotesMutation:
        """Plan creation without mutating the product database."""

        normalized = str(keyword or "").strip()
        if not normalized:
            raise InputError("Keyword text cannot be empty")
        object_id = self._create_sync_id("notes.keyword", idempotency_key)
        step = ServerOriginMutationStep(
            domain="notes.keyword",
            operation="upsert",
            object_id=object_id,
            payload={"keyword": normalized},
        )
        return PlannedNotesMutation(
            steps=(step,),
            load_result=lambda: self._load_resource_row("notes.keyword", object_id),
        )

    def plan_keyword_rename(
        self,
        keyword_id: int,
        keyword: str,
        *,
        expected_version: int | None = None,
    ) -> PlannedNotesMutation:
        """Plan a versioned keyword rename by stable identity."""

        row = self._resource_row("notes.keyword", keyword_id)
        self._require_version(row, expected_version, "Keyword")
        normalized = str(keyword or "").strip()
        if not normalized:
            raise InputError("Keyword text cannot be empty")
        object_id = str(row["sync_id"])
        return PlannedNotesMutation(
            steps=(
                ServerOriginMutationStep(
                    domain="notes.keyword",
                    operation="upsert",
                    object_id=object_id,
                    payload={"keyword": normalized},
                ),
            ),
            load_result=lambda: self._load_resource_row("notes.keyword", object_id),
        )

    def plan_resource_delete(
        self,
        domain: SyncDomain,
        local_id: int,
        *,
        expected_version: int | None = None,
    ) -> PlannedNotesMutation:
        """Plan a resource tombstone while retaining its stable identity."""

        row = self._resource_row(domain, local_id)
        self._require_version(row, expected_version, "Organization resource")
        object_id = str(row["sync_id"])
        return PlannedNotesMutation(
            steps=(
                ServerOriginMutationStep(
                    domain=domain,
                    operation="tombstone",
                    object_id=object_id,
                    payload={},
                ),
            ),
            load_result=lambda: None,
        )

    def plan_collection_change(
        self,
        collection_id: int | None,
        name: str,
        parent_id: int | None,
        *,
        idempotency_key: str | None = None,
        expected_version: int | None = None,
    ) -> PlannedNotesMutation:
        """Plan a collection create, rename, or hierarchy change."""

        normalized = str(name or "").strip()
        if not normalized:
            raise InputError("Collection name cannot be empty")
        parent_sync_id = None
        if parent_id is not None:
            parent_sync_id = str(
                self._resource_row("notes.keyword_collection", parent_id)["sync_id"]
            )
        if collection_id is None:
            object_id = self._create_sync_id(
                "notes.keyword_collection", idempotency_key
            )
        else:
            row = self._resource_row("notes.keyword_collection", collection_id)
            self._require_version(row, expected_version, "Collection")
            object_id = str(row["sync_id"])
        return PlannedNotesMutation(
            steps=(
                ServerOriginMutationStep(
                    domain="notes.keyword_collection",
                    operation="upsert",
                    object_id=object_id,
                    payload={"name": normalized, "parent_sync_id": parent_sync_id},
                    parent_id=parent_sync_id,
                ),
            ),
            load_result=lambda: self._load_resource_row(
                "notes.keyword_collection", object_id
            ),
        )

    def plan_folder_path(
        self,
        path: str,
        *,
        idempotency_key: str | None = None,
    ) -> PlannedNotesMutation:
        """Plan the canonical full folder path in parent-before-child order."""

        normalized = self.normalize_folder_path(path)
        parent_sync_id: str | None = None
        steps: list[ServerOriginMutationStep] = []
        final_sync_id: str | None = None
        parts = normalized.split("/")
        for index, name in enumerate(parts):
            segment_path = "/".join(parts[: index + 1])
            existing = self.note_db.get_note_folder_by_path(segment_path)
            if existing is not None:
                final_sync_id = str(existing["sync_id"])
                canonical_name = str(existing["name"])
            else:
                final_sync_id = self._create_sync_id(
                    "notes.folder",
                    f"{idempotency_key}:{segment_path}" if idempotency_key else None,
                )
                canonical_name = name
            steps.append(
                ServerOriginMutationStep(
                    domain="notes.folder",
                    operation="upsert",
                    object_id=final_sync_id,
                    payload={
                        "name": canonical_name,
                        "parent_sync_id": parent_sync_id,
                    },
                    parent_id=parent_sync_id,
                )
            )
            parent_sync_id = final_sync_id
        if final_sync_id is None:
            raise InputError("Folder path cannot be empty")
        return PlannedNotesMutation(
            steps=tuple(steps),
            load_result=lambda: self._load_resource_row("notes.folder", final_sync_id),
        )

    def plan_folder_change(
        self,
        folder_id: int,
        *,
        name: str,
        parent_id: int | None,
        expected_version: int | None = None,
    ) -> PlannedNotesMutation:
        """Plan a non-public folder rename or move for internal callers."""

        row = self._resource_row("notes.folder", folder_id)
        self._require_version(row, expected_version, "Folder")
        parent_sync_id = None
        if parent_id is not None:
            parent_sync_id = str(self._resource_row("notes.folder", parent_id)["sync_id"])
        object_id = str(row["sync_id"])
        return PlannedNotesMutation(
            steps=(
                ServerOriginMutationStep(
                    domain="notes.folder",
                    operation="upsert",
                    object_id=object_id,
                    payload={"name": str(name or "").strip(), "parent_sync_id": parent_sync_id},
                    parent_id=parent_sync_id,
                ),
            ),
            load_result=lambda: self._load_resource_row("notes.folder", object_id),
        )

    def plan_relationship(
        self,
        domain: SyncDomain,
        members: Mapping[str, str],
        present: bool,
    ) -> PlannedNotesMutation:
        """Plan a deterministic relationship upsert or tombstone."""

        payload = dict(members)
        member_order: dict[SyncDomain, tuple[str, ...]] = {
            "notes.keyword_link": ("subject_type", "subject_id", "keyword_sync_id"),
            "notes.keyword_collection_link": (
                "collection_sync_id",
                "keyword_sync_id",
            ),
            "notes.folder_link": ("note_id", "folder_sync_id"),
        }
        try:
            identity_members = [payload[name] for name in member_order[domain]]
        except KeyError as exc:
            raise InputError("Organization relationship members are incomplete") from exc
        object_id = organization_link_id(domain, identity_members)
        routing_metadata: dict[str, object] = {}
        if present:
            dataset = self.active_dataset()
            current_head = self.service.store.get_current_head(
                dataset.dataset_id,
                domain,
                object_id,
            )
            if current_head is not None and (
                current_head.operation == "tombstone" or current_head.deleted
            ):
                routing_metadata["restore_intent"] = True
        step = ServerOriginMutationStep(
            domain=domain,
            operation="upsert" if present else "tombstone",
            object_id=object_id,
            payload=payload,
            routing_metadata=routing_metadata,
        )
        return PlannedNotesMutation(
            steps=(step,),
            load_result=lambda: self._relationship_present(domain, object_id),
        )

    def plan_collection_with_keywords(
        self,
        *,
        collection_id: int | None,
        name: str,
        parent_id: int | None,
        keywords: Sequence[str],
        idempotency_key: str,
        expected_version: int | None = None,
    ) -> PlannedNotesMutation:
        """Plan one collection change and its complete desired keyword membership."""

        collection = self.plan_collection_change(
            collection_id,
            name,
            parent_id,
            idempotency_key=idempotency_key,
            expected_version=expected_version,
        )
        collection_step = collection.steps[0]
        steps: list[ServerOriginMutationStep] = []
        desired_keyword_ids: list[str] = []
        seen: set[str] = set()
        for raw in keywords:
            normalized = str(raw or "").strip()
            key = normalized.casefold()
            if not normalized or key in seen:
                continue
            seen.add(key)
            row = self.note_db.get_keyword_by_text(normalized)
            if row is None:
                keyword_id = self._create_sync_id(
                    "notes.keyword", f"{idempotency_key}:keyword:{key}"
                )
                canonical_keyword = normalized
            else:
                keyword_id = str(row["sync_id"])
                canonical_keyword = str(row["keyword"])
            steps.append(
                ServerOriginMutationStep(
                    domain="notes.keyword",
                    operation="upsert",
                    object_id=keyword_id,
                    payload={"keyword": canonical_keyword},
                )
            )
            desired_keyword_ids.append(keyword_id)
        steps.append(collection_step)

        current_ids: set[str] = set()
        if collection_id is not None:
            current_ids = {
                str(row["sync_id"])
                for row in self.note_db.get_keywords_for_collection(collection_id)
            }
        desired_ids = set(desired_keyword_ids)
        for keyword_sync_id in sorted(current_ids - desired_ids):
            steps.extend(
                self.plan_relationship(
                    "notes.keyword_collection_link",
                    {
                        "collection_sync_id": collection_step.object_id,
                        "keyword_sync_id": keyword_sync_id,
                    },
                    False,
                ).steps
            )
        for keyword_sync_id in desired_keyword_ids:
            if keyword_sync_id in current_ids:
                continue
            steps.extend(
                self.plan_relationship(
                    "notes.keyword_collection_link",
                    {
                        "collection_sync_id": collection_step.object_id,
                        "keyword_sync_id": keyword_sync_id,
                    },
                    True,
                ).steps
            )
        return PlannedNotesMutation(steps=tuple(steps), load_result=collection.load_result)

    def _resource_row(self, domain: SyncDomain, local_id: int) -> dict[str, Any]:
        try:
            logical_table, _ = _RESOURCE_TABLES[domain]
        except KeyError as exc:
            raise InputError("Unsupported organization resource domain") from exc
        table = self.note_db._map_table_for_backend(logical_table)
        row = self.note_db.execute_query(
            f"SELECT * FROM {table} WHERE id = ? AND deleted = ?",  # nosec B608
            (local_id, False if self.note_db.backend_type.value == "postgresql" else 0),
        ).fetchone()
        if row is None:
            raise NotesOrganizationResourceNotFoundError()
        return dict(row)

    def _load_resource_row(self, domain: SyncDomain, sync_id: str) -> dict[str, Any]:
        resource = NotesOrganizationSyncStore(self.note_db).get_resource(domain, sync_id)
        if resource is None or resource.deleted:
            raise SyncStoreError("Materialized organization resource was not found")
        if domain == "notes.keyword":
            row = self.note_db.get_keyword_by_id(resource.local_id)
        elif domain == "notes.keyword_collection":
            row = self.note_db.get_keyword_collection_by_id(resource.local_id)
        else:
            row = self.note_db.get_note_folder_by_path(self._folder_path(resource.local_id))
        if row is None:
            raise SyncStoreError("Materialized organization resource was not found")
        return row

    def _folder_path(self, local_id: int) -> str:
        row = self.note_db.execute_query(
            "SELECT path FROM note_folders WHERE id = ?", (local_id,)
        ).fetchone()
        if row is None:
            raise SyncStoreError("Materialized folder was not found")
        return str(row["path"])

    def _relationship_present(self, domain: SyncDomain, object_id: str) -> bool:
        snapshot = NotesOrganizationSyncStore(self.note_db).snapshot()
        return any(
            item.domain == domain and item.object_id == object_id
            for item in snapshot.relationships
        )

    @staticmethod
    def _require_version(
        row: Mapping[str, object], expected_version: int | None, _label: str
    ) -> None:
        if expected_version is None:
            return
        if int(row.get("version") or 0) != int(expected_version):
            raise NotesOrganizationVersionConflictError()

    @staticmethod
    def normalize_folder_path(path: str) -> str:
        """Return the canonical identity used for folder-path requests."""

        text = str(path or "").strip().replace("\\", "/").strip("/")
        parts = [part.strip() for part in text.split("/") if part.strip() and part.strip() != "."]
        if not parts or any(part == ".." for part in parts):
            raise InputError("Folder path is invalid")
        return "/".join(parts)

    @staticmethod
    def _create_sync_id(domain: SyncDomain, idempotency_key: str | None) -> str:
        if idempotency_key is None or not idempotency_key.strip():
            return new_organization_sync_id()
        digest = hashlib.sha256(
            f"notes-organization:{domain}:{idempotency_key.strip()}".encode()
        ).hexdigest()
        return str(uuid.UUID(digest[:32], version=4))


__all__ = [
    "NotesOrganizationCoordinator",
    "NotesOrganizationDomainsIncompleteError",
    "NotesOrganizationNotReadyError",
    "NotesOrganizationPreflightError",
    "NotesOrganizationResourceNotFoundError",
    "NotesOrganizationVersionConflictError",
    "PlannedNotesMutation",
]
