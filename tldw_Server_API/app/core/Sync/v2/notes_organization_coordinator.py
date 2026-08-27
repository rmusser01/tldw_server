"""Portable planning and capture for server-origin Notes organization writes."""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, cast

from tldw_Server_API.app.core.DB_Management.chacha.organization_sync_store import (
    NotesOrganizationSyncStore,
    SourceFolderTransitionPlan,
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    ConflictError,
    InputError,
)

from .errors import SyncStoreError
from .materializers.guarded_product_mutation import GuardedProductMutation
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

_REQUEST_FINGERPRINT_KEY = "notes_organization_request_fingerprint"
_RESPONSE_STATUS_KEY = "notes_organization_response_status"
_NOTE_RESPONSE_KEY = "notes_organization_note_response"
_FOLDER_PROVENANCE_KEY = "notes_folder_origin_provenance"
_KEYWORD_MERGE_RESPONSE_KEY = "notes_keyword_merge_response"
_KEYWORD_MERGE_PRECONDITION_KEY = "notes_keyword_merge_precondition"


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


class NotesKeywordMergeUnsynchronizedDependencyError(InputError):
    """A merge would rewrite a relationship outside the synchronized domains."""

    error_code = "notes_keyword_merge_unsynchronized_dependency"

    def __init__(self) -> None:
        super().__init__(self.error_code)


@dataclass(frozen=True)
class PlannedNotesMutation:
    """A read-only canonical plan and its post-materialization result loader."""

    steps: tuple[ServerOriginMutationStep, ...]
    load_result: Callable[[], object]
    response_status: int | None = None
    source_transition: SourceFolderTransitionPlan | None = None


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
        guarded_mutation: GuardedProductMutation | None = None,
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
                guarded_mutation=guarded_mutation,
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
            response_status=plan.response_status,
            source_transition=plan.source_transition,
        )

    @staticmethod
    def bind_response_status(
        plan: PlannedNotesMutation,
        response_status: int,
    ) -> PlannedNotesMutation:
        """Persist a safe immutable dynamic response status with a plan."""

        if response_status not in {200, 201}:
            raise InputError("Unsupported Notes organization response status")
        return PlannedNotesMutation(
            steps=tuple(
                replace(
                    step,
                    routing_metadata={
                        **dict(step.routing_metadata),
                        _RESPONSE_STATUS_KEY: response_status,
                    },
                )
                for step in plan.steps
            ),
            load_result=plan.load_result,
            response_status=response_status,
            source_transition=plan.source_transition,
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
        mutation_group_id = server_origin_mutation_batch_group_id(
            dataset_id=dataset.dataset_id,
            source=source,
            idempotency_key=normalized_key,
        )
        stored_group = self.service.store.list_mutation_group(
            dataset.dataset_id, mutation_group_id
        )
        group_boundary = max(
            int(envelope.server_cursor or 0) for envelope in stored_group
        )
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
        response_statuses = {
            step.routing_metadata.get(_RESPONSE_STATUS_KEY) for step in manifest
        }
        if response_statuses == {None}:
            response_status = None
        elif len(response_statuses) == 1 and next(iter(response_statuses)) in {
            200,
            201,
        }:
            response_status = int(next(iter(response_statuses)))
        else:
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
                        result_domain,
                        result_step.object_id,
                        result_step.payload,
                    )
            elif result_domain == "notes.note":
                def loader() -> object:
                    return self._load_note_response(
                        result_step,
                        dataset_id=dataset.dataset_id,
                        server_cursor_boundary=group_boundary,
                    )
            else:
                def loader() -> object:
                    return self._load_resource_row(
                        result_domain, result_step.object_id
                    )
        return PlannedNotesMutation(
            steps=manifest,
            load_result=loader,
            response_status=response_status,
        )

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
        *,
        source: str | None = None,
        idempotency_key: str | None = None,
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
        request_fingerprint: str | None = None
        if source is not None:
            request_fingerprint = self.request_fingerprint(
                "relationship.set",
                {
                    "domain": domain,
                    "members": {
                        name: payload[name] for name in member_order[domain]
                    },
                    "present": present,
                },
            )
            replay = self.replay_request_plan(
                source=source,
                idempotency_key=idempotency_key,
                request_fingerprint=request_fingerprint,
                result_domain=domain,
                relationship_result=True,
            )
            if replay is not None:
                return replay
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
        plan = PlannedNotesMutation(
            steps=(step,),
            load_result=lambda: self._relationship_present(domain, object_id, payload),
        )
        if request_fingerprint is not None:
            return self.bind_request(plan, request_fingerprint)
        return plan

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

    def plan_note_with_organization(
        self,
        *,
        note_step: ServerOriginMutationStep,
        keywords: Sequence[str] | None,
        folder_paths: Sequence[str] | None,
    ) -> PlannedNotesMutation:
        """Plan one note mutation and all requested organization deltas."""

        if note_step.domain != "notes.note" or note_step.operation != "upsert":
            raise InputError("Compound note organization requires a note upsert")
        note_id = note_step.object_id
        identity_seed = note_step.stable_key or hashlib.sha256(
            json.dumps(
                {
                    "object_id": note_id,
                    "payload": dict(note_step.payload),
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        existing_note = self.note_db.get_note_by_id(note_id)
        current_keywords = (
            self.note_db.get_keywords_for_note(note_id) if existing_note is not None else []
        )
        current_folders = (
            self.note_db.get_note_folders_for_note(note_id)
            if existing_note is not None
            else []
        )
        manual_folder_ids = (
            NotesOrganizationSyncStore(self.note_db).manual_folder_sync_ids(note_id)
            if existing_note is not None
            else set()
        )
        result_keyword_ids = [str(row["sync_id"]) for row in current_keywords]
        result_folder_ids = [str(row["sync_id"]) for row in current_folders]
        steps: list[ServerOriginMutationStep] = [note_step]

        if keywords is not None:
            desired_keywords: list[tuple[str, str]] = []
            seen_keywords: set[str] = set()
            for raw_keyword in keywords:
                keyword = str(raw_keyword or "").strip()
                key = keyword.casefold()
                if not keyword or key in seen_keywords:
                    continue
                seen_keywords.add(key)
                existing = self.note_db.get_keyword_by_text(keyword)
                if existing is None:
                    sync_id = self._create_sync_id(
                        "notes.keyword", f"{identity_seed}:keyword:{key}"
                    )
                    steps.append(
                        ServerOriginMutationStep(
                            domain="notes.keyword",
                            operation="upsert",
                            object_id=sync_id,
                            payload={"keyword": keyword},
                        )
                    )
                else:
                    sync_id = str(existing["sync_id"])
                desired_keywords.append((key, sync_id))

            current_keyword_ids = set(result_keyword_ids)
            desired_keyword_ids = {sync_id for _, sync_id in desired_keywords}
            result_keyword_ids = [sync_id for _, sync_id in desired_keywords]
            for sync_id in sorted(current_keyword_ids - desired_keyword_ids):
                steps.extend(
                    self.plan_relationship(
                        "notes.keyword_link",
                        {
                            "subject_type": "note",
                            "subject_id": note_id,
                            "keyword_sync_id": sync_id,
                        },
                        False,
                    ).steps
                )
            for _, sync_id in desired_keywords:
                if sync_id in current_keyword_ids:
                    continue
                steps.extend(
                    self.plan_relationship(
                        "notes.keyword_link",
                        {
                            "subject_type": "note",
                            "subject_id": note_id,
                            "keyword_sync_id": sync_id,
                        },
                        True,
                    ).steps
                )

        if folder_paths is not None:
            expanded_paths: list[str] = []
            seen_paths: set[str] = set()
            for raw_path in folder_paths:
                normalized_path = self.normalize_folder_path(raw_path)
                parts = normalized_path.split("/")
                for index in range(len(parts)):
                    path = "/".join(parts[: index + 1])
                    key = path.casefold()
                    if key not in seen_paths:
                        seen_paths.add(key)
                        expanded_paths.append(path)

            desired_folders: list[tuple[str, str]] = []
            parent_sync_ids: dict[str, str] = {}
            for path in expanded_paths:
                existing = self.note_db.get_note_folder_by_path(path)
                if existing is None:
                    sync_id = self._create_sync_id(
                        "notes.folder", f"{identity_seed}:folder:{path.casefold()}"
                    )
                    parent_path = path.rsplit("/", 1)[0] if "/" in path else None
                    parent_sync_id = (
                        parent_sync_ids[parent_path.casefold()]
                        if parent_path is not None
                        else None
                    )
                    steps.append(
                        ServerOriginMutationStep(
                            domain="notes.folder",
                            operation="upsert",
                            object_id=sync_id,
                            payload={
                                "name": path.rsplit("/", 1)[-1],
                                "parent_sync_id": parent_sync_id,
                            },
                            parent_id=parent_sync_id,
                        )
                    )
                else:
                    sync_id = str(existing["sync_id"])
                parent_sync_ids[path.casefold()] = sync_id
                desired_folders.append((path, sync_id))

            current_folder_ids = set(result_folder_ids)
            desired_folder_ids = {sync_id for _, sync_id in desired_folders}
            result_folder_ids = [sync_id for _, sync_id in desired_folders]
            for sync_id in sorted(current_folder_ids - desired_folder_ids):
                steps.extend(
                    self.plan_relationship(
                        "notes.folder_link",
                        {"note_id": note_id, "folder_sync_id": sync_id},
                        False,
                    ).steps
                )
            for _, sync_id in desired_folders:
                if sync_id in current_folder_ids and sync_id in manual_folder_ids:
                    continue
                steps.extend(
                    self.plan_relationship(
                        "notes.folder_link",
                        {"note_id": note_id, "folder_sync_id": sync_id},
                        True,
                    ).steps
                )

        response_time = self.service.clock()
        response_metadata = {
            "created_at": (
                str(existing_note["created_at"])
                if existing_note is not None
                else response_time
            ),
            "last_modified": response_time,
            "version": (
                int(existing_note["version"]) + 1 if existing_note is not None else 1
            ),
            "client_id": self.user_id,
            "deleted": False,
            "keyword_sync_ids": result_keyword_ids,
            "folder_sync_ids": result_folder_ids,
        }
        note_step = replace(
            note_step,
            routing_metadata={
                **dict(note_step.routing_metadata),
                _NOTE_RESPONSE_KEY: response_metadata,
            },
        )
        steps[0] = note_step
        return PlannedNotesMutation(
            steps=tuple(steps),
            load_result=lambda: self._load_note_response(note_step),
        )

    def plan_source_folder_change(
        self,
        *,
        note_id: str,
        source_id: int,
        folder_id: int,
        present: bool,
        idempotency_key: str,
        source: str = "notes-ingestion",
    ) -> PlannedNotesMutation:
        """Plan only a prospective effective source-folder transition."""

        folder = self._resource_row("notes.folder", folder_id)
        folder_sync_id = str(folder["sync_id"])
        dataset = self.require_ready()
        transition_identity = server_origin_mutation_batch_group_id(
            dataset_id=dataset.dataset_id,
            source=source,
            idempotency_key=idempotency_key,
        )
        transition = NotesOrganizationSyncStore(
            self.note_db
        ).source_folder_transition_plan(
            note_id=note_id,
            source_id=source_id,
            folder_sync_id=folder_sync_id,
            present=present,
            transition_identity=transition_identity,
        )
        if transition.operation is None:
            return PlannedNotesMutation(
                steps=(),
                load_result=lambda: None,
                source_transition=transition,
            )
        relationship = self.plan_relationship(
            "notes.folder_link",
            {"note_id": note_id, "folder_sync_id": folder_sync_id},
            transition.operation == "upsert",
        )
        provenance = {
            "operation": "source_upsert" if present else "source_delete",
            "source_id": source_id,
            "pre_state_hash": transition.pre_state_hash,
            "post_state_hash": transition.post_state_hash,
        }
        return PlannedNotesMutation(
            steps=tuple(
                replace(
                    step,
                    routing_metadata={
                        **dict(step.routing_metadata),
                        _FOLDER_PROVENANCE_KEY: provenance,
                    },
                )
                for step in relationship.steps
            ),
            load_result=relationship.load_result,
            source_transition=transition,
        )

    def apply_source_folder_provenance_only(
        self,
        *,
        note_id: str,
        source_id: int,
        folder_id: int,
        present: bool,
        transition: SourceFolderTransitionPlan,
    ) -> bool:
        """Apply a non-visible source delta only after Sync readiness succeeds."""

        self.require_ready()
        folder = self._resource_row("notes.folder", folder_id)
        return NotesOrganizationSyncStore(self.note_db).apply_source_folder_provenance(
            note_id=note_id,
            folder_sync_id=str(folder["sync_id"]),
            operation="source_upsert" if present else "source_delete",
            source_id=source_id,
            pre_state_hash=transition.pre_state_hash,
            post_state_hash=transition.post_state_hash,
            transition_identity=transition.transition_identity,
        )

    def plan_keyword_merge(
        self,
        *,
        source_keyword_id: int,
        target_keyword_id: int,
        expected_source_version: int,
        expected_target_version: int | None,
    ) -> PlannedNotesMutation:
        """Plan every synchronized relationship move and tombstone source last."""

        snapshot = self.note_db.keyword_store.synchronized_merge_snapshot(
            source_keyword_id=source_keyword_id,
            target_keyword_id=target_keyword_id,
            expected_source_version=expected_source_version,
            expected_target_version=expected_target_version,
        )
        if snapshot["has_unsynchronized_dependency"]:
            raise NotesKeywordMergeUnsynchronizedDependencyError()
        source = cast(Mapping[str, object], snapshot["source"])
        target = cast(Mapping[str, object], snapshot["target"])
        source_sync_id = str(source["sync_id"])
        target_sync_id = str(target["sync_id"])
        relationships = cast(Sequence[Mapping[str, object]], snapshot["relationships"])
        steps: list[ServerOriginMutationStep] = []
        merged_counts = {
            "merged_note_links": 0,
            "merged_conversation_links": 0,
            "merged_collection_links": 0,
        }

        for item in relationships:
            if bool(item["target_present"]):
                continue
            domain = cast(SyncDomain, item["domain"])
            members = dict(cast(Mapping[str, str], item["members"]))
            members["keyword_sync_id"] = target_sync_id
            steps.extend(self.plan_relationship(domain, members, True).steps)
            if domain == "notes.keyword_collection_link":
                merged_counts["merged_collection_links"] += 1
            elif members["subject_type"] == "note":
                merged_counts["merged_note_links"] += 1
            else:
                merged_counts["merged_conversation_links"] += 1

        for item in relationships:
            domain = cast(SyncDomain, item["domain"])
            members = dict(cast(Mapping[str, str], item["members"]))
            members["keyword_sync_id"] = source_sync_id
            steps.extend(self.plan_relationship(domain, members, False).steps)
        steps.append(
            ServerOriginMutationStep(
                domain="notes.keyword",
                operation="tombstone",
                object_id=source_sync_id,
                payload={},
                routing_metadata={
                    _KEYWORD_MERGE_PRECONDITION_KEY: {
                        "relationship_set_hash": (
                            self.note_db.keyword_store.empty_synchronized_relationship_set_hash()
                        )
                    }
                },
            )
        )
        response: dict[str, object] = {
            "source_keyword_id": source_keyword_id,
            "target_keyword_id": target_keyword_id,
            "source_deleted_version": expected_source_version + 1,
            "target_version": int(target["version"]),
            **merged_counts,
            "merged_flashcard_links": 0,
        }
        plan = PlannedNotesMutation(
            steps=tuple(
                replace(
                    step,
                    routing_metadata={
                        **dict(step.routing_metadata),
                        _KEYWORD_MERGE_RESPONSE_KEY: response,
                    },
                )
                for step in steps
            ),
            load_result=lambda: dict(response),
        )
        return plan

    @staticmethod
    def restore_keyword_merge_result(
        plan: PlannedNotesMutation,
    ) -> PlannedNotesMutation:
        """Restore an exact merge response from its immutable durable manifest."""

        responses = [
            step.routing_metadata.get(_KEYWORD_MERGE_RESPONSE_KEY)
            for step in plan.steps
        ]
        if not responses or not isinstance(responses[0], Mapping):
            raise SyncServerOriginBatchIdempotencyConflictError("invalid-merge-manifest")
        response = dict(responses[0])
        if any(item != response for item in responses):
            raise SyncServerOriginBatchIdempotencyConflictError("invalid-merge-manifest")
        expected_keys = {
            "source_keyword_id",
            "target_keyword_id",
            "source_deleted_version",
            "target_version",
            "merged_note_links",
            "merged_conversation_links",
            "merged_collection_links",
            "merged_flashcard_links",
        }
        if set(response) != expected_keys or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in response.values()
        ):
            raise SyncServerOriginBatchIdempotencyConflictError("invalid-merge-manifest")
        return PlannedNotesMutation(
            steps=plan.steps,
            load_result=lambda: dict(response),
            response_status=plan.response_status,
        )

    def _resource_row(self, domain: SyncDomain, local_id: int) -> dict[str, Any]:
        row = NotesOrganizationSyncStore(self.note_db).get_resource_row_by_local_id(
            domain,
            local_id,
        )
        if row is None:
            raise NotesOrganizationResourceNotFoundError()
        return row

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

    def _load_note_row(self, note_id: str) -> dict[str, Any]:
        row = self.note_db.get_note_by_id(note_id)
        if row is None:
            raise SyncStoreError("Materialized note was not found")
        return row

    def _load_note_response(
        self,
        note_step: ServerOriginMutationStep,
        *,
        dataset_id: str | None = None,
        server_cursor_boundary: int | None = None,
    ) -> dict[str, Any]:
        """Rebuild one immutable compound response from its durable manifest."""

        metadata = note_step.routing_metadata.get(_NOTE_RESPONSE_KEY)
        if not isinstance(metadata, Mapping):
            return self._load_note_row(note_step.object_id)
        response = {
            "id": note_step.object_id,
            "title": str(note_step.payload.get("title") or ""),
            "content": str(note_step.payload.get("content") or ""),
            "conversation_id": note_step.payload.get("conversation_id"),
            "message_id": note_step.payload.get("message_id"),
        }
        for key in ("created_at", "last_modified", "version", "client_id", "deleted"):
            if key in metadata:
                response[key] = metadata[key]
        keyword_ids = metadata.get("keyword_sync_ids")
        folder_ids = metadata.get("folder_sync_ids")
        resolved_dataset_id = dataset_id or self.active_dataset().dataset_id
        if isinstance(keyword_ids, list) and all(
            isinstance(sync_id, str) for sync_id in keyword_ids
        ):
            response["keywords"] = sorted(
                (
                    self._load_resource_response_at(
                        resolved_dataset_id,
                        "notes.keyword",
                        sync_id,
                        server_cursor_boundary,
                    )
                    for sync_id in keyword_ids
                ),
                key=lambda row: str(row.get("keyword") or "").casefold(),
            )
        if isinstance(folder_ids, list) and all(
            isinstance(sync_id, str) for sync_id in folder_ids
        ):
            response["folders"] = sorted(
                (
                    self._load_resource_response_at(
                        resolved_dataset_id,
                        "notes.folder",
                        sync_id,
                        server_cursor_boundary,
                    )
                    for sync_id in folder_ids
                ),
                key=lambda row: str(row.get("path") or "").casefold(),
            )
        return response

    def _load_resource_response_at(
        self,
        dataset_id: str,
        domain: SyncDomain,
        sync_id: str,
        server_cursor_boundary: int | None,
    ) -> dict[str, Any]:
        """Rebuild resource display data from canonical envelope history."""

        envelope = self._resource_envelope_at(
            dataset_id, domain, sync_id, server_cursor_boundary
        )
        resource = NotesOrganizationSyncStore(self.note_db).get_resource(domain, sync_id)
        if resource is None:
            raise SyncStoreError("Materialized organization resource was not found")
        row = NotesOrganizationSyncStore(self.note_db).get_resource_row_by_local_id(
            domain,
            resource.local_id,
            include_deleted=True,
        )
        if row is None:
            raise SyncStoreError("Materialized organization resource was not found")
        result = row
        if domain == "notes.keyword":
            result.update(
                {
                    "keyword": str(envelope.payload.get("keyword") or ""),
                    "last_modified": envelope.created_at_client
                    or result.get("last_modified"),
                    "version": envelope.object_revision or result.get("version"),
                    "client_id": envelope.routing_metadata.get(
                        "server_owner_user_id"
                    )
                    or envelope.device_id
                    or result.get("client_id"),
                    "deleted": False,
                }
            )
            return result
        if domain != "notes.folder":
            return result
        parent_sync_id = envelope.payload.get("parent_sync_id")
        parent_id = None
        parent_path = None
        if isinstance(parent_sync_id, str) and parent_sync_id:
            parent_resource = NotesOrganizationSyncStore(self.note_db).get_resource(
                "notes.folder", parent_sync_id
            )
            if parent_resource is None:
                raise SyncStoreError("Materialized folder parent was not found")
            parent_id = parent_resource.local_id
            parent_row = self._load_resource_response_at(
                dataset_id,
                "notes.folder",
                parent_sync_id,
                server_cursor_boundary,
            )
            parent_path = str(parent_row["path"])
        name = str(envelope.payload.get("name") or "")
        result.update(
            {
                "name": name,
                "path": f"{parent_path}/{name}" if parent_path else name,
                "parent_id": parent_id,
            }
        )
        return result

    def _resource_envelope_at(
        self,
        dataset_id: str,
        domain: SyncDomain,
        sync_id: str,
        server_cursor_boundary: int | None,
    ):
        boundary = server_cursor_boundary
        if boundary is None:
            history = self.service.store.list_envelopes_for_entity(
                dataset_id,
                domain,
                entity_id=sync_id,
                limit=1,
            )
            envelope = history[0] if history else None
        else:
            envelope = self.service.store.get_envelope_for_entity_at_or_before(
                dataset_id,
                domain,
                entity_id=sync_id,
                server_sequence=boundary,
            )
        if envelope is None:
            raise SyncStoreError("Organization resource history was not found")
        if envelope.operation != "upsert":
            raise SyncStoreError("Organization resource was not active at replay")
        return envelope

    def _folder_path(self, local_id: int) -> str:
        row = self.note_db.execute_query(
            "SELECT path FROM note_folders WHERE id = ?", (local_id,)
        ).fetchone()
        if row is None:
            raise SyncStoreError("Materialized folder was not found")
        return str(row["path"])

    def _relationship_present(
        self,
        domain: SyncDomain,
        object_id: str,
        payload: Mapping[str, object],
    ) -> bool:
        return NotesOrganizationSyncStore(self.note_db).relationship_present(
            domain=domain,
            object_id=object_id,
            payload=payload,
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
    "NotesKeywordMergeUnsynchronizedDependencyError",
    "PlannedNotesMutation",
]
