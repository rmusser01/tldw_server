"""Authenticated service boundary for the server Personal Context profile."""

from __future__ import annotations

import json
import hashlib
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError
from tldw_profile_core import (
    ActorType,
    ProfileControls,
    ProfileManifest,
    ProfilePayload,
    ProfileProposal,
    ProfileProvenance,
    ProfileRecord,
    ProfileScope,
    ProposalOperation,
    ProposalState,
    ProvenanceSource,
    RecordKind,
    RecordState,
    ScopeKind,
    SemanticKey,
    SyncMode,
)

from tldw_Server_API.app.core.exceptions import (
    ProfileConflictError,
    ProfileKeyCollisionError,
    ProfileNotFoundError,
    ProfileUnsupportedOperationError,
)
from tldw_Server_API.app.core.Personalization.personal_context_export import (
    PLAINTEXT_EXPORT_CONFIRMATION,
    RECOVERY_EXPORT_CONFIRMATION,
    encrypt_recovery_export,
    require_confirmation,
)
from tldw_Server_API.app.core.Personalization.personal_context_repository import (
    PersonalContextRepository,
)
from tldw_Server_API.app.core.Personalization.personal_context_repository_models import (
    ConcurrentProfileUpdateError,
    ProfileAlreadyExistsError,
    ProfileIntegrityError,
    ProfileSemanticKeyCollisionError,
    ProfileStorageLockedError,
    ProfileUnsupportedSchemaError,
)
from tldw_Server_API.app.core.Personalization.personal_context_runtime_policy import (
    PROFILE_RUNTIME_POLICY_ID,
    RuntimePolicyVersion,
    ServerRuntimePolicy,
    WorkspaceRuntimePolicy,
)

_PayloadAdapter = TypeAdapter(ProfilePayload)
_ControlsAdapter = TypeAdapter(ProfileControls)
_SemanticKeyAdapter = TypeAdapter(SemanticKey)


class ProfileOperationalState(StrEnum):
    """Stable server profile readiness states exposed to clients."""

    ABSENT = "absent"
    AVAILABLE = "available"
    DISABLED = "disabled"
    LOCKED = "locked"
    REVIEW_REQUIRED = "review_required"
    PURGE_PENDING = "purge_pending"
    UNSUPPORTED = "unsupported"


class ProfileStatus(BaseModel):
    """Content-free profile status suitable for authenticated API responses."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    state: ProfileOperationalState
    profile_id: str | None = None
    revision: int | None = None
    purge_generation: int | None = None


_UNSET = object()


@dataclass(frozen=True, slots=True)
class RecordMutation:
    """Fields a user-authorized record update may replace."""

    payload: Any = _UNSET
    semantic_key: Any = _UNSET
    controls: Any = _UNSET
    expires_at: Any = _UNSET
    no_expiry: Any = _UNSET


@dataclass(frozen=True, slots=True)
class PersonalContextSyncSnapshot:
    """One transactionally-read canonical snapshot authorized for Sync bootstrap."""

    manifest: ProfileManifest
    scopes: tuple[ProfileScope, ...]
    records: tuple[ProfileRecord, ...]
    proposals: tuple[ProfileProposal, ...]
    integrity_key_id: str
    integrity_key: bytes
    cursor: str


class PersonalContextService:
    """Own all authenticated Personal Context reads and mutations for one user."""

    def __init__(
        self,
        repository: PersonalContextRepository,
        *,
        clock: Callable[[], datetime] | None = None,
        id_factory: Callable[[str], str] | None = None,
        workspace_access: Callable[[str], bool] | None = None,
    ) -> None:
        self._repository = repository
        self._clock = clock or (lambda: datetime.now(UTC))
        self._id_factory = id_factory or (lambda label: f"{label}-{uuid.uuid4()}")
        self._workspace_access = workspace_access or (lambda _workspace_id: False)

    def _now(self) -> datetime:
        value = self._clock()
        if value.tzinfo is None:
            raise ValueError("Personal Context clock must be timezone-aware")
        return value

    def _profile_id(self) -> str:
        profile_ids = self._repository.profile_ids()
        if not profile_ids:
            if self._repository.has_profile_state():
                raise ProfileStorageLockedError("Personal Context key material exists without a manifest")
            raise ProfileNotFoundError("Personal context profile not found")
        if len(profile_ids) != 1:
            raise ProfileIntegrityError("Multiple Personal Context manifests exist")
        return profile_ids[0]

    def _manifest(self) -> ProfileManifest:
        profile_id = self._profile_id()
        manifest = self._repository.get_manifest(profile_id)
        if manifest is None:
            raise ProfileIntegrityError("Personal Context manifest is unavailable")
        return manifest

    def _writable_manifest(self) -> ProfileManifest:
        """Return the manifest only while the profile accepts mutations."""

        manifest = self._manifest()
        if not self._repository.list_scopes(manifest.profile_id, limit=1):
            raise ProfileUnsupportedOperationError("profile_purge_pending")
        return manifest

    def _next_manifest(self, current: ProfileManifest) -> ProfileManifest:
        return ProfileManifest.model_validate(
            {
                **current.model_dump(mode="python"),
                "revision": current.revision + 1,
                "updated_at": self._now(),
                "current_version_id": self._id_factory("manifest-version"),
            }
        )

    def sync_integrity_key(self, profile_id: str) -> tuple[str, bytes]:
        """Resolve canonical integrity custody for this user's Sync dataset."""

        if profile_id != self._profile_id():
            raise KeyError("Personal context profile not found")
        return self._repository.sync_integrity_key(profile_id)

    def sync_bootstrap_snapshot(self) -> PersonalContextSyncSnapshot:
        """Return all eligible canonical Sync heads from one repository read transaction."""

        profile_id = self._profile_id()
        manifest, scopes, records, proposals, key_id, key = self._repository.sync_bootstrap_snapshot(
            profile_id
        )
        records = tuple(
            record for record in records if record.controls.sync_mode is SyncMode.SYNCABLE
        )
        proposals = tuple(
            proposal
            for proposal in proposals
            if proposal.proposed_record is None
            or proposal.proposed_record.controls.sync_mode is SyncMode.SYNCABLE
        )
        cursor_values = [
            f"manifest:{manifest.profile_id}:{manifest.current_version_id}",
            f"purge:{manifest.purge_generation}",
            *(f"scope:{item.scope_id}:{item.version_id}" for item in scopes),
            *(f"record:{item.record_id}:{item.version_id}" for item in records),
            *(
                "proposal:"
                + item.proposal_id
                + ":"
                + hashlib.sha256(item.model_dump_json().encode("utf-8")).hexdigest()
                for item in proposals
            ),
        ]
        cursor = "personal-context-bootstrap-v1:" + hashlib.sha256(
            "\x1e".join(sorted(cursor_values)).encode("utf-8")
        ).hexdigest()
        return PersonalContextSyncSnapshot(
            manifest=manifest,
            scopes=scopes,
            records=records,
            proposals=proposals,
            integrity_key_id=key_id,
            integrity_key=key,
            cursor=cursor,
        )

    def sync_encryption_key(self, profile_id: str) -> tuple[bytes, int]:
        """Resolve canonical encryption custody for this user's Sync dataset."""

        if profile_id != self._profile_id():
            raise KeyError("Personal context profile not found")
        return self._repository.sync_encryption_key(profile_id)

    def status(self) -> ProfileStatus:
        """Return a content-free operational state without leaking profile records."""

        try:
            profile_ids = self._repository.profile_ids()
            if not profile_ids:
                state = (
                    ProfileOperationalState.LOCKED
                    if self._repository.has_profile_state()
                    else ProfileOperationalState.ABSENT
                )
                return ProfileStatus(state=state)
            if len(profile_ids) != 1:
                return ProfileStatus(state=ProfileOperationalState.LOCKED)
            profile_id = profile_ids[0]
            manifest = self._repository.get_manifest(profile_id)
            if manifest is None:
                return ProfileStatus(state=ProfileOperationalState.LOCKED)
            base = {
                "profile_id": profile_id,
                "revision": manifest.revision,
                "purge_generation": manifest.purge_generation,
            }
            if not self._repository.list_scopes(profile_id):
                return ProfileStatus(
                    state=ProfileOperationalState.PURGE_PENDING,
                    **base,
                )
            now = self._now()
            if self._has_live_pending_proposal(profile_id, now=now):
                return ProfileStatus(
                    state=ProfileOperationalState.REVIEW_REQUIRED,
                    **base,
                )
            if not self.get_runtime_policy().enabled:
                return ProfileStatus(state=ProfileOperationalState.DISABLED, **base)
            return ProfileStatus(state=ProfileOperationalState.AVAILABLE, **base)
        except ProfileUnsupportedSchemaError:
            return ProfileStatus(state=ProfileOperationalState.UNSUPPORTED)
        except (ProfileIntegrityError, ProfileStorageLockedError, ValidationError):
            return ProfileStatus(state=ProfileOperationalState.LOCKED)

    def create_profile(self, *, runtime_enabled: bool = False) -> ProfileManifest:
        """Create one empty canonical profile and its required global scope."""

        now = self._now()
        profile_id = self._id_factory("profile")
        manifest = ProfileManifest(
            profile_id=profile_id,
            revision=0,
            purge_generation=0,
            created_at=now,
            updated_at=now,
            current_version_id=self._id_factory("manifest-version"),
        )
        global_scope = ProfileScope(
            scope_id=self._id_factory("scope"),
            profile_id=profile_id,
            kind=ScopeKind.GLOBAL,
            version_id=self._id_factory("scope-version"),
            created_at=now,
            updated_at=now,
        )
        runtime_policy = ServerRuntimePolicy(enabled=True).model_dump(mode="json") if runtime_enabled else None
        runtime_version_id = self._id_factory("runtime-version") if runtime_enabled else None
        try:
            self._repository.create_profile(
                manifest,
                global_scope,
                runtime_policy=runtime_policy,
                runtime_version_id=runtime_version_id,
            )
        except ProfileAlreadyExistsError as exc:
            raise ProfileConflictError("Personal context profile already exists") from exc
        return manifest

    def apply_sync_object(
        self,
        *,
        domain: str,
        value: ProfileManifest | ProfileScope | ProfileRecord | ProfileProposal | Mapping[str, Any],
        actor_type: str,
        actor_id: str | None,
        base_object_hash: str | None = None,
    ) -> ProfileManifest | ProfileScope | ProfileRecord | ProfileProposal | Mapping[str, Any]:
        """Apply one adapter-authenticated canonical Sync object exactly once.

        Sync adapters own transport integrity and lineage hashes. This service
        remains the only mutation boundary and re-checks canonical profile,
        scope, record-version, proposal, and purge fences in repository
        transactions.
        """

        del base_object_hash
        if actor_type != "sync" or not isinstance(actor_id, str) or not actor_id:
            raise PermissionError("Personal Context sync actor is invalid")
        try:
            if domain == "personal_context.manifest":
                manifest = ProfileManifest.model_validate(value)
                current = self._manifest()
                if manifest == current:
                    return current
                if manifest.profile_id != current.profile_id:
                    raise ProfileConflictError("Personal context profile changed")
                self._repository.commit_manifest_version(
                    manifest,
                    expected_version_id=current.current_version_id,
                )
                return manifest

            profile_id = self._profile_id()
            if domain == "personal_context.scope":
                scope = ProfileScope.model_validate(value)
                if scope.profile_id != profile_id:
                    raise ProfileConflictError("Personal context profile changed")
                current_scope = self._repository.get_scope(profile_id, scope.scope_id)
                if scope == current_scope:
                    return scope
                if current_scope is not None and (
                    scope.kind is not current_scope.kind
                    or scope.created_at != current_scope.created_at
                    or scope.updated_at < current_scope.updated_at
                    or scope.version_id == current_scope.version_id
                ):
                    raise ProfileConflictError("Personal context scope changed")
                if (
                    current_scope is None
                    and scope.kind is ScopeKind.GLOBAL
                    and any(
                        candidate.kind is ScopeKind.GLOBAL
                        for candidate in self.list_scopes()
                    )
                ):
                    raise ProfileConflictError("Personal context global scope changed")
                self._repository.commit_scope(
                    scope,
                    expected_version_id=(
                        None if current_scope is None else current_scope.version_id
                    ),
                )
                return scope

            if domain == "personal_context.record":
                record = ProfileRecord.model_validate(value)
                if record.profile_id != profile_id:
                    raise ProfileConflictError("Personal context profile changed")
                self._require_scope(profile_id, record.scope_id)
                self._validate_server_controls(record.controls)
                current_record = self._repository.get_record(profile_id, record.record_id)
                if record == current_record:
                    return record
                expected_version = (
                    None if current_record is None else current_record.version_id
                )
                orphan_tombstone = (
                    current_record is None
                    and record.state is RecordState.DELETED
                    and record.payload is None
                    and record.parent_version_id is not None
                )
                if record.parent_version_id != expected_version and not orphan_tombstone:
                    raise ProfileConflictError("Personal context record changed")
                if current_record is not None and (
                    current_record.state is RecordState.DELETED
                    or record.scope_id != current_record.scope_id
                    or record.kind is not current_record.kind
                    or record.created_at != current_record.created_at
                    or record.updated_at < current_record.updated_at
                    or record.version_id == current_record.version_id
                ):
                    raise ProfileConflictError("Personal context record changed")
                self._ensure_semantic_key_available(
                    record,
                    excluding_record_id=(
                        None if current_record is None else current_record.record_id
                    ),
                )
                self._repository.commit_record_version(
                    record,
                    expected_version_id=expected_version,
                    allow_orphan_tombstone=orphan_tombstone,
                )
                return record

            if domain == "personal_context.proposal":
                proposal = ProfileProposal.model_validate(value)
                if proposal.profile_id != profile_id:
                    raise ProfileConflictError("Personal context profile changed")
                if (
                    proposal.state is ProposalState.PENDING
                    and proposal.proposed_record is not None
                    and proposal.proposed_record.controls.sync_mode
                    is SyncMode.DEVICE_ONLY
                ):
                    raise ValueError("Device-only proposals cannot synchronize")
                self._require_scope(profile_id, proposal.scope_id)
                current_proposal = self._repository.get_proposal(
                    profile_id, proposal.proposal_id
                )
                if proposal == current_proposal:
                    return proposal
                if proposal.state is ProposalState.PENDING:
                    if current_proposal is not None:
                        raise ProfileConflictError(
                            "Personal context proposal changed"
                        )
                    self.create_proposal(proposal)
                    return proposal
                if (
                    current_proposal is not None
                    and current_proposal.state is not ProposalState.PENDING
                ):
                    raise ProfileConflictError("Personal context proposal changed")
                self._repository.commit_synced_proposal_receipt(
                    proposal,
                    expected_manifest_version=self._manifest().current_version_id,
                )
                return proposal

            if domain == "personal_context.purge":
                barrier = dict(value)
                if set(barrier) != {
                    "schema_version",
                    "profile_id",
                    "purge_generation",
                }:
                    raise ValueError("Personal context purge barrier is invalid")
                current = self._manifest()
                if barrier["profile_id"] != current.profile_id:
                    raise ProfileConflictError("Personal context profile changed")
                generation = barrier["purge_generation"]
                if generation == current.purge_generation:
                    return barrier
                if generation != current.purge_generation + 1:
                    raise ProfileConflictError("Personal context purge generation changed")
                purged = ProfileManifest.model_validate(
                    {
                        **current.model_dump(mode="python"),
                        "revision": current.revision + 1,
                        "purge_generation": generation,
                        "updated_at": self._now(),
                        "current_version_id": self._id_factory("manifest-version"),
                    }
                )
                self._repository.purge_profile(
                    purged,
                    expected_manifest_version=current.current_version_id,
                )
                return barrier
        except (ConcurrentProfileUpdateError, ProfileSemanticKeyCollisionError) as exc:
            raise ProfileConflictError("Personal context changed concurrently") from exc
        raise ValueError("Unsupported Personal Context Sync domain")

    def get_manifest(self) -> ProfileManifest:
        """Return the authenticated user's manifest."""

        return self._manifest()

    def list_scopes(self) -> tuple[ProfileScope, ...]:
        """Return canonical global and workspace scopes."""

        return self._repository.list_scopes(self._profile_id())

    def create_workspace_scope(self, workspace_id: str, label: str) -> ProfileScope:
        """Create a canonical scope only after workspace ownership is proven."""

        if not self._workspace_access(workspace_id):
            raise KeyError("Workspace not found")
        manifest = self._writable_manifest()
        now = self._now()
        scope = ProfileScope(
            scope_id=self._id_factory("scope"),
            profile_id=manifest.profile_id,
            kind=ScopeKind.WORKSPACE,
            version_id=self._id_factory("scope-version"),
            created_at=now,
            updated_at=now,
        )
        next_manifest = self._next_manifest(manifest)
        policy = WorkspaceRuntimePolicy(workspace_id=workspace_id, label=label)
        try:
            self._repository.commit_scope_and_manifest(
                scope,
                next_manifest,
                expected_scope_version=None,
                expected_manifest_version=manifest.current_version_id,
                runtime_policy=policy.model_dump(mode="json"),
                runtime_version_id=self._id_factory("runtime-version"),
            )
        except ConcurrentProfileUpdateError as exc:
            raise ProfileConflictError("Personal context profile changed") from exc
        return scope

    def workspace_id_for_scope(self, scope_id: str) -> str:
        """Return an owned workspace ID from its encrypted local mapping."""

        value = self._repository.get_runtime_policy(self._profile_id(), scope_id)
        if value is None:
            raise KeyError("Workspace scope not found")
        _version_id, payload = value
        try:
            policy = WorkspaceRuntimePolicy.model_validate(payload)
        except ValidationError:
            raise ProfileIntegrityError("Runtime policy validation failed") from None
        if not self._workspace_access(policy.workspace_id):
            raise KeyError("Workspace not found")
        return policy.workspace_id

    def _require_scope(self, profile_id: str, scope_id: str) -> ProfileScope:
        scope = self._repository.get_scope(profile_id, scope_id)
        if scope is None:
            raise KeyError("Personal context scope not found")
        return scope

    @staticmethod
    def _validate_server_controls(controls: ProfileControls) -> None:
        if controls.sync_mode is SyncMode.DEVICE_ONLY:
            raise ProfileUnsupportedOperationError("server_device_only_unsupported")

    def build_manual_record(
        self,
        *,
        scope_id: str,
        payload: ProfilePayload | Mapping[str, Any],
        semantic_key: SemanticKey | Mapping[str, Any] | None,
        controls: ProfileControls | Mapping[str, Any],
        expires_at: datetime | None = None,
        no_expiry: bool = False,
    ) -> ProfileRecord:
        """Build, but do not persist, one validated manual active record."""

        manifest = self._writable_manifest()
        profile_id = manifest.profile_id
        self._require_scope(profile_id, scope_id)
        parsed_payload = _PayloadAdapter.validate_python(payload)
        parsed_controls = _ControlsAdapter.validate_python(controls)
        parsed_semantic_key = None if semantic_key is None else _SemanticKeyAdapter.validate_python(semantic_key)
        self._validate_server_controls(parsed_controls)
        now = self._now()
        return ProfileRecord(
            profile_id=profile_id,
            record_id=self._id_factory("record"),
            scope_id=scope_id,
            kind=RecordKind(parsed_payload.kind),
            payload=parsed_payload,
            semantic_key=parsed_semantic_key,
            state=RecordState.ACTIVE,
            controls=parsed_controls,
            provenance=ProfileProvenance(
                source=ProvenanceSource.MANUAL,
                actor=ActorType.USER,
                reason_code="manual_edit",
            ),
            version_id=self._id_factory("record-version"),
            parent_version_id=None,
            created_at=now,
            updated_at=now,
            expires_at=expires_at,
            no_expiry=no_expiry,
        )

    def _is_expired(self, record: ProfileRecord) -> bool:
        return record.expires_at is not None and record.expires_at <= self._now()

    def _ensure_semantic_key_available(
        self,
        record: ProfileRecord,
        *,
        excluding_record_id: str | None = None,
    ) -> None:
        if record.semantic_key is None or record.state is not RecordState.ACTIVE:
            return
        for existing in self._repository.list_records(record.profile_id):
            if (
                existing.record_id != excluding_record_id
                and existing.scope_id == record.scope_id
                and existing.kind is record.kind
                and existing.semantic_key == record.semantic_key
                and existing.state is RecordState.ACTIVE
                and not self._is_expired(existing)
            ):
                raise ProfileKeyCollisionError("Active semantic key already exists")

    def create_record(self, record: ProfileRecord) -> ProfileRecord:
        """Persist one already validated record through the mutation fence."""

        manifest = self._writable_manifest()
        if record.profile_id != manifest.profile_id:
            raise KeyError("Personal context profile not found")
        self._require_scope(manifest.profile_id, record.scope_id)
        self._validate_server_controls(record.controls)
        self._ensure_semantic_key_available(record)
        next_manifest = self._next_manifest(manifest)
        try:
            self._repository.commit_record_and_manifest(
                record,
                next_manifest,
                expected_record_version=None,
                expected_manifest_version=manifest.current_version_id,
            )
        except ProfileSemanticKeyCollisionError as exc:
            raise ProfileKeyCollisionError("Active semantic key already exists") from exc
        except ConcurrentProfileUpdateError as exc:
            raise ProfileConflictError("Personal context profile changed") from exc
        return record

    def create_manual_record(self, **values: Any) -> ProfileRecord:
        """Build and persist one manual active record."""

        return self.create_record(self.build_manual_record(**values))

    def get_record(self, record_id: str) -> ProfileRecord:
        """Return one exact-user canonical record or a uniform not-found error."""

        record = self._repository.get_record(self._profile_id(), record_id)
        if record is None or record.state is RecordState.DELETED:
            raise KeyError("Personal context record not found")
        return record

    def list_records(self, *, include_archived: bool = False) -> tuple[ProfileRecord, ...]:
        """Return readable records, excluding tombstones and expired content."""

        records = self._repository.list_records(self._profile_id())
        return tuple(
            record
            for record in records
            if record.state is not RecordState.DELETED
            and not self._is_expired(record)
            and (include_archived or record.state is RecordState.ACTIVE)
        )

    def search_records(self, query: str, *, limit: int = 5) -> tuple[ProfileRecord, ...]:
        """Search a bounded user-owned projection without indexing plaintext."""

        if not 1 <= limit <= 20:
            raise ValueError("search limit must be between 1 and 20")
        needle = query.casefold().strip()
        if not needle:
            raise ValueError("search query must not be blank")
        matches = []
        for record in self.list_records():
            haystack = json.dumps(
                record.payload.model_dump(mode="json"),
                ensure_ascii=False,
                sort_keys=True,
            ).casefold()
            if needle in haystack:
                matches.append(record)
                if len(matches) == limit:
                    break
        return tuple(matches)

    def _replacement_record(
        self,
        current: ProfileRecord,
        mutation: RecordMutation,
        *,
        state: RecordState | None = None,
    ) -> ProfileRecord:
        payload = current.payload if mutation.payload is _UNSET else _PayloadAdapter.validate_python(mutation.payload)
        if payload is not None and RecordKind(payload.kind) is not current.kind:
            raise ValueError("record kind is immutable")
        semantic_key = (
            current.semantic_key
            if mutation.semantic_key is _UNSET
            else (None if mutation.semantic_key is None else _SemanticKeyAdapter.validate_python(mutation.semantic_key))
        )
        controls = (
            current.controls if mutation.controls is _UNSET else _ControlsAdapter.validate_python(mutation.controls)
        )
        self._validate_server_controls(controls)
        expires_at = current.expires_at if mutation.expires_at is _UNSET else mutation.expires_at
        no_expiry = current.no_expiry if mutation.no_expiry is _UNSET else mutation.no_expiry
        target_state = state or current.state
        if target_state is RecordState.DELETED:
            payload = None
            semantic_key = None
            expires_at = None
            no_expiry = False
        kind = current.kind if payload is None else RecordKind(payload.kind)
        return ProfileRecord.model_validate(
            {
                **current.model_dump(mode="python"),
                "kind": kind,
                "payload": payload,
                "semantic_key": semantic_key,
                "state": target_state,
                "controls": controls,
                "provenance": ProfileProvenance(
                    source=ProvenanceSource.MANUAL,
                    actor=ActorType.USER,
                    reason_code="manual_edit",
                    derived_from_record_id=current.record_id,
                ),
                "version_id": self._id_factory("record-version"),
                "parent_version_id": current.version_id,
                "updated_at": self._now(),
                "expires_at": expires_at,
                "no_expiry": no_expiry,
            }
        )

    def _commit_replacement(
        self,
        current: ProfileRecord,
        replacement: ProfileRecord,
        *,
        expected_version_id: str,
    ) -> ProfileRecord:
        if current.version_id != expected_version_id:
            raise ProfileConflictError("Personal context record changed")
        manifest = self._writable_manifest()
        self._ensure_semantic_key_available(
            replacement,
            excluding_record_id=current.record_id,
        )
        try:
            self._repository.commit_record_and_manifest(
                replacement,
                self._next_manifest(manifest),
                expected_record_version=expected_version_id,
                expected_manifest_version=manifest.current_version_id,
            )
        except ProfileSemanticKeyCollisionError as exc:
            raise ProfileKeyCollisionError("Active semantic key already exists") from exc
        except ConcurrentProfileUpdateError as exc:
            raise ProfileConflictError("Personal context record changed") from exc
        return replacement

    def update_record(
        self,
        record_id: str,
        mutation: RecordMutation,
        *,
        expected_version_id: str,
    ) -> ProfileRecord:
        """Update one record with optimistic concurrency."""

        self._writable_manifest()
        current = self.get_record(record_id)
        if current.state is not RecordState.ACTIVE:
            raise ValueError("only active records can be updated")
        replacement = self._replacement_record(current, mutation)
        return self._commit_replacement(
            current,
            replacement,
            expected_version_id=expected_version_id,
        )

    def archive_record(self, record_id: str, *, expected_version_id: str) -> ProfileRecord:
        """Archive one active record."""

        self._writable_manifest()
        current = self.get_record(record_id)
        if current.state is not RecordState.ACTIVE:
            raise ValueError("only active records can be archived")
        replacement = self._replacement_record(
            current,
            RecordMutation(),
            state=RecordState.ARCHIVED,
        )
        return self._commit_replacement(
            current,
            replacement,
            expected_version_id=expected_version_id,
        )

    def restore_record(self, record_id: str, *, expected_version_id: str) -> ProfileRecord:
        """Restore one archived record after semantic-key validation."""

        self._writable_manifest()
        current = self.get_record(record_id)
        if current.state is not RecordState.ARCHIVED:
            raise ValueError("only archived records can be restored")
        replacement = self._replacement_record(
            current,
            RecordMutation(),
            state=RecordState.ACTIVE,
        )
        return self._commit_replacement(
            current,
            replacement,
            expected_version_id=expected_version_id,
        )

    def delete_record(self, record_id: str, *, expected_version_id: str) -> ProfileRecord:
        """Replace one record with a content-free tombstone."""

        self._writable_manifest()
        current = self.get_record(record_id)
        replacement = self._replacement_record(
            current,
            RecordMutation(),
            state=RecordState.DELETED,
        )
        return self._commit_replacement(
            current,
            replacement,
            expected_version_id=expected_version_id,
        )

    def create_proposal(self, proposal: ProfileProposal) -> ProfileProposal:
        """Persist one validated pending agent proposal for later review."""

        manifest = self._writable_manifest()
        profile_id = manifest.profile_id
        self._current_proposals(profile_id)
        if proposal.profile_id != profile_id:
            raise KeyError("Personal context profile not found")
        self._require_scope(profile_id, proposal.scope_id)
        if proposal.expires_at <= self._now():
            raise ValueError("proposal has expired")
        if proposal.proposed_record is not None:
            self._validate_server_controls(proposal.proposed_record.controls)
        try:
            self._repository.commit_proposal(
                proposal,
                expected_manifest_version=manifest.current_version_id,
            )
        except ConcurrentProfileUpdateError as exc:
            raise ProfileConflictError("Personal context proposal changed") from exc
        return proposal

    def _current_proposals(self, profile_id: str) -> tuple[ProfileProposal, ...]:
        proposals = self._repository.list_unresolved_proposals(profile_id)
        now = self._now()
        expired = tuple(
            proposal for proposal in proposals if proposal.state is ProposalState.PENDING and proposal.expires_at <= now
        )
        for proposal in expired:
            try:
                self._repository.resolve_proposal(
                    profile_id,
                    proposal.proposal_id,
                    ProposalState.EXPIRED,
                )
            except (ConcurrentProfileUpdateError, ValueError):
                continue
        return self._repository.list_unresolved_proposals(profile_id) if expired else proposals

    def _has_live_pending_proposal(
        self,
        profile_id: str,
        *,
        now: datetime,
    ) -> bool:
        """Check review status without mutating expired proposal heads."""

        return any(
            proposal.state is ProposalState.PENDING and proposal.expires_at > now
            for proposal in self._repository.list_unresolved_proposals(profile_id)
        )

    def list_proposals(
        self,
        *,
        pending_only: bool = True,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[ProfileProposal, ...]:
        """Return proposal heads, optionally including content-free receipts."""

        if not 1 <= limit <= 200 or not 0 <= offset <= 1_000:
            raise ValueError("proposal page is out of bounds")
        profile_id = self._profile_id()
        proposals = self._current_proposals(profile_id)
        if not pending_only:
            return self._repository.list_proposals(
                profile_id,
                limit=limit,
                offset=offset,
            )
        now = self._now()
        pending = tuple(
            proposal for proposal in proposals if proposal.state is ProposalState.PENDING and proposal.expires_at > now
        )
        return pending[offset : offset + limit]

    def review_proposal(
        self,
        proposal_id: str,
        *,
        action: str,
    ) -> tuple[ProfileProposal, ProfileRecord | None]:
        """Accept or reject a pending proposal and shred its proposal body."""

        manifest = self._writable_manifest()
        profile_id = manifest.profile_id
        proposal = self._repository.get_proposal(profile_id, proposal_id)
        if proposal is None:
            raise KeyError("Personal context proposal not found")
        if action not in {"accept", "reject"}:
            raise ValueError("proposal action must be accept or reject")
        if proposal.state is not ProposalState.PENDING:
            raise ValueError("only pending proposals may be reviewed")
        if proposal.expires_at <= self._now():
            self._repository.resolve_proposal(
                profile_id,
                proposal_id,
                ProposalState.EXPIRED,
            )
            raise ValueError("proposal has expired")
        if action == "reject":
            return self._repository.reject_proposal(profile_id, proposal_id), None
        current: ProfileRecord | None = None
        if proposal.operation is ProposalOperation.CREATE:
            record = proposal.proposed_record
            if record is None:
                raise ProfileIntegrityError("Create proposal content is unavailable")
        else:
            if proposal.target_record_id is None or proposal.base_version_id is None:
                raise ProfileIntegrityError("Proposal target is unavailable")
            current = self.get_record(proposal.target_record_id)
            if current.version_id != proposal.base_version_id:
                raise ProfileConflictError("Proposal target changed")
            if current.scope_id != proposal.scope_id:
                raise ProfileConflictError("Proposal target scope changed")
            if proposal.operation is ProposalOperation.UPDATE:
                if current.state is not RecordState.ACTIVE:
                    raise ProfileConflictError("Proposal target is not active")
                record = proposal.proposed_record
                if record is None:
                    raise ProfileIntegrityError("Update proposal content is unavailable")
                if record.scope_id != current.scope_id:
                    raise ProfileConflictError("Proposal cannot move its target")
                if record.kind is not current.kind:
                    raise ProfileConflictError("Proposal cannot change record kind")
                if record.created_at != current.created_at:
                    raise ProfileConflictError("Proposal cannot change record creation time")
                if record.updated_at < current.updated_at:
                    raise ProfileConflictError("Proposal record is older than its target")
                if record.controls != current.controls:
                    raise ProfileConflictError("Proposal cannot change user controls")
            elif proposal.operation is ProposalOperation.ARCHIVE:
                if current.state is not RecordState.ACTIVE:
                    raise ProfileConflictError("Proposal target is not active")
                record = self._replacement_record(
                    current,
                    RecordMutation(),
                    state=RecordState.ARCHIVED,
                )
            else:
                raise ProfileUnsupportedOperationError("server_proposal_operation_unsupported")
        self._validate_server_controls(record.controls)
        self._ensure_semantic_key_available(
            record,
            excluding_record_id=None if current is None else current.record_id,
        )
        expected_record_version = None if current is None else current.version_id
        try:
            receipt = self._repository.accept_proposal_and_record(
                profile_id,
                proposal_id,
                record,
                self._next_manifest(manifest),
                expected_record_version=expected_record_version,
                expected_manifest_version=manifest.current_version_id,
            )
        except ProfileSemanticKeyCollisionError as exc:
            raise ProfileKeyCollisionError("Active semantic key already exists") from exc
        except ConcurrentProfileUpdateError as exc:
            raise ProfileConflictError("Proposal target changed") from exc
        if receipt.state is ProposalState.EXPIRED:
            raise ValueError("proposal has expired")
        return receipt, record

    def get_runtime_policy(self) -> RuntimePolicyVersion:
        """Return the server-local profile enablement policy and version."""

        profile_id = self._profile_id()
        value = self._repository.get_runtime_policy(
            profile_id,
            PROFILE_RUNTIME_POLICY_ID,
        )
        if value is None:
            return RuntimePolicyVersion(version_id=None, enabled=False)
        version_id, payload = value
        try:
            policy = ServerRuntimePolicy.model_validate(payload)
        except ValidationError:
            raise ProfileIntegrityError("Runtime policy validation failed") from None
        return RuntimePolicyVersion(version_id=version_id, enabled=policy.enabled)

    def set_runtime_enabled(
        self,
        enabled: bool,
        *,
        expected_version_id: str | None,
    ) -> RuntimePolicyVersion:
        """Set server-local agent profile use with optimistic concurrency."""

        manifest = self._writable_manifest()
        profile_id = manifest.profile_id
        version_id = self._id_factory("runtime-version")
        policy = ServerRuntimePolicy(enabled=enabled)
        try:
            self._repository.set_runtime_policy(
                profile_id,
                PROFILE_RUNTIME_POLICY_ID,
                version_id=version_id,
                expected_version_id=expected_version_id,
                expected_manifest_version=manifest.current_version_id,
                policy=policy.model_dump(mode="json"),
            )
        except ConcurrentProfileUpdateError as exc:
            raise ProfileConflictError("Personal context runtime policy changed") from exc
        return RuntimePolicyVersion(version_id=version_id, enabled=enabled)

    def _export_snapshot(
        self,
        *,
        scope_ids: tuple[str, ...] | None = None,
    ) -> dict[str, Any]:
        manifest = self._manifest()
        scopes = self.list_scopes()
        selected = {scope.scope_id for scope in scopes}
        if scope_ids is not None:
            requested = set(scope_ids)
            if not requested.issubset(selected):
                raise KeyError("Personal context scope not found")
            selected = requested
        records = tuple(
            record
            for record in self._repository.list_records(manifest.profile_id)
            if record.scope_id in selected
        )
        return {
            "schema_version": 1,
            "manifest": manifest.model_dump(mode="json"),
            "scopes": [scope.model_dump(mode="json") for scope in scopes if scope.scope_id in selected],
            "records": [record.model_dump(mode="json") for record in records],
        }

    def export_plaintext(
        self,
        *,
        confirmation: str,
        scope_ids: tuple[str, ...] | None = None,
    ) -> dict[str, Any]:
        """Return an explicitly confirmed human-readable canonical snapshot."""

        require_confirmation(confirmation, PLAINTEXT_EXPORT_CONFIRMATION)
        return self._export_snapshot(scope_ids=scope_ids)

    def export_recovery(
        self,
        *,
        confirmation: str,
        passphrase: str,
    ) -> dict[str, str]:
        """Return an explicitly confirmed passphrase-encrypted recovery snapshot."""

        require_confirmation(confirmation, RECOVERY_EXPORT_CONFIRMATION)
        return encrypt_recovery_export(
            self._export_snapshot(),
            passphrase=passphrase,
        )

    def purge_profile(
        self,
        *,
        mode: str,
        confirmation: str,
        expected_purge_generation: int,
    ) -> ProfileManifest:
        """Advance the global purge barrier or refuse device-local deletion."""

        if mode == "local_copy":
            raise ProfileUnsupportedOperationError("server_local_copy_unsupported")
        if mode != "everywhere":
            raise ValueError("purge mode must be everywhere")
        if confirmation != "DELETE EVERYWHERE":
            raise ValueError("confirmation must be exactly 'DELETE EVERYWHERE'")
        manifest = self._writable_manifest()
        if manifest.purge_generation != expected_purge_generation:
            raise ProfileConflictError("Personal context purge generation changed")
        next_manifest = self._next_manifest(manifest)
        barrier = ProfileManifest.model_validate(
            {
                **next_manifest.model_dump(mode="python"),
                "purge_generation": manifest.purge_generation + 1,
            }
        )
        try:
            self._repository.purge_profile(
                barrier,
                expected_manifest_version=manifest.current_version_id,
            )
        except ConcurrentProfileUpdateError as exc:
            raise ProfileConflictError("Personal context profile changed") from exc
        return barrier
