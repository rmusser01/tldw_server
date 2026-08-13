"""Owner-bound coordination for canonical Notes attachment mutations."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal

from tldw_Server_API.app.core.DB_Management.chacha.note_attachment_store import (
    NoteAttachment,
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.exceptions import (
    NotesAttachmentMutationError,
    NotesAttachmentSyncNotReadyError,
)
from tldw_Server_API.app.core.Notes.attachment_policy import (
    NOTE_ATTACHMENT_MAX_FILENAME_LEN,
    canonicalize_note_attachment_file_name,
)

from .adapters import (
    AdapterAccepted,
    AdapterConflict,
    AdapterDeferred,
    AdapterRejected,
    AttachmentRefAdapter,
    SyncAdapterContext,
)
from .attachment_refs_v2 import (
    AttachmentRefV2Payload,
    attachment_ref_v2_object_hash,
    parse_attachment_ref_v2_payload,
    validate_attachment_ref_v2_routing_metadata,
)
from .errors import (
    SyncIdempotencyConflictError,
)
from .models import (
    DEFAULT_M1_ENCRYPTION_POLICY,
    SyncDataset,
    SyncEnvelope,
    SyncEnvelopeCreate,
    sync_v2_attachment_ref_v2_is_writable,
)
from .server_origin import SERVER_ORIGIN_DEVICE_ID
from .service import SyncV2Service


@dataclass(frozen=True, slots=True)
class ReadyAttachmentDataset:
    """One owner-bound canonical dataset ready for attachment mutation."""

    owner_id: str
    dataset: SyncDataset


@dataclass(frozen=True, slots=True)
class NotesAttachmentMutationPlan:
    """Canonical input needed to capture one attachment-ref mutation."""

    owner_id: str
    operation: Literal["upsert", "tombstone"]
    attachment_id: str
    payload: Mapping[str, object]
    idempotency_key: str
    source: str
    dataset_id: str | None = None
    base_server_cursor: int | None = None
    base_object_revision: int | None = None
    base_object_hash: str | None = None
    routing_metadata: Mapping[str, object] = field(default_factory=dict)
    require_available_blob: bool = False
    allocate_unique_file_name: bool = False

    def __post_init__(self) -> None:
        """Validate and freeze the public mutation-plan boundaries."""

        if not isinstance(self.owner_id, str) or not self.owner_id.strip():
            raise ValueError("owner_id must be non-empty")
        if (
            not isinstance(self.idempotency_key, str)
            or not self.idempotency_key.strip()
            or self.idempotency_key != self.idempotency_key.strip()
            or not 1 <= len(self.idempotency_key.encode("utf-8")) <= 128
        ):
            raise ValueError("idempotency_key must be between 1 and 128 bytes")
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("source must be non-empty")
        base_values = (
            self.base_server_cursor,
            self.base_object_revision,
            self.base_object_hash,
        )
        if any(value is not None for value in base_values) and not all(
            value is not None for value in base_values
        ):
            raise ValueError("attachment mutation base tuple must be complete")
        if self.allocate_unique_file_name and (
            self.operation != "upsert" or any(value is not None for value in base_values)
        ):
            raise ValueError("unique filename allocation is valid only for create")
        object.__setattr__(self, "payload", MappingProxyType(dict(self.payload)))
        object.__setattr__(
            self,
            "routing_metadata",
            MappingProxyType(dict(self.routing_metadata)),
        )


@dataclass(frozen=True, slots=True)
class NotesAttachmentMutationResult:
    """Durable response derived from an applied envelope and registry row."""

    dataset: SyncDataset
    envelope: SyncEnvelope
    attachment: NoteAttachment
    idempotent_replay: bool


@dataclass(slots=True)
class NotesAttachmentCoordinator:
    """Validate, append, and project attachment mutations under one dataset guard."""

    service: SyncV2Service
    note_db: CharactersRAGDB

    def resolve_canonical_dataset(
        self,
        *,
        owner_id: str,
        dataset_id: str | None,
    ) -> ReadyAttachmentDataset | None:
        """Resolve the owner's sole canonical default-personal Notes dataset."""

        defaults = [
            dataset
            for dataset in self.service.store.list_datasets_for_user(owner_id)
            if dataset.archived_at is None
            and dataset.scope_type == "personal"
            and dataset.owner_user_id == owner_id
            and dataset.metadata.get("default_personal") is True
            and dataset.metadata.get("client_family") == "chatbook"
        ]
        if len(defaults) != 1:
            return None
        dataset = defaults[0]
        if dataset_id is not None and dataset_id != dataset.dataset_id:
            return None
        attachment_state = dataset.metadata.get("notes_attachment_v2")
        if (
            not isinstance(attachment_state, Mapping)
            or attachment_state.get("state") != "ready"
            or not {"notes.note", "attachment.ref"}.issubset(dataset.domains)
        ):
            return None
        return ReadyAttachmentDataset(owner_id=owner_id, dataset=dataset)

    def resolve_mutation_ready(
        self,
        *,
        owner_id: str,
        dataset_id: str | None,
    ) -> ReadyAttachmentDataset | None:
        """Resolve only the owner's active canonical default-personal dataset."""

        ready = self.resolve_canonical_dataset(
            owner_id=owner_id,
            dataset_id=dataset_id,
        )
        if ready is None:
            return None
        dataset = ready.dataset
        try:
            adapter = self.service.adapters.get("attachment.ref")
        except KeyError:
            return None
        if not isinstance(adapter, AttachmentRefAdapter):
            return None
        if not sync_v2_attachment_ref_v2_is_writable(
            dataset,
            notes_attachment_sync_enabled=adapter.v2_writes_enabled,
            supports_attachments=self.service.settings.supports_attachments,
        ):
            return None
        return ready

    def require_mutation_ready(
        self,
        *,
        owner_id: str,
        dataset_id: str | None,
    ) -> ReadyAttachmentDataset:
        """Return canonical mutation authority or fail without fallback."""

        ready = self.resolve_mutation_ready(
            owner_id=owner_id,
            dataset_id=dataset_id,
        )
        if ready is None:
            raise NotesAttachmentSyncNotReadyError(
                "Notes attachment Sync is not ready for canonical mutation"
            )
        return ready

    def replay_by_idempotency_key(
        self,
        *,
        owner_id: str,
        dataset_id: str | None,
        idempotency_key: str,
    ) -> NotesAttachmentMutationResult | None:
        """Return an already captured request without requiring its object ID."""

        ready = self.require_mutation_ready(
            owner_id=owner_id,
            dataset_id=dataset_id,
        )
        stable_key = _stable_key_from_idempotency(idempotency_key)
        envelopes = self.service.store.list_envelopes_for_entity(
            ready.dataset.dataset_id,
            "attachment.ref",
            stable_key=stable_key,
            limit=1,
        )
        if not envelopes:
            return None
        return self._resume_existing(ready.dataset, envelopes[0])

    def capture(
        self,
        plan: NotesAttachmentMutationPlan,
    ) -> NotesAttachmentMutationResult:
        """Capture an exact idempotent mutation and return only applied state."""

        ready = self.require_mutation_ready(
            owner_id=plan.owner_id,
            dataset_id=plan.dataset_id,
        )
        dataset = ready.dataset
        payload = parse_attachment_ref_v2_payload(plan.operation, plan.payload)
        if str(payload.attachment_id) != plan.attachment_id:
            raise NotesAttachmentMutationError(
                "Attachment mutation identity does not match its payload"
            )
        routing_metadata = validate_attachment_ref_v2_routing_metadata(
            plan.operation,
            plan.routing_metadata,
        )
        stable_key, client_envelope_id = _mutation_identity(plan, dataset)
        existing = _existing_request(
            self.service,
            dataset_id=dataset.dataset_id,
            stable_key=stable_key,
            client_envelope_id=client_envelope_id,
        )
        if existing is not None:
            if not _request_matches_existing(
                plan,
                dataset,
                payload,
                routing_metadata,
                existing,
            ):
                raise NotesAttachmentMutationError(
                    "Attachment mutation idempotency key was reused with different content"
                )
            return self._resume_existing(dataset, existing)

        note_before = self._require_owned_note(
            plan.owner_id,
            payload,
            allow_deleted=_allows_deleted_parent(plan, routing_metadata),
        )
        head = self.service.store.get_current_head(
            dataset.dataset_id,
            "attachment.ref",
            plan.attachment_id,
        )
        _require_requested_base(plan, head)
        object_revision = 1 if head is None else (head.object_revision or 0) + 1
        envelope = _build_envelope(
            plan=plan,
            dataset=dataset,
            payload=payload,
            routing_metadata=routing_metadata,
            client_envelope_id=client_envelope_id,
            stable_key=stable_key,
            object_revision=object_revision,
        )

        stored: SyncEnvelope | None = None
        with self.service.store.materialization_guard(
            [envelope],
            require_predecessors=False,
        ) as guarded_store:
            guarded_head = guarded_store.get_current_head(
                dataset.dataset_id,
                "attachment.ref",
                plan.attachment_id,
            )
            _require_requested_base(plan, guarded_head)
            if plan.allocate_unique_file_name:
                payload = payload.model_copy(
                    update={
                        "file_name": self._allocate_unique_file_name(
                            dataset_id=dataset.dataset_id,
                            note_id=str(payload.parent_object_id),
                            requested_file_name=payload.file_name,
                        )
                    }
                )
                envelope = _build_envelope(
                    plan=plan,
                    dataset=dataset,
                    payload=payload,
                    routing_metadata=routing_metadata,
                    client_envelope_id=client_envelope_id,
                    stable_key=stable_key,
                    object_revision=object_revision,
                )
            note_after = self._require_owned_note(
                plan.owner_id,
                payload,
                allow_deleted=_allows_deleted_parent(plan, routing_metadata),
            )
            if _note_read_identity(note_before) != _note_read_identity(note_after):
                raise NotesAttachmentMutationError(
                    "The parent note changed before attachment commit"
                )
            blob = guarded_store.get_blob_object(
                dataset.dataset_id,
                payload_hash=payload.blob_hash,
                owner_user_id=plan.owner_id,
            )
            if plan.require_available_blob and (
                blob is None
                or blob.status != "available"
                or blob.size_bytes != payload.size_bytes
                or blob.content_type != payload.content_type
            ):
                raise NotesAttachmentMutationError(
                    "Attachment mutation requires an exact verified available blob"
                )
            context = SyncAdapterContext(
                prior_envelopes=((guarded_head,) if guarded_head is not None else ()),
                get_head=lambda domain, object_id: guarded_store.get_current_head(
                    dataset.dataset_id,
                    domain,
                    object_id,
                ),
                list_heads=lambda domain: self.service._list_current_heads_for_adapter(
                    dataset.dataset_id,
                    domain,
                    store=guarded_store,
                ),
                supports_attachments=True,
            )
            outcome = self.service._evaluate_envelope(
                dataset,
                envelope,
                context=context,
            )
            _require_adapter_acceptance(outcome)
            try:
                stored = guarded_store.insert_envelope(envelope)
            except SyncIdempotencyConflictError as exc:
                raise NotesAttachmentMutationError(
                    "Attachment mutation idempotency key was reused with different content"
                ) from exc
            binding = guarded_store.get_attachment_revision_binding(
                dataset.dataset_id,
                plan.attachment_id,
                object_revision,
                owner_user_id=plan.owner_id,
            )
            if plan.require_available_blob and (
                binding is None
                or binding.availability_at_acceptance != "available"
                or binding.resolved_blob_id is None
            ):
                raise NotesAttachmentMutationError(
                    "Attachment mutation did not bind the verified blob"
                )
            self.service._materialize_envelope(stored, store=guarded_store)
            stored = self.service._envelope_snapshot(stored, store=guarded_store)

        if stored is None or stored.apply_status != "applied":
            raise NotesAttachmentMutationError(
                "Attachment mutation projection did not complete"
            )
        return self._result(dataset, stored, idempotent_replay=False)

    def _resume_existing(
        self,
        dataset: SyncDataset,
        envelope: SyncEnvelope,
    ) -> NotesAttachmentMutationResult:
        """Resume materialization for one exact stored mutation request."""

        if envelope.apply_status != "applied":
            self.service._materialize_envelope(envelope)
            envelope = self.service._envelope_snapshot(envelope)
        if envelope.apply_status != "applied":
            raise NotesAttachmentMutationError(
                "Attachment mutation projection did not complete"
            )
        return self._result(dataset, envelope, idempotent_replay=True)

    def _result(
        self,
        dataset: SyncDataset,
        envelope: SyncEnvelope,
        *,
        idempotent_replay: bool,
    ) -> NotesAttachmentMutationResult:
        """Build the durable coordinator result for an applied envelope."""

        attachment = self.note_db.note_attachment_store.get(
            dataset.dataset_id,
            envelope.object_id,
        )
        if attachment is None or attachment.object_hash != envelope.payload_hash:
            raise NotesAttachmentMutationError(
                "Attachment mutation response postcondition is unavailable"
            )
        return NotesAttachmentMutationResult(
            dataset=dataset,
            envelope=envelope,
            attachment=attachment,
            idempotent_replay=idempotent_replay,
        )

    def _allocate_unique_file_name(
        self,
        *,
        dataset_id: str,
        note_id: str,
        requested_file_name: str,
    ) -> str:
        """Allocate the ordinary bounded suffix while the dataset guard is held."""

        normalized_names: set[str] = set()
        after_attachment_id: str | None = None
        scanned_count = 0
        while scanned_count <= 1000:
            page_limit = min(200, 1001 - scanned_count)
            page = self.note_db.note_attachment_store.list_page(
                dataset_id,
                note_id,
                after_attachment_id=after_attachment_id,
                limit=page_limit,
                state="live",
            )
            scanned_count += len(page)
            if scanned_count > 1000:
                raise NotesAttachmentMutationError(
                    "Attachment filename allocation exceeds its bounded search"
                )
            normalized_names.update(item.normalized_file_name for item in page)
            if len(page) < page_limit:
                break
            after_attachment_id = page[-1].attachment_id
        else:
            raise NotesAttachmentMutationError(
                "Attachment filename allocation exceeds its bounded search"
            )

        requested_display, requested_normalized = canonicalize_note_attachment_file_name(
            requested_file_name
        )
        if requested_normalized not in normalized_names:
            return requested_display
        suffixes = requested_display.split(".")
        extension = f".{suffixes[-1]}" if len(suffixes) > 1 else ""
        stem = requested_display[: -len(extension)] if extension else requested_display
        for index in range(1, 1000):
            suffix = f"-{index}"
            stem_limit = max(
                1,
                NOTE_ATTACHMENT_MAX_FILENAME_LEN - len(extension) - len(suffix),
            )
            candidate, normalized = canonicalize_note_attachment_file_name(
                f"{stem[:stem_limit]}{suffix}{extension}"
            )
            if normalized not in normalized_names:
                return candidate
        raise NotesAttachmentMutationError(
            "Attachment filename allocation exhausted its suffix boundary"
        )

    def _require_owned_note(
        self,
        owner_id: str,
        payload: AttachmentRefV2Payload,
        *,
        allow_deleted: bool,
    ) -> dict[str, Any]:
        """Return the owner-authorized parent note in the required state."""

        note = self.note_db.note_store.get_note_by_id(
            str(payload.parent_object_id),
            include_deleted=True,
        )
        if (
            note is None
            or str(note.get("client_id")) != owner_id
            or (bool(note.get("deleted")) and not allow_deleted)
        ):
            raise NotesAttachmentMutationError(
                "Attachment mutation requires an existing owned parent note"
            )
        return note


def _allows_deleted_parent(
    plan: NotesAttachmentMutationPlan,
    routing_metadata: Mapping[str, object],
) -> bool:
    """Return whether the mutation may target an owned deleted parent."""

    return plan.operation == "tombstone" or routing_metadata.get("restore_intent") is True


def _note_read_identity(note: Mapping[str, object]) -> tuple[object, ...]:
    """Return the guarded note fields that define attachment authorization."""

    return (
        note.get("id"),
        note.get("client_id"),
        note.get("version"),
        bool(note.get("deleted")),
    )


def _require_requested_base(
    plan: NotesAttachmentMutationPlan,
    head: SyncEnvelope | None,
) -> None:
    """Require the caller's optimistic base to match the current head."""

    requested = (
        plan.base_server_cursor,
        plan.base_object_revision,
        plan.base_object_hash,
    )
    if head is None:
        if any(value is not None for value in requested):
            raise NotesAttachmentMutationError(
                "Attachment mutation base does not match the current head"
            )
        if plan.operation != "upsert" or plan.routing_metadata.get("restore_intent") is True:
            raise NotesAttachmentMutationError(
                "Attachment mutation requires an existing exact base"
            )
        return
    current = (
        head.server_cursor,
        head.object_revision,
        head.payload_hash,
    )
    if requested != current:
        raise NotesAttachmentMutationError(
            "Attachment mutation base does not match the current head"
        )


def _mutation_identity(
    plan: NotesAttachmentMutationPlan,
    dataset: SyncDataset,
) -> tuple[str, str]:
    """Derive stable manifest and envelope identities for one request."""

    stable_key = _stable_key_from_idempotency(plan.idempotency_key)
    key_hash = stable_key.removeprefix("notes-attachment:")
    envelope_hash = hashlib.sha256(
        f"{dataset.dataset_id}:notes-attachment:{key_hash}".encode()
    ).hexdigest()
    return (
        stable_key,
        f"server-origin-{envelope_hash[:32]}",
    )


def _stable_key_from_idempotency(idempotency_key: str) -> str:
    """Derive the privacy-safe stable manifest key for a public request key."""

    key_hash = hashlib.sha256(idempotency_key.encode("utf-8")).hexdigest()
    return f"notes-attachment:{key_hash}"


def _build_envelope(
    *,
    plan: NotesAttachmentMutationPlan,
    dataset: SyncDataset,
    payload: AttachmentRefV2Payload,
    routing_metadata: Mapping[str, object],
    client_envelope_id: str,
    stable_key: str,
    object_revision: int,
) -> SyncEnvelopeCreate:
    """Build the canonical server-origin attachment envelope."""

    payload_hash = attachment_ref_v2_object_hash(
        plan.operation,
        payload,
        object_revision=object_revision,
    )
    encoded_payload = json.dumps(
        payload.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return SyncEnvelopeCreate(
        dataset_id=dataset.dataset_id,
        client_envelope_id=client_envelope_id,
        domain="attachment.ref",
        operation=plan.operation,
        object_id=plan.attachment_id,
        device_id=SERVER_ORIGIN_DEVICE_ID,
        base_server_cursor=plan.base_server_cursor,
        base_object_revision=plan.base_object_revision,
        base_object_hash=plan.base_object_hash,
        object_revision=object_revision,
        parent_id=str(payload.parent_object_id),
        schema_version=2,
        adapter_version=2,
        payload=payload.model_dump(mode="json"),
        payload_hash=payload_hash,
        payload_size_bytes=len(encoded_payload),
        created_at_client=payload.last_modified,
        deleted=plan.operation == "tombstone",
        encryption_metadata={"policy": DEFAULT_M1_ENCRYPTION_POLICY},
        routing_metadata=dict(routing_metadata),
        stable_key=stable_key,
    )


def _request_matches_existing(
    plan: NotesAttachmentMutationPlan,
    dataset: SyncDataset,
    payload: AttachmentRefV2Payload,
    routing_metadata: Mapping[str, object],
    envelope: SyncEnvelope,
) -> bool:
    """Return whether a stored envelope is the exact submitted request."""

    requested_payload = payload.model_dump(mode="json")
    existing_payload = dict(envelope.payload)
    for generated_field in ("created_at", "last_modified", "deleted_at"):
        requested_payload.pop(generated_field, None)
        existing_payload.pop(generated_field, None)
    if plan.allocate_unique_file_name:
        requested_name = requested_payload.pop("file_name", None)
        existing_name = existing_payload.pop("file_name", None)
        if not _is_allocated_name_for_request(requested_name, existing_name):
            return False
    return {
        "dataset_id": dataset.dataset_id,
        "operation": plan.operation,
        "attachment_id": plan.attachment_id,
        "payload": requested_payload,
        "base": (
            plan.base_server_cursor,
            plan.base_object_revision,
            plan.base_object_hash,
        ),
        "routing_metadata": dict(routing_metadata),
    } == {
        "dataset_id": envelope.dataset_id,
        "operation": envelope.operation,
        "attachment_id": envelope.object_id,
        "payload": existing_payload,
        "base": (
            envelope.base_server_cursor,
            envelope.base_object_revision,
            envelope.base_object_hash,
        ),
        "routing_metadata": dict(envelope.routing_metadata),
    }


def _is_allocated_name_for_request(requested: object, allocated: object) -> bool:
    """Return whether an allocated compatibility suffix belongs to the request."""

    if not isinstance(requested, str) or not isinstance(allocated, str):
        return False
    if requested == allocated:
        return True
    suffixes = requested.split(".")
    extension = f".{suffixes[-1]}" if len(suffixes) > 1 else ""
    requested_stem = requested[: -len(extension)] if extension else requested
    allocated_stem = allocated[: -len(extension)] if extension else allocated
    if extension and not allocated.endswith(extension):
        return False
    prefix, separator, raw_index = allocated_stem.rpartition("-")
    if separator != "-" or not raw_index.isdecimal():
        return False
    index = int(raw_index)
    if not 1 <= index < 1000:
        return False
    suffix = f"-{index}"
    stem_limit = max(
        1,
        NOTE_ATTACHMENT_MAX_FILENAME_LEN - len(extension) - len(suffix),
    )
    expected, _ = canonicalize_note_attachment_file_name(
        f"{requested_stem[:stem_limit]}{suffix}{extension}"
    )
    return allocated == expected


def _existing_request(
    service: SyncV2Service,
    *,
    dataset_id: str,
    stable_key: str,
    client_envelope_id: str,
) -> SyncEnvelope | None:
    """Load the exact prior request by manifest key or envelope identity."""

    for envelope in service.store.list_envelopes_for_entity(
        dataset_id,
        "attachment.ref",
        stable_key=stable_key,
        limit=1,
    ):
        if envelope.client_envelope_id == client_envelope_id:
            return envelope
        raise NotesAttachmentMutationError(
            "Attachment mutation idempotency key was reused with different content"
        )
    return None


def _require_adapter_acceptance(
    outcome: AdapterAccepted | AdapterRejected | AdapterDeferred | AdapterConflict,
) -> None:
    """Require adapter acceptance and map all other outcomes to failure."""

    if isinstance(outcome, AdapterAccepted):
        return
    if isinstance(outcome, AdapterRejected | AdapterDeferred | AdapterConflict):
        raise NotesAttachmentMutationError(
            outcome.message or "Attachment mutation was not accepted"
        )
    raise NotesAttachmentMutationError("Attachment mutation was not accepted")


__all__ = [
    "NotesAttachmentCoordinator",
    "NotesAttachmentMutationError",
    "NotesAttachmentMutationPlan",
    "NotesAttachmentMutationResult",
    "NotesAttachmentSyncNotReadyError",
    "ReadyAttachmentDataset",
]
