from __future__ import annotations

"""Strict Sync v2 adapters for Notes organization resources and links."""

from dataclasses import dataclass, field
from typing import cast

from ..adapters import (
    AdapterAccepted,
    AdapterConflict,
    AdapterRejected,
    SyncAdapterContext,
    SyncAdapterOutcome,
    SyncHead,
)
from ..models import (
    NOTES_ORGANIZATION_DOMAINS,
    SyncDataset,
    SyncDomain,
    SyncEnvelopeCreate,
)
from ..notes_organization import (
    NotesOrganizationValidationError,
    parse_notes_organization_payload,
    validate_organization_object_id,
)
from ._lineage import current_head, prior_envelopes

_RESOURCE_DOMAINS = frozenset(
    {"notes.keyword", "notes.keyword_collection", "notes.folder"}
)
_RELATIONSHIP_DOMAINS = frozenset(set(NOTES_ORGANIZATION_DOMAINS) - _RESOURCE_DOMAINS)
_HIERARCHY_DOMAINS = frozenset({"notes.keyword_collection", "notes.folder"})


@dataclass(slots=True)
class NotesOrganizationDomainAdapter:
    """Evaluate one of the six Notes organization domains without product writes."""

    domain: SyncDomain
    supported_adapter_versions: set[int] = field(default_factory=lambda: {1})

    def __post_init__(self) -> None:
        if self.domain not in NOTES_ORGANIZATION_DOMAINS:
            raise ValueError(f"Unsupported Notes organization adapter domain: {self.domain}")

    def evaluate_envelope(
        self,
        envelope: SyncEnvelopeCreate,
        *,
        dataset: SyncDataset,
        context: SyncAdapterContext | None = None,
    ) -> SyncAdapterOutcome:
        """Return a stable validation, conflict, or acceptance outcome."""

        if envelope.domain != self.domain or envelope.schema_version != 1:
            return _rejected(
                envelope,
                "notes_organization_payload_invalid",
                "Notes organization envelope domain or schema version is invalid",
            )
        try:
            payload = parse_notes_organization_payload(
                self.domain, envelope.operation, envelope.payload
            )
            validate_organization_object_id(self.domain, envelope.object_id, payload)
            _validate_folder_segment(self.domain, envelope.operation, payload)
        except NotesOrganizationValidationError as exc:
            code = (
                "notes_organization_identity_mismatch"
                if exc.error_code
                in {
                    "notes_organization_resource_sync_id_invalid",
                    "notes_organization_link_identity_invalid",
                }
                else exc.error_code
            )
            return _rejected(envelope, code, str(exc))

        bootstrap_capture = envelope.routing_metadata.get("bootstrap_capture")
        if bootstrap_capture not in {None, True}:
            return _rejected(
                envelope,
                "notes_organization_payload_invalid",
                "bootstrap_capture must be the boolean true when supplied",
            )
        if bootstrap_capture is True and envelope.domain not in _RELATIONSHIP_DOMAINS:
            return _rejected(
                envelope,
                "notes_organization_payload_invalid",
                "bootstrap_capture is valid only for relationship domains",
            )
        bootstrap_authorized = bootstrap_capture is True and _bootstrap_authorized(
            envelope, payload, dataset, context
        )
        readiness = _readiness_error(
            dataset,
            bootstrap_authorized=(
                bootstrap_authorized or _trusted_bootstrap_context(dataset, context)
            ),
        )
        if readiness is not None:
            return _rejected(envelope, "notes_organization_domain_not_ready", readiness)
        if bootstrap_capture is True and not bootstrap_authorized:
            return _rejected(
                envelope,
                "notes_organization_domain_not_ready",
                "Dormant relationship bootstrap was not structurally authorized",
            )

        head = _get_head(envelope, context)
        literal_replay = head is not None and _literal_replay(head, envelope)
        equivalent = head is not None and _equivalent_state(head, envelope.operation, payload)
        if head is None and _has_base(envelope):
            return _base_conflict(envelope, "The referenced base head does not exist")
        if head is not None and not literal_replay and not _exact_base(envelope, head):
            return _base_conflict(envelope, "The incoming base does not match the current head")
        if literal_replay:
            return AdapterAccepted(client_envelope_id=envelope.client_envelope_id)

        restore_intent = envelope.routing_metadata.get("restore_intent")
        if restore_intent not in {None, True} or (
            restore_intent is True and envelope.operation != "upsert"
        ):
            return _rejected(
                envelope,
                "notes_organization_payload_invalid",
                "restore_intent must be the boolean true on an upsert",
            )
        if head is not None and _is_deleted(head) != (envelope.operation == "tombstone"):
            if envelope.operation == "upsert" and restore_intent is not True:
                return _base_conflict(envelope, "Restore requires explicit restore intent")
        if restore_intent is True and (head is None or not _is_deleted(head) or not _exact_base(envelope, head)):
            return _base_conflict(envelope, "Restore requires the exact current tombstone head")

        if equivalent and envelope.operation == "tombstone":
            return AdapterAccepted(client_envelope_id=envelope.client_envelope_id)

        if (
            envelope.operation == "upsert"
            and envelope.domain in _HIERARCHY_DOMAINS
            and payload.get("parent_sync_id") == envelope.object_id
        ):
            return _conflict(
                envelope,
                "notes_organization_hierarchy_cycle",
                "An organization resource cannot parent itself",
            )

        dependency_error = _validate_dependencies(
            envelope,
            payload,
            dataset=dataset,
            context=context,
            allow_deleted=bootstrap_authorized,
        )
        if dependency_error is not None:
            return dependency_error

        if envelope.operation == "upsert":
            semantic_conflict = _validate_resource_semantics(
                envelope, payload, context=context
            )
            if semantic_conflict is not None:
                return semantic_conflict

        return AdapterAccepted(client_envelope_id=envelope.client_envelope_id)


def _readiness_error(dataset: SyncDataset, *, bootstrap_authorized: bool) -> str | None:
    if set(NOTES_ORGANIZATION_DOMAINS).difference(dataset.domains):
        return "The complete Notes organization domain group is not enrolled"
    metadata = dataset.metadata.get("notes_organization_v1")
    state = metadata.get("state") if isinstance(metadata, dict) else None
    if state == "ready" or (state == "initializing" and bootstrap_authorized):
        return None
    return "The Notes organization domain group is not ready"


def _bootstrap_authorized(
    envelope: SyncEnvelopeCreate,
    payload: dict[str, object],
    dataset: SyncDataset,
    context: SyncAdapterContext | None,
) -> bool:
    metadata = dataset.metadata.get("notes_organization_v1")
    bootstrap_id = metadata.get("bootstrap_id") if isinstance(metadata, dict) else None
    return bool(
        context is not None
        and context.trusted_server_origin
        and context.organization_group_state == "initializing"
        and bootstrap_id is not None
        and context.organization_bootstrap_id == bootstrap_id
        and context.bootstrap_relationship_verifier is not None
        and context.bootstrap_relationship_verifier(
            envelope.domain, envelope.object_id, payload
        )
    )


def _trusted_bootstrap_context(
    dataset: SyncDataset,
    context: SyncAdapterContext | None,
) -> bool:
    metadata = dataset.metadata.get("notes_organization_v1")
    return bool(
        context is not None
        and context.trusted_server_origin
        and context.organization_group_state == "initializing"
        and isinstance(metadata, dict)
        and isinstance(metadata.get("bootstrap_id"), str)
        and context.organization_bootstrap_id == metadata.get("bootstrap_id")
    )


def _get_head(
    envelope: SyncEnvelopeCreate,
    context: SyncAdapterContext | None,
) -> SyncHead | None:
    if context is not None and context.get_head is not None:
        return context.get_head(envelope.domain, envelope.object_id)
    matching = [
        item
        for item in prior_envelopes(envelope, context)
        if item.domain == envelope.domain and item.object_id == envelope.object_id
    ]
    return current_head(matching)


def _domain_heads(
    domain: SyncDomain,
    context: SyncAdapterContext | None,
) -> dict[str, SyncHead]:
    candidates: list[SyncHead] = []
    if context is not None:
        if context.list_heads is not None:
            candidates.extend(context.list_heads(domain))
        candidates.extend(item for item in context.prior_envelopes if item.domain == domain)
    grouped: dict[str, list[SyncHead]] = {}
    for item in candidates:
        grouped.setdefault(item.object_id, []).append(item)
    heads: dict[str, SyncHead] = {}
    for object_id, items in grouped.items():
        head = current_head(items)
        if head is not None:
            heads[object_id] = head
    return heads


def _validate_dependencies(
    envelope: SyncEnvelopeCreate,
    payload: dict[str, object],
    *,
    dataset: SyncDataset,
    context: SyncAdapterContext | None,
    allow_deleted: bool,
) -> AdapterRejected | None:
    references = _dependency_references(envelope.domain, payload)
    for domain, object_id in references:
        if domain not in dataset.domains:
            return _rejected(
                envelope,
                "notes_organization_dependency_missing",
                f"Required dependency domain is not enrolled: {domain}",
            )
        head = context.get_head(domain, object_id) if context and context.get_head else None
        if head is None:
            return _rejected(
                envelope,
                "notes_organization_dependency_missing",
                f"Required dependency is missing: {domain}",
            )
        if head.dataset_id != dataset.dataset_id:
            return _rejected(
                envelope,
                "notes_organization_ownership_mismatch",
                "Required dependency belongs to another dataset",
            )
        if _is_deleted(head) and not allow_deleted:
            return _rejected(
                envelope,
                "notes_organization_dependency_deleted",
                f"Required dependency is deleted: {domain}",
            )
    return None


def _dependency_references(
    domain: SyncDomain,
    payload: dict[str, object],
) -> tuple[tuple[SyncDomain, str], ...]:
    if domain == "notes.keyword_link":
        subject_domain: SyncDomain = (
            "notes.note" if payload["subject_type"] == "note" else "chat.conversation"
        )
        return (
            (subject_domain, cast(str, payload["subject_id"])),
            ("notes.keyword", cast(str, payload["keyword_sync_id"])),
        )
    if domain == "notes.keyword_collection_link":
        return (
            ("notes.keyword_collection", cast(str, payload["collection_sync_id"])),
            ("notes.keyword", cast(str, payload["keyword_sync_id"])),
        )
    if domain == "notes.folder_link":
        return (
            ("notes.note", cast(str, payload["note_id"])),
            ("notes.folder", cast(str, payload["folder_sync_id"])),
        )
    if domain in _HIERARCHY_DOMAINS and payload.get("parent_sync_id") is not None:
        return ((domain, cast(str, payload["parent_sync_id"])),)
    return ()


def _validate_resource_semantics(
    envelope: SyncEnvelopeCreate,
    payload: dict[str, object],
    *,
    context: SyncAdapterContext | None,
) -> AdapterConflict | None:
    if envelope.domain in {"notes.keyword", "notes.keyword_collection"}:
        field_name = "keyword" if envelope.domain == "notes.keyword" else "name"
        wanted = cast(str, payload[field_name]).casefold()
        for object_id, head in _domain_heads(envelope.domain, context).items():
            if object_id == envelope.object_id or _is_deleted(head):
                continue
            candidate = head.payload.get(field_name)
            if isinstance(candidate, str) and candidate.casefold() == wanted:
                return _conflict(
                    envelope,
                    "notes_organization_name_conflict",
                    "An active resource already uses this case-insensitive name",
                )

    if envelope.domain in _HIERARCHY_DOMAINS:
        return _validate_hierarchy(envelope, payload, context=context)
    return None


def _validate_hierarchy(
    envelope: SyncEnvelopeCreate,
    payload: dict[str, object],
    *,
    context: SyncAdapterContext | None,
) -> AdapterConflict | None:
    heads = _domain_heads(envelope.domain, context)
    heads[envelope.object_id] = envelope
    active = {object_id: head for object_id, head in heads.items() if not _is_deleted(head)}
    parent_id = payload.get("parent_sync_id")
    if parent_id == envelope.object_id:
        return _conflict(
            envelope,
            "notes_organization_hierarchy_cycle",
            "An organization resource cannot parent itself",
        )

    paths: dict[str, str] = {}
    try:
        for start_id in active:
            if start_id in paths:
                continue
            chain: list[str] = []
            seen: set[str] = set()
            cursor: str | None = start_id
            hidden_by_deleted_ancestor = False
            while cursor is not None and cursor not in paths:
                if cursor in seen or len(chain) > len(heads):
                    raise ValueError("cycle")
                head = heads.get(cursor)
                if head is None:
                    raise KeyError(cursor)
                if _is_deleted(head):
                    hidden_by_deleted_ancestor = True
                    break
                seen.add(cursor)
                chain.append(cursor)
                raw_parent = head.payload.get("parent_sync_id")
                cursor = cast(str, raw_parent) if raw_parent is not None else None

            if hidden_by_deleted_ancestor:
                continue
            prefix = paths.get(cursor, "") if cursor is not None else ""
            while chain:
                object_id = chain.pop()
                name = active[object_id].payload.get("name")
                if not isinstance(name, str):
                    raise ValueError("invalid name")
                prefix = f"{prefix}/{name}" if prefix else name
                paths[object_id] = prefix
    except (KeyError, ValueError):
        return _conflict(
            envelope,
            "notes_organization_hierarchy_cycle",
            "The organization hierarchy contains a cycle or corrupt parent chain",
        )

    if envelope.domain == "notes.folder":
        folded = [path.casefold() for path in paths.values()]
        if any(len(path) > 500 for path in paths.values()) or len(folded) != len(set(folded)):
            return _conflict(
                envelope,
                "notes_organization_path_conflict",
                "The derived folder path is too long or conflicts case-insensitively",
            )
    return None


def _validate_folder_segment(
    domain: SyncDomain,
    operation: str,
    payload: dict[str, object],
) -> None:
    if domain != "notes.folder" or operation != "upsert":
        return
    name = cast(str, payload["name"])
    if name in {".", ".."} or "/" in name or "\\" in name:
        raise NotesOrganizationValidationError(
            "notes_organization_payload_invalid",
            "Folder name must be one relative path segment",
        )


def _has_base(envelope: SyncEnvelopeCreate) -> bool:
    return any(
        value is not None
        for value in (
            envelope.base_server_cursor,
            envelope.base_object_revision,
            envelope.base_object_hash,
            envelope.base_version,
        )
    )


def _exact_base(envelope: SyncEnvelopeCreate, head: SyncHead) -> bool:
    revision_matches = (
        envelope.base_object_revision is None
        or envelope.base_object_revision == head.object_revision
    )
    head_version = head.entity_version if head.entity_version is not None else head.object_revision
    version_matches = (
        envelope.base_version is None
        or str(envelope.base_version) == str(head_version)
    )
    has_version_token = (
        envelope.base_object_revision is not None or envelope.base_version is not None
    )
    if head.server_cursor is None:
        cursor_matches = envelope.base_server_cursor in {None, 0}
    else:
        cursor_matches = envelope.base_server_cursor in {None, head.server_cursor}
    return bool(
        has_version_token
        and envelope.base_object_hash == head.payload_hash
        and revision_matches
        and version_matches
        and cursor_matches
    )


def _literal_replay(head: SyncHead, envelope: SyncEnvelopeCreate) -> bool:
    fingerprint_fields = (
        "dataset_id",
        "domain",
        "object_id",
        "stable_key",
        "operation",
        "client_envelope_id",
        "device_id",
        "client_profile_id",
        "client_sequence",
        "created_at_client",
        "base_server_cursor",
        "base_object_revision",
        "base_object_hash",
        "object_revision",
        "parent_id",
        "schema_version",
        "base_version",
        "entity_version",
        "dependencies",
        "routing_metadata",
        "payload_ciphertext",
        "payload",
        "payload_hash",
        "payload_size_bytes",
        "deleted",
        "encryption_metadata",
        "adapter_version",
        "status",
    )
    if any(getattr(head, name) != getattr(envelope, name) for name in fingerprint_fields):
        return False
    if head.mutation_group_id is None and envelope.mutation_group_id is None:
        return True
    return all(
        getattr(head, name) == getattr(envelope, name)
        for name in (
            "mutation_group_id",
            "mutation_step",
            "mutation_step_count",
            "mutation_plan_hash",
        )
    )


def _equivalent_state(head: SyncHead, operation: str, payload: dict[str, object]) -> bool:
    return head.operation == operation and dict(head.payload) == payload


def _is_deleted(head: SyncHead) -> bool:
    return head.operation == "tombstone" or head.deleted


def _rejected(
    envelope: SyncEnvelopeCreate,
    error_code: str,
    message: str,
) -> AdapterRejected:
    return AdapterRejected(
        client_envelope_id=envelope.client_envelope_id,
        error_code=error_code,
        message=message,
    )


def _conflict(
    envelope: SyncEnvelopeCreate,
    conflict_type: str,
    message: str,
) -> AdapterConflict:
    return AdapterConflict(
        client_envelope_id=envelope.client_envelope_id,
        domain=envelope.domain,
        entity_id=envelope.object_id,
        conflict_type=conflict_type,
        message=message,
    )


def _base_conflict(envelope: SyncEnvelopeCreate, message: str) -> AdapterConflict:
    return _conflict(envelope, "notes_organization_base_conflict", message)


__all__ = ["NotesOrganizationDomainAdapter"]
