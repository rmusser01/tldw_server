from __future__ import annotations

from dataclasses import fields, replace
from pathlib import Path
from typing import cast

import pytest

from tldw_Server_API.app.core.DB_Management.Sync_DB import SyncDatabase
from tldw_Server_API.app.core.Sync.v2.adapters import (
    AdapterAccepted,
    AdapterConflict,
    AdapterRejected,
    SyncAdapterContext,
    SyncHead,
)
from tldw_Server_API.app.core.Sync.v2.domain_adapters.notes_organization import (
    NotesOrganizationDomainAdapter,
)
from tldw_Server_API.app.core.Sync.v2.models import (
    NOTES_ORGANIZATION_DOMAINS,
    SyncDataset,
    SyncDatasetCreate,
    SyncDomain,
    SyncEnvelope,
    SyncEnvelopeCreate,
)
from tldw_Server_API.app.core.Sync.v2.notes_organization import organization_link_id
from tldw_Server_API.app.core.Sync.v2.store import SyncV2Store

KEYWORD_ID = "11111111-1111-4111-8111-111111111111"
OTHER_KEYWORD_ID = "22222222-2222-4222-8222-222222222222"
COLLECTION_ID = "33333333-3333-4333-8333-333333333333"
OTHER_COLLECTION_ID = "44444444-4444-4444-8444-444444444444"
FOLDER_ID = "55555555-5555-4555-8555-555555555555"
OTHER_FOLDER_ID = "66666666-6666-4666-8666-666666666666"
NOTE_ID = "77777777-7777-4777-8777-777777777777"
CONVERSATION_ID = "conversation-1"


def _dataset(
    *,
    owner_user_id: str = "user-1",
    organization_state: str = "ready",
    domains: list[SyncDomain] | None = None,
) -> SyncDataset:
    return SyncDataset(
        dataset_id="dataset-1",
        owner_user_id=owner_user_id,
        scope_type="personal",
        encryption_policy="server_trusted_v1",
        domains=domains
        or [
            "notes.note",
            "chat.conversation",
            *NOTES_ORGANIZATION_DOMAINS,
        ],
        workspace_id=None,
        metadata={
            "notes_organization_v1": {
                "state": organization_state,
                "bootstrap_id": "bootstrap-1",
            }
        },
        created_at="2026-08-08T00:00:00+00:00",
        updated_at="2026-08-08T00:00:00+00:00",
    )


def _payload(domain: SyncDomain) -> dict[str, object]:
    return {
        "notes.keyword": {"keyword": "Research"},
        "notes.keyword_link": {
            "subject_type": "note",
            "subject_id": NOTE_ID,
            "keyword_sync_id": KEYWORD_ID,
        },
        "notes.keyword_collection": {
            "name": "Projects",
            "parent_sync_id": None,
        },
        "notes.keyword_collection_link": {
            "collection_sync_id": COLLECTION_ID,
            "keyword_sync_id": KEYWORD_ID,
        },
        "notes.folder": {"name": "Work", "parent_sync_id": None},
        "notes.folder_link": {
            "note_id": NOTE_ID,
            "folder_sync_id": FOLDER_ID,
        },
    }[domain]


def _object_id(domain: SyncDomain, payload: dict[str, object]) -> str:
    if domain == "notes.keyword":
        return KEYWORD_ID
    if domain == "notes.keyword_collection":
        return COLLECTION_ID
    if domain == "notes.folder":
        return FOLDER_ID
    if domain == "notes.keyword_link":
        members = [
            cast(str, payload["subject_type"]),
            cast(str, payload["subject_id"]),
            cast(str, payload["keyword_sync_id"]),
        ]
    elif domain == "notes.keyword_collection_link":
        members = [
            cast(str, payload["collection_sync_id"]),
            cast(str, payload["keyword_sync_id"]),
        ]
    else:
        members = [cast(str, payload["note_id"]), cast(str, payload["folder_sync_id"])]
    return organization_link_id(domain, members)


def _envelope(
    domain: SyncDomain,
    *,
    operation: str = "upsert",
    payload: dict[str, object] | None = None,
    object_id: str | None = None,
    client_envelope_id: str = "incoming",
    dataset_id: str = "dataset-1",
    base_server_cursor: int | None = None,
    base_object_revision: int | None = None,
    base_object_hash: str | None = None,
    object_revision: int | None = 1,
    base_version: int | str | None = None,
    entity_version: int | str | None = None,
    routing_metadata: dict[str, object] | None = None,
) -> SyncEnvelopeCreate:
    canonical_payload = dict(_payload(domain) if payload is None else payload)
    if operation == "tombstone" and domain in {
        "notes.keyword",
        "notes.keyword_collection",
        "notes.folder",
    }:
        canonical_payload = {} if payload is None else canonical_payload
    return SyncEnvelopeCreate(
        dataset_id=dataset_id,
        client_envelope_id=client_envelope_id,
        domain=domain,
        operation=cast(object, operation),
        object_id=object_id or _object_id(domain, canonical_payload or _payload(domain)),
        device_id="device-1",
        base_server_cursor=base_server_cursor,
        base_object_revision=base_object_revision,
        base_object_hash=base_object_hash,
        object_revision=object_revision,
        base_version=base_version,
        entity_version=entity_version,
        schema_version=1,
        payload=canonical_payload,
        payload_hash=f"hash:{client_envelope_id}",
        payload_size_bytes=64,
        routing_metadata=dict(routing_metadata or {}),
    )


def _stored(
    envelope: SyncEnvelopeCreate,
    *,
    sequence: int = 1,
    dataset_id: str | None = None,
) -> SyncEnvelope:
    create_fields = {field.name for field in fields(SyncEnvelopeCreate)}
    return SyncEnvelope(
        **{
            field_name: getattr(envelope, field_name)
            for field_name in create_fields
            if field_name
            not in {
                "dataset_id",
                "server_cursor",
                "server_sequence",
                "server_timestamp",
            }
        },
        dataset_id=dataset_id or envelope.dataset_id,
        server_cursor=sequence,
        server_timestamp="2026-08-08T00:00:00+00:00",
    )


def _context(*heads: SyncHead, **attestation: object) -> SyncAdapterContext:
    by_identity = {(head.domain, head.object_id): head for head in heads}

    def get_head(domain: SyncDomain, object_id: str) -> SyncHead | None:
        return by_identity.get((domain, object_id))

    def list_heads(domain: SyncDomain) -> tuple[SyncHead, ...]:
        return tuple(head for head in heads if head.domain == domain)

    return SyncAdapterContext(
        prior_envelopes=tuple(heads),
        get_head=get_head,
        list_heads=list_heads,
        **attestation,
    )


def _active_dependencies(domain: SyncDomain) -> tuple[SyncEnvelope, ...]:
    dependencies: list[SyncEnvelope] = []
    if domain in {"notes.keyword_link", "notes.keyword_collection_link"}:
        dependencies.append(_stored(_envelope("notes.keyword"), sequence=10))
    if domain == "notes.keyword_link":
        dependencies.append(
            _stored(
                _envelope(
                    "notes.note",
                    payload={"title": "Note", "content": "Body"},
                    object_id=NOTE_ID,
                ),
                sequence=11,
            )
        )
    if domain == "notes.keyword_collection_link":
        dependencies.append(
            _stored(_envelope("notes.keyword_collection"), sequence=12)
        )
    if domain == "notes.folder_link":
        dependencies.extend(
            [
                _stored(
                    _envelope(
                        "notes.note",
                        payload={"title": "Note", "content": "Body"},
                        object_id=NOTE_ID,
                    ),
                    sequence=13,
                ),
                _stored(_envelope("notes.folder"), sequence=14),
            ]
        )
    return tuple(dependencies)


@pytest.mark.parametrize("domain", NOTES_ORGANIZATION_DOMAINS)
def test_adapter_rejects_unknown_payload_fields_before_state_lookup(domain: SyncDomain) -> None:
    payload = _payload(domain)
    payload["unexpected"] = True
    calls = 0

    def get_head(_domain: SyncDomain, _object_id: str) -> SyncHead | None:
        nonlocal calls
        calls += 1
        return None

    context = SyncAdapterContext(get_head=get_head, list_heads=lambda _domain: ())

    outcome = NotesOrganizationDomainAdapter(domain).evaluate_envelope(
        _envelope(domain, payload=payload), dataset=_dataset(), context=context
    )

    assert isinstance(outcome, AdapterRejected)
    assert outcome.error_code == "notes_organization_payload_invalid"
    assert calls == 0


@pytest.mark.parametrize("domain", NOTES_ORGANIZATION_DOMAINS)
def test_adapter_rejects_noncanonical_object_identity(domain: SyncDomain) -> None:
    outcome = NotesOrganizationDomainAdapter(domain).evaluate_envelope(
        _envelope(domain, object_id="wrong-identity"),
        dataset=_dataset(),
        context=_context(),
    )

    assert isinstance(outcome, AdapterRejected)
    assert outcome.error_code == "notes_organization_identity_mismatch"


def test_adapter_rejects_nonready_or_partially_enrolled_group() -> None:
    partial = _dataset(domains=["notes.keyword"])
    initializing = _dataset(organization_state="initializing")
    adapter = NotesOrganizationDomainAdapter("notes.keyword")

    for dataset in (partial, initializing):
        outcome = adapter.evaluate_envelope(
            _envelope("notes.keyword"), dataset=dataset, context=_context()
        )
        assert isinstance(outcome, AdapterRejected)
        assert outcome.error_code == "notes_organization_domain_not_ready"


def test_resource_update_requires_exact_base_revision_and_hash() -> None:
    head = _stored(
        replace(
            _envelope("notes.keyword", client_envelope_id="head", object_revision=3),
            payload_hash="hash:head",
        ),
        sequence=7,
    )
    incoming = _envelope(
        "notes.keyword",
        payload={"keyword": "Renamed"},
        object_revision=4,
        base_object_revision=3,
        base_object_hash="hash:wrong",
    )

    outcome = NotesOrganizationDomainAdapter("notes.keyword").evaluate_envelope(
        incoming, dataset=_dataset(), context=_context(head)
    )

    assert isinstance(outcome, AdapterConflict)
    assert outcome.conflict_type == "notes_organization_base_conflict"


def test_exact_resource_base_and_idempotent_replay_are_accepted() -> None:
    head = _stored(
        replace(
            _envelope("notes.keyword", client_envelope_id="head", object_revision=3),
            payload_hash="hash:head",
        ),
        sequence=7,
    )
    adapter = NotesOrganizationDomainAdapter("notes.keyword")

    update = adapter.evaluate_envelope(
        _envelope(
            "notes.keyword",
            payload={"keyword": "Renamed"},
            object_revision=4,
            base_object_revision=3,
            base_object_hash="hash:head",
        ),
        dataset=_dataset(),
        context=_context(head),
    )
    replay = adapter.evaluate_envelope(
        replace(
            _envelope(
                "notes.keyword",
                client_envelope_id="replay",
                object_revision=3,
                base_object_revision=3,
                base_object_hash="hash:head",
            ),
            payload_hash="hash:head",
        ),
        dataset=_dataset(),
        context=_context(head),
    )

    assert isinstance(update, AdapterAccepted)
    assert isinstance(replay, AdapterAccepted)


def test_literal_original_retry_is_accepted_but_id_or_payload_drift_is_not() -> None:
    original = _envelope(
        "notes.keyword",
        client_envelope_id="literal-create",
        object_revision=1,
    )
    head = _stored(original, sequence=7)
    adapter = NotesOrganizationDomainAdapter("notes.keyword")

    literal_retry = adapter.evaluate_envelope(
        original,
        dataset=_dataset(),
        context=_context(head),
    )
    same_id_drift = adapter.evaluate_envelope(
        replace(
            original,
            payload={"keyword": "Changed"},
            payload_hash="hash:changed",
        ),
        dataset=_dataset(),
        context=_context(head),
    )
    new_baseless_change = adapter.evaluate_envelope(
        replace(original, client_envelope_id="new-envelope"),
        dataset=_dataset(),
        context=_context(head),
    )

    assert isinstance(literal_retry, AdapterAccepted)
    assert isinstance(same_id_drift, AdapterConflict)
    assert same_id_drift.conflict_type == "notes_organization_base_conflict"
    assert isinstance(new_baseless_change, AdapterConflict)
    assert new_baseless_change.conflict_type == "notes_organization_base_conflict"


def test_literal_retry_requires_symmetric_mutation_group_metadata() -> None:
    grouped = replace(
        _envelope(
            "notes.keyword",
            client_envelope_id="grouped-create",
            object_revision=1,
        ),
        mutation_group_id="group-1",
        mutation_step=0,
        mutation_step_count=1,
        mutation_plan_hash="a" * 64,
    )
    head = _stored(grouped, sequence=7)
    adapter = NotesOrganizationDomainAdapter("notes.keyword")

    exact_grouped = adapter.evaluate_envelope(
        grouped,
        dataset=_dataset(),
        context=_context(head),
    )
    omitted_group = adapter.evaluate_envelope(
        replace(
            grouped,
            mutation_group_id=None,
            mutation_step=None,
            mutation_step_count=None,
            mutation_plan_hash=None,
        ),
        dataset=_dataset(),
        context=_context(head),
    )
    drifted_group = adapter.evaluate_envelope(
        replace(grouped, mutation_plan_hash="b" * 64),
        dataset=_dataset(),
        context=_context(head),
    )

    assert isinstance(exact_grouped, AdapterAccepted)
    assert isinstance(omitted_group, AdapterConflict)
    assert omitted_group.conflict_type == "notes_organization_base_conflict"
    assert isinstance(drifted_group, AdapterConflict)
    assert drifted_group.conflict_type == "notes_organization_base_conflict"


def test_base_version_is_canonical_lineage_for_updates_and_equivalent_replays() -> None:
    head = _stored(
        replace(
            _envelope(
                "notes.keyword",
                client_envelope_id="versioned-head",
                object_revision=3,
                entity_version="v3",
            ),
            payload_hash="hash:versioned-head",
        ),
        sequence=7,
    )
    adapter = NotesOrganizationDomainAdapter("notes.keyword")

    correct = adapter.evaluate_envelope(
        _envelope(
            "notes.keyword",
            payload={"keyword": "Renamed"},
            base_version="v3",
            base_object_hash="hash:versioned-head",
        ),
        dataset=_dataset(),
        context=_context(head),
    )
    stale = adapter.evaluate_envelope(
        _envelope(
            "notes.keyword",
            payload={"keyword": "Renamed"},
            base_version="v2",
            base_object_hash="hash:versioned-head",
        ),
        dataset=_dataset(),
        context=_context(head),
    )
    divergent_duplicate = adapter.evaluate_envelope(
        replace(
            _envelope(
                "notes.keyword",
                client_envelope_id="divergent-duplicate",
                object_revision=3,
                base_version="v2",
                base_object_hash="hash:versioned-head",
            ),
            payload_hash="hash:versioned-head",
        ),
        dataset=_dataset(),
        context=_context(head),
    )

    assert isinstance(correct, AdapterAccepted)
    assert isinstance(stale, AdapterConflict)
    assert stale.conflict_type == "notes_organization_base_conflict"
    assert isinstance(divergent_duplicate, AdapterConflict)
    assert divergent_duplicate.conflict_type == "notes_organization_base_conflict"


def test_restore_requires_exact_current_tombstone_lineage() -> None:
    tombstone = _stored(
        replace(
            _envelope(
                "notes.keyword",
                operation="tombstone",
                client_envelope_id="deleted",
                object_revision=4,
            ),
            payload_hash="hash:deleted",
        ),
        sequence=9,
    )
    adapter = NotesOrganizationDomainAdapter("notes.keyword")

    stale = adapter.evaluate_envelope(
        _envelope(
            "notes.keyword",
            base_object_revision=3,
            base_object_hash="hash:older",
            routing_metadata={"restore_intent": True},
        ),
        dataset=_dataset(),
        context=_context(tombstone),
    )
    exact = adapter.evaluate_envelope(
        _envelope(
            "notes.keyword",
            base_object_revision=4,
            base_object_hash="hash:deleted",
            routing_metadata={"restore_intent": True},
        ),
        dataset=_dataset(),
        context=_context(tombstone),
    )

    assert isinstance(stale, AdapterConflict)
    assert stale.conflict_type == "notes_organization_base_conflict"
    assert isinstance(exact, AdapterAccepted)


def test_update_delete_divergence_is_reviewable() -> None:
    active = _stored(_envelope("notes.keyword", client_envelope_id="active"), sequence=1)
    deleted = _stored(
        _envelope("notes.keyword", operation="tombstone", client_envelope_id="deleted"),
        sequence=2,
    )
    adapter = NotesOrganizationDomainAdapter("notes.keyword")

    delete_vs_update = adapter.evaluate_envelope(
        _envelope("notes.keyword", operation="tombstone"),
        dataset=_dataset(),
        context=_context(active),
    )
    update_vs_delete = adapter.evaluate_envelope(
        _envelope("notes.keyword"),
        dataset=_dataset(),
        context=_context(deleted),
    )

    assert isinstance(delete_vs_update, AdapterConflict)
    assert delete_vs_update.conflict_type == "notes_organization_base_conflict"
    assert isinstance(update_vs_delete, AdapterConflict)
    assert update_vs_delete.conflict_type == "notes_organization_base_conflict"


@pytest.mark.parametrize(
    ("domain", "missing_domain", "missing_id"),
    [
        ("notes.keyword_link", "notes.note", NOTE_ID),
        ("notes.keyword_collection_link", "notes.keyword", KEYWORD_ID),
        ("notes.folder_link", "notes.folder", FOLDER_ID),
    ],
)
def test_relationship_rejects_missing_deleted_and_foreign_dependencies(
    domain: SyncDomain,
    missing_domain: SyncDomain,
    missing_id: str,
) -> None:
    dependencies = list(_active_dependencies(domain))
    adapter = NotesOrganizationDomainAdapter(domain)
    missing = tuple(
        head for head in dependencies if (head.domain, head.object_id) != (missing_domain, missing_id)
    )
    target = next(
        head for head in dependencies if (head.domain, head.object_id) == (missing_domain, missing_id)
    )
    deleted = replace(target, operation="tombstone", deleted=True, payload={})
    foreign = replace(target, dataset_id="dataset-foreign")

    outcomes = [
        adapter.evaluate_envelope(
            _envelope(domain), dataset=_dataset(), context=_context(*missing)
        ),
        adapter.evaluate_envelope(
            _envelope(domain), dataset=_dataset(), context=_context(*missing, deleted)
        ),
        adapter.evaluate_envelope(
            _envelope(domain), dataset=_dataset(), context=_context(*missing, foreign)
        ),
    ]

    assert [cast(AdapterRejected, outcome).error_code for outcome in outcomes] == [
        "notes_organization_dependency_missing",
        "notes_organization_dependency_deleted",
        "notes_organization_ownership_mismatch",
    ]


def test_conversation_keyword_link_requires_active_conversation_head() -> None:
    payload = {
        "subject_type": "conversation",
        "subject_id": CONVERSATION_ID,
        "keyword_sync_id": KEYWORD_ID,
    }
    envelope = _envelope("notes.keyword_link", payload=payload)
    keyword = _stored(_envelope("notes.keyword"), sequence=1)
    conversation = _stored(
        _envelope(
            "chat.conversation",
            payload={"title": "Conversation"},
            object_id=CONVERSATION_ID,
        ),
        sequence=2,
    )

    missing = NotesOrganizationDomainAdapter("notes.keyword_link").evaluate_envelope(
        envelope, dataset=_dataset(), context=_context(keyword)
    )
    accepted = NotesOrganizationDomainAdapter("notes.keyword_link").evaluate_envelope(
        envelope, dataset=_dataset(), context=_context(keyword, conversation)
    )

    assert isinstance(missing, AdapterRejected)
    assert missing.error_code == "notes_organization_dependency_missing"
    assert isinstance(accepted, AdapterAccepted)


@pytest.mark.parametrize(
    ("domain", "existing_id", "payload"),
    [
        ("notes.keyword", OTHER_KEYWORD_ID, {"keyword": "research"}),
        (
            "notes.keyword_collection",
            OTHER_COLLECTION_ID,
            {"name": "projects", "parent_sync_id": None},
        ),
    ],
)
def test_resource_names_are_unique_case_insensitively(
    domain: SyncDomain, existing_id: str, payload: dict[str, object]
) -> None:
    existing = _stored(_envelope(domain, object_id=existing_id), sequence=1)

    outcome = NotesOrganizationDomainAdapter(domain).evaluate_envelope(
        _envelope(domain, payload=payload),
        dataset=_dataset(),
        context=_context(existing),
    )

    assert isinstance(outcome, AdapterConflict)
    assert outcome.conflict_type == "notes_organization_name_conflict"


def test_collection_hierarchy_rejects_self_parent_and_preexisting_cycle() -> None:
    first = _stored(
        _envelope(
            "notes.keyword_collection",
            object_id=COLLECTION_ID,
            payload={"name": "First", "parent_sync_id": OTHER_COLLECTION_ID},
        ),
        sequence=1,
    )
    second = _stored(
        _envelope(
            "notes.keyword_collection",
            object_id=OTHER_COLLECTION_ID,
            payload={"name": "Second", "parent_sync_id": COLLECTION_ID},
        ),
        sequence=2,
    )
    adapter = NotesOrganizationDomainAdapter("notes.keyword_collection")

    self_parent = adapter.evaluate_envelope(
        _envelope(
            "notes.keyword_collection",
            payload={"name": "Projects", "parent_sync_id": COLLECTION_ID},
        ),
        dataset=_dataset(),
        context=_context(),
    )
    corrupt = adapter.evaluate_envelope(
        _envelope(
            "notes.keyword_collection",
            object_id="88888888-8888-4888-8888-888888888888",
            payload={"name": "Third", "parent_sync_id": None},
        ),
        dataset=_dataset(),
        context=_context(first, second),
    )

    assert isinstance(self_parent, AdapterConflict)
    assert self_parent.conflict_type == "notes_organization_hierarchy_cycle"
    assert isinstance(corrupt, AdapterConflict)
    assert corrupt.conflict_type == "notes_organization_hierarchy_cycle"


def test_deleted_ancestor_does_not_invalidate_retained_descendants_or_other_mutations() -> None:
    parent = _stored(
        _envelope(
            "notes.folder",
            object_id=FOLDER_ID,
            payload={"name": "Archived", "parent_sync_id": None},
            client_envelope_id="parent-active",
        ),
        sequence=1,
    )
    deleted_parent = _stored(
        _envelope(
            "notes.folder",
            operation="tombstone",
            object_id=FOLDER_ID,
            client_envelope_id="parent-deleted",
        ),
        sequence=2,
    )
    child = _stored(
        _envelope(
            "notes.folder",
            object_id=OTHER_FOLDER_ID,
            payload={"name": "Retained", "parent_sync_id": FOLDER_ID},
            client_envelope_id="retained-child",
        ),
        sequence=3,
    )

    outcome = NotesOrganizationDomainAdapter("notes.folder").evaluate_envelope(
        _envelope(
            "notes.folder",
            object_id="88888888-8888-4888-8888-888888888888",
            payload={"name": "Unrelated", "parent_sync_id": None},
        ),
        dataset=_dataset(),
        context=_context(parent, deleted_parent, child),
    )

    assert isinstance(outcome, AdapterAccepted)


def test_folder_rejects_cycle_duplicate_derived_path_and_long_path() -> None:
    parent = _stored(
        _envelope(
            "notes.folder",
            object_id=OTHER_FOLDER_ID,
            payload={"name": "Parent", "parent_sync_id": None},
        ),
        sequence=1,
    )
    existing_child = _stored(
        _envelope(
            "notes.folder",
            object_id=FOLDER_ID,
            payload={"name": "Child", "parent_sync_id": OTHER_FOLDER_ID},
        ),
        sequence=2,
    )
    duplicate_id = "99999999-9999-4999-8999-999999999999"
    adapter = NotesOrganizationDomainAdapter("notes.folder")

    cycle = adapter.evaluate_envelope(
        _envelope(
            "notes.folder",
            object_id=OTHER_FOLDER_ID,
            payload={"name": "Parent", "parent_sync_id": FOLDER_ID},
            base_object_revision=1,
            base_object_hash=parent.payload_hash,
        ),
        dataset=_dataset(),
        context=_context(parent, existing_child),
    )
    duplicate = adapter.evaluate_envelope(
        _envelope(
            "notes.folder",
            object_id=duplicate_id,
            payload={"name": "child", "parent_sync_id": OTHER_FOLDER_ID},
        ),
        dataset=_dataset(),
        context=_context(parent, existing_child),
    )
    long_path = adapter.evaluate_envelope(
        _envelope(
            "notes.folder",
            object_id=duplicate_id,
            payload={"name": "x" * 495, "parent_sync_id": OTHER_FOLDER_ID},
        ),
        dataset=_dataset(),
        context=_context(parent),
    )

    assert isinstance(cycle, AdapterConflict)
    assert cycle.conflict_type == "notes_organization_hierarchy_cycle"
    assert isinstance(duplicate, AdapterConflict)
    assert duplicate.conflict_type == "notes_organization_path_conflict"
    assert isinstance(long_path, AdapterConflict)
    assert long_path.conflict_type == "notes_organization_path_conflict"


@pytest.mark.parametrize(
    "domain", ["notes.keyword_link", "notes.keyword_collection_link", "notes.folder_link"]
)
def test_relationship_duplicate_upserts_and_tombstones_are_idempotent(domain: SyncDomain) -> None:
    dependencies = _active_dependencies(domain)
    active = _stored(_envelope(domain, client_envelope_id="active"), sequence=20)
    tombstone = _stored(
        _envelope(domain, operation="tombstone", client_envelope_id="deleted"),
        sequence=21,
    )
    adapter = NotesOrganizationDomainAdapter(domain)

    duplicate_upsert = adapter.evaluate_envelope(
        _envelope(
            domain,
            base_object_revision=active.object_revision,
            base_object_hash=active.payload_hash,
        ),
        dataset=_dataset(),
        context=_context(*dependencies, active),
    )
    duplicate_delete = adapter.evaluate_envelope(
        _envelope(
            domain,
            operation="tombstone",
            base_object_revision=tombstone.object_revision,
            base_object_hash=tombstone.payload_hash,
        ),
        dataset=_dataset(),
        context=_context(*dependencies, tombstone),
    )

    assert isinstance(duplicate_upsert, AdapterAccepted)
    assert isinstance(duplicate_delete, AdapterAccepted)


def test_bootstrap_capture_is_fail_closed_without_all_structural_attestations() -> None:
    dependency = _stored(
        _envelope(
            "notes.note",
            operation="tombstone",
            payload={},
            object_id=NOTE_ID,
        ),
        sequence=1,
    )
    keyword = _stored(_envelope("notes.keyword"), sequence=2)
    envelope = _envelope(
        "notes.keyword_link", routing_metadata={"bootstrap_capture": True}
    )
    initializing = _dataset(organization_state="initializing")
    adapter = NotesOrganizationDomainAdapter("notes.keyword_link")

    untrusted = adapter.evaluate_envelope(
        envelope,
        dataset=initializing,
        context=_context(
            dependency,
            keyword,
            trusted_server_origin=False,
            organization_group_state="initializing",
            bootstrap_relationship_verifier=lambda *_args: True,
        ),
    )
    unverified = adapter.evaluate_envelope(
        envelope,
        dataset=initializing,
        context=_context(
            dependency,
            keyword,
            trusted_server_origin=True,
            organization_group_state="initializing",
        ),
    )
    authorized = adapter.evaluate_envelope(
        envelope,
        dataset=initializing,
        context=_context(
            dependency,
            keyword,
            trusted_server_origin=True,
            organization_group_state="initializing",
            organization_bootstrap_id="bootstrap-1",
            bootstrap_relationship_verifier=lambda domain, object_id, payload: (
                domain == envelope.domain
                and object_id == envelope.object_id
                and dict(payload) == envelope.payload
            ),
        ),
    )
    wrong_bootstrap = adapter.evaluate_envelope(
        envelope,
        dataset=initializing,
        context=_context(
            dependency,
            keyword,
            trusted_server_origin=True,
            organization_group_state="initializing",
            organization_bootstrap_id="bootstrap-stale",
            bootstrap_relationship_verifier=lambda *_args: True,
        ),
    )

    assert isinstance(untrusted, AdapterRejected)
    assert untrusted.error_code == "notes_organization_domain_not_ready"
    assert isinstance(unverified, AdapterRejected)
    assert unverified.error_code == "notes_organization_domain_not_ready"
    assert isinstance(authorized, AdapterAccepted)
    assert isinstance(wrong_bootstrap, AdapterRejected)
    assert wrong_bootstrap.error_code == "notes_organization_domain_not_ready"


def test_bootstrap_capture_is_relationship_only_and_bound_to_exact_payload() -> None:
    resource = _envelope(
        "notes.keyword", routing_metadata={"bootstrap_capture": True}
    )
    relationship = _envelope(
        "notes.keyword_link", routing_metadata={"bootstrap_capture": True}
    )
    initializing = _dataset(organization_state="initializing")
    trusted = {
        "trusted_server_origin": True,
        "organization_group_state": "initializing",
        "organization_bootstrap_id": "bootstrap-1",
    }

    resource_outcome = NotesOrganizationDomainAdapter("notes.keyword").evaluate_envelope(
        resource,
        dataset=initializing,
        context=_context(
            bootstrap_relationship_verifier=lambda *_args: True,
            **trusted,
        ),
    )
    mismatched = NotesOrganizationDomainAdapter(
        "notes.keyword_link"
    ).evaluate_envelope(
        relationship,
        dataset=initializing,
        context=_context(
            *_active_dependencies("notes.keyword_link"),
            bootstrap_relationship_verifier=lambda domain, object_id, payload: (
                domain == relationship.domain
                and object_id == relationship.object_id
                and dict(payload) == {"different": True}
            ),
            **trusted,
        ),
    )

    assert isinstance(resource_outcome, AdapterRejected)
    assert resource_outcome.error_code == "notes_organization_payload_invalid"
    assert isinstance(mismatched, AdapterRejected)
    assert mismatched.error_code == "notes_organization_domain_not_ready"


def test_current_head_store_queries_are_owner_scoped_and_bounded(tmp_path: Path) -> None:
    store = SyncV2Store(SyncDatabase(sqlite_path=tmp_path / "sync.db"))
    store.enroll_dataset(
        SyncDatasetCreate(
            dataset_id="dataset-1",
            owner_user_id="user-1",
            scope_type="personal",
            encryption_policy="server_trusted_v1",
            domains=["notes.note"],
            metadata={},
        )
    )
    for index, note_id in enumerate((NOTE_ID, OTHER_KEYWORD_ID), start=1):
        store.insert_envelope(
            _envelope(
                "notes.note",
                payload={"title": f"Note {index}", "content": "Body"},
                object_id=note_id,
                client_envelope_id=f"note-{index}",
            )
        )
    store.insert_envelope(
        _envelope(
            "notes.note",
            object_id=NOTE_ID,
            payload={"title": "Renamed", "content": "Body"},
            client_envelope_id="note-new-head",
            base_server_cursor=1,
            base_object_revision=1,
            base_object_hash="hash:note-1",
            object_revision=2,
        )
    )

    head = store.get_current_head("dataset-1", "notes.note", NOTE_ID)
    first_page = store.list_current_heads(
        "dataset-1", "notes.note", limit=1, offset=0
    )
    second_page = store.list_current_heads(
        "dataset-1", "notes.note", limit=1, offset=1
    )

    assert head is not None
    assert head.client_envelope_id == "note-new-head"
    assert len(first_page) == 1
    assert len(second_page) == 1
    assert first_page[0].object_id != second_page[0].object_id
    assert store.get_current_head("dataset-foreign", "notes.note", NOTE_ID) is None


def test_nonaccepted_envelope_never_advances_or_repairs_as_current_head(
    tmp_path: Path,
) -> None:
    store = SyncV2Store(SyncDatabase(sqlite_path=tmp_path / "sync.db"))
    store.enroll_dataset(
        SyncDatasetCreate(
            dataset_id="dataset-1",
            owner_user_id="user-1",
            scope_type="personal",
            encryption_policy="server_trusted_v1",
            domains=["notes.note"],
            metadata={},
        )
    )
    accepted = store.insert_envelope(
        _envelope(
            "notes.note",
            payload={"title": "Accepted", "content": "Body"},
            object_id=NOTE_ID,
            client_envelope_id="accepted-head",
        )
    )
    conflict = store.insert_envelope(
        replace(
            _envelope(
                "notes.note",
                payload={"title": "Conflict", "content": "Body"},
                object_id=NOTE_ID,
                client_envelope_id="conflict-head",
                base_server_cursor=accepted.server_cursor,
                base_object_revision=1,
                base_object_hash=accepted.payload_hash,
                object_revision=2,
            ),
            status="conflict",
        )
    )

    store.db.execute(
        "UPDATE sync_current_heads SET latest_server_cursor = ? "
        "WHERE dataset_id = ? AND domain = ? AND object_id = ?",
        (conflict.server_cursor, "dataset-1", "notes.note", NOTE_ID),
    )
    store.db.ensure_schema()

    head = store.get_current_head("dataset-1", "notes.note", NOTE_ID)
    assert head is not None
    assert head.client_envelope_id == "accepted-head"


def test_current_head_history_backfill_runs_only_when_projection_is_created(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = SyncV2Store(SyncDatabase(sqlite_path=tmp_path / "sync.db"))
    store.enroll_dataset(
        SyncDatasetCreate(
            dataset_id="dataset-1",
            owner_user_id="user-1",
            scope_type="personal",
            encryption_policy="server_trusted_v1",
            domains=["notes.note"],
            metadata={},
        )
    )
    store.insert_envelope(
        _envelope(
            "notes.note",
            object_id=NOTE_ID,
            payload={"title": "Accepted", "content": "Body"},
            client_envelope_id="accepted-for-upgrade",
        )
    )
    with store.db.backend.transaction() as connection:
        store.db.execute("DROP TABLE sync_current_heads", connection=connection)
    store.db.ensure_schema()
    upgraded_head = store.get_current_head("dataset-1", "notes.note", NOTE_ID)
    assert upgraded_head is not None
    assert upgraded_head.client_envelope_id == "accepted-for-upgrade"

    statements: list[str] = []
    original_execute = store.db.execute

    def record_execute(sql: str, *args: object, **kwargs: object) -> object:
        statements.append(" ".join(sql.split()))
        return original_execute(sql, *args, **kwargs)

    monkeypatch.setattr(store.db, "execute", record_execute)
    store.db.ensure_schema()

    assert not any(
        "GROUP BY dataset_id, domain, entity_id" in statement
        for statement in statements
    )
