from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from tldw_Server_API.app.core.DB_Management.chacha.note_link_store import NotesLinkStore
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.Sync_DB import SyncDatabase
from tldw_Server_API.app.core.Sync.v2.adapters import (
    AdapterAccepted,
    AdapterConflict,
    AdapterDeferred,
    SyncAdapterContext,
)
from tldw_Server_API.app.core.Sync.v2.domain_adapters.notes_link import NotesLinkDomainAdapter
from tldw_Server_API.app.core.Sync.v2.errors import SyncMaterializationContractError
from tldw_Server_API.app.core.Sync.v2.materializers.guarded_product_mutation import (
    GuardedProductMutation,
)
from tldw_Server_API.app.core.Sync.v2.materializers.notes_link import (
    NotesLinkMaterializer,
    _mark_applied,
)
from tldw_Server_API.app.core.Sync.v2.models import (
    DEFAULT_M1_ENCRYPTION_POLICY,
    SyncDataset,
    SyncDatasetCreate,
    SyncEnvelope,
    SyncEnvelopeCreate,
)
from tldw_Server_API.app.core.Sync.v2.store import SyncV2Store

DATASET_ID = "dataset-1"
OWNER_ID = "owner-1"
EDGE_ID = "11111111-1111-4111-8111-111111111111"
OTHER_EDGE_ID = "22222222-2222-4222-8222-222222222222"
SOURCE_NOTE_ID = "33333333-3333-4333-8333-333333333333"
TARGET_NOTE_ID = "44444444-4444-4444-8444-444444444444"
OTHER_NOTE_ID = "55555555-5555-4555-8555-555555555555"
CREATED_AT = "2026-08-10T12:00:00+00:00"


def _dataset(*, state: str = "ready") -> SyncDataset:
    return SyncDataset(
        dataset_id=DATASET_ID,
        owner_user_id=OWNER_ID,
        scope_type="personal",
        encryption_policy=DEFAULT_M1_ENCRYPTION_POLICY,
        domains=["notes.note", "notes.link"],
        workspace_id=None,
        metadata={"notes_link_v1": {"state": state}},
        created_at=CREATED_AT,
        updated_at=CREATED_AT,
    )


def _payload(
    *,
    source_note_id: str = SOURCE_NOTE_ID,
    target_note_id: str = TARGET_NOTE_ID,
    weight: float = 1.0,
    modified_at: str = CREATED_AT,
) -> dict[str, object]:
    source_note_id, target_note_id = sorted((source_note_id, target_note_id))
    return {
        "source_note_id": source_note_id,
        "target_note_id": target_note_id,
        "type": "manual",
        "directed": False,
        "weight": weight,
        "label": None,
        "properties": {},
        "created_at": CREATED_AT,
        "last_modified": modified_at,
        "created_by": "device-1",
    }


def _incoming(
    *,
    operation: str = "upsert",
    object_id: str = EDGE_ID,
    payload: dict[str, object] | None = None,
    base: SyncEnvelopeCreate | SyncEnvelope | None = None,
    restore: bool = False,
    suffix: str = "incoming",
) -> SyncEnvelopeCreate:
    normalized = _payload() if payload is None else payload
    if operation == "tombstone":
        normalized = {
            **normalized,
            "last_modified": "2026-08-10T12:00:02+00:00",
            "deleted_at": "2026-08-10T12:00:02+00:00",
            "reason": "manual-delete",
        }
    return SyncEnvelopeCreate(
        dataset_id=DATASET_ID,
        client_envelope_id=f"env-{suffix}",
        domain="notes.link",
        operation=operation,
        object_id=object_id,
        device_id="device-1",
        base_server_cursor=base.server_cursor if base else None,
        base_object_revision=base.object_revision if base else None,
        base_object_hash=base.payload_hash if base else None,
        base_version=base.entity_version if base else None,
        object_revision=1 if base is None else int(base.object_revision or 1) + 1,
        entity_version=1 if base is None else int(base.entity_version or 1) + 1,
        schema_version=1,
        payload=normalized,
        payload_hash=f"sha256:{suffix}",
        created_at_client=str(normalized["last_modified"]),
        routing_metadata={"restore_intent": True} if restore else {},
        status="accepted",
    )


def _head(
    *,
    operation: str = "upsert",
    object_id: str = EDGE_ID,
    payload: dict[str, object] | None = None,
    cursor: int = 10,
    revision: int = 1,
) -> SyncEnvelope:
    create = _incoming(
        operation=operation,
        object_id=object_id,
        payload=payload,
        suffix=f"head-{object_id}-{revision}",
    )
    return SyncEnvelope(
        **{
            field_name: getattr(create, field_name)
            for field_name in create.__dataclass_fields__
            if field_name
            not in {
                "server_cursor",
                "server_sequence",
                "object_revision",
                "entity_version",
                "payload_hash",
            }
        },
        server_cursor=cursor,
        object_revision=revision,
        entity_version=revision,
        payload_hash=f"sha256:head-{revision}",
    )


def _context(
    edge_head: SyncEnvelope | None = None,
    *,
    note_heads: dict[str, SyncEnvelope] | None = None,
    link_heads: tuple[SyncEnvelope, ...] = (),
) -> SyncAdapterContext:
    notes = note_heads or {
        SOURCE_NOTE_ID: _note_head(SOURCE_NOTE_ID),
        TARGET_NOTE_ID: _note_head(TARGET_NOTE_ID),
        OTHER_NOTE_ID: _note_head(OTHER_NOTE_ID),
    }

    def get_head(domain: str, object_id: str):
        if domain == "notes.link" and edge_head is not None and object_id == edge_head.object_id:
            return edge_head
        if domain == "notes.note":
            return notes.get(object_id)
        return None

    def list_heads(domain: str):
        return link_heads if domain == "notes.link" else tuple(notes.values())

    return SyncAdapterContext(get_head=get_head, list_heads=list_heads)


def _note_head(note_id: str, *, deleted: bool = False) -> SyncEnvelope:
    return SyncEnvelope(
        dataset_id=DATASET_ID,
        client_envelope_id=f"note-{note_id}",
        domain="notes.note",
        operation="tombstone" if deleted else "upsert",
        object_id=note_id,
        device_id="device-1",
        server_cursor=1,
        object_revision=1,
        entity_version=1,
        payload={"title": note_id, "content": "body"},
        payload_hash=f"sha256:{note_id}",
        created_at_client=CREATED_AT,
        deleted=deleted,
    )


def test_adapter_accepts_create_exact_update_and_explicit_restore() -> None:
    adapter = NotesLinkDomainAdapter()
    created = _incoming()
    assert isinstance(
        adapter.evaluate_envelope(created, dataset=_dataset(), context=_context()),
        AdapterAccepted,
    )

    current = _head()
    exact_replay = replace(
        created,
        client_envelope_id=current.client_envelope_id,
        payload_hash=current.payload_hash,
    )
    assert isinstance(
        adapter.evaluate_envelope(
            exact_replay,
            dataset=_dataset(),
            context=_context(current),
        ),
        AdapterAccepted,
    )

    update = _incoming(
        payload=_payload(weight=2.0, modified_at="2026-08-10T12:00:01+00:00"),
        base=current,
        suffix="update",
    )
    assert isinstance(
        adapter.evaluate_envelope(update, dataset=_dataset(), context=_context(current)),
        AdapterAccepted,
    )

    tombstone = _head(operation="tombstone", cursor=11, revision=2)
    restore = _incoming(
        payload=_payload(modified_at="2026-08-10T12:00:03+00:00"),
        base=tombstone,
        restore=True,
        suffix="restore",
    )
    assert isinstance(
        adapter.evaluate_envelope(
            restore,
            dataset=_dataset(),
            context=_context(tombstone),
        ),
        AdapterAccepted,
    )


@pytest.mark.parametrize("operation", ["upsert", "tombstone"])
def test_adapter_conflicts_stale_update_and_delete(operation: str) -> None:
    adapter = NotesLinkDomainAdapter()
    current = _head()
    stale_base = replace(current, server_cursor=9, payload_hash="sha256:stale")
    incoming = _incoming(operation=operation, base=stale_base, suffix=f"stale-{operation}")
    result = adapter.evaluate_envelope(
        incoming,
        dataset=_dataset(),
        context=_context(current),
    )
    assert isinstance(result, AdapterConflict)
    assert result.conflict_type == "notes_link_base_conflict"


def test_adapter_rejects_immutable_retarget_and_requires_explicit_restore() -> None:
    adapter = NotesLinkDomainAdapter()
    current = _head()
    retarget = _incoming(
        payload=_payload(target_note_id=OTHER_NOTE_ID, modified_at="2026-08-10T12:00:01+00:00"),
        base=current,
        suffix="retarget",
    )
    result = adapter.evaluate_envelope(
        retarget,
        dataset=_dataset(),
        context=_context(current),
    )
    assert isinstance(result, AdapterConflict)
    assert result.conflict_type == "notes_link_identity_conflict"

    tombstone = _head(operation="tombstone", cursor=11, revision=2)
    missing_intent = _incoming(
        payload=_payload(modified_at="2026-08-10T12:00:03+00:00"),
        base=tombstone,
        suffix="restore-without-intent",
    )
    result = adapter.evaluate_envelope(
        missing_intent,
        dataset=_dataset(),
        context=_context(tombstone),
    )
    assert isinstance(result, AdapterConflict)
    assert result.conflict_type == "notes_link_base_conflict"


def test_adapter_defers_missing_endpoint_and_accepts_deleted_endpoint_identity() -> None:
    adapter = NotesLinkDomainAdapter()
    missing = {SOURCE_NOTE_ID: _note_head(SOURCE_NOTE_ID)}
    result = adapter.evaluate_envelope(
        _incoming(),
        dataset=_dataset(),
        context=_context(note_heads=missing),
    )
    assert isinstance(result, AdapterDeferred)

    deleted = {
        SOURCE_NOTE_ID: _note_head(SOURCE_NOTE_ID),
        TARGET_NOTE_ID: _note_head(TARGET_NOTE_ID, deleted=True),
    }
    result = adapter.evaluate_envelope(
        _incoming(),
        dataset=_dataset(),
        context=_context(note_heads=deleted),
    )
    assert isinstance(result, AdapterAccepted)


def test_adapter_blocks_duplicate_logical_identity_under_another_object_id() -> None:
    duplicate = _head(object_id=OTHER_EDGE_ID)
    result = NotesLinkDomainAdapter().evaluate_envelope(
        _incoming(),
        dataset=_dataset(),
        context=_context(link_heads=(duplicate,)),
    )
    assert isinstance(result, AdapterConflict)
    assert result.conflict_type == "notes_link_logical_identity_conflict"


@pytest.fixture()
def materialization_stack(tmp_path: Path):
    note_db = CharactersRAGDB(tmp_path / "notes-link-product.db", client_id=OWNER_ID)
    for note_id in (SOURCE_NOTE_ID, TARGET_NOTE_ID, OTHER_NOTE_ID):
        note_db.note_store.add_note(note_id, "body", note_id=note_id)
    sync_store = SyncV2Store(SyncDatabase(sqlite_path=tmp_path / "notes-link-sync.db"))
    sync_store.enroll_dataset(
        SyncDatasetCreate(
            dataset_id=DATASET_ID,
            owner_user_id=OWNER_ID,
            domains=["notes.note"],
        )
    )
    sync_store.db.execute(
        "UPDATE sync_datasets SET domain_set_json = ?, metadata_json = ? WHERE dataset_id = ?",
        (
            json.dumps(["notes.note", "notes.link"]),
            json.dumps({"notes_link_v1": {"state": "ready"}}),
            DATASET_ID,
        ),
    )
    try:
        yield note_db, sync_store
    finally:
        note_db.close_connection()


def _stored(
    store: SyncV2Store,
    *,
    payload: dict[str, object],
    operation: str = "upsert",
    base: SyncEnvelope | None = None,
    revision: int = 1,
    suffix: str,
    restore: bool = False,
) -> SyncEnvelope:
    return store.insert_envelope(
        SyncEnvelopeCreate(
            dataset_id=DATASET_ID,
            client_envelope_id=f"stored-{suffix}",
            domain="notes.link",
            operation=operation,
            object_id=EDGE_ID,
            device_id="device-1",
            base_server_cursor=base.server_cursor if base else None,
            base_object_revision=base.object_revision if base else None,
            base_object_hash=base.payload_hash if base else None,
            base_version=base.entity_version if base else None,
            object_revision=revision,
            entity_version=revision,
            payload=payload,
            payload_hash=f"sha256:{suffix}",
            created_at_client=str(payload["last_modified"]),
            routing_metadata={"restore_intent": True} if restore else {},
            status="accepted",
        )
    )


def test_materializer_applies_and_crash_replay_does_not_repeat_product_write(
    materialization_stack,
) -> None:
    note_db, sync_store = materialization_stack
    materializer = NotesLinkMaterializer(note_db)
    created = _stored(
        sync_store,
        payload=_payload(),
        suffix="create",
    )
    result = materializer.apply(created, store=sync_store)
    assert result.status == "applied"
    first = note_db.notes_link_store.get(EDGE_ID)
    assert first is not None and first.version == 1

    updated_payload = _payload(weight=2.0, modified_at="2026-08-10T12:00:01+00:00")
    updated = _stored(
        sync_store,
        payload=updated_payload,
        base=created,
        revision=2,
        suffix="update",
    )
    product_revision_before = note_db.execute_query(
        "SELECT revision FROM note_graph_revisions WHERE singleton_id = 1"
    ).fetchone()["revision"]
    # An exact live postcondition short-circuits the later stale-base replay, as restore does.
    NotesLinkStore(note_db).upsert(
        edge_id=EDGE_ID,
        payload=updated_payload,
        expected_version=1,
    )
    product_revision_after = note_db.execute_query(
        "SELECT revision FROM note_graph_revisions WHERE singleton_id = 1"
    ).fetchone()["revision"]
    assert product_revision_after == product_revision_before + 1

    result = materializer.apply(updated, store=sync_store)
    assert result.status == "applied"
    assert note_db.notes_link_store.get(EDGE_ID).version == 2
    replay_revision = note_db.execute_query(
        "SELECT revision FROM note_graph_revisions WHERE singleton_id = 1"
    ).fetchone()["revision"]
    assert replay_revision == product_revision_after
    state = sync_store.get_object_state(DATASET_ID, "notes.link", EDGE_ID)
    assert state is not None and state.latest_server_cursor == updated.server_cursor


def test_materializer_exact_replay_executes_guard_against_live_postcondition(
    materialization_stack,
) -> None:
    note_db, sync_store = materialization_stack
    materializer = NotesLinkMaterializer(note_db)
    created = _stored(sync_store, payload=_payload(), suffix="guarded-replay")
    assert materializer.apply(created, store=sync_store).status == "applied"
    events: list[tuple[str, str | None]] = []
    guard = GuardedProductMutation(
        expected_domain="notes.link",
        expected_object_id=EDGE_ID,
        before=lambda _conn: events.append(("before", None)),
        after=lambda _conn, identity: events.append(("after", identity)),
    )

    result = materializer.apply(
        created,
        store=sync_store,
        guarded_mutation=guard,
    )

    assert result.status == "applied"
    assert events == [("before", None), ("after", EDGE_ID)]
    assert note_db.notes_link_store.get(EDGE_ID).version == 1


@pytest.mark.parametrize(
    ("operation", "restore", "suffix"),
    (("tombstone", False, "guarded-tombstone"), ("upsert", True, "guarded-restore")),
)
def test_invalid_guarded_mutation_raises_a_typed_materialization_contract_error(
    materialization_stack,
    operation: str,
    restore: bool,
    suffix: str,
) -> None:
    note_db, sync_store = materialization_stack
    envelope = _stored(
        sync_store,
        payload=_payload(),
        operation=operation,
        restore=restore,
        suffix=suffix,
    )
    guard = GuardedProductMutation(
        expected_domain="notes.link",
        expected_object_id=EDGE_ID,
        before=lambda _conn: None,
        after=lambda _conn, _identity: None,
    )

    with pytest.raises(
        SyncMaterializationContractError,
        match="sync_materialization_contract_invalid",
    ):
        NotesLinkMaterializer(note_db).apply(
            envelope,
            store=sync_store,
            guarded_mutation=guard,
        )


def test_materializer_returns_safe_conflict_for_divergent_product_state(
    materialization_stack,
) -> None:
    note_db, sync_store = materialization_stack
    materializer = NotesLinkMaterializer(note_db)
    created = _stored(sync_store, payload=_payload(), suffix="create-conflict")
    assert materializer.apply(created, store=sync_store).status == "applied"
    NotesLinkStore(note_db).upsert(
        edge_id=EDGE_ID,
        payload=_payload(weight=9.0, modified_at="2026-08-10T12:00:01+00:00"),
        expected_version=1,
    )
    incoming = _stored(
        sync_store,
        payload=_payload(weight=2.0, modified_at="2026-08-10T12:00:02+00:00"),
        base=created,
        revision=2,
        suffix="divergent",
    )

    result = materializer.apply(incoming, store=sync_store)

    assert result.status == "conflict"
    assert result.conflict_type == "notes_link_product_conflict"
    assert "payload" not in result.metadata
    stored = sync_store.get_envelope_by_server_cursor(incoming.server_cursor)
    assert stored is not None and stored.apply_status == "conflict"


def test_mark_applied_returns_failed_when_status_storage_is_unavailable() -> None:
    class _UnavailableStore:
        calls = 0

        def mark_envelope_apply_status(self, *_args, **_kwargs) -> None:
            self.calls += 1
            raise RuntimeError("storage unavailable")

    store = _UnavailableStore()

    result = _mark_applied(_head(), store)  # type: ignore[arg-type]

    assert result.status == "failed"
    assert result.error_code == "notes_link_projection_failed"
    assert store.calls == 2
