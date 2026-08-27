from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path
from typing import Any

import pytest

from tldw_Server_API.app.core.DB_Management.chacha.organization_sync_store import (
    NotesOrganizationSyncStore,
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.Sync_DB import SyncDatabase
from tldw_Server_API.app.core.Sync.v2.adapters import (
    StaticSyncAdapter,
    SyncAdapterRegistry,
)
from tldw_Server_API.app.core.Sync.v2.domain_adapters.notes_link import (
    NotesLinkDomainAdapter,
)
from tldw_Server_API.app.core.Sync.v2.materializers.guarded_product_mutation import (
    GuardedProductMutation,
    GuardedProductMutationIdentityError,
)
from tldw_Server_API.app.core.Sync.v2.materializers.notes_link import (
    NotesLinkMaterializer,
)
from tldw_Server_API.app.core.Sync.v2.materializers.notes_organization import (
    NotesOrganizationMaterializer,
)
from tldw_Server_API.app.core.Sync.v2.models import (
    DEFAULT_M1_ENCRYPTION_POLICY,
    NOTES_ORGANIZATION_DOMAINS,
    SyncDatasetCreate,
    SyncEnvelopeCreate,
    SyncObjectState,
)
from tldw_Server_API.app.core.Sync.v2.notes_link_coordinator import (
    NotesLinkCoordinator,
)
from tldw_Server_API.app.core.Sync.v2.notes_organization import organization_link_id
from tldw_Server_API.app.core.Sync.v2.notes_organization_coordinator import (
    NotesOrganizationCoordinator,
)
from tldw_Server_API.app.core.Sync.v2.security import (
    server_trusted_encryption_status_from_config,
)
from tldw_Server_API.app.core.Sync.v2.server_origin_batch import (
    ServerOriginMutationStep,
    SyncServerOriginBatchMaterializationError,
)
from tldw_Server_API.app.core.Sync.v2.service import SyncV2Service, SyncV2Settings
from tldw_Server_API.app.core.Sync.v2.store import SyncV2Store

OWNER_ID = "owner-1"
DATASET_ID = "dataset-1"
SOURCE_ID = "11111111-1111-4111-8111-111111111111"
TARGET_ID = "22222222-2222-4222-8222-222222222222"
KEYWORD_ID = "33333333-3333-4333-8333-333333333333"
NOW = "2026-08-26T12:00:00+00:00"


def _ready_encryption():
    return server_trusted_encryption_status_from_config(
        mode="managed_storage",
        server_trusted_enabled=True,
        auth_mode="multi_user",
    )


@pytest.fixture()
def guarded_context(
    tmp_path: Path,
) -> tuple[SyncV2Service, SyncV2Store, CharactersRAGDB]:
    note_db = CharactersRAGDB(tmp_path / "notes.db", client_id=OWNER_ID)
    for note_id in (SOURCE_ID, TARGET_ID):
        note_db.note_store.add_note(note_id, "body", note_id=note_id)

    store = SyncV2Store(SyncDatabase(sqlite_path=tmp_path / "sync.db"))
    domains = [
        "notes.note",
        "notes.link",
        *NOTES_ORGANIZATION_DOMAINS,
    ]
    store.enroll_dataset(
        SyncDatasetCreate(
            dataset_id=DATASET_ID,
            owner_user_id=OWNER_ID,
            encryption_policy=DEFAULT_M1_ENCRYPTION_POLICY,
            domains=domains,
            metadata={
                "default_personal": True,
                "client_family": "chatbook",
                "notes_link_v1": {"state": "ready"},
                "notes_organization_v1": {"state": "ready"},
            },
        )
    )
    for index, note_id in enumerate((SOURCE_ID, TARGET_ID), start=1):
        envelope = store.insert_envelope(
            SyncEnvelopeCreate(
                dataset_id=DATASET_ID,
                client_envelope_id=f"note-{index}",
                domain="notes.note",
                operation="upsert",
                object_id=note_id,
                device_id="device-1",
                object_revision=1,
                entity_version=1,
                payload={"title": note_id, "content": "body"},
                payload_hash=f"sha256:note-{index}",
                created_at_client=NOW,
                status="accepted",
            )
        )
        assert envelope.server_cursor is not None
        store.upsert_object_state(
            SyncObjectState(
                dataset_id=DATASET_ID,
                domain="notes.note",
                object_id=note_id,
                object_revision=1,
                object_hash=envelope.payload_hash or "",
                latest_server_cursor=envelope.server_cursor,
                deleted=False,
            )
        )
        store.mark_envelope_apply_status(
            envelope.server_cursor,
            apply_status="applied",
        )

    adapters = [NotesLinkDomainAdapter()]
    adapters.extend(StaticSyncAdapter(domain=domain) for domain in NOTES_ORGANIZATION_DOMAINS)
    materializers = {
        "notes.link": NotesLinkMaterializer(note_db),
        **{domain: NotesOrganizationMaterializer(note_db, domain) for domain in NOTES_ORGANIZATION_DOMAINS},
    }
    service = SyncV2Service(
        store=store,
        adapters=SyncAdapterRegistry(adapters),
        materializers=materializers,
        clock=lambda: NOW,
        settings=SyncV2Settings(server_trusted_encryption=_ready_encryption()),
    )
    with note_db.transaction() as conn:
        conn.execute(
            "CREATE TABLE guarded_test_state (id INTEGER PRIMARY KEY, phase TEXT NOT NULL, resource_identity TEXT)"
        )
        conn.execute("INSERT INTO guarded_test_state(id, phase) VALUES (1, 'accepting')")
    try:
        yield service, store, note_db
    finally:
        note_db.close_connection()


def _phase(note_db: CharactersRAGDB) -> tuple[str, str | None]:
    with note_db.transaction() as conn:
        row = conn.execute("SELECT phase, resource_identity FROM guarded_test_state WHERE id = 1").fetchone()
    return str(row["phase"]), (str(row["resource_identity"]) if row["resource_identity"] is not None else None)


def _link_guard(
    *,
    edge_id: str,
    fail_after: bool = False,
) -> GuardedProductMutation:
    def before(conn: Any) -> None:
        row = conn.execute("SELECT phase FROM guarded_test_state WHERE id = 1").fetchone()
        if str(row["phase"]) != "accepting":
            raise RuntimeError("acceptance fence is no longer current")
        conn.execute("UPDATE guarded_test_state SET phase = 'guarded' WHERE id = 1")

    def after(conn: Any, resource_identity: str) -> None:
        row = conn.execute(
            "SELECT edge_id FROM note_edges WHERE edge_id = ? AND deleted = 0",
            (resource_identity,),
        ).fetchone()
        assert row is not None
        if fail_after:
            raise RuntimeError("injected finalizer failure")
        conn.execute(
            "UPDATE guarded_test_state SET phase = 'accepted', resource_identity = ? WHERE id = 1",
            (resource_identity,),
        )

    return GuardedProductMutation(
        expected_domain="notes.link",
        expected_object_id=edge_id,
        before=before,
        after=after,
    )


def _edge_id_for_key(key: str) -> str:
    digest = hashlib.sha256(f"{DATASET_ID}:notes.graph.link.create:{key}".encode()).digest()[:16]
    return str(uuid.UUID(bytes=digest, version=4))


def _create_link(
    service: SyncV2Service,
    store: SyncV2Store,
    note_db: CharactersRAGDB,
    *,
    key: str,
    guard: GuardedProductMutation | None,
):
    coordinator = NotesLinkCoordinator(
        service,
        note_db,
        OWNER_ID,
        store.get_dataset(DATASET_ID),
    )
    return coordinator.create(
        source_note_id=SOURCE_ID,
        target_note_id=TARGET_ID,
        directed=False,
        weight=1.0,
        label=None,
        properties={},
        idempotency_key=key,
        guarded_mutation=guard,
    )


def test_valid_guard_commits_canonical_link_and_finalization_together(
    guarded_context: tuple[SyncV2Service, SyncV2Store, CharactersRAGDB],
) -> None:
    service, store, note_db = guarded_context
    expected_edge_id = _edge_id_for_key("guard-valid")

    link = _create_link(
        service,
        store,
        note_db,
        key="guard-valid",
        guard=_link_guard(edge_id=expected_edge_id),
    )

    assert link.edge_id == expected_edge_id
    assert _phase(note_db) == ("accepted", expected_edge_id)
    head = store.get_current_head(DATASET_ID, "notes.link", expected_edge_id)
    assert head is not None
    json.dumps(head.payload)
    json.dumps(head.routing_metadata)
    assert "guarded_mutation" not in head.payload
    assert "guarded_mutation" not in head.routing_metadata


def test_replaced_fence_and_finalizer_failure_write_no_canonical_link(
    guarded_context: tuple[SyncV2Service, SyncV2Store, CharactersRAGDB],
) -> None:
    service, store, note_db = guarded_context
    edge_id = _edge_id_for_key("guard-replaced")
    with note_db.transaction() as conn:
        conn.execute("UPDATE guarded_test_state SET phase = 'replaced' WHERE id = 1")

    with pytest.raises(SyncServerOriginBatchMaterializationError):
        _create_link(
            service,
            store,
            note_db,
            key="guard-replaced",
            guard=_link_guard(edge_id=edge_id),
        )

    assert note_db.notes_link_store.get(edge_id) is None
    assert _phase(note_db) == ("replaced", None)

    with note_db.transaction() as conn:
        conn.execute("UPDATE guarded_test_state SET phase = 'accepting' WHERE id = 1")
    with pytest.raises(SyncServerOriginBatchMaterializationError):
        _create_link(
            service,
            store,
            note_db,
            key="guard-replaced",
            guard=_link_guard(edge_id=edge_id, fail_after=True),
        )

    assert note_db.notes_link_store.get(edge_id) is None
    assert _phase(note_db) == ("accepting", None)


def test_exact_replay_rechecks_guard_and_finalizes_existing_postcondition(
    guarded_context: tuple[SyncV2Service, SyncV2Store, CharactersRAGDB],
) -> None:
    service, store, note_db = guarded_context
    edge_id = _edge_id_for_key("guard-replay")
    first = _create_link(
        service,
        store,
        note_db,
        key="guard-replay",
        guard=_link_guard(edge_id=edge_id),
    )
    with note_db.transaction() as conn:
        conn.execute("UPDATE guarded_test_state SET phase = 'accepting', resource_identity = NULL WHERE id = 1")

    replay = _create_link(
        service,
        store,
        note_db,
        key="guard-replay",
        guard=_link_guard(edge_id=edge_id),
    )

    assert replay == first
    assert _phase(note_db) == ("accepted", edge_id)


def test_guard_identity_mismatch_fails_before_any_product_mutation(
    guarded_context: tuple[SyncV2Service, SyncV2Store, CharactersRAGDB],
) -> None:
    service, _store, note_db = guarded_context
    coordinator = NotesOrganizationCoordinator(service, note_db, OWNER_ID)
    keyword_link_id = organization_link_id(
        "notes.keyword_link",
        ["note", SOURCE_ID, KEYWORD_ID],
    )
    steps = (
        ServerOriginMutationStep(
            domain="notes.keyword",
            operation="upsert",
            object_id=KEYWORD_ID,
            payload={"keyword": "Research"},
        ),
        ServerOriginMutationStep(
            domain="notes.keyword_link",
            operation="upsert",
            object_id=keyword_link_id,
            payload={
                "subject_type": "note",
                "subject_id": SOURCE_ID,
                "keyword_sync_id": KEYWORD_ID,
            },
        ),
    )
    guard = GuardedProductMutation(
        expected_domain="notes.keyword_link",
        expected_object_id="44444444-4444-4444-8444-444444444444",
        before=lambda _conn: None,
        after=lambda _conn, _identity: None,
    )

    with pytest.raises(GuardedProductMutationIdentityError):
        coordinator.capture(
            steps=steps,
            source="notes_graph_suggestion",
            idempotency_key="guard-mismatch",
            guarded_mutation=guard,
        )

    assert note_db.get_keyword_by_text("Research") is None


def test_keyword_creation_cannot_finalize_guarded_relationship(
    guarded_context: tuple[SyncV2Service, SyncV2Store, CharactersRAGDB],
) -> None:
    service, _store, note_db = guarded_context
    coordinator = NotesOrganizationCoordinator(service, note_db, OWNER_ID)
    payload = {
        "subject_type": "note",
        "subject_id": SOURCE_ID,
        "keyword_sync_id": KEYWORD_ID,
    }
    keyword_link_id = organization_link_id(
        "notes.keyword_link",
        ["note", SOURCE_ID, KEYWORD_ID],
    )
    steps = (
        ServerOriginMutationStep(
            domain="notes.keyword",
            operation="upsert",
            object_id=KEYWORD_ID,
            payload={"keyword": "Research"},
        ),
        ServerOriginMutationStep(
            domain="notes.keyword_link",
            operation="upsert",
            object_id=keyword_link_id,
            payload=payload,
        ),
    )

    def before(conn: Any) -> None:
        conn.execute("UPDATE guarded_test_state SET phase = 'guarded' WHERE id = 1")

    def fail_after(_conn: Any, _resource_identity: str) -> None:
        raise RuntimeError("injected relationship finalizer failure")

    guard = GuardedProductMutation(
        expected_domain="notes.keyword_link",
        expected_object_id=keyword_link_id,
        before=before,
        after=fail_after,
    )

    with pytest.raises(SyncServerOriginBatchMaterializationError):
        coordinator.capture(
            steps=steps,
            source="notes_graph_suggestion",
            idempotency_key="new-keyword",
            guarded_mutation=guard,
        )

    assert note_db.get_keyword_by_text("Research") is not None
    assert not NotesOrganizationSyncStore(note_db).relationship_present(
        domain="notes.keyword_link",
        object_id=keyword_link_id,
        payload=payload,
    )
    assert _phase(note_db) == ("accepting", None)


def test_guarded_keyword_creation_checks_fence_without_running_finalizer(
    guarded_context: tuple[SyncV2Service, SyncV2Store, CharactersRAGDB],
) -> None:
    service, _store, note_db = guarded_context
    coordinator = NotesOrganizationCoordinator(service, note_db, OWNER_ID)
    events: list[str] = []
    guard = GuardedProductMutation(
        expected_domain="notes.keyword",
        expected_object_id=KEYWORD_ID,
        before=lambda _conn: events.append("before"),
        after=lambda _conn, _identity: events.append("after"),
    )

    result = coordinator.capture(
        steps=(
            ServerOriginMutationStep(
                domain="notes.keyword",
                operation="upsert",
                object_id=KEYWORD_ID,
                payload={"keyword": "Research"},
            ),
        ),
        source="notes_graph_suggestion",
        idempotency_key="guarded-keyword",
        guarded_mutation=guard,
    )

    assert result.fully_applied is True
    assert note_db.get_keyword_by_text("Research") is not None
    assert events == ["before"]


def test_ordinary_unguarded_caller_is_unchanged(
    guarded_context: tuple[SyncV2Service, SyncV2Store, CharactersRAGDB],
) -> None:
    service, store, note_db = guarded_context

    link = _create_link(
        service,
        store,
        note_db,
        key="ordinary",
        guard=None,
    )

    assert note_db.notes_link_store.get(link.edge_id) == link
    assert _phase(note_db) == ("accepting", None)
