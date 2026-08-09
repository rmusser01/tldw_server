from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.Sync_DB import SyncDatabase
from tldw_Server_API.app.core.Sync.v2.materializers.notes_organization import (
    NotesOrganizationMaterializer,
)
from tldw_Server_API.app.core.Sync.v2.models import (
    NOTES_ORGANIZATION_DOMAINS,
    SyncDatasetCreate,
    SyncDomain,
    SyncEnvelope,
    SyncEnvelopeCreate,
)
from tldw_Server_API.app.core.Sync.v2.notes_organization import organization_link_id
from tldw_Server_API.app.core.Sync.v2.store import SyncV2Store

pytestmark = pytest.mark.unit

KEYWORD_ID = "11111111-1111-4111-8111-111111111111"
COLLECTION_ID = "22222222-2222-4222-8222-222222222222"
PARENT_COLLECTION_ID = "33333333-3333-4333-8333-333333333333"
FOLDER_ID = "44444444-4444-4444-8444-444444444444"
PARENT_FOLDER_ID = "55555555-5555-4555-8555-555555555555"
NOTE_ID = "66666666-6666-4666-8666-666666666666"


@pytest.fixture()
def note_db(tmp_path: Path) -> CharactersRAGDB:
    db = CharactersRAGDB(str(tmp_path / "notes.sqlite"), client_id="owner-1")
    yield db
    db.close_connection()


@pytest.fixture()
def sync_store(tmp_path: Path) -> SyncV2Store:
    store = SyncV2Store(SyncDatabase(sqlite_path=tmp_path / "sync.sqlite"))
    store.enroll_dataset(
        SyncDatasetCreate(
            dataset_id="dataset-1",
            owner_user_id="owner-1",
            domains=["notes.note", "chat.conversation"],
        )
    )
    # Task 7 owns public organization enrollment. Task 6 exercises already-enrolled
    # materializers by installing the accepted dataset contract directly.
    store.db.execute(
        "UPDATE sync_datasets SET domain_set_json = ?, metadata_json = ? "
        "WHERE dataset_id = ?",
        (
            json.dumps(["notes.note", "chat.conversation", *NOTES_ORGANIZATION_DOMAINS]),
            json.dumps({"notes_organization_v1": {"state": "ready"}}),
            "dataset-1",
        ),
    )
    return store


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
            "parent_sync_id": PARENT_COLLECTION_ID,
        },
        "notes.keyword_collection_link": {
            "collection_sync_id": COLLECTION_ID,
            "keyword_sync_id": KEYWORD_ID,
        },
        "notes.folder": {"name": "Work", "parent_sync_id": PARENT_FOLDER_ID},
        "notes.folder_link": {"note_id": NOTE_ID, "folder_sync_id": FOLDER_ID},
    }[domain]


def _object_id(domain: SyncDomain, payload: dict[str, object]) -> str:
    if domain == "notes.keyword":
        return KEYWORD_ID
    if domain == "notes.keyword_collection":
        return COLLECTION_ID
    if domain == "notes.folder":
        return FOLDER_ID
    if domain == "notes.keyword_link":
        members = [payload["subject_type"], payload["subject_id"], payload["keyword_sync_id"]]
    elif domain == "notes.keyword_collection_link":
        members = [payload["collection_sync_id"], payload["keyword_sync_id"]]
    else:
        members = [payload["note_id"], payload["folder_sync_id"]]
    return organization_link_id(domain, [cast(str, member) for member in members])


def _stored_envelope(
    store: SyncV2Store,
    domain: SyncDomain,
    *,
    operation: str = "upsert",
    payload: dict[str, object] | None = None,
    object_id: str | None = None,
    revision: int = 1,
    base: SyncEnvelope | None = None,
    restore: bool = False,
    suffix: str | None = None,
    routing_metadata: dict[str, object] | None = None,
) -> SyncEnvelope:
    normalized = _payload(domain) if payload is None else payload
    identity = object_id or _object_id(domain, normalized)
    tag = suffix or f"{operation}-{revision}"
    return store.insert_envelope(
        SyncEnvelopeCreate(
            dataset_id="dataset-1",
            client_envelope_id=f"env-{domain}-{tag}",
            domain=domain,
            operation=operation,
            object_id=identity,
            object_revision=revision,
            base_server_cursor=base.server_cursor if base else None,
            base_object_revision=base.object_revision if base else None,
            base_object_hash=base.payload_hash if base else None,
            payload=normalized,
            payload_hash=f"sha256:{domain}:{tag}",
            routing_metadata={"restore_intent": True} if restore else (routing_metadata or {}),
            status="accepted",
        )
    )


def _seed_dependencies(note_db: CharactersRAGDB, domain: SyncDomain) -> None:
    from tldw_Server_API.app.core.DB_Management.chacha.organization_sync_store import (
        NotesOrganizationSyncStore,
    )

    projection = NotesOrganizationSyncStore(note_db)
    if domain in {"notes.keyword_link", "notes.folder_link"}:
        note_db.add_note("Linked note", "Body", note_id=NOTE_ID)
    if domain in {"notes.keyword_link", "notes.keyword_collection_link"}:
        projection.apply_resource(
            domain="notes.keyword",
            object_id=KEYWORD_ID,
            operation="upsert",
            payload={"keyword": "Dependency"},
        )
    if domain in {"notes.keyword_collection", "notes.keyword_collection_link"}:
        projection.apply_resource(
            domain="notes.keyword_collection",
            object_id=PARENT_COLLECTION_ID,
            operation="upsert",
            payload={"name": "Root", "parent_sync_id": None},
        )
    if domain == "notes.keyword_collection_link":
        projection.apply_resource(
            domain="notes.keyword_collection",
            object_id=COLLECTION_ID,
            operation="upsert",
            payload={"name": "Projects", "parent_sync_id": PARENT_COLLECTION_ID},
        )
    if domain in {"notes.folder", "notes.folder_link"}:
        projection.apply_resource(
            domain="notes.folder",
            object_id=PARENT_FOLDER_ID,
            operation="upsert",
            payload={"name": "Root", "parent_sync_id": None},
        )
    if domain == "notes.folder_link":
        projection.apply_resource(
            domain="notes.folder",
            object_id=FOLDER_ID,
            operation="upsert",
            payload={"name": "Work", "parent_sync_id": PARENT_FOLDER_ID},
        )


def _projection_count(note_db: CharactersRAGDB, domain: SyncDomain) -> int:
    sql = {
        "notes.keyword": "SELECT COUNT(*) AS count FROM keywords WHERE sync_id = ?",
        "notes.keyword_link": (
            "SELECT COUNT(*) AS count FROM note_keywords l "
            "JOIN keywords k ON k.id = l.keyword_id WHERE l.note_id = ? AND k.sync_id = ?"
        ),
        "notes.keyword_collection": (
            "SELECT COUNT(*) AS count FROM keyword_collections WHERE sync_id = ?"
        ),
        "notes.keyword_collection_link": (
            "SELECT COUNT(*) AS count FROM collection_keywords l "
            "JOIN keyword_collections c ON c.id = l.collection_id "
            "JOIN keywords k ON k.id = l.keyword_id "
            "WHERE c.sync_id = ? AND k.sync_id = ?"
        ),
        "notes.folder": "SELECT COUNT(*) AS count FROM note_folders WHERE sync_id = ?",
        "notes.folder_link": (
            "SELECT COUNT(*) AS count FROM note_folder_memberships l "
            "JOIN note_folders f ON f.id = l.folder_id WHERE l.note_id = ? AND f.sync_id = ?"
        ),
    }[domain]
    params = {
        "notes.keyword": (KEYWORD_ID,),
        "notes.keyword_link": (NOTE_ID, KEYWORD_ID),
        "notes.keyword_collection": (COLLECTION_ID,),
        "notes.keyword_collection_link": (COLLECTION_ID, KEYWORD_ID),
        "notes.folder": (FOLDER_ID,),
        "notes.folder_link": (NOTE_ID, FOLDER_ID),
    }[domain]
    with note_db.transaction() as conn:
        return int(conn.execute(sql, params).fetchone()["count"])


@pytest.mark.parametrize("domain", NOTES_ORGANIZATION_DOMAINS)
def test_each_domain_upserts_tombstones_restores_and_reapplies_idempotently(
    domain: SyncDomain,
    note_db: CharactersRAGDB,
    sync_store: SyncV2Store,
) -> None:
    _seed_dependencies(note_db, domain)
    materializer = NotesOrganizationMaterializer(note_db, domain)

    created = _stored_envelope(sync_store, domain)
    assert materializer.apply(created, store=sync_store).status == "applied"
    assert _projection_count(note_db, domain) == 1
    local_id_before: int | None = None
    if domain in {"notes.keyword", "notes.keyword_collection", "notes.folder"}:
        from tldw_Server_API.app.core.DB_Management.chacha.organization_sync_store import (
            NotesOrganizationSyncStore,
        )

        resource = NotesOrganizationSyncStore(note_db).get_resource(
            domain, created.object_id
        )
        assert resource is not None
        local_id_before = resource.local_id

    tombstone_payload = {} if domain in {
        "notes.keyword",
        "notes.keyword_collection",
        "notes.folder",
    } else _payload(domain)
    tombstone = _stored_envelope(
        sync_store,
        domain,
        operation="tombstone",
        payload=tombstone_payload,
        revision=2,
        base=created,
    )
    assert materializer.apply(tombstone, store=sync_store).status == "applied"
    assert _projection_count(note_db, domain) == (
        1 if domain in {"notes.keyword", "notes.keyword_collection", "notes.folder"} else 0
    )

    restored = _stored_envelope(
        sync_store,
        domain,
        revision=3,
        base=tombstone,
        restore=True,
    )
    assert materializer.apply(restored, store=sync_store).status == "applied"
    assert materializer.apply(restored, store=sync_store).status == "applied"
    assert _projection_count(note_db, domain) == 1
    if local_id_before is not None:
        from tldw_Server_API.app.core.DB_Management.chacha.organization_sync_store import (
            NotesOrganizationSyncStore,
        )

        restored_resource = NotesOrganizationSyncStore(note_db).get_resource(
            domain, restored.object_id
        )
        assert restored_resource is not None
        assert restored_resource.local_id == local_id_before
    state = sync_store.get_object_state("dataset-1", domain, restored.object_id)
    assert state is not None
    assert (state.latest_server_cursor, state.deleted) == (restored.server_cursor, False)

    if domain in {"notes.keyword_collection", "notes.folder"}:
        table = "keyword_collections" if domain == "notes.keyword_collection" else "note_folders"
        parent_sync_id = PARENT_COLLECTION_ID if domain == "notes.keyword_collection" else PARENT_FOLDER_ID
        with note_db.transaction() as conn:
            row = conn.execute(
                f"SELECT child.parent_id, parent.id AS expected_parent_id "  # nosec B608 - fixed test table matrix
                f"FROM {table} child JOIN {table} parent ON parent.sync_id = ? "
                "WHERE child.sync_id = ?",
                (parent_sync_id, _object_id(domain, _payload(domain))),
            ).fetchone()
            assert row["parent_id"] == row["expected_parent_id"]
            if domain == "notes.folder":
                path = conn.execute(
                    "SELECT path FROM note_folders WHERE sync_id = ?", (FOLDER_ID,)
                ).fetchone()["path"]
                assert path == "Root/Work"


@pytest.mark.parametrize("domain", NOTES_ORGANIZATION_DOMAINS)
def test_each_domain_rejects_a_stale_non_head_apply_without_overwriting_product_state(
    domain: SyncDomain,
    note_db: CharactersRAGDB,
    sync_store: SyncV2Store,
) -> None:
    _seed_dependencies(note_db, domain)
    materializer = NotesOrganizationMaterializer(note_db, domain)
    created = _stored_envelope(sync_store, domain)
    assert materializer.apply(created, store=sync_store).status == "applied"
    current = _stored_envelope(
        sync_store,
        domain,
        payload=_payload(domain),
        revision=2,
        base=created,
        suffix="current",
    )
    assert materializer.apply(current, store=sync_store).status == "applied"

    stale = _stored_envelope(
        sync_store,
        domain,
        payload=_payload(domain),
        revision=3,
        base=created,
        suffix="stale",
    )
    result = materializer.apply(stale, store=sync_store)

    assert result.status == "conflict"
    assert _projection_count(note_db, domain) == 1
    state = sync_store.get_object_state("dataset-1", domain, stale.object_id)
    assert state is not None
    assert state.latest_server_cursor == current.server_cursor


@pytest.mark.parametrize(
    ("domain", "table"),
    [
        ("notes.keyword", "keywords"),
        ("notes.keyword_link", "note_keywords"),
        ("notes.keyword_collection", "keyword_collections"),
        ("notes.keyword_collection_link", "collection_keywords"),
        ("notes.folder", "note_folders"),
        ("notes.folder_link", "note_folder_memberships"),
    ],
)
def test_each_domain_rolls_back_product_failure_and_records_retryable_apply_error(
    domain: SyncDomain,
    table: str,
    note_db: CharactersRAGDB,
    sync_store: SyncV2Store,
) -> None:
    _seed_dependencies(note_db, domain)
    trigger = f"fail_{table}_insert"
    with note_db.transaction() as conn:
        conn.execute(
            f"CREATE TRIGGER {trigger} BEFORE INSERT ON {table} "  # nosec B608 - fixed test table matrix
            "BEGIN SELECT RAISE(ABORT, 'injected product failure'); END"
        )
    envelope = _stored_envelope(sync_store, domain)

    result = NotesOrganizationMaterializer(note_db, domain).apply(
        envelope, store=sync_store
    )

    assert result.status == "failed"
    assert result.error_code == "notes_organization_projection_failed"
    assert _projection_count(note_db, domain) == 0
    stored = sync_store.list_envelopes_for_entity(
        "dataset-1", domain, entity_id=envelope.object_id, limit=10
    )
    assert stored[0].apply_status == "failed"
    assert stored[0].apply_error_code == "notes_organization_projection_failed"
    assert sync_store.get_object_state("dataset-1", domain, envelope.object_id) is None


def test_resource_tombstones_preserve_relationships_source_rows_and_child_parent_pointers(
    note_db: CharactersRAGDB,
    sync_store: SyncV2Store,
) -> None:
    from tldw_Server_API.app.core.DB_Management.chacha.organization_sync_store import (
        NotesOrganizationSyncStore,
    )

    projection = NotesOrganizationSyncStore(note_db)
    note_db.add_note("Linked note", "Body", note_id=NOTE_ID)
    projection.apply_resource(
        domain="notes.folder",
        object_id=PARENT_FOLDER_ID,
        operation="upsert",
        payload={"name": "Root", "parent_sync_id": None},
    )
    projection.apply_resource(
        domain="notes.folder",
        object_id=FOLDER_ID,
        operation="upsert",
        payload={"name": "Child", "parent_sync_id": PARENT_FOLDER_ID},
    )
    link_payload = {"note_id": NOTE_ID, "folder_sync_id": PARENT_FOLDER_ID}
    projection.apply_relationship(
        domain="notes.folder_link",
        object_id=_object_id("notes.folder_link", link_payload),
        operation="upsert",
        payload=link_payload,
        routing_metadata={},
    )
    parent = _stored_envelope(
        sync_store,
        "notes.folder",
        payload={"name": "Root", "parent_sync_id": None},
        object_id=PARENT_FOLDER_ID,
    )
    materializer = NotesOrganizationMaterializer(note_db, "notes.folder")
    assert materializer.apply(parent, store=sync_store).status == "applied"
    tombstone = _stored_envelope(
        sync_store,
        "notes.folder",
        operation="tombstone",
        payload={},
        object_id=PARENT_FOLDER_ID,
        revision=2,
        base=parent,
    )

    assert materializer.apply(tombstone, store=sync_store).status == "applied"
    with note_db.transaction() as conn:
        child = conn.execute(
            "SELECT parent_id FROM note_folders WHERE sync_id = ?", (FOLDER_ID,)
        ).fetchone()
        parent_row = conn.execute(
            "SELECT id FROM note_folders WHERE sync_id = ?", (PARENT_FOLDER_ID,)
        ).fetchone()
        membership_count = conn.execute(
            "SELECT COUNT(*) AS count FROM note_folder_memberships "
            "WHERE note_id = ? AND folder_id = ?",
            (NOTE_ID, parent_row["id"]),
        ).fetchone()["count"]
    assert child["parent_id"] == parent_row["id"]
    assert membership_count == 1


def test_folder_link_tombstone_suppresses_source_membership_and_restore_ignores_remote_provenance(
    note_db: CharactersRAGDB,
    sync_store: SyncV2Store,
) -> None:
    from tldw_Server_API.app.core.DB_Management.chacha.organization_sync_store import (
        NotesOrganizationSyncStore,
    )

    _seed_dependencies(note_db, "notes.folder_link")
    projection = NotesOrganizationSyncStore(note_db)
    folder = projection.get_resource("notes.folder", FOLDER_ID)
    assert folder is not None
    with note_db.transaction() as conn:
        conn.execute(
            "INSERT INTO note_folder_source_memberships(note_id, folder_id, source_id, created_at) "
            "VALUES (?, ?, ?, ?)",
            (NOTE_ID, folder.local_id, 77, "2026-08-08T00:00:00+00:00"),
        )
    materializer = NotesOrganizationMaterializer(note_db, "notes.folder_link")
    upsert = _stored_envelope(
        sync_store,
        "notes.folder_link",
        routing_metadata={
            "origin": "remote",
            "source_id": 999,
            "source_key": "must-not-be-applied",
        },
    )
    assert materializer.apply(upsert, store=sync_store).status == "applied"
    tombstone = _stored_envelope(
        sync_store,
        "notes.folder_link",
        operation="tombstone",
        payload=_payload("notes.folder_link"),
        revision=2,
        base=upsert,
    )
    assert materializer.apply(tombstone, store=sync_store).status == "applied"
    assert projection.snapshot().relationships == ()
    with note_db.transaction() as conn:
        assert conn.execute(
            "SELECT COUNT(*) AS count FROM note_folder_source_memberships "
            "WHERE note_id = ? AND folder_id = ?",
            (NOTE_ID, folder.local_id),
        ).fetchone()["count"] == 1
        assert conn.execute(
            "SELECT COUNT(*) AS count FROM note_folder_source_memberships WHERE source_id = 999"
        ).fetchone()["count"] == 0
        assert conn.execute(
            "SELECT COUNT(*) AS count FROM note_folder_sync_suppressions "
            "WHERE note_id = ? AND folder_id = ?",
            (NOTE_ID, folder.local_id),
        ).fetchone()["count"] == 1


def test_materializer_rejects_stored_payload_and_identity_drift_before_product_write(
    note_db: CharactersRAGDB,
    sync_store: SyncV2Store,
) -> None:
    envelope = _stored_envelope(sync_store, "notes.keyword")
    malformed = replace(envelope, payload={"keyword": "Research", "unexpected": True})

    result = NotesOrganizationMaterializer(note_db, "notes.keyword").apply(
        malformed, store=sync_store
    )

    assert result.status == "failed"
    assert result.error_code == "notes_organization_projection_failed"
    assert _projection_count(note_db, "notes.keyword") == 0


def test_materializer_rejects_valid_relationship_payload_with_mismatched_object_id(
    note_db: CharactersRAGDB,
    sync_store: SyncV2Store,
) -> None:
    _seed_dependencies(note_db, "notes.keyword_link")
    envelope = _stored_envelope(
        sync_store,
        "notes.keyword_link",
        object_id="notes.keyword_link:sha256:" + "0" * 64,
    )

    result = NotesOrganizationMaterializer(note_db, "notes.keyword_link").apply(
        envelope, store=sync_store
    )

    assert result.status == "failed"
    assert result.error_code == "notes_organization_projection_failed"
    assert result.message == "Notes organization envelope validation failed"
    assert _projection_count(note_db, "notes.keyword_link") == 0


@pytest.mark.parametrize(
    "domain",
    ("notes.keyword_link", "notes.keyword_collection_link", "notes.folder_link"),
)
def test_relationship_materialization_fails_retryably_for_missing_owner_bound_dependencies(
    domain: SyncDomain,
    note_db: CharactersRAGDB,
    sync_store: SyncV2Store,
) -> None:
    envelope = _stored_envelope(sync_store, domain)

    result = NotesOrganizationMaterializer(note_db, domain).apply(
        envelope, store=sync_store
    )

    assert result.status == "failed"
    assert result.error_code == "notes_organization_projection_failed"
    assert _projection_count(note_db, domain) == 0
    stored = sync_store.list_envelopes_for_entity(
        "dataset-1", domain, entity_id=envelope.object_id, limit=10
    )[0]
    assert stored.apply_status == "failed"


def test_relationship_materialization_fails_retryably_for_soft_deleted_dependency(
    note_db: CharactersRAGDB,
    sync_store: SyncV2Store,
) -> None:
    from tldw_Server_API.app.core.DB_Management.chacha.organization_sync_store import (
        NotesOrganizationSyncStore,
    )

    _seed_dependencies(note_db, "notes.keyword_link")
    NotesOrganizationSyncStore(note_db).apply_resource(
        domain="notes.keyword",
        object_id=KEYWORD_ID,
        operation="tombstone",
        payload={},
    )
    envelope = _stored_envelope(sync_store, "notes.keyword_link")

    result = NotesOrganizationMaterializer(note_db, "notes.keyword_link").apply(
        envelope, store=sync_store
    )

    assert result.status == "failed"
    assert result.error_code == "notes_organization_projection_failed"
    assert result.message == "Notes organization dependency or hierarchy validation failed"
    assert _projection_count(note_db, "notes.keyword_link") == 0


def test_projection_failure_message_never_persists_user_labels_paths_or_backend_text(
    note_db: CharactersRAGDB,
    sync_store: SyncV2Store,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel_label = "LEAK-SENTINEL-PRIVATE-LABEL"
    sentinel_path = "/private/owner/secret-folder"
    sentinel_backend = "DETAIL: duplicate key value contains user data"
    envelope = _stored_envelope(
        sync_store,
        "notes.keyword",
        payload={"keyword": sentinel_label},
    )

    def _fail_with_sensitive_backend_text(*args, **kwargs):
        raise RuntimeError(f"{sentinel_backend}: {sentinel_label} at {sentinel_path}")

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Sync.v2.materializers.notes_organization."
        "NotesOrganizationSyncStore.apply_resource",
        _fail_with_sensitive_backend_text,
    )

    result = NotesOrganizationMaterializer(note_db, "notes.keyword").apply(
        envelope, store=sync_store
    )

    stored = sync_store.list_envelopes_for_entity(
        "dataset-1", "notes.keyword", entity_id=KEYWORD_ID, limit=10
    )[0]
    assert result.status == "failed"
    assert result.error_code == "notes_organization_projection_failed"
    assert result.message == "Notes organization projection failed"
    assert stored.apply_error_message == "Notes organization projection failed"
    for secret in (sentinel_label, sentinel_path, sentinel_backend):
        assert secret not in result.message
        assert secret not in stored.apply_error_message
