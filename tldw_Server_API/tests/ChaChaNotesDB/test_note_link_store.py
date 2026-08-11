from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest

from tldw_Server_API.app.core.DB_Management.chacha.note_link_store import NotesLinkStore
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    ConflictError,
    InputError,
)

pytestmark = pytest.mark.unit


class _ZeroRowcount:
    rowcount = 0


class _LostCasConnection:
    """Proxy one transaction while simulating a concurrent version advance."""

    def __init__(self, connection) -> None:
        self._connection = connection
        self.update_sql: str | None = None

    def execute(self, sql, parameters=()):
        if sql.startswith("UPDATE note_edges SET"):
            self.update_sql = sql
            return _ZeroRowcount()
        return self._connection.execute(sql, parameters)


def _timestamp(second: int) -> str:
    return datetime(2026, 8, 10, 12, 0, second, tzinfo=timezone.utc).isoformat()


def _payload(
    first_note_id: str,
    second_note_id: str,
    *,
    modified_at: str,
    weight: float = 1.0,
    label: str | None = None,
    properties: dict[str, object] | None = None,
) -> dict[str, object]:
    source_note_id, target_note_id = sorted((first_note_id, second_note_id))
    return {
        "source_note_id": source_note_id,
        "target_note_id": target_note_id,
        "type": "manual",
        "directed": False,
        "weight": weight,
        "label": label,
        "properties": properties or {},
        "created_at": _timestamp(0),
        "last_modified": modified_at,
        "created_by": "device:test",
    }


def _graph_revision(db: CharactersRAGDB) -> int:
    row = db.execute_query("SELECT revision FROM note_graph_revisions WHERE singleton_id = 1").fetchone()
    assert row is not None
    return int(row["revision"])


@pytest.fixture()
def link_db(tmp_path):
    db = CharactersRAGDB(tmp_path / "notes-link-store.db", client_id="owner-1")
    note_ids = (str(uuid4()), str(uuid4()), str(uuid4()))
    for index, note_id in enumerate(note_ids):
        db.note_store.add_note(f"Note {index}", "body", note_id=note_id)
    try:
        yield db, NotesLinkStore(db), note_ids
    finally:
        db.close_connection()


def test_link_lifecycle_is_versioned_and_exact_replays_do_not_write(link_db):
    db, store, (first_note_id, second_note_id, _) = link_db
    edge_id = str(uuid4())
    created_payload = _payload(
        first_note_id,
        second_note_id,
        modified_at=_timestamp(0),
        label="related",
        properties={"kind": "reference"},
    )

    before_create = _graph_revision(db)
    created = store.upsert(edge_id=edge_id, payload=created_payload, expected_version=None)
    assert created.changed is True
    assert created.link.version == 1
    assert created.link.deleted is False
    assert _graph_revision(db) == before_create + 1

    replay_revision = _graph_revision(db)
    replayed = store.upsert(edge_id=edge_id, payload=created_payload, expected_version=None)
    assert replayed.changed is False
    assert replayed.link == created.link
    assert _graph_revision(db) == replay_revision

    updated_payload = _payload(
        first_note_id,
        second_note_id,
        modified_at=_timestamp(1),
        weight=2.5,
        label="supports",
        properties={"kind": "citation"},
    )
    updated = store.upsert(edge_id=edge_id, payload=updated_payload, expected_version=1)
    assert updated.changed is True
    assert updated.link.version == 2
    assert updated.link.weight == 2.5

    update_replay_revision = _graph_revision(db)
    update_replay = store.upsert(edge_id=edge_id, payload=updated_payload, expected_version=1)
    assert update_replay.changed is False
    assert update_replay.link == updated.link
    assert _graph_revision(db) == update_replay_revision

    tombstone_payload = {
        **updated_payload,
        "last_modified": _timestamp(2),
        "deleted_at": _timestamp(2),
        "reason": "manual-delete",
    }
    tombstoned = store.tombstone(
        edge_id=edge_id,
        payload=tombstone_payload,
        expected_version=2,
    )
    assert tombstoned.changed is True
    assert tombstoned.link.deleted is True
    assert tombstoned.link.version == 3
    assert store.list_for_notes([first_note_id, second_note_id]) == ()

    with pytest.raises(ConflictError, match="logical"):
        store.upsert(
            edge_id=str(uuid4()),
            payload=updated_payload,
            expected_version=None,
        )

    tombstone_replay_revision = _graph_revision(db)
    tombstone_replay = store.tombstone(
        edge_id=edge_id,
        payload=tombstone_payload,
        expected_version=2,
    )
    assert tombstone_replay.changed is False
    assert tombstone_replay.link == tombstoned.link
    assert _graph_revision(db) == tombstone_replay_revision

    restored_payload = {
        **updated_payload,
        "last_modified": _timestamp(3),
    }
    restored = store.restore(
        edge_id=edge_id,
        payload=restored_payload,
        expected_version=3,
    )
    assert restored.changed is True
    assert restored.link.deleted is False
    assert restored.link.deleted_at is None
    assert restored.link.version == 4
    assert store.get(edge_id) == restored.link

    restore_replay_revision = _graph_revision(db)
    restore_replay = store.restore(
        edge_id=edge_id,
        payload=restored_payload,
        expected_version=3,
    )
    assert restore_replay.changed is False
    assert restore_replay.link == restored.link
    assert _graph_revision(db) == restore_replay_revision


def test_link_identity_is_immutable_and_versions_are_optimistic(link_db):
    _, store, (first_note_id, second_note_id, third_note_id) = link_db
    edge_id = str(uuid4())
    payload = _payload(first_note_id, second_note_id, modified_at=_timestamp(0))
    store.upsert(edge_id=edge_id, payload=payload, expected_version=None)

    with pytest.raises(ConflictError, match="version"):
        store.upsert(
            edge_id=edge_id,
            payload={**payload, "weight": 2.0, "last_modified": _timestamp(1)},
            expected_version=7,
        )

    with pytest.raises(InputError, match="identity"):
        store.upsert(
            edge_id=edge_id,
            payload=_payload(first_note_id, third_note_id, modified_at=_timestamp(1)),
            expected_version=1,
        )

    with pytest.raises(ConflictError, match="logical"):
        store.upsert(
            edge_id=str(uuid4()),
            payload=payload,
            expected_version=None,
        )


@pytest.mark.parametrize("operation", ["upsert", "tombstone", "restore"])
def test_link_mutations_enforce_version_with_database_compare_and_swap(link_db, operation):
    db, store, (first_note_id, second_note_id, _) = link_db
    edge_id = str(uuid4())
    created_payload = _payload(first_note_id, second_note_id, modified_at=_timestamp(0))
    store.upsert(edge_id=edge_id, payload=created_payload, expected_version=None)

    expected_version = 1
    if operation == "upsert":
        payload = {**created_payload, "weight": 2.0, "last_modified": _timestamp(1)}
    elif operation == "tombstone":
        payload = {
            **created_payload,
            "last_modified": _timestamp(1),
            "deleted_at": _timestamp(1),
            "reason": "manual-delete",
        }
    else:
        tombstone_payload = {
            **created_payload,
            "last_modified": _timestamp(1),
            "deleted_at": _timestamp(1),
            "reason": "manual-delete",
        }
        store.tombstone(
            edge_id=edge_id,
            payload=tombstone_payload,
            expected_version=expected_version,
        )
        expected_version = 2
        payload = {**created_payload, "last_modified": _timestamp(2)}

    def mutate(conn):
        return getattr(store, operation)(
            edge_id=edge_id,
            payload=payload,
            expected_version=expected_version,
            conn=conn,
        )

    with db.transaction() as connection:
        proxy = _LostCasConnection(connection)
        with pytest.raises(ConflictError, match="version"):
            mutate(proxy)

    assert proxy.update_sql is not None
    assert "AND version = ?" in proxy.update_sql


def test_link_lifecycle_uses_one_canonical_properties_encoding(link_db):
    db, store, (first_note_id, second_note_id, _) = link_db
    edge_id = str(uuid4())
    created_payload = _payload(
        first_note_id,
        second_note_id,
        modified_at=_timestamp(0),
        properties={"display": "café"},
    )
    store.upsert(edge_id=edge_id, payload=created_payload, expected_version=None)

    store.tombstone(
        edge_id=edge_id,
        payload={
            **created_payload,
            "last_modified": _timestamp(1),
            "deleted_at": _timestamp(1),
            "reason": "manual-delete",
        },
        expected_version=1,
    )
    row = db.execute_query(
        "SELECT properties, metadata FROM note_edges WHERE edge_id = ?",
        (edge_id,),
    ).fetchone()

    assert row["properties"] == '{"display":"café"}'
    assert row["metadata"] == row["properties"]


def test_hard_deleting_an_endpoint_cascades_the_explicit_link(link_db):
    db, store, (first_note_id, second_note_id, _) = link_db
    edge_id = str(uuid4())
    store.upsert(
        edge_id=edge_id,
        payload=_payload(first_note_id, second_note_id, modified_at=_timestamp(0)),
        expected_version=None,
    )

    assert db.note_store.delete_note(first_note_id, hard_delete=True) is True
    assert store.get(edge_id) is None


def test_public_create_requires_live_owned_endpoints_but_history_can_use_deleted(link_db):
    db, store, (first_note_id, second_note_id, third_note_id) = link_db
    assert db.note_store.soft_delete_note(second_note_id, expected_version=1) is True

    payload = _payload(first_note_id, second_note_id, modified_at=_timestamp(0))
    with pytest.raises(InputError, match="live endpoints"):
        store.upsert(edge_id=str(uuid4()), payload=payload, expected_version=None)

    historical = store.upsert(
        edge_id=str(uuid4()),
        payload=payload,
        expected_version=None,
        allow_deleted_endpoints=True,
    )
    assert historical.changed is True
    assert store.list_for_notes([first_note_id, second_note_id]) == ()
    assert store.list_for_notes(
        [first_note_id, second_note_id],
        include_deleted_endpoints=True,
    ) == (historical.link,)

    foreign_note_id = str(uuid4())
    now = _timestamp(4)
    with db.transaction() as conn:
        conn.execute(
            "INSERT INTO notes(id, title, content, last_modified, client_id, version, deleted, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (foreign_note_id, "foreign", "body", now, "owner-2", 1, 0, now),
        )
    with pytest.raises(InputError, match="owned endpoints"):
        store.upsert(
            edge_id=str(uuid4()),
            payload=_payload(first_note_id, foreign_note_id, modified_at=_timestamp(4)),
            expected_version=None,
            allow_deleted_endpoints=True,
        )

    foreign_edge_id = str(uuid4())
    with db.transaction() as conn:
        conn.execute(
            "INSERT INTO note_edges(edge_id, user_id, from_note_id, to_note_id, type, directed, "
            "weight, properties, created_at, last_modified, created_by, version, deleted) "
            "VALUES (?, ?, ?, ?, 'manual', 0, 1.0, '{}', ?, ?, ?, 1, 0)",
            (
                foreign_edge_id,
                "owner-2",
                min(foreign_note_id, third_note_id),
                max(foreign_note_id, third_note_id),
                now,
                now,
                "device:foreign",
            ),
        )
    assert store.get(foreign_edge_id) is None

    cross_owner_edge_id = str(uuid4())
    with db.transaction() as conn:
        conn.execute(
            "INSERT INTO note_edges(edge_id, user_id, from_note_id, to_note_id, type, directed, "
            "weight, properties, created_at, last_modified, created_by, version, deleted) "
            "VALUES (?, ?, ?, ?, 'manual', 1, 1.0, '{}', ?, ?, ?, 1, 0)",
            (
                cross_owner_edge_id,
                "owner-1",
                first_note_id,
                foreign_note_id,
                now,
                now,
                "device:tampered",
            ),
        )
    assert store.get(cross_owner_edge_id) is None
    assert all(link.edge_id != cross_owner_edge_id for link in store.snapshot())


def test_legacy_manual_edge_methods_delegate_to_owner_bound_soft_delete(link_db):
    db, store, (first_note_id, second_note_id, _) = link_db

    with pytest.raises(InputError, match="authenticated owner"):
        db.create_manual_note_edge(
            user_id="other-owner",
            from_note_id=first_note_id,
            to_note_id=second_note_id,
            created_by="user:other-owner",
        )

    created = db.create_manual_note_edge(
        user_id="owner-1",
        from_note_id=second_note_id,
        to_note_id=first_note_id,
        directed=False,
        metadata={"kind": "legacy"},
        created_by="user:owner-1",
    )
    edge_id = str(created["edge_id"])
    assert created["from_note_id"] < created["to_note_id"]
    assert created["metadata"] == {"kind": "legacy"}
    assert db.delete_manual_note_edge(user_id="owner-1", edge_id=edge_id) is True

    stored = store.get(edge_id)
    assert stored is not None and stored.deleted is True
    assert db.delete_manual_note_edge(user_id="owner-1", edge_id=edge_id) is True
    row = db.execute_query(
        "SELECT COUNT(*) AS total FROM note_edges WHERE edge_id = ?",
        (edge_id,),
    ).fetchone()
    assert int(row["total"]) == 1


def test_external_transaction_rolls_back_link_and_revision_together(link_db):
    db, store, (first_note_id, second_note_id, _) = link_db
    edge_id = str(uuid4())
    payload = _payload(first_note_id, second_note_id, modified_at=_timestamp(0))
    before = _graph_revision(db)

    with pytest.raises(RuntimeError, match="abort"):
        with db.transaction() as conn:
            result = store.upsert(
                edge_id=edge_id,
                payload=payload,
                expected_version=None,
                conn=conn,
            )
            assert result.changed is True
            raise RuntimeError("abort")

    assert store.get(edge_id) is None
    assert _graph_revision(db) == before


def test_link_keyset_query_plan_uses_owner_live_index_without_temp_sort(link_db):
    db, store, (first_note_id, second_note_id, _) = link_db
    store.upsert(
        edge_id=str(uuid4()),
        payload=_payload(first_note_id, second_note_id, modified_at=_timestamp(0)),
        expected_version=None,
    )

    rows = db.execute_query(
        "EXPLAIN QUERY PLAN "
        "SELECT edge.edge_id FROM note_edges edge "
        "JOIN notes source ON source.id = edge.from_note_id "
        "JOIN notes target ON target.id = edge.to_note_id "
        "WHERE edge.user_id = ? AND source.client_id = ? AND target.client_id = ? "
        "AND edge.type = 'manual' AND edge.edge_id > ? AND edge.deleted = 0 "
        "AND source.deleted = 0 AND target.deleted = 0 "
        "ORDER BY edge.edge_id LIMIT ?",
        ("owner-1", "owner-1", "owner-1", "", 51),
    ).fetchall()
    details = [str(row["detail"]) for row in rows]

    assert any("idx_note_edges_owner_live" in detail for detail in details)
    assert not any("USE TEMP B-TREE" in detail for detail in details)
