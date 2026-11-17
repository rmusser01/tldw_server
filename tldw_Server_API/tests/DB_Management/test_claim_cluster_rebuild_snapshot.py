"""Bounded source and cleanup integrity contracts for cluster publication."""

import hashlib
import importlib
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError

import pytest

from tldw_Server_API.app.core.DB_Management.media_db.api import managed_media_database
from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase
from tldw_Server_API.app.core.DB_Management.scope_context import scoped_context

pytestmark = pytest.mark.unit
RUNTIME = "tldw_Server_API.app.core.DB_Management.media_db.runtime.claims_cluster_rebuild_snapshot"
TYPES = "tldw_Server_API.app.core.DB_Management.media_db.claims_cluster_rebuild_types"


@pytest.fixture
def db(tmp_path):
    handle = MediaDatabase(db_path=str(tmp_path / "source.db"), client_id="42")
    yield handle
    handle.close_connection()


def seed(db, *, owner=42, text="Claim", chunk=0, visibility="personal"):
    """Use public ingestion helpers; all fixture rows are temporary."""
    with scoped_context(user_id=owner, is_admin=False, team_ids=[9], org_ids=[8]):
        media, _, _ = db.add_media_with_keywords(
            title=text,
            media_type="text",
            content=text,
            keywords=None,
            owner_user_id=owner,
            visibility=visibility,
        )
        db.upsert_claims(
            [
                {
                    "media_id": media,
                    "chunk_index": chunk,
                    "claim_text": text,
                    "extractor": "test",
                    "extractor_version": "1",
                    "chunk_hash": text,
                }
            ]
        )
        row = db.execute_query("SELECT id FROM Claims WHERE media_id = ?", (media,)).fetchone()
    return int(media), int(row["id"])


def snapshot(db, **kwargs):
    with scoped_context(user_id=42, is_admin=False, team_ids=[9], org_ids=[8]):
        return db.read_claim_cluster_source(
            owner_user_id="42",
            claim_limit=kwargs.get("claim_limit", 50_000),
            text_byte_limit=kwargs.get("text_byte_limit", 33_554_432),
        )


def cleanup(db, limit=50_000):
    module = importlib.import_module(RUNTIME)
    with scoped_context(user_id=42, is_admin=False):
        with db.transaction() as conn:
            return module.lock_claim_cluster_cleanup(db, conn, owner_user_id="42", limit=limit)


def test_owner_source_order_eligibility_and_fallback(db):
    _, first = seed(db, text="First")
    _, second = seed(db, text="Second", chunk=2)
    foreign, _ = seed(db, owner=43, text="Shared", visibility="team")
    deleted_media, _ = seed(db, text="Deleted media")
    _, deleted_claim = seed(db, text="Deleted claim")
    _, blank = seed(db, text="   ")
    db.execute_query("UPDATE Media SET team_id = 9, version = version + 1 WHERE id = ?", (foreign,), commit=True)
    db.execute_query("UPDATE Media SET deleted = 1, version = version + 1 WHERE id = ?", (deleted_media,), commit=True)
    db.execute_query("UPDATE Claims SET deleted = 1 WHERE id = ?", (deleted_claim,), commit=True)
    db.execute_query(
        "UPDATE Media SET owner_user_id = NULL, client_id = '42', version = version + 1 WHERE id IN "
        "(SELECT media_id FROM Claims WHERE id = ?)",
        (second,),
        commit=True,
    )
    result = snapshot(db)
    assert [r.claim_id for r in result.rows] == [first, second]
    assert result.owner_user_id == "42"
    assert blank not in [r.claim_id for r in result.rows]


@pytest.mark.parametrize("change", ["text", "chunk", "media", "add", "delete", "transfer"])
def test_digest_changes_for_every_source_field(db, change):
    media, claim = seed(db)
    before = snapshot(db).digest
    if change == "text":
        db.execute_query("UPDATE Claims SET claim_text = 'Changed' WHERE id = ?", (claim,), commit=True)
    elif change == "chunk":
        db.execute_query("UPDATE Claims SET chunk_index = 7 WHERE id = ?", (claim,), commit=True)
    elif change == "media":
        other, _ = seed(db, text="Other")
        db.execute_query("UPDATE Claims SET media_id = ? WHERE id = ?", (other, claim), commit=True)
    elif change == "add":
        seed(db, text="Added")
    elif change == "delete":
        db.execute_query("UPDATE Claims SET deleted = 1 WHERE id = ?", (claim,), commit=True)
    else:
        db.execute_query(
            "UPDATE Media SET owner_user_id = 43, version = version + 1 WHERE id = ?", (media,), commit=True
        )
    assert snapshot(db).digest != before


def test_snapshot_limits_are_independent_utf8_and_strict(db):
    seed(db, text="\u00e9\u00e9")
    assert len(snapshot(db, text_byte_limit=4).rows) == 1
    error = importlib.import_module(TYPES).ClusterRebuildStorageError
    with pytest.raises(error, match="claims_cluster_resource_limit"):
        snapshot(db, text_byte_limit=3)
    seed(db, text="Other")
    with pytest.raises(error, match="claims_cluster_resource_limit"):
        snapshot(db, claim_limit=1)
    for bad in (True, 0, 50_001):
        with pytest.raises(ValueError):
            snapshot(db, claim_limit=bad)


def test_digest_is_length_delimited_and_keyset_pages_are_consistent(db, monkeypatch):
    from hypothesis import given, settings
    from hypothesis import strategies as st

    module = importlib.import_module(RUNTIME)
    monkeypatch.setattr(module, "_PAGE_SIZE", 1)
    _, first = seed(db, text="First")
    media, second = seed(db, text="Second", chunk=2)
    queries = []
    fetch = db._fetchall_with_connection

    def record(conn, query, params=None):
        if "FROM Claims c" in query:
            queries.append((query, params))
        return fetch(conn, query, params)

    monkeypatch.setattr(db, "_fetchall_with_connection", record)
    result = snapshot(db)
    assert [row.claim_id for row in result.rows] == [first, second]
    assert all("OFFSET" not in query and "ORDER BY c.id" in query for query, _ in queries)
    assert [params[1] for _, params in queries] == [0, first, second]

    @settings(max_examples=12, deadline=None)
    @given(st.text(alphabet="ab|:\u00e9", min_size=1, max_size=30))
    def check_digest(text):
        db.execute_query("UPDATE Claims SET claim_text = ? WHERE id = ?", (text, second), commit=True)
        current = snapshot(db)
        expected = hashlib.sha256(b"claims-cluster-source-v1\x00")
        for value in ("42", first, result.rows[0].media_id, 0, "First", second, media, 2, text):
            raw = str(value).encode("utf-8")
            expected.update(len(raw).to_bytes(8, "big"))
            expected.update(raw)
        assert current.digest == expected.hexdigest()

    check_digest()


def test_oversized_text_is_not_materialized_from_storage(db, monkeypatch):
    seed(db, text="x" * 1000)
    fetch = db._fetchall_with_connection
    observed = []

    def record(conn, query, params=None):
        rows = fetch(conn, query, params)
        if "FROM Claims c" in query:
            observed.extend(rows)
        return rows

    monkeypatch.setattr(db, "_fetchall_with_connection", record)
    with pytest.raises(importlib.import_module(TYPES).ClusterRebuildStorageError):
        snapshot(db, text_byte_limit=100)
    assert observed[0]["claim_text"] is None


def test_source_scan_cannot_evade_claim_limit_with_blank_rows(db):
    seed(db, text=" ")
    seed(db, text="  ")
    with pytest.raises(
        importlib.import_module(TYPES).ClusterRebuildStorageError, match="claims_cluster_resource_limit"
    ):
        snapshot(db, claim_limit=1)


def test_sqlite_snapshot_is_deferred_and_does_not_rollback_foreign_transaction(db, monkeypatch):
    seed(db)
    statements = []
    conn = db.get_connection()
    conn.set_trace_callback(statements.append)
    try:
        monkeypatch.setattr(db, "get_connection", lambda: conn)
        snapshot(db)
        assert "BEGIN DEFERRED" in statements
        assert "BEGIN IMMEDIATE" not in statements
        conn.execute("BEGIN")
        with pytest.raises(ValueError):
            snapshot(db)
        assert conn.in_transaction
    finally:
        conn.set_trace_callback(None)
        conn.rollback()
        db.backend.get_pool().return_connection(conn)


def test_existing_only_snapshot_never_initializes(db, monkeypatch):
    seed(db)
    before = snapshot(db)
    path = db.db_path_str
    with managed_media_database(db_path=path, client_id="42", existing_only=True) as existing:
        monkeypatch.setattr(existing, "initialize_db", lambda: pytest.fail("runtime provisioning"))
        assert snapshot(existing) == before


@pytest.mark.parametrize("reference", ["assignment", "membership", "representative"])
def test_transferred_reference_is_not_hidden_by_owner_snapshot(db, reference):
    media, claim = seed(db)
    cluster = db.create_claim_cluster(user_id="42")
    if reference == "assignment":
        db.execute_query("UPDATE Claims SET claim_cluster_id = ? WHERE id = ?", (cluster["id"], claim), commit=True)
    elif reference == "membership":
        db.execute_query(
            "INSERT INTO claim_cluster_membership (cluster_id, claim_id) VALUES (?, ?)",
            (cluster["id"], claim),
            commit=True,
        )
    else:
        db.execute_query(
            "UPDATE claim_clusters SET representative_claim_id = ? WHERE id = ?", (claim, cluster["id"]), commit=True
        )
    db.execute_query("UPDATE Media SET owner_user_id = 43, version = version + 1 WHERE id = ?", (media,), commit=True)
    error = importlib.import_module(TYPES).ClusterRebuildStorageError
    assert snapshot(db).rows == ()
    with pytest.raises(error) as caught:
        cleanup(db)
    assert caught.value.code == "claims_cluster_foreign_reference"
    assert not caught.value.retryable
    assert db.get_claim_cluster(cluster["id"])


@pytest.mark.parametrize("boundary", ["clusters", "assignments", "memberships", "unique"])
def test_cleanup_limits_each_input_independently(db, boundary):
    _, first = seed(db, text="First")
    _, second = seed(db, text="Second")
    cluster = db.create_claim_cluster(user_id="42", representative_claim_id=first)
    if boundary == "clusters":
        db.create_claim_cluster(user_id="42")
    elif boundary == "assignments":
        db.execute_query("UPDATE Claims SET claim_cluster_id = ?", (cluster["id"],), commit=True)
    elif boundary == "memberships":
        for claim in (first, second):
            db.execute_query(
                "INSERT INTO claim_cluster_membership (cluster_id, claim_id) VALUES (?, ?)",
                (cluster["id"], claim),
                commit=True,
            )
    else:
        db.execute_query("UPDATE Claims SET claim_cluster_id = ? WHERE id = ?", (cluster["id"], second), commit=True)
    error = importlib.import_module(TYPES).ClusterRebuildStorageError
    with pytest.raises(error, match="claims_cluster_resource_limit"):
        cleanup(db, limit=1)
    assert db.get_claim_cluster(cluster["id"])


def test_cleanup_includes_inactive_references_and_uses_caller_transaction(db):
    _, claim = seed(db)
    cluster = db.create_claim_cluster(user_id="42", representative_claim_id=claim)
    db.add_claim_to_cluster(cluster_id=cluster["id"], claim_id=claim)
    db.execute_query("UPDATE Claims SET deleted = 1 WHERE id = ?", (claim,), commit=True)
    assert cleanup(db) == (claim,)
    module = importlib.import_module(RUNTIME)
    with pytest.raises(RuntimeError, match="rollback"):
        with db.transaction() as conn:
            db.execute_query("UPDATE Claims SET deleted = 0 WHERE id = ?", (claim,), connection=conn)
            assert (
                module.read_claim_cluster_source_on_connection(
                    db,
                    conn,
                    owner_user_id="42",
                    claim_limit=50,
                    text_byte_limit=100,
                    lock=True,
                )
                .rows[0]
                .claim_id
                == claim
            )
            raise RuntimeError("rollback")
    assert snapshot(db).rows == ()


def test_connection_helpers_reject_missing_connection(db):
    module = importlib.import_module(RUNTIME)
    with pytest.raises(ValueError):
        module.lock_claim_cluster_cleanup(db, None, owner_user_id="42", limit=50)
    with pytest.raises(ValueError):
        module.read_claim_cluster_source_on_connection(
            db,
            None,
            owner_user_id="42",
            claim_limit=50,
            text_byte_limit=100,
            lock=True,
        )


def test_add_revalidates_missing_destination_before_writing(db):
    _, claim = seed(db)
    error = importlib.import_module(TYPES).ClusterRebuildStorageError
    with pytest.raises(error):
        db.add_claim_to_cluster(cluster_id=999, claim_id=claim)
    assert db.execute_query("SELECT claim_cluster_id FROM Claims WHERE id = ?", (claim,)).fetchone()[0] is None


def test_publication_result_copies_and_freezes_nested_mapping():
    types = importlib.import_module(TYPES)
    data = {"clusters_created": 2}
    result = types.ClusterPublicationResult(data, True)
    data["clusters_created"] = 99
    assert result.result["clusters_created"] == 2
    with pytest.raises(TypeError):
        result.result["clusters_created"] = 3


def test_domain_values_defensively_copy_sequences_and_remain_frozen():
    types = importlib.import_module(TYPES)
    row = types.ClusterSourceRow(1, 2, 3, "Claim")
    rows = [row]
    source = types.ClusterSourceSnapshot("42", rows, "a" * 64)
    members = [types.ClusterMember(1, 1.0)]
    assignment = types.ClusterAssignment("Claim", 1, members)
    assignments = [assignment]
    calculation = types.ClusterCalculation(assignments, "ok", None, 0)
    identity = types.ClusterPublicationIdentity("42", "job", "authority", 1, "b" * 64)
    rows.clear()
    members.clear()
    assignments.clear()
    assert source.rows == (row,)
    assert assignment.members == (types.ClusterMember(1, 1.0),)
    assert calculation.assignments == (assignment,)
    with pytest.raises(FrozenInstanceError):
        identity.generation = 2


def test_sqlite_keyset_snapshot_is_consistent_when_writer_commits(db, monkeypatch):
    _, first = seed(db)
    module = importlib.import_module(RUNTIME)
    monkeypatch.setattr(module, "_PAGE_SIZE", 1)
    fetch = db._fetchall_with_connection
    inserted = []

    def insert():
        with managed_media_database(db_path=db.db_path_str, client_id="42", existing_only=True) as other:
            return seed(other, text="Concurrent arrival")[1]

    def fetch_then_insert(conn, query, params=None):
        page = fetch(conn, query, params)
        if "FROM Claims c" in query and not inserted:
            with ThreadPoolExecutor(max_workers=1) as pool:
                inserted.append(pool.submit(insert).result(timeout=5))
        return page

    monkeypatch.setattr(db, "_fetchall_with_connection", fetch_then_insert)
    assert [r.claim_id for r in snapshot(db).rows] == [first]
    assert [r.claim_id for r in snapshot(db).rows] == [first, inserted[0]]


@pytest.mark.parametrize("upgrade", [False, True])
def test_sqlite_fresh_and_upgrade_publication_schema(db, upgrade):
    if upgrade:
        db.execute_query("DROP TABLE IF EXISTS claims_cluster_publication_receipts", commit=True)
        db.execute_query("DROP TABLE IF EXISTS claims_cluster_publication_heads", commit=True)
        db.execute_query("UPDATE schema_version SET version = 26", commit=True)
        db.initialize_db()
    tables = {row["name"] for row in db.execute_query("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()}
    assert {"claims_cluster_publication_heads", "claims_cluster_publication_receipts"} <= tables
    assert db.execute_query("SELECT version FROM schema_version").fetchone()[0] == 27
    with db.transaction() as conn:
        conn.execute("INSERT INTO claims_cluster_publication_heads (owner_user_id) VALUES ('42')")
        row = conn.execute(
            "SELECT authority_uuid, generation, job_uuid FROM claims_cluster_publication_heads"
        ).fetchone()
        assert tuple(row) == (None, 0, None)
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("UPDATE claims_cluster_publication_heads SET generation = 1")
