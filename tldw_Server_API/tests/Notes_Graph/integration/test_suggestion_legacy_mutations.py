"""Real product-store controls for the inactive-Sync suggestion adapter.

The registered scope isolates guarded product mutations. Fresh unbound admission
and factory integration are separate controls and must not be inferred from it.
"""

from importlib import import_module

import pytest

from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.chacha.operation_scope import chacha_operation
from tldw_Server_API.app.core.DB_Management.chacha.organization_sync_store import (
    GuardedKeywordIdentityCollision,
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, InputError
from tldw_Server_API.app.core.Notes_Graph.suggestion_service import build_suggestion_decision_service
from tldw_Server_API.app.core.Sync.v2 import server_origin
from tldw_Server_API.app.core.Sync.v2.materializers.guarded_product_mutation import (
    GuardedProductMutation,
    GuardedProductMutationIdentityError,
)
from tldw_Server_API.app.core.Sync.v2.notes_organization import organization_link_id
from tldw_Server_API.tests.Notes_Graph.integration.test_suggestion_acceptance import (
    DATASET_ID,
    NOW,
    OWNER_ID,
    SOURCE_ID,
    TARGET_ID,
    _fingerprint,
    _publish,
)

pytestmark = pytest.mark.integration
EDGE_ID = "33333333-3333-4333-8333-333333333333"
KEYWORD_ID = "44444444-4444-4444-8444-444444444444"


@pytest.fixture(params=["sqlite", pytest.param("postgres", marks=pytest.mark.postgres)])
def local_product_db(request, tmp_path, monkeypatch):
    backend = (
        DatabaseBackendFactory.create_backend(request.getfixturevalue("pg_database_config"))
        if request.param == "postgres"
        else None
    )
    db = CharactersRAGDB(tmp_path / "local-suggestions.db", client_id=OWNER_ID, backend=backend)
    monkeypatch.setattr(server_origin, "get_active_server_origin_sync_service_for_user", lambda _owner: None)
    try:
        with chacha_operation(independent=True):
            db.add_note("Source", "source body", note_id=SOURCE_ID)
            db.add_note("Target", "target body", note_id=TARGET_ID)
            yield db
    finally:
        db.close_all_connections()
        if backend is not None:
            backend.get_pool().close_all()


def _adapter(db):
    module = import_module("tldw_Server_API.app.core.Notes_Graph.suggestion_local_mutations")
    return module.LocalSuggestionMutations(db)


def _claim(db, *, kind="related_note"):
    # Existing canonical storage setup is intentional for this adapter-only
    # boundary; it does not assert that a fresh local scope is authorized yet.
    with db.transaction() as conn:
        conn.execute(
            "INSERT INTO note_task_scope_authority(owner_user_id,dataset_id) VALUES (?,?)",
            (OWNER_ID, DATASET_ID),
        )
    _publish(db, suggestion_id="local-mutation", kind=kind)
    return db.note_graph_suggestion_store.claim_acceptance(
        dataset_id=DATASET_ID,
        suggestion_id="local-mutation",
        expected_revision=1,
        expected_source_fingerprint=_fingerprint(db, SOURCE_ID),
        expected_target_fingerprint=_fingerprint(db, TARGET_ID) if kind == "related_note" else None,
        idempotency_key="local-mutation-decision",
        now=NOW,
    ).suggestion


def _guard(db, fence, domain, identity, *, fail_after=False):
    finalized = []

    def before(conn):
        db.note_graph_suggestion_store.guard_acceptance_in_transaction(conn=conn, fence=fence, now=NOW)

    def after(conn, result_id):
        finalized.append(
            db.note_graph_suggestion_store.finalize_acceptance_in_transaction(
                conn=conn,
                fence=fence,
                accepted_resource_identity=result_id,
                now=NOW,
            )
        )
        if fail_after:
            raise RuntimeError("synthetic finalizer interruption")

    return GuardedProductMutation(
        expected_domain=domain,
        expected_object_id=identity,
        before=before,
        after=None if domain == "notes.keyword" else after,
    ), finalized


def test_fresh_inactive_factory_provides_real_decisions(local_product_db):
    db = local_product_db
    decisions = build_suggestion_decision_service(
        note_db=db,
        owner_user_id=OWNER_ID,
        dataset_id=f"legacy:{OWNER_ID}",
    )
    assert decisions is not None


def test_local_adapter_commits_link_and_exact_acceptance_together(local_product_db):
    db = local_product_db
    fence = _claim(db)
    guard, finalized = _guard(db, fence, "notes.link", EDGE_ID)
    link = _adapter(db).create_related_link(
        edge_id=EDGE_ID,
        source_note_id=TARGET_ID,
        target_note_id=SOURCE_ID,
        guarded_mutation=guard,
    )
    assert (link.edge_id, link.directed, link.weight, link.label, link.properties) == (
        EDGE_ID,
        False,
        1.0,
        None,
        {},
    )
    assert finalized[0].envelope["state"] == "accepted"
    assert (
        db.note_graph_suggestion_store.get_suggestion(dataset_id=DATASET_ID, suggestion_id=fence.id).state.value
        == "accepted"
    )


def test_local_adapter_finalizer_failure_rolls_back_product_and_decision(local_product_db):
    db = local_product_db
    fence = _claim(db)
    guard, _ = _guard(db, fence, "notes.link", EDGE_ID, fail_after=True)
    with pytest.raises(RuntimeError, match="synthetic finalizer interruption"):
        _adapter(db).create_related_link(
            edge_id=EDGE_ID,
            source_note_id=SOURCE_ID,
            target_note_id=TARGET_ID,
            guarded_mutation=guard,
        )
    assert db.notes_link_store.get(EDGE_ID) is None
    assert (
        db.note_graph_suggestion_store.get_suggestion(dataset_id=DATASET_ID, suggestion_id=fence.id).state.value
        == "accepting"
    )


def test_local_adapter_keyword_alone_never_finalizes_acceptance(local_product_db):
    db = local_product_db
    fence = _claim(db, kind="tag")
    adapter = _adapter(db)
    keyword_guard, _ = _guard(db, fence, "notes.keyword", KEYWORD_ID)
    resource = adapter.create_keyword(keyword_sync_id=KEYWORD_ID, display="Research", guarded_mutation=keyword_guard)
    assert resource.sync_id == KEYWORD_ID
    assert db.get_keywords_for_note(SOURCE_ID) == []
    assert (
        db.note_graph_suggestion_store.get_suggestion(dataset_id=DATASET_ID, suggestion_id=fence.id).state.value
        == "accepting"
    )
    identity = organization_link_id("notes.keyword_link", ["note", SOURCE_ID, KEYWORD_ID])
    membership_guard, finalized = _guard(db, fence, "notes.keyword_link", identity)
    adapter.link_keyword(note_id=SOURCE_ID, keyword_sync_id=KEYWORD_ID, guarded_mutation=membership_guard)
    assert finalized[0].envelope["state"] == "accepted"
    assert [row["sync_id"] for row in db.get_keywords_for_note(SOURCE_ID)] == [KEYWORD_ID]


def test_local_adapter_identity_mismatch_cannot_create_product(local_product_db):
    db = local_product_db
    fence = _claim(db)
    guard, _ = _guard(db, fence, "notes.link", EDGE_ID)
    with pytest.raises(GuardedProductMutationIdentityError):
        _adapter(db).create_related_link(
            edge_id=KEYWORD_ID,
            source_note_id=SOURCE_ID,
            target_note_id=TARGET_ID,
            guarded_mutation=guard,
        )
    assert db.notes_link_store.snapshot() == ()


def test_local_adapter_preserves_outer_caller_rollback(local_product_db):
    db = local_product_db
    fence = _claim(db)
    guard, _ = _guard(db, fence, "notes.link", EDGE_ID)
    with pytest.raises(RuntimeError, match="caller rollback"):
        with db.transaction():
            _adapter(db).create_related_link(
                edge_id=EDGE_ID,
                source_note_id=SOURCE_ID,
                target_note_id=TARGET_ID,
                guarded_mutation=guard,
            )
            raise RuntimeError("caller rollback")
    assert db.notes_link_store.get(EDGE_ID) is None
    assert (
        db.note_graph_suggestion_store.get_suggestion(dataset_id=DATASET_ID, suggestion_id=fence.id).state.value
        == "accepting"
    )


def test_local_adapter_failed_membership_preserves_only_prior_keyword(local_product_db):
    db = local_product_db
    fence = _claim(db, kind="tag")
    adapter = _adapter(db)
    keyword_guard, _ = _guard(db, fence, "notes.keyword", KEYWORD_ID)
    adapter.create_keyword(keyword_sync_id=KEYWORD_ID, display="Research", guarded_mutation=keyword_guard)
    identity = organization_link_id("notes.keyword_link", ["note", SOURCE_ID, KEYWORD_ID])
    membership_guard, _ = _guard(db, fence, "notes.keyword_link", identity, fail_after=True)
    with pytest.raises(RuntimeError, match="synthetic finalizer interruption"):
        adapter.link_keyword(note_id=SOURCE_ID, keyword_sync_id=KEYWORD_ID, guarded_mutation=membership_guard)
    assert db.get_keyword_by_text("Research")["sync_id"] == KEYWORD_ID
    assert db.get_keywords_for_note(SOURCE_ID) == []
    assert (
        db.note_graph_suggestion_store.get_suggestion(dataset_id=DATASET_ID, suggestion_id=fence.id).state.value
        == "accepting"
    )


def test_local_adapter_keyword_replay_preserves_identity_and_version(local_product_db):
    db = local_product_db
    fence = _claim(db, kind="tag")
    adapter = _adapter(db)
    guard, _ = _guard(db, fence, "notes.keyword", KEYWORD_ID)
    first = adapter.create_keyword(keyword_sync_id=KEYWORD_ID, display="Research", guarded_mutation=guard)
    replay = adapter.create_keyword(keyword_sync_id=KEYWORD_ID, display="Research", guarded_mutation=guard)
    assert (replay.local_id, replay.sync_id, replay.version) == (first.local_id, KEYWORD_ID, first.version)
    assert len(db.list_keywords()) == 1


def test_local_adapter_reports_normalized_collision_without_replacing_keyword(local_product_db):
    db = local_product_db
    existing_id = db.add_keyword("Research")
    existing = db.get_keyword_by_id(existing_id)
    fence = _claim(db, kind="tag")
    guard, _ = _guard(db, fence, "notes.keyword", KEYWORD_ID)
    with pytest.raises(GuardedKeywordIdentityCollision) as caught:
        _adapter(db).create_keyword(keyword_sync_id=KEYWORD_ID, display="RESEARCH", guarded_mutation=guard)
    assert caught.value.canonical_sync_id == existing["sync_id"]
    assert db.get_keyword_by_id(existing_id) == existing
    assert len(db.list_keywords()) == 1
    assert db.get_keywords_for_note(SOURCE_ID) == []


def test_local_adapter_same_tag_label_is_owner_scoped(local_product_db, tmp_path):
    db = local_product_db
    # PostgreSQL owners share physical tables; SQLite retains its per-owner file.
    other = CharactersRAGDB(
        tmp_path / "foreign-owner.db",
        client_id="other-owner",
        backend=db.backend if db.backend_type == BackendType.POSTGRESQL else None,
    )
    try:
        foreign_local_id = other.add_keyword("Research")
        foreign = other.get_keyword_by_id(foreign_local_id)
        fence = _claim(db, kind="tag")
        adapter = _adapter(db)
        keyword_guard, _ = _guard(db, fence, "notes.keyword", KEYWORD_ID)
        owned = adapter.create_keyword(keyword_sync_id=KEYWORD_ID, display="research", guarded_mutation=keyword_guard)
        assert owned.sync_id != foreign["sync_id"]
        foreign_identity = organization_link_id("notes.keyword_link", ["note", SOURCE_ID, foreign["sync_id"]])
        foreign_guard, _ = _guard(db, fence, "notes.keyword_link", foreign_identity)
        with pytest.raises(InputError, match="missing or deleted"):
            adapter.link_keyword(note_id=SOURCE_ID, keyword_sync_id=foreign["sync_id"], guarded_mutation=foreign_guard)
        assert db.get_keywords_for_note(SOURCE_ID) == []
        owned_identity = organization_link_id("notes.keyword_link", ["note", SOURCE_ID, KEYWORD_ID])
        owned_guard, finalized = _guard(db, fence, "notes.keyword_link", owned_identity)
        adapter.link_keyword(note_id=SOURCE_ID, keyword_sync_id=KEYWORD_ID, guarded_mutation=owned_guard)
        assert finalized[0].envelope["state"] == "accepted"
        assert [row["sync_id"] for row in db.get_keywords_for_note(SOURCE_ID)] == [KEYWORD_ID]
        assert other.get_keyword_by_id(foreign_local_id) == foreign
    finally:
        other.close_all_connections()
