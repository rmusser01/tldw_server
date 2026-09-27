"""Actual local publication/decision controls for durable merge survivors."""

import pytest

from tldw_Server_API.app.core.Notes_Graph.suggestion_service import build_suggestion_decision_service
from tldw_Server_API.tests.Notes_Graph.integration import test_suggestion_acceptance as acceptance_fixture
from tldw_Server_API.tests.Notes_Graph.integration.test_suggestion_legacy_mutations import (
    local_product_db as local_product_db,
)

pytestmark = pytest.mark.integration


def _setup(db, monkeypatch):
    dataset = f"legacy:{db.client_id}"
    monkeypatch.setattr(acceptance_fixture, "DATASET_ID", dataset)
    source = db.get_keyword_by_id(db.add_keyword("Before merge"))
    target = db.get_keyword_by_id(db.add_keyword("Surviving keyword"))
    return dataset, source, target


def _merge(db, source, target):
    return db.merge_keywords(
        source_keyword_id=source["id"],
        target_keyword_id=target["id"],
        expected_source_version=source["version"],
        expected_target_version=target["version"],
    )


def _accept(db, dataset, suggestion_id):
    decisions = build_suggestion_decision_service(note_db=db, owner_user_id=db.client_id, dataset_id=dataset)
    assert decisions is not None
    decisions._clock = lambda: acceptance_fixture.NOW
    return decisions.accept(
        dataset_id=dataset,
        suggestion_id=suggestion_id,
        expected_revision=1,
        expected_source_fingerprint=acceptance_fixture._fingerprint(db, acceptance_fixture.SOURCE_ID),
        expected_target_fingerprint=None,
        idempotency_key=f"accept-{suggestion_id}",
    )


def test_pending_local_tag_follows_real_merge_at_acceptance(local_product_db, monkeypatch):
    db = local_product_db
    dataset, source, target = _setup(db, monkeypatch)
    acceptance_fixture._publish(
        db,
        suggestion_id="merge-pending",
        kind="tag",
        keyword_sync_id=source["sync_id"],
        normalized_tag="before merge",
        display_tag="Before merge",
    )
    _merge(db, source, target)
    result = _accept(db, dataset, "merge-pending")
    assert result.envelope["state"] == "accepted"
    assert [row["sync_id"] for row in db.get_keywords_for_note(acceptance_fixture.SOURCE_ID)] == [target["sync_id"]]


def test_inflight_local_tag_snapshot_survives_merge_before_publication(local_product_db, monkeypatch):
    db = local_product_db
    dataset, source, target = _setup(db, monkeypatch)
    original = db.note_graph_suggestion_store.stage_suggestions

    def merge_after_candidate_snapshot(**kwargs):
        # The generated candidate has already captured the old portable ID.
        # Interleave a real ordinary merge before the actual stage/publish path.
        assert kwargs["candidates"][0]["keyword_sync_id"] == source["sync_id"]
        _merge(db, source, target)
        return original(**kwargs)

    monkeypatch.setattr(db.note_graph_suggestion_store, "stage_suggestions", merge_after_candidate_snapshot)
    acceptance_fixture._publish(
        db,
        suggestion_id="merge-inflight",
        kind="tag",
        keyword_sync_id=source["sync_id"],
        normalized_tag="before merge",
        display_tag="Before merge",
    )
    page = db.note_graph_suggestion_store.list_suggestions(
        dataset_id=dataset,
        source_note_id=acceptance_fixture.SOURCE_ID,
        source_fingerprint=acceptance_fixture._fingerprint(db, acceptance_fixture.SOURCE_ID),
        states=("pending",),
        limit=20,
        after=None,
    )
    assert len(page.items) == 1
    assert page.items[0].display_tag == target["keyword"]
    assert page.items[0].keyword_sync_id == source["sync_id"]
    assert _accept(db, dataset, "merge-inflight").envelope["state"] == "accepted"
    assert [row["sync_id"] for row in db.get_keywords_for_note(acceptance_fixture.SOURCE_ID)] == [target["sync_id"]]


def test_local_acceptance_rechecks_survivor_after_source_restore(local_product_db, monkeypatch):
    db = local_product_db
    dataset, source, target = _setup(db, monkeypatch)
    acceptance_fixture._publish(db, suggestion_id="merge-restore-race", kind="tag", keyword_sync_id=source["sync_id"])
    _merge(db, source, target)
    decisions = build_suggestion_decision_service(note_db=db, owner_user_id=db.client_id, dataset_id=dataset)
    decisions._clock = lambda: acceptance_fixture.NOW
    original = decisions.local.link_keyword

    def restore_before_product_guard(**kwargs):
        assert kwargs["keyword_sync_id"] == target["sync_id"]
        assert db.add_keyword(source["keyword"]) == source["id"]
        return original(**kwargs)

    monkeypatch.setattr(decisions.local, "link_keyword", restore_before_product_guard)
    result = decisions.accept(
        dataset_id=dataset,
        suggestion_id="merge-restore-race",
        expected_revision=1,
        expected_source_fingerprint=acceptance_fixture._fingerprint(db, acceptance_fixture.SOURCE_ID),
        expected_target_fingerprint=None,
        idempotency_key="restore-race",
    )
    assert result.envelope["state"] == "pending"
    assert db.get_keywords_for_note(acceptance_fixture.SOURCE_ID) == []


@pytest.mark.parametrize("local_product_db", ["sqlite"], indirect=True)
@pytest.mark.parametrize("candidate", ["existing", "merged", "new-same-label", "new-unicode-label"])
def test_sqlite_local_acceptance_preserves_prior_device_keyword_identity(local_product_db, monkeypatch, candidate):
    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

    db = local_product_db
    prior = CharactersRAGDB(db.db_path, client_id="prior-device")
    try:
        source = prior.get_keyword_by_id(
            prior.add_keyword("Café STRASSE" if candidate == "new-unicode-label" else "Prior Research")
        )
        target = prior.get_keyword_by_id(prior.add_keyword("Surviving Research")) if candidate == "merged" else source
        if candidate == "merged":
            _merge(prior, source, target)
    finally:
        prior.close_all_connections()
    dataset = f"legacy:{db.client_id}"
    monkeypatch.setattr(acceptance_fixture, "DATASET_ID", dataset)
    acceptance_fixture._publish(
        db,
        suggestion_id="prior-device-tag",
        kind="tag",
        keyword_sync_id=None if candidate.startswith("new-") else source["sync_id"],
        normalized_tag="café strasse" if candidate == "new-unicode-label" else "prior research",
        display_tag="Cafe\u0301 Straße" if candidate == "new-unicode-label" else "Prior Research",
    )
    before = db.get_keyword_by_id(target["id"])
    result = _accept(db, dataset, "prior-device-tag")
    assert result.envelope["state"] == "accepted"
    assert [row["sync_id"] for row in db.get_keywords_for_note(acceptance_fixture.SOURCE_ID)] == [target["sync_id"]]
    assert db.get_keyword_by_id(target["id"]) == before


@pytest.mark.parametrize("local_product_db", ["sqlite"], indirect=True)
def test_sqlite_local_publication_suppresses_already_linked_prior_device_keyword(local_product_db, monkeypatch):
    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

    db = local_product_db
    prior = CharactersRAGDB(db.db_path, client_id="prior-device")
    try:
        keyword = prior.get_keyword_by_id(prior.add_keyword("Prior Research"))
        prior.link_note_to_keyword(acceptance_fixture.SOURCE_ID, keyword["id"])
    finally:
        prior.close_all_connections()
    dataset = f"legacy:{db.client_id}"
    monkeypatch.setattr(acceptance_fixture, "DATASET_ID", dataset)
    acceptance_fixture._publish(
        db,
        suggestion_id="prior-device-duplicate",
        kind="tag",
        keyword_sync_id=keyword["sync_id"],
        normalized_tag="prior research",
        display_tag="Prior Research",
    )
    page = db.note_graph_suggestion_store.list_suggestions(
        dataset_id=dataset,
        source_note_id=acceptance_fixture.SOURCE_ID,
        source_fingerprint=acceptance_fixture._fingerprint(db, acceptance_fixture.SOURCE_ID),
        states=("pending",),
        limit=20,
        after=None,
    )
    assert not page.items
    assert [row["sync_id"] for row in db.get_keywords_for_note(acceptance_fixture.SOURCE_ID)] == [keyword["sync_id"]]


@pytest.mark.parametrize("local_product_db", ["sqlite"], indirect=True)
def test_prior_device_keyword_membership_and_finalizer_roll_back_together(local_product_db, monkeypatch):
    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

    db = local_product_db
    prior = CharactersRAGDB(db.db_path, client_id="prior-device")
    try:
        keyword = prior.get_keyword_by_id(prior.add_keyword("Prior Research"))
    finally:
        prior.close_all_connections()
    dataset = f"legacy:{db.client_id}"
    monkeypatch.setattr(acceptance_fixture, "DATASET_ID", dataset)
    acceptance_fixture._publish(
        db,
        suggestion_id="prior-device-rollback",
        kind="tag",
        keyword_sync_id=keyword["sync_id"],
        normalized_tag="prior research",
        display_tag="Prior Research",
    )
    store = db.note_graph_suggestion_store
    original = store.finalize_acceptance_in_transaction

    def fail_after_finalization(**kwargs):
        assert original(**kwargs).envelope["state"] == "accepted"
        raise RuntimeError("synthetic finalizer interruption")

    monkeypatch.setattr(store, "finalize_acceptance_in_transaction", fail_after_finalization)
    with pytest.raises(RuntimeError, match="synthetic finalizer interruption"):
        _accept(db, dataset, "prior-device-rollback")
    assert db.get_keywords_for_note(acceptance_fixture.SOURCE_ID) == []
    assert db.get_keyword_by_id(keyword["id"]) == keyword
    assert store.get_suggestion(dataset_id=dataset, suggestion_id="prior-device-rollback").state.value == "accepting"
