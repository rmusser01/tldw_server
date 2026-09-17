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
