"""Actual published evidence serialization, filtering and strict schema controls."""

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from tldw_Server_API.app.api.v1.endpoints import notes_graph_suggestions as endpoint
from tldw_Server_API.app.api.v1.schemas.notes_graph_suggestions import (
    SuggestionEvidenceResponse,
    SuggestionRunCreateRequest,
)
from tldw_Server_API.app.core.DB_Management.chacha.operation_scope import chacha_operation
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.tests.Notes_Graph.integration.test_suggestion_fresh_reads import (
    fresh_suggestions as fresh_suggestions,
)
from tldw_Server_API.tests.Notes_Graph.unit import test_suggestion_store as stored

pytestmark = pytest.mark.integration


def _published(fixture, monkeypatch, *, unbound=False):
    db, client, _, _ = fixture
    monkeypatch.setattr(stored, "DATASET_ID", "legacy:1")
    with chacha_operation(independent=True):
        with db.transaction() as conn:
            conn.execute(
                "INSERT INTO note_task_scope_authority(owner_user_id,dataset_id) VALUES (?,?)", ("1", "legacy:1")
            )
        db.add_note("Source", "source body", note_id=stored.SOURCE_ID)
        db.add_note("Target", "target body", note_id=stored.TARGET_ID)
        stored._stage_and_activate(db, key="publish-evidence", suggestion_id="evidence-visible")
        if unbound:
            with db.transaction() as conn:
                conn.execute("DELETE FROM note_task_scope_authority WHERE owner_user_id=?", ("1",))
    return db, client, f"/api/v1/notes/{stored.SOURCE_ID}/graph/suggestions?state=pending&limit=20"


@pytest.mark.parametrize("unbound", [False, True], ids=["registered", "unbound"])
def test_populated_evidence_survives_real_router_serialization(fresh_suggestions, monkeypatch, unbound):
    _, client, url = _published(fresh_suggestions, monkeypatch, unbound=unbound)
    response = client.get(url)
    assert response.status_code == 200
    item = response.json()["items"][0]
    assert item["id"] == "evidence-visible" and item["target_title"] == "Target"
    assert item["evidence"] == [
        {
            "side": "source",
            "note_id": stored.SOURCE_ID,
            "field": "content",
            "start_offset": 0,
            "end_offset": 6,
            "text": "source",
        },
        {
            "side": "target",
            "note_id": stored.TARGET_ID,
            "field": "content",
            "start_offset": 0,
            "end_offset": 6,
            "text": "target",
        },
    ]


@pytest.mark.parametrize("target_state", ["range_changed", "deleted", "fingerprint_changed", "oversized"])
def test_filtered_target_evidence_is_not_serialized(fresh_suggestions, monkeypatch, target_state):
    db, client, url = _published(fresh_suggestions, monkeypatch)
    with chacha_operation(independent=True), db.transaction() as conn:
        if target_state == "range_changed":
            conn.execute("UPDATE note_graph_suggestion_evidence SET end_offset=? WHERE side=?", (999, "target"))
        elif target_state == "deleted":
            conn.execute("UPDATE notes SET deleted=? WHERE id=?", (True, stored.TARGET_ID))
        else:
            content = "changed" if target_state == "fingerprint_changed" else "x" * 1_000_001
            conn.execute("UPDATE notes SET content=? WHERE id=?", (content, stored.TARGET_ID))
    response = client.get(url)
    assert response.status_code == 200
    item = response.json()["items"][0]
    assert item["target_title"] is None
    assert [evidence["side"] for evidence in item["evidence"]] == ["source"]


@pytest.mark.parametrize("source_state,expected", [("foreign", 404), ("fingerprint_changed", 200)])
def test_source_scope_and_fingerprint_remain_required(fresh_suggestions, monkeypatch, source_state, expected):
    db, client, url = _published(fresh_suggestions, monkeypatch)
    if source_state == "foreign":
        other = CharactersRAGDB(
            db.db_path.parent / "foreign-owner.db",
            client_id="2",
            backend=db.backend if db.backend_type.value == "postgresql" else None,
        )
        client.app.dependency_overrides[endpoint.get_chacha_db_for_user] = lambda: other
        client.app.dependency_overrides[endpoint.get_request_user] = lambda: SimpleNamespace(id=2, id_str="2")
        try:
            response = client.get(url)
        finally:
            other.close_all_connections()
    else:
        with chacha_operation(independent=True), db.transaction() as conn:
            conn.execute("UPDATE notes SET content=? WHERE id=?", ("new source", stored.SOURCE_ID))
        response = client.get(url)
    assert response.status_code == expected
    if expected == 200:
        assert response.json()["items"] == []


@pytest.mark.parametrize(
    "changes",
    [
        {"side": "other"},
        {"field": "password"},
        {"start_offset": -1},
        {"end_offset": 0},
        {"text": "x" * 481},
        {"text": None},
        {"extra": "unexpected"},
    ],
)
def test_evidence_schema_remains_bounded_and_closed(changes):
    payload = {"side": "source", "note_id": "note", "field": "content", "start_offset": 0, "end_offset": 1, "text": "x"}
    with pytest.raises(ValidationError):
        SuggestionEvidenceResponse.model_validate({**payload, **changes})


def test_request_schema_still_rejects_response_fields():
    with pytest.raises(ValidationError):
        SuggestionRunCreateRequest.model_validate({"evidence": []})
