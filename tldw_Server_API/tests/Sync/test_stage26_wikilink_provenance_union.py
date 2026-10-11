"""Keep the released owner-bound wikilink guard with the default provenance registry."""

import pytest

from tldw_Server_API.app.core.Sync.v2.models import SyncEnvelopeCreate
from tldw_Server_API.tests.Sync.test_sync_v2_notes_provenance_lifecycle import env as env
from tldw_Server_API.tests.Sync.test_sync_v2_notes_provenance_lifecycle import save


@pytest.mark.parametrize("race", [False, True])
def test_default_registry_keeps_wikilink_product_guard_and_provenance(env, monkeypatch, race):
    service, db = env
    save(env)
    retained = db.note_provenance_store.get("note")
    state = service.store.get_object_state("personal", "notes.note", "note")
    envelope = service.store.insert_envelope(
        SyncEnvelopeCreate(
            dataset_id="personal",
            client_envelope_id="wikilink-guard",
            domain="notes.note",
            operation="upsert",
            object_id="note",
            device_id="server-origin",
            object_revision=2,
            base_server_cursor=state.latest_server_cursor,
            base_object_revision=state.object_revision,
            base_object_hash=state.object_hash,
            payload={"title": "Title", "content": "Rewritten [[New title]]."},
            payload_hash="sha256:rewritten",
            routing_metadata={
                "source": "notes-wikilink-rewrite",
                "origin": "server",
                "server_device_id": "server-origin",
                "server_owner_user_id": "alice",
                "notes_ingestion_expected_product_version": 1,
            },
        )
    )
    upsert = db.upsert_note_from_sync

    def concurrent_edit(**kwargs):
        db.update_note("note", {"content": "Concurrent canonical edit"}, expected_version=1)
        return upsert(**kwargs)

    with monkeypatch.context() as scoped:
        if race:
            scoped.setattr(db, "upsert_note_from_sync", concurrent_edit)
        result = service._materialize_envelope(envelope)
    assert result.status == ("conflict" if race else "applied")
    assert db.get_note_by_id("note")["content"] == ("Concurrent canonical edit" if race else "Rewritten [[New title]].")
    assert db.note_provenance_store.get("note") == retained
    if race:
        assert service._materialize_envelope(service._envelope_snapshot(envelope)).status == "conflict"
        assert db.get_note_by_id("note")["content"] == "Concurrent canonical edit"


@pytest.mark.parametrize("env", ["sqlite", pytest.param("postgres", marks=pytest.mark.postgres)], indirect=True)
@pytest.mark.parametrize("stale", [False, True])
def test_guarded_note_provenance_pair_retains_conflict_after_product_rollback(env, stale):
    from tldw_Server_API.app.core.Sync.v2.notes_provenance import provenance_step
    from tldw_Server_API.app.core.Sync.v2.server_origin_batch import (
        ServerOriginMutationStep,
        SyncServerOriginBatchMaterializationError,
        capture_server_origin_mutation_batch,
    )

    service, db = env
    save(env)
    retained = db.note_provenance_store.get("note")
    if stale:
        db.update_note("note", {"content": "Concurrent canonical edit"}, expected_version=1)
    dataset = service.store.get_dataset("personal")
    steps = [
        ServerOriginMutationStep(
            domain="notes.note",
            operation="upsert",
            object_id="note",
            payload={"title": "Title", "content": "Rewritten [[New title]]."},
            routing_metadata={"notes_ingestion_expected_product_version": 1},
        ),
        provenance_step(
            service=service,
            dataset=dataset,
            note_id="note",
            payload={"origin": "knowledge_qa", "trust_state": "cited_answer"},
            expected_version=1,
        ),
    ]
    args = {
        "service": service,
        "user_id": "alice",
        "steps": steps,
        "source": "notes-wikilink-rewrite",
        "idempotency_key": "paired-wikilink",
    }
    if not stale:
        result = capture_server_origin_mutation_batch(**args)
        assert result.fully_applied
        assert len(result.envelopes) == 2
        assert db.get_note_by_id("note")["content"] == "Rewritten [[New title]]."
        assert db.note_provenance_store.get("note")["version"] == 2
        return
    for _ in range(2):
        with pytest.raises(SyncServerOriginBatchMaterializationError) as error:
            capture_server_origin_mutation_batch(**args)
        assert error.value.retryable is False
        assert len(error.value.result.envelopes) == 2
        assert all(member.apply_status == "conflict" for member in error.value.result.envelopes)
        assert db.get_note_by_id("note")["content"] == "Concurrent canonical edit"
        assert db.note_provenance_store.get("note") == retained
        assert service.store.get_object_state("personal", "notes.note", "note").object_revision == 1
        assert service.store.get_object_state("personal", "notes.provenance", "note").object_revision == 1


@pytest.mark.parametrize("version,owner", [(None, "alice"), (True, "alice"), (0, "alice"), (1, "other")])
def test_default_registry_rejects_invalid_wikilink_guard_with_provenance(env, version, owner):
    service, db = env
    save(env)
    retained = db.note_provenance_store.get("note")
    state = service.store.get_object_state("personal", "notes.note", "note")
    routing = {
        "source": "notes-wikilink-rewrite",
        "origin": "server",
        "server_device_id": "server-origin",
        "server_owner_user_id": owner,
    }
    if version is not None:
        routing["notes_ingestion_expected_product_version"] = version
    envelope = service.store.insert_envelope(
        SyncEnvelopeCreate(
            dataset_id="personal",
            client_envelope_id="invalid-wikilink-guard",
            domain="notes.note",
            operation="upsert",
            object_id="note",
            device_id="server-origin",
            object_revision=2,
            base_server_cursor=state.latest_server_cursor,
            base_object_revision=state.object_revision,
            base_object_hash=state.object_hash,
            payload={"title": "Must not write", "content": "Invalid"},
            payload_hash="sha256:invalid",
            routing_metadata=routing,
        )
    )
    assert service._materialize_envelope(envelope).status == "failed"
    assert service._materialize_envelope(service._envelope_snapshot(envelope)).status == "failed"
    assert (db.get_note_by_id("note")["content"], db.get_note_by_id("note")["version"]) == ("Body", 1)
    assert db.note_provenance_store.get("note") == retained
