"""Fresh read eligibility and capability authority through real storage/routes."""

from types import SimpleNamespace

import pytest

from tldw_Server_API.app.api.v1.endpoints import notes_graph_suggestions as endpoint
from tldw_Server_API.app.core.DB_Management.chacha.note_graph_suggestion_store import (
    NotesGraphDatasetScopeError,
    NotesGraphFTSNotReadyError,
)
from tldw_Server_API.app.core.DB_Management.chacha.operation_scope import chacha_operation
from tldw_Server_API.app.core.Notes_Graph import suggestion_provider, suggestion_service
from tldw_Server_API.app.core.Notes_Graph.suggestion_content import content_fingerprint
from tldw_Server_API.tests.Notes_Graph.integration.test_suggestion_fresh_reads import (
    fresh_suggestions as fresh_suggestions,
)
from tldw_Server_API.tests.Notes_Graph.unit import test_suggestion_store as stored

pytestmark = pytest.mark.integration


def _register(db, owner="1", dataset="legacy:1"):
    with chacha_operation(independent=True), db.transaction() as conn:
        conn.execute(
            "INSERT INTO note_task_scope_authority(owner_user_id,dataset_id) VALUES (?,?)",
            (owner, dataset),
        )


def _counts(db):
    with chacha_operation(independent=True), db.transaction() as conn:
        return tuple(
            conn.execute(query).fetchone()["n"]
            for query in (
                "SELECT COUNT(*) AS n FROM note_task_scope_authority",
                "SELECT COUNT(*) AS n FROM note_graph_suggestion_runs",
                "SELECT COUNT(*) AS n FROM note_graph_suggestions",
                "SELECT COUNT(*) AS n FROM note_graph_suggestion_operation_receipts",
            )
        )


def _ready_provider(monkeypatch):
    # Resolve actual audited provider capability; never call its transport.
    monkeypatch.setattr(suggestion_provider, "get_default_provider", lambda: "openai")
    monkeypatch.setattr(suggestion_provider, "get_default_model_for_provider", lambda _: "fixture-model")
    monkeypatch.setattr(suggestion_provider, "loaded_config_data", {"openai_api": {"api_key": "synthetic-test-only"}})
    resolved = suggestion_provider.resolve_generation_capability(provider=None, model=None)
    assert resolved.capabilities.generation_available
    return resolved.capabilities


@pytest.mark.parametrize(
    "binding_owner,binding_dataset,requested,allowed",
    [
        (None, None, "legacy:1", True),
        (None, None, "legacy:2", False),
        (None, None, "arbitrary", False),
        ("2", "dataset-other", "legacy:1", True),
        ("1", "dataset-one", "legacy:1", False),
        ("1", "dataset-one", "dataset-one", True),
        ("1", "legacy:1", "legacy:1", True),
    ],
)
def test_read_scope_uses_exact_owner_and_preserves_authority(
    fresh_suggestions, binding_owner, binding_dataset, requested, allowed
):
    db, _, note, _ = fresh_suggestions
    if binding_owner:
        _register(db, binding_owner, binding_dataset)
    before = _counts(db)
    with chacha_operation(independent=True):
        store = db.note_graph_suggestion_store
        if allowed:
            source = store.load_source_note(dataset_id=requested, note_id=note)
            assert source.note_id == note
            assert store.ensure_fts_ready(dataset_id=requested) is None
            fingerprint = content_fingerprint(source.title, source.content)
            assert (
                store.list_suggestions(
                    dataset_id=requested,
                    source_note_id=note,
                    source_fingerprint=fingerprint,
                    states=("pending",),
                    limit=20,
                    after=None,
                ).items
                == ()
            )
            assert (
                store.list_suggestion_evidence(
                    dataset_id=requested,
                    source_note_id=note,
                    source_fingerprint=fingerprint,
                    suggestion_ids=("missing",),
                    limit=6,
                )
                == ()
            )
            assert (
                store.get_rejection_set(dataset_id=requested, source_note_id=note, source_fingerprint=fingerprint)
                is None
            )
        else:
            with pytest.raises(NotesGraphDatasetScopeError):
                store.load_source_note(dataset_id=requested, note_id=note)
    assert _counts(db) == before


@pytest.mark.parametrize("kind,expected", [("missing", 404), ("foreign", 404), ("deleted", 404), ("oversized", 422)])
def test_fresh_route_retains_source_safety(fresh_suggestions, kind, expected):
    db, client, note, _ = fresh_suggestions
    with chacha_operation(independent=True), db.transaction() as conn:
        if kind == "foreign":
            conn.execute("UPDATE notes SET client_id=? WHERE id=?", ("2", note))
        elif kind == "deleted":
            conn.execute("UPDATE notes SET deleted=? WHERE id=?", (True, note))
        elif kind == "oversized":
            conn.execute("UPDATE notes SET content=? WHERE id=?", ("x" * 1_000_001, note))
        else:
            note = "missing-note"
    response = client.get(f"/api/v1/notes/{note}/graph/suggestions/capabilities")
    assert response.status_code == expected


def test_missing_decision_authority_removes_actions_and_changes_etag(fresh_suggestions, monkeypatch):
    monkeypatch.setattr(suggestion_service, "build_suggestion_decision_service", lambda **_: None)
    db, client, note, _ = fresh_suggestions
    base = _ready_provider(monkeypatch)
    monkeypatch.setenv("NOTES_GRAPH_SUGGESTIONS_WORKER_ENABLED", "1")
    client.app.dependency_overrides[endpoint.try_get_job_manager] = lambda: SimpleNamespace()
    before = _counts(db)
    first = client.get(f"/api/v1/notes/{note}/graph/suggestions/capabilities")
    again = client.get(f"/api/v1/notes/{note}/graph/suggestions/capabilities")
    assert first.status_code == 200
    value = first.json()
    assert value["generation_available"] is False
    assert value["unavailable_reason"] == "notes_graph_sync_not_ready"
    assert value["allowed_actions"] == []
    assert value["revision"] != base.revision
    assert first.headers["etag"] == f'"{value["revision"]}"' == again.headers["etag"]
    assert again.json() == value
    assert _counts(db) == before


def test_readiness_precedence_and_revision_transition(fresh_suggestions, monkeypatch):
    monkeypatch.setattr(suggestion_service, "build_suggestion_decision_service", lambda **_: None)
    db, client, note, _ = fresh_suggestions
    _register(db)
    base = _ready_provider(monkeypatch)
    monkeypatch.setenv("NOTES_GRAPH_SUGGESTIONS_WORKER_ENABLED", "1")
    client.app.dependency_overrides[endpoint.try_get_job_manager] = lambda: SimpleNamespace()
    url = f"/api/v1/notes/{note}/graph/suggestions/capabilities"
    no_decisions = client.get(url).json()
    client.app.dependency_overrides[endpoint.try_get_job_manager] = lambda: None
    no_worker = client.get(url).json()
    assert no_worker["unavailable_reason"] == "notes_graph_suggestions_worker_unavailable"
    assert no_decisions["allowed_actions"] == ["cancel"]
    assert no_worker["allowed_actions"] == ["cancel"]
    assert no_worker["revision"] != no_decisions["revision"]
    monkeypatch.setattr(suggestion_service, "build_suggestion_decision_service", lambda **_: SimpleNamespace())
    canonical_worker = client.get(url).json()
    assert canonical_worker["unavailable_reason"] == "notes_graph_suggestions_worker_unavailable"
    assert canonical_worker["revision"] == base.revision
    assert canonical_worker["allowed_actions"] == list(base.allowed_actions)
    client.app.dependency_overrides[endpoint.try_get_job_manager] = lambda: SimpleNamespace()
    ready = client.get(url).json()
    assert ready["generation_available"] is True and ready["revision"] == base.revision


def test_missing_coordinator_blocks_http_admission_and_retired_scope_blocks_store_writes(
    fresh_suggestions, monkeypatch
):
    monkeypatch.setattr(suggestion_service, "build_suggestion_decision_service", lambda **_: None)
    db, client, note, _ = fresh_suggestions
    _ready_provider(monkeypatch)
    monkeypatch.setenv("NOTES_GRAPH_SUGGESTIONS_WORKER_ENABLED", "1")

    class NoJobs:
        def get_job_or_archived_by_idempotency_key(self, **scope):
            # The established coordinator checks idempotency before readiness.
            return None

        def create_job(self, **kwargs):
            raise AssertionError("unavailable coordinator must not enqueue")

    client.app.dependency_overrides[endpoint.try_get_job_manager] = lambda: NoJobs()
    before = _counts(db)
    response = client.get(f"/api/v1/notes/{note}/graph/suggestions/capabilities")
    assert response.status_code == 200
    admission = client.post(
        f"/api/v1/notes/{note}/graph/suggestions/runs",
        json={},
        headers={"If-Match": response.headers["etag"], "Idempotency-Key": "unbound-rejected"},
    )
    assert admission.status_code == 503
    with chacha_operation(independent=True), db.transaction() as conn:
        assert [row["state"] for row in conn.execute("SELECT state FROM note_graph_suggestion_runs").fetchall()] == [
            "failed"
        ]
        assert conn.execute("SELECT COUNT(*) AS n FROM note_graph_suggestions").fetchone()["n"] == 0
    _register(db, dataset="canonical-one")
    before = _counts(db)
    with chacha_operation(independent=True):
        store = db.note_graph_suggestion_store
        with pytest.raises(NotesGraphDatasetScopeError):
            store.admit_run(
                dataset_id="legacy:1",
                source_note_id=note,
                source_fingerprint="sha256:" + "a" * 64,
                provider="openai",
                model="fixture-model",
                capability_revision=response.json()["revision"],
                prompt_contract_version="fixture",
                idempotency_key="strict-direct",
                now=stored.NOW,
            )
    assert _counts(db) == before


def test_populated_legacy_reads_preserve_published_evidence_and_rejections(fresh_suggestions, monkeypatch):
    db, client, note, _ = fresh_suggestions
    _register(db)
    monkeypatch.setattr(stored, "DATASET_ID", "legacy:1")
    note = stored.SOURCE_ID
    with chacha_operation(independent=True):
        db.add_note("Source fixture", "source body", note_id=note)
        target = db.add_note("Target fixture", "source related target body", note_id=stored.TARGET_ID)
        run = stored._stage_and_activate(db, key="publish", suggestion_id="visible", target_id=target)
        other = db.add_note("Other fixture", "other target body", note_id=stored.OTHER_ID)
        rejected = stored._stage_and_activate(db, key="reject", suggestion_id="dismiss", target_id=other)
        suggestions = db.note_graph_suggestion_store.list_suggestions(
            dataset_id="legacy:1",
            source_note_id=note,
            source_fingerprint=stored._fingerprint(db, note),
            states=("pending",),
            limit=20,
            after=None,
        ).items
        suggestion = next(item for item in suggestions if item.id == "dismiss")
        db.note_graph_suggestion_store.reject_suggestion(
            dataset_id="legacy:1",
            suggestion_id=suggestion.id,
            expected_revision=suggestion.revision,
            expected_source_fingerprint=suggestion.source_fingerprint,
            expected_target_fingerprint=suggestion.target_fingerprint,
            idempotency_key="dismiss",
            now=stored.NOW,
        )
        with db.transaction() as conn:
            conn.execute("DELETE FROM note_task_scope_authority WHERE owner_user_id=?", ("1",))
    response = client.get(f"/api/v1/notes/{note}/graph/suggestions?state=rejected&limit=20")
    assert response.status_code == 200
    value = response.json()
    assert value["rejection_count"] == 1 and value["rejection_set_revision"] >= 1
    assert value["items"][0]["state"] == "rejected"
    assert run.id != rejected.id
    assert value["items"][0]["evidence"] == []  # Rejection deliberately erases evidence.
    # Exercise Stage A storage/facade reads here; populated HTTP serialization
    # has its own UAT226 actual-router regression.
    from tldw_Server_API.app.core.Notes_Graph.suggestion_api import build_notes_graph_suggestions_api

    with chacha_operation(independent=True):
        api = build_notes_graph_suggestions_api(note_db=db, owner_user_id="1", dataset_id="legacy:1", jobs=None)
        pending = api.list_suggestions(note_id=note, states=("pending",), limit=20, cursor=None)
    assert pending.items[0].suggestion.id == "visible"
    assert {item.side for item in pending.items[0].evidence} == {"source", "target"}
    with chacha_operation(independent=True), db.transaction() as conn:
        conn.execute("UPDATE notes SET content=? WHERE id=?", ("changed target", target))
    with chacha_operation(independent=True):
        stale = api.list_suggestions(note_id=note, states=("pending",), limit=20, cursor=None)
    assert {item.side for item in stale.items[0].evidence} == {"source"}


def test_expected_fts_unavailability_is_distinct_from_unexpected_storage_error(fresh_suggestions, monkeypatch):
    db, client, note, _ = fresh_suggestions
    _register(db)
    _ready_provider(monkeypatch)
    monkeypatch.setenv("NOTES_GRAPH_SUGGESTIONS_WORKER_ENABLED", "1")
    client.app.dependency_overrides[endpoint.try_get_job_manager] = lambda: SimpleNamespace()
    monkeypatch.setattr(suggestion_service, "build_suggestion_decision_service", lambda **_: SimpleNamespace())
    store_type = type(db.note_graph_suggestion_store)

    def fts_unavailable(*_args, **_kwargs):
        raise NotesGraphFTSNotReadyError("notes_graph_fts_not_ready")

    monkeypatch.setattr(store_type, "_ensure_fts_ready", fts_unavailable)
    response = client.get(f"/api/v1/notes/{note}/graph/suggestions/capabilities")
    assert response.status_code == 200 and response.json()["unavailable_reason"] == "notes_graph_fts_not_ready"

    def storage_error(*_args, **_kwargs):
        raise RuntimeError("synthetic-private-storage-fault")

    monkeypatch.setattr(store_type, "load_source_note", storage_error)
    response = client.get(f"/api/v1/notes/{note}/graph/suggestions/capabilities")
    assert response.status_code == 503 and "synthetic-private" not in response.text


@pytest.mark.parametrize("operation", ["reject", "reset", "cancel"])
def test_retired_local_write_helpers_remain_strict(fresh_suggestions, operation):
    db, _, note, _ = fresh_suggestions
    _register(db, dataset="canonical-one")
    before = _counts(db)
    with chacha_operation(independent=True):
        store = db.note_graph_suggestion_store
        common = {"dataset_id": "legacy:1", "idempotency_key": "strict-write", "now": stored.NOW}
        with pytest.raises(NotesGraphDatasetScopeError):
            if operation == "reject":
                store.reject_suggestion(
                    **common,
                    suggestion_id="missing",
                    expected_revision=1,
                    expected_source_fingerprint="sha256:" + "a" * 64,
                    expected_target_fingerprint=None,
                )
            elif operation == "reset":
                store.reset_rejections(
                    **common, source_note_id=note, source_fingerprint="sha256:" + "a" * 64, expected_revision=1
                )
            else:
                store.admit_run_cancellation(
                    **common, run_id="missing", expected_state="queued", expected_revision=1, reason="user_cancelled"
                )
    assert _counts(db) == before


def test_unbound_read_preserves_nested_caller_write_and_rollback(fresh_suggestions):
    db, _, note, kind = fresh_suggestions
    with chacha_operation(independent=True):
        original = db.get_note_by_id(note)["title"]
        with pytest.raises(RuntimeError, match="caller rollback"):
            with db.transaction() as conn:
                conn.execute("UPDATE notes SET title=? WHERE id=?", ("pending caller title", note))
                with db.transaction():
                    assert (
                        db.note_graph_suggestion_store.load_source_note(dataset_id="legacy:1", note_id=note).title
                        == "pending caller title"
                    )
                if kind == "postgres":
                    assert (
                        conn.execute("SELECT current_setting('app.current_dataset_id', true) AS v").fetchone()["v"]
                        == "legacy:1"
                    )
                raise RuntimeError("caller rollback")
        assert db.get_note_by_id(note)["title"] == original
        if kind == "postgres":
            with db.transaction() as conn:
                assert conn.execute("SELECT current_setting('app.current_dataset_id', true) AS v").fetchone()["v"] in (
                    None,
                    "",
                )


def test_unbound_eligibility_is_rechecked_after_owner_binding(fresh_suggestions):
    db, _, note, _ = fresh_suggestions
    with chacha_operation(independent=True):
        store = db.note_graph_suggestion_store
        assert store.load_source_note(dataset_id="legacy:1", note_id=note).note_id == note
        with db.transaction() as conn:
            conn.execute(
                "INSERT INTO note_task_scope_authority(owner_user_id,dataset_id) VALUES (?,?)", ("1", "canonical-one")
            )
        with pytest.raises(NotesGraphDatasetScopeError):
            store.load_source_note(dataset_id="legacy:1", note_id=note)
        assert store.load_source_note(dataset_id="canonical-one", note_id=note).note_id == note


@pytest.mark.parametrize("reason", ["disabled", "provider"])
def test_missing_decisions_preserve_higher_priority_unavailability(fresh_suggestions, monkeypatch, reason):
    monkeypatch.setattr(suggestion_service, "build_suggestion_decision_service", lambda **_: None)
    from tldw_Server_API.app.core.Notes_Graph import graph_service

    db, client, note, _ = fresh_suggestions
    _register(db)
    if reason == "disabled":
        _ready_provider(monkeypatch)
        monkeypatch.setattr(graph_service, "NOTES_GRAPH_ENABLED", lambda: False)
    monkeypatch.setenv("NOTES_GRAPH_SUGGESTIONS_WORKER_ENABLED", "1")
    client.app.dependency_overrides[endpoint.try_get_job_manager] = lambda: SimpleNamespace()
    response = client.get(f"/api/v1/notes/{note}/graph/suggestions/capabilities")
    assert response.status_code == 200
    expected = "notes_graph_suggestions_disabled" if reason == "disabled" else "notes_graph_provider_disallowed"
    assert response.json()["unavailable_reason"] == expected
    assert response.json()["allowed_actions"] == ["cancel"]


def test_registration_storage_error_is_not_unbound_capability(fresh_suggestions, monkeypatch):
    monkeypatch.setattr(suggestion_service, "build_suggestion_decision_service", lambda **_: None)
    db, client, note, _ = fresh_suggestions
    _register(db)

    def fail(*_args, **_kwargs):
        raise RuntimeError("synthetic-private-registration-fault")

    monkeypatch.setattr(type(db.note_graph_suggestion_store), "is_dataset_scope_registered", fail, raising=False)
    response = client.get(f"/api/v1/notes/{note}/graph/suggestions/capabilities")
    assert response.status_code == 503
    assert "synthetic-private" not in response.text


@pytest.mark.parametrize("dataset", ["legacy:1", "canonical-one"])
def test_registered_scope_can_cancel_without_decision_coordinator(fresh_suggestions, monkeypatch, dataset):
    monkeypatch.setattr(suggestion_service, "build_suggestion_decision_service", lambda **_: None)
    from tldw_Server_API.app.core.Notes_Graph.suggestion_api import build_notes_graph_suggestions_api

    db, _, note, _ = fresh_suggestions
    _register(db, dataset=dataset)
    _ready_provider(monkeypatch)
    calls = []

    class Jobs:
        def get_job_or_archived_by_uuid(self, job_id, **scope):
            calls.append(("read", scope))
            return {
                "id": 7,
                "uuid": job_id,
                "owner_user_id": "1",
                "domain": "notes",
                "queue": "graph-suggestions",
                "job_type": "note_graph_suggestions",
                "status": "pending",
            }

        def cancel_job(self, job_id, **scope):
            calls.append(("cancel", scope))
            assert job_id == 7
            return True

    with chacha_operation(independent=True):
        store = db.note_graph_suggestion_store
        admitted = store.admit_run(
            dataset_id=dataset,
            source_note_id=note,
            source_fingerprint=stored._fingerprint(db, note),
            provider="openai",
            model="fixture-model",
            capability_revision="cap-v1",
            prompt_contract_version="prompt-v1",
            idempotency_key="seed-run",
            now=stored.NOW,
        )
        queued = store.bind_admitted_run(
            dataset_id=dataset,
            run_id=admitted.run.id,
            expected_state="admitting",
            expected_revision=admitted.run.revision,
            job_id="scoped-job",
            completion_token=f"completion-{admitted.run.id}",
            replay_envelope={"run_id": admitted.run.id, "state": "queued"},
            now=stored.NOW,
        )
        api = build_notes_graph_suggestions_api(note_db=db, owner_user_id="1", dataset_id=dataset, jobs=Jobs())
        capability = api.get_capabilities(note_id=note, provider=None, model=None)
        assert capability.allowed_actions == ("cancel",)
        assert not capability.generation_available
        result = api.cancel_run(
            note_id=note, run_id=queued.id, expected_revision=queued.revision, idempotency_key="cancel-scoped"
        )
        assert result.accepted is True
        assert result.cancellation.replay_envelope["state"] == "cancelling"
        replay = api.cancel_run(
            note_id=note, run_id=queued.id, expected_revision=queued.revision, idempotency_key="cancel-scoped"
        )
        assert replay.accepted is True
    assert len(calls) == 2
    assert calls[0] == ("read", {"domain": "notes", "owner_user_id": "1"})
    assert calls[1][1]["expected_uuid"] == "scoped-job"
