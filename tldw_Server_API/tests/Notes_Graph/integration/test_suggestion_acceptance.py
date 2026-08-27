"""Guarded canonical acceptance integration tests."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.chacha.note_graph_suggestion_models import (
    NoteGraphSuggestionRun,
)
from tldw_Server_API.app.core.DB_Management.chacha.note_link_store import NotesLinkStore
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.Sync_DB import SyncDatabase
from tldw_Server_API.app.core.Notes_Graph.suggestion_content import content_fingerprint
from tldw_Server_API.app.core.Notes_Graph.suggestion_decisions import SuggestionDecisionService
from tldw_Server_API.app.core.Sync.v2.adapters import StaticSyncAdapter, SyncAdapterRegistry
from tldw_Server_API.app.core.Sync.v2.domain_adapters.notes_link import NotesLinkDomainAdapter
from tldw_Server_API.app.core.Sync.v2.materializers.notes_link import NotesLinkMaterializer
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
from tldw_Server_API.app.core.Sync.v2.notes_link_coordinator import NotesLinkCoordinator
from tldw_Server_API.app.core.Sync.v2.notes_organization_coordinator import (
    NotesOrganizationCoordinator,
)
from tldw_Server_API.app.core.Sync.v2.security import (
    server_trusted_encryption_status_from_config,
)
from tldw_Server_API.app.core.Sync.v2.server_origin_batch import (
    ServerOriginMutationStep,
    SyncServerOriginBatchMaterializationError,
    server_origin_mutation_batch_group_id,
)
from tldw_Server_API.app.core.Sync.v2.service import SyncV2Service, SyncV2Settings
from tldw_Server_API.app.core.Sync.v2.store import SyncV2Store

pytestmark = pytest.mark.integration

OWNER_ID = "owner-1"
DATASET_ID = "dataset-1"
SOURCE_ID = "11111111-1111-4111-8111-111111111111"
TARGET_ID = "22222222-2222-4222-8222-222222222222"
NOW = datetime(2026, 8, 27, 16, 0, tzinfo=timezone.utc)


def _fingerprint(db: CharactersRAGDB, note_id: str) -> str:
    note = db.get_note_by_id(note_id, include_deleted=True)
    assert note is not None
    return content_fingerprint(note["title"], note["content"])


def _sync_service(tmp_path: Path, note_db: CharactersRAGDB) -> tuple[SyncV2Service, SyncV2Store]:
    store = SyncV2Store(SyncDatabase(sqlite_path=tmp_path / "sync.db"))
    store.enroll_dataset(
        SyncDatasetCreate(
            dataset_id=DATASET_ID,
            owner_user_id=OWNER_ID,
            encryption_policy=DEFAULT_M1_ENCRYPTION_POLICY,
            domains=["notes.note", "notes.link", *NOTES_ORGANIZATION_DOMAINS],
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
                created_at_client=NOW.isoformat(),
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
        store.mark_envelope_apply_status(envelope.server_cursor, apply_status="applied")
    adapters = [NotesLinkDomainAdapter()]
    adapters.extend(StaticSyncAdapter(domain=domain) for domain in NOTES_ORGANIZATION_DOMAINS)
    materializers = {
        "notes.link": NotesLinkMaterializer(note_db),
        **{
            domain: NotesOrganizationMaterializer(note_db, domain)
            for domain in NOTES_ORGANIZATION_DOMAINS
        },
    }
    encryption = server_trusted_encryption_status_from_config(
        mode="managed_storage",
        server_trusted_enabled=True,
        auth_mode="multi_user",
    )
    return (
        SyncV2Service(
            store=store,
            adapters=SyncAdapterRegistry(adapters),
            materializers=materializers,
            clock=lambda: NOW.isoformat(),
            settings=SyncV2Settings(server_trusted_encryption=encryption),
        ),
        store,
    )


@pytest.fixture()
def context(tmp_path: Path):
    note_db = CharactersRAGDB(tmp_path / "notes.db", client_id=OWNER_ID)
    note_db.add_note(SOURCE_ID, "body", note_id=SOURCE_ID)
    note_db.add_note(TARGET_ID, "body", note_id=TARGET_ID)
    with note_db.transaction() as conn:
        conn.execute(
            "INSERT INTO note_task_scope_authority(owner_user_id,dataset_id) VALUES (?,?)",
            (OWNER_ID, DATASET_ID),
        )
    sync, sync_store = _sync_service(tmp_path, note_db)
    links = NotesLinkCoordinator(sync, note_db, OWNER_ID, sync_store.get_dataset(DATASET_ID))
    organization = NotesOrganizationCoordinator(sync, note_db, OWNER_ID)
    decisions = SuggestionDecisionService(
        store=note_db.note_graph_suggestion_store,
        link_coordinator=links,
        organization_coordinator=organization,
        clock=lambda: NOW,
    )
    try:
        yield note_db, links, decisions
    finally:
        note_db.close_all_connections()


def _stage(
    db: CharactersRAGDB,
    *,
    suggestion_id: str,
    kind: str,
    keyword_sync_id: str | None = None,
    normalized_tag: str = "research",
    display_tag: str = "Research",
    model: str = "model-a",
    include_evidence: bool = False,
) -> NoteGraphSuggestionRun:
    store = db.note_graph_suggestion_store
    admitted = store.admit_run(
        dataset_id=DATASET_ID,
        source_note_id=SOURCE_ID,
        source_fingerprint=_fingerprint(db, SOURCE_ID),
        provider="openai",
        model=model,
        capability_revision="cap-v1",
        prompt_contract_version="prompt-v1",
        idempotency_key=f"run-{suggestion_id}",
        now=NOW,
    )
    queued = store.bind_admitted_run(
        dataset_id=DATASET_ID,
        run_id=admitted.run.id,
        expected_state="admitting",
        expected_revision=admitted.run.revision,
        job_id=f"job-{suggestion_id}",
        completion_token=f"completion-{suggestion_id}",
        replay_envelope={"run_id": admitted.run.id, "state": "queued"},
        now=NOW,
    )
    running = store.start_run(
        dataset_id=DATASET_ID,
        run_id=queued.id,
        expected_state="queued",
        expected_revision=queued.revision,
        expected_job_id=queued.job_id,
        acquired_completion_token=f"worker-{suggestion_id}",
        now=NOW,
    )
    candidate = (
        {
            "id": suggestion_id,
            "kind": "related_note",
            "target_note_id": TARGET_ID,
            "target_fingerprint": _fingerprint(db, TARGET_ID),
            "match_strength": "strong",
            "rationale": "Bounded rationale",
            "evidence": (
                (
                    {
                        "side": "source",
                        "ordinal": 0,
                        "note_id": SOURCE_ID,
                        "field": "content",
                        "content_fingerprint": _fingerprint(db, SOURCE_ID),
                        "start_offset": 0,
                        "end_offset": 4,
                    },
                    {
                        "side": "target",
                        "ordinal": 0,
                        "note_id": TARGET_ID,
                        "field": "content",
                        "content_fingerprint": _fingerprint(db, TARGET_ID),
                        "start_offset": 0,
                        "end_offset": 4,
                    },
                )
                if include_evidence
                else ()
            ),
        }
        if kind == "related_note"
        else {
            "id": suggestion_id,
            "kind": "tag",
            "normalized_tag": normalized_tag,
            "display_tag": display_tag,
            "keyword_sync_id": keyword_sync_id,
            "match_strength": "possible",
            "rationale": "Bounded rationale",
            "evidence": (),
        }
    )
    publishing = store.stage_suggestions(
        dataset_id=DATASET_ID,
        run_id=running.id,
        expected_state="running",
        expected_revision=running.revision,
        expected_job_id=running.job_id,
        expected_completion_token=running.expected_completion_token,
        result_digest=f"sha256:{'a' * 64}",
        candidates=(candidate,),
        invalid_item_count=0,
        now=NOW,
    )
    return publishing


def _activate(db: CharactersRAGDB, publishing: NoteGraphSuggestionRun) -> None:
    db.note_graph_suggestion_store.activate_staged_run(
        dataset_id=DATASET_ID,
        run_id=publishing.id,
        expected_state="publishing",
        expected_revision=publishing.revision,
        observed_job_id=publishing.job_id,
        observed_completion_token=publishing.expected_completion_token,
        observed_result_digest=publishing.result_digest,
        now=NOW,
    )


def _publish(
    db: CharactersRAGDB,
    *,
    suggestion_id: str,
    kind: str,
    keyword_sync_id: str | None = None,
    normalized_tag: str = "research",
    display_tag: str = "Research",
    model: str = "model-a",
    include_evidence: bool = False,
) -> None:
    _activate(
        db,
        _stage(
            db,
            suggestion_id=suggestion_id,
            kind=kind,
            keyword_sync_id=keyword_sync_id,
            normalized_tag=normalized_tag,
            display_tag=display_tag,
            model=model,
            include_evidence=include_evidence,
        ),
    )


def test_related_acceptance_commits_link_and_decision_in_one_guarded_transaction(context) -> None:
    note_db, _links, decisions = context
    _publish(note_db, suggestion_id="related-accept", kind="related_note")

    result = decisions.accept(
        dataset_id=DATASET_ID,
        suggestion_id="related-accept",
        expected_revision=1,
        expected_source_fingerprint=_fingerprint(note_db, SOURCE_ID),
        expected_target_fingerprint=_fingerprint(note_db, TARGET_ID),
        idempotency_key="related-accept-request",
    )
    replay = decisions.accept(
        dataset_id=DATASET_ID,
        suggestion_id="related-accept",
        expected_revision=1,
        expected_source_fingerprint=_fingerprint(note_db, SOURCE_ID),
        expected_target_fingerprint=_fingerprint(note_db, TARGET_ID),
        idempotency_key="related-accept-request",
    )

    assert result.envelope == replay.envelope
    link = note_db.notes_link_store.get(result.envelope["accepted_resource_identity"])
    assert link is not None
    assert (link.directed, link.weight, link.label, link.properties) == (False, 1.0, None, {})


def test_new_tag_acceptance_requires_final_selected_note_membership(context) -> None:
    note_db, _links, decisions = context
    _publish(note_db, suggestion_id="tag-accept", kind="tag")

    result = decisions.accept(
        dataset_id=DATASET_ID,
        suggestion_id="tag-accept",
        expected_revision=1,
        expected_source_fingerprint=_fingerprint(note_db, SOURCE_ID),
        expected_target_fingerprint=None,
        idempotency_key="tag-accept-request",
    )

    assert result.envelope["state"] == "accepted"
    keyword = note_db.get_keyword_by_text("Research")
    assert keyword is not None
    assert [row["sync_id"] for row in note_db.get_keywords_for_note(SOURCE_ID)] == [
        keyword["sync_id"]
    ]


def test_new_tag_crash_after_keyword_creation_retries_only_final_membership(
    context,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    note_db, _links, decisions = context
    _publish(note_db, suggestion_id="tag-crash", kind="tag")
    original_capture = NotesOrganizationCoordinator.capture
    capture_count = 0

    def interrupt_membership(self, **kwargs):
        nonlocal capture_count
        capture_count += 1
        if capture_count == 2:
            raise ConnectionError("interrupted before membership capture")
        return original_capture(self, **kwargs)

    monkeypatch.setattr(NotesOrganizationCoordinator, "capture", interrupt_membership)
    request = {
        "dataset_id": DATASET_ID,
        "suggestion_id": "tag-crash",
        "expected_revision": 1,
        "expected_source_fingerprint": _fingerprint(note_db, SOURCE_ID),
        "expected_target_fingerprint": None,
        "idempotency_key": "tag-crash-request",
    }
    with pytest.raises(ConnectionError, match="membership"):
        decisions.accept(**request)
    assert note_db.get_keyword_by_text("Research") is not None
    assert note_db.get_keywords_for_note(SOURCE_ID) == []

    monkeypatch.setattr(NotesOrganizationCoordinator, "capture", original_capture)
    retried = decisions.accept(**request)

    assert retried.envelope["state"] == "accepted"
    assert len(note_db.get_keywords_for_note(SOURCE_ID)) == 1


def test_concurrent_new_tag_name_collision_converges_on_existing_portable_keyword(
    context,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    note_db, _links, decisions = context
    _publish(note_db, suggestion_id="tag-collision", kind="tag")
    organization = decisions.organization
    original_capture = NotesOrganizationCoordinator.capture
    injected = False

    def collide(self, **kwargs):
        nonlocal injected
        if not injected:
            injected = True
            external = organization.plan_keyword_create(
                "Research",
                idempotency_key="external-collision-keyword",
            )
            original_capture(
                self,
                steps=external.steps,
                source="notes_api",
                idempotency_key="external-collision-keyword",
            )
            raise SyncServerOriginBatchMaterializationError(
                SimpleNamespace(),
                retryable=False,
            )
        return original_capture(self, **kwargs)

    monkeypatch.setattr(NotesOrganizationCoordinator, "capture", collide)
    result = decisions.accept(
        dataset_id=DATASET_ID,
        suggestion_id="tag-collision",
        expected_revision=1,
        expected_source_fingerprint=_fingerprint(note_db, SOURCE_ID),
        expected_target_fingerprint=None,
        idempotency_key="tag-collision-request",
    )

    assert result.envelope["state"] == "accepted"
    keywords = note_db.keyword_store.list_keywords()
    assert [row["keyword"] for row in keywords] == ["Research"]
    assert note_db.get_keywords_for_note(SOURCE_ID)[0]["sync_id"] == keywords[0]["sync_id"]


@pytest.mark.parametrize(
    ("case_id", "canonical_display", "suggested_display", "normalized"),
    (
        ("ascii", "research", "Research", "research"),
        ("casefold", "STRASSE", "Stra\u00dfe", "strasse"),
        ("nfc", "CAF\u00c9", "cafe\u0301", "caf\u00e9"),
    ),
)
def test_normalized_new_tag_collision_supersedes_alias_and_uses_canonical_membership(
    context,
    monkeypatch: pytest.MonkeyPatch,
    case_id: str,
    canonical_display: str,
    suggested_display: str,
    normalized: str,
) -> None:
    note_db, _links, decisions = context
    organization = decisions.organization
    suggestion_id = f"normalized-{case_id}"
    _publish(
        note_db,
        suggestion_id=suggestion_id,
        kind="tag",
        normalized_tag=normalized,
        display_tag=suggested_display,
    )
    original_capture = NotesOrganizationCoordinator.capture
    injected = False

    def capture_with_external_winner(self, **kwargs):
        nonlocal injected
        if not injected and kwargs.get("guarded_mutations"):
            injected = True
            external = organization.plan_keyword_create(
                canonical_display,
                idempotency_key=f"external-{case_id}",
            )
            original_capture(
                organization,
                steps=external.steps,
                source="notes_api",
                idempotency_key=f"external-{case_id}",
            )
        return original_capture(self, **kwargs)

    monkeypatch.setattr(NotesOrganizationCoordinator, "capture", capture_with_external_winner)

    result = decisions.accept(
        dataset_id=DATASET_ID,
        suggestion_id=suggestion_id,
        expected_revision=1,
        expected_source_fingerprint=_fingerprint(note_db, SOURCE_ID),
        expected_target_fingerprint=None,
        idempotency_key=f"accept-{case_id}",
    )

    assert result.envelope["state"] == "accepted"
    canonical = note_db.get_keyword_by_text(canonical_display)
    assert canonical is not None
    assert [row["sync_id"] for row in note_db.get_keywords_for_note(SOURCE_ID)] == [
        canonical["sync_id"]
    ]
    assert len(note_db.keyword_store.list_keywords()) == 1
    group_id = server_origin_mutation_batch_group_id(
        dataset_id=DATASET_ID,
        source="notes_graph_suggestion",
        idempotency_key=f"notes-graph:{suggestion_id}:keyword",
    )
    group = organization.service.store.list_mutation_group(DATASET_ID, group_id)
    assert len(group) == 1
    assert group[0].apply_status == "superseded"
    assert (
        organization.service.store.get_object_state(
            DATASET_ID,
            "notes.keyword",
            group[0].object_id,
        )
        is None
    )


def test_external_exact_relationship_win_finalizes_actual_canonical_edge(
    context,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    note_db, _links, decisions = context
    _publish(note_db, suggestion_id="related-external-win", kind="related_note")
    external_dir = tmp_path / "external-sync"
    external_dir.mkdir()
    external_sync, external_store = _sync_service(external_dir, note_db)
    external_links = NotesLinkCoordinator(
        external_sync,
        note_db,
        OWNER_ID,
        external_store.get_dataset(DATASET_ID),
    )
    original_upsert = NotesLinkStore.upsert
    external_edge_id: list[str] = []

    def upsert_with_external_winner(self, **kwargs):
        if kwargs.get("before") is not None and not external_edge_id:
            external = external_links.create(
                source_note_id=SOURCE_ID,
                target_note_id=TARGET_ID,
                directed=False,
                weight=1.0,
                label=None,
                properties={},
                idempotency_key="external-exact-winner",
            )
            external_edge_id.append(external.edge_id)
        return original_upsert(self, **kwargs)

    monkeypatch.setattr(NotesLinkStore, "upsert", upsert_with_external_winner)

    result = decisions.accept(
        dataset_id=DATASET_ID,
        suggestion_id="related-external-win",
        expected_revision=1,
        expected_source_fingerprint=_fingerprint(note_db, SOURCE_ID),
        expected_target_fingerprint=_fingerprint(note_db, TARGET_ID),
        idempotency_key="related-external-win-request",
    )

    assert result.envelope["state"] == "accepted"
    assert result.envelope["accepted_resource_identity"] == external_edge_id[0]
    assert note_db.execute_query(
        "SELECT COUNT(*) AS count FROM note_edges WHERE deleted=0"
    ).fetchone()["count"] == 1


def test_finalizer_failure_rolls_back_relationship_and_accept_receipt(
    context,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    note_db, _links, decisions = context
    _publish(note_db, suggestion_id="related-finalizer-failure", kind="related_note")

    def fail_finalizer(_self, **_kwargs):
        raise RuntimeError("injected finalizer failure")

    monkeypatch.setattr(
        type(decisions.store),
        "finalize_acceptance_in_transaction",
        fail_finalizer,
    )

    with pytest.raises(SyncServerOriginBatchMaterializationError):
        decisions.accept(
            dataset_id=DATASET_ID,
            suggestion_id="related-finalizer-failure",
            expected_revision=1,
            expected_source_fingerprint=_fingerprint(note_db, SOURCE_ID),
            expected_target_fingerprint=_fingerprint(note_db, TARGET_ID),
            idempotency_key="related-finalizer-failure-request",
        )

    assert note_db.execute_query(
        "SELECT COUNT(*) AS count FROM note_edges WHERE deleted=0"
    ).fetchone()["count"] == 0
    suggestion = decisions.store.get_suggestion(
        dataset_id=DATASET_ID,
        suggestion_id="related-finalizer-failure",
    )
    assert suggestion.state.value == "accepting"
    receipt = note_db.execute_query(
        "SELECT state FROM note_graph_suggestion_operation_receipts WHERE id=?",
        (suggestion.decision_receipt_id,),
    ).fetchone()
    assert receipt["state"] == "in_progress"


def test_accept_vs_edit_barrier_rolls_back_link_after_fingerprint_invalidation(
    context,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    note_db, _links, decisions = context
    _publish(note_db, suggestion_id="related-edit-race", kind="related_note")
    entered = Event()
    release = Event()
    original_create = NotesLinkCoordinator.create

    def blocked_create(self, **kwargs):
        if kwargs.get("guarded_mutation") is not None:
            entered.set()
            assert release.wait(10)
        return original_create(self, **kwargs)

    monkeypatch.setattr(NotesLinkCoordinator, "create", blocked_create)
    request = {
        "dataset_id": DATASET_ID,
        "suggestion_id": "related-edit-race",
        "expected_revision": 1,
        "expected_source_fingerprint": _fingerprint(note_db, SOURCE_ID),
        "expected_target_fingerprint": _fingerprint(note_db, TARGET_ID),
        "idempotency_key": "related-edit-race-request",
    }
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(decisions.accept, **request)
        assert entered.wait(10)
        assert note_db.update_note(SOURCE_ID, {"content": "edited"}, expected_version=1)
        release.set()
        with pytest.raises(SyncServerOriginBatchMaterializationError):
            future.result(timeout=10)

    assert note_db.execute_query(
        "SELECT COUNT(*) AS count FROM note_edges WHERE deleted=0"
    ).fetchone()["count"] == 0
    assert decisions.store.get_suggestion(
        dataset_id=DATASET_ID,
        suggestion_id="related-edit-race",
    ).state.value == "stale"


def test_regeneration_during_acceptance_stales_only_pending_duplicates_in_finalizer(
    tmp_path: Path,
    pg_database_config,
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> None:
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    request.addfinalizer(backend.get_pool().close_all)
    note_db = CharactersRAGDB(":memory:", client_id=OWNER_ID, backend=backend)
    request.addfinalizer(note_db.close_all_connections)
    note_db.add_note(SOURCE_ID, "body", note_id=SOURCE_ID)
    note_db.add_note(TARGET_ID, "body", note_id=TARGET_ID)
    with note_db.transaction() as conn:
        conn.execute(
            "INSERT INTO note_task_scope_authority(owner_user_id,dataset_id) VALUES (?,?)",
            (OWNER_ID, DATASET_ID),
        )
    sync, sync_store = _sync_service(tmp_path, note_db)
    links = NotesLinkCoordinator(sync, note_db, OWNER_ID, sync_store.get_dataset(DATASET_ID))
    decisions = SuggestionDecisionService(
        store=note_db.note_graph_suggestion_store,
        link_coordinator=links,
        organization_coordinator=NotesOrganizationCoordinator(sync, note_db, OWNER_ID),
        clock=lambda: NOW,
    )
    accepting_id = "related-accepting-generation"
    terminal_id = "related-terminal-generation"
    protected_id = "related-protected-accepting"
    pending_id = "related-pending-generation"
    all_ids = (accepting_id, terminal_id, protected_id, pending_id)
    _publish(
        note_db,
        suggestion_id=accepting_id,
        kind="related_note",
        include_evidence=True,
    )
    coordinator_entered = Event()
    release_coordinator = Event()
    product_transaction_entered = Event()
    release_product_transaction = Event()
    original_create = NotesLinkCoordinator.create
    original_guard = type(decisions.store).guard_acceptance_in_transaction
    original_finalize = type(decisions.store).finalize_acceptance_in_transaction
    finalizer_snapshots: list[dict[str, object]] = []

    def durable_snapshot(conn):
        rows = conn.execute(
            "SELECT * FROM note_graph_suggestions WHERE id IN (?,?,?,?) ORDER BY id",
            all_ids,
        ).fetchall()
        evidence = conn.execute(
            "SELECT * FROM note_graph_suggestion_evidence "
            "WHERE suggestion_id IN (?,?,?,?) ORDER BY suggestion_id,side,ordinal",
            all_ids,
        ).fetchall()
        evidence_by_id = {suggestion_id: [] for suggestion_id in all_ids}
        for row in evidence:
            evidence_by_id[str(row["suggestion_id"])].append(dict(row))
        return (
            {str(row["id"]): dict(row) for row in rows},
            {suggestion_id: tuple(values) for suggestion_id, values in evidence_by_id.items()},
        )

    def blocked_create(self, **kwargs):
        if kwargs.get("guarded_mutation") is not None:
            coordinator_entered.set()
            assert release_coordinator.wait(10)
        return original_create(self, **kwargs)

    def blocked_guard(self, *, conn, **kwargs):
        original_guard(self, conn=conn, **kwargs)
        product_transaction_entered.set()
        assert release_product_transaction.wait(10)

    def observed_finalize(self, *, conn, **kwargs):
        result = original_finalize(self, conn=conn, **kwargs)
        rows, evidence = durable_snapshot(conn)
        link = conn.execute(
            "SELECT * FROM note_edges WHERE user_id=? AND edge_id=?",
            (OWNER_ID, result.envelope["accepted_resource_identity"]),
        ).fetchone()
        receipt = conn.execute(
            "SELECT * FROM note_graph_suggestion_operation_receipts "
            "WHERE owner_user_id=? AND dataset_id=? AND id=?",
            (OWNER_ID, DATASET_ID, kwargs["fence"].decision_receipt_id),
        ).fetchone()
        finalizer_snapshots.append(
            {
                "suggestions": rows,
                "evidence": evidence,
                "link": dict(link) if link is not None else None,
                "receipt": dict(receipt) if receipt is not None else None,
            }
        )
        return result

    monkeypatch.setattr(NotesLinkCoordinator, "create", blocked_create)
    monkeypatch.setattr(type(decisions.store), "guard_acceptance_in_transaction", blocked_guard)
    monkeypatch.setattr(
        type(decisions.store),
        "finalize_acceptance_in_transaction",
        observed_finalize,
    )
    accept_request = {
        "dataset_id": DATASET_ID,
        "suggestion_id": accepting_id,
        "expected_revision": 1,
        "expected_source_fingerprint": _fingerprint(note_db, SOURCE_ID),
        "expected_target_fingerprint": _fingerprint(note_db, TARGET_ID),
        "idempotency_key": "related-accepting-generation-request",
    }
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(decisions.accept, **accept_request)
        try:
            assert coordinator_entered.wait(10)
            _publish(
                note_db,
                suggestion_id=terminal_id,
                kind="related_note",
                model="model-b",
                include_evidence=True,
            )
            _publish(
                note_db,
                suggestion_id=protected_id,
                kind="related_note",
                model="model-c",
                include_evidence=True,
            )
            terminal_before = decisions.store.get_suggestion(
                dataset_id=DATASET_ID,
                suggestion_id=terminal_id,
            )
            assert (terminal_before.state.value, terminal_before.decision_reason) == (
                "stale",
                "superseded_by_run",
            )
            protected_claim = decisions.store.claim_acceptance(
                dataset_id=DATASET_ID,
                suggestion_id=protected_id,
                expected_revision=1,
                expected_source_fingerprint=_fingerprint(note_db, SOURCE_ID),
                expected_target_fingerprint=_fingerprint(note_db, TARGET_ID),
                idempotency_key="related-protected-accepting-request",
                now=NOW,
            )
            assert protected_claim.suggestion is not None
            pending_publishing = _stage(
                note_db,
                suggestion_id=pending_id,
                kind="related_note",
                model="model-d",
                include_evidence=True,
            )
            with note_db.transaction() as conn:
                decisions.store._set_dataset_scope(conn, DATASET_ID)
                rows_before, evidence_before = durable_snapshot(conn)
            assert evidence_before[terminal_id]
            assert evidence_before[protected_id]

            release_coordinator.set()
            assert product_transaction_entered.wait(10)
            _activate(note_db, pending_publishing)
            assert decisions.store.get_suggestion(
                dataset_id=DATASET_ID,
                suggestion_id=pending_id,
            ).state.value == "pending"
        finally:
            release_coordinator.set()
            release_product_transaction.set()
        try:
            result = future.result(timeout=10)
        except SyncServerOriginBatchMaterializationError as exc:
            statuses = [
                (row.apply_status, row.apply_error_code, row.apply_error_message)
                for row in exc.result.envelopes
            ]
            pytest.fail(f"guarded relationship acceptance failed: {statuses}")

    assert result.envelope["state"] == "accepted"
    assert len(finalizer_snapshots) == 1
    in_transaction = finalizer_snapshots[0]
    in_transaction_rows = in_transaction["suggestions"]
    assert in_transaction_rows[accepting_id]["state"] == "accepted"
    assert in_transaction_rows[accepting_id]["decision_reason"] == "user_accepted"
    assert in_transaction_rows[accepting_id]["rationale"] is None
    assert in_transaction_rows[pending_id]["state"] == "stale"
    assert in_transaction_rows[pending_id]["decision_reason"] == "canonical_accepted"
    assert in_transaction_rows[pending_id]["rationale"] is None
    assert in_transaction_rows[terminal_id] == rows_before[terminal_id]
    assert in_transaction_rows[protected_id] == rows_before[protected_id]

    in_transaction_evidence = in_transaction["evidence"]
    assert in_transaction_evidence[accepting_id] == ()
    assert in_transaction_evidence[pending_id] == ()
    assert in_transaction_evidence[terminal_id] == evidence_before[terminal_id]
    assert in_transaction_evidence[protected_id] == evidence_before[protected_id]

    link = in_transaction["link"]
    assert link is not None
    assert link["edge_id"] == result.envelope["accepted_resource_identity"]
    assert (link["from_note_id"], link["to_note_id"]) == (SOURCE_ID, TARGET_ID)
    assert link["type"] == "manual"
    assert not bool(link["directed"])
    assert float(link["weight"]) == 1.0
    assert link["label"] is None
    assert decisions.store._properties_are_empty(link["properties"])
    assert not bool(link["deleted"])

    receipt = in_transaction["receipt"]
    assert receipt is not None
    assert receipt["operation_kind"] == "suggestion_accept"
    assert receipt["resource_identity"] == accepting_id
    assert receipt["state"] == "completed"
    assert receipt["http_status"] == 200
    assert receipt["completed_at"] is not None
    assert json.loads(str(receipt["replay_envelope"])) == result.envelope

    pending = decisions.store.get_suggestion(
        dataset_id=DATASET_ID,
        suggestion_id=pending_id,
    )
    assert (pending.state.value, pending.decision_reason, pending.rationale) == (
        "stale",
        "canonical_accepted",
        None,
    )
    with note_db.transaction() as conn:
        decisions.store._set_dataset_scope(conn, DATASET_ID)
        rows_after, evidence_after = durable_snapshot(conn)
    assert rows_after[terminal_id] == rows_before[terminal_id]
    assert rows_after[protected_id] == rows_before[protected_id]
    assert evidence_after[terminal_id] == evidence_before[terminal_id]
    assert evidence_after[protected_id] == evidence_before[protected_id]
    assert note_db.notes_link_store.get(result.envelope["accepted_resource_identity"]) is not None


def test_activation_after_finalizer_reconciliation_waits_for_product_commit(
    tmp_path: Path,
    pg_database_config,
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> None:
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    request.addfinalizer(backend.get_pool().close_all)
    note_db = CharactersRAGDB(":memory:", client_id=OWNER_ID, backend=backend)
    request.addfinalizer(note_db.close_all_connections)
    note_db.add_note(SOURCE_ID, "body", note_id=SOURCE_ID)
    note_db.add_note(TARGET_ID, "body", note_id=TARGET_ID)
    with note_db.transaction() as conn:
        conn.execute(
            "INSERT INTO note_task_scope_authority(owner_user_id,dataset_id) VALUES (?,?)",
            (OWNER_ID, DATASET_ID),
        )
    sync, sync_store = _sync_service(tmp_path, note_db)
    decisions = SuggestionDecisionService(
        store=note_db.note_graph_suggestion_store,
        link_coordinator=NotesLinkCoordinator(
            sync,
            note_db,
            OWNER_ID,
            sync_store.get_dataset(DATASET_ID),
        ),
        organization_coordinator=NotesOrganizationCoordinator(sync, note_db, OWNER_ID),
        clock=lambda: NOW,
    )
    accepting_id = "related-post-finalizer-accepting"
    duplicate_id = "related-post-finalizer-duplicate"
    _publish(note_db, suggestion_id=accepting_id, kind="related_note")
    duplicate_run = _stage(
        note_db,
        suggestion_id=duplicate_id,
        kind="related_note",
        model="model-post-finalizer",
    )
    finalizer_reconciled = Event()
    release_product_commit = Event()
    activation_started = Event()
    canonical_read_entered = Event()
    original_finalize = type(decisions.store).finalize_acceptance_in_transaction
    original_has_current_link = type(decisions.store)._has_current_link

    def held_after_finalizer(self, *, conn, **kwargs):
        result = original_finalize(self, conn=conn, **kwargs)
        finalizer_reconciled.set()
        assert release_product_commit.wait(10)
        return result

    def observe_canonical_read(self, conn, source_note_id, target_note_id):
        canonical_read_entered.set()
        return original_has_current_link(self, conn, source_note_id, target_note_id)

    def activate_duplicate():
        activation_started.set()
        _activate(note_db, duplicate_run)

    monkeypatch.setattr(
        type(decisions.store),
        "finalize_acceptance_in_transaction",
        held_after_finalizer,
    )
    monkeypatch.setattr(type(decisions.store), "_has_current_link", observe_canonical_read)
    accept_request = {
        "dataset_id": DATASET_ID,
        "suggestion_id": accepting_id,
        "expected_revision": 1,
        "expected_source_fingerprint": _fingerprint(note_db, SOURCE_ID),
        "expected_target_fingerprint": _fingerprint(note_db, TARGET_ID),
        "idempotency_key": "related-post-finalizer-request",
    }
    with ThreadPoolExecutor(max_workers=2) as executor:
        accept_future = executor.submit(decisions.accept, **accept_request)
        assert finalizer_reconciled.wait(10)
        activation_future = executor.submit(activate_duplicate)
        assert activation_started.wait(10)
        try:
            assert not canonical_read_entered.wait(0.5)
            assert not activation_future.done()
        finally:
            release_product_commit.set()
        accepted = accept_future.result(timeout=10)
        activation_future.result(timeout=10)

    assert accepted.envelope["state"] == "accepted"
    assert canonical_read_entered.is_set()
    assert note_db.execute_query(
        "SELECT COUNT(*) AS count FROM note_graph_suggestions "
        "WHERE id=? AND state='pending'",
        (duplicate_id,),
    ).fetchone()["count"] == 0
    assert note_db.execute_query(
        "SELECT COUNT(*) AS count FROM note_edges WHERE user_id=? AND deleted=?",
        (OWNER_ID, False),
    ).fetchone()["count"] == 1


def test_old_fence_late_worker_cannot_write_after_expired_takeover(
    context,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    note_db, _links, decisions = context
    _publish(note_db, suggestion_id="related-old-fence", kind="related_note")
    entered = Event()
    release = Event()
    original_create = NotesLinkCoordinator.create

    def blocked_create(self, **kwargs):
        if kwargs.get("guarded_mutation") is not None:
            entered.set()
            assert release.wait(10)
        return original_create(self, **kwargs)

    monkeypatch.setattr(NotesLinkCoordinator, "create", blocked_create)
    request = {
        "dataset_id": DATASET_ID,
        "suggestion_id": "related-old-fence",
        "expected_revision": 1,
        "expected_source_fingerprint": _fingerprint(note_db, SOURCE_ID),
        "expected_target_fingerprint": _fingerprint(note_db, TARGET_ID),
        "idempotency_key": "related-old-fence-request",
    }
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(decisions.accept, **request)
        assert entered.wait(10)
        claims = decisions.store.claim_expired_acceptances(
            dataset_id=DATASET_ID,
            limit=1,
            now=NOW + timedelta(minutes=6),
        )
        assert len(claims) == 1
        decisions.store.resolve_expired_acceptance(
            fence=claims[0],
            accepted_resource_identity=None,
            resolved_keyword_sync_id=None,
            now=NOW + timedelta(minutes=6),
        )
        release.set()
        with pytest.raises(SyncServerOriginBatchMaterializationError):
            future.result(timeout=10)

    assert note_db.execute_query(
        "SELECT COUNT(*) AS count FROM note_edges WHERE deleted=0"
    ).fetchone()["count"] == 0
    assert decisions.store.get_suggestion(
        dataset_id=DATASET_ID,
        suggestion_id="related-old-fence",
    ).state.value == "pending"


def test_deleted_keyword_restore_wins_before_fenced_stale_closure(
    context,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    note_db, _links, decisions = context
    organization = decisions.organization
    keyword_plan = organization.plan_keyword_create("Restorable", idempotency_key="restorable")
    _capture_plan(organization, keyword_plan, key="restorable")
    keyword = keyword_plan.load_result()
    _publish(
        note_db,
        suggestion_id="tag-restored",
        kind="tag",
        keyword_sync_id=keyword["sync_id"],
        normalized_tag="restorable",
        display_tag="Restorable",
    )
    deletion = organization.plan_resource_delete(
        "notes.keyword",
        keyword["id"],
        expected_version=keyword["version"],
    )
    _capture_plan(organization, deletion, key="delete-restorable")
    missing_seen = Event()
    restored = Event()
    original_resolve = decisions._resolve_keyword
    first = True

    def blocked_missing_resolution(*args, **kwargs):
        nonlocal first
        resolution = original_resolve(*args, **kwargs)
        if first:
            first = False
            assert resolution[1] is True
            missing_seen.set()
            assert restored.wait(10)
        return resolution

    monkeypatch.setattr(decisions, "_resolve_keyword", blocked_missing_resolution)
    request = {
        "dataset_id": DATASET_ID,
        "suggestion_id": "tag-restored",
        "expected_revision": 1,
        "expected_source_fingerprint": _fingerprint(note_db, SOURCE_ID),
        "expected_target_fingerprint": None,
        "idempotency_key": "tag-restored-request",
    }
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(decisions.accept, **request)
        assert missing_seen.wait(10)
        restore = ServerOriginMutationStep(
            domain="notes.keyword",
            operation="upsert",
            object_id=keyword["sync_id"],
            payload={"keyword": "Restorable"},
            routing_metadata={"restore_intent": True},
        )
        organization.capture(
            steps=(restore,),
            source="notes_api",
            idempotency_key="restore-restorable",
        )
        restored.set()
        result = future.result(timeout=10)

    assert result.envelope["state"] == "accepted"
    assert [row["sync_id"] for row in note_db.get_keywords_for_note(SOURCE_ID)] == [
        keyword["sync_id"]
    ]


def _capture_plan(organization: NotesOrganizationCoordinator, plan, *, key: str) -> None:
    organization.capture(
        steps=plan.steps,
        source="notes_api",
        idempotency_key=key,
    )


def test_existing_tag_rename_and_merge_follow_current_portable_identity(context) -> None:
    note_db, _links, decisions = context
    organization = decisions.organization
    source_plan = organization.plan_keyword_create("Research", idempotency_key="source-keyword")
    _capture_plan(organization, source_plan, key="source-keyword")
    source = source_plan.load_result()
    target_plan = organization.plan_keyword_create("Knowledge", idempotency_key="target-keyword")
    _capture_plan(organization, target_plan, key="target-keyword")
    target = target_plan.load_result()
    _publish(
        note_db,
        suggestion_id="tag-merged",
        kind="tag",
        keyword_sync_id=source["sync_id"],
    )
    rename = organization.plan_keyword_rename(
        source["id"],
        "Deep Research",
        expected_version=source["version"],
    )
    _capture_plan(organization, rename, key="rename-source")
    renamed = rename.load_result()
    merge = organization.plan_keyword_merge(
        source_keyword_id=renamed["id"],
        target_keyword_id=target["id"],
        expected_source_version=renamed["version"],
        expected_target_version=target["version"],
    )
    _capture_plan(organization, merge, key="merge-source")

    result = decisions.accept(
        dataset_id=DATASET_ID,
        suggestion_id="tag-merged",
        expected_revision=1,
        expected_source_fingerprint=_fingerprint(note_db, SOURCE_ID),
        expected_target_fingerprint=None,
        idempotency_key="tag-merged-request",
    )

    assert result.envelope["state"] == "accepted"
    memberships = note_db.get_keywords_for_note(SOURCE_ID)
    assert [row["sync_id"] for row in memberships] == [target["sync_id"]]


def test_existing_tag_deletion_durably_marks_acceptance_stale(context) -> None:
    note_db, _links, decisions = context
    organization = decisions.organization
    keyword_plan = organization.plan_keyword_create("Disposable", idempotency_key="delete-keyword")
    _capture_plan(organization, keyword_plan, key="delete-keyword")
    keyword = keyword_plan.load_result()
    _publish(
        note_db,
        suggestion_id="tag-deleted",
        kind="tag",
        keyword_sync_id=keyword["sync_id"],
        normalized_tag="disposable",
        display_tag="Disposable",
    )
    deletion = organization.plan_resource_delete(
        "notes.keyword",
        keyword["id"],
        expected_version=keyword["version"],
    )
    _capture_plan(organization, deletion, key="delete-keyword-resource")

    result = decisions.accept(
        dataset_id=DATASET_ID,
        suggestion_id="tag-deleted",
        expected_revision=1,
        expected_source_fingerprint=_fingerprint(note_db, SOURCE_ID),
        expected_target_fingerprint=None,
        idempotency_key="tag-deleted-request",
    )

    assert result.envelope["state"] == "stale"
    assert result.envelope["error_code"] == "notes_graph_canonical_resource_stale"
