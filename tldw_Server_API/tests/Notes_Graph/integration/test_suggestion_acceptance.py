"""Guarded canonical acceptance integration tests."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

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
    SyncServerOriginBatchMaterializationError,
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


def _publish(
    db: CharactersRAGDB,
    *,
    suggestion_id: str,
    kind: str,
    keyword_sync_id: str | None = None,
    normalized_tag: str = "research",
    display_tag: str = "Research",
) -> None:
    store = db.note_graph_suggestion_store
    admitted = store.admit_run(
        dataset_id=DATASET_ID,
        source_note_id=SOURCE_ID,
        source_fingerprint=_fingerprint(db, SOURCE_ID),
        provider="openai",
        model="model-a",
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
            "evidence": (),
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
    store.activate_staged_run(
        dataset_id=DATASET_ID,
        run_id=publishing.id,
        expected_state="publishing",
        expected_revision=publishing.revision,
        observed_job_id=publishing.job_id,
        observed_completion_token=publishing.expected_completion_token,
        observed_result_digest=publishing.result_digest,
        now=NOW,
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
