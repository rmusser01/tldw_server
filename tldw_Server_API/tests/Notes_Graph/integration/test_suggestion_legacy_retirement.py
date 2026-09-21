"""Retired local review work is fenced, cancelled and cleaned without rekeying."""

from datetime import timedelta

import pytest

from tldw_Server_API.app.core.DB_Management.chacha.note_graph_suggestion_store import NotesGraphDatasetScopeError
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Notes_Graph.suggestion_jobs import JOB_QUEUE, SuggestionAdmissionService, _payload
from tldw_Server_API.app.core.Notes_Graph.suggestion_maintenance import MaintenanceScope, SuggestionMaintenance
from tldw_Server_API.app.core.Notes_Graph.suggestion_service import build_suggestion_decision_service
from tldw_Server_API.tests.Notes_Graph.integration import test_suggestion_acceptance as fixture
from tldw_Server_API.tests.Notes_Graph.integration.test_suggestion_legacy_mutations import (
    local_product_db as local_product_db,
)

pytestmark = pytest.mark.integration


def _rows(db, table):
    assert table in {"note_graph_suggestions", "note_graph_suggestion_runs", "note_graph_suggestion_operation_receipts"}
    with db.transaction() as conn:
        # The identifier is the fixed test-helper allowlist above.
        return [dict(row) for row in conn.execute(f"SELECT * FROM {table} ORDER BY id").fetchall()]  # nosec B608


def _arguments(db):
    return {
        "dataset_id": f"legacy:{db.client_id}",
        "source_note_id": fixture.SOURCE_ID,
        "source_fingerprint": fixture._fingerprint(db, fixture.SOURCE_ID),
        "provider": "openai",
        "model": "synthetic-no-provider",
        "capability_revision": "cap-v1",
        "prompt_contract_version": "prompt-v1",
        "idempotency_key": "retirement-admission",
        "now": fixture.NOW,
    }


def _maintenance(db, jobs):
    return SuggestionMaintenance(
        jobs=jobs, scopes=[MaintenanceScope(store=db.note_graph_suggestion_store, dataset_id=f"legacy:{db.client_id}")]
    )


@pytest.mark.parametrize("decision", ["pending", "accepting", "accepted", "rejected"])
def test_retirement_stales_only_unfinished_review_and_preserves_terminal_receipts(
    local_product_db, monkeypatch, decision
):
    db = local_product_db
    dataset = f"legacy:{db.client_id}"
    monkeypatch.setattr(fixture, "DATASET_ID", dataset)
    fixture._publish(db, suggestion_id="retire-review", kind="related_note")
    store = db.note_graph_suggestion_store
    args = {
        "dataset_id": dataset,
        "suggestion_id": "retire-review",
        "expected_revision": 1,
        "expected_source_fingerprint": fixture._fingerprint(db, fixture.SOURCE_ID),
        "expected_target_fingerprint": fixture._fingerprint(db, fixture.TARGET_ID),
        "idempotency_key": "decision",
    }
    if decision == "accepted":
        service = build_suggestion_decision_service(note_db=db, owner_user_id=db.client_id, dataset_id=dataset)
        service._clock = lambda: fixture.NOW
        service.accept(**args)
    elif decision == "accepting":
        store.claim_acceptance(**args, now=fixture.NOW)
    elif decision == "rejected":
        store.reject_suggestion(**args, now=fixture.NOW)
    before_receipts = {
        row["id"]: row for row in _rows(db, "note_graph_suggestion_operation_receipts") if row["state"] != "in_progress"
    }
    store.reserve_canonical_scope(dataset_id="real-canonical")
    with pytest.raises(NotesGraphDatasetScopeError):
        store.get_suggestion(dataset_id=dataset, suggestion_id="retire-review")
    result = _maintenance(db, object()).run_pass(now=fixture.NOW + timedelta(seconds=1))
    row = _rows(db, "note_graph_suggestions")[0]
    assert row["state"] == ("stale" if decision in {"pending", "accepting"} else decision)
    assert result.claimed <= 100
    after = {row["id"]: row for row in _rows(db, "note_graph_suggestion_operation_receipts")}
    assert all(after[key] == value for key, value in before_receipts.items())
    assert all(row["state"] != "in_progress" for row in after.values())
    assert all(row["dataset_id"] == dataset for row in after.values())
    if decision == "accepted":
        assert db.notes_link_store.get(row["accepted_resource_identity"]) is not None


@pytest.mark.parametrize(
    "timing", ["queued", "enqueue-before-bind", "enqueue-after-first-maintenance", "enqueue-after-grace"]
)
def test_retired_jobs_are_discovered_and_cancelled_without_binding_or_publication(
    local_product_db, tmp_path, monkeypatch, timing
):
    db = local_product_db
    store = db.note_graph_suggestion_store
    monkeypatch.setenv("JOBS_ALLOWED_QUEUES_NOTES", JOB_QUEUE)
    jobs = JobManager(tmp_path / "retired-jobs.db")
    service = SuggestionAdmissionService(store=store, jobs=jobs, owner_user_id=db.client_id)
    create = jobs.create_job
    first_pass = []

    def enrolling_create(**kwargs):
        store.reserve_canonical_scope(dataset_id="real-canonical")
        if timing in {"enqueue-after-first-maintenance", "enqueue-after-grace"}:
            elapsed = timedelta(minutes=11) if timing == "enqueue-after-grace" else timedelta(seconds=1)
            first_pass.append(_maintenance(db, jobs).run_pass(now=fixture.NOW + elapsed))
            assert _rows(db, "note_graph_suggestion_runs")[0]["state"] == (
                "failed" if timing == "enqueue-after-grace" else "admitting"
            )
        return create(**kwargs)

    if timing == "queued":
        service.admit(**_arguments(db))
        store.reserve_canonical_scope(dataset_id="real-canonical")
    else:
        monkeypatch.setattr(jobs, "create_job", enrolling_create)
        with pytest.raises(NotesGraphDatasetScopeError):
            service.admit(**_arguments(db))
    run_before = _rows(db, "note_graph_suggestion_runs")[0]
    job_before = jobs.get_job_or_archived_by_idempotency_key(
        idempotency_key=run_before["id"],
        domain="notes",
        queue=JOB_QUEUE,
        job_type="note_graph_suggestions",
        owner_user_id=db.client_id,
    )
    _maintenance(db, jobs).run_pass(
        now=fixture.NOW + (timedelta(minutes=12) if timing == "enqueue-after-grace" else timedelta(seconds=2))
    )
    run = _rows(db, "note_graph_suggestion_runs")[0]
    job = jobs.get_job_or_archived_by_uuid(job_before["uuid"], domain="notes", owner_user_id=db.client_id)
    assert job["status"] == "cancelled"
    assert job["payload"] == job_before["payload"]
    assert run["state"] == ("stale" if timing == "queued" else "failed")
    assert run["dataset_id"] == f"legacy:{db.client_id}"
    assert run["job_id"] == run_before["job_id"]
    assert _rows(db, "note_graph_suggestions") == []
    receipts = _rows(db, "note_graph_suggestion_operation_receipts")
    assert all(row["state"] != "in_progress" for row in receipts)
    if first_pass:
        assert first_pass[0].released == (0 if timing == "enqueue-after-grace" else 1)


def test_retired_missing_job_keeps_existing_grace_then_closes_without_product(local_product_db, tmp_path, monkeypatch):
    db = local_product_db
    monkeypatch.setenv("JOBS_ALLOWED_QUEUES_NOTES", JOB_QUEUE)
    jobs = JobManager(tmp_path / "missing-jobs.db")
    db.note_graph_suggestion_store.admit_run(**_arguments(db))
    db.note_graph_suggestion_store.reserve_canonical_scope(dataset_id="real-canonical")
    maintenance = _maintenance(db, jobs)
    maintenance.run_pass(now=fixture.NOW + timedelta(minutes=9))
    assert _rows(db, "note_graph_suggestion_runs")[0]["state"] == "admitting"
    maintenance.run_pass(now=fixture.NOW + timedelta(minutes=10))
    assert _rows(db, "note_graph_suggestion_runs")[0]["state"] == "failed"
    assert _rows(db, "note_graph_suggestions") == []


def test_enrollment_between_keyword_and_membership_never_reports_acceptance(local_product_db, monkeypatch):
    db = local_product_db
    dataset = f"legacy:{db.client_id}"
    monkeypatch.setattr(fixture, "DATASET_ID", dataset)
    fixture._publish(db, suggestion_id="keyword-before-enrollment", kind="tag")
    decisions = build_suggestion_decision_service(note_db=db, owner_user_id=db.client_id, dataset_id=dataset)
    decisions._clock = lambda: fixture.NOW
    create = decisions.local.create_keyword
    created = []

    def create_then_enroll(**kwargs):
        resource = create(**kwargs)
        created.append(resource.sync_id)
        db.note_graph_suggestion_store.reserve_canonical_scope(dataset_id="real-canonical")
        return resource

    monkeypatch.setattr(decisions.local, "create_keyword", create_then_enroll)
    with pytest.raises(NotesGraphDatasetScopeError):
        decisions.accept(
            dataset_id=dataset,
            suggestion_id="keyword-before-enrollment",
            expected_revision=1,
            expected_source_fingerprint=fixture._fingerprint(db, fixture.SOURCE_ID),
            expected_target_fingerprint=None,
            idempotency_key="enrollment-keyword",
        )
    assert len(created) == 1
    assert db.keyword_store.resolve_merge_survivor(created[0]) is not None
    assert db.get_keywords_for_note(fixture.SOURCE_ID) == []
    _maintenance(db, object()).run_pass(now=fixture.NOW + timedelta(seconds=1))
    assert _rows(db, "note_graph_suggestions")[0]["state"] == "stale"
    receipt = next(
        row
        for row in _rows(db, "note_graph_suggestion_operation_receipts")
        if row["operation_kind"] == "suggestion_accept"
    )
    assert receipt["state"] == "completed" and receipt["http_status"] == 409


def test_enrollment_after_maintenance_scope_check_defers_acceptance_reconciliation(local_product_db, monkeypatch):
    db = local_product_db
    dataset = f"legacy:{db.client_id}"
    monkeypatch.setattr(fixture, "DATASET_ID", dataset)
    fixture._publish(db, suggestion_id="retire-between-maintenance-checks", kind="related_note")
    maintenance = _maintenance(db, object())
    decisions = maintenance._scopes[0].decision_service
    reconcile = decisions.reconcile_expired

    def enroll_before_claim(**kwargs):
        db.note_graph_suggestion_store.reserve_canonical_scope(dataset_id="real-canonical")
        return reconcile(**kwargs)

    monkeypatch.setattr(decisions, "reconcile_expired", enroll_before_claim)
    maintenance.run_pass(now=fixture.NOW + timedelta(seconds=1))
    maintenance.run_pass(now=fixture.NOW + timedelta(seconds=2))
    assert _rows(db, "note_graph_suggestions")[0]["state"] == "stale"


def test_late_job_discovery_rotates_under_budget_without_extending_retention(local_product_db, tmp_path, monkeypatch):
    db = local_product_db
    store = db.note_graph_suggestion_store
    monkeypatch.setenv("JOBS_ALLOWED_QUEUES_NOTES", JOB_QUEUE)
    jobs = JobManager(tmp_path / "fair-jobs.db")
    runs = []
    for index in range(4):
        args = {**_arguments(db), "idempotency_key": f"older-{index}", "now": fixture.NOW + timedelta(seconds=index)}
        run = store.admit_run(**args).run
        runs.append(
            store.fail_admission(
                dataset_id=args["dataset_id"],
                run_id=run.id,
                expected_state="admitting",
                expected_revision=run.revision,
                error_code="notes_graph_capabilities_changed_before_queue",
                guidance_key="retry_generation",
                now=args["now"],
            )
        )
    store.reserve_canonical_scope(dataset_id="real-canonical")
    receipt_before = _rows(db, "note_graph_suggestion_operation_receipts")
    expiry_before = {row["id"]: row["expires_at"] for row in _rows(db, "note_graph_suggestion_runs")}
    late = jobs.create_job(
        domain="notes",
        queue=JOB_QUEUE,
        job_type="note_graph_suggestions",
        owner_user_id=db.client_id,
        payload=_payload(runs[-1], f"legacy:{db.client_id}"),
        max_retries=0,
        idempotency_key=runs[-1].id,
    )
    maintenance = _maintenance(db, jobs)
    for _ in range(4):
        result = maintenance.run_pass(now=fixture.NOW + timedelta(hours=1), limit=1)
        assert result.claimed == 1
    assert (
        jobs.get_job_or_archived_by_uuid(late["uuid"], domain="notes", owner_user_id=db.client_id)["status"]
        == "cancelled"
    )
    assert {row["id"]: row["expires_at"] for row in _rows(db, "note_graph_suggestion_runs")} == expiry_before
    assert _rows(db, "note_graph_suggestion_operation_receipts") == receipt_before
    assert maintenance.run_pass(now=fixture.NOW + timedelta(hours=1), limit=1).claimed == 0
    maintenance.run_pass(now=fixture.NOW + timedelta(days=31))
    assert _rows(db, "note_graph_suggestion_runs") == []
    assert _rows(db, "note_graph_suggestion_operation_receipts") == receipt_before


def test_retired_jobs_lookup_outage_preserves_retry_and_terminal_receipts(local_product_db, tmp_path, monkeypatch):
    db = local_product_db
    store = db.note_graph_suggestion_store
    monkeypatch.setenv("JOBS_ALLOWED_QUEUES_NOTES", JOB_QUEUE)
    jobs = JobManager(tmp_path / "outage-jobs.db")
    admitted = SuggestionAdmissionService(store=store, jobs=jobs, owner_user_id=db.client_id).admit(**_arguments(db))
    store.reserve_canonical_scope(dataset_id="real-canonical")
    before = _rows(db, "note_graph_suggestion_operation_receipts")
    lookup = jobs.get_job_or_archived_by_uuid

    def unavailable(*args, **kwargs):
        raise ConnectionError("synthetic temporary Jobs outage")

    monkeypatch.setattr(jobs, "get_job_or_archived_by_uuid", unavailable)
    assert _maintenance(db, jobs).run_pass(now=fixture.NOW + timedelta(seconds=1)).released == 1
    assert _rows(db, "note_graph_suggestion_runs")[0]["state"] == "queued"
    assert _rows(db, "note_graph_suggestion_operation_receipts") == before

    monkeypatch.setattr(jobs, "get_job_or_archived_by_uuid", lookup)
    _maintenance(db, jobs).run_pass(now=fixture.NOW + timedelta(seconds=2))
    assert lookup(admitted.job["uuid"], domain="notes", owner_user_id=db.client_id)["status"] == "cancelled"
    assert _rows(db, "note_graph_suggestion_operation_receipts") == before


@pytest.mark.parametrize("local_product_db", ["postgres"], indirect=True)
def test_restricted_role_local_acceptance_and_retirement_use_only_exact_owner(local_product_db, monkeypatch):
    from uuid import uuid4

    db = local_product_db
    store = db.note_graph_suggestion_store
    dataset = f"legacy:{db.client_id}"
    monkeypatch.setattr(fixture, "DATASET_ID", dataset)
    role = db.backend.escape_identifier(f"uat225_{uuid4().hex[:12]}")
    created = False
    try:
        with db.backend.transaction() as conn:
            db.backend.execute(f"CREATE ROLE {role} NOLOGIN NOSUPERUSER NOBYPASSRLS", connection=conn)
            db.backend.execute(f"GRANT USAGE ON SCHEMA public TO {role}", connection=conn)
            db.backend.execute(
                f"GRANT SELECT,INSERT,UPDATE,DELETE ON ALL TABLES IN SCHEMA public TO {role}", connection=conn
            )
            db.backend.execute(f"GRANT USAGE,SELECT ON ALL SEQUENCES IN SCHEMA public TO {role}", connection=conn)
            db.backend.execute(f"GRANT {role} TO CURRENT_USER", connection=conn)
        created = True
        with db.transaction() as conn:
            conn.execute(f"SET LOCAL ROLE {role}")
            flags = conn.execute("SELECT rolsuper,rolbypassrls FROM pg_roles WHERE rolname=current_user").fetchone()
            assert not flags["rolsuper"] and not flags["rolbypassrls"]
            fixture._publish(db, suggestion_id="restricted-product", kind="tag")
            decisions = build_suggestion_decision_service(note_db=db, owner_user_id=db.client_id, dataset_id=dataset)
            decisions._clock = lambda: fixture.NOW
            result = decisions.accept(
                dataset_id=dataset,
                suggestion_id="restricted-product",
                expected_revision=1,
                expected_source_fingerprint=fixture._fingerprint(db, fixture.SOURCE_ID),
                expected_target_fingerprint=None,
                idempotency_key="restricted-accept",
            )
            assert result.envelope["state"] == "accepted"
            fixture._publish(db, suggestion_id="restricted-retire", kind="related_note")
            store.reserve_canonical_scope(dataset_id="real-canonical")
            _maintenance(db, object()).run_pass(now=fixture.NOW + timedelta(seconds=1))
            rows = {row["id"]: row["state"] for row in _rows(db, "note_graph_suggestions")}
            assert rows == {"restricted-product": "accepted", "restricted-retire": "stale"}
            for foreign_scope in ("legacy:another-owner", "another-canonical"):
                with pytest.raises(NotesGraphDatasetScopeError):
                    store.retire_local_suggestions(dataset_id=foreign_scope, now=fixture.NOW, limit=1)
    finally:
        if created:
            with db.backend.transaction() as conn:
                db.backend.execute(f"DROP OWNED BY {role}", connection=conn)
                db.backend.execute(f"DROP ROLE {role}", connection=conn)
