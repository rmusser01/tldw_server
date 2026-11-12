"""Atomic output, immutable replay and domain fencing on real Media databases."""

import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from uuid import uuid4

import pytest

from tldw_Server_API.app.core.claims_cluster_rebuild_contract import (
    ClusterRebuildLimits,
    ClusterRebuildSettings,
    canonical_cluster_settings,
)
from tldw_Server_API.app.core.DB_Management.media_db.claims_cluster_rebuild_types import (
    ClusterAssignment,
    ClusterCalculation,
    ClusterMember,
    ClusterPublicationIdentity,
    ClusterRebuildStorageError,
)
from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase
from tldw_Server_API.app.core.DB_Management.scope_context import scoped_context
from tldw_Server_API.tests.DB_Management.test_claim_cluster_rebuild_snapshot import seed, snapshot

pytestmark = pytest.mark.unit
LIMITS = ClusterRebuildLimits(50_000, 5_000, 33_554_432, 4_096, 2_000_000, 50_000_000, 50_000)
PUBLICATION = "tldw_Server_API.app.core.DB_Management.media_db.runtime.claims_cluster_rebuild_publication"


@pytest.fixture
def db(tmp_path):
    handle = MediaDatabase(db_path=str(tmp_path / "publication.db"), client_id="42")
    yield handle
    handle.close_connection()


def request(db, *, generation=1, authority=None, method="exact", empty=False):
    """Build a hand-checked singleton assignment, independent of the calculator."""
    if not empty and not snapshot(db).rows:
        seed(db)
    source = snapshot(db)
    settings = ClusterRebuildSettings(1, method, 1, 0.85 if method == "embeddings" else None, 1)
    fingerprint = hashlib.sha256(
        json.dumps(canonical_cluster_settings(settings), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assignments = tuple(
        ClusterAssignment(row.claim_text.strip(), row.claim_id, (ClusterMember(row.claim_id, 1.0),))
        for row in source.rows
    )
    return {
        "identity": ClusterPublicationIdentity("42", str(uuid4()), authority or str(uuid4()), generation, fingerprint),
        "settings": settings,
        "snapshot": source,
        "calculation": ClusterCalculation(assignments, "ok", None, 0),
        "limits": LIMITS,
    }


def state(db):
    """Compare all mutated domain tables, not just the public result."""
    return {
        table: [tuple(row) for row in db.execute_query(f"SELECT * FROM {table} ORDER BY 1").fetchall()]
        for table in (
            "claim_clusters",
            "claim_cluster_membership",
            "Claims",
            "claims_cluster_publication_heads",
            "claims_cluster_publication_receipts",
        )
    }


def test_replay_preserves_ids_counts_and_immutable_result(db):
    req = request(db)
    assert db.load_claim_cluster_publication(identity=req["identity"]) is None
    first = db.publish_claim_cluster_rebuild(**req)
    before = state(db)
    replay = db.publish_claim_cluster_rebuild(**req)
    loaded = db.load_claim_cluster_publication(identity=req["identity"])
    assert dict(first.result) == {
        "clusters_created": 1,
        "claims_assigned": 1,
        "claims_skipped": 0,
        "method": "exact",
        "similarity_threshold": None,
        "outcome": "ok",
        "reason": None,
    }
    assert first.newly_published and not replay.newly_published and not loaded.newly_published
    assert first.result == replay.result == loaded.result
    assert state(db) == before
    cluster = db.execute_query("SELECT id, representative_claim_id FROM claim_clusters").fetchone()
    claim = req["snapshot"].rows[0].claim_id
    assert cluster["representative_claim_id"] == claim
    assert tuple(db.execute_query("SELECT cluster_id, claim_id FROM claim_cluster_membership").fetchone()) == (
        cluster["id"],
        claim,
    )
    assert db.execute_query("SELECT claim_cluster_id FROM Claims WHERE id = ?", (claim,)).fetchone()[0] == cluster["id"]
    with pytest.raises(TypeError):
        first.result["clusters_created"] = 99


def test_old_receipt_replays_after_newer_head_without_restoring_output(db):
    old = request(db)
    first = db.publish_claim_cluster_rebuild(**old)
    new = request(db, generation=2, authority=old["identity"].authority_uuid)
    db.publish_claim_cluster_rebuild(**new)
    before = state(db)
    assert db.publish_claim_cluster_rebuild(**old).result == first.result
    assert db.load_claim_cluster_publication(identity=old["identity"]).result == first.result
    assert state(db) == before


def test_late_generation_is_superseded_without_receipt_or_mutation(db):
    new = request(db, generation=3)
    db.publish_claim_cluster_rebuild(**new)
    old = request(db, generation=2, authority=new["identity"].authority_uuid)
    before = state(db)
    result = db.publish_claim_cluster_rebuild(**old)
    assert (result.result["outcome"], result.result["reason"], result.newly_published) == (
        "skipped",
        "superseded",
        False,
    )
    assert state(db) == before
    assert db.load_claim_cluster_publication(identity=old["identity"]) is None


@pytest.mark.parametrize(
    "collision", ["authority", "equal_generation", "historical_generation", "receipt_generation", "receipt_settings"]
)
def test_identity_collisions_fail_without_mutation(db, collision):
    req = request(db)
    db.publish_claim_cluster_rebuild(**req)
    identity = req["identity"]
    if collision == "authority":
        identity = replace(identity, job_uuid=str(uuid4()), authority_uuid=str(uuid4()), generation=2)
    elif collision == "equal_generation":
        identity = replace(identity, job_uuid=str(uuid4()))
    elif collision == "historical_generation":
        db.publish_claim_cluster_rebuild(**request(db, generation=3, authority=identity.authority_uuid))
        identity = replace(identity, job_uuid=str(uuid4()))
    elif collision == "receipt_generation":
        identity = replace(identity, generation=2)
    else:
        identity = replace(identity, settings_fingerprint="f" * 64)
    before = state(db)
    with pytest.raises(ClusterRebuildStorageError, match="claims_cluster_corruption"):
        db.publish_claim_cluster_rebuild(**{**req, "identity": identity})
    assert state(db) == before


@pytest.mark.parametrize("change", ["edit", "delete", "transfer", "insert"])
def test_changed_source_rolls_back_head_output_and_receipt(db, change):
    req = request(db)
    row = req["snapshot"].rows[0]
    if change == "edit":
        db.execute_query("UPDATE Claims SET claim_text = 'Changed' WHERE id = ?", (row.claim_id,), commit=True)
    elif change == "delete":
        db.execute_query("DELETE FROM Claims WHERE id = ?", (row.claim_id,), commit=True)
    elif change == "transfer":
        db.execute_query(
            "UPDATE Media SET owner_user_id = 43, visibility = 'team', team_id = 9, version = version + 1 WHERE id = ?",
            (row.media_id,),
            commit=True,
        )
    else:
        seed(db, text="New arrival")
    before = state(db)
    with pytest.raises(ClusterRebuildStorageError) as caught:
        db.publish_claim_cluster_rebuild(**req)
    assert (caught.value.code, caught.value.retryable) == ("claims_cluster_source_changed", True)
    assert state(db) == before


def test_skip_validates_source_preserves_output_and_fences_older_jobs(db):
    old = request(db)
    db.publish_claim_cluster_rebuild(**old)
    skip = request(db, generation=3, authority=old["identity"].authority_uuid, method="embeddings")
    skip["calculation"] = ClusterCalculation((), "skipped", "no_embeddings", 1)
    before = state(db)
    result = db.publish_claim_cluster_rebuild(**skip)
    after = state(db)
    assert result.newly_published and result.result["reason"] == "no_embeddings"
    for table in ("claim_clusters", "claim_cluster_membership", "Claims"):
        assert after[table] == before[table]
    late = request(db, generation=2, authority=old["identity"].authority_uuid)
    assert db.publish_claim_cluster_rebuild(**late).result["reason"] == "superseded"
    changed = request(db, generation=4, authority=old["identity"].authority_uuid, method="embeddings")
    changed["calculation"] = skip["calculation"]
    seed(db, text="Changed skip source")
    before = state(db)
    with pytest.raises(ClusterRebuildStorageError, match="claims_cluster_source_changed"):
        db.publish_claim_cluster_rebuild(**changed)
    assert state(db) == before


def test_exact_empty_clears_owner_output_and_inactive_assignment_only(db):
    req = request(db)
    db.publish_claim_cluster_rebuild(**req)
    own = req["snapshot"].rows[0].claim_id
    _, foreign = seed(db, owner=43, text="Foreign")
    with scoped_context(user_id=43, is_admin=False):
        foreign_cluster = db.create_claim_cluster(user_id="43", representative_claim_id=foreign)
        db.add_claim_to_cluster(cluster_id=foreign_cluster["id"], claim_id=foreign)
    db.execute_query("UPDATE Claims SET deleted = 1 WHERE id = ?", (own,), commit=True)
    empty = request(db, generation=2, authority=req["identity"].authority_uuid, empty=True)
    result = db.publish_claim_cluster_rebuild(**empty)
    assert result.result["clusters_created"] == 0
    assert db.execute_query("SELECT claim_cluster_id FROM Claims WHERE id = ?", (own,)).fetchone()[0] is None
    assert (
        db.execute_query("SELECT claim_cluster_id FROM Claims WHERE id = ?", (foreign,)).fetchone()[0]
        == foreign_cluster["id"]
    )


@pytest.mark.parametrize("reference", ["assignment", "membership", "representative"])
@pytest.mark.parametrize("retry", [False, True])
def test_foreign_old_references_fail_closed_even_outside_source_or_after_retry(db, reference, retry):
    req = request(db)
    row = req["snapshot"].rows[0]
    cluster = db.create_claim_cluster(
        user_id="42",
        representative_claim_id=row.claim_id if reference == "representative" else None,
    )
    if reference == "membership":
        db.execute_query(
            "INSERT INTO claim_cluster_membership (cluster_id, claim_id) VALUES (?, ?)",
            (cluster["id"], row.claim_id),
            commit=True,
        )
    if reference == "assignment":
        db.execute_query(
            "UPDATE Claims SET claim_cluster_id = ? WHERE id = ?", (cluster["id"], row.claim_id), commit=True
        )
    if retry:
        db.execute_query("UPDATE Claims SET claim_text = 'Edit' WHERE id = ?", (row.claim_id,), commit=True)
        with pytest.raises(ClusterRebuildStorageError, match="claims_cluster_source_changed"):
            db.publish_claim_cluster_rebuild(**req)
    db.execute_query(
        "UPDATE Media SET owner_user_id = 43, visibility = 'team', team_id = 9, version = version + 1 WHERE id = ?",
        (row.media_id,),
        commit=True,
    )
    req["snapshot"] = snapshot(db)
    req["calculation"] = ClusterCalculation((), "ok", None, 0)
    before = state(db)
    with pytest.raises(ClusterRebuildStorageError) as caught:
        db.publish_claim_cluster_rebuild(**req)
    assert (caught.value.code, caught.value.retryable) == ("claims_cluster_foreign_reference", False)
    assert state(db) == before


def test_sqlite_duplicate_attempts_publish_once_through_commit(db, monkeypatch):
    req = request(db)
    other = MediaDatabase(db_path=db.db_path_str, client_id="42", existing_only=True)
    execute = db._execute_with_connection
    started = threading.Event()
    futures = []

    def duplicate():
        started.set()
        return other.publish_claim_cluster_rebuild(**req)

    with ThreadPoolExecutor(max_workers=1) as pool:

        def pause(conn, sql, params=None):
            result = execute(conn, sql, params)
            if sql.startswith("INSERT INTO claims_cluster_publication_receipts"):
                futures.append(pool.submit(duplicate))
                assert started.wait(5)
                assert not futures[0].done()
            return result

        monkeypatch.setattr(db, "_execute_with_connection", pause)
        try:
            first = db.publish_claim_cluster_rebuild(**req)
            replay = futures[0].result(timeout=5)
            assert first.newly_published and not replay.newly_published
            assert first.result == replay.result
            assert len(db.execute_query("SELECT * FROM claim_clusters").fetchall()) == 1
        finally:
            other.close_connection()


def test_domain_receipt_survives_actual_jobs_and_request_receipt_pruning(db, tmp_path, monkeypatch):
    from tldw_Server_API.app.core.Jobs.manager import JobManager
    from tldw_Server_API.tests.Jobs.test_jobs_resource_admission_sqlite import make_command, make_request, sql

    monkeypatch.setenv("JOBS_ALLOWED_QUEUES_CLAIMS", "default")
    monkeypatch.setenv("JOBS_ARCHIVE_BEFORE_DELETE", "false")
    manager = JobManager(tmp_path / "pruned-jobs.db")
    command = make_command(owner_user_id="42", payload={**make_command().payload, "owner_user_id": "42", "min_size": 1})
    job = manager.admit_resource_job(command, request=make_request(command)).job
    req = request(db, authority=job["resource_authority_uuid"])
    req["identity"] = replace(req["identity"], job_uuid=job["uuid"], generation=job["resource_generation"])
    first = db.publish_claim_cluster_rebuild(**req)
    before = state(db)
    assert manager.cancel_job(job["id"])
    sql(manager, "UPDATE job_resource_request_receipts SET expires_at = '2000-01-01'")
    assert manager.prune_jobs(older_than_days=0, domain="claims") == 1
    assert manager.get_job_by_uuid(job["uuid"]) is None
    assert sql(manager, "SELECT * FROM job_resource_request_receipts", rows=True) == []
    assert db.load_claim_cluster_publication(identity=req["identity"]).result == first.result
    assert state(db) == before


def test_oversized_old_memberships_fail_before_destruction(db):
    req = request(db)
    _, extra = seed(db, text="Inactive")
    cluster = db.create_claim_cluster(user_id="42")
    for claim in (req["snapshot"].rows[0].claim_id, extra):
        db.add_claim_to_cluster(cluster_id=cluster["id"], claim_id=claim)
    db.execute_query("UPDATE Claims SET deleted = 1 WHERE id = ?", (extra,), commit=True)
    req["limits"] = replace(LIMITS, cleanup_references=1)
    before = state(db)
    with pytest.raises(ClusterRebuildStorageError, match="claims_cluster_resource_limit"):
        db.publish_claim_cluster_rebuild(**req)
    assert state(db) == before


FAULTS = [
    "DELETE FROM claim_cluster_membership",
    "UPDATE Claims SET claim_cluster_id = NULL",
    "DELETE FROM claim_clusters",
    "INSERT INTO claim_clusters ",
    "INSERT INTO claim_cluster_membership",
    "UPDATE Claims SET claim_cluster_id = ?",
    "UPDATE claims_cluster_publication_heads",
    "INSERT INTO claims_cluster_publication_receipts",
]


@pytest.mark.parametrize("fault", FAULTS)
def test_every_destructive_and_receipt_fault_rolls_back_all_tables(db, monkeypatch, fault):
    req = request(db)
    db.publish_claim_cluster_rebuild(**req)
    new = request(db, generation=2, authority=req["identity"].authority_uuid)
    before = state(db)
    execute = db._execute_with_connection
    reached = []

    def fail_after(conn, sql, params=None):
        result = execute(conn, sql, params)
        if sql.startswith(fault):
            reached.append(sql)
            raise RuntimeError("injected publication fault")
        return result

    monkeypatch.setattr(db, "_execute_with_connection", fail_after)
    with pytest.raises(RuntimeError, match="injected publication fault"):
        db.publish_claim_cluster_rebuild(**new)
    assert reached
    assert state(db) == before


@pytest.mark.parametrize(
    "field,value",
    [
        ("job_uuid", "x" * 36),
        ("job_uuid", str(uuid4()).upper()),
        ("authority_uuid", "x" * 36),
        ("generation", True),
        ("generation", 0),
        ("generation", 2**63),
        ("settings_fingerprint", "Z" * 64),
        ("owner_user_id", "../42"),
    ],
)
def test_identity_boundary_rejects_noncanonical_values(db, field, value):
    req = request(db)
    req["identity"] = replace(req["identity"], **{field: value})
    before = state(db)
    with pytest.raises(ValueError):
        db.publish_claim_cluster_rebuild(**req)
    with pytest.raises(ValueError):
        db.load_claim_cluster_publication(identity=req["identity"])
    assert state(db) == before


@pytest.mark.parametrize(
    "bad", ["foreign_member", "foreign_rep", "bool_id", "duplicate", "nan", "text", "reason", "count"]
)
def test_calculation_boundary_is_fail_closed(db, bad):
    req = request(db)
    own = req["snapshot"].rows[0].claim_id
    _, foreign = seed(db, owner=43, text="Foreign calculation")
    assignment = req["calculation"].assignments[0]
    if bad == "foreign_member":
        assignment = replace(assignment, members=(ClusterMember(foreign, 1.0),))
    elif bad == "foreign_rep":
        assignment = replace(assignment, representative_claim_id=foreign)
    elif bad == "bool_id":
        assignment = replace(assignment, members=(ClusterMember(True, 1.0),))
    elif bad == "duplicate":
        assignment = replace(assignment, members=(ClusterMember(own, 1.0), ClusterMember(own, 1.0)))
    elif bad == "nan":
        assignment = replace(assignment, members=(ClusterMember(own, float("nan")),))
    elif bad == "text":
        assignment = replace(assignment, canonical_claim_text="x" * (LIMITS.text_bytes + 1))
    calculation = replace(req["calculation"], assignments=(assignment,))
    if bad == "reason":
        calculation = replace(calculation, reason="secret text")
    if bad == "count":
        calculation = replace(calculation, claims_skipped=True)
    req["calculation"] = calculation
    before = state(db)
    with pytest.raises(ClusterRebuildStorageError):
        db.publish_claim_cluster_rebuild(**req)
    assert state(db) == before


def test_receipt_metadata_tampering_is_not_replay(db):
    req = request(db)
    db.publish_claim_cluster_rebuild(**req)
    db.execute_query(
        "UPDATE claims_cluster_publication_receipts SET settings_fingerprint = ?", ("a" * 64,), commit=True
    )
    before = state(db)
    with pytest.raises(ClusterRebuildStorageError, match="claims_cluster_corruption"):
        db.load_claim_cluster_publication(identity=req["identity"])
    with pytest.raises(ClusterRebuildStorageError, match="claims_cluster_corruption"):
        db.publish_claim_cluster_rebuild(**req)
    assert state(db) == before


@pytest.mark.parametrize("damage", ["missing", "authority", "equal_uuid", "rewound"])
def test_receipt_cannot_rebind_a_missing_or_corrupt_head(db, damage):
    req = request(db, generation=2)
    db.publish_claim_cluster_rebuild(**req)
    if damage == "missing":
        db.execute_query("DELETE FROM claims_cluster_publication_heads", commit=True)
    elif damage == "authority":
        db.execute_query("UPDATE claims_cluster_publication_heads SET authority_uuid = ?", (str(uuid4()),), commit=True)
    elif damage == "equal_uuid":
        db.execute_query("UPDATE claims_cluster_publication_heads SET job_uuid = ?", (str(uuid4()),), commit=True)
    else:
        db.execute_query("UPDATE claims_cluster_publication_heads SET generation = 1", commit=True)
    before = state(db)
    with pytest.raises(ClusterRebuildStorageError, match="claims_cluster_corruption"):
        db.load_claim_cluster_publication(identity=req["identity"])
    with pytest.raises(ClusterRebuildStorageError, match="claims_cluster_corruption"):
        db.publish_claim_cluster_rebuild(**req)
    assert state(db) == before


def test_nested_publication_does_not_claim_an_uncommitted_publication(db):
    req = request(db)
    before = state(db)
    with db.transaction():
        with pytest.raises(ValueError, match="own transaction"):
            db.publish_claim_cluster_rebuild(**req)
    assert state(db) == before


@pytest.mark.parametrize("method", ["exact", "embeddings"])
def test_calculator_handoff_publishes_real_members_and_partial_coverage(db, method):
    from tldw_Server_API.app.core.Claims_Extraction.claims_cluster_rebuild import calculate_claim_cluster_rebuild

    seed(db, text="First")
    seed(db, text="Second")
    req = request(db, method=method)
    req["calculation"] = calculate_claim_cluster_rebuild(
        req["snapshot"],
        settings=req["settings"],
        limits=req["limits"],
        read_vectors=lambda ids: {ids[0]: [1.0, 0.0]},
    )
    result = db.publish_claim_cluster_rebuild(**req)
    expected = (2, 2, 0) if method == "exact" else (1, 1, 1)
    assert tuple(result.result[key] for key in ("clusters_created", "claims_assigned", "claims_skipped")) == expected
    assert len(db.execute_query("SELECT * FROM claim_cluster_membership").fetchall()) == expected[1]
