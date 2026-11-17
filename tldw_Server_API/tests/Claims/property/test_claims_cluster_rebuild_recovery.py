"""Generated real-store replay and interleaved generation qualification."""

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tldw_Server_API.app.core.Claims_Extraction.claims_cluster_rebuild import calculate_claim_cluster_rebuild
from tldw_Server_API.app.core.DB_Management.media_db.claims_cluster_rebuild_types import ClusterPublicationIdentity
from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase
from tldw_Server_API.app.core.DB_Management.scope_context import scoped_context
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.tests.Claims.property.test_claims_cluster_rebuild_limits import LIMITS
from tldw_Server_API.tests.Claims.test_claims_cluster_rebuild_jobs_worker import EMBEDDINGS, EXACT, enqueue
from tldw_Server_API.tests.DB_Management.test_claim_cluster_rebuild_publication import state
from tldw_Server_API.tests.DB_Management.test_claim_cluster_rebuild_snapshot import seed

pytestmark = pytest.mark.integration


@settings(max_examples=25, deadline=None)
@given(st.permutations(range(1, 6)), st.sets(st.integers(1, 5)), st.lists(st.integers(1, 5), max_size=10))
def test_interleaved_generations_and_duplicate_replays_preserve_latest_output(order, skips, duplicates):
    """A missing fence or replay receipt would change stored IDs/head here."""
    with TemporaryDirectory() as directory, pytest.MonkeyPatch.context() as env:
        env.delenv("JOBS_DB_URL", raising=False)
        env.setenv("JOBS_DB_PATH", str(Path(directory) / "observer.db"))
        manager = JobManager(Path(directory) / "jobs.db")
        db = MediaDatabase(db_path=Path(directory) / "Media_DB_v2.db", client_id="42")
        try:
            seed(db, text="A fact")
            with scoped_context(user_id=42, is_admin=False):
                source = db.read_claim_cluster_source(owner_user_id="42", claim_limit=50000, text_byte_limit=33554432)
                requests = {}
                for generation in range(1, 6):
                    effective = EMBEDDINGS if generation in skips else EXACT
                    job = enqueue(manager, settings=effective).job
                    assert job["resource_generation"] == generation
                    requests[generation] = {
                        "identity": ClusterPublicationIdentity(
                            "42",
                            job["uuid"],
                            job["resource_authority_uuid"],
                            generation,
                            job["resource_settings_fingerprint"],
                        ),
                        "settings": effective,
                        "snapshot": source,
                        "calculation": calculate_claim_cluster_rebuild(
                            source, settings=effective, limits=LIMITS, read_vectors=None
                        ),
                        "limits": LIMITS,
                    }
                    assert manager.cancel_job(job["id"], expected_uuid=job["uuid"])
                high_water = 0
                published = {}
                for generation in [*order, *duplicates]:
                    before = state(db)
                    result = db.publish_claim_cluster_rebuild(**requests[generation])
                    if generation in published:
                        assert not result.newly_published
                        assert result.result == published[generation]
                        assert state(db) == before
                    elif generation < high_water:
                        assert not result.newly_published and result.result["reason"] == "superseded"
                        assert state(db) == before
                    else:
                        high_water = generation
                        published[generation] = result.result
                        assert result.newly_published
                        if generation in skips:
                            assert state(db)["claim_clusters"] == before["claim_clusters"]
                    head = db.execute_query(
                        "SELECT generation,job_uuid FROM claims_cluster_publication_heads"
                    ).fetchone()
                    assert head["generation"] == high_water
                    assert head["job_uuid"] == requests[high_water]["identity"].job_uuid
                    assert len(db.execute_query("SELECT * FROM claims_cluster_publication_receipts").fetchall()) == len(
                        published
                    )
                # Replaying a receipt never returns newly_published, even when
                # a later preserving skip advanced the head.
                for generation, result in published.items():
                    assert db.load_claim_cluster_publication(identity=requests[generation]["identity"]).result == result
        finally:
            db.close_connection()


@settings(max_examples=20, deadline=None)
@given(st.integers(1, 30), st.integers(1, 8))
def test_failed_bounded_rebuild_leaves_prior_receipt_and_output_intact(text_size, excess):
    from tldw_Server_API.app.core.DB_Management.media_db.claims_cluster_rebuild_types import ClusterRebuildStorageError

    with TemporaryDirectory() as directory, pytest.MonkeyPatch.context() as env:
        env.delenv("JOBS_DB_URL", raising=False)
        env.setenv("JOBS_DB_PATH", str(Path(directory) / "observer.db"))
        manager = JobManager(Path(directory) / "jobs.db")
        db = MediaDatabase(db_path=Path(directory) / "Media_DB_v2.db", client_id="42")
        try:
            seed(db, text="x" * (text_size + excess))
            with scoped_context(user_id=42, is_admin=False):
                source = db.read_claim_cluster_source(owner_user_id="42", claim_limit=50000, text_byte_limit=33554432)
                job = enqueue(manager).job
                identity = ClusterPublicationIdentity(
                    "42", job["uuid"], job["resource_authority_uuid"], 1, job["resource_settings_fingerprint"]
                )
                calculation = calculate_claim_cluster_rebuild(source, settings=EXACT, limits=LIMITS, read_vectors=None)
                db.publish_claim_cluster_rebuild(
                    identity=identity, settings=EXACT, snapshot=source, calculation=calculation, limits=LIMITS
                )
                before = state(db)
                with pytest.raises(ClusterRebuildStorageError, match="claims_cluster_resource_limit"):
                    calculate_claim_cluster_rebuild(
                        source, settings=EXACT, limits=replace(LIMITS, text_bytes=text_size), read_vectors=None
                    )
                assert state(db) == before
                assert db.load_claim_cluster_publication(identity=identity).result["claims_assigned"] == 1
        finally:
            db.close_connection()
