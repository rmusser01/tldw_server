"""HTTP -> Jobs -> actual WorkerSDK -> Media, including retained UUID links."""

import asyncio

import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.endpoints import claims
from tldw_Server_API.app.core.Claims_Extraction import claims_job_handlers as handlers
from tldw_Server_API.app.core.Claims_Extraction.claims_embeddings import claim_embedding_id
from tldw_Server_API.app.core.Embeddings.ChromaDB_Library import ChromaDBManager
from tldw_Server_API.tests.Claims.test_claims_cluster_rebuild_jobs_api import PATH, build_app
from tldw_Server_API.tests.Claims.test_claims_cluster_rebuild_jobs_api import manager as manager
from tldw_Server_API.tests.Claims.test_claims_cluster_rebuild_jobs_worker import EMBEDDINGS, output, run_worker
from tldw_Server_API.tests.Claims.test_claims_cluster_rebuild_recovery import (
    jobs_pg_dsn as jobs_pg_dsn,
)
from tldw_Server_API.tests.Claims.test_claims_cluster_rebuild_recovery import (
    recovery_scenario as recovery_scenario,
)
from tldw_Server_API.tests.Claims.test_claims_cluster_rebuild_recovery import (
    recovery_storage as recovery_storage,
)
from tldw_Server_API.tests.DB_Management.test_claim_cluster_rebuild_publication import state
from tldw_Server_API.tests.DB_Management.test_claim_cluster_rebuild_snapshot import seed
from tldw_Server_API.tests.Jobs.test_jobs_owned_status_api import principal
from tldw_Server_API.tests.Jobs.test_jobs_resource_admission_sqlite import sql

pytestmark = pytest.mark.integration
pytest_plugins = ["tldw_Server_API.tests._plugins.authnz_isolated_fixtures"]
USE_SHARED_JOBS_POSTGRES = True


@pytest.mark.parametrize("filler_count", [1, 199, 200])
async def test_colliding_embedding_ids_publish_last_source_once_across_reader_batches(
    recovery_scenario, filler_count, tmp_path, monkeypatch
):
    s = recovery_scenario
    media, first_claim = seed(s.db, text="Repeated claim")
    s.db.upsert_claims(
        [
            {"media_id": media, "chunk_index": i, "claim_text": f"Unique claim {i}", "chunk_hash": str(i)}
            for i in range(1, filler_count + 1)
        ]
        + [{"media_id": media, "chunk_index": 0, "claim_text": "Repeated claim", "chunk_hash": "duplicate"}]
    )
    source = s.db.read_claim_cluster_source(owner_user_id="42", claim_limit=5000, text_byte_limit=33554432)
    ids = tuple(dict.fromkeys(claim_embedding_id(row.media_id, row.chunk_index, row.claim_text) for row in source.rows))
    config = {
        "USER_DB_BASE_DIR": str(tmp_path / "vectors"),
        "chroma_client_settings": {"anonymized_telemetry": False},
    }
    monkeypatch.setitem(handlers.settings, "USER_DB_BASE_DIR", config["USER_DB_BASE_DIR"])
    monkeypatch.setitem(handlers.settings, "EMBEDDING_CONFIG", config)
    with ChromaDBManager("42", config) as vectors:
        collection = vectors.client.create_collection(
            "claims_for_42", embedding_function=None, configuration={"hnsw": {"batch_size": 2, "sync_threshold": 2}}
        )
        collection.upsert(ids=list(ids), embeddings=[[1.0, 0.0] for _ in ids])
    job = s.admit(settings=EMBEDDINGS).job
    await run_worker(s.manager)
    stored = s.manager.get_job(job["id"])
    assert stored["status"] == "completed", (stored["error_code"], stored["last_error"])
    assert stored["result"]["claims_assigned"] == filler_count + 1
    assert stored["result"]["claims_skipped"] == 1
    members = s.db.execute_query("SELECT claim_id FROM claim_cluster_membership ORDER BY claim_id").fetchall()
    assert [int(member["claim_id"]) for member in members] == [
        row.claim_id for row in source.rows if row.claim_id != first_claim
    ]
    assert len(s.inventory()["receipts"]) == 1


@pytest.mark.parametrize("method", ["exact", "embeddings"])
def test_http_to_sdk_publication_and_terminal_replay(recovery_scenario, method, tmp_path, monkeypatch):
    s = recovery_scenario
    seed(s.db, text="A fact")
    seed(s.db, text="a FACT")
    monkeypatch.setenv("CLAIMS_JOBS_ENABLED", "true")
    monkeypatch.setenv("CLAIMS_CLUSTER_REBUILD_JOBS_ENABLED", "true")
    if method == "embeddings":
        config = {
            **dict(handlers.settings.get("EMBEDDING_CONFIG") or {}),
            "USER_DB_BASE_DIR": str(tmp_path / "vectors"),
        }
        monkeypatch.setitem(handlers.settings, "USER_DB_BASE_DIR", config["USER_DB_BASE_DIR"])
        source = s.db.read_claim_cluster_source(owner_user_id="42", claim_limit=5000, text_byte_limit=33554432)
        with ChromaDBManager("42", config) as vectors:
            collection = vectors.client.create_collection("claims_for_42", embedding_function=None)
            collection.upsert(
                ids=[claim_embedding_id(row.media_id, row.chunk_index, row.claim_text) for row in source.rows],
                embeddings=[[1.0, 0.0], [1.0, 0.0]],
            )
    before_admission = state(s.db)
    with TestClient(build_app(principal(42))) as client:
        accepted = client.post(PATH, params={"method": method}, headers={"Idempotency-Key": "transport"})
        assert accepted.status_code == 202, accepted.text
        reference = accepted.json()
        assert reference["reused"] is False
        assert accepted.headers["location"] == reference["status_url"]
        assert state(s.db) == before_admission
        asyncio.run(run_worker(s.manager))
        status = client.get(reference["status_url"])
        assert status.status_code == 200, status.text
        assert status.json()["status"] == "completed"
        assert status.json()["result"]["claims_assigned"] == 2
        assert status.json()["result"]["clusters_created"] == 1
        before = output(s.db)
        replay = client.post(PATH, params={"method": method}, headers={"Idempotency-Key": "transport"})
        assert replay.json() == {**reference, "reused": True}
        assert output(s.db) == before
        assert len(s.db.execute_query("SELECT * FROM claims_cluster_publication_receipts").fetchall()) == 1


def test_lost_http_ack_replays_committed_admission_through_sdk(recovery_scenario, monkeypatch):
    s = recovery_scenario
    seed(s.db, text="A fact")
    seed(s.db, text="a FACT")
    monkeypatch.setenv("CLAIMS_JOBS_ENABLED", "true")
    monkeypatch.setenv("CLAIMS_CLUSTER_REBUILD_JOBS_ENABLED", "true")
    producer = claims.get_job_manager()
    real_admit = producer.admit_resource_job

    def lose_ack(*args, **kwargs):
        real_admit(*args, **kwargs)
        raise ConnectionError("lost transport acknowledgement")

    before = state(s.db)
    with TestClient(build_app(principal(42))) as client:
        with monkeypatch.context() as patch:
            patch.setattr(producer, "admit_resource_job", lose_ack)
            lost = client.post(PATH, params={"method": "exact"}, headers={"Idempotency-Key": "lost"})
        assert lost.status_code == 503
        assert lost.json() == {"detail": "Claims cluster rebuild admission unavailable"}
        admitted = s.manager.list_jobs(domain="claims")
        assert len(admitted) == 1
        assert state(s.db) == before
        accepted = client.post(PATH, params={"method": "exact"}, headers={"Idempotency-Key": "lost"})
        assert accepted.status_code == 202, accepted.text
        reference = accepted.json()
        assert reference["reused"] is True
        assert reference["job_uuid"] == admitted[0]["uuid"]
        asyncio.run(run_worker(s.manager))
        status = client.get(reference["status_url"])
        assert status.status_code == 200, status.text
        assert status.json()["status"] == "completed"
        assert status.json()["result"]["claims_assigned"] == 2
        assert status.json()["result"]["clusters_created"] == 1
        published = state(s.db)
        replay = client.post(PATH, params={"method": "exact"}, headers={"Idempotency-Key": "lost"})
        assert replay.json() == reference
        assert state(s.db) == published


@pytest.mark.parametrize("foreign", [False, True])
def test_real_status_links_remain_exact_after_archive_id_reuse_and_pruning(manager, monkeypatch, foreign):
    """SQLite actually reuses IDs; neither URL may resolve its replacement."""
    monkeypatch.setenv("JOBS_ARCHIVE_BEFORE_DELETE", "true")
    user = principal(1, roles=["admin"], permissions=["*"]) if foreign else principal(42)
    app = build_app(user, admin=foreign)
    with TestClient(app) as client:
        accepted = client.post(PATH, params={"method": "exact", **({"user_id": 42} if foreign else {})})
        assert accepted.status_code == 202, accepted.text
        ref = accepted.json()
        assert manager.cancel_job(ref["job_id"], expected_uuid=ref["job_uuid"])
        assert manager.prune_jobs(older_than_days=0, domain="claims") == 1
        replacement = manager.create_job(
            domain="claims", queue="default", job_type="other", payload={}, owner_user_id="99"
        )
        assert replacement["id"] == ref["job_id"] and replacement["uuid"] != ref["job_uuid"]
        archived = client.get(ref["status_url"])
        assert archived.status_code == 200, archived.text
        assert archived.json()["uuid"] == ref["job_uuid"]
        # Archive retention's final deletion is fixture state; there is no
        # Claims history reconstruction after ordinary Jobs retention ends.
        sql(manager, "DELETE FROM jobs_archive WHERE uuid=?", (ref["job_uuid"],))
        assert client.get(ref["status_url"]).status_code == 404
