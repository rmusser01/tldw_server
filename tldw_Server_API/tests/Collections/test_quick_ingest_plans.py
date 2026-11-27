"""Durable setup and claim contracts across both Collections backends."""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.core.DB_Management.backends.base import BackendType, DatabaseConfig
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.Collections_DB import CollectionsDatabase


@pytest.fixture(params=["sqlite", "postgres"])
def backend(request, tmp_path):
    config = (
        DatabaseConfig(backend_type=BackendType.SQLITE, sqlite_path=str(tmp_path / "collections.db"))
        if request.param == "sqlite"
        else request.getfixturevalue("pg_database_config")
    )
    backend = DatabaseBackendFactory.create_backend(config)
    yield backend
    CollectionsDatabase._bootstrapped_backend_targets.discard(CollectionsDatabase._backend_target_key(backend))
    backend.get_pool().close_all()


def setup(db, keys=("talk",), title="Original"):
    return db.put_quick_ingest_plan(
        "draft",
        {"name": title, "kind": "conference"},
        [
            (
                key,
                {
                    "source_url": "https://example.test/same",
                    "title": title,
                    "metadata": {"quick_ingest_item_id": "editable"},
                },
            )
            for key in keys
        ],
    )


def test_plan_replay_preserves_current_binding_after_edit_and_after_claim(backend):
    db = CollectionsDatabase.from_backend("alice", backend)
    collection, original = setup(db)
    first = original["talk"]
    assert first.idempotency_key
    db.update_media_collection_item(
        first.id, ordinal=9, title="Stored edit", metadata={"quick_ingest_item_id": "changed"}
    )
    recovered, rows = setup(db, ("another", "talk"), "New")
    assert recovered == collection
    assert (rows["talk"].id, rows["talk"].ordinal, rows["talk"].title, rows["talk"].idempotency_key) == (
        first.id,
        9,
        "Stored edit",
        first.idempotency_key,
    )
    assert rows["another"].id != first.id
    assert rows["another"].title == "New"
    assert db.claim_quick_ingest_source("draft", "talk")[0]
    assert not db.claim_quick_ingest_source("draft", "talk")[0]
    db.begin_quick_ingest_dispatch(
        "draft", "talk", retry_count=0, idempotency_key=first.idempotency_key, transport="jobs"
    )
    db.finish_quick_ingest_dispatch(
        "draft",
        "talk",
        retry_count=0,
        idempotency_key=first.idempotency_key,
        transport="jobs",
        status="processing",
        latest_job_id="33",
    )
    db.resolve_media_collection_item(first.id, media_id=25, status="completed", latest_job_id="33")
    assert setup(db)[1]["talk"].media_id == 25
    assert db.claim_quick_ingest_source("draft", "talk")[1].latest_job_id == "33"
    other = CollectionsDatabase.from_backend("bob", backend)
    with pytest.raises(KeyError):
        other.claim_quick_ingest_source("draft", "talk")
    other_collection, other_rows = setup(other)
    assert other_collection != collection and other_rows["talk"].id != first.id


def test_concurrent_resume_and_dispatch_has_one_winner(backend):
    db = CollectionsDatabase.from_backend("alice", backend)
    start = Barrier(2)

    def resume(extra):
        start.wait(timeout=5)
        collection, rows = setup(db, ("talk", extra))
        claimed, _ = db.claim_quick_ingest_source("draft", "talk")
        return collection, rows["talk"].id, claimed

    with ThreadPoolExecutor(max_workers=2) as pool:
        first, second = pool.map(resume, ("one", "two"))
    assert first[:2] == second[:2] and first[2] != second[2]
    assert len(db.get_media_collection(first[0]).items) == 3


def test_plan_failure_rolls_back_all_rows(backend, monkeypatch):
    db = CollectionsDatabase.from_backend("alice", backend)
    original = db.add_media_collection_item

    def fail_second(**kwargs):
        if kwargs["source_url"].endswith("second"):
            raise RuntimeError("injected before commit")
        return original(**kwargs)

    monkeypatch.setattr(db, "add_media_collection_item", fail_second)
    with pytest.raises(RuntimeError, match="injected"):
        db.put_quick_ingest_plan(
            "draft",
            {"name": "Original", "kind": "conference"},
            [
                ("a", {"source_url": "first"}),
                ("b", {"source_url": "second"}),
            ],
        )
    assert db.list_media_collections()[1] == 0
    assert len(setup(db)[1]) == 1


def test_plan_api_validates_authority_and_recovers_current_receipt(tmp_path, monkeypatch):
    from tldw_Server_API.app.api.v1.API_Deps.Collections_DB_Deps import get_collections_db_for_user
    from tldw_Server_API.app.api.v1.endpoints.media import collections

    monkeypatch.setenv("TEST_MODE", "true")
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("SINGLE_USER_API_KEY", "test-api-key-12345")
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path))
    monkeypatch.delenv("JOBS_DB_URL", raising=False)
    monkeypatch.setenv("JOBS_DB_PATH", str(tmp_path / "jobs.sqlite"))
    app = FastAPI()
    app.include_router(collections.router, prefix="/media")
    with (
        CollectionsDatabase.for_user("1") as db,
        TestClient(app, headers={"X-API-KEY": "test-api-key-12345"}) as client,
    ):
        app.dependency_overrides[get_collections_db_for_user] = lambda: db
        url = "/media/collections/quick-ingest-plans/draft"
        source = {"source_key": "talk", "item": {"source_url": "https://example.test", "title": "First"}}

        def put(items):
            return client.put(url, json={"collection": {"name": "Talks"}, "items": items})

        source_status = url + "/sources/talk"
        assert client.get(source_status).status_code == 404
        assert db.list_media_collections()[1] == 0
        for key in (".", "..", "embedded/slash"):
            assert put([{**source, "source_key": key}]).status_code == 422
        assert put([source, source]).status_code == 422
        assert put([{**source, "item": {**source["item"], "media_id": 88}}]).status_code == 422
        # A direct source can be planned and reserved even when Jobs storage is offline.
        from tldw_Server_API.app.api.v1.endpoints.media import ingest_jobs

        def jobs_offline():
            raise RuntimeError("Jobs storage unavailable")

        with monkeypatch.context() as patch:
            app.dependency_overrides[ingest_jobs.get_job_manager] = jobs_offline
            patch.setattr(ingest_jobs, "get_job_manager", jobs_offline)
            response = put([source])
            assert response.status_code == 200, response.text
        app.dependency_overrides.pop(ingest_jobs.get_job_manager)
        before = response.json()
        assert client.get(source_status).json() == {"collection_id": before["collection_id"], **before["items"][0]}
        assert client.get(url + "/sources/absent").status_code == 404
        assert len(db.get_media_collection(before["collection_id"]).items) == 1
        assert before["items"][0]["source_key"] == "talk"
        assert put([source]).json() == before
        claim_url = url + "/sources/talk/claim"
        assert client.post(claim_url).json()["claimed"]
        assert not client.post(claim_url).json()["claimed"]
        row = before["items"][0]["item"]
        token = {"retry_count": 0, "idempotency_key": row["idempotency_key"], "transport": "web"}
        db.begin_quick_ingest_dispatch("draft", "talk", **token)
        db.finish_quick_ingest_dispatch("draft", "talk", status="failed", **token)
        retry = {"retry_count": 1, "idempotency_key": "next", "source_url": "https://example.test/corrected"}
        won = client.post(claim_url, json=retry).json()
        assert won["claimed"] and won["item"]["source_url"] == retry["source_url"]
        assert not client.post(claim_url, json=retry).json()["claimed"]
        item_url = f"/media/collections/{before['collection_id']}/items/{row['id']}"
        assert client.patch(item_url, json={"status": "completed", "media_id": 999}).status_code == 409
        stale = {"status": "failed", "retry_count": 0, "idempotency_key": row["idempotency_key"]}
        assert client.patch(item_url, json=stale).status_code == 409
        assert client.patch(item_url, json={"title": "Author edit"}).status_code == 200
        assert db.get_media_collection_item(row["id"]).idempotency_key == "next"
        assert db.get_media_collection_item(row["id"]).media_id is None
        # The Jobs transaction can commit before its HTTP handler projects the ID into Collections.
        from tldw_Server_API.app.api.v1.endpoints.media.ingest_jobs import get_job_manager

        manager = get_job_manager()
        assert db.begin_quick_ingest_dispatch("draft", "talk", retry_count=1, idempotency_key="next", transport="jobs")[
            0
        ]
        payload = {
            "collection_id": str(row["collection_id"]),
            "planned_item_id": str(row["id"]),
            "idempotency_key": "next",
            "source_kind": "url",
            "source": retry["source_url"],
        }
        job = manager.create_job(
            domain="media_ingest",
            queue="default",
            job_type="media_ingest_item",
            owner_user_id="1",
            idempotency_key="media-collection-url:1:1:next",
            payload=payload,
        )
        observed = client.get(source_status).json()
        assert observed["item"]["latest_job_id"] == str(job["id"])
        assert db.get_media_collection_item(row["id"]).latest_job_id is None
        recovered = put([source]).json()["items"][0]
        assert recovered["item"]["latest_job_id"] == str(job["id"])
        lease = manager.acquire_next_job(
            domain="media_ingest", queue="default", lease_seconds=60, worker_id="pdf-worker"
        )
        manager.complete_job(
            job["id"], worker_id="pdf-worker", lease_id=lease["lease_id"], result={"status": "Success", "media_id": 82}
        )
        recovered = put([source]).json()["items"][0]
        assert (recovered["dispatch_state"], recovered["item"]["media_id"]) == ("settled", 82)
        with CollectionsDatabase.for_user("2") as other:
            app.dependency_overrides[get_collections_db_for_user] = lambda: other
            assert client.get(source_status).status_code == 404
            assert other.list_media_collections()[1] == 0
        app.dependency_overrides[get_collections_db_for_user] = lambda: db
        db.delete_media_collection(before["collection_id"])
        assert client.get(source_status).status_code == 404


def test_recovered_reservation_fences_transport_and_stale_attempts(backend):
    from tldw_Server_API.app.core.DB_Management.Collections_DB import QuickIngestPlanConflict

    db = CollectionsDatabase.from_backend("alice", backend)
    original = setup(db)[1]["talk"]
    key = original.idempotency_key
    assert db.claim_quick_ingest_source("draft", "talk")[0]
    assert setup(db)[1]["talk"].dispatch_state == "reserved"

    def begin(mode, attempt=0, token=key):
        return db.begin_quick_ingest_dispatch(
            "draft", "talk", retry_count=attempt, idempotency_key=token, transport=mode
        )

    def finish(mode, status, **kwargs):
        return db.finish_quick_ingest_dispatch(
            "draft", "talk", retry_count=0, idempotency_key=key, transport=mode, status=status, **kwargs
        )

    assert not begin("web", token="another-key")[0]
    assert begin("jobs")[0]
    assert not begin("web")[0]
    assert not begin("add")[0]
    assert finish("jobs", "fallback_ready").dispatch_state == "fallback_ready"
    assert begin("add")[0]
    assert not begin("jobs")[0]
    assert not begin("add")[0]
    finish("add", "failed")
    assert db.claim_quick_ingest_source("draft", "talk", 1, "retry-one", "https://example.test/fixed")[0]
    assert not db.claim_quick_ingest_source("draft", "talk", 1, "retry-one")[0]
    with pytest.raises(QuickIngestPlanConflict):
        finish("add", "completed", media_id=100)
    assert begin("web", 1, "retry-one")[0]
    db.finish_quick_ingest_dispatch(
        "draft", "talk", retry_count=1, idempotency_key="retry-one", transport="web", status="completed", media_id=71
    )
    assert not begin("web", 1, "retry-one")[0]
    assert setup(db)[1]["talk"].media_id == 71


def test_one_qualified_consumer_can_dispatch_after_lost_claim_ack(backend):
    db = CollectionsDatabase.from_backend("alice", backend)
    row = setup(db)[1]["talk"]
    db.claim_quick_ingest_source("draft", "talk")
    gate = Barrier(2)

    def deliver(mode):
        gate.wait(timeout=5)
        return db.begin_quick_ingest_dispatch(
            "draft", "talk", retry_count=0, idempotency_key=row.idempotency_key, transport=mode
        )[0]

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(deliver, ["jobs", "web"])) == [False, True]


def test_attempt_tokens_are_never_reassigned_to_another_source_or_attempt(backend):
    from tldw_Server_API.app.core.DB_Management.Collections_DB import QuickIngestPlanConflict

    db = CollectionsDatabase.from_backend("alice", backend)
    rows = setup(db, ("one", "two"))[1]
    for name, row in rows.items():
        db.claim_quick_ingest_source("draft", name)
        args = {"retry_count": 0, "idempotency_key": row.idempotency_key, "transport": "web"}
        db.begin_quick_ingest_dispatch("draft", name, **args)
        db.finish_quick_ingest_dispatch("draft", name, status="failed", **args)
    with pytest.raises(QuickIngestPlanConflict):
        db.claim_quick_ingest_source("draft", "one", 1, rows["two"].idempotency_key)
    assert db.claim_quick_ingest_source("draft", "one", 1, "new")[0]
    args = {"retry_count": 1, "idempotency_key": "new", "transport": "jobs"}
    db.begin_quick_ingest_dispatch("draft", "one", **args)
    db.finish_quick_ingest_dispatch("draft", "one", status="processing", latest_job_id="51", **args)
    for rejected in ("failed", "cancelled", "fallback_ready"):
        with pytest.raises(QuickIngestPlanConflict):
            db.finish_quick_ingest_dispatch("draft", "one", status=rejected, **args)
    assert db.get_quick_ingest_source("draft", "one").status == "processing"
    # A late response from an unrelated/old worker cannot overwrite the new receipt.
    assert db.resolve_media_collection_item(rows["one"].id, media_id=48, latest_job_id="50").media_id is None
    db.finish_quick_ingest_dispatch("draft", "one", status="failed", latest_job_id="51", **args)
    with pytest.raises(QuickIngestPlanConflict):
        db.claim_quick_ingest_source("draft", "one", 2, rows["one"].idempotency_key)


def test_confirmed_duplicate_can_settle_without_fabricating_a_media_id(backend):
    db = CollectionsDatabase.from_backend("alice", backend)
    row = setup(db)[1]["talk"]
    args = {"retry_count": 0, "idempotency_key": row.idempotency_key, "transport": "web"}
    db.claim_quick_ingest_source("draft", "talk")
    db.begin_quick_ingest_dispatch("draft", "talk", **args)
    with pytest.raises(ValueError):
        db.finish_quick_ingest_dispatch("draft", "talk", status="completed", **args)
    db.finish_quick_ingest_dispatch("draft", "talk", status="skipped_existing", **args)
    observed = db.get_quick_ingest_source("draft", "talk")
    assert (observed.dispatch_state, observed.status, observed.media_id) == ("settled", "skipped_existing", None)
    assert not db.claim_quick_ingest_source("draft", "talk", 1, "later")[0]


def test_before_submission_stop_only_wins_against_unconsumed_reservation(backend):
    from tldw_Server_API.app.core.DB_Management.Collections_DB import QuickIngestPlanConflict

    db = CollectionsDatabase.from_backend("alice", backend)
    rows = setup(db, ("waiting", "sent"))[1]
    for name, row in rows.items():
        db.claim_quick_ingest_source("draft", name)
        token = {"retry_count": 0, "idempotency_key": row.idempotency_key}
        cancel = {**token, "status": "cancelled", "error_summary": "Stopped before submission."}
        if name == "sent":
            db.begin_quick_ingest_dispatch("draft", name, transport="web", **token)
            with pytest.raises(QuickIngestPlanConflict):
                db.apply_quick_ingest_item_patch(row.id, cancel)
            assert db.get_quick_ingest_source("draft", name).dispatch_state == "web"
        else:
            stopped = db.apply_quick_ingest_item_patch(row.id, cancel)
            assert (stopped.status, stopped.error_summary) == ("cancelled", "Stopped before submission.")
            assert not db.begin_quick_ingest_dispatch("draft", name, transport="jobs", **token)[0]
