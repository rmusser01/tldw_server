"""Verify that the receiving transport consumes one owner-bound Quick Ingest attempt."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import HTTPException, Response

from tldw_Server_API.app.api.v1.endpoints.media import quick_ingest_dispatch as dispatch
from tldw_Server_API.app.core.DB_Management.Collections_DB import CollectionsDatabase
from tldw_Server_API.app.core.Ingestion_Media_Processing import input_sourcing
from tldw_Server_API.app.core.Usage import quota_checks
from tldw_Server_API.app.services.enhanced_web_scraping_service import WebScrapingService


def args(transport):
    if transport == "web":
        return {
            "payload": SimpleNamespace(
                url_input="https://example.test/a", mode="persist", scrape_method="Individual URLs"
            )
        }
    return {"form_data": SimpleNamespace(urls=["https://example.test/a"]), "files": []}


@pytest.fixture
def plan_store(tmp_path, monkeypatch):
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path / "users"))
    with CollectionsDatabase.for_user("1") as store:
        yield store


@pytest.fixture
def attempt(plan_store):
    query = _reserve(plan_store, "https://example.test/a", "url")
    return dispatch.PlanAttempt("plan", "source", db=plan_store, **query)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "transport,reply,identity",
    [
        ("web", {"status": "persist-ok", "media_ids": [42]}, ("media_id", 42)),
        ("jobs", {"batch_id": "b", "jobs": [{"id": 51}]}, ("latest_job_id", "51")),
    ],
)
async def test_lost_ack_returns_original_receipt_without_dispatching_again(attempt, transport, reply, identity):
    call = AsyncMock(return_value=reply)
    first = await dispatch.receive(attempt, transport, args(transport), call)
    replay = await dispatch.receive(attempt, transport, args(transport), call)
    assert first["dispatched"] and not replay["dispatched"]
    assert first["result"] == reply and getattr(replay["item"], identity[0]) == identity[1]
    call.assert_awaited_once()


@pytest.mark.asyncio
async def test_dispatch_records_json_receipt_from_memoryview_response(attempt):
    response = Response(memoryview(b'{"media_ids":[42]}'), media_type="application/json")
    saved = await dispatch.receive(attempt, "web", args("web"), AsyncMock(return_value=response))
    assert saved["result"] == {"media_ids": [42]}
    assert attempt.db.get_quick_ingest_source("plan", "source").media_id == 42


@pytest.mark.asyncio
async def test_only_a_definitive_jobs_quota_rejection_allows_the_sync_transport(attempt):
    from tldw_Server_API.app.core.exceptions import BadRequestError

    api_error = HTTPException(429, "Concurrent job limit exceeded")
    api_error.__cause__ = BadRequestError("Concurrent job limit exceeded")
    with pytest.raises(HTTPException) as rejected:
        await dispatch.receive(attempt, "jobs", args("jobs"), AsyncMock(side_effect=api_error))
    assert rejected.value.detail["code"] == "plan_jobs_quota_rejected"
    assert not (await dispatch.receive(attempt, "jobs", args("jobs"), AsyncMock()))["dispatched"]
    call = AsyncMock(return_value={"results": [{"status": "Success", "db_id": 42}]})
    saved = await dispatch.receive(attempt, "add", args("add"), call)
    assert saved["item"].media_id == 42
    assert not (await dispatch.receive(attempt, "add", args("add"), call))["dispatched"]
    call.assert_awaited_once()


@pytest.mark.asyncio
async def test_ambiguous_interruption_does_not_enable_another_write_or_another_transport(attempt):
    with pytest.raises(RuntimeError, match="lost"):
        await dispatch.receive(attempt, "jobs", args("jobs"), AsyncMock(side_effect=RuntimeError("lost")))
    call = AsyncMock()
    assert not (await dispatch.receive(attempt, "add", args("add"), call))["dispatched"]
    assert not (await dispatch.receive(attempt, "jobs", args("jobs"), call))["dispatched"]
    call.assert_not_awaited()


@pytest.mark.asyncio
async def test_mismatched_source_or_attempt_does_not_consume_the_valid_source(attempt):
    call = AsyncMock()
    wrong = args("web")
    wrong["payload"].url_input = "https://example.test/other"
    with pytest.raises(HTTPException) as mismatch:
        await dispatch.receive(attempt, "web", wrong, call)
    assert mismatch.value.status_code == 409
    assert attempt.db.get_quick_ingest_source("plan", "source").dispatch_state == "reserved"
    stale = dispatch.PlanAttempt("plan", "source", 1, "old", attempt.db)
    with pytest.raises(HTTPException) as mismatch:
        await dispatch.receive(stale, "web", args("web"), call)
    assert mismatch.value.status_code == 409
    assert attempt.db.get_quick_ingest_source("plan", "source").dispatch_state == "reserved"
    call.assert_not_awaited()


@pytest.mark.asyncio
async def test_changed_attempt_between_lookup_and_begin_does_not_replay_an_unrelated_result(attempt, monkeypatch):
    def switch_attempt(plan_key, source_key, **_kwargs):
        row = attempt.db.get_quick_ingest_source(plan_key, source_key)
        return False, replace(row, retry_count=1, idempotency_key="new-attempt")

    monkeypatch.setattr(attempt.db, "begin_quick_ingest_dispatch", switch_attempt)
    with pytest.raises(HTTPException) as stale:
        await dispatch.receive(attempt, "web", args("web"), AsyncMock())
    assert stale.value.status_code == 409


def test_planned_routes_reuse_every_existing_transport_permission_and_validated_input():
    from fastapi import FastAPI
    from fastapi.routing import APIRoute

    app = FastAPI()
    app.include_router(dispatch.router, prefix="/api/v1/media")
    paths = app.openapi()["paths"]
    for transport, module, path in [
        ("web", dispatch.process_web_scraping, "/process-web-scraping"),
        ("jobs", dispatch.ingest_jobs, "/ingest/jobs"),
        ("add", dispatch.add, "/add"),
    ]:
        original = next(route for route in module.router.routes if isinstance(route, APIRoute) and route.path == path)
        planned = next(route for route in dispatch.router.routes if route.path.endswith("/" + transport))
        assert planned.dependencies == original.dependencies
        operation = paths["/api/v1/media" + planned.path]["post"]
        assert {(p["name"], p["in"]) for p in operation["parameters"]} >= {
            ("plan_key", "path"),
            ("source_key", "path"),
            ("retry_count", "query"),
            ("idempotency_key", "query"),
        }
        assert (
            "application/json" in operation["requestBody"]["content"]
            if transport == "web"
            else ("multipart/form-data" in operation["requestBody"]["content"])
        )


@pytest.fixture
def plan_client(plan_store, tmp_path, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from tldw_Server_API.app.api.v1.API_Deps.Collections_DB_Deps import (
        get_collections_db_for_user,
        try_get_collections_db_for_user,
    )

    monkeypatch.setenv("TEST_MODE", "true")
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("SINGLE_USER_API_KEY", "test-api-key-12345")
    monkeypatch.setenv("JOBS_DB_PATH", str(tmp_path / "jobs.db"))
    monkeypatch.delenv("JOBS_DB_URL", raising=False)
    app = FastAPI()
    app.include_router(dispatch.router, prefix="/api/v1/media")
    app.include_router(dispatch.ingest_jobs.router, prefix="/api/v1/media")
    app.dependency_overrides[get_collections_db_for_user] = lambda: plan_store
    app.dependency_overrides[try_get_collections_db_for_user] = lambda: plan_store
    with TestClient(app, headers={"X-API-KEY": "test-api-key-12345"}) as client:
        yield plan_store, client


def _reserve(store, source_url, source_kind):
    _, rows = store.put_quick_ingest_plan(
        "plan",
        {"name": "Plan", "kind": "conference"},
        [
            ("source", {"source_kind": source_kind, "source_url": source_url}),
        ],
    )
    assert store.claim_quick_ingest_source("plan", "source")[0]
    return {"retry_count": 0, "idempotency_key": rows["source"].idempotency_key}


_PATH = "/api/v1/media/collections/quick-ingest-plans/plan/sources/source/dispatch"


def test_web_transport_publishes_real_plan_receipt_before_a_lost_http_ack(plan_client, monkeypatch):
    """A second client can inspect the in-flight save and recover its exact receipt."""
    import asyncio
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    store, client = plan_client
    url = "https://example.test/plan-source"
    query = _reserve(store, url, "url")
    entered, finish = Event(), Event()

    async def scrape_once(**_options):
        entered.set()
        assert await asyncio.to_thread(finish.wait, 6)
        return {"status": "persist-ok", "media_ids": [42]}

    scrape = AsyncMock(side_effect=scrape_once)
    monkeypatch.setattr(WebScrapingService, "process_web_scraping_task", scrape)
    body = {
        "url_input": url,
        "scrape_method": "Individual URLs",
        "mode": "persist",
        "summarize_checkbox": False,
        "keywords": "telescope, observatory",
    }
    with ThreadPoolExecutor(max_workers=1) as threads:
        first = threads.submit(client.post, _PATH + "/web", params=query, json=body)
        try:
            assert entered.wait(6)
            in_flight = client.post(_PATH + "/web", params=query, json=body)
            assert in_flight.status_code == 200, in_flight.text
            assert in_flight.json()["dispatch_state"] == "web" and not in_flight.json()["dispatched"]
        finally:
            finish.set()
        assert first.result(timeout=6).status_code == 200

    recovered = client.post(_PATH + "/web", params=query, json=body)
    assert recovered.status_code == 200, recovered.text
    assert not recovered.json()["dispatched"]
    assert recovered.json()["item"]["media_id"] == store.get_quick_ingest_source("plan", "source").media_id == 42
    assert recovered.json()["dispatch_state"] == "settled"
    scrape.assert_awaited_once()


@pytest.mark.parametrize("quota_rejected", [False, True])
def test_native_dispatch_keeps_the_original_job_or_fallback_media(plan_client, monkeypatch, quota_rejected):
    """Replay after a lost native acknowledgement returns its existing Job or saved Media."""
    from pathlib import Path

    from tldw_Server_API.app.core.Jobs.manager import JobManager

    store, client = plan_client
    query = _reserve(store, "local://plan.txt", "file")
    data = {"media_type": "document", "perform_analysis": "false", "generate_embeddings": "false"}
    files = {"files": ("plan.txt", b"Receiver-bound plan document.", "text/plain")}

    def post(kind):
        return client.post(_PATH + "/" + kind, params=query, data=data, files=files)

    if quota_rejected:
        from tldw_Server_API.app.services.storage_quota_service import StorageQuotaService

        # The isolated transport fixture has no account quota ledger.
        monkeypatch.setattr(StorageQuotaService, "check_quota", AsyncMock(return_value=(True, {})))

        monkeypatch.setenv("JOBS_MAX_PER_USER", "1")
        JobManager().create_job(
            domain="media_ingest", queue="default", job_type="ingest", payload={}, owner_user_id="1"
        )
        denied = post("jobs")
        assert denied.status_code == 429, denied.text
        assert denied.json()["detail"]["code"] == "plan_jobs_quota_rejected"
        assert store.get_quick_ingest_source("plan", "source").dispatch_state == "fallback_ready"
        saved = post("add")
        assert saved.status_code == 200, saved.text
        media_id = saved.json()["result"]["results"][0]["db_id"]
        assert media_id > 0 and saved.json()["item"]["media_id"] == media_id
        recovered = post("add").json()
        assert not recovered["dispatched"] and recovered["item"]["media_id"] == media_id
    else:
        accepted = post("jobs")
        assert accepted.status_code == 200, accepted.text
        job_id = accepted.json()["result"]["jobs"][0]["id"]
        row = JobManager().get_job(job_id)
        payload = row["payload"]
        source = Path(payload["source"])
        try:
            assert source.read_bytes() == b"Receiver-bound plan document."
            assert payload["idempotency_key"] == query["idempotency_key"]
            recovered = post("jobs").json()
            assert not recovered["dispatched"] and recovered["item"]["latest_job_id"] == str(job_id)
            assert not post("add").json()["dispatched"]
        finally:
            source.unlink(missing_ok=True)
            source.parent.rmdir()


@pytest.mark.parametrize("binding", ["item", "key"])
def test_legacy_jobs_cannot_publish_for_a_reserved_plan_source(plan_client, binding, monkeypatch):
    """Both old-client identities are refused before multipart staging or Jobs admission."""
    store, client = plan_client
    query = _reserve(store, "local://plan.txt", "file")
    item = store.get_quick_ingest_source("plan", "source")
    data = {"media_type": "document"}
    if binding == "item":
        data["media_collection_item_id"] = str(item.id)
    else:
        data["idempotency_key"] = query["idempotency_key"]
    files = {"files": ("plan.txt", b"reserved", "text/plain")}
    with monkeypatch.context() as patch:

        async def no_staging(*args, **kwargs):
            pytest.fail("The legacy route reached the upload staging boundary")

        patch.setattr(input_sourcing, "save_uploaded_files", no_staging)
        resp = client.post("/api/v1/media/ingest/jobs", data=data, files=files)
    assert resp.status_code == 409, resp.text
    assert store.get_quick_ingest_source("plan", "source").dispatch_state == "reserved"


@pytest.mark.asyncio
async def test_late_quota_error_cannot_publish_fallback_for_a_new_attempt(attempt, monkeypatch):
    from tldw_Server_API.app.core.exceptions import BadRequestError

    def stale_finish(*_args, **_kwargs):
        raise ValueError("attempt changed")

    monkeypatch.setattr(attempt.db, "finish_quick_ingest_dispatch", stale_finish)
    api_error = HTTPException(429, "Concurrent job limit exceeded")
    api_error.__cause__ = BadRequestError("Concurrent job limit exceeded")
    with pytest.raises(HTTPException) as stale:
        await dispatch.receive(attempt, "jobs", args("jobs"), AsyncMock(side_effect=api_error))
    assert stale.value.status_code == 409


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reply,status,media_id",
    [
        (
            {"status": "duplicate", "media_ids": [], "stored_articles": 0, "duplicate_articles": 1},
            "skipped_existing",
            None,
        ),
        (
            {"status": "duplicate", "media_ids": [], "media_id": 23, "stored_articles": 0, "duplicate_articles": 1},
            "skipped_existing",
            23,
        ),
        (
            {
                "status": "persist-ok",
                "stored_articles": 0,
                "total_articles": 1,
                "extraction_failures": [{"code": "source_access_denied"}],
            },
            "failed",
            None,
        ),
    ],
)
async def test_direct_scrape_retains_confirmed_duplicate_or_presave_failure(attempt, reply, status, media_id):
    result = await dispatch.receive(attempt, "web", args("web"), AsyncMock(return_value=reply))
    assert result["item"].status == status and result["item"].media_id == media_id
    assert result["result"] == reply


def test_empty_native_file_can_retry_the_same_source_after_validation(plan_client):
    """The receiver knows no Job was published and the corrected bytes may start a fresh attempt."""
    from pathlib import Path

    from tldw_Server_API.app.core.Jobs.manager import JobManager

    store, client = plan_client
    query = _reserve(store, "local://plan.txt", "file")

    def submit(params, content):
        return client.post(
            _PATH + "/jobs",
            params=params,
            data={"media_type": "document"},
            files={"files": ("plan.txt", content, "text/plain")},
        )

    denied = submit(query, b"")
    assert denied.status_code == 200, denied.text
    assert denied.json()["item"]["status"] == "failed"
    assert "empty" in denied.json()["item"]["error_summary"].lower()
    assert not JobManager().list_jobs(domain="media_ingest")
    won, _ = store.claim_quick_ingest_source("plan", "source", retry_count=1, idempotency_key="next-file")
    assert won
    fixed = submit({"retry_count": 1, "idempotency_key": "next-file"}, b"corrected content")
    assert fixed.status_code == 200, fixed.text
    source = Path(fixed.json()["result"]["jobs"][0]["source"])
    try:
        assert source.read_bytes() == b"corrected content"
    finally:
        source.unlink(missing_ok=True)
        source.parent.rmdir()


def _exhaust_analysis_quota(monkeypatch, retry_after: int) -> None:
    """Give this user a spent token budget and a known reset interval."""

    async def user_limit(user_id, key):
        return 20 if user_id == 1 and key == quota_checks.LLM_TOKENS_PER_MONTH_KEY else None

    monkeypatch.setattr(quota_checks, "user_quota", user_limit)
    monkeypatch.setattr(quota_checks, "llm_tokens_this_month", AsyncMock(return_value=20))
    monkeypatch.setattr(quota_checks, "seconds_until_utc_month_start", lambda: retry_after)


def test_web_summary_quota_denied_before_fetch_permits_explicit_retry(plan_client, monkeypatch):
    store, client = plan_client
    url = "https://example.test/plan-source"
    query = _reserve(store, url, "url")
    _exhaust_analysis_quota(monkeypatch, retry_after=60)
    scrape = AsyncMock(return_value={"status": "persist-ok", "media_ids": [22]})
    monkeypatch.setattr(WebScrapingService, "process_web_scraping_task", scrape)
    body = {"url_input": url, "scrape_method": "Individual URLs", "mode": "persist", "summarize_checkbox": True}
    denied = client.post(_PATH + "/web", params=query, json=body)
    assert denied.status_code == 402 and denied.headers["Retry-After"] == "60"
    assert denied.json()["detail"]["category"] == "llm_tokens_month"
    scrape.assert_not_awaited()
    item = store.get_quick_ingest_source("plan", "source")
    assert item.status == "failed" and "quota" in item.error_summary.lower()
    won, _ = store.claim_quick_ingest_source("plan", "source", retry_count=1, idempotency_key="next-web")
    assert won
    corrected = client.post(
        _PATH + "/web",
        params={"retry_count": 1, "idempotency_key": "next-web"},
        json={**body, "summarize_checkbox": False},
    )
    assert corrected.status_code == 200 and corrected.json()["item"]["media_id"] == 22


def test_empty_crawl_can_retry_the_original_source_without_repeating_a_write(plan_client, monkeypatch):
    from tldw_Server_API.app.core.DB_Management.db_path_utils import get_user_media_db_path
    from tldw_Server_API.app.core.DB_Management.media_db.api import managed_media_database
    from tldw_Server_API.app.core.Web_Scraping.contracts.execution import CrawlOutcome, CrawlReport
    from tldw_Server_API.app.services import web_scraping_service

    store, client = plan_client
    url = "https://example.test/lecture"
    query = _reserve(store, url, "url")
    item = store.get_quick_ingest_source("plan", "source")
    body = {"url_input": url, "scrape_method": "Individual URLs", "mode": "persist", "perform_chunking": False}
    crawl = AsyncMock(
        side_effect=[
            CrawlReport((), CrawlOutcome("failed", "queue_overloaded", False)),
            CrawlReport(
                ({"url": url, "title": "Lecture", "content": "Lecture notes", "extraction_successful": True},),
                CrawlOutcome("completed", "frontier_exhausted", False),
            ),
        ]
    )
    service = WebScrapingService()
    service._initialized = True
    service.scraper = SimpleNamespace(_batch_report=crawl, _set_progress=Mock())
    persist = AsyncMock(wraps=service._store_persistent)
    monkeypatch.setattr(service, "_store_persistent", persist)
    monkeypatch.setattr(web_scraping_service, "get_web_scraping_service", lambda: service)

    denied = client.post(_PATH + "/web", params=query, json=body)
    assert denied.status_code == 503
    assert denied.json()["detail"] == {
        "message": "Web crawl failed",
        "crawl_outcome": {"state": "failed", "stop_reason": "queue_overloaded", "has_partial_results": False},
    }
    persist.assert_not_awaited()
    replay = client.post(_PATH + "/web", params=query, json=body).json()
    assert not replay["dispatched"] and replay["dispatch_state"] == "settled"
    assert replay["item"]["status"] == "failed" and replay["item"]["media_id"] is None
    crawl.assert_awaited_once()

    retry = {"retry_count": 1, "idempotency_key": "lecture-retry"}
    assert store.claim_quick_ingest_source("plan", "source", **retry)[0]
    saved = client.post(_PATH + "/web", params=retry, json=body)
    assert saved.status_code == 200, saved.text
    receipt = saved.json()["item"]
    assert (receipt["collection_id"], receipt["id"]) == (item.collection_id, item.id)
    (media_id,) = saved.json()["result"]["media_ids"]
    assert receipt["media_id"] == media_id
    with managed_media_database(client_id="1", db_path=get_user_media_db_path(1), existing_only=True) as db:
        media = db.get_media_by_id(media_id)
        assert media["url"] == url and "Lecture notes" in media["content"]
    assert not client.post(_PATH + "/web", params=retry, json=body).json()["dispatched"]
    assert crawl.await_count == 2
    persist.assert_awaited_once()


@pytest.mark.asyncio
async def test_unconfirmed_web_error_cannot_release_a_claimed_source(attempt):
    error = HTTPException(
        503,
        {
            "message": "Web crawl failed",
            "crawl_outcome": {"state": "failed", "stop_reason": "queue_overloaded", "has_partial_results": False},
        },
    )
    operation = AsyncMock(side_effect=error)
    with pytest.raises(HTTPException) as denied:
        await dispatch.receive(attempt, "web", args("web"), operation)
    assert denied.value is error
    replay = await dispatch.receive(attempt, "web", args("web"), operation)
    assert not replay["dispatched"] and replay["dispatch_state"] == "web"
    assert not attempt.db.claim_quick_ingest_source("plan", "source", retry_count=1, idempotency_key="next")[0]
    operation.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stopped_during_save,reported,expected_state",
    [(False, True, "settled"), (False, False, "web"), (True, True, "web")],
)
async def test_confirmed_crawl_stop_only_releases_the_source_before_media_storage(
    attempt, monkeypatch, stopped_during_save, reported, expected_state
):
    import asyncio

    from tldw_Server_API.app.core.Web_Scraping.contracts.execution import CrawlOutcome, CrawlReport

    stop = asyncio.CancelledError("crawl stopped")
    if reported:
        stop.report = CrawlReport.interrupted([], "cancelled")
    article = {"url": "https://example.test/a", "content": "Lecture notes"}
    completed = CrawlReport((article,), CrawlOutcome("completed", "frontier_exhausted", False))
    crawl = AsyncMock(return_value=completed, side_effect=None if stopped_during_save else stop)
    service = WebScrapingService()
    service._initialized = True
    service.scraper = SimpleNamespace(_batch_report=crawl, _set_progress=Mock(), _progress={})
    persist = AsyncMock(side_effect=stop)
    monkeypatch.setattr(service, "_store_persistent", persist)

    async def submit(*, payload):
        return await service.process_web_scraping_task(
            scrape_method="Individual URLs",
            url_input=payload.url_input,
            user_id=1,
        )

    with pytest.raises(asyncio.CancelledError) as stopped:
        await dispatch.receive(attempt, "web", args("web"), submit)
    assert stopped.value is stop
    replay = await dispatch.receive(attempt, "web", args("web"), submit)
    assert not replay["dispatched"] and replay["dispatch_state"] == expected_state
    if stopped_during_save:
        persist.assert_awaited_once()
    else:
        persist.assert_not_awaited()
    assert replay["item"].status == ("cancelled" if expected_state == "settled" else "processing")
    claimed, _ = attempt.db.claim_quick_ingest_source("plan", "source", retry_count=1, idempotency_key="next")
    assert claimed == (expected_state == "settled")


@pytest.mark.parametrize("during_save", [False, True], ids=["before-media-save", "during-media-save"])
def test_web_summary_policy_refusal_settles_only_a_confirmed_unwritten_source(plan_client, monkeypatch, during_save):
    from tldw_Server_API.app.core.exceptions import LLMPolicyRefusal
    from tldw_Server_API.app.core.Web_Scraping.contracts.execution import CrawlOutcome, CrawlReport
    from tldw_Server_API.app.services import web_scraping_service

    store, client = plan_client
    url = "https://example.test/lecture"
    query = _reserve(store, url, "url")
    article = {"url": url, "title": "Lecture", "content": "Lecture notes", "extraction_successful": True}
    crawl = AsyncMock(return_value=CrawlReport((article,), CrawlOutcome("completed", "frontier_exhausted", False)))
    refusal = LLMPolicyRefusal("Input violates moderation policy. " * 35)
    refusal.headers = {"X-Policy-Decision": "refused"}
    summarize = AsyncMock(return_value="Summary", side_effect=None if during_save else refusal)
    persist = AsyncMock(side_effect=refusal)
    service = WebScrapingService()
    service._initialized = True
    service.scraper = SimpleNamespace(_batch_report=crawl, _summarize_content=summarize, _set_progress=Mock())
    monkeypatch.setattr(service, "_store_persistent", persist)
    monkeypatch.setattr(web_scraping_service, "get_web_scraping_service", lambda: service)
    body = {"url_input": url, "scrape_method": "Individual URLs", "mode": "persist", "summarize_checkbox": True}

    denied = client.post(_PATH + "/web", params=query, json=body)
    assert denied.status_code == 400 and denied.json()["detail"] == refusal.detail
    assert denied.headers["X-Policy-Decision"] == "refused"
    replay = client.post(_PATH + "/web", params=query, json=body).json()
    assert not replay["dispatched"]
    assert replay["dispatch_state"] == ("web" if during_save else "settled")
    assert replay["item"]["status"] == ("processing" if during_save else "failed")
    assert replay["item"]["media_id"] is None
    if during_save:
        persist.assert_awaited_once()
    else:
        persist.assert_not_awaited()
        assert replay["item"]["error_summary"] == refusal.detail[:1000]
    summarize.assert_awaited_once()
    crawl.assert_awaited_once()


def test_fallback_analysis_quota_before_save_projects_retryable_failed_receipt(plan_client, monkeypatch):
    from tldw_Server_API.app.services.storage_quota_service import StorageQuotaService

    store, client = plan_client
    query = _reserve(store, "local://plan.txt", "file")
    won, _ = store.begin_quick_ingest_dispatch("plan", "source", transport="jobs", **query)
    assert won
    store.finish_quick_ingest_dispatch("plan", "source", transport="jobs", status="fallback_ready", **query)
    monkeypatch.setattr(StorageQuotaService, "check_quota", AsyncMock(return_value=(True, {})))
    _exhaust_analysis_quota(monkeypatch, retry_after=80)
    response = client.post(
        _PATH + "/add",
        params=query,
        data={"media_type": "document", "perform_analysis": "true", "api_name": "OpenAI", "overwrite_existing": "true"},
        files={"files": ("plan.txt", b"would have been ingested", "text/plain")},
    )
    assert response.status_code == 402 and response.headers["Retry-After"] == "80"
    assert response.json()["detail"]["category"] == "llm_tokens_month"
    item = store.get_quick_ingest_source("plan", "source")
    assert item.status == "failed" and "quota" in item.error_summary.lower()
