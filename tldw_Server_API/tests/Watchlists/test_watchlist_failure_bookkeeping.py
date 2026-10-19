"""Failed extraction must not become saved content or successful monitoring."""

import json

import httpx
import pytest

from tldw_Server_API.app.core.DB_Management.Collections_DB import CollectionsDatabase
from tldw_Server_API.app.core.DB_Management.Watchlists_DB import WatchlistsDatabase
from tldw_Server_API.app.core.Watchlists import fetchers, pipeline
from tldw_Server_API.app.core.Watchlists.fetchers import fetch_site_article
from tldw_Server_API.app.core.Watchlists.pipeline import run_watchlist_job
from tldw_Server_API.app.core.Web_Scraping import orchestration
from tldw_Server_API.app.core.Web_Scraping.orchestration.article_models import article_failure_result

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "result",
    [
        article_failure_result("extraction_error"),
        {"title": "Denied", "content": "denial detail", "extraction_successful": False},
        {"title": "Empty", "content": "   "},
    ],
)
def test_article_failure_does_not_become_an_ingestable_article(monkeypatch, result):
    monkeypatch.setattr(orchestration, "scrape_article_blocking", lambda _url: result)
    assert fetch_site_article("https://example.test/denied") is None  # nosec B101 - Pytest failure-state regression assertion.


def test_article_without_legacy_success_flag_retains_real_content(monkeypatch):
    monkeypatch.setattr(
        orchestration,
        "scrape_article_blocking",
        lambda _url: {"title": "Rowan", "author": "Observer", "content": "The Selene telescope is ready."},
    )
    assert fetch_site_article("https://example.test/valid") == {  # nosec B101 - Pytest failure-state regression assertion.
        "title": "Rowan",
        "author": "Observer",
        "content": "The Selene telescope is ready.",
        "url": "https://example.test/valid",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("include_success", [False, True])
@pytest.mark.parametrize("source_mode", ["site", "rss", "rules", "skip_fetch"])
async def test_failed_source_preserves_good_content_and_recovers_without_duplicate_ingestion(
    monkeypatch, tmp_path, include_success, source_mode
):
    monkeypatch.setenv("TEST_MODE", "0")
    monkeypatch.setenv("TLDW_TEST_MODE", "0")
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path / "users"))
    monkeypatch.setenv("JOBS_DB_URL", "sqlite:///" + str(tmp_path / "jobs.db"))
    monkeypatch.setenv("MONITORING_NOTIFY_ENABLED", "false")
    user_id = 9701
    db = WatchlistsDatabase.for_user(user_id)
    failed = db.create_source(
        name="Denied",
        url="https://example.test/denied",
        source_type="rss" if source_mode == "rss" else "site",
        active=True,
        settings_json=None,
        tags=[],
        group_ids=[],
    )
    sources = [failed]
    if include_success:
        sources.append(
            db.create_source(
                name="Good",
                url="https://example.test/good",
                source_type="rss" if source_mode == "rss" else "site",
                active=True,
                settings_json=None,
                tags=[],
                group_ids=[],
            )
        )
    if source_mode != "site":
        for source in sources:
            settings = {}
            if source_mode in {"rules", "skip_fetch"}:
                settings = {"scrape_rules": {"list_url": source.url, "skip_article_fetch": source_mode == "skip_fetch"}}
            db.update_source(source.id, {"settings_json": json.dumps(settings)})
    job = db.create_job(
        name="Recovery",
        description=None,
        scope_json=json.dumps({"sources": [source.id for source in sources]}),
        schedule_expr=None,
        schedule_timezone="UTC",
        active=True,
        max_concurrency=None,
        per_host_delay_ms=None,
        retry_policy_json=None,
        output_prefs_json=json.dumps({"briefing_pipeline": {"text": {"enabled": True}}}),
    )
    failed_now = True

    def external_article(url):
        if failed_now and url.endswith("/denied"):
            return article_failure_result("extraction_error")
        return {
            "title": url.rsplit("/", 1)[-1],
            "content": f"The Selene telescope is ready for {url.rsplit('/', 1)[-1]}.",
            "extraction_successful": True,
        }

    async def external_feed(url, **kwargs):
        if kwargs.get("etag") == '"v1"':
            return {"status": 304, "items": []}
        return {"status": 200, "etag": '"v1"', "items": [{"url": url, "title": "Rowan", "summary": "Feed description"}]}

    async def external_rules(base_url, **kwargs):
        content = "" if failed_now and base_url.endswith("/denied") else external_article(base_url)["content"]
        return [{"url": base_url, "title": "Rowan", "content": content}]

    monkeypatch.setattr(pipeline, "fetch_rss_feed_history", external_feed)
    monkeypatch.setattr(pipeline, "fetch_site_items_with_rules", external_rules)
    monkeypatch.setattr(orchestration, "scrape_article_blocking", external_article)
    first = await run_watchlist_job(user_id, job.id)
    run = db.get_run(first["run_id"])
    items, _ = db.list_items(run_id=run.id, job_id=job.id, status=None, limit=100, offset=0)
    assert {item.source_id: item.status for item in items} == {  # nosec B101 - Pytest failure-state regression assertion.
        source.id: "error" if source.id == failed.id else "ingested" for source in sources
    }
    assert run.status == "failed"  # nosec B101 - Pytest failure-state regression assertion.
    with pytest.raises(KeyError):
        db.get_briefing_occurrence_for_run(run.id)
    stats = json.loads(run.stats_json)
    assert stats["items_ingested"] == int(include_success)  # nosec B101 - Pytest failure-state regression assertion.
    assert stats["source_errors"] == 1  # nosec B101 - Pytest failure-state regression assertion.
    assert db.get_source(failed.id).consec_errors == 1  # nosec B101 - Pytest failure-state regression assertion.
    assert not db.has_seen_item(failed.id, "https://example.test/denied")  # nosec B101 - Pytest failure-state regression assertion.
    collection = CollectionsDatabase.for_user(user_id)
    _, total = collection.list_content_items(page=1, size=10)
    assert total == int(include_success)  # nosec B101 - Pytest failure-state regression assertion.

    failed_now = False
    second = await run_watchlist_job(user_id, job.id)
    recovered = db.get_run(second["run_id"])
    assert recovered.status == "succeeded"  # nosec B101 - Pytest failure-state regression assertion.
    assert db.get_briefing_occurrence_for_run(recovered.id).artifact_status == "ready"  # nosec B101 - Pytest failure-state regression assertion.
    assert json.loads(recovered.stats_json)["items_ingested"] == 1  # nosec B101 - Pytest failure-state regression assertion.
    assert db.get_source(failed.id).status == "ok"  # nosec B101 - Pytest failure-state regression assertion.
    _, total = collection.list_content_items(page=1, size=10)
    assert total == len(sources)  # nosec B101 - Pytest failure-state regression assertion.
    if source_mode == "rss":
        assert db.get_source(failed.id).etag == '"v1"'  # nosec B101 - Pytest successful recovery validator assertion.
        unchanged = await run_watchlist_job(user_id, job.id)
        assert db.get_run(unchanged["run_id"]).status == "succeeded" and unchanged["items_ingested"] == 0  # nosec B101 - Pytest unchanged RSS regression assertion.
        assert {entry["status"] for entry in unchanged["source_statuses"]} == {"not_modified"}  # nosec B101 - Pytest unchanged RSS regression assertion.


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("source_type", "failure"), [("rss", "cancelled"), ("rss", "persistence"), ("site", "persistence")]
)
async def test_unfinished_source_is_not_successful_or_cached(monkeypatch, tmp_path, source_type, failure):
    monkeypatch.setenv("TEST_MODE", "0")
    monkeypatch.setenv("TLDW_TEST_MODE", "0")
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path / "users"))
    monkeypatch.setenv("JOBS_DB_URL", "sqlite:///" + str(tmp_path / "jobs.db"))
    monkeypatch.setenv("MONITORING_NOTIFY_ENABLED", "false")
    db = WatchlistsDatabase.for_user(9702)
    source = db.create_source(
        name="RSS",
        url="https://example.test/feed",
        source_type=source_type,
        active=True,
        settings_json=None,
        tags=[],
        group_ids=[],
    )
    job = db.create_job(
        name="RSS recovery",
        description=None,
        scope_json=json.dumps({"sources": [source.id]}),
        schedule_expr=None,
        schedule_timezone="UTC",
        active=True,
        max_concurrency=None,
        per_host_delay_ms=None,
        retry_policy_json=None,
        output_prefs_json=None,
    )

    async def feed(_url, **kwargs):
        return {
            "status": 200,
            "etag": '"new"',
            "last_modified": "new-date",
            "items": [
                {"url": "https://example.test/article", "title": "Rowan"},
            ],
        }

    def article(_url):
        if failure == "cancelled":
            runs, _ = db.list_runs_for_job(job.id, limit=10, offset=0)
            db.update_run(runs[0].id, status="cancelled")
        return {"title": "Rowan", "content": "The Selene telescope is ready."}

    monkeypatch.setattr(pipeline, "fetch_rss_feed_history", feed)
    monkeypatch.setattr(orchestration, "scrape_article_blocking", article)
    if failure == "persistence":
        monkeypatch.setattr(CollectionsDatabase, "upsert_content_item", lambda *args, **kwargs: None)
    result = await run_watchlist_job(9702, job.id)
    assert db.get_run(result["run_id"]).status == ("cancelled" if failure == "cancelled" else "failed")  # nosec B101 - Pytest incomplete RSS regression assertion.
    source = db.get_source(source.id)
    assert source.etag is None and source.last_modified is None  # nosec B101 - Pytest incomplete RSS regression assertion.


@pytest.mark.asyncio
@pytest.mark.parametrize("history", [False, True])
@pytest.mark.parametrize("invalid", [False, True])
async def test_invalid_feed_is_distinct_from_valid_empty_feed(monkeypatch, history, invalid):
    from tldw_Server_API.app.core import http_client

    async def external_response(**kwargs):
        return httpx.Response(
            200,
            text="<rss><broken>" if invalid else "<rss><channel/></rss>",
            headers={"ETag": '"feed"'},
            request=httpx.Request("GET", kwargs["url"]),
        )

    monkeypatch.setattr(http_client, "afetch", external_response)
    monkeypatch.setattr(fetchers, "is_url_allowed_for_tenant", lambda *_: True)
    fetch = fetchers.fetch_rss_feed_history if history else fetchers.fetch_rss_feed
    result = await fetch("https://example.test/feed")
    assert result["items"] == []  # nosec B101 - Pytest feed parsing regression assertion.
    if invalid:
        assert result["status"] == 502  # nosec B101 - Pytest feed parsing regression assertion.
        assert result["error"] == "Invalid RSS/Atom XML"  # nosec B101 - Pytest feed parsing regression assertion.
        assert not result.get("etag")  # nosec B101 - Invalid input must not poison cache validators.
    else:
        assert result["status"] == 200  # nosec B101 - Empty valid feed is successful, not a parse failure.
        assert result["etag"] == '"feed"'  # nosec B101 - Valid feed caching contract.
