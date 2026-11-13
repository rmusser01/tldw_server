"""Public snapshots exclude execution data without scrubbing article prose."""

import json
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.Web_Scraping.jobs import projections
from tldw_Server_API.app.core.Web_Scraping.jobs.models import ScrapingJob
from tldw_Server_API.app.services.enhanced_web_scraping_service import WebScrapingService

pytestmark = pytest.mark.unit
URL = "https://user:sentinel-pass@example.com/article?token=sentinel-token#sentinel-fragment"


def private_job(job_type=ScrapingJob):
    return job_type(
        "job",
        URL,
        "auto",
        user_id="7",
        metadata={"options": {"custom_headers": {"Authorization": "sentinel-auth"}, "cookie": "sentinel-cookie"}},
        error="sentinel-auth",
        result={
            "url": URL,
            "content": "Public article text",
            "raw_html": "<p>Requested public raw content</p>",
            "_discovery_html": "sentinel-discovery",
            "discovery": {"html": "sentinel-discovery"},
            "metadata": {
                "labels": ["public"],
                "nested": {
                    "custom_headers": {"X-Whatever": "sentinel-header"},
                    "link": URL,
                    "options": {"arbitrary": "sentinel-option"},
                    "error": "sentinel-auth",
                },
            },
            "preflight_analysis": {"nested": {"Cookie": "sentinel-cookie"}, "url": URL},
        },
    )


def test_nested_execution_secrets_never_project():
    job = private_job()
    public = projections.public_job(job)
    serialized = json.dumps(public)
    assert "sentinel-" not in serialized
    assert public["url"] == "https://example.com/article"
    assert public["metadata"] == {}
    assert public["result"]["metadata"]["labels"] == ["public"]
    assert public["result"]["metadata"]["nested"]["error"] == "execution_failed"
    assert job.result["url"] == URL


def test_intentionally_public_article_raw_content_is_not_secret_scanned():
    job = private_job()
    job.result["content"] = "Article discussing Cookie and Authorization: sentinel-public"
    public = projections.public_job(job)
    assert public["result"]["content"] == job.result["content"]
    assert public["result"]["raw_html"] == "<p>Requested public raw content</p>"


def test_progress_is_allowlisted_and_detached_not_an_execution_payload():
    source = {
        "status": "in_progress",
        "processed_urls": 2,
        "total_urls": 5,
        "current_url": URL,
        "owner_user_id": "7",
        "options": {"arbitrary": "sentinel-option"},
        "_discovery_html": "sentinel-discovery",
        "articles": [private_job().result],
        "error": "sentinel-auth",
        "remaining": 3,
    }
    public = projections.public_progress(source)
    assert public["processed_urls"] == 2
    assert public["current_url"] == "https://example.com/article"
    assert public["remaining"] == 3
    assert "sentinel-" not in json.dumps(public)
    assert "articles" not in public and "owner_user_id" not in public
    public["processed_urls"] = 9
    assert source["processed_urls"] == 2


@pytest.mark.parametrize("legacy", [False, True])
def test_history_redacts_nested_urls_and_discovery_but_keeps_public_content(legacy):
    job_type = ScrapingJob
    if legacy:
        from tldw_Server_API.app.core.Web_Scraping.enhanced_web_scraping import ScrapingJob as LegacyJob
        from tldw_Server_API.app.core.Web_Scraping.enhanced_web_scraping import ScrapingJobQueue

        job_type = LegacyJob
    job = private_job(job_type)
    if legacy:
        queue = ScrapingJobQueue()
        queue._record_completed_job(job)
        retained = queue.get_job("job")
    else:
        history = projections.History(1, 10000)
        history.record(job, job.result)
        retained = history.get("job")
    assert "sentinel-" not in json.dumps(retained.to_dict())
    assert retained.result["content"] == "Public article text"
    assert retained.result["raw_html"] == "<p>Requested public raw content</p>"
    assert job.result["_discovery_html"] == "sentinel-discovery"


@pytest.mark.asyncio
@pytest.mark.parametrize("legacy", [False, True])
async def test_service_status_projects_both_legacy_and_new_jobs(legacy):
    job_type = ScrapingJob
    if legacy:
        from tldw_Server_API.app.core.Web_Scraping.enhanced_web_scraping import ScrapingJob as LegacyJob

        job_type = LegacyJob
    svc = WebScrapingService()
    svc._initialized = True
    svc.scraper = SimpleNamespace(job_queue=SimpleNamespace(get_job=lambda _: private_job(job_type)))
    public = await svc.get_job_status("job", SimpleNamespace(id=7))
    assert "sentinel-" not in json.dumps(public)
    assert public["result"]["content"] == "Public article text"


def test_crawl_outcome_projection_retains_terminal_classification():
    from tldw_Server_API.app.core.Web_Scraping.contracts.execution import CrawlOutcome

    public = projections.public_job(private_job(), crawl_outcome=CrawlOutcome("partial", "budget_exhausted", True))
    assert public["crawl_outcome"] == {
        "state": "partial",
        "stop_reason": "budget_exhausted",
        "has_partial_results": True,
    }


@pytest.mark.asyncio
async def test_progress_endpoint_projects_after_owner_lookup(monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints import web_scraping as endpoints

    def progress(task_id, *, owner_user_id):
        assert (task_id, owner_user_id) == ("task", "7")
        return {"current_url": URL, "processed_urls": 2, "options": {"cookie": "sentinel-cookie"}}

    svc = SimpleNamespace(_initialized=True, scraper=SimpleNamespace(get_progress=progress))
    monkeypatch.setattr(endpoints, "get_web_scraping_service", lambda: svc)
    result = await endpoints.get_scraping_progress("task", SimpleNamespace(id=7))
    assert result["progress"] == {"current_url": "https://example.com/article", "processed_urls": 2}


def test_recursive_progress_preserves_existing_counts():
    snapshot = {"pages_scraped": 2, "total_pages": 10, "queue_size": 3, "visited": 4}
    assert projections.public_progress(snapshot) == snapshot


def test_option_header_aliases_and_schemeless_url_credentials_are_private():
    job = private_job()
    job.result["metadata"] = {
        "request_options": {"arbitrary": "sentinel-option"},
        "request_headers": {"X-Arbitrary": "sentinel-header"},
        "link": "//user:sentinel-password@example.com/path?key=sentinel-query",
    }
    assert "sentinel-" not in json.dumps(projections.public_job(job))


@pytest.mark.parametrize("mode", ["polling", "history", "legacy_history"])
@pytest.mark.parametrize("key", ["content", "raw_html", "raw_content"])
@pytest.mark.parametrize("container", ["direct", "nested", "list"])
def test_prose_exemption_applies_only_at_article_boundary(mode, key, container):
    def wrapped(value):
        leaf = {key: value}
        if container == "nested":
            return {"result": leaf}
        if container == "list":
            return {"items": [leaf]}
        return leaf

    job = private_job()
    prose = {"content": URL, "raw_html": f"<p>{URL}</p>", "raw_content": URL}
    job.result = {**prose, "metadata": wrapped(URL), "preflight_analysis": wrapped(URL)}
    if mode == "polling":
        result = projections.public_job(job)["result"]
    elif mode == "history":
        history = projections.History(1, 10000)
        history.record(job, job.result)
        result = history.get("job").result
    else:
        from tldw_Server_API.app.core.Web_Scraping.enhanced_web_scraping import ScrapingJobQueue

        queue = ScrapingJobQueue()
        queue._record_completed_job(job)
        result = queue.get_job("job").result
    assert result["metadata"] == wrapped("https://example.com/article")
    assert result["preflight_analysis"] == wrapped("https://example.com/article")
    assert {key: result[key] for key in prose} == prose
    assert job.result["metadata"] == wrapped(URL)


def test_nested_prose_redaction_preserves_exact_history_byte_limit():
    job = private_job()
    job.result = {"content": URL, "raw_html": f"<p>{URL}</p>", "metadata": {"content": URL}}
    expected = {
        **job.to_dict(),
        "url": "https://example.com/article",
        "metadata": {},
        "error": "execution_failed",
        "result": {"content": URL, "raw_html": f"<p>{URL}</p>", "metadata": {"content": "https://example.com/article"}},
    }
    size = len(json.dumps(expected, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))
    exact = projections.History(1, size)
    exact.record(job, job.result)
    retained = exact.get("job")
    assert retained is not None
    assert retained.to_dict() == expected
    assert exact.bytes == size
    smaller = projections.History(1, size - 1)
    smaller.record(job, job.result)
    assert smaller.get("job") is None
    assert smaller.bytes == 0
