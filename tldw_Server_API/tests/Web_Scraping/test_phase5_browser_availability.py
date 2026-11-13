"""Enhanced no-browser fallback stays policy-admitted canonical HTTP work."""

import asyncio

import pytest

from tldw_Server_API.app.core.Web_Scraping import enhanced_web_scraping as enhanced
from tldw_Server_API.app.core.Web_Scraping.browser_transport import decide_browser_transport
from tldw_Server_API.app.core.Web_Scraping.orchestration import article_browser
from tldw_Server_API.tests.Web_Scraping.test_phase5_default_batch import BASE
from tldw_Server_API.tests.Web_Scraping.test_phase5_default_batch import batch as _batch_fixture

pytestmark = pytest.mark.unit


@pytest.fixture
def batch(monkeypatch, tmp_path):
    return _batch_fixture.__wrapped__(monkeypatch, tmp_path)


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["httpx", "auto", "playwright"])
async def test_default_batch_preserves_uninitialized_public_browser_fallback(batch, monkeypatch, backend):
    admissions, launches = [], []
    batch.scraper.config.update(web_scraper_default_backend=backend, web_scraper_retry_count=1)

    def admitted():
        admissions.append("transport")
        return decide_browser_transport(
            configured_mode="url_guarded", auth_mode="single_user", outbound_policy_mode="compat"
        )

    async def unavailable(self):
        launches.append("browser")
        raise article_browser.ArticleFailure("browser_error", "launch")

    monkeypatch.setattr(enhanced, "default_browser_transport_decision", admitted)
    monkeypatch.setattr(article_browser._DefaultPlaywrightLauncher, "start", unavailable)
    try:
        assert batch.scraper._browser is None
        old = await batch.scraper.scrape_article(BASE + "old", method="playwright")
        report = await batch.scraper._batch_report([BASE + "new"], method="playwright")
        assert old["extraction_successful"] and report.articles[0]["extraction_successful"]
        assert old["method"] == report.articles[0]["method"] == "trafilatura"
        assert len(batch.calls) == 2 and launches == []
        assert admissions == ["transport", "transport"]
        snapshot = batch.contexts[-1].snapshot()
        assert snapshot.page_attempts == snapshot.network_requests == 1 and snapshot.browser_attempts == 0
    finally:
        await batch.scraper.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["disabled", "attested_proxy"])
async def test_uninitialized_batch_transport_denial_prevents_http_fallback(batch, monkeypatch, mode):
    admissions = []

    def denied():
        admissions.append("transport")
        return decide_browser_transport(configured_mode=mode, auth_mode="multi_user", outbound_policy_mode="strict")

    monkeypatch.setattr(enhanced, "default_browser_transport_decision", denied)
    try:
        old = await batch.scraper.scrape_article(BASE + "old", method="playwright")
        report = await batch.scraper._batch_report([BASE + "new"], method="playwright")
        assert old["error"] == report.articles[0]["error"] == "browser_transport_unavailable"
        assert old["capability"] == report.articles[0]["capability"]
        assert len(admissions) >= 2 and batch.calls == []
        assert batch.contexts[-1].snapshot().browser_attempts == 0
    finally:
        await batch.scraper.stop()


@pytest.mark.asyncio
async def test_uninitialized_advised_browser_fallback_preserves_preflight_and_instance_metadata(batch, monkeypatch):
    from tldw_Server_API.app.core.Web_Scraping.contracts import PreflightResult

    batch.scraper.config.update(
        web_scraper_default_backend="auto",
        web_scraper_preflight_analyzers=True,
        web_scraper_preflight_include_results=True,
    )
    pipeline = batch.scraper._extract_from_html_with_pipeline

    def metadata(*args, **kwargs):
        data = pipeline(*args, **kwargs)
        data["instance_extra"] = {"labels": ["preserved"]}
        return data

    async def analysis(*args, **kwargs):
        return PreflightResult(analysis={"results": {"js": {"status": "success", "js_required": True}}})

    monkeypatch.setattr(batch.scraper, "_extract_from_html_with_pipeline", metadata)
    monkeypatch.setattr(enhanced.preflight_facade, "run_preflight", analysis)
    try:
        old = await batch.scraper.scrape_article(BASE + "old")
        report = await batch.scraper._batch_report([BASE + "new"])
        data = report.articles[0]
        assert old["method"] == data["method"] == "trafilatura"
        assert old["preflight_analysis"] == data["preflight_analysis"]
        assert old["instance_extra"] == data["instance_extra"] == {"labels": ["preserved"]}
        assert len(batch.calls) == 2 and batch.contexts[-1].snapshot().browser_attempts == 0
    finally:
        await batch.scraper.stop()


@pytest.mark.asyncio
async def test_uninitialized_browser_fallback_is_not_reached_after_real_preflight_parent_stop(batch, monkeypatch):
    admissions = []
    monkeypatch.setattr(enhanced, "default_browser_transport_decision", lambda: admissions.append("forbidden"))
    batch.limits(network_requests=1)
    batch.scraper.config.update(web_scraper_preflight_analyzers=True, web_scraper_preflight_enable_external_tools=False)
    try:
        report = await batch.scraper._batch_report([BASE + "one"], method="playwright")
        assert report.outcome.stop_reason == "budget_exhausted" and admissions == []
        assert len(batch.calls) == 1 and batch.calls[0][0].endswith("/robots.txt")
        assert batch.contexts[-1].snapshot().browser_attempts == 0
    finally:
        await batch.scraper.stop()


@pytest.mark.asyncio
async def test_uninitialized_browser_fallback_is_not_reached_after_policy_denial(batch, monkeypatch):
    admissions = []
    monkeypatch.setattr(enhanced, "default_browser_transport_decision", lambda: admissions.append("forbidden"))
    try:
        result = await batch.scraper.scrape_multiple(["file:///private/article"], method="playwright")
        assert result[0]["policy_reason"] and admissions == [] and batch.calls == []
    finally:
        await batch.scraper.stop()


@pytest.mark.asyncio
async def test_uninitialized_browser_fallback_is_not_reached_after_caller_cancels_preflight(batch, monkeypatch):
    entered = asyncio.Event()
    admissions = []

    async def analysis(*args, **kwargs):
        entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(enhanced.preflight_facade, "run_preflight", analysis)
    monkeypatch.setattr(enhanced, "default_browser_transport_decision", lambda: admissions.append("forbidden"))
    batch.scraper.config["web_scraper_preflight_analyzers"] = True
    task = asyncio.create_task(batch.scraper._batch_report([BASE + "one"], method="playwright"))
    try:
        await asyncio.wait_for(entered.wait(), 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert admissions == [] and batch.calls == []
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await batch.scraper.stop()


@pytest.mark.asyncio
async def test_public_recursive_manual_browser_availability_uses_same_canonical_fallback(batch, monkeypatch):
    from tldw_Server_API.app.core.Web_Scraping.orchestration import crawl, crawl_context

    monkeypatch.setattr(crawl, "CrawlContext", crawl_context.CrawlContext)
    batch.scraper.config["web_scraper_default_backend"] = "playwright"
    try:
        result = await batch.scraper.recursive_scrape(BASE + "walk", max_pages=1)
        assert result[0]["extraction_successful"] and len(batch.calls) == 1
        snapshot = batch.contexts[-1].snapshot()
        assert snapshot.page_attempts == snapshot.network_requests == 1 and snapshot.browser_attempts == 0
    finally:
        await batch.scraper.stop()


@pytest.mark.asyncio
async def test_uninitialized_fallback_http_byte_stop_never_retries_browser(batch):
    batch.limits(transfer_bytes=1)
    try:
        report = await batch.scraper._batch_report([BASE + "one"], method="playwright")
        assert report.outcome.stop_reason == "budget_exhausted" and len(batch.calls) == 1
        snapshot = batch.contexts[-1].snapshot()
        assert snapshot.page_attempts == snapshot.network_requests == 1 and snapshot.browser_attempts == 0
    finally:
        await batch.scraper.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [200, 404])
async def test_manual_recursive_http_fallback_is_terminal_despite_js_hints_or_http_failure(batch, monkeypatch, status):
    launches = []

    async def unavailable(self):
        launches.append("forbidden native bootstrap")
        raise article_browser.ArticleFailure("browser_error", "launch")

    monkeypatch.setattr(article_browser._DefaultPlaywrightLauncher, "start", unavailable)
    batch.scraper.config.update(web_scraper_default_backend="playwright", web_scraper_retry_count=1)
    batch.body = batch.body.replace("</body>", "<footer>please enable javascript</footer></body>")
    batch.status = lambda: status
    try:
        result = await batch.scraper.recursive_scrape(BASE + "walk", max_pages=1)
        assert batch.calls and launches == []
        if status == 200:
            assert result[0]["extraction_successful"]
        else:
            assert not any(item.get("extraction_successful") for item in result)
    finally:
        await batch.scraper.stop()
