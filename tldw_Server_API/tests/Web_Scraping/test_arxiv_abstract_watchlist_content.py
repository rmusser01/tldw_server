"""Canonical arXiv abstract selection through real scraping/Watchlist consumers."""

from __future__ import annotations

import hashlib
import json
from unittest.mock import AsyncMock
from urllib.parse import urlsplit

import pytest

from tldw_Server_API.app.core.DB_Management.Collections_DB import CollectionsDatabase
from tldw_Server_API.app.core.DB_Management.Watchlists_DB import WatchlistsDatabase
from tldw_Server_API.app.core.Watchlists import fetchers, pipeline
from tldw_Server_API.app.core.Web_Scraping.content import ContentMetadataHandler
from tldw_Server_API.app.core.Web_Scraping.extraction import extract_article_with_pipeline
from tldw_Server_API.app.core.Web_Scraping.extraction.strategies.trafilatura import extract_with_trafilatura
from tldw_Server_API.app.core.Web_Scraping.orchestration import article
from tldw_Server_API.app.core.Web_Scraping.orchestration.article_models import (
    ArticleLimits,
    ArticlePlan,
    DirectBrowserProfile,
)
from tldw_Server_API.app.core.Web_Scraping.preflight import PreflightTarget
from tldw_Server_API.app.core.Web_Scraping.runtime import FetchResponse, PolicyDecision, RuntimeRequestContext

pytestmark = pytest.mark.unit

URL = "https://arxiv.org/abs/2610.11410"
ABSTRACT = "Atlas sensors detected exactly seven cobalt pulses. Independent measurements confirmed the orbital period."
GENERIC = "The community laboratory offers independent software integrations and collaborative experiments."
LABS = "arXivLabs is a framework for collaborators to develop and share new arXiv features."


def _html(*, meta: str | None = ABSTRACT, block: str | None = None) -> str:
    abstract_meta = f'<meta name="citation_abstract" content="{meta}">' if meta is not None else ""
    abstract_block = (
        f'<blockquote class="abstract mathjax"><span class="descriptor">Abstract:</span> {block}</blockquote>'
        if block is not None
        else ""
    )
    # Publisher-shaped abstract metadata/block, followed by a much larger unrelated section.
    paragraphs = "".join(
        f"<p>{LABS} {GENERIC} This is unrelated service information {index}.</p>" for index in range(12)
    )
    return (
        "<html><head><title>Synthetic Atlas study</title>"
        '<meta name="citation_title" content="Synthetic Atlas study">'
        '<meta name="citation_author" content="Mira Example">'
        f'{abstract_meta}</head><body><div id="abs-outer"><h1>Synthetic Atlas study</h1>'
        f'{abstract_block}<section id="arxivlabs"><h2>arXivLabs</h2>{paragraphs}</section></div></body></html>'
    )


def _body(result: dict) -> str:
    return ContentMetadataHandler.strip_metadata(result["content"]).strip()


@pytest.mark.parametrize("meta,block", [(ABSTRACT, None), (None, ABSTRACT), ("  ", ABSTRACT)])
def test_real_trafilatura_selects_publisher_abstract_instead_of_labs(meta, block):
    result = extract_with_trafilatura(_html(meta=meta, block=block), URL)
    assert result["extraction_successful"] is True  # nosec B101 # pytest behavioral oracle, not production authorization
    assert _body(result) == ABSTRACT  # nosec B101 # pytest behavioral oracle, not production authorization
    assert result["url"] == URL  # nosec B101 # pytest behavioral oracle, not production authorization
    assert result["title"] == "Synthetic Atlas study"  # nosec B101 # pytest behavioral oracle, not production authorization
    assert ContentMetadataHandler.METADATA_START in result["content"]  # nosec B101 # pytest behavioral oracle, not production authorization


@pytest.mark.parametrize(
    "url",
    [
        "https://arxiv.org/abs/2610.11410v2",
        "https://arxiv.org/abs/hep-th/9901001v3",
    ],
)
def test_canonical_versioned_and_legacy_abstract_paths_select_the_abstract(url):
    result = extract_with_trafilatura(_html(), url)
    assert _body(result) == ABSTRACT  # nosec B101 # pytest behavioral oracle, not production authorization
    assert result["url"] == url  # nosec B101 # pytest behavioral oracle, not production authorization


@pytest.mark.parametrize(
    "url",
    [
        "https://arxiv.org.evil.invalid/abs/2610.11410",
        "https://arxiv.org@evil.invalid/abs/2610.11410",
        "https://example.com/abs/2610.11410",
        "https://arxiv.org/html/2610.11410",
        "https://arxiv.org/pdf/2610.11410",
        "https://arxiv.org/abs/2610.11410/extra",
        "https://arxiv.org/abs/not-a-paper",
    ],
)
def test_noncanonical_hosts_and_nonabstract_paths_keep_generic_body(url):
    result = extract_with_trafilatura(_html(), url)
    assert result["extraction_successful"] is True  # nosec B101 # pytest behavioral oracle, not production authorization
    assert GENERIC in _body(result)  # nosec B101 # pytest behavioral oracle, not production authorization
    assert ABSTRACT not in _body(result)  # nosec B101 # pytest behavioral oracle, not production authorization
    assert result["url"] == url  # nosec B101 # pytest behavioral oracle, not production authorization


@pytest.mark.parametrize("meta,block", [(None, None), ("  ", "  ")])
def test_missing_or_empty_abstract_keeps_existing_generic_fallback(meta, block):
    result = extract_with_trafilatura(_html(meta=meta, block=block), URL)
    assert result["extraction_successful"] is True  # nosec B101 # pytest behavioral oracle, not production authorization
    assert GENERIC in _body(result)  # nosec B101 # pytest behavioral oracle, not production authorization


def test_empty_page_keeps_unsuccessful_extraction():
    result = extract_with_trafilatura("", URL)
    assert result["extraction_successful"] is False  # nosec B101 # pytest behavioral oracle, not production authorization
    assert result["content"] == ""  # nosec B101 # pytest behavioral oracle, not production authorization


class _Fetch:
    def __init__(self, html: str, *, redirect: str | None = None):
        self.html = html
        self.redirect = redirect
        self.requests = []

    def fetch(self, request):
        self.requests.append(request)
        if self.redirect and len(self.requests) == 1:
            return FetchResponse(
                url=request.url, status=302, text="", headers={"Location": self.redirect}, backend="httpx"
            )
        return FetchResponse(url=request.url, status=200, text=self.html, headers={}, backend="httpx")


class _Executor:
    async def run(self, function, /, *args, **kwargs):
        return function(*args, **kwargs)


class _NoBrowser:
    async def acquire(self, *_args, **_kwargs):
        raise AssertionError("The blocking HTTP fixture must not launch a browser")


def _install_blocking(monkeypatch, html, *, denied=None, redirect=None):
    fetch = _Fetch(html, redirect=redirect)
    policy_urls = []

    async def evaluate(url, **_kwargs):
        policy_urls.append(url)
        allowed = url != denied
        return PreflightTarget(
            url=url,
            decision=PolicyDecision(
                allowed=allowed,
                mode="test",
                reason="allowed" if allowed else "private_ip",
                stage="pre_fetch",
                source="article_extract_blocking",
            ),
            request_context=RuntimeRequestContext(source="article_extract_blocking", stage="pre_fetch"),
        )

    dependencies = article.ArticleDependencies(
        load_config=lambda: {"web_scraper": {"web_scraper_preflight_analyzers": False}},
        resolve_plan=lambda url, _config: ArticlePlan(
            url=url,
            domain=urlsplit(url).hostname or "",
            backend="httpx",
            handler="",
            browser=DirectBrowserProfile(
                user_agent="fixture",
                custom_cookies=(),
                retries=1,
                timeout_ms=1000,
                stealth_enabled=False,
                stealth_wait_ms=0,
            ),
            limits=ArticleLimits(max_article_bytes=100000, max_browser_transfer_bytes=100000),
        ),
        evaluate_target=evaluate,
        run_preflight=AsyncMock(side_effect=AssertionError("Disabled preflight must not run")),
        apply_preflight_advice=lambda result, **kwargs: (kwargs["backend"], kwargs["method"], result),
        fetch_client=fetch,
        browser=_NoBrowser(),
        executor=_Executor(),
        extract=extract_article_with_pipeline,
        build_preflight_context=lambda *_args, **_kwargs: None,
        preflight_options=lambda _values: None,
        public_preflight_payload=lambda *_args: None,
        resolve_handler=lambda _path: None,
        js_required=lambda *_args: False,
        convert_content=article.convert_html_to_markdown,
        increment_counter=lambda *_args, **_kwargs: None,
        observe_histogram=lambda *_args, **_kwargs: None,
        clock=lambda: 0.0,
        log=lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(article, "_build_default_dependencies", lambda _cookies: dependencies)
    return fetch, policy_urls


def test_blocking_watchlist_consumer_retains_exact_abstract_and_source_identity(monkeypatch):
    fetch, policy_urls = _install_blocking(monkeypatch, _html())
    result = fetchers.fetch_site_article(URL)
    assert result["content"].strip() == ABSTRACT  # nosec B101 # pytest behavioral oracle, not production authorization
    assert result["url"] == URL  # nosec B101 # pytest behavioral oracle, not production authorization
    assert result["title"] == "Synthetic Atlas study"  # nosec B101 # pytest behavioral oracle, not production authorization
    assert [request.url for request in fetch.requests] == [URL]  # nosec B101 # pytest behavioral oracle, not production authorization
    assert policy_urls == [URL]  # nosec B101 # pytest behavioral oracle, not production authorization


@pytest.mark.parametrize("redirect", [None, "http://127.0.0.1/private"])
def test_blocking_policy_denial_prevents_abstract_acquisition(monkeypatch, redirect):
    denied = redirect or URL
    fetch, policy_urls = _install_blocking(monkeypatch, _html(), denied=denied, redirect=redirect)
    result = fetchers.fetch_site_article(URL)
    assert not result["content"]  # nosec B101 # pytest behavioral oracle, not production authorization
    assert policy_urls == ([URL, redirect] if redirect else [URL])  # nosec B101 # pytest behavioral oracle, not production authorization
    assert len(fetch.requests) == (1 if redirect else 0)  # nosec B101 # pytest behavioral oracle, not production authorization


@pytest.mark.asyncio
async def test_real_watchlist_run_persists_paper_abstract_in_owned_item_and_collection(monkeypatch, tmp_path):
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path / "users"))
    monkeypatch.setenv("TEST_MODE", "0")
    monkeypatch.setenv("TLDW_TEST_MODE", "0")
    fetch, _policy_urls = _install_blocking(monkeypatch, _html())
    user_id = 921
    db = WatchlistsDatabase.for_user(user_id)
    source = db.create_source(
        name="Synthetic arXiv",
        url="https://rss.arxiv.org/rss/cs.AI",
        source_type="rss",
        active=True,
        settings_json=json.dumps({"limit": 1}),
        tags=[],
        group_ids=[],
    )
    job = db.create_job(
        name="Manual paper briefing",
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

    async def feed(*_args, **_kwargs):
        return {
            "status": 200,
            "etag": None,
            "last_modified": None,
            "items": [{"title": "Synthetic Atlas study", "url": URL, "summary": ABSTRACT, "published": None}],
        }

    monkeypatch.setattr(pipeline, "fetch_rss_feed", feed)
    monkeypatch.setattr(pipeline, "fetch_rss_feed_history", feed)
    monkeypatch.setattr(pipeline, "enqueue_embeddings_job_for_item", AsyncMock(return_value=None))
    result = await pipeline.run_watchlist_job(user_id, job.id)
    items, total = db.list_items(run_id=result["run_id"])
    assert (result["items_found"], result["items_ingested"], result["source_errors"], total) == (1, 1, 0, 1)  # nosec B101 # pytest behavioral oracle, not production authorization
    assert (items[0].summary, items[0].content, items[0].url, items[0].source_id) == (
        ABSTRACT,
        ABSTRACT,
        URL,
        source.id,
    )  # nosec B101 # pytest behavioral oracle, not production authorization
    collection_items, collection_total = CollectionsDatabase.for_user(user_id).list_content_items(page=1, size=10)
    assert collection_total == 1  # nosec B101 # pytest behavioral oracle, not production authorization
    assert (
        collection_items[0].summary,
        collection_items[0].url,
        collection_items[0].source_id,
        collection_items[0].content_hash,
    ) == (ABSTRACT, URL, source.id, hashlib.sha256(ABSTRACT.encode()).hexdigest())  # nosec B101 # pytest behavioral oracle, not production authorization
    assert [request.url for request in fetch.requests] == [URL]  # nosec B101 # pytest behavioral oracle, not production authorization


@pytest.mark.parametrize(
    "options",
    [
        {"strategy_order": ["cluster", "trafilatura"]},
        {"schema_rules": {}},
        {"llm_settings": {}},
        {"regex_settings": {}},
        {"cluster_settings": {}},
    ],
)
def test_real_pipeline_preserves_each_explicit_strategy_settings_override(options):
    result = extract_article_with_pipeline(_html(), URL, allow_llm_extraction=False, **options)
    assert result["extraction_strategy"] == "cluster"  # nosec B101 # pytest behavioral oracle, not production authorization
    assert GENERIC in _body(result)  # nosec B101 # pytest behavioral oracle, not production authorization
    assert ABSTRACT not in _body(result)  # nosec B101 # pytest behavioral oracle, not production authorization


def test_real_pipeline_preserves_explicit_handler():
    calls = []

    def handler(page_html, url):
        calls.append((page_html, url))
        return {"url": url, "content": "Caller-selected article content.", "extraction_successful": True}

    result = extract_article_with_pipeline(_html(), URL, handler=handler, allow_llm_extraction=False)
    assert result["extraction_strategy"] == "schema"  # nosec B101 # pytest behavioral oracle, not production authorization
    assert result["content"] == "Caller-selected article content."  # nosec B101 # pytest behavioral oracle, not production authorization
    assert calls == [(_html(), URL)]  # nosec B101 # pytest behavioral oracle, not production authorization


def test_real_pipeline_preserves_explicit_fallback_extractor_order():
    calls = []

    def fallback(page_html, url):
        calls.append((page_html, url))
        return {"url": url, "content": "Caller-selected fallback.", "extraction_successful": True}

    result = extract_article_with_pipeline(_html(), URL, fallback_extractor=fallback, allow_llm_extraction=False)
    assert result["extraction_strategy"] == "cluster"  # nosec B101 # pytest behavioral oracle, not production authorization
    assert GENERIC in _body(result)  # nosec B101 # pytest behavioral oracle, not production authorization
    assert calls == []  # nosec B101 # pytest behavioral oracle, not production authorization


@pytest.mark.parametrize("meta,block", [(None, None), (" ", " ")])
def test_real_pipeline_without_usable_abstract_preserves_default_generic_selection(meta, block):
    result = extract_article_with_pipeline(_html(meta=meta, block=block), URL, allow_llm_extraction=False)
    assert result["extraction_strategy"] == "cluster"  # nosec B101 # pytest behavioral oracle, not production authorization
    assert GENERIC in _body(result)  # nosec B101 # pytest behavioral oracle, not production authorization


@pytest.mark.parametrize("url", ["https://example.com/abs/2610.11410", "https://arxiv.org/html/2610.11410"])
def test_real_pipeline_preserves_nonarxiv_and_full_paper_default_selection(url):
    result = extract_article_with_pipeline(_html(), url, allow_llm_extraction=False)
    assert result["extraction_strategy"] == "cluster"  # nosec B101 # pytest behavioral oracle, not production authorization
    assert GENERIC in _body(result)  # nosec B101 # pytest behavioral oracle, not production authorization
    assert result["url"] == url  # nosec B101 # pytest behavioral oracle, not production authorization
