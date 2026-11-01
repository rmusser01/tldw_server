"""Phase 5 parity baselines, not regression tests for intentional corrections.

False completion/checkpoint deletion and unsafe public projections are NOT frozen
here: owning lifecycle/checkpoint/projection tasks must test those corrections.
"""

from __future__ import annotations

import asyncio
import inspect
from enum import Enum
from urllib.parse import urlparse

import pytest

from tldw_Server_API.tests.Web_Scraping.phase5_fakes import (
    ENHANCED,
    LEGACY,
    PROFILE_MANIFEST,
    SERVICE,
    current_crawl_entrypoints,
    resolve,
)

pytestmark = pytest.mark.unit


def test_profile_manifest_covers_public_crawl_imports() -> None:
    assert current_crawl_entrypoints() - PROFILE_MANIFEST.entrypoints == set()


def test_profile_manifest_has_exact_profile_names_and_resolvable_adapters() -> None:
    assert set(PROFILE_MANIFEST.profiles) == {
        "legacy_recursive",
        "enhanced_recursive",
        "legacy_sitemap",
        "legacy_url_level",
        "enhanced_sitemap",
        "enhanced_url_level",
        "enhanced_batch",
    }
    for path in PROFILE_MANIFEST.entrypoints:
        assert callable(resolve(path)), path


def harness_for(profile, outcomes):
    from tldw_Server_API.tests.Web_Scraping.phase5_fakes import make_harness

    return make_harness(profile=profile, outcomes=outcomes)


BASE = "https://example.com/blog/start"


@pytest.mark.asyncio
@pytest.mark.parametrize("profile", list(PROFILE_MANIFEST.profiles))
async def test_every_profile_runs_the_existing_adapter_and_retains_article_payload(profile):
    harness = harness_for(profile, [{"url": BASE, "content": "kept article"}])
    result = await harness.run()
    articles = result["articles"] if profile == "enhanced_url_level" else result
    assert articles[0]["content"] == "kept article"
    assert articles[0]["metadata"]["nested"] == {"labels": ["baseline"]}
    assert articles[0]["custom_extra"] == {"rank": 7}
    assert articles[0]["preflight_analysis"] == {
        "analysis": {"results": {"js": {"js_required": True}}},
        "advice": {"backend": "curl", "method": "playwright", "notes": ["js_required"]},
    }
    assert type(articles[0]["metadata"]) is dict
    assert harness.calls


@pytest.mark.asyncio
async def test_legacy_midcrawl_transport_failure_contract():
    harness = harness_for(
        "legacy_recursive",
        [
            {"url": "https://example.com", "content": "kept article", "links": ["/blog/next"]},
            {"url": "https://example.com/blog/next", "transport_error": True},
        ],
    )
    articles = await harness.run(base_url="https://example.com", max_pages=3)
    assert articles[0]["content"] == "kept article"
    assert articles[1]["error"] == "browser_transport_unavailable"
    assert articles[1]["extraction_successful"] is False


@pytest.mark.asyncio
async def test_legacy_initial_transport_failure_is_a_structured_list():
    harness = harness_for("legacy_recursive", [{"url": BASE}])
    harness.transport_allowed = False
    articles = await harness.run()
    assert articles[0]["url"] == BASE
    assert articles[0]["error"] == "browser_transport_unavailable"
    assert harness.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "profile, expected",
    [
        ("legacy_recursive", ["/bad", "/good"]),
        ("enhanced_recursive", ["/bad"]),
    ],
)
async def test_profile_page_counters_from_a_failed_seed(profile, expected):
    harness = harness_for(
        profile,
        [
            {"url": "https://example.com/bad", "extraction_successful": False, "links": ["/good"]},
            {"url": "https://example.com/good"},
        ],
    )
    options = {"max_pages": 1}
    if profile == "legacy_recursive":
        options.update(
            base_url="https://example.com",
            resume_frontier=[
                ("https://example.com/bad", 0),
                ("https://example.com/good", 0),
            ],
        )
    await harness.run(**options)
    assert [urlparse(url).path for url in harness.attempted_urls] == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "strategy, expected",
    [
        ("default", [BASE, "https://example.com/short", "https://example.com/blog/a/deep"]),
        ("best_first", [BASE, "https://example.com/blog/a/deep", "https://example.com/short"]),
    ],
)
async def test_enhanced_fifo_and_best_first_have_distinct_order_and_exact_tags(strategy, expected):
    harness = harness_for(
        "enhanced_recursive",
        [
            {"url": BASE, "links": ["/short", "/blog/a/deep"]},
            {"url": "https://example.com/short"},
            {"url": "https://example.com/blog/a/deep"},
        ],
    )
    result = await harness.run(max_pages=3, max_depth=2, crawl_strategy=strategy)
    assert [article["url"] for article in result] == expected
    assert result[1]["metadata"]["parent_url"] == BASE
    assert result[1]["metadata"]["depth"] == 1
    assert result[1]["metadata"]["crawl_strategy"] == strategy
    assert result[1]["metadata"]["score_type"] == "composite_relevance"
    assert result[1]["metadata"]["ordering_score_type"] == "path_depth"
    # PathDepthScorer: optimal depth 3. Composite also includes content type and freshness.
    assert result[1]["metadata"]["ordering_score"] == pytest.approx(1.0 if strategy == "best_first" else 1 / 3)
    assert result[1]["metadata"]["score"] != result[1]["metadata"]["ordering_score"]


@pytest.mark.asyncio
async def test_legacy_fifo_uses_real_discovery_and_filters_without_adding_enhanced_tags():
    harness = harness_for(
        "legacy_recursive",
        [
            {
                "url": "https://example.com",
                "links": ["/blog/z", "/blog/a", "/login", "https://other.example/blog/x", "/blog/z"],
            },
            {"url": "https://example.com/blog/z"},
            {"url": "https://example.com/blog/a"},
        ],
    )
    result = await harness.run(base_url="https://example.com", max_pages=4, max_depth=1)
    assert [article["url"] for article in result] == [
        "https://example.com",
        "https://example.com/blog/z",
        "https://example.com/blog/a",
    ]
    assert all("crawl_strategy" not in article["metadata"] for article in result)
    assert all("_discovered_links" not in article for article in result)
    assert harness.clock.sleeps == [0.0, 0.0, 0.0]
    assert harness.callbacks[:2] == [
        "Scraping page 1/4: https://example.com",
        "Scraping page 2/4: https://example.com/blog/z",
    ]
    # Do not freeze the unconditional terminal callback/checkpoint deletion bug.


@pytest.mark.asyncio
@pytest.mark.parametrize("profile", ["legacy_recursive", "enhanced_recursive", "enhanced_sitemap", "enhanced_batch"])
async def test_profile_credentials_and_explicit_analysis_intent_reach_page_execution(profile):
    harness = harness_for(profile, [{"url": BASE}])
    await harness.run(custom_cookies=harness.credentials["custom_cookies"], allow_llm_extraction=False)
    assert all(call["kwargs"]["allow_llm_extraction"] is False for call in harness.calls)
    if profile == "legacy_recursive":
        assert harness.transport_calls[0]["cookies"] == tuple(harness.credentials["custom_cookies"])
    else:
        assert harness.calls[0]["kwargs"]["custom_cookies"] == harness.credentials["custom_cookies"]


@pytest.mark.asyncio
@pytest.mark.parametrize("profile", ["legacy_sitemap", "enhanced_sitemap"])
@pytest.mark.parametrize(
    "kind, expected_count",
    [
        ("namespaced", 2),
        ("plain", 0),
        ("index", 1),
        ("malformed", 0),
    ],
)
async def test_sitemap_formats_preserve_each_adapter_current_parser_contract(profile, kind, expected_count):
    harness = harness_for(profile, [{"url": BASE}, {"url": "https://example.com/blog/second"}])
    harness.sitemap_kind = kind
    result = await harness.run()
    # Enhanced BeautifulSoup accepts unnamespaced locs and recovers malformed XML.
    if profile == "enhanced_sitemap" and kind in {"plain", "malformed"}:
        expected_count = 2 if kind == "plain" else 1
    assert len(result) == expected_count
    assert harness.responses and all(response.closed for response in harness.responses)
    if kind == "index":
        assert result[0]["url"] == "https://example.com/child-sitemap.xml"
        assert len(harness.fetch_calls) == 1  # Current adapters scrape locs; no nested expansion.


@pytest.mark.asyncio
async def test_enhanced_sitemap_filter_precedes_max_urls_and_keeps_credentials():
    harness = harness_for(
        "enhanced_sitemap",
        [
            {"url": "https://example.com/blog/skip"},
            {"url": BASE},
            {"url": "https://example.com/blog/last"},
        ],
    )
    result = await harness.run(
        filter_func=lambda url: not url.endswith("skip"),
        max_urls=1,
        custom_cookies=harness.credentials["custom_cookies"],
        user_agent="phase5-agent",
        custom_headers={"X-Phase5": "test"},
    )
    assert [article["url"] for article in result] == [BASE]
    assert harness.fetch_calls[0]["headers"] == {"User-Agent": "phase5-agent", "X-Phase5": "test"}
    assert harness.fetch_calls[0]["cookies"] == {"session": "test-cookie"}


@pytest.mark.asyncio
@pytest.mark.parametrize("profile, respect", [("legacy_sitemap", False), ("enhanced_sitemap", True)])
async def test_sitemap_robots_policy_denial_prevents_fetch(profile, respect):
    harness = harness_for(profile, [{"url": BASE}])
    harness.denied_urls.add(harness.sitemap_url)
    assert await harness.run() == []
    assert harness.fetch_calls == []
    assert harness.policy_calls[0]["respect_robots"] is respect


@pytest.mark.asyncio
@pytest.mark.parametrize("profile", ["legacy_url_level", "enhanced_url_level"])
async def test_url_level_is_absolute_path_depth_not_seed_relative(profile):
    harness = harness_for(
        profile,
        [
            {"url": BASE, "links": ["/blog/child", "/blog/child/deep"]},
            {"url": "https://example.com/blog/child"},
            {"url": "https://example.com/blog/child/deep"},
        ],
    )
    result = await harness.run(level=2)
    articles = result["articles"] if profile == "enhanced_url_level" else result
    assert {article["url"] for article in articles} == {BASE, "https://example.com/blog/child"}
    if profile == "enhanced_url_level":
        assert result["method"] == "URL Level"
        assert result["url_level"] == 2
        assert result["total_articles"] == 2


@pytest.mark.asyncio
async def test_batch_failure_envelope_retains_nested_metadata_and_input_order():
    harness = harness_for(
        "enhanced_batch",
        [
            {"url": BASE},
            {"url": "https://example.com/bad", "error": "fixture extraction failure", "extraction_successful": False},
        ],
    )
    result = await harness.run(user_id="owner-5", allow_llm_extraction=False)
    assert [article["url"] for article in result] == [BASE, "https://example.com/bad"]
    assert result[1]["error"] == "fixture extraction failure"
    assert result[1]["metadata"]["nested"] == {"labels": ["baseline"]}
    assert result[1]["custom_extra"] == {"rank": 7}
    assert [job.user_id for job in harness.jobs] == ["owner-5", "owner-5"]
    assert len([call for call in harness.calls if call["url"].endswith("bad")]) == 4


@pytest.mark.asyncio
async def test_fake_transport_and_executor_gates_park_real_legacy_adapter():
    harness = harness_for("legacy_recursive", [{"url": BASE}])
    harness.transport_release.clear()
    harness.executor_release.clear()
    task = asyncio.create_task(harness.run())
    try:
        await asyncio.wait_for(harness.transport_started.wait(), 2)
        assert not task.done()
        harness.transport_release.set()
        await asyncio.wait_for(harness.executor_started.wait(), 2)
        assert not task.done()
        harness.executor_release.set()
        assert (await asyncio.wait_for(task, 2))[0]["url"] == BASE
    finally:
        harness.transport_release.set()
        harness.executor_release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize("profile", ["legacy_recursive", "legacy_sitemap", "legacy_url_level"])
def test_sync_legacy_adapters_return_materialized_lists(profile):
    harness = harness_for(profile, [{"url": BASE}])
    result = harness.run_sync()
    assert isinstance(result, list)
    assert not inspect.isawaitable(result)
    assert result[0]["content"] == "kept article"


LEGACY_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/58.0.3029.110 Safari/537.3"
)

SIGNATURES = {
    f"{LEGACY}.recursive_scrape": (
        "base_url max_pages max_depth delay resume_file user_agent custom_cookies progress_callback allow_llm_extraction",
        {
            "delay": 1.0,
            "resume_file": "scrape_progress.json",
            "user_agent": LEGACY_USER_AGENT,
            "custom_cookies": None,
            "progress_callback": None,
            "allow_llm_extraction": True,
        },
        (),
        True,
    ),
    f"{LEGACY}.sync_recursive_scrape": (
        "url_input max_pages max_depth delay custom_cookies allow_llm_extraction",
        {"delay": 1.0, "custom_cookies": None, "allow_llm_extraction": True},
        (),
        False,
    ),
    f"{LEGACY}.scrape_from_sitemap": (
        "sitemap_url allow_llm_extraction",
        {"allow_llm_extraction": True},
        ("allow_llm_extraction",),
        False,
    ),
    f"{LEGACY}.scrape_by_url_level": (
        "base_url level allow_llm_extraction",
        {"allow_llm_extraction": True},
        ("allow_llm_extraction",),
        False,
    ),
    f"{ENHANCED}.recursive_scrape": (
        "base_url max_pages max_depth url_filter custom_cookies user_agent custom_headers task_id user_id include_external_override score_threshold_override crawl_strategy allow_llm_extraction",
        {
            "max_pages": 100,
            "max_depth": 3,
            "url_filter": None,
            "custom_cookies": None,
            "user_agent": None,
            "custom_headers": None,
            "task_id": None,
            "user_id": None,
            "include_external_override": None,
            "score_threshold_override": None,
            "crawl_strategy": None,
            "allow_llm_extraction": True,
        },
        ("include_external_override", "score_threshold_override", "crawl_strategy", "allow_llm_extraction"),
        True,
    ),
    f"{ENHANCED}.scrape_sitemap": (
        "sitemap_url filter_func max_urls custom_cookies user_agent custom_headers task_id user_id allow_llm_extraction",
        {
            "filter_func": None,
            "max_urls": None,
            "custom_cookies": None,
            "user_agent": None,
            "custom_headers": None,
            "task_id": None,
            "user_id": None,
            "allow_llm_extraction": True,
        },
        (),
        True,
    ),
    f"{ENHANCED}.scrape_multiple": (
        "urls method priority summarize user_id allow_llm_extraction kwargs",
        {"method": "auto", "priority": "NORMAL", "summarize": False, "user_id": None, "allow_llm_extraction": True},
        (),
        True,
    ),
    f"{SERVICE}._scrape_by_url_level": (
        "base_url url_level max_pages summarize custom_prompt api_name api_key keywords system_prompt temperature priority custom_cookies user_agent custom_headers user_id include_external score_threshold crawl_strategy task_id summary_prompt_overrides",
        {
            "user_id": None,
            "include_external": None,
            "score_threshold": None,
            "crawl_strategy": None,
            "task_id": None,
            "summary_prompt_overrides": None,
        },
        ("user_id", "include_external", "score_threshold", "crawl_strategy", "task_id", "summary_prompt_overrides"),
        True,
    ),
}


@pytest.mark.parametrize("entrypoint", list(SIGNATURES))
def test_profile_signature_snapshot_preserves_defaults_kinds_and_async_contract(entrypoint):
    adapter = resolve(entrypoint)
    parameters = {
        name: parameter for name, parameter in inspect.signature(adapter).parameters.items() if name != "self"
    }
    names, defaults, keyword_only, asynchronous = SIGNATURES[entrypoint]
    assert list(parameters) == names.split()
    assert {
        name: parameter.default.name if isinstance(parameter.default, Enum) else parameter.default
        for name, parameter in parameters.items()
        if parameter.default is not inspect.Parameter.empty
    } == defaults
    assert (
        tuple(name for name, parameter in parameters.items() if parameter.kind is inspect.Parameter.KEYWORD_ONLY)
        == keyword_only
    )
    assert inspect.iscoroutinefunction(adapter) is asynchronous
    for name, parameter in parameters.items():
        expected_kind = (
            inspect.Parameter.VAR_KEYWORD
            if name == "kwargs"
            else (inspect.Parameter.KEYWORD_ONLY if name in keyword_only else inspect.Parameter.POSITIONAL_OR_KEYWORD)
        )
        assert parameter.kind is expected_kind


@pytest.mark.asyncio
@pytest.mark.parametrize("strategy", ["default", "best_first"])
@pytest.mark.parametrize("override", [False, True])
async def test_enhanced_explicit_false_and_zero_override_configured_external_and_score_filters(strategy, override):
    harness = harness_for(
        "enhanced_recursive",
        [
            {"url": BASE, "links": ["https://external.example/blog/a", "/blog/child"]},
            {"url": "https://external.example/blog/a"},
            {"url": "https://example.com/blog/child"},
        ],
    )
    harness.config.update(web_crawl_include_external=True, web_crawl_score_threshold=0.99)
    options = {"include_external_override": False, "score_threshold_override": 0.0} if override else {}
    result = await harness.run(max_pages=3, crawl_strategy=strategy, **options)
    assert [article["url"] for article in result] == ([BASE, "https://example.com/blog/child"] if override else [BASE])


@pytest.mark.asyncio
async def test_enhanced_batch_omitted_credentials_and_analysis_defaults_reach_executor():
    harness = harness_for("enhanced_batch", [{"url": BASE}])
    await harness.run()
    assert harness.calls[0]["kwargs"] == {
        "method": "auto",
        "custom_cookies": None,
        "user_agent": None,
        "custom_headers": None,
        "allow_llm_extraction": True,
    }
    assert harness.jobs[0].metadata == {"allow_llm_extraction": True}


@pytest.mark.asyncio
async def test_fake_clock_advances_without_sleeping_the_process():
    from tldw_Server_API.tests.Web_Scraping.phase5_fakes import FakeClock

    clock = FakeClock()
    await clock.sleep(1.25)
    await clock.sleep(0.75)
    assert clock.now == 2.0
    assert clock.sleeps == [1.25, 0.75]
