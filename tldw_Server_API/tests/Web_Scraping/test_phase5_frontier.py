"""Atomic frontier occupancy, ordering and continuation invariants."""

from dataclasses import replace

import pytest
from hypothesis import given
from hypothesis import strategies as st

from tldw_Server_API.app.core.Web_Scraping.contracts.execution import DiscoveryResult, PageData
from tldw_Server_API.app.core.Web_Scraping.contracts.requests import CrawlRequest
from tldw_Server_API.app.core.Web_Scraping.crawl.frontier import Frontier
from tldw_Server_API.app.core.Web_Scraping.crawl.models import CrawlLimits, FrontierEntry
from tldw_Server_API.app.core.Web_Scraping.crawl.profiles import resolve_crawl_plan
from tldw_Server_API.app.core.Web_Scraping.runtime.controls import ParentStop

pytestmark = pytest.mark.unit
BASE = "https://example.com/blog/start"


def plan(profile="enhanced_recursive", **options):
    return resolve_crawl_plan(
        CrawlRequest(BASE),
        profile=profile,
        options=options,
        supplied=frozenset(options),
        config={},
        limits=CrawlLimits(),
    )


def entry(url, sequence=0, depth=0, score=0.0):
    return FrontierEntry(url, None, depth, sequence, score)


def page(url):
    return PageData({"url": url, "content": "kept", "extraction_successful": True}, DiscoveryResult(url, url, ()))


def test_unfinished_restores_admission_order():
    frontier = Frontier(plan(crawl_strategy="default"))
    entries = [entry(BASE + str(i), i) for i in range(3)]
    for item in entries:
        assert frontier.admit(item)
    first, second = frontier.take(), frontier.take()
    frontier.restore_unfinished()
    assert [frontier.take(), frontier.take(), frontier.take()] == [first, second, entries[2]]


def test_inflight_consumes_capacity_until_atomic_commit():
    frontier = Frontier(replace(plan(), limits=CrawlLimits(frontier_entries=1)))
    first = entry(BASE)
    frontier.admit(first)
    assert frontier.take() == first
    with pytest.raises(ParentStop, match="budget_exhausted"):
        frontier.admit(entry(BASE + "2", 1))
    assert frontier.snapshot().inflight == (first,)
    frontier.commit(first, page(BASE))
    assert frontier.admit(entry(BASE + "2", 1))
    assert frontier.snapshot().results == (page(BASE).article,)
    with pytest.raises(ValueError):
        frontier.commit(first, page(BASE))


@pytest.mark.parametrize("strategy,want", [("default", ["/b", "/a/x/y", "/a"]), ("best_first", ["/a/x/y", "/a", "/b"])])
def test_strategy_order_and_lexical_tiebreak(strategy, want):
    frontier = Frontier(plan(crawl_strategy=strategy))
    for i, (path, score) in enumerate([("/b", 0.5), ("/a/x/y", 1.0), ("/a", 0.5)]):
        frontier.admit(entry("https://example.com" + path, i, 1, score))
    assert [frontier.take().url for _ in range(3)] == ["https://example.com" + path for path in want]


def test_normalized_identity_deduplicates_without_charging_candidates():
    frontier = Frontier(plan())
    assert frontier.admit(entry("https://example.com/a?utm_source=x"))
    assert not frontier.admit(entry("https://EXAMPLE.com:443/a#frag", 1))
    assert len(frontier.snapshot().queued) == 1


@given(st.lists(st.integers(min_value=0, max_value=12), max_size=40))
def test_fifo_dispositions_and_results_match_unique_admission_order(numbers):
    frontier = Frontier(plan(crawl_strategy="default"))
    unique = list(dict.fromkeys(numbers))
    for sequence, number in enumerate(numbers):
        frontier.admit(entry(f"https://example.com/{number}", sequence))
    observed = []
    while (item := frontier.take()) is not None:
        observed.append(int(item.url.rsplit("/", 1)[1]))
        frontier.commit(item, page(item.url))
        state = frontier.snapshot()
        assert not set(state.queued) & set(state.inflight)
        assert not set(state.completed) & set(state.inflight)
        assert len(state.results) == len(state.completed)
    assert observed == unique


@pytest.mark.parametrize("field", ["max_url_bytes", "discovered_url_bytes"])
@pytest.mark.parametrize("bad", [True, False, 0, -1, 1.5, float("inf"), float("nan")])
def test_url_bounds_are_trusted_positive_integers(field, bad):
    with pytest.raises(ValueError):
        CrawlLimits(**{field: bad})
