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


def test_traversal_decision_freezes_safe_nested_metadata_and_rejects_unbounded_reasons():
    from tldw_Server_API.app.core.Web_Scraping.crawl.models import TraversalDecision

    source = {"nested": {"rank": 7}, "api_key": "private"}
    decision = TraversalDecision(metadata=source)
    source["nested"]["rank"] = 8
    assert decision.metadata["nested"]["rank"] == 7
    assert "api_key" not in decision.metadata
    with pytest.raises(TypeError):
        decision.metadata["nested"]["rank"] = 9
    with pytest.raises(ValueError):
        TraversalDecision(reason="private error detail")


def test_frontier_rejects_oversized_identity_before_normalizing_or_retaining():
    frontier = Frontier(replace(plan(), limits=CrawlLimits(max_url_bytes=20)))
    with pytest.raises(ParentStop, match="budget_exhausted"):
        frontier.admit(entry("https://example.com/" + "x" * 21))
    assert frontier.snapshot().queued == ()


def test_explicit_seed_cannot_resolve_against_an_oversized_request_base(monkeypatch):
    from tldw_Server_API.app.core.Web_Scraping.crawl import frontier as module

    resolved = replace(
        plan(), request=CrawlRequest("https://example.com/" + "x" * 50), limits=CrawlLimits(max_url_bytes=40)
    )
    calls = []
    original = module.normalize_reference

    def normalize(url, base, plan):
        calls.append(base)
        return original(url, base, plan)

    monkeypatch.setattr(module, "normalize_reference", normalize)
    frontier = Frontier(resolved)
    with pytest.raises(ParentStop, match="budget_exhausted"):
        frontier.admit(entry("https://e.test/x"))
    assert calls == []


@given(st.lists(st.integers(min_value=0, max_value=30), min_size=2, max_size=15, unique=True))
def test_legacy_url_level_uses_existing_set_pop_traversal(numbers):
    frontier = Frontier(plan("legacy_url_level"))
    pending = set()
    for sequence, number in enumerate(numbers):
        url = f"https://example.com/{number}"
        frontier.admit(entry(url, sequence))
        pending.add(url)
    expected = [pending.pop() for _ in numbers]
    assert [frontier.take().url for _ in numbers] == expected


@pytest.mark.parametrize("allowance", [48, 49])
def test_seed_identity_admits_unicode_query_before_normalized_allocation(monkeypatch, allowance):
    from tldw_Server_API.app.core.Web_Scraping import url_utils

    original, encoded = url_utils.urlencode, []

    def encode(*args, **kwargs):
        encoded.append(True)
        assert allowance == 49, "over-limit seed identity reached allocating urlencode"
        return original(*args, **kwargs)

    monkeypatch.setattr(url_utils, "urlencode", encode)
    frontier = Frontier(
        replace(plan(), request=CrawlRequest("https://e.test/a"), limits=CrawlLimits(max_url_bytes=allowance))
    )
    seed = entry("https://e.test/x?q=" + "\u00e9" * 5)
    if allowance == 48:
        with pytest.raises(ParentStop, match="budget_exhausted"):
            frontier.admit(seed)
        assert frontier.snapshot().queued == () and not encoded
    else:
        assert frontier.admit(seed)
        assert not frontier.admit(entry("https://e.test/x?q=" + "%C3%A9" * 5, 1))
        assert encoded
