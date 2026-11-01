"""Discovery must charge encounters and derived bytes before retaining links."""

from dataclasses import replace

import pytest
from hypothesis import given
from hypothesis import strategies as st

from tldw_Server_API.app.core.Web_Scraping.crawl.discovery import discover
from tldw_Server_API.app.core.Web_Scraping.crawl.models import CrawlHooks, CrawlLimits
from tldw_Server_API.app.core.Web_Scraping.orchestration.crawl_context import CrawlContext
from tldw_Server_API.app.core.Web_Scraping.runtime.controls import ParentStop
from tldw_Server_API.tests.Web_Scraping.test_phase5_frontier import BASE, plan

pytestmark = pytest.mark.unit


def run(document, *, profile="enhanced_recursive", limits=None, request=BASE, final=BASE, hooks=None, **options):
    resolved = replace(plan(profile, **options), limits=limits or CrawlLimits())
    context = CrawlContext(resolved.limits, clock=lambda: 0)
    result = discover(document, request_url=request, final_url=final, plan=resolved, controls=context, hooks=hooks)
    return result, context.snapshot()


def test_duplicates_are_all_charged_before_deduplication():
    result, usage = run('<a href="/x">one</a>' * 3)
    assert usage.discovered_candidates == 3
    assert [item.url for item in result.links] == ["https://example.com/x"] * 3
    assert usage.discovered_url_bytes == 2 * len(BASE.encode()) + 3 * (len("https://example.com/x") + len(BASE))
    assert usage.transfer_bytes == 0


def test_duplicate_candidate_overflow_is_explicit():
    with pytest.raises(ParentStop, match="budget_exhausted"):
        run('<a href="/x">one</a>' * 4, limits=CrawlLimits(discovered_candidates=3))


def test_derived_memory_admission_checks_long_base_before_collection_growth():
    base = "https://example.com/" + "x" * 32 + "/"
    document = "".join(f'<a href="p{i}">x</a>' for i in range(16))
    with pytest.raises(ParentStop, match="budget_exhausted"):
        run(document, request=base, final=base, limits=CrawlLimits(max_url_bytes=100, discovered_url_bytes=400))


def test_oversized_derived_temporary_is_rejected_before_resolution(monkeypatch):
    from tldw_Server_API.app.core.Web_Scraping.crawl import discovery

    original = discovery.urljoin
    resolved = []

    def observe(base, reference):
        value = original(base, reference)
        resolved.append(value)
        return value

    monkeypatch.setattr(discovery, "urljoin", observe)
    with pytest.raises(ParentStop, match="budget_exhausted"):
        run(
            '<a href="' + "x" * 30 + '">x</a>',
            request="https://e.test/a",
            final="https://e.test/a",
            limits=CrawlLimits(max_url_bytes=40),
        )
    assert all(len(value.encode()) <= 40 for value in resolved)


@pytest.mark.parametrize("extra", [0, 1])
def test_collective_byte_boundary_accepts_exact_and_rejects_one_over(extra):
    # request/final 2*31; target /x is 21, parent 31: total 114.
    request = "https://example.com/blog/start"
    cost = 2 * len(request) + len("https://example.com/x") + len(request)
    if extra:
        with pytest.raises(ParentStop, match="budget_exhausted"):
            run('<a href="/x">x</a>', limits=CrawlLimits(discovered_url_bytes=cost - 1))
    else:
        result, usage = run('<a href="/x">x</a>', limits=CrawlLimits(discovered_url_bytes=cost))
        assert len(result.links) == 1 and usage.discovered_url_bytes == cost


@pytest.mark.parametrize("kind", ["raw", "base", "target", "utf8"])
def test_individual_url_limit_covers_inputs_and_derived_targets(kind):
    kwargs = {"request": "https://e.test/a", "final": "https://e.test/a"}
    document = '<a href="/x">x</a>'
    if kind == "raw":
        document = '<a href="' + "x" * 41 + '">x</a>'
    elif kind == "base":
        document = '<base href="' + "x" * 41 + '">' + document
    elif kind == "target":
        document = '<a href="' + "x" * 30 + '">x</a>'
    else:
        document = '<a href="/' + "\u00e9" * 15 + '">x</a>'
    with pytest.raises(ParentStop, match="budget_exhausted"):
        run(document, limits=CrawlLimits(max_url_bytes=40), **kwargs)


@pytest.mark.parametrize(
    "profile,target",
    [
        ("legacy_recursive", "https://example.com/assets/child"),
        ("enhanced_recursive", "https://example.com/blog/child"),
    ],
)
def test_final_and_request_locations_keep_profile_resolution_bases(profile, target):
    result, _ = run(
        '<base href="/assets/"><a href="child">x</a>', profile=profile, final="https://example.com/redirect/end"
    )
    assert result.request_url == BASE
    assert result.final_url == "https://example.com/redirect/end"
    assert result.links[0].url == target
    assert result.links[0].parent_url == BASE


def test_malformed_references_count_but_are_not_retained():
    result, usage = run('<a href="http://[bad">bad</a><a href="/ok">ok</a>')
    assert usage.discovered_candidates == 2
    assert [link.url for link in result.links] == ["https://example.com/ok"]


@pytest.mark.parametrize("profile,plain_count", [("legacy_sitemap", 0), ("enhanced_sitemap", 2)])
def test_sitemap_namespace_contract(profile, plain_count):
    plain = "<urlset><url><loc>https://example.com/a</loc></url><url><loc>https://example.com/b</loc></url></urlset>"
    result, _ = run(plain, profile=profile)
    assert len(result.links) == plain_count
    namespaced = plain.replace("<urlset>", '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">')
    result, usage = run(namespaced, profile=profile)
    assert len(result.links) == 2 and usage.discovered_candidates == 2


def test_sitemap_live_filter_sees_stripped_loc_before_truthy_max_urls():
    seen = []

    def accept(url):
        seen.append(url)
        return not url.endswith("skip")

    document = (
        "<urlset>"
        + "".join(f"<url><loc> https://example.com/{item} </loc></url>" for item in ("skip", "a", "b"))
        + "</urlset>"
    )
    result, usage = run(document, profile="enhanced_sitemap", max_urls=1, hooks=CrawlHooks(url_filter=accept))
    assert seen == ["https://example.com/skip", "https://example.com/a"]
    assert [link.url for link in result.links] == ["https://example.com/a"]
    assert usage.discovered_candidates == 2


@given(st.lists(st.integers(min_value=0, max_value=20), max_size=30))
def test_normalization_keeps_encounter_order_and_accounting(numbers):
    document = "".join(f'<a href="/p/{number}?utm_source=x#f">x</a>' for number in numbers)
    result, usage = run(document)
    assert [link.url for link in result.links] == [f"https://example.com/p/{number}" for number in numbers]
    assert usage.discovered_candidates == len(numbers)
