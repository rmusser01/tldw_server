import pytest

from tldw_Server_API.tests.Web_Scraping.phase5_fakes import make_harness


@pytest.mark.integration
@pytest.mark.asyncio
async def test_best_first_metadata_exposes_relevance_and_ordering_scores():
    base = "http://example.com"
    fixture = make_harness(
        profile="enhanced_recursive",
        outcomes=[
            {"url": base, "links": ["/a"]},
            {"url": base + "/a"},
        ],
    )
    res = await fixture.run(max_pages=2, max_depth=2)
    assert len(res) == 2
    md = res[1]["metadata"]
    assert md["crawl_strategy"] == "best_first"
    assert md["score_type"] == "composite_relevance"
    assert md["ordering_score_type"] == "path_depth"
    assert isinstance(md["score"], float)
    assert isinstance(md["ordering_score"], float)
    assert md["score"] != md["ordering_score"]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_fifo_metadata_is_strategy_tagged_and_keeps_score_fields():
    base = "http://example.com/base"
    fixture = make_harness(
        profile="enhanced_recursive",
        outcomes=[
            {"url": base, "links": [base + "/a", base + "/b"]},
            {"url": base + "/a"},
            {"url": base + "/b"},
        ],
    )
    res = await fixture.run(max_pages=3, max_depth=2, crawl_strategy="default")
    assert len(res) == 3
    md = res[1]["metadata"]
    assert md["crawl_strategy"] == "default"
    assert md["score_type"] == "composite_relevance"
    assert md["ordering_score_type"] == "path_depth"
    assert isinstance(md["score"], float)
    assert isinstance(md["ordering_score"], float)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_best_first_path_depth_guard_is_relative_to_base_path():
    base = "http://example.com/docs/v1"
    child = base + "/topic"
    fixture = make_harness(
        profile="enhanced_recursive",
        outcomes=[
            {"url": base, "links": [child, child + "/deep"]},
            {"url": child},
        ],
    )
    res = await fixture.run(max_pages=5, max_depth=1)
    assert [r["url"] for r in res] == [base, child]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_best_first_strict_policy_skips_candidate_on_robots_error():
    base = "http://example.com"
    blocked = base + "/blocked"
    fixture = make_harness(
        profile="enhanced_recursive",
        outcomes=[
            {"url": base, "links": [blocked]},
        ],
    )
    fixture.denied_urls.add(blocked)
    res = await fixture.run(max_pages=5, max_depth=2)
    assert [r["url"] for r in res] == [base]
    assert any(call["url"] == blocked and call["respect_robots"] for call in fixture.policy_calls)


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("strategy", ["default", "best_first"])
async def test_phase5_failed_children_are_omitted_without_consuming_success_capacity(strategy):
    from tldw_Server_API.tests.Web_Scraping.phase5_fakes import make_harness

    root = "https://example.com/blog/start"
    harness = make_harness(
        profile="enhanced_recursive",
        outcomes=[
            {"url": root, "links": ["/blog/bad", "/blog/good"]},
            {"url": "https://example.com/blog/bad", "extraction_successful": False, "error": "fixture failure"},
            {"url": "https://example.com/blog/good", "links": ["/blog/last"]},
            {"url": "https://example.com/blog/last"},
        ],
    )
    result = await harness.run(max_pages=3, max_depth=2, crawl_strategy=strategy)
    assert [article["url"] for article in result] == [
        root,
        "https://example.com/blog/good",
        "https://example.com/blog/last",
    ]
    assert harness.attempted_urls == [
        root,
        "https://example.com/blog/bad",
        "https://example.com/blog/good",
        "https://example.com/blog/last",
    ]


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("strategy", ["default", "best_first"])
async def test_phase5_seed_relative_depth_filters_robots_and_exact_scores(strategy):
    from tldw_Server_API.tests.Web_Scraping.phase5_fakes import make_harness

    base = "https://example.com/docs/v1"
    child = f"{base}/topic"
    harness = make_harness(
        profile="enhanced_recursive",
        outcomes=[
            {
                "url": base,
                "links": [
                    child,
                    child,
                    f"{child}/deep",
                    f"{base}/blocked",
                    f"{base}/manual.pdf",
                    "https://external.example/a",
                ],
            },
            {"url": child},
        ],
    )
    harness.denied_urls.add(f"{base}/blocked")
    result = await harness.run(max_pages=8, max_depth=1, crawl_strategy=strategy)
    assert [article["url"] for article in result] == [base, child]
    assert result[1]["metadata"] == {
        "nested": {"labels": ["baseline"]},
        "depth": 1,
        "parent_url": base,
        "score": pytest.approx(5 / 6),
        "ordering_score": 1.0,
        "score_type": "composite_relevance",
        "ordering_score_type": "path_depth",
        "crawl_strategy": strategy,
    }
    candidate_calls = [call for call in harness.policy_calls if call["stage"] == "discovery"]
    assert [call["url"] for call in candidate_calls] == [child, f"{base}/blocked"]
    assert all(call["respect_robots"] is True for call in candidate_calls)
