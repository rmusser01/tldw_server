import pytest

from tldw_Server_API.tests.Web_Scraping.phase5_fakes import make_harness


@pytest.mark.integration
@pytest.mark.asyncio
async def test_priority_crawl_order():
    base = "http://example.com"
    fixture = make_harness(
        profile="enhanced_recursive",
        outcomes=[
            {"url": base, "links": ["/a", "/a/b", "/a/b/c"]},
            {"url": base + "/a"},
            {"url": base + "/a/b"},
        ],
    )
    res = await fixture.run(max_pages=4, max_depth=2)
    urls = [r["url"] for r in res]
    assert urls == [base, base + "/a/b", base + "/a"]
    assert isinstance(res[1]["metadata"]["depth"], int)
    assert "parent_url" in res[1]["metadata"]
    assert "score" in res[1]["metadata"]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_max_pages_and_visited():
    base = "http://example.com"
    fixture = make_harness(
        profile="enhanced_recursive",
        outcomes=[
            {"url": base, "links": ["/x", "/x#frag", "/x?utm_source=zzz"]},
            {"url": base + "/x"},
        ],
    )
    res = await fixture.run(max_pages=2, max_depth=2)
    assert [r["url"] for r in res] == [base, base + "/x"]
    assert fixture.attempted_urls == [base, base + "/x"]
