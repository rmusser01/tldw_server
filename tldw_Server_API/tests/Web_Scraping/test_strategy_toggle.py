import pytest

from tldw_Server_API.tests.Web_Scraping.phase5_fakes import make_harness


@pytest.mark.integration
@pytest.mark.asyncio
async def test_fifo_order_when_strategy_default():
    base = "http://example.com/a"
    b1, b2 = "http://example.com/b1", "http://example.com/b2"
    fixture = make_harness(
        profile="enhanced_recursive",
        outcomes=[
            {"url": base, "links": [b1, b2]},
            {"url": b1},
            {"url": b2},
        ],
    )
    res = await fixture.run(max_pages=3, max_depth=2, crawl_strategy="default")
    assert [r["url"] for r in res] == [base, b1, b2]
