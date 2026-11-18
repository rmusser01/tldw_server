"""Exercise server-owned progress artifacts through the actual recursive service."""

import json
from pathlib import Path

import pytest

from tldw_Server_API.app.core.Web_Scraping.enhanced_web_scraping import JobPriority
from tldw_Server_API.app.services.enhanced_web_scraping_service import WebScrapingService
from tldw_Server_API.tests.Web_Scraping.phase5_fakes import make_harness

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def test_service_saved_observation_is_redacted_loadable_and_owner_bound(monkeypatch, tmp_path):
    url = "https://user:password-sentinel@example.com/blog/article?token=query-sentinel"
    harness = make_harness(profile="enhanced_recursive", outcomes=[{"url": url, "content": "Kept article"}])
    monkeypatch.chdir(tmp_path)
    with harness.boundaries():
        service = WebScrapingService()
        service.scraper = harness.scraper
        harness.scraper._progress["service"] = {
            "owner_user_id": "9",
            "extra": {"label": "kept", "custom_headers": {"Authorization": "header-sentinel"}},
        }
        try:
            result = await service._scrape_recursive(
                url,
                1,
                0,
                False,
                None,
                None,
                None,
                "",
                None,
                0.7,
                None,
                JobPriority.NORMAL,
                None,
                None,
                user_id=9,
                task_id="service",
            )
            path = Path(result["progress_file"])
            encoded = path.read_text()
            assert not any(secret in encoded for secret in ("password-sentinel", "query-sentinel", "header-sentinel"))
            await harness.scraper.load_progress("loaded", path)
            saved = harness.scraper.get_progress("loaded", owner_user_id="9")
            assert saved == json.loads(encoded) and saved["extra"]["label"] == "kept"
            assert (saved["pages_scraped"], saved["crawl_outcome"]["state"], saved["owner_user_id"]) == (
                1,
                "completed",
                "9",
            )
            assert harness.scraper.get_progress("loaded", owner_user_id="10") == {}
            assert result["articles"][0]["content"] == "Kept article"
        finally:
            await harness.scraper.stop()
