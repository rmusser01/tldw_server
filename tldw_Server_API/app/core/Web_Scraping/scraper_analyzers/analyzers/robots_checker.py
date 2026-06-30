from __future__ import annotations

from typing import Any
from urllib.parse import urlparse, urlunparse

from tldw_Server_API.app.core.http_client import fetch as http_fetch


def check_robots_txt(url: str) -> dict[str, Any]:
    """
    Fetch and parse robots.txt to check for scraping directives.

    Returns crawl delay and whether scraping is disallowed for all user agents.
    """
    try:
        parsed_url = urlparse(url)
        robots_url = urlunparse((parsed_url.scheme, parsed_url.netloc, "robots.txt", "", "", ""))

        headers = {"User-Agent": "Mozilla/5.0 (compatible; caniscrape-bot/1.0)"}
        resp = http_fetch(method="GET", url=robots_url, timeout=10, headers=headers, allow_redirects=True)

        if resp.status_code == 200:
            if "text/html" in resp.headers.get("Content-Type", "").lower():
                return {"status": "not_found"}

            crawl_delay: float | None = None
            scraping_disallowed = False
            is_generic_agent_block = False

            for raw_line in resp.text.splitlines():
                line = raw_line.strip().lower()
                if not line or line.startswith("#"):
                    continue

                if line.startswith("user-agent:"):
                    agent = line.split(":", 1)[1].strip()
                    is_generic_agent_block = agent == "*"
                    continue

                if not is_generic_agent_block:
                    continue

                if line.startswith("disallow:"):
                    path = line.split(":", 1)[1].strip()
                    if path == "/":
                        scraping_disallowed = True
                elif line.startswith("crawl-delay:"):
                    try:
                        delay_str = line.split(":", 1)[1].strip()
                        crawl_delay = float(delay_str)
                    except (ValueError, IndexError):
                        continue

            return {
                "status": "success",
                "crawl_delay": crawl_delay,
                "scraping_disallowed": scraping_disallowed,
            }

        if 400 <= resp.status_code < 500:
            return {"status": "not_found"}

        return {"status": "error", "message": str(resp.status_code)}
    except Exception as exc:
        return {"status": "error", "message": "Robots.txt check failed."}
