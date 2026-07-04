# Web Scraping Architecture

## Overview
The Web_Scraping module powers article and site content extraction, as well as the guided WebSearch workflow. It consists of:

- Enhanced asynchronous pipeline with job queue, rate limiting, session/cookie management, and multiple extraction strategies (`enhanced_web_scraping.py`).
- Legacy utilities maintained for compatibility and offline/synchronous flows (`Article_Extractor_Lib.py`).
- Search orchestration and relevance evaluation (`WebSearch_APIs.py`).

This document explains the production pipeline, fallbacks, configuration, and operational guidance.

## Enhanced Pipeline
- Rate limiting across second/minute/hour windows to avoid throttling.
- Priority job queue with retry/backoff and resumability hooks.
- Session and cookie management (per-domain sessions with user-agent and header scoping; supports Playwright-style cookies and plain name/value cookies).
- Deduplication via normalized content hashing.
- Multiple extraction methods:
  - `trafilatura` (fast, robust for static pages)
  - `playwright` (JavaScript-rendered sites; guarded fallback to `trafilatura` if Playwright isn’t initialized)
  - `beautifulsoup` (lightweight HTML parsing when appropriate)

### Crawling Strategy (Best-First)

Recursive crawling uses a best-first priority queue with a normalized composite score. The pipeline:

- Normalizes URLs and removes tracking params for robust deduplication.
- Applies a FilterChain (domain allow/deny, content type, and default URL patterns).
- Optionally enforces robots.txt with a per-domain cache (honors centralized egress guard).
- Scores each candidate using PathDepth, ContentType, Freshness, and optional Keyword/Domain scorers.
- Enqueues candidates above a configurable threshold until `max_pages` or `max_depth` is reached.

Result metadata provides `depth`, `parent_url`, and `score`; the service persists `crawl_depth`, `crawl_parent_url`, and `crawl_score` into Media DB safe metadata.

For configuration and request overrides, see `Docs/Product/Completed/WebCrawl_Priority_BFS.md`.

Start/stop lifecycle initializes Playwright if available; absence of Playwright no longer breaks scraping-`playwright` calls transparently fall back to `trafilatura`.

## Legacy Helpers and Fallbacks
Some flows still need synchronous execution (e.g., sitemap and URL-level crawling from worker threads). To prevent async/sync mixups:

- `scrape_article_blocking(url)` fetches with `requests` and extracts via `trafilatura`, used by:
  - `scrape_from_sitemap`
  - `scrape_by_url_level`
  - `scrape_from_filtered_sitemap`

- `async_scrape_and_no_summarize_then_ingest(...)` is the async-safe variant. The old `scrape_and_no_summarize_then_ingest` is a thin sync wrapper and must not be called from async contexts.

These changes remove nested event loop errors and eliminate returning coroutines from sync code.

## Configuration
Configuration is loaded via `load_and_log_configs()` and normalized before use.

- `web_scraper_retry_count` - integer retry attempts.
- `web_scraper_retry_timeout` - seconds per navigation; converted to milliseconds for Playwright.
- `web_scraper_stealth_playwright` - boolean; string values like "true", "1", "yes" are accepted.

Stealth waits are configurable via `STEALTH_WAIT_MS` if present; otherwise default 5000 ms.

## WebSearch Orchestration
The WebSearch pipeline supports subquery generation, filtering, and aggregation. To remove interactive code from runtime:

- The module-level test/demos were moved to the test suite).
- `review_and_select_results` now accepts an optional selector callback. If none is provided, no interactive prompts are used and all results are forwarded.

## Operational Guidance
- Production should prefer the enhanced scraper. If Playwright is unavailable, the system gracefully falls back to static extraction.
- Rate limits and concurrency are tunable in config; adjust for resource-constrained environments.
- Cookies can be provided as Playwright-style dicts ({name, value, ...}) or as plain mappings.
- Use the article extraction benchmark (Docs/Evals/WebScraping_Article_Benchmark.md) to quantify changes to extraction logic.
- Keep `include_external=false` unless you need cross-domain exploration. External crawling increases latency and the likelihood of anti-bot challenges.
- Consider a modest `web_crawl_score_threshold` (0.2-0.4) to emphasize likely content pages; enable the keyword scorer when targeting topic-specific sections.

## Testing
Key regression tests cover:
- Sitemap and URL-level flows using blocking extraction helpers.
- Non-interactive selection in WebSearch.
- Playwright guard fallback to `trafilatura`.
- Cookie injection using name/value dicts.

### Phase 1 Contracts

Phase 1 adds internal Web_Scraping contracts under `tldw_Server_API/app/core/Web_Scraping/contracts/` and compatibility tests for legacy wrapper imports and dict-shaped results. These contracts are additive, stdlib-only, and not wired into runtime behavior yet; later phases should depend on these contracts before moving behavior out of `Article_Extractor_Lib.py`, `enhanced_web_scraping.py`, or `WebSearch_APIs.py`.

Implementation plan: `Docs/superpowers/plans/2026-07-04-web-scraping-phase-1-contracts-compatibility.md`

## References
- `Docs/Design/WebScraping_Refactor_Import_Inventory.md` records the current compatibility import surface for the modular refactor. Update it with `Helper_Scripts/web_scraping_refactor_inventory.py` whenever Web_Scraping compatibility imports are added, removed, or migrated.
- https://github.com/scrapinghub/article-extraction-benchmark
- https://github.com/D4Vinci/Scrapling
- https://github.com/rmusser01/nicar-2025-scraping
- https://www.diffordsguide.com/
- https://github.com/ulixee/hero
- https://github.com/devflowinc/firecrawl-simple
