# Web_Scraping

This package owns governed web acquisition, article extraction, crawl-oriented
scraping, and multi-provider web search.

See the [Web Scraping architecture](../../../../Docs/Design/WebScraping.md) for
detailed ownership, sequencing, security boundaries, lifecycle, and migration
scope.

## Public Entry Points

Use canonical package imports for single-page article work:

```python
from tldw_Server_API.app.core.Web_Scraping.content import ContentMetadataHandler
from tldw_Server_API.app.core.Web_Scraping.extraction import extract_article_data_from_html
from tldw_Server_API.app.core.Web_Scraping.orchestration import (
    scrape_article,
    scrape_article_blocking,
    scrape_article_sync,
)
```

- `scrape_article`: async single-page article result.
- `scrape_article_blocking`: synchronous article result for threads without an
  active event loop.
- `scrape_article_sync`: legacy synchronous raw-browser HTML result.
- `extract_article_data_from_html`: extraction when acquisition is already
  complete.
- `ContentMetadataHandler`: shared metadata normalization.

`Article_Extractor_Lib.py` remains a supported compatibility module. Do not add
new internal imports to it when a canonical owner exists.

With `strategy_order=None`, regex matches are non-terminal enrichment. They are
copied onto a later article success or final no-content result but never make
default extraction successful by themselves. An explicit strategy order keeps
ordered first-success behavior, including terminal regex success.

## HTTP Integration

- `POST /api/v1/research/websearch` runs provider search with optional subquery,
  relevance, and aggregation stages.
- `/api/v1/web-scraping/status`, `/api/v1/web-scraping/job/{job_id}`,
  `/api/v1/web-scraping/service/*`, `/api/v1/web-scraping/progress/*`,
  `/api/v1/web-scraping/cookies/*`, and
  `/api/v1/web-scraping/duplicates/check` manage the enhanced scraper.

The management router uses the `web-scraping` API route key and may be disabled
by route policy. Use `WebScrapingService` for in-process enhanced-scraper
integration; do not import endpoint functions as a service API.

## Architecture

The canonical article path is `orchestration.scrape_article`: immutable plan,
primary target admission, optional Phase 3 preflight, bounded HTTP or guarded
browser acquisition, fresh checks for later destinations, bounded extraction
offload, and the compatibility dictionary result. Enhanced crawl/job behavior
remains in `enhanced_web_scraping.py`; provider workflows remain in
`WebSearch_APIs.py`.

Preflight is advisory and fail-open for extraction. Primary policy admission is
blocking and happens before analyzer or acquisition network work. Explicit route
and configuration choices override analyzer advice. Cancellation propagates.

## Configuration

Important settings in the `[Web-Scraper]` section of
`tldw_Server_API/Config_Files/config.txt` include:

- `web_scraper_default_backend`: `auto`, `curl`, `httpx`, or `playwright`.
- `web_scraper_retry_count`: direct-browser retry count.
- `web_scraper_retry_timeout`: direct-browser timeout in seconds.
- `web_scraper_stealth_playwright`: enables optional browser stealth behavior.
- `web_scraper_preflight_analyzers`: enables the pre-scrape analyzer.
- `web_scraper_preflight_timeout_s`, `web_scraper_preflight_scan_depth`,
  `web_scraper_preflight_find_all_waf`, and
  `web_scraper_preflight_impersonate`: analyzer controls.
- `web_scraper_preflight_include_results`: attaches successful analyzer output
  as `preflight_analysis`.
- `web_scraper_playwright_no_sandbox`: controls the analyzer browser launch
  option where supported.
- `web_scraper_preflight_enable_external_tools`: controls governed external
  probes. The absent-setting Phase 3 compatibility behavior is retained until
  Phase 7.

The `[Web-Scraping]` section owns the direct article runtime limits and wait:

- `stealth_wait_ms`: delay after browser navigation when stealth is enabled.
- `web_scraper_max_article_bytes`: lightweight response and rendered-HTML
  limit, default 16 MiB.
- `web_scraper_max_browser_transfer_bytes`: aggregate browser transfer limit,
  default 64 MiB.

The single-page limits accept positive integers no greater than 1 GiB. Invalid
values use defaults. They do not alter enhanced scraper or recursive crawl
acquisition in Phase 4.

Web-search settings live in `[Search-Engines]` and are normalized under
`load_and_log_configs()["search_engines"]`. Configure
`search_provider_default`, language and result controls, and the applicable
provider key, engine ID, country, or endpoint. Common keys cover Google, Brave,
Kagi, Tavily, Serper, Exa, Firecrawl, Yandex, and Searx; DuckDuckGo does not
require credentials.

## Security Boundary

Target admission and actual network dispatch are separate decisions. HTTP
redirects and browser HTTP/WebSocket destinations receive fresh egress checks.
The guarded browser installs interception before navigation, blocks service
workers, accounts for transfer bytes, and fails closed when required controls
are unavailable.

Browser route checks validate URLs but do not pin Chromium DNS resolution.
Deployments requiring resolved-IP pinning must use a capable HTTP transport and
disable Playwright selection and fallback.

The direct-browser profile preserves the effective user agent, caller cookie
dictionaries, retries, timeout, stealth behavior, and established viewport.
Navigation first waits for `domcontentloaded`, followed by the configured stealth
delay when stealth is enabled or a `networkidle` wait otherwise.
Plan headers, plan cookies, and proxies remain lightweight-only inputs until a
separate cross-origin credential review approves browser support.

## Results And Failures

Article results retain the legacy dictionary fields such as `url`, `title`,
`author`, `date`, `content`, and `extraction_successful`. Orchestration-owned
faults use `policy_error`, `fetch_error`, `browser_error`,
`response_too_large`, or `extraction_error`. Deliberate policy denials preserve
compatibility strings such as `Blocked by outbound policy` or
`Egress denied: <reason>` plus bounded `policy_*` fields. Ordinary no-content
results may be unsuccessful without any `error` field. Optional successful
preflight output is attached under `preflight_analysis` only when configured.

Metrics use sanitized, bounded stages, backends, outcomes, and failure codes;
they have no hostname or domain labels. Logs may include a bounded sanitized
hostname. Do not add full URLs, query strings, raw errors, cookies, headers, or
unbounded metric labels.

## Compatibility And Deferred Scope

Keep legacy imports for external compatibility and for crawl, sitemap,
bookmark, source-file, ingestion, batching, progress, and queue/job behavior
that has not moved. `scrape_article_async(context, ...)` remains crawl-bound and
is not the canonical single-page facade.

Phase 5 will move crawl discovery, sitemap processing, recursive traversal,
budgets, cancellation, progress, and job state into `crawl` and `jobs` packages.
The enhanced scraper remains separate until its retry, job, and extraction
semantics have their own migration design. Phase 6 WebSearch migration is
deferred and will move WebSearch workflows, provider adapters, and provider
result parsers. Phase 7 compatibility-wrapper and proven dead-code removal is
also deferred until inventory-backed migration and deprecation gates are met.

## Extension Points

For a web-search provider, add its `[Search-Engines]` loader entries when needed,
implement `search_web_<provider>` and `parse_<provider>_results`, and register it
in provider dispatch, result processing, and `SUPPORTED_WEBSEARCH_ENGINES`.
Immediately before every provider request, call
`_enforce_provider_outbound_policy(url, source="websearch_<provider>")`; provider
APIs use the shared raw-egress decision with `respect_robots=False`. Add focused
provider, parser, endpoint, policy-denial, and sanitization tests.

For crawl-specific behavior, extend `EnhancedWebScraper`, expose the operation
through `WebScrapingService`, and add a management route only when remote control
is required. Preserve service lifecycle, queue, retry, progress, and
deduplication ownership; canonical single-page changes belong in
`orchestration`.

## Development

Regenerate the compatibility inventory after import changes:

```bash
python Helper_Scripts/web_scraping_refactor_inventory.py \
  --root . \
  --json Docs/Design/web_scraping_refactor_import_inventory.json \
  --markdown Docs/Design/WebScraping_Refactor_Import_Inventory.md
python -m pytest -q \
  tldw_Server_API/tests/WebScraping/test_refactor_import_inventory.py
```

Run both `tldw_Server_API/tests/Web_Scraping` and
`tldw_Server_API/tests/WebScraping` before changing compatibility behavior.
Focused suites include:

- `tldw_Server_API/tests/Web_Scraping/test_phase4_extraction_pipeline.py`;
- `tldw_Server_API/tests/Web_Scraping/test_phase4_article_orchestration.py`;
- `tldw_Server_API/tests/Web_Scraping/test_phase4_article_compatibility.py`;
- `tldw_Server_API/tests/Web_Scraping/test_phase3_preflight_*.py`;
- `tldw_Server_API/tests/Web_Scraping/test_enhanced_web_scraping_guards.py`;
- `tldw_Server_API/tests/WebSearch/integration/test_websearch_endpoint.py`;
- `tldw_Server_API/tests/WebSearch/integration/test_websearch_engines_endpoint.py`;
- `tldw_Server_API/tests/WebScraping/test_refactor_import_inventory.py`.
