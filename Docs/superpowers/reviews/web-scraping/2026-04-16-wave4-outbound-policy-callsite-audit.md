# Wave 4 Outbound Policy Call-Site Audit

Date: 2026-04-16

## Scope

Wave 4 was bounded to scrape and websearch data-plane callers that previously performed direct raw egress checks or branch-local robots handling.

The migration target was the shared helper layer in:

- `tldw_Server_API/app/core/Web_Scraping/outbound_policy.py`

That helper now exposes:

- `decide_web_outbound_policy()` for async scrape-style entry points and recursive discovery gates
- `decide_web_outbound_policy_sync()` for provider-style raw egress checks and legacy sync helpers

## In-Scope Call Sites Before Migration

### `Article_Extractor_Lib.py`

- `scrape_article()`
  - direct `evaluate_url_policy(url)` pre-fetch gate
  - branch-local async robots check via `is_allowed_by_robots_async()`
- `scrape_article_blocking()`
  - direct `evaluate_url_policy(url)` pre-fetch gate
- `scrape_from_sitemap()`
  - direct `evaluate_url_policy(sitemap_url)` pre-fetch gate
- `collect_internal_links()`
  - direct `evaluate_url_policy(base_url)` pre-fetch gate

### `enhanced_web_scraping.py`

- `EnhancedWebScraper.scrape_article()`
  - direct `evaluate_url_policy(url)` pre-fetch gate
  - branch-local async robots check before fetch/navigation
- `EnhancedWebScraper.scrape_sitemap()`
  - direct `evaluate_url_policy(sitemap_url)` pre-fetch gate
- `EnhancedWebScraper.recursive_scrape()`
  - direct `evaluate_url_policy(base_url)` pre-fetch gate
  - branch-local recursive candidate robots gating via `RobotsFilter.allowed()`

### `WebSearch_APIs.py`

- Provider and provider-like raw egress checks:
  - Baidu stub
  - Brave
  - DuckDuckGo HTML
  - Google
  - Kagi
  - Searx
  - Serper
  - Tavily
  - Exa
  - Firecrawl
  - 4chan catalog fetch
  - 4chan archive fetch
  - 4chan archived thread fetch
  - Yandex stub

## Post-Migration Routing

### Async scrape-style callers

These now route through `decide_web_outbound_policy()`:

- `Article_Extractor_Lib.scrape_article()`
- `EnhancedWebScraper.scrape_article()`
- `EnhancedWebScraper.recursive_scrape()` candidate discovery gates

Notes:

- Strict mode blocks scrape-style requests when robots fetches are unreachable.
- Recursive crawl candidate checks reuse the existing `RobotsFilter` instance by passing it into the shared helper, so cache behavior stays aligned.

### Sync/raw egress callers

These now route through `decide_web_outbound_policy_sync()`:

- `Article_Extractor_Lib.scrape_article_blocking()`
- `Article_Extractor_Lib.scrape_from_sitemap()`
- `Article_Extractor_Lib.collect_internal_links()`
- provider and provider-like call sites in `WebSearch_APIs.py`
- `EnhancedWebScraper.scrape_sitemap()`
- `EnhancedWebScraper.recursive_scrape()` base URL guard

Notes:

- Provider API requests stay on raw egress policy only.
- Robots policy is not synthesized for provider endpoints in this wave.

## Research Agent Result

`tldw_Server_API/app/core/RAG/rag_service/research_agent.py` did not require code changes.

Reason:

- the `scrape_url` action already delegates to `Article_Extractor_Lib.scrape_article()`
- after Task 2, it inherited the shared outbound-policy block behavior without a branch-local bypass

## Remaining Direct `evaluate_url_policy()` Usage

Direct usage was removed from the in-scope scrape/websearch modules listed above.

Remaining direct usage after the migration is limited to the explicitly excluded files:

- `tldw_Server_API/app/services/document_processing_service.py`
- `tldw_Server_API/app/services/workflows_webhook_dlq_service.py`

## Explicit Exclusions

These stay out of scope for Wave 4:

- `tldw_Server_API/app/services/document_processing_service.py`
- `tldw_Server_API/app/services/workflows_webhook_dlq_service.py`

Reason:

- they do not participate in the same scrape/websearch data-plane contract
- migrating them in this wave would have expanded scope beyond the approved outbound-safety rollout
