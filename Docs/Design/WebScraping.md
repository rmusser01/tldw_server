# Web Scraping Architecture

## Overview

`Web_Scraping` provides three related but separate capabilities:

- governed single-page article acquisition and extraction;
- enhanced crawl and job-queue scraping;
- multi-provider web search and result aggregation.

Phase 4 made the single-page path canonical without removing the Phase 3
pre-scrape analyzer or the legacy compatibility surface. Enhanced crawling and
web search remain distinct because their retry, queue, scoring, and provider
contracts differ from single-page article orchestration.

## Package Ownership

| Package or module | Responsibility |
| --- | --- |
| `content` | HTML-to-Markdown conversion and content metadata normalization. |
| `selectors` | Canonical selector models, catalog loading, and selector resolution. |
| `safe_regex.py` | Bounded regular-expression execution and stable regex failures. |
| `extraction` | Extraction strategies, strategy ordering, caches, traces, and normalized results. |
| `orchestration` | Immutable article plans, policy admission, optional preflight, HTTP/browser acquisition, extraction offload, and public article results. |
| `runtime` | Shared request, timeout, fetch, browser, and policy protocols used by orchestration. |
| `preflight` | Governed advisory analyzers, scoring, recommendations, and advice precedence. |
| `enhanced_web_scraping.py` | Crawl-oriented sessions, queues, retries, scoring, and deduplication. |
| `WebSearch_APIs.py` | Search-provider dispatch, normalization, review, relevance analysis, and aggregation. |
| `Article_Extractor_Lib.py` | Compatibility exports plus crawl, sitemap, batching, ingestion, and other responsibilities deferred to later phases. |

Internal consumers should import moved single-page responsibilities from
`content`, `extraction`, or `orchestration`. Legacy imports remain supported for
external callers, compatibility tests, and responsibilities that have not yet
moved.

## Integration Surfaces

Application code should use the canonical Python facades documented in
`tldw_Server_API/app/core/Web_Scraping/README.md`. The HTTP integration points
are:

- `POST /api/v1/research/websearch` for provider search, optional subqueries,
  relevance analysis, and aggregation;
- `/api/v1/web-scraping/status`, `/api/v1/web-scraping/job/{job_id}`,
  `/api/v1/web-scraping/service/*`, `/api/v1/web-scraping/progress/*`,
  `/api/v1/web-scraping/cookies/*`, and
  `/api/v1/web-scraping/duplicates/check` for enhanced-scraper management.

The management router is registered under the `web-scraping` API route key and
can be disabled through the normal route policy. Callers must handle a disabled
route as an unavailable capability rather than importing endpoint internals.

## Approved Phase 4 Behavior Changes

Phase 4 permits only the following runtime behavior changes from the approved
design:

1. With `strategy_order=None`, regex is bounded, non-terminal enrichment. Regex
   matches may enrich a later success or the final failure, but cannot make
   default article extraction successful by themselves. Any explicit strategy
   order preserves ordered first-success behavior, including terminal regex
   extraction.
2. Caller cancellation is re-raised instead of becoming fallback, retry, or a
   failure dictionary.
3. Moved synchronous scrape entry points reject active-event-loop calls before
   policy, configuration, metrics, browser, or network side effects.
4. Invalid, oversized, or timed-out generated and configured regexes return
   stable sanitized failures under bounded execution.
5. The individual-URL service caller uses the supported `system_message`
   keyword instead of `system_prompt`.
6. The legacy raw-browser synchronous path performs governed admission before
   network access while preserving its distinct result shape.
7. Public boundaries replace raw provider, regex, selector, and transport
   exception text with deterministic sanitized failure codes.
8. Extraction submission uses bounded admission, the existing worker setting,
   a 64-worker ceiling, and a conservative default rather than an unbounded
   executor queue.
9. Direct async Playwright acquisition installs target, redirect, subresource,
   service-worker, and WebSocket egress controls before navigation.
10. Moved observability removes full URLs, query strings, `url`, `base_url`, and
    raw-error fields while retaining existing metric names and non-sensitive,
    low-cardinality labels.
11. Direct article acquisition bounds HTTP response bodies, browser transfer
    totals, and rendered HTML and reports `response_too_large` when a limit is
    exceeded.

The `content` and `selectors` packages are compatibility-preserving ownership
moves, not additional semantic changes; existing content envelopes, selector
DSL behavior, caches, and compatibility exports remain intact except for the
approved bounded-regex failures in selector transforms. Single-page article
paths are canonical through `orchestration.scrape_article`,
`scrape_article_blocking`, and `scrape_article_sync`. Collections, Evaluations,
RAG, Watchlists, Workflows, WebSearch, and web-scraping services use canonical
facades where the moved responsibility is available; mixed and deferred
consumers retain safe legacy imports. Legacy imports remain supported.
The Phase 3 preflight remains integrated, optional, advisory, and fail-open for
extraction after successful primary admission; explicit caller choices still
take precedence. All behavior outside this allowlist remains compatible.

## Single-Page Article Flow

`orchestration.scrape_article` is the canonical async entry point:

1. Load configuration and snapshot an immutable `ArticlePlan`.
2. Evaluate the target through the shared outbound-policy boundary.
3. Stop before analyzer or acquisition work when target admission is denied or
   fails.
4. If enabled, run the Phase 3 pre-scrape analyzer in one governed execution
   context.
5. Apply successful analyzer advice only to automatic choices. Explicit route
   or configuration choices take precedence.
6. Attempt bounded lightweight acquisition unless Playwright was selected.
7. Re-evaluate egress for redirects and each later network destination; target
   admission and preflight advice are not reusable network authorization.
8. Use guarded browser fallback only when routing and compatibility rules allow
   it.
9. Enforce browser-transfer and rendered-HTML limits before extraction.
10. Offload synchronous extraction through the bounded extraction executor.
11. Attach successful public preflight metadata only when configured and return
    the legacy dictionary result shape.

Optional preflight failures and timeouts are fail-open for extraction and retain
the configured route. Caller cancellation always propagates. The analyzer
results remain attached under `preflight_analysis` only when
`web_scraper_preflight_include_results` is enabled and the overall preflight
result is successful.

Phase 3 also preserves the legacy external-tool default until Phase 7. When
`web_scraper_preflight_enable_external_tools` is absent, an installed
`wafw00f` remains enabled and its first use emits one process-level
compatibility warning and metric. Explicit true or false values are
authoritative; malformed explicit values disable the tool and emit a sanitized
warning.

`scrape_article_blocking` provides the synchronous article result. The legacy
`scrape_article_sync` entry point preserves its raw-browser HTML result shape.
Both reject calls from a thread with an active event loop before configuration,
metrics, browser startup, or network side effects.

## Acquisition Security

Primary target admission uses `preflight.evaluate_target`; actual HTTP and
browser dispatches use fresh egress decisions. HTTP redirects retain the shared
HTTP client's resolution and credential-stripping controls. The guarded article
browser installs HTTP and WebSocket interception before navigation, blocks
service workers, checks each destination, and fails closed when required
interception or transfer accounting is unavailable.

Browser route validation is URL-level enforcement. It does not pin Chromium's
DNS resolution to an address approved earlier, so it must not be described as
resolved-IP pinning. Deployments that require transport-level pinning should use
a pin-capable HTTP transport and disable Playwright selection and fallback.

The direct Playwright compatibility profile intentionally supports the effective
user agent, caller-provided Playwright cookie dictionaries, retry and timeout
settings, optional stealth behavior, and the established headless viewport.
Navigation first waits for `domcontentloaded`, followed by the configured stealth
delay when stealth is enabled or a `networkidle` wait otherwise.
Plan cookies, extra headers, and proxies remain lightweight-path inputs in Phase
4. Enabling them for browser contexts requires a separate credential and
cross-origin security review.

## Limits And Failures

Each article request snapshots two byte limits:

- `web_scraper_max_article_bytes`: 16,777,216 bytes by default for lightweight
  response bodies and rendered HTML passed to extraction.
- `web_scraper_max_browser_transfer_bytes`: 67,108,864 bytes by default for
  aggregate encoded browser HTTP and WebSocket transfer.

Only positive integer values up to 1 GiB are accepted; invalid values use the
defaults. These Phase 4 limits apply to the canonical async, blocking, and raw
browser single-page entry points. They do not change enhanced-scraper or
crawl-bound acquisition.

Faults owned by article orchestration use stable `error` codes:
`policy_error`, `fetch_error`, `browser_error`, `response_too_large`, and
`extraction_error`. Regex, selector, and provider components have their own
stable scoped codes, including `regex_invalid`, `regex_too_large`,
`regex_timeout`, `selector_invalid`, and `provider_error`; those codes are not a
promise that every unsuccessful article dictionary has an `error` field.

Deliberate policy denials retain compatibility payloads. Robots denials use
`error="Blocked by outbound policy"`; other egress denials use
`error="Egress denied: <reason>"` and include bounded `policy_*` fields. The
blocking compatibility path may use `error="Outbound policy evaluation failed"`
for a policy-check failure. An ordinary no-content result instead reports
`extraction_successful=False` and may have no `error` field at all.

Logs and metrics sanitize URLs and errors to avoid credentials, query strings,
raw failures, and high-cardinality labels. Metrics use bounded stage, backend,
outcome, and failure-code labels without hostnames; logs may include only a
bounded sanitized hostname.

## Executor Lifecycle

Synchronous extraction runs in generation-owned bounded executors. Reload and
shutdown never forcibly cancel admitted synchronous futures. Reload closes
admission to the old generation, drains its admitted work with
`cancel_futures=False`, opens a replacement generation, and resumes waiters
against that replacement. Shutdown rejects waiting and new submissions, enters
the terminal shut-down state, and drains admitted work with
`cancel_futures=False` without opening a replacement. Forked children reset
inherited executor state before accepting work. Lifecycle helpers live in
`orchestration.executor`; application shutdown and tests must use those helpers
rather than closing executor internals directly.

## Enhanced Crawling And Web Search

The enhanced scraper retains rate limiting, prioritized jobs, cookie/session
management, content deduplication, and best-first crawl scoring. Crawl discovery,
sitemap processing, recursive traversal, crawl budgets, progress, and queue/job
state are deferred to Phase 5.

`WebSearch_APIs.py` retains provider adapters, subquery generation, normalized
results, optional callback-based review, relevance analysis, and final
aggregation. Provider calls use raw egress policy without synthesizing robots
checks for provider API endpoints.

### Web Search Provider Configuration

Provider settings live in the `[Search-Engines]` section of
`tldw_Server_API/Config_Files/config.txt` and are normalized under
`load_and_log_configs()["search_engines"]`. Shared settings include
`search_provider_default`, query/result languages, result limits, subquery and
rerank controls, and relevance/final-answer LLM choices. Provider settings
include Google API key and engine ID, Brave keys and country, Kagi, Tavily,
Serper, Exa, Firecrawl, and Yandex credentials, plus the Searx endpoint. A
provider that does not require credentials, such as DuckDuckGo, still uses the
same dispatch and outbound-policy boundary.

To add or extend a provider:

1. Add its `[Search-Engines]` settings and normalized loader entries when the
   provider needs configuration.
2. Implement `search_web_<provider>` and `parse_<provider>_results`, returning
   the existing normalized result shape.
3. Immediately before every provider network request, call
   `_enforce_provider_outbound_policy(url, source="websearch_<provider>")`.
   This is required; it delegates to the shared outbound-policy adapter with
   `respect_robots=False` for provider APIs.
4. Register the provider in `perform_websearch`,
   `process_web_search_results`, and `SUPPORTED_WEBSEARCH_ENGINES`.
5. Add provider, parser, endpoint-routing, policy-denial, and sanitization tests.

### Enhanced Scraper Extension Point

Add crawl-specific acquisition, queue, retry, scoring, cookie, or deduplication
behavior to `EnhancedWebScraper`, expose the operation through
`WebScrapingService`, and add a management endpoint only when remote control is
required. Keep service lifecycle and job state in those owners. Do not bypass
canonical target admission or move canonical single-page behavior into the
enhanced scraper before the Phase 5 migration design.

## Compatibility And Follow-On Work

`Article_Extractor_Lib.py` remains importable and re-exports canonical Phase 4
APIs. It also owns deferred crawl, sitemap, bookmark, source-file, ingestion,
batching, and progress behavior until those responsibilities move. Do not remove
these exports or migrate mixed consumers solely to reduce inventory counts.

After Phase 4, Phase 5 moves crawl discovery, sitemap processing, recursive
traversal, budgets, cancellation, progress, and queue/job state into explicit
`crawl` and `jobs` packages that call the canonical article orchestrator for
each page. Phase 6 WebSearch migration is explicitly deferred: it will move
WebSearch workflows, provider adapters, and provider result parsers. Phase 7
compatibility-wrapper and proven dead-code removal is also deferred until its
inventory-backed migration and deprecation gates are satisfied.

## Inventory And Verification

Regenerate compatibility inventory whenever imports change:

```bash
python Helper_Scripts/web_scraping_refactor_inventory.py \
  --root . \
  --json Docs/Design/web_scraping_refactor_import_inventory.json \
  --markdown Docs/Design/WebScraping_Refactor_Import_Inventory.md
python -m pytest -q \
  tldw_Server_API/tests/WebScraping/test_refactor_import_inventory.py
```

Focused coverage is organized as follows:

- extraction contracts and ordering:
  `tldw_Server_API/tests/Web_Scraping/test_phase4_extraction_*.py`;
- article planning, orchestration, browser, compatibility, and executor:
  `tldw_Server_API/tests/Web_Scraping/test_phase4_article_*.py` and
  `tldw_Server_API/tests/Web_Scraping/test_phase4_extraction_executor.py`;
- Phase 3 preflight compatibility and governed analyzers:
  `tldw_Server_API/tests/Web_Scraping/test_phase3_*.py` and
  `tldw_Server_API/tests/WebScraping/integration/test_phase3_preflight_browser_smoke.py`;
- enhanced scraping and management endpoints:
  `tldw_Server_API/tests/Web_Scraping/test_enhanced_web_scraping_guards.py` and
  `tldw_Server_API/tests/WebScraping/integration/test_web_scraping_endpoints.py`;
- provider and endpoint behavior: `tldw_Server_API/tests/WebSearch/` and
  `tldw_Server_API/tests/Web_Scraping/test_websearch_*.py`;
- import compatibility:
  `tldw_Server_API/tests/WebScraping/test_refactor_import_inventory.py`.

Broad regression coverage lives under `tldw_Server_API/tests/Web_Scraping` and
`tldw_Server_API/tests/WebScraping`. Cross-consumer certification includes
Collections, Evaluations, RAG, Watchlists, Workflows, WebSearch, and scraping
services.

## References

- `Docs/Design/WebScraping_Refactor_Import_Inventory.md`
- `Docs/superpowers/specs/2026-07-26-web-scraping-phase-4-extraction-article-orchestration-design.md`
- `Docs/superpowers/plans/2026-07-27-web-scraping-phase-4-extraction-article-orchestration.md`
- `Docs/Product/Completed/WebCrawl_Priority_BFS.md`
- `Docs/Evals/WebScraping_Article_Benchmark.md`
