# Web_Scraping Modular Refactor Design

Date: 2026-07-03
Task: TASK-12025
Status: Draft for user review

## Purpose

Refactor `tldw_Server_API.app.core.Web_Scraping` into smaller, stable modules that are easier to maintain, extend, and reason about without losing current behavior.

The refactor must preserve:

- Governed pre-scrape analyzer functionality.
- Existing public imports and dict-shaped return contracts during migration.
- Direct single-article extraction APIs used by WebSearch and other callers.
- Authenticated scraping support through cookies and browser cookie cloning.
- WebSearch provider behavior and normalized result fields.
- Current hardening guarantees for outbound policy, robots behavior, cancellation, redaction, extraction defaults, and Playwright sandbox configuration.

This design describes the target architecture and migration strategy. It does not authorize implementation yet; implementation planning should happen only after user review.

## Current Problems

The current module has three large centers of gravity:

- `Article_Extractor_Lib.py`: direct scraping, extraction strategies, regex/PII/schema/cluster/LLM enrichment, crawl helpers, sitemap helpers, bookmark/file URL parsing, compatibility helpers, and legacy wrapper behavior.
- `enhanced_web_scraping.py`: runtime scraping, queueing, rate limiting, cookies, browser lifecycle, deduplication, extraction backend choice, crawl behavior, and management support.
- `WebSearch_APIs.py`: query generation, provider adapters, provider parsers, relevance scoring, page scraping, aggregation, and older demo/test-style functions.

These files mix policy checks, runtime I/O, extraction logic, service state, and compatibility behavior. That makes changes difficult to review and increases the risk of duplicate policy, timeout, cancellation, logging, and analyzer behavior.

The recent hardening work added a governed analyzer boundary. The larger refactor should preserve and strengthen that boundary instead of folding analyzer logic back into extraction or browser code.

## Design Principles

- Keep compatibility first. Old imports stay available while internals move behind them.
- Move behavior by responsibility, not by file chunk.
- Use internal dataclasses and enums for new module boundaries; keep Pydantic at API boundaries unless an existing API schema requires it.
- Convert internal results to public dict shapes in one conversion layer.
- Keep preflight analyzer output advisory. Preflight returns signals; orchestration chooses what to do.
- Keep cancellation as an exception internally. Only job/progress views may expose cancellation as state.
- Treat policy, timeout, budget, cleanup, and redaction as cross-cutting requirements, not optional helper behavior.
- Avoid making legacy wrapper files dependency targets for new code.

## Target Package Shape

```text
tldw_Server_API/app/core/Web_Scraping/
  contracts/
    requests.py
    results.py
    statuses.py
    errors.py
    conversion.py

  config/
    settings.py

  policy/
    outbound.py
    robots.py

  runtime/
    fetch.py
    browser.py
    sessions.py
    rate_limits.py
    resource_governance.py
    timeouts.py
    cancellation.py
    observability.py

  routing/
    backend_router.py
    handlers.py
    url_utils.py
    ua_profiles.py

  content/
    metadata.py
    markdown.py
    deduplication.py

  sources/
    bookmarks.py
    csv_urls.py
    url_files.py

  preflight/
    context.py
    runner.py
    scoring.py
    recommendations.py
    analyzers/

  extraction/
    pipeline.py
    strategies/
      trafilatura_strategy.py
      playwright_strategy.py
      beautifulsoup_strategy.py
    enrichment/
      regex_entities.py
      pii.py
      jsonld.py
      schema_rules.py
      clusters.py
      llm.py

  crawl/
    plans.py
    sitemap.py
    link_discovery.py
    bfs.py
    filters.py
    scoring.py

  cookies/
    manager.py
    cloner.py
    domains.py

  search/
    orchestration.py
    query_planning.py
    relevance.py
    aggregation.py
    normalization.py

  search_providers/
    google.py
    brave.py
    duckduckgo.py
    searx.py
    serper.py
    tavily.py
    kagi.py
    exa.py
    firecrawl.py
    fourchan.py

  orchestration/
    scrape_plan.py
    backend_selection.py
    crawl_plan.py

  jobs/
    queue.py
    progress.py
    retries.py

  Article_Extractor_Lib.py
  enhanced_web_scraping.py
  WebSearch_APIs.py
```

`Article_Extractor_Lib.py`, `enhanced_web_scraping.py`, and `WebSearch_APIs.py` remain as compatibility modules while migration is in progress. New code should import the new packages, not these legacy wrapper files.

## Dependency Boundaries

Foundational modules:

- `contracts` defines internal request/result dataclasses, errors, and status enums.
- `config` normalizes web-scraping settings.
- `policy` wraps outbound policy and robots decisions.
- `runtime` owns fetch/browser/session/timeout/cancellation primitives.
- `routing`, `content`, and `sources` hold shared helpers that are currently mixed into the large legacy files.

Feature modules:

- `preflight`, `extraction`, `crawl`, `search`, `search_providers`, `cookies`, `routing`, `content`, and `sources` depend on `contracts`, `config`, and selected `policy` or `runtime` interfaces.
- `runtime` must not directly import `policy`. Runtime helpers accept policy checkers, request contexts, or route guards from callers.
- `preflight` must not perform article extraction.
- `extraction` must not know analyzer internals.
- `crawl` must not parse articles directly; it schedules scrape requests.
- `search` must not know extraction strategy internals; it asks for page content through the scrape facade.
- `orchestration` coordinates feature modules and makes backend-selection decisions.
- Services, API endpoints, and compatibility wrappers sit above orchestration.

The intended direction is:

```text
contracts/config
  -> policy and runtime interfaces
  -> routing / content / sources / preflight / extraction / crawl / cookies / search_providers
  -> orchestration / search
  -> services and compatibility wrappers
```

Lower-level modules must not import `orchestration`, services, API endpoints, or the legacy wrapper files.

## Core Data Flow

### Single-Page Scrape

1. Caller provides current public arguments or a new internal `ScrapeRequest`.
2. Compatibility wrapper or service converts public arguments into `ScrapeRequest`.
3. `orchestration.scrape_plan` normalizes URL, headers, cookies, backend preference, config, and request context.
4. `policy` checks the primary target and robots settings before network work.
5. `preflight` runs only when enabled. It uses runtime interfaces for guarded HTTP/browser work and returns analyzer results, score, and recommendations.
6. `orchestration.backend_selection` combines caller preference, config, and preflight advice to choose an extraction backend.
7. `extraction.pipeline` runs the selected strategy and optional enrichment steps.
8. `contracts.conversion` converts the internal `ExtractionResult` to the existing public dict shape.

Preflight is a standard scrape-plan phase, but config decides whether it executes.

### Crawl

1. Caller provides crawl options that convert to a `CrawlRequest`.
2. `crawl` owns sitemap parsing, link discovery, BFS scoring, page/depth caps, domain filtering, and URL prioritization.
3. A shared crawl context owns cancellation, crawl-wide budgets, session reuse, and observability metadata.
4. Each accepted URL becomes a normal `ScrapeRequest`.
5. The single-page scrape flow handles policy, preflight, extraction, cookies, and result conversion.
6. Crawl metadata such as parent URL, depth, and score is attached outside extraction.

Crawl must not duplicate extraction backend logic or analyzer logic.

### WebSearch

1. `search` owns query planning, relevance, aggregation, and normalized result assembly.
2. `search_providers` own provider-specific HTTP calls and parsers.
3. Provider calls use provider-appropriate outbound policy. Robots checks are not synthesized for provider API endpoints.
4. When WebSearch needs page content, it calls the public scrape facade or a stable orchestration API.
5. WebSearch result dict fields stay compatible during migration.

Provider modules should not import `Article_Extractor_Lib.py` directly after migration begins.

## Preflight Analyzer Preservation

The governed preflight analyzer boundary is a core part of the target design.

`preflight` owns:

- Analyzer execution context.
- Primary target checks for analyzer execution.
- Analyzer subrequest policy checks.
- Total, browser, and active-probe budgets.
- Overall and per-analyzer timeouts.
- Sanitized URL labels and redacted logs.
- External tool opt-in handling.
- Analyzer scoring and recommendations.
- Stable analyzer result keys.

`runtime` provides guarded fetch/browser primitives, but preflight controls how analyzers consume budgets and how analyzer failures are represented.

`orchestration` consumes preflight output. It may choose Playwright when JavaScript is required, keep static extraction when analyzer signals are unknown, or attach preflight metadata when configured. It must not inspect analyzer implementation details.

Existing preflight config keys remain valid:

- `web_scraper_preflight_analyzers`
- `web_scraper_preflight_timeout_s`
- `web_scraper_preflight_scan_depth`
- `web_scraper_preflight_find_all_waf`
- `web_scraper_preflight_impersonate`
- `web_scraper_preflight_include_results`
- `web_scraper_preflight_enable_external_tools`
- `web_scraper_playwright_no_sandbox`

## Contracts And Conversion

New internal dataclasses should represent:

- `ScrapeRequest`
- `ScrapeContext`
- `ExtractionResult`
- `PreflightResult`
- `CrawlRequest`
- `CrawlResult`
- `SearchRequest`
- `SearchResult`
- `RuntimeFailure`

Status enums should cover:

- `ok`
- `blocked`
- `policy_denied`
- `timeout`
- `budget_exhausted`
- `external_tool_disabled`
- `unavailable`
- `error`

Cancellation is not an ordinary status internally. It remains `asyncio.CancelledError`.

Compatibility conversion must preserve public dict keys such as:

- `url`
- `title`
- `author`
- `date`
- `content`
- `extraction_successful`
- `error`
- `backend`
- `method`
- `preflight_analysis`
- WebSearch `results`
- WebSearch `processing_error`
- Provider-normalized result fields currently consumed by endpoints and tests.

Conversion should live in one place, such as `contracts/conversion.py`, rather than being scattered across wrappers.

## Runtime And Shared Context

Runtime should provide interfaces for:

- Async guarded HTTP fetch.
- Blocking fetch when needed by compatibility wrappers.
- Browser launch and page/context lifecycle.
- Sandboxed Playwright launch arguments.
- Rate-limit enforcement primitives.
- Resource-governance hooks currently used by enhanced scraping.
- Per-request timeout helpers.
- Cancellation-safe cleanup.
- Session reuse.
- Redacted stage/outcome observability.

Shared request context should allow safe reuse of:

- Normalized target URL.
- Effective user agent and headers.
- Cookie material after validation.
- Policy decisions where cache keys include all decision-relevant options.
- Fetch response metadata where reuse is safe and does not bypass policy or privacy constraints.
- Preflight results and backend advice.

Fetch/body reuse must be conservative. It should only happen when the same URL, headers, cookies, policy mode, and caller context are compatible.

## Routing, Content, And Sources

Several helpers are currently embedded in the large files but do not belong to extraction, preflight, or search.

`routing` should own:

- Backend routing and handler resolution.
- User-agent profiles and browser-like headers.
- URL normalization helpers used by crawl, collections, dedupe, and services.

`content` should own:

- Markdown conversion.
- Content metadata extraction.
- Content deduplication and hash helpers.

`sources` should own:

- Chromium and Firefox bookmark parsing.
- CSV and file-based URL collection.
- Helpers that turn imported source lists into scrape or crawl requests.

These modules are shared utilities, not orchestration layers. They should not import services, API endpoints, or legacy wrapper modules.

## Extraction

`extraction/strategies` owns article extraction backends:

- Trafilatura.
- Playwright.
- BeautifulSoup.
- Future extractors.

`extraction/enrichment` owns optional enrichment:

- Regex entities.
- PII masking or detection.
- JSON-LD and microdata.
- Schema rules.
- Cluster extraction.
- LLM-based extraction.

The default extraction order must preserve the recent hardening decision: regex and PII detection cannot short-circuit normal article extraction unless explicitly requested by a caller.

`scrape_article_blocking` remains importable, but it must be documented as valid only outside a running event loop and must use the same policy/runtime contracts.

## Crawl And Jobs

`crawl` owns crawl-specific decisions:

- Sitemap loading.
- Link discovery.
- URL filtering.
- BFS priority scoring.
- Page and depth limits.
- External-domain inclusion.
- Crawl-wide budget and cancellation behavior.

`jobs` owns service/job concerns:

- Queue state.
- Progress snapshots.
- Retry state.
- Job status.
- Persisted job metadata if retained.

Extraction strategies should not know about job queues. Jobs schedule orchestration work; they do not perform extraction directly.

## Search And Search Providers

`search` owns workflow-level behavior:

- Query planning.
- Subquery generation.
- Optional result review hooks.
- Relevance analysis.
- Final aggregation.
- Normalized result assembly.

`search_providers` own provider details:

- API parameter construction.
- Provider outbound policy enforcement.
- HTTP calls.
- Provider-specific parsing.
- Provider-specific error normalization.

Production `test_*` and demo-style functions in `WebSearch_APIs.py` should not be preserved as public compatibility APIs unless tests show real callers. They should move to test files or be dropped during cleanup.

## Error Handling

Hardening behavior must remain intact:

- `asyncio.CancelledError` always propagates internally.
- Primary target policy denial returns a clear blocked result before network work.
- Analyzer failures remain analyzer-scoped.
- Analyzer timeout and budget exhaustion return structured analyzer statuses.
- Crawl cancellation stops scheduling new URLs and cancels pending page work.
- Runtime browser/context/session cleanup runs in `finally` paths.
- Provider errors remain provider-scoped and normalize into existing WebSearch fields.
- Compatibility wrappers convert typed internal errors to current dict-shaped responses.

Only job/progress surfaces may expose cancellation as a persisted or user-visible state.

## Observability And Redaction

Every phase should emit consistent stage/outcome metadata:

- `stage`
- `backend`
- `provider`
- `status`
- `duration_ms`
- sanitized URL label when needed
- policy mode where useful

Logs and metrics must not expose:

- Cookie values.
- API keys.
- Full secret-bearing query strings.
- Browser profile paths unless explicitly needed for local debugging and already sanitized.
- Raw page content.

Redaction tests should be added per phase as code migrates.

## Compatibility Entry Points

These entry points stay importable during migration:

- `Article_Extractor_Lib.scrape_article`
- `Article_Extractor_Lib.scrape_article_blocking`
- `Article_Extractor_Lib.extract_article_with_pipeline`
- `Article_Extractor_Lib.extract_article_data_from_html`
- `Article_Extractor_Lib.scrape_and_summarize_multiple`
- `Article_Extractor_Lib.async_scrape_and_no_summarize_then_ingest`
- `Article_Extractor_Lib.scrape_and_no_summarize_then_ingest`
- `enhanced_web_scraping.EnhancedWebScraper`
- `enhanced_web_scraping.CookieManager` until cookie management is fully moved.
- `enhanced_web_scraping.create_enhanced_scraper`
- `WebSearch_APIs.generate_and_search`
- `WebSearch_APIs.analyze_and_aggregate`
- `WebSearch_APIs.perform_websearch`
- `WebSearch_APIs.search_discussions`
- `WebSearch_APIs.review_and_select_results`
- `WebSearch_APIs.process_web_search_results`
- `WebSearch_APIs.summarize`
- Existing provider functions and parser names that tests or callers currently import.
- `Article_Extractor_Lib.convert_html_to_markdown`
- `Article_Extractor_Lib.is_content_page`
- `Article_Extractor_Lib.collect_bookmarks`
- `Article_Extractor_Lib.collect_urls_from_file`
- `Article_Extractor_Lib.ContentMetadataHandler`
- Existing enrichment helpers such as regex, JSON-LD, cluster, schema, and LLM extraction functions that tests or callers currently import.
- `enhanced_web_scraping.RateLimiter`
- `enhanced_web_scraping.ContentDeduplicator`
- `enhanced_web_scraping.ScrapingJobQueue`

Compatibility tests should lock these public contracts before moving implementation.

The current import surface extends beyond Web_Scraping tests. RAG, MCP, Workflows, Watchlists, Collections, Research providers, DB import checks, and services import these modules directly. The implementation plan must inventory those imports before moving code.

## Migration Phases

### Phase 0: Import Inventory And Guardrails

Create an import inventory and compatibility map before moving behavior.

Success criteria:

- Current imports from application code and tests are listed for `Article_Extractor_Lib.py`, `enhanced_web_scraping.py`, `WebSearch_APIs.py`, `scraper_analyzers`, `url_utils.py`, `ua_profiles.py`, `handlers.py`, `scraper_router.py`, and `scoring.py`.
- Public compatibility wrappers are selected from actual imports and tests, not guesses.
- A lightweight guardrail test or static check prevents new internal modules from importing legacy wrapper files.
- The implementation plan names which compatibility tests protect each moved responsibility.

The Phase 0 implementation should produce `Docs/Design/WebScraping_Refactor_Import_Inventory.md` and `Docs/Design/web_scraping_refactor_import_inventory.json`, with tests that fail when the inventory no longer matches current imports.

### Phase 1: Contracts And Compatibility Tests

Add internal dataclasses, status enums, conversion helpers, and contract tests for public entry points.

Success criteria:

- Existing public dict shapes are captured in tests.
- No behavior moves yet.
- Compatibility conversion is available for later phases.

### Phase 2: Runtime And Policy Boundary

Move guarded fetch, browser launch, timeout, cancellation, session, and robots/policy plumbing behind explicit interfaces.

Success criteria:

- Runtime does not import legacy wrapper files.
- Runtime does not directly import policy; policy checkers or request contexts are injected.
- Existing hardening tests still pass.

### Phase 3: Preflight Package Move

Move the governed analyzer context, runner, scoring, recommendations, and analyzer modules into `preflight`.

Success criteria:

- Analyzer result keys remain stable.
- Config keys remain stable.
- Preflight is still optional but part of the standard scrape plan.
- Analyzer policy, budget, timeout, and redaction tests pass.

### Phase 4: Extraction Package Move

Move article pipeline, extraction strategies, and enrichment helpers out of `Article_Extractor_Lib.py`.

Success criteria:

- Direct article APIs still work through wrappers.
- Regex/PII cannot short-circuit normal extraction by default.
- Blocking scrape behavior is documented and protected from event-loop misuse.

### Phase 5: Crawl And Jobs Split

Move sitemap, link discovery, BFS, filtering, crawl budgets, and job state into `crawl` and `jobs`.

Success criteria:

- Crawl uses normal scrape orchestration for each page.
- Crawl-wide cancellation and budgets are enforced.
- Queue/progress/retry state is not mixed into extraction strategies.

### Phase 6: Search Provider Split

Move provider adapters and parsers into `search_providers`; keep workflow behavior in `search`.

Success criteria:

- Provider-specific tests pass without real network.
- Secret redaction remains enforced.
- WebSearch result fields remain compatible.

### Phase 7: Thin Wrappers And Cleanup

Reduce legacy files to compatibility wrappers and remove only proven-private dead code.

Success criteria:

- New code imports new packages.
- Old imports continue to work.
- Demo/test-style functions are moved to tests or removed only when no callers depend on them.

## Test Strategy

Each phase should include focused tests before moving behavior:

- Public compatibility contract tests for old imports.
- Policy denial and robots behavior tests.
- Cancellation propagation tests.
- Timeout and cleanup tests.
- Preflight result-shape and recommendation tests.
- Extraction backend selection tests.
- Regex/PII default-order tests.
- Cookie redaction and domain matching tests.
- Crawl cap, cancellation, and metadata tests.
- Provider parser and redaction tests.
- Conversion tests from internal dataclasses to public dicts.

No phase should require real network or real browser execution for its regression suite.

## Rollout And Risk Controls

- Keep wrappers in place until all direct imports are known and migrated.
- Start with an import inventory and wrapper compatibility map.
- Move one responsibility per phase.
- Do not change public API shape and internal module boundaries in the same commit when avoidable.
- Keep config keys stable.
- Keep old docs accurate while migration is partial.
- Run focused Web_Scraping/WebScraping/WebSearch tests after every phase.
- Run Bandit on touched production paths before completing each phase.

## Open Decisions For Implementation Planning

- Whether to use one `ScrapeContext` for both single-page scraping and crawling, or a `CrawlContext` that wraps page-level contexts.
- Whether search page-content retrieval should call compatibility wrappers during early phases or a new orchestration API immediately.
- Which existing `WebSearch_APIs.py` provider/demo functions are truly public by import usage and tests.
- Whether job state should remain in the service layer or move into `Web_Scraping/jobs`.
- Whether the duplicate `app/core/WebSearch/Web_Search.py` path should remain as a compatibility shim or be consolidated during the search-provider split.
- Whether source import helpers should remain under `sources` or move closer to Collections after compatibility wrappers are stable.

These are implementation-plan decisions, not blockers for the architecture.

## Spec Self-Review

- Placeholder scan: no placeholder markers or incomplete sections remain.
- Consistency check: package boundaries, dependency rules, and migration phases all keep preflight advisory and governed.
- Scope check: design covers the full end-state but phases implementation to avoid a high-risk rewrite.
- Ambiguity check: compatibility wrappers, dict-shaped public contracts, cancellation handling, runtime-policy separation, import inventory, and preflight preservation are explicit.
- Follow-up review amendment: explicit homes were added for rate limits, resource governance, routing helpers, source import helpers, content metadata, markdown conversion, and deduplication. The compatibility surface now calls out RAG, MCP, Workflows, Watchlists, Collections, Research providers, DB import checks, and service callers.
