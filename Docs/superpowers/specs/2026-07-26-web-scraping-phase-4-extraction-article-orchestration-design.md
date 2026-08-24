# Web Scraping Phase 4: Extraction And Article Orchestration Design

**Status:** Approved for implementation planning
**Date:** 2026-07-26
**Backlog:** TASK-12988
**Roadmap phase:** Phase 4, Extraction Package Move
**Depends on:** Phase 3 governed preflight package, merged in PR #2752

## Summary

Phase 4 moves article extraction and governed single-page scraping out of
`Article_Extractor_Lib.py` without removing the pre-scrape analyzer, changing
established public result dictionaries, or collapsing the distinct behavior of
the enhanced scraper.

The target architecture has two primary layers:

- `extraction` owns HTML-to-result behavior, extraction strategies,
  enrichment, caches, and their narrow dependencies.
- `orchestration.article` owns scrape-plan resolution, outbound admission,
  governed preflight, HTTP and browser acquisition, cancellation, async
  offloading, and synchronous compatibility entry points.

Neutral content, selector, and bounded-regex helpers sit below both layers.
`Article_Extractor_Lib.py` remains an explicit compatibility surface and keeps
the crawl, jobs, ingestion, bookmark, and sitemap responsibilities assigned to
later phases.

The work is delivered as four sequential, independently reviewable units:
Phase 4A shared leaf components, Phase 4B extraction, Phase 4C article
orchestration and direct consumers, and Phase 4D final integration and gates.

## Context

`Article_Extractor_Lib.py` currently combines more than four thousand lines of
unrelated responsibilities:

- extraction strategy order, retries, traces, metrics, and caches;
- JSON-LD, schema, regex, LLM, cluster, and Trafilatura extraction;
- content formatting and metadata envelopes;
- outbound policy, optional preflight, HTTP fetch, and Playwright fallback;
- synchronous and asynchronous article entry points;
- summarization, ingestion, sitemap traversal, recursive crawl, bookmarks,
  source-file parsing, hashing, and progress state.

Phase 1 established typed compatibility contracts. Phase 2 established runtime
and policy protocols. Phase 3 moved the complete pre-scrape analyzer into the
governed `preflight` package and routed article and enhanced consumers through
its package-level facade. Phase 4 must build on those boundaries rather than
moving or bypassing them.

The Phase 0 import inventory records production and test consumers outside the
Web Scraping package, including Collections, Evaluations, RAG, Watchlists,
Workflows, WebSearch, services, and extraction tests. Compatibility decisions
in this design apply to names captured by that inventory and by the approved
roadmap design. Arbitrary private globals and private monkeypatch paths are not
public compatibility contracts.

## Goals

1. Establish one canonical implementation for article extraction strategies,
   enrichment, caches, formatting, and single-page orchestration.
2. Preserve governed preflight as an optional part of the standard article
   scrape plan.
3. Preserve per-dispatch outbound enforcement independently of preflight
   analysis or advice.
4. Keep inventoried imports, callable signatures, coroutine classification,
   result dictionaries, strategy-specific fields, and enhanced-scraper
   behavior compatible.
5. Prevent regex and PII matches from short-circuiting normal article
   extraction when the caller uses the inferred default strategy order.
6. Keep blocking work off active event-loop threads and propagate caller
   cancellation consistently.
7. Bound generated and configured regular-expression execution.
8. Bound article response bodies and rendered HTML before extraction.
9. Remove current import cycles and give each package one clear ownership
   boundary.
10. Deliver the migration in small units that can be tested, reviewed, rebased,
   and merged independently.

## Non-Goals

Phase 4 does not:

- move recursive crawl, sitemap traversal, URL-depth filtering, bookmark or
  source-file collection, ingestion batching, progress/resume state, or job
  queues; those belong to Phase 5;
- move WebSearch workflows, provider adapters, or provider result parsers;
  those belong to Phase 6;
- remove compatibility wrappers or unproven dead code; that belongs to Phase 7;
- rename public configuration keys or redesign public result schemas;
- unify the direct and enhanced Trafilatura contracts;
- change provider selection, LLM prompts, PII masking defaults, or
  settings-over-environment precedence;
- make plan headers, plan cookies, or plan proxies effective in the direct
  Playwright path where the current implementation ignores them;
- apply the new direct-article browser acquisition limits or routing behavior
  to the enhanced scraper, whose acquisition behavior remains separate;
- promise forcible termination of a synchronous provider call already running
  on a worker thread;
- preserve the effect of monkeypatching private globals in a legacy wrapper on
  a canonical consumer.

## Approved Behavior Changes

Only the following behavior changes are approved in this phase:

1. With `strategy_order=None`, regex is non-terminal enrichment. It cannot make
   a normal article extraction successful by itself.
2. Caller cancellation is re-raised rather than converted into fallback,
   retry, or a failure dictionary.
3. Moved synchronous scrape entry points reject calls from an active event-loop
   thread before policy, configuration, or network side effects.
4. Generated and configured regexes that are invalid, oversized, or exceed an
   execution deadline return stable sanitized failures instead of running
   without a bound.
5. The individual-URL service path passes the existing `system_message`
   keyword instead of the unsupported `system_prompt` keyword.
6. The legacy raw-browser sync path performs governed admission before network
   access while preserving its distinct result shape.
7. Raw provider, regex, selector, and transport exception text is replaced at
   public boundaries by the stable sanitized codes defined in this design.
8. Submission of extraction work is bounded by the existing worker setting,
   an explicit 64-worker ceiling, and a conservative default instead of relying
   on an unbounded default executor queue.
9. The direct async Playwright article path installs target, redirect,
   subresource, service-worker, and WebSocket egress controls before navigation.
   The current path performs only target-level admission.
10. Moved code removes full-URL, query-string, `url`, `base_url`, and raw-error
    metric/log fields. Existing metric names and non-sensitive low-cardinality
    labels remain stable.
11. Direct article acquisition rejects HTTP bodies, browser transfer totals, and
    rendered HTML that exceed the bounded article limits with the stable
    `response_too_large` code.

All other observed behavior is preserved unless a later reviewed design
explicitly changes it.

## Architecture

### Dependency Direction

```text
contracts       runtime protocols       policy adapters       preflight facade
    |                   |                      |                      |
    +-------------------+----------------------+----------------------+
                                |
safe_regex.py        content/        selectors/
      |                  |                |
      +------------------+----------------+
                         |
                    extraction/
                         |
     config + routing + handlers + runtime + policy + preflight
                         |
                orchestration/article.py
                         |
       Article_Extractor_Lib compatibility surface
```

The arrows describe allowed dependency flow toward higher-level composition.
The following rules are mandatory:

- `safe_regex.py`, `content`, and `selectors` do not import extraction,
  orchestration, Watchlists, enhanced scraping, WebSearch, or legacy wrappers.
- `extraction` does not import policy, preflight, Playwright, routing,
  `Article_Extractor_Lib.py`, enhanced scraping, WebSearch, or Watchlists.
- `orchestration.article` may compose config, routing, handlers, runtime,
  policy, preflight, and extraction.
- New internal packages never import a legacy wrapper.
- Legacy modules may import and explicitly re-export canonical names.
- `handlers.py`, `enhanced_web_scraping.py`, and internal consumers use
  canonical packages after their migration slice.

Architecture tests enforce these directions using the existing import-inventory
and AST guard pattern.

### Shared Leaf Components

#### `safe_regex.py`

This module provides bounded compilation and search for untrusted or
configuration-provided regular expressions. It uses the project's existing
`regex` dependency and enforces:

- a default maximum pattern length of 4,096 characters;
- a default maximum router input length of 8,192 characters;
- a default maximum generated-regex sample input of 1,000,000 characters;
- a default search deadline of 100 milliseconds;
- explicit flag normalization;
- stable errors for invalid, oversized, and timed-out expressions.

If an LLM-generated expression is valid but the sample HTML exceeds the sample
limit, generation may still succeed, but sample matching is skipped with a
bounded status field. Router patterns that cannot be evaluated safely do not
match and cannot block plan resolution. Static built-in regex catalogs may keep
their existing compiled representation because their patterns are trusted and
covered by tests.

The limits are internal defaults, injectable in tests, and do not add public
configuration keys in Phase 4.

#### `content`

`content.formatting` owns `convert_html_to_markdown` and other direct
HTML-to-display formatting moved in this phase. `content.metadata` owns
`ContentMetadataHandler`.

The metadata implementation preserves:

- `[METADATA]` and `[/METADATA]` envelope markers;
- the current metadata fields and `Trafilatura` pipeline value;
- the 64-level nesting guard;
- malformed-envelope pass-through behavior;
- body-only hashing and content-change semantics;
- inventoried legacy imports through explicit re-exports.

Neutral ownership is required because extraction, crawl persistence, services,
Collections, Watchlists, and Evaluations consume this behavior.

#### `selectors`

The shared selector package owns schema DSL normalization, CSS/XPath safety,
validation, execution, transforms, compiled-selector caches, cache stats, and
cache clearing currently implemented in Watchlists fetchers.

Watchlists keeps explicit compatibility exports for
`validate_selector_rules`, `extract_schema_fields`, selector-cache stats, and
selector-cache clearing. Extraction imports the neutral selector facade rather
than Watchlists. This removes the current upward dependency and associated
cycle without changing endpoint or test import paths.

Cached selector objects remain bounded and thread-safe. Selector result
dictionaries, validation errors, warnings, counts, field normalization,
transforms, and schema DSL behavior remain differential-test equivalent except
for approved bounded-regex failures in regex transforms.

### `extraction`

The extraction package is the canonical HTML-to-result boundary. Its logical
components are:

- `__init__.py`: explicit supported facade and `__all__`;
- `dependencies.py`: immutable dependency bundle and default factory;
- `pipeline.py`: order normalization, strategy dispatch, traces, fallback, and
  result assembly;
- `enrichment.py`: non-destructive enrichment merge rules;
- `caches.py`: schema, cluster, and LLM throttle/cache lifecycle;
- `strategies/`: JSON-LD, schema, regex, LLM, cluster, and Trafilatura
  implementations.

Exact file splitting may combine very small support modules, but the ownership
and dependency rules above do not change.

The dependency bundle supplies narrow callables or interfaces for selector
execution, LLM calls, metrics, clock, sleep, and cooperative cancellation.
Canonical public functions construct defaults at call time while tests and
higher-level orchestration may inject deterministic replacements. This is the
supported monkeypatch and test seam after migration.

LLM concurrency admission uses one stable process-wide limiter per canonical
provider identity. A smaller positive limit supplied by a later request tightens
that limiter after already-admitted calls drain; a live limiter is never replaced
or widened. Explicit throttle-state lifecycle reset or process restart creates a
new limiter. This conservative rule prevents differently configured requests
from creating overlapping semaphore generations that exceed either limit.

Cluster `min_word_count=0` retains the predecessor's truthy-fallback behavior and
therefore selects the default positive threshold. This is distinct from finite
similarity and prefilter thresholds, where an explicit `0.0` remains meaningful
and is preserved.

The package preserves canonical public functions and constants including:

- `DEFAULT_EXTRACTION_STRATEGY_ORDER`;
- `extract_article_with_pipeline`;
- `extract_article_data_from_html`;
- regex, JSON-LD, cluster, and LLM extraction helpers;
- schema and regex generation helpers;
- extraction cache stats and clearing.

### `orchestration.article`

The article orchestrator owns single-page acquisition and composition:

- scraper configuration and rule loading;
- user-agent, headers, cookies, proxy, and backend plan construction;
- typed article acquisition limits and a direct-browser compatibility profile;
- primary outbound policy admission;
- optional governed preflight execution;
- successful preflight advice application;
- lightweight runtime fetch;
- JavaScript-required detection and guarded browser fallback;
- HTTP body, browser transfer, and rendered-HTML budget enforcement;
- bounded extraction offloading;
- preflight payload and public result attachment;
- synchronous compatibility profiles and event-loop guards;
- resource cleanup, cancellation, and sanitized errors.

The canonical facade exposes the existing signatures of `scrape_article` and
`scrape_article_blocking`. The legacy `scrape_article_sync` raw-browser helper
keeps its historical raw HTML result shape through a governed compatibility
adapter. Both synchronous entry points use the same before-side-effects
active-event-loop guard.

The context-bound `scrape_article_async(context, url, ...)` helper is owned by
the recursive crawler and remains in the Phase 5 area. Phase 4 changes only its
extraction dependency to the canonical extraction facade. Its import path and
behavior remain compatible.

### `Article_Extractor_Lib.py`

The legacy module remains importable and contains:

- explicit imports and exports for canonical Phase 4 APIs;
- crawl, sitemap, bookmark, source-file, ingestion, batching, progress, and
  related compatibility code deferred to Phase 5;
- compatibility-only helpers not yet assigned to a canonical later package.

There is one implementation of each moved responsibility. The legacy module
does not retain copied extraction, selector, content, or article-orchestration
implementations.

## Article Data Flow

### Standard Async Flow

`scrape_article` follows this sequence:

1. Resolve configuration and the domain scrape plan.
2. Construct the effective user-agent, headers, cookies, proxy, backend, and
   extraction settings.
3. Perform primary outbound policy admission through the Phase 3 preflight
   facade's target evaluation boundary.
4. If denied or policy evaluation fails, return the stable sanitized policy
   failure dictionary without probing or fetching.
5. If enabled, build one governed preflight context, run the analyzer, and
   close its resources.
6. Apply advice only when the overall preflight result is successful and the
   relevant method or backend selection is `auto`. Explicit configuration wins.
7. Attempt the lightweight runtime fetch with the article response-body limit
   unless routing selected Playwright.
8. Apply a fresh egress decision before every scrape-side HTTP redirect or
   dispatch. The preflight decision and advice never authorize a later network
   request.
9. If eligible lightweight behavior fails, yields no extractable content, or
   indicates JavaScript is required, use the guarded browser adapter.
10. Route browser navigation, redirects, subresources, and web sockets through
    the guarded browser adapter described below. Apply the direct-browser
    compatibility profile rather than silently enabling currently ignored plan
    fields.
11. Enforce the browser transfer and rendered-HTML limits before HTML crosses
    into extraction.
12. Offload synchronous extraction through the bounded extraction executor.
13. Copy and enrich the extraction result, attach an optional public preflight
    payload, and return the legacy dictionary shape.

Optional preflight operational failure or timeout remains fail-open and uses
the configured/default scrape plan. Caller cancellation always propagates.
Preflight payload inclusion remains controlled by the existing option and is
limited to successful overall preflight results.

### Blocking Flow

`scrape_article_blocking` and `scrape_article_sync` first check
`asyncio.get_running_loop()` in the calling thread. If a loop is active, each
raises the same documented compatibility `RuntimeError` before configuration,
policy, metrics, browser startup, or network side effects.

Outside an active loop, it uses the shared article orchestrator with an explicit
blocking compatibility profile. That profile preserves the current robots
setting, 30-second timeout, cookie reduction, HTTP status handling, content
conversion, generic extraction settings, and result fields. Route-specific
extraction handlers and strategy settings are cleared while transport and
browser routing remain available. Optional preflight may run through the approved
blocking adapter's fresh event loop after the active-loop guard, but it does not
silently change those compatibility settings. It does not use the legacy
per-analyzer background-loop bridge. Browser fallback occurs only where the
compatibility profile and existing routing rules permit it.

`scrape_article_sync` performs governed target admission and then delegates
browser acquisition to the same async guarded article-browser adapter through a
fresh local event loop. It does not maintain a second synchronous routing
implementation. The adapter result is translated back to the historical raw
HTML dictionary without adding extraction fields or changing the public
signature.

### Policy And Egress Separation

The target-level policy decision answers whether analysis and a scrape plan may
proceed. It is not a reusable authorization token for future network I/O.

HTTP redirects, curl dispatch, and proxy use retain their existing runtime
egress controls, fresh resolution, credential stripping, and fail-closed
behavior. The direct article Playwright path does not currently provide
equivalent per-request routing. Phase 4 deliberately adds those browser controls
under approved behavior change 9 while reusing the Phase 2/3 protocols and
concrete policy adapters; it does not introduce a second policy implementation.

The guarded article browser adapter installs HTTP and WebSocket interception
before the first navigation, blocks service workers, and asks the shared egress
guard for a fresh decision before continuing each intercepted destination. A
request is aborted when validation fails. If required route or WebSocket
interception is unavailable, that capability fails closed instead of silently
continuing.

Playwright may independently resolve a hostname after URL-level validation, so
the design does not claim that route interception pins the browser's DNS result.
The current policy decision has no transport-pinning requirement, and Phase 4
does not infer one merely because `resolved_ips` is non-empty; doing so would
disable ordinary browser requests without adding a usable security guarantee.
The remaining DNS-rebinding window is an explicit residual risk of Playwright
dispatch. Deployments that require transport-level resolved-IP pinning must use
a pin-capable HTTP transport and must not select or fall back to Playwright. A
later security design may add a typed policy requirement and browser-transport
capability negotiation. Tests distinguish URL-level egress validation from
resolved-IP pinning.

### Direct Browser Compatibility Profile

The guarded article browser uses a typed request profile. Its field behavior is
fixed for Phase 4 so the refactor does not accidentally make currently ignored
configuration effective:

| Input or behavior | Phase 4 direct Playwright behavior |
| --- | --- |
| Effective user agent | Preserve the current plan/profile-derived user agent. |
| Caller `custom_cookies` | Copy and pass the original Playwright cookie dictionaries to the browser context. |
| Plan cookies | Remain ignored by direct Playwright; they continue to affect only the lightweight path. |
| Plan extra headers | Remain ignored by direct Playwright; they continue to affect only the lightweight path. |
| Plan proxies | Validate for lightweight use but remain ignored by direct Playwright. Enabling browser proxies requires a later reviewed behavior change. |
| Stealth setting | Preserve the existing optional stealth hook and wait behavior. |
| Retry count and timeout | Preserve the existing browser retry count and configured timeout normalization. |
| Navigation waits | Preserve `domcontentloaded`, followed by the existing stealth delay or `networkidle` wait. |
| Launch mode and viewport | Preserve headless Chromium and the existing 1280 by 720 viewport. |

The adapter never converts plan headers into context-wide Playwright headers.
Caller cookies remain subject to Playwright's domain, path, secure, and same-site
rules and are never copied into a manual `Cookie` header. Route continuations do
not add authorization, proxy-authorization, or cookie headers. If a later phase
adds browser headers or proxies, it must use the shared cross-origin credential
sanitizer and receive an explicit compatibility/security review.

### Article Acquisition Limits

Phase 4 adds two normalized server settings:

- `web_scraper_max_article_bytes`, default 16,777,216 bytes, bounds the
  lightweight response body and rendered HTML passed to extraction;
- `web_scraper_max_browser_transfer_bytes`, default 67,108,864 bytes, bounds
  aggregate encoded browser response data for one article navigation.

Only positive integers no greater than 1,073,741,824 bytes (1 GiB) are accepted;
absent, malformed, zero, negative, overlong-decimal, or larger values use the
defaults. Decimal length is checked before integer conversion so the bound does
not depend on the interpreter's integer-string protections. The normalized
limits are immutable for one scrape request and injectable in tests.

The limits apply to `scrape_article`, `scrape_article_blocking`, and
`scrape_article_sync`. They do not change the Phase 5 crawl-bound
`scrape_article_async(context, ...)` helper or enhanced-scraper acquisition in
Phase 4.

The lightweight adapter passes `max_response_bytes` to the central HTTP helper,
which applies the bound while accumulating the response. The guarded Chromium
adapter installs transfer accounting before navigation and stops the page when
the aggregate encoded HTTP response and WebSocket payload budget is exceeded.
If the required accounting capability is unavailable, guarded article-browser
acquisition fails closed.

The internal `FetchRequest` contract gains an optional `max_response_bytes`
field with a default of `None`, preserving all existing callers. Article
orchestration always supplies the normalized article limit. `DefaultFetchClient`
forwards it to the selected HTTP backend, and a backend that cannot enforce a
non-`None` limit fails closed instead of dispatching an unbounded request.
Phase 4C extends the central simple `http_client.fetch` path so both its httpx
and curl backends enforce the optional bound while streaming/accumulating the
body. The curl backend uses curl-cffi's synchronous native `content_callback`,
not its background producer queue, and retains at most the configured bytes in
the application buffer. Extra bytes are consumed without retention until the
hop status is available: terminal overflow fails, while redirect bodies remain
ignored as before. Callers that omit the option retain existing behavior.

Before returning rendered HTML, one browser-side operation serializes the
document, measures its UTF-8 byte length, and returns the string only when it is
within `web_scraper_max_article_bytes`. This avoids transferring an oversized
DOM string into the application process. The operation preserves the document
and doctype semantics characterized for the current `page.content()` path.

These limits bound application extraction input and normal browser transfer.
They do not claim to cap every allocation inside Chromium while JavaScript is
executing. Browser process memory isolation and operating-system resource limits
remain deployment controls. Limit failures close owned resources and return
`response_too_large`; they do not trigger an alternate backend that would repeat
the oversized acquisition.

## Extraction Pipeline Semantics

### Strategy Order

The public default list remains exactly:

```text
jsonld, schema, regex, llm, cluster, trafilatura
```

Aliases, normalization, duplicate removal, unknown-strategy traces, and
`allow_llm_extraction=False` behavior remain compatible. Disallowing LLM removes
only `llm` from the effective order.

### Default Regex Enrichment

When `strategy_order is None`, regex is non-terminal:

- If execution reaches regex and matches are found, the trace records a
  successful enrichment outcome and retains `regex_matches`.
- The pipeline continues to later content-producing strategies.
- A later successful content result receives a copied `regex_matches` field.
- Regex does not become the primary `extraction_strategy` in default mode.
- If all content strategies fail, the final failure result retains
  `regex_matches` but is not labeled as successful article extraction.

If an earlier JSON-LD or schema strategy succeeds, normal first-success behavior
still returns before regex is reached. Phase 4 does not add an unconditional
full-document enrichment pass.

### Explicit Strategy Compatibility

When a caller supplies any explicit `strategy_order`, the existing ordered
first-success semantics are preserved. This includes regex as a terminal
strategy when explicitly selected, whether alone or in a mixed explicit order.
`strategy_order=["regex"]` therefore remains a successful standalone regex
extraction when matches exist.

An explicit empty list or a list containing only unknown/blank entries still
normalizes to the public default order as it does today, but it remains an
explicit call for regex-terminal semantics. Unknown entries keep their skipped
trace records. Only the literal `None` selects non-terminal regex enrichment.

This distinction fixes the default false-positive article behavior without
breaking callers that deliberately configured regex extraction.

### Result Assembly

Every pipeline result preserves the base fields:

- `url`;
- `title`;
- `author`;
- `date`;
- `content`;
- `extraction_successful`.

Pipeline results preserve:

- `extraction_trace`;
- `extraction_strategy`;
- `extraction_strategy_order`.

Strategy-specific keys remain available, including regex matches, JSON-LD
types and summary, schema validation/count/cache fields, cluster block/tag
fields, LLM extraction/schema/provider/mode/usage fields, and existing error
codes.

JSON-LD summary carry-forward remains unchanged: a summary from an otherwise
non-terminal JSON-LD result enriches a later successful result only when that
result lacks a non-empty summary.

Result assembly copies strategy and cache payloads before adding traces,
enrichment, summaries, conversion fields, or preflight data. Cached values are
never mutated by a caller-specific result.

### Enhanced Scraper

The enhanced scraper imports canonical extraction functions where it already
shares article extraction. Its distinct behavior remains unchanged:

- its Trafilatura path continues to use JSON output, include tables, and return
  plain content;
- the direct article Trafilatura path continues to exclude tables and add the
  metadata envelope;
- enhanced Playwright and BeautifulSoup DOM fallbacks remain in place;
- enhanced trace entries, retries, job integration, and queue behavior remain
  unchanged;
- Phase 4 does not replace enhanced extraction wholesale with direct article
  orchestration.

## Compatibility Policy

Phase 4 follows the approved Phase 3 compatibility policy.

The migration preserves for inventoried public surfaces:

- old import paths;
- public names and signatures;
- positional versus keyword-only parameters;
- defaults and annotations where contract tests currently bind them;
- coroutine classification;
- callable identity where an explicit canonical re-export permits it;
- base and strategy-specific result dictionaries;
- policy and preflight payload fields;
- metadata envelope behavior.

Direct canonical re-exports have identity tests. Compatibility wrappers that
must adapt dependencies or enforce event-loop behavior have importability,
signature, coroutine-classification, and behavior tests instead of an identity
requirement.

Legacy private-module monkeypatches are not forwarded into canonical consumers.
Internal consumer tests patch or inject the canonical facade or dependency
bundle. This prevents wrappers from becoming a second dependency-injection
framework.

## Cancellation And Concurrency

`asyncio.CancelledError` is not part of any recoverable exception tuple in moved
code. It propagates through:

- policy admission;
- preflight execution and cleanup;
- fetch and redirect handling;
- browser acquisition, navigation, retry, and cleanup;
- extraction queueing and result wait;
- retry delays and inter-strategy checkpoints.

Extraction and LLM retry delays cap the complete exponential-base-plus-jitter
delay with `EXTRACTOR_RETRY_MAX_DELAY_MS`. The default cap is 30 seconds when
the setting is absent or invalid; `0` disables retry sleeping without changing
the configured attempt count.

An `ExtractionExecutorManager` owns one process-scoped executor generation. A
generation contains its process ID, monotonically increasing generation ID,
normalized worker count, `ThreadPoolExecutor`, `threading.BoundedSemaphore`, and
closed flag. The worker count is captured when the generation is created: a
positive `EXTRACTOR_MAX_WORKERS` value no greater than 64 is used when
configured, otherwise the default is four. Decimal length is checked before
conversion so Python 3.10 and later interpreters apply the same bound.
Environment changes do not silently resize a live generation.

A manager lock protects generation creation, replacement, and submission. Async
callers attempt non-blocking acquisition on the current generation's semaphore
and wait with cancellation-aware bounded backoff when capacity is unavailable.
Backoff starts at 10 milliseconds and is capped at 100 milliseconds; each wait
ends on admission, cancellation, or the admission deadline. The default
admission budget is 30 seconds and may be set with the positive finite
`EXTRACTOR_ADMISSION_TIMEOUT_SECONDS` environment value; injected managers may
provide the caller's remaining orchestration budget directly. After
acquiring a permit, submission re-enters the manager lock and verifies that the
generation is still current, has the current process ID, and is open. A stale
permit is released to its owning generation and acquisition restarts against the
new generation.

Submission occurs while replacement is excluded by the manager lock. After
successful submission, the permit is released by the concurrent future's done
callback to the exact generation that issued it. If submission itself fails,
the submitter releases that generation's permit before propagating the sanitized
failure. Callback release is idempotence-guarded so shutdown and cancellation
cannot over-release. This is safe across test loops and the blocking adapter's
fresh loop, and at most the worker limit is running or submitted to one live
generation.
Additional callers remain cancellable outside the executor instead of
accumulating in its internal queue. A cooperative token is set when the caller
is cancelled. Moved strategies check it before starting, between retries, and
before dispatching the next strategy.

Cancelling an await of already-running synchronous work does not terminate the
underlying thread. The caller is released immediately, the eventual result is
discarded, and no later strategy is dispatched. The executor slot remains
occupied until that call returns. Provider and strategy deadlines, bounded
admission, and saturation metrics prevent repeated cancellation from producing
unbounded abandoned work.

Explicit reload first closes admission to the old generation, drains it, and
then installs a generation using the newly normalized setting. Normal process
shutdown atomically detaches and closes the generation before waiting outside
the manager lock. A child-process PID mismatch or registered after-fork hook
discards the inherited executor state and creates a fresh generation; it never
waits on parent threads that do not exist in the child. Old callbacks release
only old-generation permits and cannot affect the replacement. Tests prove that
concurrent submission, cancellation, reload, shutdown, and fork reset do not
leak threads, over-release permits, submit to a closed executor, or grow an
executor's internal queue.

The manager distinguishes running, reloading, and terminally shut-down states.
Waiters poll through reload and resume against the replacement generation.
Shutdown causes waiting or later submissions to fail with `extraction_error` and
does not lazily recreate an executor. Only explicit process startup or the
documented test reset may leave the shut-down state.

## Error Handling And Cleanup

- Policy denial and policy evaluation failure are sanitized and fail closed.
- Optional preflight operational failure is sanitized and fail open.
- Strategy failures add sanitized trace entries and continue only when a valid
  fallback remains.
- Unknown strategies retain their existing skipped trace entries.
- HTTP responses, browser pages, contexts, launchers, and subprocesses close in
  `finally` blocks.
- Cancellation cleanup uses one bounded grace period. Cleanup failures are
  recorded internally and do not replace the original cancellation, timeout,
  or result.
- Public errors exclude credentials, cookies, authorization headers, proxy
  credentials, query strings, raw subprocess output, and provider exception
  text that may contain secrets.
- Public boundary failures use deterministic codes: `policy_error`,
  `regex_invalid`, `regex_too_large`, `regex_timeout`, `selector_invalid`,
  `provider_error`, `fetch_error`, `browser_error`, `response_too_large`, and
  `extraction_error`. Existing contract-bound safe codes remain unchanged.
- Internal failure logs contain the exception class, stable code, bounded stage,
  and sanitized host context. Raw exception messages are not logged by moved
  code because provider and transport messages may embed credentials or URLs.
- Failed and partial schema results are not cached.
- Cache reads return safe copies and cache writes store copies.
- Cache clearing and stats include the selector caches under their existing
  public key names.

### Public Failure Mapping

Sanitization does not collapse strategy-specific result fields. The mapping is:

| Boundary | Existing public field | Phase 4 value |
| --- | --- | --- |
| Target policy evaluation failure | Article `error` | `policy_error` |
| Explicit policy denial | Existing blocked-result fields | Preserve existing contract-bound safe denial code and shape. |
| Lightweight acquisition failure | Article `error` | `fetch_error` |
| Guarded browser startup, navigation, or routing failure | Article `error` | `browser_error` |
| HTTP, browser-transfer, or rendered-HTML limit | Article `error` | `response_too_large` |
| Pipeline boundary failure without a strategy result | Article `error` | `extraction_error` |
| Generated regex validation | Generator `error` | `regex_invalid`, `regex_too_large`, or `regex_timeout` |
| Generated regex/schema provider exception | Generator `error` | `provider_error`; existing safe precondition codes such as `regex_llm_empty_html` remain unchanged. |
| LLM extraction provider exception | Extraction `llm_error` | `provider_error`; safe state codes such as `llm_provider_missing` and `llm_empty_text` remain unchanged. |
| Selector parse/evaluation exception | Validation entry `error` | `selector_invalid`; existing deterministic `selector_too_complex:*` values remain unchanged. |
| Cluster and JSON-LD state failures | `cluster_error` or `jsonld_error` | Preserve existing deterministic safe codes; remove raw exception fragments. |

Strategy traces use the same stable codes in `reason` or `detail` and never copy
an exception message. The exact field and code pairs are contract fixtures, so a
caller does not need to inspect prose or parse prefixed exception strings.

## Observability

Existing scrape fetch, latency, content-length, strategy, cache, and fallback
metric names remain compatible. Non-sensitive low-cardinality label keys and
values remain compatible. Under approved behavior change 10, moved code removes
the existing `url`, `base_url`, and raw `error` metric labels instead of carrying
high-cardinality or sensitive values into the canonical packages. It does not
replace them with hostnames, hashes, paths, or other unbounded labels.

Phase 4 adds only bounded, low-cardinality observability for:

- regex enrichment versus explicit regex extraction;
- bounded-regex rejection and timeout reasons;
- extraction executor queued, running, saturated, cancelled, and discarded
  outcomes;
- article response-body, browser-transfer, and rendered-HTML limit outcomes;
- blocking active-event-loop rejection;
- sanitized orchestration stage failures.

New and moved metrics and logs do not include full URLs, query strings, content,
regex source, cookies, provider payloads, raw exception messages, or credentials.
Logs may include a bounded sanitized hostname from the shared observability
sanitizer; metrics do not. Deferred Phase 5 legacy code that still emits URL
fields is recorded as follow-on debt and is not silently described as remediated
by Phase 4.

## Delivery Units

Phase 4 is one architectural phase delivered as four sequential merge units.
Each unit starts from the merged predecessor, has its own Backlog child task,
uses RED/GREEN tests, receives an independent review, and is merged before the
next production unit begins.

### Phase 4A: Shared Leaf Components

Move and establish:

- bounded regex helpers and router/generated-regex integration;
- content formatting and metadata envelope helpers;
- selector engine, selector caches, and Watchlists compatibility exports;
- Watchlists imports migrated to the canonical selector facade;
- architecture guards for the new leaf packages.

Phase 4A does not move the extraction pipeline. Its differential tests prove
that content, metadata, selector, schema DSL, cache, Watchlists endpoint, and
router behavior remain compatible except for approved bounded-regex failures.

### Phase 4B: Extraction Package

Move and establish:

- extraction dependency bundle and canonical facade;
- strategy implementations and support helpers;
- extraction caches, throttles, retries, metrics, and traces;
- extraction metric/log sanitization for URL and raw-error fields;
- pipeline and result assembly;
- default regex enrichment and explicit-order compatibility;
- canonical extraction imports for handlers, enhanced scraping, and the
  crawl-bound context helper.

At the end of 4B, `Article_Extractor_Lib.py` explicitly re-exports moved
extraction names and contains no duplicate extraction implementation.

### Phase 4C: Governed Article Orchestration

Move and establish:

- standard async article orchestration;
- policy and optional preflight composition;
- advice precedence and guarded HTTP/browser acquisition;
- the direct-browser compatibility profile and per-request routing controls;
- HTTP body, browser transfer, and rendered-HTML limits;
- opt-in bounded simple-fetch support for both central httpx and curl backends;
- bounded extraction executor and cancellation lifecycle;
- blocking compatibility profile and active-loop rejection;
- governed raw-browser sync compatibility adapter;
- explicit legacy article exports;
- orchestration metric/log sanitization for URL and raw-error fields;
- canonical article imports for Collections, Evaluations, RAG, Watchlists,
  Workflows, WebSearch, and web-scraping services where the canonical facade is
  available;
- the `system_prompt` to `system_message` caller fix.

At the end of 4C, canonical article orchestration owns all moved single-page
entry points, and preflight remains part of the standard scrape plan.

### Phase 4D: Final Integration And Gates

Complete:

- any remaining safe canonical import migration identified by regenerated
  inventory;
- README, design references, import inventory, and generated inventory updates;
- dependency-direction, compatibility, focused, cross-consumer, security, and
  broad regression gates;
- final whole-phase review.

Compatibility imports remain for external and deferred Phase 5/7 callers.

## Test Strategy

### Characterization And Differential Tests

Before each physical move, tests capture the behavior owned by that unit. The
predecessor commit generates checked-in deterministic input/result fixtures for
pure extraction, selector, formatting, and orchestration-fake scenarios. Each
fixture records the predecessor commit ID and a schema version. Timing, random
IDs, cache warmth, and metric emission are normalized or asserted separately;
external providers and network operations use deterministic fakes.

Canonical tests read those immutable fixtures. They do not import a copied
legacy implementation or execute a second production implementation at test
time. Fixture regeneration is an explicit reviewed operation, never an automatic
consequence of a test run.

The differential harness has an explicit change allowlist limited to the eleven
numbered approved behavior changes in this design. Each permitted difference is
tagged with its behavior-change number. A difference outside that list is a
regression, not an incidental refactor adjustment.

### Compatibility Tests

Tests bind:

- inventoried import paths and supported names;
- exact signatures, defaults, annotations, and positional compatibility;
- coroutine classification;
- direct re-export identity where applicable;
- base result dictionaries and policy fields;
- extraction traces and effective strategy order;
- strategy-specific result fields;
- metadata envelope grammar and malformed input behavior.

### Pipeline Tests

Tests cover:

- default order, aliases, duplicates, and unknown strategies;
- `None`, explicit empty, and explicit unknown-only strategy-order semantics;
- LLM allowed and disallowed modes;
- default non-terminal regex and explicit terminal regex;
- default regex matches retained on later success and final failure;
- JSON-LD summary carry-forward;
- schema validation-before-execution and success-only caching;
- PII catalog limits, overlap rules, IP validation, Luhn filtering, and masking
  precedence;
- cluster fallback, tags, cache behavior, and concurrency;
- LLM parse, usage, retry, throttle, and sanitized failure behavior;
- exact public failure field/code mappings and trace sanitization;
- Trafilatura direct and enhanced contract separation;
- copy-on-read and non-mutating enrichment.

Property-based tests cover strategy normalization, bounded-regex invariants,
cache size/eviction invariants, and result merge immutability.

### Orchestration Tests

Deterministic fakes cover:

- preflight disabled and enabled;
- successful advice and explicit-setting precedence;
- preflight timeout and operational failure fallback;
- policy denial and policy checker failure;
- cancellation at policy, preflight, fetch, redirect, browser, extraction,
  retry, and cleanup stages;
- HTTP success, no-content, JavaScript-required, and error fallbacks;
- lightweight response-body limits and no-fallback behavior after an oversized
  response;
- acquisition-limit defaulting, malformed-value normalization, immutable
  per-request snapshots, `FetchRequest` default compatibility, and enforcement
  by every selectable lightweight backend;
- central simple-fetch compatibility when the new optional bound is omitted;
- redirect, DNS, proxy, browser navigation, subresource, service-worker, and web
  socket egress enforcement;
- browser interception capability failure and URL-level validation, including
  confirmation that a non-empty `resolved_ips` decision is not misrepresented
  as browser transport pinning;
- direct-browser compatibility-table coverage for user agent, caller cookies,
  ignored plan headers/cookies/proxies, stealth, retries, waits, viewport, and
  launch mode;
- browser transfer accounting capability failure, HTTP/WebSocket transfer-budget
  rejection, browser-side rendered-HTML rejection, and cleanup after each limit
  outcome;
- optional successful-preflight payload attachment on article success and on a
  final article extraction failure reached after that successful preflight;
- active-loop rejection before any side effect for both
  `scrape_article_blocking` and `scrape_article_sync`;
- blocking compatibility settings and result shape;
- executor saturation across simultaneous independent event loops, queued
  cancellation, running cancellation, discarded results, submit-failure permit
  release, concurrent reload/shutdown, stale-generation retry, dynamic-setting
  snapshot behavior, terminal shutdown admission, and process/fork reset;
- an event-loop heartbeat while extraction runs off-thread;
- moved metric calls contain no URL, base-URL, hostname, raw-error, or other
  unbounded label values, while existing non-sensitive label schemas remain
  stable.

One opt-in Playwright smoke test uses a local server when Playwright is
available. It never uses the public network.

### Cross-Consumer Tests

The required matrix includes focused suites for:

- Web Scraping Phase 1, 2, and 3 contracts;
- Watchlists selector DSL, validation, endpoints, and fetchers;
- Collections reading service;
- Evaluations article extraction benchmark;
- RAG research agent;
- Workflows RAG adapters;
- WebSearch article consumers;
- web scraping services;
- enhanced scraping, handlers, and crawl-bound extraction helpers.

### Verification Gates

Each delivery unit runs:

- focused tests for its behavior and compatibility surface;
- changed Python-file compilation;
- Ruff and Black checks on touched scope;
- `git diff --check`;
- Bandit on touched production paths;
- import-inventory regeneration and byte-equivalence checks where applicable;
- independent code review.

The final unit runs the broad Web Scraping and cross-consumer matrix. Required
Python 3.10 coverage runs in CI; local Python 3.10 execution is recorded when
that interpreter is available.

Any broad failure claimed as pre-existing must reproduce on the exact
`origin/dev` base in the same dependency and environment conditions. Touched
scope must pass its own gates even when the repository has unrelated baseline
debt.

## Risks And Mitigations

### Import Cycles

**Risk:** The legacy module, enhanced scraper, handlers, and Watchlists currently
use lazy imports to survive cycles.

**Mitigation:** Move shared behavior into leaf packages first, migrate consumers
to canonical facades, prohibit new-to-legacy imports, and run import smoke tests
after every unit.

### Selector Behavior Drift

**Risk:** The selector engine has broad schema DSL, endpoint, and cache usage.

**Mitigation:** Move it before extraction, keep direct compatibility exports,
use differential fixtures, and run Watchlists/API tests in Phase 4A and 4D.

### Result Mutation And Cache Leakage

**Risk:** Adding traces or enrichment to cached dictionaries can leak fields
between requests.

**Mitigation:** Copy on cache read/write and before every result merge; add
concurrency and immutability tests.

### Cancellation Of Synchronous Work

**Risk:** A cancelled caller cannot forcibly terminate a synchronous LLM call.

**Mitigation:** Use bounded workers and admission, provider deadlines,
cooperative checkpoints, discarded late results, lifecycle cleanup, and
saturation tests. Do not claim stronger cancellation than Python can provide.

### Executor Generation Races

**Risk:** Reload, shutdown, or fork can race with admission and cause submission
to a closed executor, permit over-release, or parent-thread waits in a child.

**Mitigation:** Use the locked generation model, bind every permit callback to
its owning generation, drain before explicit reload, reset on PID change, and
exercise submission/replacement races deterministically.

### Oversized Article Content

**Risk:** A target can return an oversized response or create a large rendered
DOM, exhausting memory even when extraction worker count is bounded.

**Mitigation:** Enforce the HTTP body, aggregate browser transfer, and
browser-side serialization limits before extraction; close resources and do not
retry an oversized page through another backend. Keep Chromium process-level
resource isolation as an explicit deployment responsibility.

### Egress Regression

**Risk:** Treating preflight admission as fetch authorization could reopen
redirect, DNS-rebinding, proxy, or browser subresource paths.

**Mitigation:** Keep target admission separate from per-dispatch guards, install
browser interception before navigation, fail closed when required interception
capabilities are unavailable, and reuse Phase 2/3 policy adapters instead of
creating a second policy implementation. Document the Playwright DNS-rebinding
window and do not describe URL-level validation as resolved-IP pinning.

### Browser Compatibility Drift

**Risk:** A generalized guarded browser request could start applying plan
headers, cookies, or proxies that the current direct browser ignores, leak a
sensitive header cross-origin, or drop caller cookies and stealth behavior.

**Mitigation:** Bind the direct-browser compatibility table in tests, never set
context-wide plan headers, rely on Playwright cookie scoping, and require a
separate reviewed change before enabling currently ignored fields.

### Observability Contract Drift

**Risk:** Preserving sensitive URL labels conflicts with the low-cardinality and
credential-safety requirements, while changing unrelated labels can break
dashboards.

**Mitigation:** Preserve metric names and non-sensitive label schemas, remove
only the explicitly allowlisted URL/base-URL/raw-error labels in moved code, and
assert label keys and bounded values in tests.

### Compatibility Drift

**Risk:** Moving functions changes signatures, result fields, traces, or old
imports.

**Mitigation:** Characterize first, use explicit exports and `__all__`, bind
contracts in tests, and permit only the approved behavior-change allowlist.

### Review And Rebase Size

**Risk:** Moving extraction, selectors, orchestration, and consumers in one PR
would be difficult to review and conflict-prone.

**Mitigation:** Use the sequential Phase 4A-4D merge train. Rebase each unit on
the merged predecessor and do not begin the next production move before its
dependency has merged.

## Completion Criteria

Phase 4 is complete only when all four delivery units are merged and:

- one canonical implementation owns every moved responsibility;
- no new internal package imports a legacy wrapper;
- inventoried old import paths and approved public contracts remain usable;
- strategy-specific fields, traces, metadata envelopes, policy fields, and
  optional preflight payloads remain compatible;
- preflight remains optional and integrated into standard article
  orchestration;
- every scrape-side network dispatch remains independently governed;
- direct article Playwright routing is treated as an explicit behavior change,
  not incorrectly characterized as pre-existing behavior;
- direct browser inputs match the compatibility table and do not silently enable
  currently ignored plan fields;
- HTTP bodies, browser transfers, and rendered HTML are bounded before
  extraction and report `response_too_large`;
- inferred-default regex cannot masquerade as successful article extraction;
- explicit strategy orders preserve their legacy semantics;
- `scrape_article_blocking` and `scrape_article_sync` reject active event-loop
  use before side effects;
- cancellation propagates and bounded worker growth is verified;
- executor reload, shutdown, and fork reset preserve generation and permit
  invariants under concurrent submission;
- enhanced scraper behavior remains differential-test equivalent;
- the verified service keyword mismatch is fixed by a real-signature test;
- import inventories and Web Scraping documentation are current;
- moved metrics/logs contain no full URLs, query strings, raw errors, or
  high-cardinality replacement labels;
- touched compile, format, lint, security, compatibility, focused, and
  cross-consumer gates pass;
- any unchanged broad failure has exact-base reproduction evidence;
- each unit and the final combined phase pass independent review.

## Follow-On Work

After Phase 4, the next roadmap item is Phase 5: move crawl discovery, sitemap
processing, recursive traversal, budgets, cancellation, progress, and queue/job
state into explicit `crawl` and `jobs` packages that call the canonical article
orchestrator for each page.

Two explicit follow-ups remain outside Phase 4:

- decide whether plan headers, plan cookies, and plan proxies should become
  effective in direct browser acquisition, with cross-origin credential rules;
- design guarded browser transfer and observability parity for the enhanced
  scraper without collapsing its distinct extraction, jobs, and retry behavior.
