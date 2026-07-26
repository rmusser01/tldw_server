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
8. Remove current import cycles and give each package one clear ownership
   boundary.
9. Deliver the migration in small units that can be tested, reviewed, rebased,
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
8. Submission of extraction work is bounded by the existing worker setting and
   a conservative default instead of relying on an unbounded default executor
   queue.

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
- primary outbound policy admission;
- optional governed preflight execution;
- successful preflight advice application;
- lightweight runtime fetch;
- JavaScript-required detection and guarded browser fallback;
- bounded extraction offloading;
- preflight payload and public result attachment;
- synchronous compatibility profiles and event-loop guards;
- resource cleanup, cancellation, and sanitized errors.

The canonical facade exposes the existing signatures of `scrape_article` and
`scrape_article_blocking`. The legacy raw-browser sync helper keeps its
historical raw HTML result shape through a governed compatibility adapter.

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
7. Attempt the lightweight runtime fetch unless routing selected Playwright.
8. Apply a fresh egress decision before every scrape-side HTTP redirect or
   dispatch. The preflight decision and advice never authorize a later network
   request.
9. If eligible lightweight behavior fails, yields no extractable content, or
   indicates JavaScript is required, use the guarded browser adapter.
10. Route browser navigation, redirects, subresources, and web sockets through
    the guarded browser adapter described below, and validate proxies before
    browser launch.
11. Offload synchronous extraction through the bounded extraction executor.
12. Copy and enrich the extraction result, attach an optional public preflight
    payload, and return the legacy dictionary shape.

Optional preflight operational failure or timeout remains fail-open and uses
the configured/default scrape plan. Caller cancellation always propagates.
Preflight payload inclusion remains controlled by the existing option and is
limited to successful overall preflight results.

### Blocking Flow

`scrape_article_blocking` first checks `asyncio.get_running_loop()` in the
calling thread. If a loop is active, it raises the documented compatibility
`RuntimeError` before configuration, policy, metrics, or network side effects.

Outside an active loop, it uses the shared article orchestrator with an explicit
blocking compatibility profile. That profile preserves the current robots
setting, 30-second timeout, cookie reduction, HTTP status handling, content
conversion, and result fields. Optional preflight may run through the approved
blocking adapter's fresh event loop after the active-loop guard, but it does not
silently change those compatibility settings. It does not use the legacy
per-analyzer background-loop bridge. Browser fallback occurs only where the
compatibility profile and existing routing rules permit it.

### Policy And Egress Separation

The target-level policy decision answers whether analysis and a scrape plan may
proceed. It is not a reusable authorization token for future network I/O.

HTTP redirects, curl dispatch, proxy use, browser top-level navigation,
subresources, service workers, and web sockets retain their existing runtime
egress controls, fresh resolution, credential stripping, and fail-closed
behavior. Phase 4 reuses the Phase 2/3 protocols and concrete policy adapters;
it does not introduce a second policy implementation.

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

The process-scoped extraction executor uses a positive
`EXTRACTOR_MAX_WORKERS` value when configured and a default of four workers
otherwise. A process-scoped `threading.BoundedSemaphore`, not an
event-loop-bound asyncio primitive, guards submission. Async callers attempt a
non-blocking acquisition and wait with cancellation-aware bounded backoff when
capacity is unavailable. Backoff starts at 10 milliseconds and is capped at
100 milliseconds; each wait ends on admission, cancellation, or the caller's
orchestration deadline. After successful submission, the permit is released by
the concurrent future's done callback. If submission itself fails, the submitter
releases the acquired permit before propagating the sanitized failure. This is
safe across test loops and the blocking adapter's fresh loop, and at most the
worker limit is running or submitted to the executor.
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

The executor has explicit shutdown and process/fork reset behavior. Tests prove
that cancellation, shutdown, and reset do not leak threads, permits, or queued
futures.

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
  `provider_error`, `fetch_error`, `browser_error`, and `extraction_error`.
  Existing contract-bound safe codes remain unchanged. Raw exception text is
  available only to sanitized internal logging.
- Failed and partial schema results are not cached.
- Cache reads return safe copies and cache writes store copies.
- Cache clearing and stats include the selector caches under their existing
  public key names.

## Observability

Existing scrape fetch, latency, content-length, strategy, cache, and fallback
metric names and label semantics remain compatible.

Phase 4 adds only bounded, low-cardinality observability for:

- regex enrichment versus explicit regex extraction;
- bounded-regex rejection and timeout reasons;
- extraction executor queued, running, saturated, cancelled, and discarded
  outcomes;
- blocking active-event-loop rejection;
- sanitized orchestration stage failures.

Metrics and logs do not include full URLs, query strings, content, regex source,
cookies, provider payloads, or credentials.

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
- bounded extraction executor and cancellation lifecycle;
- blocking compatibility profile and active-loop rejection;
- governed raw-browser sync compatibility adapter;
- explicit legacy article exports;
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

Before each physical move, tests capture the behavior owned by that unit.
Differential fixtures compare canonical results with the characterized baseline
for unchanged scenarios.

The differential harness has an explicit change allowlist limited to the eight
approved behavior changes in this design. A difference outside that list is a
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
- LLM allowed and disallowed modes;
- default non-terminal regex and explicit terminal regex;
- default regex matches retained on later success and final failure;
- JSON-LD summary carry-forward;
- schema validation-before-execution and success-only caching;
- PII catalog limits, overlap rules, IP validation, Luhn filtering, and masking
  precedence;
- cluster fallback, tags, cache behavior, and concurrency;
- LLM parse, usage, retry, throttle, and sanitized failure behavior;
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
- redirect, DNS, proxy, browser navigation, subresource, service-worker, and web
  socket egress enforcement;
- browser interception capability failure and URL-level validation, including
  confirmation that a non-empty `resolved_ips` decision is not misrepresented
  as browser transport pinning;
- optional successful-preflight payload attachment on article success and on a
  final article extraction failure reached after that successful preflight;
- active-loop rejection before any side effect;
- blocking compatibility settings and result shape;
- executor saturation across simultaneous independent event loops, queued
  cancellation, running cancellation, discarded results, submit-failure permit
  release, shutdown, and process/fork reset;
- an event-loop heartbeat while extraction runs off-thread.

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

### Egress Regression

**Risk:** Treating preflight admission as fetch authorization could reopen
redirect, DNS-rebinding, proxy, or browser subresource paths.

**Mitigation:** Keep target admission separate from per-dispatch guards, install
browser interception before navigation, fail closed when required interception
capabilities are unavailable, and reuse Phase 2/3 policy adapters instead of
creating a second policy implementation. Document the Playwright DNS-rebinding
window and do not describe URL-level validation as resolved-IP pinning.

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
- inferred-default regex cannot masquerade as successful article extraction;
- explicit strategy orders preserve their legacy semantics;
- blocking entry points reject active event-loop use before side effects;
- cancellation propagates and bounded worker growth is verified;
- enhanced scraper behavior remains differential-test equivalent;
- the verified service keyword mismatch is fixed by a real-signature test;
- import inventories and Web Scraping documentation are current;
- touched compile, format, lint, security, compatibility, focused, and
  cross-consumer gates pass;
- any unchanged broad failure has exact-base reproduction evidence;
- each unit and the final combined phase pass independent review.

## Follow-On Work

After Phase 4, the next roadmap item is Phase 5: move crawl discovery, sitemap
processing, recursive traversal, budgets, cancellation, progress, and queue/job
state into explicit `crawl` and `jobs` packages that call the canonical article
orchestrator for each page.
