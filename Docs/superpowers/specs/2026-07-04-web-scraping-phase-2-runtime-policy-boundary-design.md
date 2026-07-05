# Web_Scraping Phase 2 Runtime And Policy Boundary Design

Date: 2026-07-04
Task: TASK-12159
Status: Draft for user review

## Purpose

Design Phase 2 of the Web_Scraping modular refactor: introduce explicit runtime and policy boundaries while preserving current scraping behavior, especially the governed pre-scrape analyzer.

Phase 2 is intentionally conservative. It should create the interfaces and adapters needed for later runtime movement, then wire one small production path through those adapters to prove the shape works.

## Background

Phase 0 produced the import inventory and guardrails. Phase 1 added internal contracts and compatibility tests without moving runtime behavior.

The larger refactor design defines Phase 2 as:

> Move guarded fetch, browser launch, timeout, cancellation, session, and robots/policy plumbing behind explicit interfaces.

The user-approved refinement for this phase is:

- Add runtime and policy contracts/adapters.
- Wire one tiny production integration point.
- Preserve the pre-scrape analyzer and all public behavior.
- Defer analyzer package relocation and broad runtime movement.

## Non-Goals

Phase 2 must not:

- Move `scraper_analyzers/` into `preflight/`.
- Change analyzer result keys, recommendations, scoring, timeout behavior, or optional inclusion behavior.
- Change public `scrape_article(...)`, enhanced scraper, WebSearch, or crawl signatures.
- Replace the article extraction pipeline.
- Move Playwright browser lifecycle out of legacy wrappers yet.
- Move recursive crawl, sitemap, WebSearch provider, cookie cloning, or job queue behavior.
- Make `Article_Extractor_Lib._fetch_with_curl` or any other legacy local helper a public runtime primitive.

## Proposed Package Shape

Add a new package:

```text
tldw_Server_API/app/core/Web_Scraping/runtime/
  __init__.py
  requests.py
  responses.py
  policy.py
  fetch.py
  browser.py
  sessions.py
  timeouts.py
  cancellation.py

tldw_Server_API/app/core/Web_Scraping/policy/
  __init__.py
  adapters.py
```

This package is lower-level than the Phase 1 `contracts/` package. Runtime contracts should describe fetch and policy primitives, not article extraction results.

The `runtime/` package must remain policy-neutral. It may define policy-checker protocols and decision dataclasses, but it must not import `Web_Scraping.outbound_policy`, egress helpers, robots helpers, or legacy wrapper modules. Concrete policy adapters live outside `runtime/` and are injected by callers.

### Runtime Requests

`runtime.requests` should define frozen dataclasses for low-level runtime input:

- `RuntimeRequestContext`
  - `source`
  - `stage`
  - `user_id`
  - `request_id`
  - `metadata`
- `FetchRequest`
  - `url`
  - `method`
  - `headers`
  - `cookies`
  - `timeout`
  - `backend`
  - `allow_redirects`
  - `impersonate`
  - `proxies`
  - `context`

The request objects should normalize mutable mappings to immutable string-keyed mappings, following the style of the Phase 1 contracts.

### Runtime Responses

`runtime.responses` should define:

- `FetchResponse`
  - `url`
  - `status`
  - `headers`
  - `text`
  - `backend`
  - `elapsed_seconds`
  - `metadata`
- `PolicyDecision`
  - `allowed`
  - `reason`
  - `mode`
  - `stage`
  - `source`

The response layer should avoid raw exception text in public fields. If adapter failures need to be represented as structured failures later, they should use sanitized messages consistent with Phase 1 `RuntimeFailure`.

### Policy Adapter

`runtime.policy` should define only the protocol-like boundary for scrape-level policy checks:

- `OutboundPolicyChecker`
  - async `decide(url, *, respect_robots, user_agent, context, config) -> PolicyDecision`

`policy.adapters` should provide the default adapter that delegates to the existing `Web_Scraping.outbound_policy.decide_web_outbound_policy`.

Policy timing must remain unchanged for `scrape_article`: the scrape-level policy check happens before preflight analyzer execution. This preserves the current guarantee that analyzers do not run for denied URLs.

### Fetch Adapter

`runtime.fetch` should define:

- `FetchClient`
  - sync `fetch(request: FetchRequest) -> FetchResponse`
- `DefaultFetchClient`
  - delegates to `tldw_Server_API.app.core.http_client.fetch`

The curl backend should be requested by passing `backend="curl"` to the central HTTP client. The adapter should not call or expose `Article_Extractor_Lib._fetch_with_curl`.

The central `http_client.fetch` has two call modes today:

- passing `method=` delegates to the response-object API,
- omitting `method=` uses the simplified Web_Scraping path with `backend`, `impersonate`, redirect, and curl support.

For the Phase 2 production seam, `DefaultFetchClient` should use the simplified path for GET article fetches so `backend="curl"` keeps working. Non-GET support should be rejected or explicitly deferred until a later phase instead of silently falling into the wrong HTTP helper mode.

`DefaultFetchClient` should normalize both mapping-like and object-like responses into `FetchResponse`, preserving the legacy tolerance currently provided by `_resp_get` for fields such as `status`, `status_code`, `headers`, `text`, `url`, and `backend`.

The central HTTP client already performs lower-level egress checks. Phase 2 should keep that as defense in depth while making the scrape-level policy decision explicit and injectable.

### Browser, Session, Timeout, And Cancellation Contracts

Phase 2 should define contract placeholders for the remaining runtime responsibilities named in the larger refactor design:

- `runtime.browser`
  - protocol for browser launch/context/page acquisition,
  - no production Playwright wiring in this phase.
- `runtime.sessions`
  - dataclasses or protocols for cookie/session state passed to fetch or browser adapters,
  - no cookie cloning move in this phase.
- `runtime.timeouts`
  - small timeout/budget dataclasses for fetch and browser operations,
  - no global timeout policy rewrite in this phase.
- `runtime.cancellation`
  - cancellation marker/protocol helpers that preserve `asyncio.CancelledError` behavior,
  - no job-state cancellation rewrite in this phase.

These modules should be small and import-boundary tested. They exist to make Phase 3+ plans concrete, not to move runtime behavior prematurely.

## Production Integration Point

Wire only the lightweight HTTP fetch path inside `Article_Extractor_Lib.scrape_article`.

The integration point should cover:

1. Existing scrape-level policy decision before preflight.
2. Existing preflight analyzer logic and advisory backend selection.
3. Existing lightweight fetch attempt for `httpx`/`curl`.

It should not cover:

- Playwright launch and retry loop.
- Stealth behavior.
- Browser cookie application.
- Extraction pipeline movement.
- Enhanced scraper or WebSearch call paths.

Implementation should keep the public `scrape_article(url, custom_cookies=None)` signature unchanged. If dependency injection is needed for tests, add it to a private helper rather than the public function.

## Expected Flow

The intended `scrape_article` order after Phase 2:

1. Resolve router plan and headers as today.
2. Use the injected default policy adapter to make the existing `pre_fetch` decision.
3. Return the same blocked article dict when denied.
4. Run preflight analyzers exactly as today when enabled.
5. Let preflight advice continue to adjust `backend_choice` and `preflight_method`.
6. For non-Playwright lightweight fetch, construct a `FetchRequest`.
7. Use `DefaultFetchClient` to perform the fetch through central `http_client.fetch`.
8. Preserve the current curl-to-httpx fallback: if a curl lightweight fetch fails, retry with an httpx lightweight fetch before falling back to Playwright.
9. Convert `FetchResponse` back into the existing local extraction path.
10. Fall back to Playwright exactly as today on JS-required, error, or no-extract outcomes.

## Compatibility Requirements

Phase 2 must preserve:

- `scrape_article(...)` return dictionaries, including policy fields and preflight payload attachment.
- Existing router backend semantics for `auto`, `curl`, `httpx`, and `playwright`.
- Existing preflight behavior:
  - policy check before analyzer execution,
  - optional analyzer execution by config,
  - optional result inclusion by config,
  - JS-required advice selecting Playwright,
  - TLS advice selecting curl when backend is auto.
- Existing fallback metrics and labels where the lightweight path currently records them.
- Existing hardening behavior for egress policy, robots policy, redaction, and cancellation handling.

## Test Plan

Add focused no-network tests:

- Runtime dataclass immutability and normalization.
- Runtime package import boundary:
  - runtime must not import `Article_Extractor_Lib`, `enhanced_web_scraping`, or `WebSearch_APIs`.
  - runtime must not import `Web_Scraping.outbound_policy`, core egress policy, or robots helpers directly.
  - runtime contracts should not depend on article extraction contracts.
- Default policy adapter delegates to `decide_web_outbound_policy` and maps allow/deny fields.
- Default fetch adapter delegates to central `http_client.fetch`, including `backend="curl"` without touching `_fetch_with_curl`.
- Default fetch adapter uses the simplified GET path without `method=` for Phase 2 article fetches, so curl support is not accidentally bypassed.
- Default fetch adapter normalizes mapping-like and object-like response fields.
- Browser, session, timeout, and cancellation contract modules import only stdlib and local runtime primitives.
- `scrape_article` uses the runtime policy adapter for the pre-fetch decision and preserves the blocked dict shape.
- `scrape_article` uses the runtime fetch adapter for lightweight HTTP fetch and preserves extraction success output.
- `scrape_article` preserves curl-to-httpx fallback before Playwright fallback.
- Preflight compatibility test where analyzer advice still changes `backend_choice` or `preflight_method` and still attaches the same optional payload.
- Existing tests:
  - `test_phase1_contracts.py`
  - `test_router_backend_selection.py`
  - `test_enhanced_web_scraping_guards.py`
  - `test_outbound_policy.py`
  - `test_http_client_fetch.py`

Security verification should run Bandit on touched Web_Scraping files when implementation begins. For this design-only task, record that no Python code changed.

## Risks And Mitigations

### Risk: Analyzer Regression

Moving the policy decision or fetch code could accidentally change when preflight runs.

Mitigation: keep policy before preflight and add a test proving denied URLs do not run analyzer code.

### Risk: Double Policy Confusion

`http_client.fetch` already enforces lower-level egress policy. The runtime policy adapter adds an explicit scrape-level decision.

Mitigation: document the two layers. The scrape-level adapter owns user-facing policy metadata and robots behavior; the HTTP client remains defense in depth for redirects, proxies, and low-level network calls.

### Risk: Runtime Abstractions Become Too Broad

It would be tempting to introduce a full scraping engine in Phase 2.

Mitigation: keep runtime contracts limited to policy and fetch primitives. Article extraction, orchestration, browser lifecycle, and analyzer relocation remain later phases.

### Risk: Legacy Helper Leakage

Wrapping `_fetch_with_curl` would make a private helper part of the new architecture.

Mitigation: call the central HTTP client directly with `backend="curl"`.

### Risk: Test Injection Pollutes Public API

Adding runtime parameters to `scrape_article` would change a public compatibility surface.

Mitigation: use a private helper or module-level default adapters for tests, keeping public signatures stable.

### Risk: Wrong Central HTTP Helper Mode

Calling `http_client.fetch(method="GET", backend="curl", ...)` would select the response-object path and bypass the simplified curl-capable Web_Scraping path.

Mitigation: the Phase 2 default fetch adapter must call the simplified GET path without `method=` for article fetches and include a regression test proving `backend="curl"` reaches the curl-capable branch.

## Spec Self-Review

Review findings addressed before user review:

- The first draft placed the default policy adapter inside `runtime/`, which would violate the larger refactor rule that runtime must not directly import policy. The design now keeps `runtime.policy` protocol-only and puts concrete adapters under `Web_Scraping/policy/`.
- The first draft under-modeled browser, session, timeout, and cancellation boundaries. The design now adds contract-only placeholders for those modules while keeping production wiring limited to policy and lightweight fetch.
- A later review found that `http_client.fetch` has separate response-object and simplified Web_Scraping modes. The design now requires the simplified GET path for Phase 2 article fetches so curl backend behavior is preserved.
- The production seam remains intentionally small: no Playwright lifecycle move, no analyzer relocation, and no public signature change.

## Implementation Notes For Next Plan

The implementation plan should be test-first and staged:

0. Rebase or recreate the implementation branch on latest `dev`, then re-check Web_Scraping touchpoints before editing Python code.
1. Runtime package contracts and import boundary tests.
2. Policy adapter outside `runtime/` and fetch adapter with unit tests.
3. Browser, session, timeout, and cancellation contract placeholders with import-boundary tests.
4. Private helper around `scrape_article` policy and lightweight fetch path.
5. Compatibility tests for blocked, success, curl, and preflight-advice cases.
6. Verification and Backlog finalization.

Implementation should use `superpowers:writing-plans` next, then the existing subagent-driven/checkpoint workflow for actual code changes.
