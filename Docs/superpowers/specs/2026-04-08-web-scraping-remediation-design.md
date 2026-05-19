# Web Scraping Remediation Design

Date: 2026-04-08
Topic: `Web_Scraping` ingest remediation
Status: Proposed design

## Goal

Address the issues found in the `Web_Scraping` ingest review without widening scope into unrelated scraper or lifecycle refactors.

The remediation must:

- remove the order-sensitive endpoint test failure around `/api/v1/media/process-web-scraping`
- close the reachable enhanced curl request-safety gap by reusing centralized outbound policy enforcement
- keep robots handling behavior unchanged while making its best-effort contract explicit in tests and nearby code

## Confirmed Problems

1. The `/process-web-scraping` custom-header regression test is order-sensitive.
   - Root cause: one test uses `TestClient`, which runs app shutdown and leaves the shared app in draining state, while the next test reuses the same app through `ASGITransport` without a fresh lifespan startup.
   - This produces a real `503 Service Unavailable`, but it is a test-harness isolation defect rather than a proven stable production-path route failure.

2. The enhanced scraper curl path is still a reachable request path that does not clearly prove the same redirect-hop and egress enforcement as the centralized `http_client` path.
   - Current issue location: `tldw_Server_API/app/core/Web_Scraping/enhanced_web_scraping.py`
   - Relevant central policy path: `tldw_Server_API/app/core/http_client.py`

3. Robots handling is intentionally configurable and fail-open when robots retrieval fails.
   - This is not a bug to change in this pass.
   - The problem is incomplete contract clarity and incomplete regression coverage for that behavior.

## Approved Constraints

The remediation will follow these decisions already agreed with the user:

- Keep the current robots best-effort behavior.
- Prefer a structural fix for the curl path, not test-only evidence.
- Accept a small production-code cleanup for the endpoint path when it improves determinism, but do not remove compatibility shims still used by friendly-ingest service flows.

## Primary Scope

- `tldw_Server_API/app/api/v1/endpoints/media/process_web_scraping.py`
- `tldw_Server_API/app/services/web_scraping_service.py`
- `tldw_Server_API/app/core/Web_Scraping/enhanced_web_scraping.py`
- `tldw_Server_API/app/core/http_client.py`
- `tldw_Server_API/tests/WebScraping/test_webscraping_usage_events.py`
- `tldw_Server_API/tests/WebScraping/test_custom_headers_support.py`
- `tldw_Server_API/tests/Web_Scraping/test_http_client_fetch.py`
- `tldw_Server_API/tests/Web_Scraping/test_robots_enforcement.py`

Secondary touched scope only if needed:

- `tldw_Server_API/tests/Media_Ingestion_Modification/test_media_compat_patchpoints.py`
- `tldw_Server_API/tests/Web_Scraping/test_filters_and_robots.py`
- `tldw_Server_API/tests/Web_Scraping/test_router_validation.py`

## Explicit Non-Goals

- No global FastAPI lifecycle redesign.
- No removal of `media.process_web_scraping_task` as a compatibility seam for friendly-ingest service paths.
- No fail-closed robots policy change.
- No broad scraper refactor outside the reachable ingest path.
- No attempt to make the e2e workflow runnable in this sandbox.

## Design

### 1. Endpoint And Test Determinism

The endpoint fix should be narrow and aimed at the actual failure mode.

The primary fix is test-harness determinism, not a larger production routing change. The confirmed `503` comes from mixing app lifecycle strategies across the two tests, so the first responsibility of this task is to make those tests use one consistent lifecycle-safe client pattern.

`process_web_scraping.py` may gain a module-local resolution seam that defaults to the imported service task, but only if that materially simplifies deterministic patching for the endpoint tests. This keeps the endpoint patch surface local and explicit without widening the change into friendly-ingest or service compatibility paths.

Important constraint:

- Do not remove the `media.process_web_scraping_task` export from `tldw_Server_API/app/api/v1/endpoints/media/__init__.py`.
- Friendly-ingest service paths in `web_scraping_service.py` still rely on that seam, and those flows are not being refactored in this pass.

The order-sensitive `503` fix should be completed primarily in the tests:

- standardize the affected endpoint tests on one lifecycle-safe client strategy
- stop mixing `TestClient`-driven lifespan shutdown with direct `ASGITransport(app=app)` reuse for this specific endpoint slice
- patch one stable endpoint-local target in both usage-events and custom-header tests

Recommended concrete direction:

- keep `test_cookie_manager_stores_and_returns_cookies()` async and independent
- convert the endpoint custom-header coverage to use the same `client_with_single_user` fixture style as the usage-events test, or a local wrapper around that fixture
- stop reusing the shared app directly through `ASGITransport(app=app)` for this endpoint slice after a `TestClient`-driven test has run
- update the compat patchpoint assertion to match the intentional endpoint-local resolution change

Success criteria:

- `test_webscraping_usage_events.py` and `test_custom_headers_support.py` pass together in one pytest invocation
- header forwarding remains asserted end to end at the route boundary
- friendly-ingest tests that patch `media.process_web_scraping_task` continue to pass unchanged

### 2. Curl Request-Safety Parity

The curl remediation should centralize policy rather than duplicate it.

The current enhanced curl path in `enhanced_web_scraping.py` should stop calling `CurlCffiSession.get()` directly. Instead, the enhanced scraper should route curl-backed HTML fetches through the centralized simple fetch path in `http_client.py`.

To make that safe enough for this use case, the centralized simple curl path itself must be hardened first:

- disable implicit curl redirect following
- apply the same lightweight `_is_url_allowed()` gate on the initial URL and on each redirect hop
- reuse the same redirect policy decisions already used by the simple httpx path:
  - same-host redirects allowed by default
  - cross-host redirects configurable
  - scheme downgrade blocked unless explicitly allowed
  - max redirect cap enforced
- keep proxy validation in the central path

After that hardening, `enhanced_web_scraping.py` should call the central fetch helper with `backend="curl"` and adapt the returned mapping into the existing `(html, backend_used, elapsed)` contract.

This keeps outbound policy logic in one place and removes the reachable direct-curl bypass shape from the enhanced scraper.

Success criteria:

- blocked curl URLs fail before fetch
- blocked redirect hops are not followed
- allowed same-host redirects still succeed under the central policy defaults
- successful curl fetches still return HTML content and preserve custom headers and cookies
- the enhanced scraper no longer owns separate curl redirect or egress logic

### 3. Robots Contract Clarification

Robots behavior should remain functionally unchanged.

This pass should:

- keep fail-open behavior when robots retrieval fails
- keep explicit block behavior when robots disallows the path
- add or tighten tests so this contract is visible and deliberate
- add a short clarifying code comment near the enhanced scraper robots branch to make the policy choice explicit for future readers

This is a contract-clarity change, not a behavior change.

Success criteria:

- robots tests clearly prove fail-open-on-fetch-error and deny-on-explicit-disallow behavior
- no request path changes from the existing user-visible robots policy

## Testing Strategy

The remediation should be developed test-first.

Minimum required targeted validation after implementation:

- `tldw_Server_API/tests/WebScraping/test_webscraping_usage_events.py`
- `tldw_Server_API/tests/WebScraping/test_custom_headers_support.py`
- `python -m pytest -v tldw_Server_API/tests/WebScraping/test_webscraping_usage_events.py tldw_Server_API/tests/WebScraping/test_custom_headers_support.py`
- `tldw_Server_API/tests/Web_Scraping/test_friendly_ingest_crawl_flags.py`
- `tldw_Server_API/tests/Web_Scraping/test_http_client_fetch.py`
- `tldw_Server_API/tests/Web_Scraping/test_robots_enforcement.py`

Run these additional tests if touched by the implementation:

- `tldw_Server_API/tests/Media_Ingestion_Modification/test_media_compat_patchpoints.py`
- `tldw_Server_API/tests/Web_Scraping/test_filters_and_robots.py`
- `tldw_Server_API/tests/Web_Scraping/test_router_validation.py`

Security validation:

- run Bandit on the touched web scraping and HTTP client files after the code change

## Risks And Mitigations

- Risk: changing the endpoint seam breaks friendly-ingest or compatibility expectations
  - Mitigation: keep the `media` shim export in place and limit the resolution change to the endpoint module only

- Risk: hardening curl redirect handling changes successful scrape behavior on legitimate redirect chains
  - Mitigation: preserve same-host redirects by default and add regression tests for allowed redirect behavior as needed

- Risk: tests become deterministic only by overfitting to current fixtures
  - Mitigation: standardize on one client pattern for this endpoint slice rather than sprinkling lifecycle resets across unrelated tests

## Implementation Shape

The work should be executed in three implementation tasks:

1. Stabilize endpoint task resolution and the affected endpoint tests.
2. Harden the centralized simple curl fetch path and reroute the enhanced scraper curl branch through it.
3. Tighten robots contract tests and add the minimal clarifying comment.

## Expected Outcome

After this remediation:

- the web scraping endpoint tests are deterministic and no longer fail because the shared app was left draining by an earlier test
- the reachable enhanced curl branch inherits centralized redirect and egress policy enforcement
- the project explicitly preserves best-effort robots semantics with regression coverage instead of leaving that behavior as an implicit assumption
