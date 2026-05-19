# Web Scraping Ingest Review Design

Date: 2026-04-07
Topic: `Web_Scraping` ingest-path review
Status: Approved design

## Goal

Review the web scraping ingest path in `tldw_server` for correctness issues, latent bugs, regression risks, security problems, reliability concerns, maintainability issues, performance costs, and missing test coverage.

The deliverable is a broader audit, not just a code-review bug list. The final review should produce prioritized findings with file references, evidence, validation results, and a remediation roadmap.

## Primary Scope

The review is limited to the end-to-end ingest path for public web scraping ingest requests:

- Endpoint entry and request forwarding:
  - `tldw_Server_API/app/api/v1/endpoints/media/ingest_web_content.py`
  - `tldw_Server_API/app/api/v1/endpoints/media/process_web_scraping.py`
- Request models and request-shaping logic used by those routes:
  - `tldw_Server_API/app/api/v1/schemas/media_request_models.py`
- Service orchestration, normalization, fallback, and mode handling:
  - `tldw_Server_API/app/services/web_scraping_service.py`
  - `tldw_Server_API/app/services/enhanced_web_scraping_service.py`

From those files, the review should follow only reachable code paths into helper functions and core scraping components that are actually used by the ingest flow.

## Secondary Reachable Scope

These files are in scope only where the ingest call graph reaches them:

- `tldw_Server_API/app/core/Web_Scraping/Article_Extractor_Lib.py`
- `tldw_Server_API/app/core/Web_Scraping/enhanced_web_scraping.py`
- `tldw_Server_API/app/core/Web_Scraping/scraper_router.py`
- `tldw_Server_API/app/core/Web_Scraping/filters.py`
- `tldw_Server_API/app/core/Web_Scraping/scoring.py`
- persistence helpers and DB-management functions used by the services
- ephemeral-storage helpers used by the services

## Explicitly Out Of Scope

These areas may be mentioned only when they directly affect the ingest-path contract:

- Optional web scraping management router:
  - `tldw_Server_API/app/api/v1/endpoints/web_scraping.py`
- General web search/provider orchestration:
  - `tldw_Server_API/app/core/Web_Scraping/WebSearch_APIs.py`
- Unrelated frontend/UI behavior
- Broad refactors outside the ingest-path boundaries

## Review Questions

The audit should answer these concrete questions:

1. Does the endpoint validate and forward request data consistently, including crawl flags, headers, cookies, and mode selection?
2. Do the service layers normalize inputs correctly and preserve contract semantics across the enhanced path and legacy fallback path?
3. Are `mode="ephemeral"` and `mode="persist"` both correct, observable, and consistent in how they shape results and handle failures?
4. Are crawl metadata, persistence metadata, cookie/header data, and observability fields preserved accurately after recent service and DB-lifecycle refactors?
5. Are error paths precise enough to preserve actionable HTTP behavior, or do broad catches turn contract problems into generic 500s?
6. Do any reachable outbound fetch paths weaken SSRF/egress protections, cookie safety, or header propagation guarantees?
7. Are there maintainability or performance problems in the reachable ingest-path code that materially increase future bug risk?
8. Does the current test suite cover the highest-risk behaviors, or are there important blind spots?

## Review Method

The audit should proceed in four passes.

### Pass 1: Contract And Control-Flow Mapping

Trace the request from the endpoint through:

- request schema and forwarded parameters
- input normalization and validation
- enhanced-service dispatch
- fallback gating and degraded-control handling
- ephemeral storage
- persistent storage
- crawl-config metadata shaping
- returned response contracts

The output of this pass should be a compact call-graph and a list of high-risk seams.

### Pass 2: Risk-Focused Code Audit

Inspect the highest-risk ingest-path boundaries for:

- endpoint/service contract mismatches
- enhanced-service initialization and shutdown risks
- fallback-path regressions and silent degradation
- config-default versus request-override precedence bugs
- metadata loss or coercion problems
- broad exception handling that hides root causes
- maintainability hotspots such as mixed responsibilities or duplicated logic

### Pass 3: Targeted Validation

Run a focused validation set rather than the full test suite.

Primary validation targets:

- `tldw_Server_API/tests/Web_Scraping/test_friendly_ingest_crawl_flags.py`
- `tldw_Server_API/tests/Web_Scraping/test_ingest_cookie_parsing.py`
- `tldw_Server_API/tests/Web_Scraping/test_process_web_scraping_strategy_validation.py`
- `tldw_Server_API/tests/Web_Scraping/test_crawl_config_precedence.py`
- `tldw_Server_API/tests/Web_Scraping/test_legacy_fallback_behavior.py`
- `tldw_Server_API/tests/Web_Scraping/test_persistence_crawl_metadata.py`
- `tldw_Server_API/tests/Web_Scraping/test_http_client_fetch.py`
- `tldw_Server_API/tests/Web_Scraping/test_robots_enforcement.py`
- `tldw_Server_API/tests/WebScraping/test_webscraping_usage_events.py`
- `tldw_Server_API/tests/WebScraping/test_custom_headers_support.py`

Optional validation target when the environment supports it:

- `tldw_Server_API/tests/server_e2e_tests/test_web_scraping_workflow.py`

The validation matrix must explicitly cover:

- request validation
- enhanced-path config precedence
- fallback-path behavior
- `mode="ephemeral"`
- `mode="persist"`
- metadata persistence
- error surfacing
- negative-path error contracts
- usage logging
- header and cookie forwarding where applicable
- request-safety enforcement where applicable

### Pass 4: Security And Request-Safety Audit

This is not limited to a generic static scan. The audit must explicitly verify:

- outbound URL safety and SSRF/egress enforcement on every reachable fetch path
- whether fallback or helper paths can bypass those checks
- propagation and handling of `custom_headers`, cookies, and `user_agent`
- whether fallback behavior weakens request-safety guarantees
- whether broad exception handling could mask a security-significant enforcement failure

Bandit should be run on the touched scope as a supplement, not as the primary security method.

## Evidence Rules

Findings must be labeled by evidence strength.

- Confirmed bug:
  - backed by a reproducer, failing targeted test, runtime behavior, or direct path proof
- Confirmed security issue:
  - backed by a concrete bypass path, reachable weak enforcement path, or demonstrated unsafe behavior
- Probable risk:
  - backed by static analysis showing a weak contract, ambiguous behavior, or fragile implementation pattern
- Maintainability or performance concern:
  - backed by static analysis and tied to real engineering cost or future bug risk

Correctness claims should not be presented as confirmed without proof. Architectural and maintainability risks may remain static findings if they are clearly labeled as such.

## Environment Assumptions

All commands should run from the repository root.

Before running Python, pytest, or Bandit commands:

```bash
source .venv/bin/activate
```

Environment-conditioned behaviors should be treated carefully:

- missing Playwright is an environment condition unless it breaks the documented ingest-path contract
- skipped e2e tests are not defects by themselves
- disabled optional routers are out of scope unless they leak into the ingest path

## Expected Output Format

The final review should include:

1. Prioritized findings first, ordered by severity
2. File references for each finding
3. Evidence type for each finding
4. Notes on current coverage:
   - covered and failing
   - covered but weak
   - uncovered
5. Open questions or assumptions
6. A remediation roadmap grouped into:
   - fix now
   - fix soon
   - opportunistic cleanup

## Non-Goals

This review design does not authorize implementation changes yet. The next step after spec approval is to write a concrete execution plan for the review work itself.
