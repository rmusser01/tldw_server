# Wave 4 Web Scraping And Outbound Safety Design

## Summary

Wave 4 hardens the shared outbound fetch surface used by web scraping, research websearch, and URL-scrape actions without turning the entire subsystem into a breaking-change rewrite.

The key design decision is to introduce one explicit runtime policy mode for these shared data-plane paths:

- `compat`: preserve current behavior
- `strict`: fail closed on policy uncertainty where the current scrape stack still relies on best-effort or branch-local semantics

This wave should ship the strict mode first, prove it with tests, and document it as the supported hardening path. It should not silently flip all current behavior to strict by default in the same wave.

The design intentionally distinguishes between:

- data-plane fetch paths that create real outbound network risk
- control-plane management endpoints that mostly inspect or mutate service state

The data-plane paths are in scope. The management surface is only in scope where it depends on or exposes the same policy contract.

## Goals

- Tighten runtime outbound-policy behavior for shared scraping and websearch paths behind an explicit strict profile first.
- Make the shared egress and robots contract explicit instead of letting each branch implement its own fail-open or fail-closed behavior.
- Keep user-facing endpoint behavior deterministic while the stricter policy is introduced.
- Cover all user-facing paths that share the same outbound fetch stack rather than only the ingest route.
- Prove strict-mode behavior with focused tests across scrape, search-provider, and URL-scrape action branches.
- Preserve a compatibility mode so rollout can happen without forcing an immediate global behavior change.
- Emit enough structured reason and metric data to compare `compat` and `strict` behavior during rollout instead of treating strict-mode enablement as a blind switch.

## Non-Goals

- Rewriting the entire scraping stack, provider catalog, or browser automation subsystem.
- Refactoring `/web-scraping` management endpoints unless they directly participate in shared policy behavior.
- Flipping strict enforcement on by default for all deployments in this wave.
- General outbound hardening for unrelated subsystems that do not share the same scrape and websearch data-plane.
- Redesigning the public request schemas unless a contract change is required to express the new strict-mode behavior.
- Folding in other raw-`evaluate_url_policy()` consumers such as document file download or workflow webhook DLQ delivery when they do not share the same scrape and websearch robots or fetch contract.

## Current State

### In-Scope Shared Surface

The current tree has multiple user-facing paths that share outbound policy concerns:

- media ingest route:
  - `tldw_Server_API/app/api/v1/endpoints/media/process_web_scraping.py`
- ingest service and reachable scrape branches:
  - `tldw_Server_API/app/services/web_scraping_service.py`
  - `tldw_Server_API/app/services/enhanced_web_scraping_service.py`
  - `tldw_Server_API/app/core/Web_Scraping/Article_Extractor_Lib.py`
  - `tldw_Server_API/app/core/Web_Scraping/enhanced_web_scraping.py`
- research and websearch surface:
  - `tldw_Server_API/app/api/v1/endpoints/research.py`
  - `tldw_Server_API/app/core/Web_Scraping/WebSearch_APIs.py`
  - `tldw_Server_API/app/core/RAG/rag_service/research_agent.py`
- shared outbound enforcement layer:
  - `tldw_Server_API/app/core/Security/egress.py`
  - `tldw_Server_API/app/core/http_client.py`

The `/web-scraping` management endpoints live in:

- `tldw_Server_API/app/api/v1/endpoints/web_scraping.py`

They mainly expose service status, progress, cookies, and job control. They do not appear to introduce separate outbound fetch branches on their own, so they should remain secondary to the data-plane paths above.

### Explicit Exclusions

There are other user-visible services that call `evaluate_url_policy()` directly, but they should stay out of Wave 4 unless implementation proves they actually share the same scrape and websearch policy contract:

- `tldw_Server_API/app/services/document_processing_service.py`
- `tldw_Server_API/app/services/workflows_webhook_dlq_service.py`

Those paths share raw egress enforcement, but they do not currently share the scrape-side robots semantics, result-shape contract, or websearch fetch flow that this wave is tightening. Naming them explicitly prevents “same egress helper” from turning Wave 4 into a generic outbound-policy rewrite.

### Current Evidence On `main`

The new Wave 4 worktree was created from `main`, not from the earlier review branch, and the following focused baseline passed on the current tree:

- `tldw_Server_API/tests/Web_Scraping/test_http_client_fetch.py`
- `tldw_Server_API/tests/Web_Scraping/test_robots_enforcement.py`
- `tldw_Server_API/tests/Web_Scraping/test_process_web_scraping_strategy_validation.py`
- `tldw_Server_API/tests/WebScraping/test_webscraping_usage_events.py`
- `tldw_Server_API/tests/WebScraping/test_custom_headers_support.py`

Result:

- `16 passed, 5 warnings in 10.18s`

That means the current mainline already has a stable baseline for:

- centralized redirect-aware `http_client` behavior
- current robots semantics
- `/process-web-scraping` strategy validation
- endpoint-level usage-event logging and custom-header forwarding

Wave 4 should preserve that determinism and build on it.

### Current Policy Behavior

#### Egress

`tldw_Server_API/app/core/Security/egress.py` already fails closed on many raw URL-policy checks:

- unsupported schemes
- invalid or missing hosts
- disallowed ports
- denylisted hosts
- strict-profile hosts not on the allowlist
- private or reserved IP resolution
- resolution failures

This is good baseline behavior for raw egress policy evaluation.

#### Scrape Branches

The scrape stack is less uniform than the egress helper itself:

- `EnhancedWebScraper.scrape_article()` pre-checks `evaluate_url_policy(url)` and returns a blocked result on denial.
- `Article_Extractor_Lib.scrape_article()` also pre-checks the egress policy before fetch.
- `EnhancedWebScraper._fetch_html_curl()` already routes through `http_client.fetch(..., backend="curl", follow_redirects=True)`, so the old “direct curl path bypass” concern should no longer be treated as a live assumption on `main`.

The current weak point is robots semantics, not raw egress routing:

- `tldw_Server_API/tests/Web_Scraping/test_robots_enforcement.py::test_is_allowed_by_robots_allows_when_unreachable`
- `Article_Extractor_Lib` comments and code
- `EnhancedWebScraper.scrape_article()` comments and code

Current behavior is best described as:

- explicit robots disallow blocks the request
- robots retrieval or evaluation failures fail open

That behavior may be acceptable in compatibility mode, but it is not a strong runtime hardening contract.

#### Websearch And Research Paths

`WebSearch_APIs.py` already uses `evaluate_url_policy(...)` at provider entry points and then routes network calls through `http_client.fetch(...)`.

`research_agent.py` uses `Article_Extractor_Lib.scrape_article()` for the `scrape_url` action, which means its runtime behavior inherits the same scrape-side policy semantics rather than defining its own.

The current websearch egress coverage is narrow but real:

- `tldw_Server_API/tests/Security/test_websearch_egress_guard.py`

That test proves denial behavior for at least one provider branch, but the shared strict-mode contract is not yet expressed centrally or exhaustively across providers and scrape actions.

## Requirements Confirmed With User

- Wave 4 should include the other user-facing surfaces if they share the same egress code, not just the ingest endpoint.
- The wave should tighten runtime policy, not only add proof or documentation.
- The tightening should ship behind an explicit strict profile or flag first, rather than as an immediate hard default for every deployment.

## Approaches Considered

### Approach 1: Flip The Existing Behavior To Strict Everywhere Now

Pros:

- Maximum immediate hardening
- Simple story at the code level

Cons:

- Highest regression risk
- Breaks the current documented and tested robots fail-open behavior in one step
- Makes it difficult to separate policy correctness work from rollout fallout

### Approach 2: Add A Shared Strict Profile And Migrate Shared Data-Plane Paths To It

Pros:

- Best balance of risk reduction and controllable rollout
- Makes the runtime contract explicit
- Lets tests prove both current compatibility behavior and stricter behavior
- Avoids branch-local policy drift between scraping and websearch callers

Cons:

- Slightly more design work up front
- Requires discipline to centralize behavior instead of adding more local conditionals

### Approach 3: Expand Tests And Docs Without Changing Runtime Policy

Pros:

- Lowest immediate regression risk
- Good fit if the main problem were uncertainty only

Cons:

- Does not actually tighten the runtime contract
- Leaves best-effort behavior in place where the user explicitly asked for stronger policy

## Recommendation

Use Approach 2.

Introduce one shared outbound-policy mode for scraping and websearch data-plane callers, ship `strict` as an explicit supported mode, and migrate the in-scope call sites to one common contract.

This wave should deliver real behavior change in strict mode, not just more tests, while leaving `compat` available so the default can remain stable until rollout confidence exists.

## Proposed Architecture

### Scope Boundary

Wave 4 should treat these as one logical subsystem because they share the same outbound fetch and scrape-policy boundary:

- `/api/v1/media/process-web-scraping`
- reachable legacy and enhanced scrape branches
- websearch provider fetches in `WebSearch_APIs.py`
- `research_agent` URL-scrape action

The following remain secondary:

- `/web-scraping` management endpoints

They should only change if needed to expose or document the policy mode or if tests reveal they accidentally depend on branch-local scrape semantics.

### Shared Outbound Policy Layer

Add one shared scraping and websearch outbound-policy layer above raw `evaluate_url_policy()`. This layer should live close to the scraping and websearch code, not as a generic catch-all security abstraction for unrelated subsystems.

Recommended shape:

- raw egress/IP/redirect enforcement remains in `app/core/Security/egress.py` and `app/core/http_client.py`
- reuse the existing `tldw_Server_API/app/core/Web_Scraping/filters.py` and `RobotsFilter` machinery where practical, or add one thin sibling helper such as:
  - `tldw_Server_API/app/core/Web_Scraping/outbound_policy.py`
- do not duplicate robots fetch, cache, and parser behavior in parallel helpers unless the implementation first proves the existing `filters.py` layer cannot carry the strict-mode contract cleanly

That helper should be responsible for:

- reading the configured mode
- normalizing policy decisions into one result shape
- handling robots behavior consistently
- exposing one small decision API that scrape and websearch callers can reuse
- resolving mode at call time, or through an injectable resolver, rather than capturing configuration once at import time where tests and rollout toggles become brittle

Suggested result contract:

- `allowed`
- `mode`
- `reason`
- `stage`
- `source`
- optional `details`

The important part is not the exact field names. The important part is that scrape and websearch callers stop inventing their own local policy outcomes.

Implementation guardrail:

- the implementation plan should enumerate the exact in-scope direct `evaluate_url_policy()` call sites and either migrate them to the shared helper or explicitly justify why a remaining direct call is still required

### Policy Modes

Recommended modes:

- `compat`
- `strict`

`compat` behavior:

- preserve current robots fail-open semantics
- preserve current endpoint contract where blocked results are returned in-band for scrape operations
- keep current provider-level denial behavior for websearch paths

`strict` behavior:

- fail closed on robots retrieval or evaluation errors when robots enforcement is enabled for the path
- fail closed on policy-evaluation exceptions, not just explicit denials
- require all in-scope scrape and websearch data-plane callers to route through the shared policy helper before navigation or fetch
- keep redirect and host-level egress enforcement delegated to `http_client.fetch()` where possible instead of open-coding redirect logic per branch

Strict mode should not silently override every user-requested behavior. In particular:

- existing `respect_robots=False` contracts should not be reinterpreted unless the deployment explicitly chooses a stronger policy in a later wave

This wave is about making strict hardening real and reusable, not about erasing every compatibility escape hatch.

### Scrape Path Behavior

The following branches should use the shared policy helper:

- legacy article scrape path
- enhanced trafilatura or curl-backed path
- enhanced browser-backed path
- sitemap and recursive discovery entry points where policy applies before fetch

The browser-backed path matters even if Playwright is not always available. Strict mode is only credible if browser navigation is blocked before outbound navigation when the same URL would be blocked for non-browser fetches.

### Websearch And Research Behavior

The following callers should also use the shared policy helper or a thin adapter over it:

- provider entry points in `WebSearch_APIs.py`
- article-scrape follow-ups triggered from websearch aggregation
- `research_agent` `scrape_url` action

The design goal is one policy story across:

- provider search requests
- result URL scraping
- agent-triggered URL scraping

This avoids ending up with strict semantics in ingest while research mode still behaves as best-effort.

Clarification:

- robots enforcement is relevant for scrape-style URL fetching and follow-up page retrieval, not for every provider API endpoint call in `WebSearch_APIs.py`
- provider API requests should still use the shared helper for raw egress-mode evaluation and consistent reason handling, but strict-mode robots failures should not be synthesized for provider endpoints that are not performing page scraping

### Error And Response Model

This wave should keep user-facing behavior boring and explicit.

Recommended contract:

- scrape-style operations continue returning structured blocked results where that is already the established API shape
- provider-style websearch operations continue raising or returning explicit upstream failure states as appropriate
- strict-mode failures should include a machine-readable reason code wherever the current API contract already supports structured error detail

The design should avoid adding a new family of ambiguous “maybe blocked” or silently downgraded outcomes.

### Configuration Contract

Introduce one explicit configuration key for the shared data-plane policy mode, rather than scattering booleans across unrelated files.

Recommended intent:

- one mode selector under the existing web scraping or web search configuration surface
- clear documented values:
  - `compat`
  - `strict`

The exact final key name can be chosen during implementation to fit the existing config conventions, but it should read as one shared mode for web outbound behavior rather than a route-specific toggle.

## Testing Strategy

### Baseline Preservation

The current focused baseline that already passes on `main` should remain green:

- `test_http_client_fetch.py`
- `test_robots_enforcement.py`
- `test_process_web_scraping_strategy_validation.py`
- `test_webscraping_usage_events.py`
- `test_custom_headers_support.py`

### New Strict-Mode Coverage

Add explicit tests for:

- `compat` versus `strict` robots behavior in `Article_Extractor_Lib`
- `compat` versus `strict` robots behavior in `EnhancedWebScraper`
- browser-backed scrape path refusing navigation under strict mode before Playwright fetch work starts
- websearch provider calls honoring the same strict-mode contract
- `research_agent` `scrape_url` action inheriting the same strict-mode behavior rather than bypassing it
- focused coverage proving the existing `RobotsFilter` path and any new shared helper do not diverge on the same URL and mode inputs

### Regression Focus

Tests should prove:

- current compatibility behavior still works when the mode is `compat`
- stricter behavior really changes the outcome when the mode is `strict`
- no in-scope branch skips the shared policy helper
- every remaining in-scope direct `evaluate_url_policy()` call is either removed or deliberately documented as a raw preflight that still feeds the shared result contract

This wave should not rely on indirect confidence like “it probably goes through the same helper.”

## Rollout Plan

Wave 4 should deliver:

1. one explicit shared policy mode
2. migration of the in-scope data-plane call sites to that contract
3. focused tests proving compat and strict behavior
4. structured counters or logs that distinguish compat-allow versus strict-block decisions at the shared policy layer
5. documentation describing the strict mode as the supported hardening path

Wave 4 should not automatically change the default mode for all deployments. That should be a later operational decision once strict mode is field-tested.

## Risks And Tradeoffs

- The biggest risk is accidental breakage of scrape flows that currently depend on robots fail-open behavior.
- The biggest design risk is only partially migrating call sites and ending with “strict” behavior that still varies by branch.
- The biggest testing risk is proving curl and browser parity weakly rather than explicitly.

Those risks are why the strict-profile-first rollout is the right choice.

## Open Questions

There are no blocking product questions left for this wave.

The remaining decisions are implementation-level:

- exact config key naming
- exact shared helper location
- whether any management endpoint needs a small policy-status exposure for observability

Those should be resolved in the implementation plan, not by broadening the design scope further.
