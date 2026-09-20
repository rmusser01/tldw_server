# Web_Scraping Phase 3 Governed Preflight Package Design

Task: TASK-12968
Status: Draft for user review
Date: 2026-07-14

## Purpose

Design Phase 3 of the `Web_Scraping` modular refactor: move the pre-scrape
analyzer implementation into a governed `preflight` package and make that
package the shared preflight boundary for both article scraping paths.

The move must improve maintainability, extensibility, and runtime safety
without losing analyzer functionality or changing public scrape contracts.

## Background

Phase 1 introduced typed request and result contracts. Phase 2 introduced
runtime and outbound-policy boundaries and preserved the rule that primary
target policy runs before preflight. The analyzer implementation still lives
under `scraper_analyzers`, and preflight orchestration remains duplicated in:

- `Article_Extractor_Lib.scrape_article`
- `enhanced_web_scraping.EnhancedWebScraper`

Both consumers independently parse the same configuration, run the same
analyzers, interpret JavaScript and TLS signals, apply routing advice, and
construct the same optional payload. Individual analyzers also create HTTP
sessions, Playwright instances, and subprocesses directly. That makes policy,
timeout, cancellation, cleanup, and redaction behavior difficult to enforce
consistently.

Phase 3 resolves those ownership problems while preserving the compatibility
surface recorded by Phase 0.

## Goals

- Make `Web_Scraping/preflight` the sole implementation owner for analyzer
  orchestration, analyzers, scoring, recommendations, and analyzer utilities.
- Route both article scraping consumers through one typed facade.
- Preserve all current preflight config keys and their existing behavior.
- Preserve analyzer result keys, successful result values, scoring,
  recommendations, routing advice, and optional public payload shape.
- Govern HTTP, browser, and external-tool probes through injected adapters.
- Keep preflight optional and advisory; analyzer failure must not fail an
  otherwise valid scrape.
- Preserve primary policy denial as blocking before analyzer or extraction
  network work.
- Keep old `scraper_analyzers` imports working through temporary shims.
- Make behavior deterministic and testable without real network, browser, or
  external-tool dependencies.

## Non-Goals

Phase 3 will not:

- Move article extraction strategies out of `Article_Extractor_Lib`.
- Split crawl, jobs, search providers, or other later-phase responsibilities.
- Introduce an analyzer plugin registry or configurable analyzer scheduling.
- Parallelize the top-level analyzer sequence.
- Reuse fetched bodies across analyzers or extraction.
- Add user-facing request, browser, or active-probe budget config keys.
- Change public scrape function signatures or return dictionaries.
- Change successful analyzer thresholds, scoring weights, or recommendations.
- Remove compatibility shims; removal remains a Phase 7 responsibility.
- Make the extraction Playwright lifecycle use the new analyzer browser
  adapter.
- Proxy or inspect network requests made internally by external executables.
- Provide DNS-pinned browser transport or an egress-enforcing browser proxy.

## Selected Approach

Use a facade-led governed migration.

The analyzer implementation moves physically into `preflight`. The package
adds typed options, a URL-bound target policy evaluation, an execution context,
governed probe adapters, centralized advice, and centralized payload
eligibility. Both scrape consumers adopt that facade in this phase. The old
package becomes a temporary explicit re-export layer.

This approach is preferred over a package-only move because a package-only
move would preserve direct-network and event-loop problems. It is preferred
over a new analyzer engine because Phase 3 is compatibility-sensitive and
does not need a registry, new scheduling model, or new public configuration.

## Target Package Layout

```text
Web_Scraping/
├── contracts/
├── policy/
├── runtime/
├── preflight/
│   ├── __init__.py
│   ├── facade.py
│   ├── options.py
│   ├── context.py
│   ├── probes.py
│   ├── runner.py
│   ├── compatibility.py
│   ├── analyzers/
│   ├── scoring/
│   ├── recommendations/
│   └── utils/
└── scraper_analyzers/       # temporary compatibility shims only
```

`preflight` owns implementation. `scraper_analyzers` owns no runtime logic.

## Dependency Direction

The allowed production dependency direction is:

```text
Article_Extractor_Lib / EnhancedWebScraper
                    |
                    v
             preflight facade
                    |
                    v
      contracts + injected runtime/policy protocols
```

The following rules are mandatory:

- `preflight` must not import either legacy scraper.
- `runtime` must not import `preflight` or concrete policy implementations.
- `runtime` may define probe-egress protocols; concrete adapters belong under
  `policy` and may delegate to centralized security policy evaluation.
- The two scrape consumers must not import analyzer, scoring, or
  recommendation internals.
- Moved analyzers must not create HTTP sessions, Playwright instances, or
  subprocesses outside the governed probe adapters.
- New application code must import `preflight`, not `scraper_analyzers`.

## Core Contracts

### PreflightOptions

`PreflightOptions.from_mapping(...)` is the only production parser for
preflight configuration. It accepts the existing configuration mapping and
normalizes:

- `web_scraper_preflight_analyzers`
- `web_scraper_preflight_timeout_s`
- `web_scraper_preflight_scan_depth`
- `web_scraper_preflight_find_all_waf`
- `web_scraper_preflight_impersonate`
- `web_scraper_preflight_include_results`
- `web_scraper_preflight_enable_external_tools`
- `web_scraper_playwright_no_sandbox`

Existing defaults remain unchanged:

- analyzers are disabled unless enabled;
- non-positive or malformed overall timeout values mean no overall timeout;
- invalid scan depth becomes `default`;
- find-all WAF, impersonation, result inclusion, and no-sandbox remain false
  unless enabled.

External tools require a compatibility transition. When
`web_scraper_preflight_enable_external_tools` is absent, an installed
`wafw00f` remains usable, matching current behavior. An explicit false value
disables it; an explicit true value enables it. A malformed explicit value
fails closed to false and produces only a sanitized configuration warning.

When the absent setting activates the installed-tool fallback, a
concurrency-safe process-level once guard emits one safe warning without a URL
and increments the bounded metric once:
`web_scraping_preflight_legacy_external_tool_default_total{tool="wafw00f"}`.
An explicit true or false setting emits neither warning nor compatibility
metric. Phase 7 changes the absent-setting default to disabled after migration
telemetry and documentation have made the transition visible.

No new user-facing budget keys are introduced. Tests and future callers may
inject optional limits through the execution context.

### PreflightTarget

The facade evaluates the primary target through the Phase 2
`OutboundPolicyChecker` and returns an immutable `PreflightTarget` containing:

- the normalized target URL;
- the resulting `PolicyDecision`;
- the `RuntimeRequestContext` used for the decision.

This binds the policy decision to the URL and request metadata instead of
passing an unbound boolean. Both scrape consumers use the same decision:

- denied decisions are converted through their existing blocked-result
  adapters;
- policy evaluation failure retains the existing generic extraction failure;
- allowed targets may be passed to `run_preflight` and then to extraction.

`run_preflight` rejects a denied `PreflightTarget` as programmer error. It
never performs analyzer work for a denied target.

### ProbeEgressGuard

The Phase 2 `OutboundPolicyChecker` remains a scrape-level boundary. It owns
the primary egress and optional robots decision and is not reused for every
redirect, browser asset, or WebSocket attempt.

Phase 3 adds a narrower protocol-only `ProbeEgressGuard` under `runtime` and a
concrete adapter under `policy`. The guard returns a small immutable
`ProbeEgressDecision` and delegates to centralized
`Security.egress.evaluate_url_policy`. It does not evaluate robots.

The execution context uses this guard for every HTTP redirect destination,
browser request, and explicit external-tool launch target, including
exact-target dispatches. Positive decisions are not cached across dispatches.
The HTTP runtime performs its existing egress check again immediately before
transport work.

### PreflightExecutionContext

One request-scoped execution context owns:

- the runtime request metadata associated with the explicit target;
- the outbound policy checker;
- the probe egress guard;
- governed HTTP, browser, and external-tool adapters;
- an overall deadline derived from the existing timeout setting;
- optional request, browser, and active-probe limits;
- atomic consumed-budget counters;
- deterministic browser-identity selection;
- sanitized logging helpers.

Budget limits default to `None`, meaning unbounded. This preserves current
capacity. Counters are still maintained so finite limits can be injected and
tested without redesigning analyzer APIs.

Reservations are atomic. The rate-limit analyzer's existing concurrent burst
reserves one request slot per request before dispatch. A failed reservation
returns a legacy analyzer error with `error_code: "budget_exhausted"` and never
makes the request. Counters must never become negative or exceed a finite
limit.

### PreflightResult and PreflightAdvice

Phase 3 reuses the Phase 1 `PreflightResult` and `PreflightAdvice` contracts.
It does not introduce a competing analyzer result model.

`run_preflight(...)` returns:

- `None` when analyzers are disabled;
- `PreflightResult(status=OK, ...)` when the runner completes, including when
  individual analyzer result entries report errors;
- a non-OK `PreflightResult` with a sanitized `RuntimeFailure` when the overall
  preflight run times out or its orchestration fails.

The `analysis` mapping for a successful run retains the exact top-level keys:

- `results`
- `score`
- `recommendations`

The `results` mapping retains the exact analyzer keys and order:

- `robots`
- `tls`
- `js`
- `behavioral`
- `captcha`
- `fingerprint`
- `integrity`
- `rate_limit`
- `waf`

Per-analyzer failures continue to use legacy dictionary entries with
`status: "error"`, a safe message, and a stable error code where applicable.
Overall typed status and failure fields are not added to public payloads.

## Facade API and Responsibilities

The facade owns three operations:

1. `evaluate_target(...) -> PreflightTarget`
2. `run_preflight(target, options, context) -> PreflightResult | None`
3. `public_preflight_payload(result: PreflightResult | None, include_results) -> dict | None`

It also owns `apply_preflight_advice(...)`, which applies a typed
`PreflightAdvice` to the caller's current backend and method without exposing
analyzer internals to the caller.

`public_preflight_payload` is the only payload eligibility gate. It returns the
existing `{"analysis": ..., "advice": ...}` dictionary only when inclusion is
enabled and the overall result status is `OK`. It returns `None` for disabled,
timed-out, or otherwise non-OK overall runs. Individual analyzer errors,
including probe budget exhaustion or unavailable optional capabilities, do not
make an otherwise completed run ineligible.

The existing `preflight_result_to_public_dict` remains the shape converter,
but production consumers call it only through the eligibility gate.

## Analyzer Execution

The runner preserves the current top-level analyzer order. It does not run
independent analyzers concurrently. Existing concurrency internal to an
analyzer, specifically the rate-limit burst, remains intact.

Each analyzer receives the same execution context and performs outbound work
only through its probe interfaces. Passive parsing and scoring helpers remain
ordinary pure functions.

Each moved analyzer has two deliberately separate entry surfaces:

- a private async implementation that requires the execution context and is
  the only surface used by the runner and facade;
- a public compatibility entry that preserves the historical name, signature,
  and sync/async classification.

The historically synchronous JS, integrity, WAF, captcha, behavioral, robots,
and fingerprint entries remain synchronous to direct callers. TLS and
rate-limit entries remain coroutine functions. Compatibility wrappers adapt
to the internal async implementation without allowing the standard facade to
call synchronous wrappers.

Unexpected analyzer exceptions are isolated to that analyzer key. The runner
continues with the remaining analyzers and then calculates scoring and
recommendations over the complete result map. `asyncio.CancelledError` is
never converted into an analyzer error and always propagates.

Scoring or recommendation orchestration failure is an overall preflight
failure, matching current behavior where runner failure discards preflight
advice and payload. It does not fail extraction.

## Governed HTTP Probes

HTTP probes use an async adapter built on the existing runtime fetch boundary.
The adapter:

- reserves request budget before each dispatch;
- caps the operation timeout at the smaller of the analyzer timeout and the
  remaining overall deadline;
- disables transport-level automatic redirects;
- resolves relative redirects explicitly;
- applies the probe egress guard before every hop;
- uses the existing `http_client.DEFAULT_MAX_REDIRECTS` limit and redirect
  safety behavior rather than defining a second limit;
- strips sensitive headers and cookies at origin boundaries through the
  existing HTTP client behavior;
- closes every response in success, failure, timeout, and cancellation paths.

An exact-target probe reuses the allowed URL-bound `PreflightTarget` decision
instead of repeating the scrape-level robots decision. Every dispatch,
including an exact-target dispatch, receives a fresh probe egress decision.
The runtime HTTP boundary then repeats its existing egress validation
immediately before transport work. Redirect and subrequest checks use a
`preflight_subrequest` stage and never evaluate robots. This keeps primary
robots timing unchanged while governing every actual outbound destination.

Redirect loops, invalid locations, denied destinations, policy-check errors,
and exhausted budgets become analyzer-scoped safe errors. No denied redirect
is dispatched.

## Governed Browser Probes

The current runtime browser protocol is insufficient because it cannot install
request interception. Phase 3 extends the protocol with the minimum hooks
needed to govern analyzer browser work and adds a production async Playwright
adapter.

The guarded browser adapter:

- reserves browser budget before launch or context creation;
- creates contexts with service workers blocked because Playwright routing
  cannot reliably intercept requests owned by service workers;
- installs HTTP and WebSocket routing before creating or navigating a page;
- applies the probe egress guard to HTTP and HTTPS navigation, redirects,
  subresources, and WebSocket attempts;
- aborts denied requests without exposing the full URL in logs;
- uses the remaining overall deadline to cap navigation and wait timeouts;
- closes pages, contexts, and browser processes in `finally` paths.

Phase 3 raises the Playwright dependency floor to `>=1.48.0` in the base,
`web_research`, and `scrape-analyzers` dependency groups because
`browser_context.route_web_socket` was added in Playwright 1.48. The adapter
also performs a runtime capability check for environments installed before the
dependency change. If HTTP routing, WebSocket routing, or service-worker
blocking is unavailable, the affected browser analyzer returns `unavailable`.
It must not fall back to direct synchronous Playwright access.

These requirements follow the Playwright
[`BrowserContext` API](https://playwright.dev/python/docs/api/class-browsercontext),
which documents the WebSocket routing version and recommends blocking service
workers when relying on request routing.

Browser routing provides URL-level egress enforcement; it does not pin DNS
resolution to the address approved by the guard. Full DNS-rebinding protection
for browser probes requires a governed proxy or pinned browser transport and
is outside Phase 3. This limitation is explicit and must not be described as
equivalent to the runtime HTTP transport's dispatch-time validation.

Network and browser operations use native async APIs. `asyncio.to_thread` is
not used for cancellable network, browser, or subprocess work because task
cancellation cannot stop the underlying thread. Thread offload remains
permitted only for the existing bounded DNS resolver bridge, the probe-egress
adapter around the synchronous central evaluator, bounded non-I/O parsing, and
the isolated legacy synchronous compatibility bridge.

## Governed External Tools

`wafw00f` discovery and execution are injected adapter dependencies. The
adapter:

- honors the compatibility config behavior described above;
- requires the allowed URL-bound target and obtains a fresh probe egress
  decision before launch without repeating the primary robots decision;
- reserves one active-probe budget slot;
- executes an argument list without a shell;
- uses an async subprocess;
- enforces the existing 60-second tool timeout capped by the overall deadline;
- terminates, then kills and awaits the process if needed on timeout or
  cancellation;
- never logs raw arguments, stdout, stderr, or exception text;
- returns parsed, sanitized WAF results through the existing result shape.

An external executable is an opaque active probe. Phase 3 can govern whether
it starts, its approved target, budget, timeout, cancellation, and exposed
output, but cannot policy-check redirects or requests the executable makes
internally. Proxying or sandboxing those internal requests is explicitly out
of scope. The tool also resolves its own destination after the launch check,
so the design does not claim DNS pinning or per-hop governance inside
`wafw00f`.

## Advice Semantics

Advice generation moves into `preflight` and preserves current behavior:

- a successful JS result with `js_required` or `is_spa` recommends Playwright
  only when the current method remains automatic;
- a TLS result whose `status` is `active` recommends curl only when the
  configured backend remains automatic;
- missing, malformed, denied, timed-out, or error results do not change
  routing;
- advice notes remain `js_required` and `tls_active` in their existing order.

The two scrape consumers call `apply_preflight_advice` and do not inspect
`results["js"]` or `results["tls"]`.

## Failure and Cancellation Semantics

Failure behavior is intentionally asymmetric:

- Primary policy denial or policy-evaluation failure remains blocking before
  analyzer and extraction network work.
- Analyzer-level policy denial, timeout, budget exhaustion, missing dependency,
  unavailable capability, or unexpected error affects only that analyzer key.
- Overall preflight timeout or runner failure preserves the caller's original
  backend and method, records a typed internal failure, and omits the public
  preflight payload.
- Advice is generated only from explicit successful analyzer signals.
- External cancellation propagates after cleanup and is never normalized into
  `timeout`, `error`, or a successful result.

Policy-check failure for an analyzer probe denies that probe. It never fails
open into an ungoverned outbound request.

The execution context derives one monotonic deadline at preflight start. Every
operation caps its local timeout against the remaining time. Expiry of that
deadline becomes the existing overall preflight timeout result; cancellation
of the caller remains `asyncio.CancelledError` and is never confused with
deadline expiry. If the deadline timer and caller cancellation race, observed
caller cancellation wins.

Resource cleanup runs in a shielded task with one shared two-second grace
period for the preflight run so caller cancellation does not immediately
interrupt close operations. If graceful cleanup exceeds that bound, adapters
force-close resources and terminate, then kill and await subprocesses as
applicable. Cleanup errors are sanitized and recorded internally but never
replace the original timeout or cancellation outcome.

## Redaction

Preflight logs use sanitized host/path labels and exclude:

- URL credentials;
- query strings and fragments;
- cookies and authorization headers;
- proxy credentials;
- subprocess arguments and raw output;
- exception text that may embed request data.

Public analyzer error values use safe messages and stable error codes. This may
intentionally replace unsafe raw error text, but successful analyzer outputs
and public field names remain unchanged.

## Compatibility Shims

The physical implementation moves into `preflight`. Every old import path
recorded by Phase 0 remains importable under `scraper_analyzers`, including
deep analyzer, scoring, recommendation, and utility modules.

Canonical public compatibility wrappers live in `preflight`; old modules use
explicit re-exports and explicit `__all__` values. Private internal async
implementations are not re-exported. The wrappers and shims preserve:

- import paths;
- public names and callable signatures;
- callable identity where an explicit re-export permits it;
- legacy result dictionaries;
- synchronous `run_analysis` event-loop rejection behavior.

The shims do not promise that monkeypatching an old internal module changes a
new consumer. Tests for consumer behavior inject or patch the new facade.
Direct callers of legacy `gather_analysis` and `run_analysis` receive the same
`results`/`score`/`recommendations` shape through a default governed context.
The synchronous wrapper still raises inside an active event loop.

Historically synchronous per-analyzer wrappers run their internal coroutine
through a lazily started, process-scoped background event-loop thread. The
bridge propagates return values and exceptions, supports the wrapper's timeout,
cancels timed-out submissions without abandoning their cleanup, and shuts down
at process exit. This bridge is compatibility-only and is never used by the
facade or runner. `run_analysis` deliberately does not use the bridge; it
retains its historical active-event-loop rejection. Historically async TLS and
rate-limit public entries remain coroutine functions.

For a policy-denied direct legacy call, the compatibility layer returns the
stable top-level and analyzer-key structure with safe `policy_denied` analyzer
errors and performs no probes. This is the required safety change for a direct
entry point that previously had no primary policy boundary.

For a direct legacy call whose policy checker fails, the same stable structure
uses safe `policy_error` analyzer codes and performs no probes. The wrapper
does not expose exception text.

Shims are documented as deprecated but emit no runtime deprecation warnings.
They remain until Phase 7 proves no in-repository application consumers depend
on them.

## Consumer Migration

Both `Article_Extractor_Lib` and `EnhancedWebScraper` adopt the same sequence:

1. Build `PreflightOptions` from the existing config mapping.
2. Evaluate the primary target once through the facade.
3. Preserve each consumer's existing blocked or policy-error conversion.
4. Run optional preflight with a request-scoped execution context.
5. Apply typed advice through the facade helper.
6. Run extraction using the resulting backend and method.
7. Attach the centralized eligible payload to every existing success or
   failure return path exactly as today.

Duplicated preflight config parsing, `asyncio.to_thread(run_analysis, ...)`, JS
and TLS inspection, payload construction, and attachment eligibility are
removed from both consumers.

## Testing Strategy

All required regression tests use deterministic fakes and perform no external
network, real browser, or real external-tool work.

### Characterization and Contract Tests

- Replay fixed runtime responses through every analyzer.
- Inject deterministic browser identity selection.
- Assert exact successful analyzer values, score cards, recommendations,
  advice, notes, and optional payload shape.
- Assert the stable top-level and analyzer-key sets for every runner outcome.
- Cover every existing config key with absent, valid, and malformed values.
- Assert the absent external-tool setting emits its process-level warning and
  metric at most once, while explicit true or false emits neither.
- Cover both scrape consumers and all existing return paths that attach
  preflight metadata.

### Policy and Probe Tests

- Denied primary targets run no analyzers or extraction probes.
- Exact-target probes reuse the scrape-level target and robots decision, every
  dispatch receives a probe egress decision, and runtime HTTP repeats egress
  validation immediately before transport work.
- Redirect tests cover relative locations, loops, maximum hops, scheme changes,
  denied/private targets, missing locations, and policy-check failure.
- Browser fakes verify service workers are blocked, routing is installed before
  page creation, and denied navigation, redirects, subresources, and WebSocket
  attempts are aborted.
- Browser capability tests verify environments below Playwright 1.48 return
  `unavailable` without launching a browser.
- External-tool availability and execution are fully injected so tests cannot
  execute a locally installed `wafw00f` accidentally.
- External-tool launch tests require a fresh allowed probe egress decision and
  prove denial or guard failure prevents process creation.

### Budget, Timeout, and Cancellation Tests

- Atomic request reservations cover the rate-limit analyzer's concurrent
  burst.
- Finite counters never become negative or exceed their limits.
- Effective operation timeouts never exceed remaining overall time.
- Overall timeout omits advice and public payload while extraction continues.
- Monotonic deadline expiry and caller cancellation remain distinct outcomes.
- External cancellation propagates, bounded shielded cleanup receives its
  two-second grace, and forced cleanup preserves the original outcome.
- Responses, browser resources, and subprocesses close or terminate on every
  exit path.

Property-based tests cover budget invariants, timeout capping, option
normalization, and stable result-key completeness.

### Redaction Tests

Generated URLs and errors include credentials, secret-like query values,
fragments, cookies, authorization headers, and subprocess output. Assertions
verify none appear in logs, typed failures, or public analyzer errors.

### Compatibility and Architecture Tests

- Every old import path from the Phase 0 inventory resolves.
- Explicit shims expose the expected names, signatures, and result shapes.
- `inspect.signature` and `inspect.iscoroutinefunction` pin every historical
  public analyzer entry's signature and sync/async classification.
- Synchronous compatibility wrappers exercise the background-loop bridge,
  while `run_analysis` still rejects calls from an active event loop.
- An AST-based dependency guard with a small explicit allowlist rejects:
  - imports from `preflight` to either legacy scraper;
  - analyzer-internal HTTP, Playwright, or subprocess creation;
  - scraper-consumer imports of analyzer internals;
  - new application imports from `scraper_analyzers`.
- The import inventory is regenerated and reviewed after the move.

### Browser Adapter Coverage

Required tests use fake browser and routing protocols. A separately marked,
optional smoke test uses async Playwright against a local test server. It skips
when the browser extra is absent and should run in a suitable optional CI job.
It never accesses the public network.

## Rollout

Implementation proceeds in reviewable, continuously passing stages:

1. Add typed options, target, execution context, probe protocols, deterministic
   fakes, and characterization tests.
2. Add governed HTTP, browser, and external-tool adapters, raise the Playwright
   floor, and add focused capability, policy, timeout, cancellation, cleanup,
   and redaction tests.
3. Move analyzers, scoring, recommendations, and utilities into `preflight`,
   adding explicit compatibility shims.
4. Migrate `Article_Extractor_Lib` to the facade and run its compatibility
   suite.
5. Migrate `EnhancedWebScraper` to the facade and run its compatibility suite.
6. Remove duplicated orchestration, update the import inventory and docs, and
   run final architecture and security gates.

The implementation plan may assign these stages to independent review tasks,
but the Phase 3 branch is not merge-ready until both consumers use the shared
facade and all compatibility gates pass.

## Verification Gates

Completion requires:

- focused preflight unit and property tests;
- Phase 1 and Phase 2 compatibility suites;
- the broader `WebScraping` and `Web_Scraping` test suites;
- import-inventory and AST architecture checks;
- Python import/compile checks for touched modules;
- the repository's configured formatting and lint checks for touched files;
- Bandit on touched Python paths;
- no required test that accesses external network, requires a browser install,
  or executes a real external tool.

The optional local-browser smoke test is reported separately when its extra is
available.

## Risks and Mitigations

### Risk: The move changes successful analyzer behavior

Mitigation: deterministic characterization fixtures pin current successful
values, order, scoring, recommendations, and advice before implementation is
moved.

### Risk: Cancellation leaves blocking work alive

Mitigation: use native async HTTP, async Playwright, and async subprocess APIs;
reserve threads for bounded parsing only; assert cleanup on cancellation.

### Risk: Browser interception is unavailable or incomplete

Mitigation: require interception capability before browser analyzer execution
and return `unavailable` rather than using an ungoverned fallback. Cover
navigation, subresources, service workers, and WebSockets in adapter contract
tests. Document that URL routing is not DNS-pinned transport; a governed proxy
or pinned browser transport is later hardening work.

### Risk: Synchronous compatibility creates a second execution model

Mitigation: isolate the background event-loop bridge behind historical public
wrappers, pin signatures and coroutine classification in tests, and prohibit
the facade and runner from calling the bridge.

### Risk: Compatibility shims become permanent implementation homes

Mitigation: shims contain explicit re-exports only, architecture checks reject
new application imports, and Phase 7 owns removal after inventory proof.

### Risk: Public payload exposes internal failure details

Mitigation: one eligibility helper gates conversion and permits only overall
`OK` results when inclusion is enabled.

### Risk: External-tool governance is overstated

Mitigation: classify the command as one opaque active probe and document that
its internal requests cannot be intercepted in Phase 3.

### Risk: The phase grows into a new analyzer framework

Mitigation: preserve existing analyzer functions, order, outputs, and internal
rate-limit burst; defer registries, scheduling modes, fetch reuse, and public
budget configuration.

## Success Criteria

Phase 3 is successful when:

- `preflight` is the sole analyzer implementation owner.
- Both article scraping consumers use the same facade.
- Scrape-level primary policy and per-dispatch probe egress are separate,
  explicit, and tested.
- Analyzer HTTP, browser, and external-tool work uses governed adapters.
- Existing config keys and successful output behavior remain compatible.
- Analyzer failures are isolated and extraction remains fail-open.
- Cancellation propagates and all resources clean up.
- Optional payloads remain unchanged and never expose overall failures.
- Legacy imports remain functional through temporary explicit shims.
- Required tests pass without external services or optional executables.

## Spec Self-Review

- Placeholder scan: no unresolved markers or incomplete requirements remain.
- Consistency check: primary policy remains blocking; analyzer failures remain
  advisory and fail-open; overall failures are typed internally and omitted
  publicly.
- Scope check: the design is one Phase 3 migration with staged implementation;
  extraction, crawl, jobs, search, registries, scheduling, and public budget
  configuration remain outside the phase.
- Ambiguity check: config defaults, external-tool compatibility, redirect cap,
  browser interception fallback, cancellation, payload eligibility, shim
  guarantees, and verification gates are explicit.
- Review refinements incorporated: native async cancellation, browser protocol
  capability, opaque external-tool limitation, atomic burst budgets,
  deterministic characterization, structural dependency checks, centralized
  payload eligibility, property invariants, and optional real-browser smoke
  coverage are all specified.
- Final review refinements incorporated: historical sync/async callable
  classification, separate scrape and probe policy roles, Playwright 1.48 and
  service-worker requirements, URL-routing DNS limitations, monotonic deadline
  and bounded shielded cleanup semantics, and the external-tool default sunset
  signal are explicit and testable.
