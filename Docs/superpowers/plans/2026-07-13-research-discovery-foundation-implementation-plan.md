# Research Discovery Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` or `superpowers:executing-plans` to execute this plan. Use `superpowers:test-driven-development` for every behavior change and `superpowers:verification-before-completion` before each commit.

**Goal:** Deliver TASK-12968.2's offline, gateway-backed V2 discovery foundation for the existing eight-source catalog without changing production selection or consumer behavior.

**Architecture:** Keep the current V1 catalog, router, service, result IDs, and endpoint path intact. Add frozen V2 contracts, a product-owned route registry, a pure planner, a one-hop gateway facade that consumes TASK-12971, gateway-only V2 adapters, and an in-memory executor journal. Exercise V2 only with frozen fixtures and synthetic inputs; TASK-12968.3 owns production standalone cutover and TASK-12968.4 owns durable Deep Research journaling.

**Tech Stack:** Python 3.10+, frozen standard-library dataclasses, asyncio, existing FastAPI/Pydantic surfaces only where already required, pytest, Hypothesis where installed, explicit immutable discovery route policy, side-effect-free Security normalization helpers, and the public one-hop primitive delivered by TASK-12971.

## Global Constraints

- Work only in the isolated feature worktree; never use the dirty root worktree.
- TASK-12968.1, TASK-12971, and TASK-12968.7 must be complete before runtime edits begin.
- Do not wrap `afetch_json`; it does not satisfy connected-peer or streaming-limit requirements.
- Do not modify V1 `DiscoveryResult.result_id`, canonical fingerprint behavior, legacy evidence IDs, absent-field selection semantics, current endpoint envelopes, or active Deep Research runs.
- Do not enable V2 in production, shadow real user traffic, or double-fetch a production query. V2 execution is frozen-fixture, synthetic, or explicitly opted-in only.
- Do not add page retrieval, result-link dereference, Media ingestion, cookies, credentials, Playwright, authenticated scraping, ambient proxy state, or a generic crawler.
- Every initial request, pagination request, redirect hop, or retry is one separately reserved dispatch with a fresh `dispatch_id`, fresh route-policy validation, and its own debit. No layer may hide automatic redirects, retries, or pages.
- Freeze independent ceilings for route attempts, physical dispatches, pages per route, redirects, retries, aggregate wall time, and returned results. A failure or post-dispatch cancellation still consumes its dispatch budget; only a definitely unused pre-dispatch reservation may be released.
- Cursors are untrusted provider data. Reconstruct the next request from an approved route template, enforce the same origin and query schema, reject absolute cursor URLs, and stop at the declared page and aggregate deadlines.
- Retrying requires an idempotent method plus an explicitly retryable outcome; otherwise fail without a second dispatch. Duplicate-work risk is reported rather than hidden.
- The OpenAlex V2 route is `api_key` gated under current provider policy. TASK-12968.2 keeps its foundation declaration secret-free and typed unavailable/skipped, with zero executable attempts, physical reservations, or gateway calls. Do not add a secret-reference field or positive credentialed branch; authenticated OpenAlex enablement belongs to a separately authorized future program. V1 behavior is unchanged by this plan.
- Keep the first implementation boring: immutable values, explicit functions, injected dependencies, and an in-memory journal. Durable attempt persistence belongs to TASK-12968.4.

## Execution Preflight

- [x] Read the official Backlog task-execution workflow, confirm TASK-12968.1, TASK-12971, and TASK-12968.7 are complete, move TASK-12968.2 to In Progress, and link this plan before the first runtime edit.
- [x] Verify the worktree is isolated and clean apart from the approved task/plan changes.
- [x] Re-read the TASK-12971 public transport contract and exact focused-test paths; update only the Stage 3 import/test path if its delivered public name differs.

## Shared Gate Before Every Commit

Before every stage commit, run all tests introduced in that stage and every earlier stage, the impacted existing discovery tests, `python -m compileall -q` on touched Python paths, `ruff check` and `black --check` on every touched Python file, and `git diff --check`. Fix failures before committing; do not defer these gates to Stage 5.

## Task 1 / Stage 1: Freeze Legacy Execution Behavior

**Goal:** Characterize what the current eight-source path actually does before adding a parallel V2 path.

**Success Criteria:** A checked-in golden fixture and focused tests freeze selection, provider invocation, ordering, warnings, source statuses, partial/all-failure behavior, and serialized response projections without network access.

**Tests:** Existing legacy selection contract plus new recording-provider execution characterization.

**Status:** Complete

### Files

- Add `Docs/Design/research_source_inventory/research-discovery-legacy-execution-v1.json`.
- Add `tldw_Server_API/tests/Research/test_research_discovery_legacy_execution_contract.py`.
- Update `tldw_Server_API/tests/Research/test_research_broker.py` with hard-coded legacy `src_...` and `note_...` identity goldens.
- Read but do not change `tldw_Server_API/app/core/Research/discovery/catalog.py`, `router.py`, `service.py`, `models.py`, and `identity.py` unless the characterization exposes a defect that receives separate approval.

### Test-first steps

- [x] Write tests with injected deterministic adapters for all eight current sources.
- [x] Inject a deterministic no-I/O OA resolver and a tripwire that would fail if the default resolver were constructed or called.
- [x] Cover omitted, empty, explicit-source, category-only, and source/category-union selection through the real V1 service boundary.
- [x] Preserve the observed V1 semantics: omitted or empty selection defaults to the three `open_research_graph` sources, result aggregation follows source priority rather than adapter completion order, and any malformed item makes that source `internal_error`.
- [x] Pin source priority, result ordering, provider call arguments/counts, source status ordering, warning ordering, partial failure, all failure, valid-empty, malformed provider payload, and a documented stable serialization projection.
- [x] Omit only volatile `SourceStatus.elapsed_ms`, `DiscoveryMetrics.elapsed_ms`, and the generated discovery ID from the golden projection. Assert exact ID shape, snapshot-ID equality, elapsed types/ranges, and timezone-aware snapshot timestamp ordering/retention separately.
- [x] Serialize through `ResearchDiscoverySearchResponse.model_validate(...).model_dump(mode="json")`, freeze every remaining public field, and assert the same projection from the persisted snapshot together with its persisted request and effective configuration.
- [x] Add fixed-input hard-coded Research broker goldens through `collect_focus_area`: local `src_7aef3f6cf7e9` / `note_9c1e03fe5b64`, academic `src_72b0a1007cc7` / `note_3fd24e3f8693`, and web `src_47fc41c2bc20` / `note_817ae2d4c1e1`. Do not freeze `retrieved_at`; parse it as timezone-aware and bound it to the collection interval.
- [x] Generate the golden JSON mechanically from reviewed stable expected values, then compare canonical sorted-key JSON with one trailing newline.
- [x] Run the existing discovery catalog/router/service/selection/identity suites to prove the fixture describes current behavior rather than changing it.

### Verify

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest -q \
  tldw_Server_API/tests/Research/test_research_discovery_legacy_selection_contract.py \
  tldw_Server_API/tests/Research/test_research_discovery_legacy_execution_contract.py \
  tldw_Server_API/tests/Research/test_research_discovery_catalog.py \
  tldw_Server_API/tests/Research/test_research_discovery_router.py \
  tldw_Server_API/tests/Research/test_research_discovery_service.py \
  tldw_Server_API/tests/Research/test_research_discovery_selection.py \
  tldw_Server_API/tests/Research/test_research_discovery_identity.py \
  tldw_Server_API/tests/Research/test_research_broker.py
python -m compileall -q \
  tldw_Server_API/tests/Research/test_research_discovery_legacy_execution_contract.py \
  tldw_Server_API/tests/Research/test_research_broker.py
ruff check \
  tldw_Server_API/tests/Research/test_research_discovery_legacy_execution_contract.py \
  tldw_Server_API/tests/Research/test_research_broker.py
black --check \
  tldw_Server_API/tests/Research/test_research_discovery_legacy_execution_contract.py \
  tldw_Server_API/tests/Research/test_research_broker.py
git diff --check
```

### Commit

`test(research): freeze legacy discovery execution contract`

Independent-review hardening: `test(research): harden legacy execution golden`

## Task 2 / Stage 2: Add Pure V2 Contracts, Registry, and Planner

**Goal:** Compile source intent into a deterministic, budgeted effective plan without performing I/O or touching V1 identities.

**Success Criteria:** Frozen V2 values model source, route, backend, policy, attempt, dispatch intent, dispatch allowance, source predicate, readiness, provenance, budget, and outcome identity; the registry supports aliases and multiple routes; the planner deterministically coalesces compatible backend work and rejects impossible budgets without creating runtime journal reservations.

**Tests:** Unit and property tests for construction, referential integrity, determinism, coalescing, fallback order, and physical-request budgets.

**Status:** Complete

### Files

- Add `tldw_Server_API/app/core/Research/discovery/contracts.py`.
- Add `tldw_Server_API/app/core/Research/discovery/registry.py`.
- Add `tldw_Server_API/app/core/Research/discovery/planner.py`.
- Add `tldw_Server_API/tests/Research/test_research_discovery_contracts.py`.
- Add `tldw_Server_API/tests/Research/test_research_discovery_planner.py`.
- Add `tldw_Server_API/tests/Research/test_research_discovery_registry_reconciliation.py`.

### Test-first steps

- [x] Start with failing constructor tests for frozen, slots-based dataclasses and validated catalog IDs, route references, policy digests, exact origins, query modes, typed source predicates, credential requirements, immutable readiness overlays, and explicit `offline_fixture` or `synthetic` execution modes. There is no production-default execution mode.
- [x] Define a frozen `DispatchIntent` containing route ID, policy digest, operation kind, method, path, typed query pairs, and limits. It describes work but cannot dispatch or debit work itself.
- [x] Add an additive route-independent V2 document identity; assert existing `build_fingerprint`, `stable_result_id`, serialized V1 result IDs, and Deep Research evidence IDs remain byte-for-byte unchanged.
- [x] Build a V2 registry for the eight existing targets without mutating `default_source_catalog()`. Record exact routes/backends and make legacy `site_hosts` descriptive only.
- [x] Mark OpenAlex's V2 API route credentialed and typed unavailable/skipped, carry no secret material or secret-reference interface in the foundation registry, and never inherit the stale V1 `requires_credentials=False` claim. Leave V1 unchanged.
- [x] Start with an OpenAlex V2 selection regression: the planner returns a typed unavailable/skipped outcome and emits zero executable attempts and zero dispatch allowance. The zero-gateway-call tripwire belongs to Stage 4E, after the gateway boundary exists. Do not add a positive credentialed branch; authenticated enablement is deferred to a separately authorized future program.
- [x] Reconcile all eight existing targets (arXiv, PubMed, Semantic Scholar, Zenodo, OpenAlex, OSF, Figshare, and Crossref) against the frozen ledger's target, route, backend, credential, query-mode, and source-predicate declarations. The Crossref Metadata Search seed row resolves to the existing stable `crossref` product ID; do not create a parallel `crossref_metadata_search` runtime identity.
- [x] Compile explicit V2 selections plus an immutable readiness overlay into stable ordered attempts with selection reasons, route fallback order, policy/catalog versions, and declared dispatch allowances. Mark the seven credentialless routes fixture-executable and OpenAlex unavailable.
- [x] Coalesce only requests whose backend, normalized query, filters, policy, and source predicates are actually compatible. Preserve requested-target attribution separately from backend identity.
- [x] Count PubMed ESearch plus conditional ESummary as a deterministic allowance of at most two physical dispatches. An empty ESearch leaves unused allowance; it does not create or release an `AttemptJournal` reservation.
- [x] Model and report separate ceilings for route attempts, physical dispatches, per-route pages, redirects, retries, aggregate wall time, and returned results.
- [x] Add a synthetic two-target shared-backend aggregator to tests so coalescing plus matching, nonmatching, and ambiguous source predicates are exercised even though the eight production foundation routes are direct/native.
- [x] Add Hypothesis invariants where available: deterministic plan bytes, no negative allowance, coalescing never increases physical requests, and no planned dimension exceeds its declared ceiling. Runtime journal accounting invariants belong to Stage 4A.

### Verify

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest -q \
  tldw_Server_API/tests/Research/test_research_discovery_contracts.py \
  tldw_Server_API/tests/Research/test_research_discovery_planner.py \
  tldw_Server_API/tests/Research/test_research_discovery_registry_reconciliation.py \
  tldw_Server_API/tests/Research/test_research_discovery_identity.py \
  tldw_Server_API/tests/Research/test_research_broker.py \
  tldw_Server_API/tests/Research/test_research_discovery_legacy_execution_contract.py
python -m compileall -q \
  tldw_Server_API/app/core/Research/discovery/contracts.py \
  tldw_Server_API/app/core/Research/discovery/registry.py \
  tldw_Server_API/app/core/Research/discovery/planner.py \
  tldw_Server_API/tests/Research/test_research_discovery_contracts.py \
  tldw_Server_API/tests/Research/test_research_discovery_planner.py \
  tldw_Server_API/tests/Research/test_research_discovery_registry_reconciliation.py
ruff check \
  tldw_Server_API/app/core/Research/discovery/contracts.py \
  tldw_Server_API/app/core/Research/discovery/registry.py \
  tldw_Server_API/app/core/Research/discovery/planner.py \
  tldw_Server_API/tests/Research/test_research_discovery_contracts.py \
  tldw_Server_API/tests/Research/test_research_discovery_planner.py \
  tldw_Server_API/tests/Research/test_research_discovery_registry_reconciliation.py
black --check \
  tldw_Server_API/app/core/Research/discovery/contracts.py \
  tldw_Server_API/app/core/Research/discovery/registry.py \
  tldw_Server_API/app/core/Research/discovery/planner.py \
  tldw_Server_API/tests/Research/test_research_discovery_contracts.py \
  tldw_Server_API/tests/Research/test_research_discovery_planner.py \
  tldw_Server_API/tests/Research/test_research_discovery_registry_reconciliation.py
git diff --check
```

### Commit

`feat(research): add pure discovery v2 planning contracts`

### Completion evidence

- RED: the initial focused suite failed during collection because the V2 contract modules did not exist. Review regressions then failed for attribution-only coalescing, nested type safety, complete physical-dispatch accounting, zero-work result allowances, import side effects, legacy lazy-submodule compatibility, and zero-page attempt accounting.
- GREEN: 57 focused Task 2 tests, 97 exact plan tests, 107 impacted package/import tests, and the complete 574-test Research suite passed. The complete Research suite preceded the final lazy-submodule edge hardening; the subsequent focused, exact, and impacted suites cover those final changes.
- Quality gates: compileall, Ruff, Black, Python 3.10 AST parsing, and `git diff --check` passed. Bandit reported zero findings and zero errors across 1,487 production lines.
- Independent review: five Important findings and two final compatibility/accounting edge cases were fixed RED-first; the same reviewer returned a final CLEAN verdict.
- External-review fix RED: 2/2 predicate regressions failed before canonicalization, and the consolidated remaining suite stopped on the absent logical-attempt/dispatch-group contract boundary before production edits.
- External-review fix GREEN: 64/64 focused contract/planner tests, 115/115 exact plan tests, 200/200 impacted package/jobs/endpoint tests, and 594/594 complete Research tests passed on the settled diff.
- External-review fix gates: compileall, Ruff, Black, Python 3.10 AST parsing, and diff hygiene passed; Bandit reported zero findings and zero errors across 1,161 touched production lines. Task 2 remains In Progress pending external controller re-review.
- Minor-fix RED/GREEN: three synthetic imports proved the purity scanner permitted the delivered `Security.http_hop` facade before the test-only rule; all three then passed alongside the existing static and subprocess purity guards.
- Minor-fix gates: 78/78 contracts/planner/registry tests, compileall, Ruff, Black, and diff hygiene passed. Bandit is not applicable because no production Python changed. Task 2 remains In Progress pending controller re-review.

## Task 3 / Stage 3: Consume TASK-12971 Through a One-Hop Discovery Gateway

**Goal:** Apply discovery route policy to exactly one separately accounted physical hop using the reusable secure transport primitive.

**Success Criteria:** The discovery gateway validates a frozen route request, delegates one hop to TASK-12971, returns sanitized typed transport evidence, and never follows a redirect or performs a retry itself.

**Tests:** Unit, security, cancellation, revocation, and streaming-boundary tests with an injected fake one-hop primitive; focused integration tests against TASK-12971's local test server only.

**Status:** Complete

### Files

- Add `tldw_Server_API/app/core/Research/discovery/gateway.py`.
- Add `tldw_Server_API/tests/Research/test_research_discovery_gateway.py`.
- Import `HTTPHopLimits`, `NormalizedHTTPHopRequest`, `HTTPHopResponse`, `HTTPHopError`, and `request_http_hop` from `tldw_Server_API.app.core.Security.http_hop`; do not copy its transport code into Research or import its private resolver/backend seams.
- Do not call `tldw_Server_API/app/core/Security/egress.py:evaluate_url_policy`: it resolves DNS and reads ambient egress configuration. Reuse only side-effect-free normalization helpers whose inputs are explicit. TASK-12971 exclusively owns DNS-answer and connected-peer enforcement.
- Do not reuse Web Scraping `FetchRequest` defaults that permit cookies, proxies, or redirects. The discovery request contract is explicit and narrower.

### Test-first steps

- [x] Stop if TASK-12971 is not complete or its focused security tests are not green; record the blocker instead of implementing an alternate client.
- [x] Write failing tests for exact scheme/host/port/method/path/query enforcement, canonical policy-digest recomputation and binding, revocation before dispatch, query minimization, and independence from environment/config-file allowlists.
- [x] Define `dispatch_once(...)` so one call can perform one physical hop only. Redirect responses and retryable failures return typed data to the executor; they are not followed internally.
- [x] Prove no ambient proxy, `.netrc`, cookie, authorization, client certificate, or credential state enters the primitive request.
- [x] Preserve TASK-12971's exposed resolved-address, connected-peer, header-byte, and wire-byte evidence. Derive requested Host/SNI and configured ceilings from the validated request, decoded bytes from bounded `len(body)`, and elapsed time around the public call; do not import private TASK-12971 seams or claim fields the public response does not expose.
- [x] Return bounded typed errors without query text, response bodies, local paths, secrets, or unsafe provider detail.

### Verify

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest -q \
  tldw_Server_API/tests/Research/test_research_discovery_gateway.py \
  tldw_Server_API/tests/Security/test_http_hop_contract.py \
  tldw_Server_API/tests/Security/test_http_hop_transport.py \
  tldw_Server_API/tests/Security/test_http_hop_streaming.py \
  tldw_Server_API/tests/Research/test_research_discovery_contracts.py \
  tldw_Server_API/tests/Research/test_research_discovery_planner.py \
  tldw_Server_API/tests/Research/test_research_discovery_registry_reconciliation.py \
  tldw_Server_API/tests/Research/test_research_discovery_legacy_execution_contract.py
python -m compileall -q \
  tldw_Server_API/app/core/Research/discovery/gateway.py \
  tldw_Server_API/tests/Research/test_research_discovery_gateway.py
ruff check \
  tldw_Server_API/app/core/Research/discovery/gateway.py \
  tldw_Server_API/tests/Research/test_research_discovery_gateway.py
black --check \
  tldw_Server_API/app/core/Research/discovery/gateway.py \
  tldw_Server_API/tests/Research/test_research_discovery_gateway.py
git diff --check
```

### Commit

`feat(research): add one-hop discovery gateway facade`

## Task 4 / Stage 4A: Add the Executor-Owned Dispatch Boundary and Journal

**Goal:** Make one component solely responsible for reservation, policy revalidation, dispatch IDs, gateway calls, debits, cancellation, and candidate commit.

**Success Criteria:** Adapters receive only an executor-owned `dispatch(intent)` capability bound to one planned attempt, never the raw gateway; every physical operation is journaled exactly once; allowances remain distinct from runtime reservations and debits; and scripted offline tests cover every state transition and budget dimension.

**Status:** Not Started

### Files

- Add `tldw_Server_API/app/core/Research/discovery/executor.py`.
- Add `tldw_Server_API/tests/Research/test_research_discovery_executor.py`.

### Test-first steps

- [ ] Use a scripted adapter to prove it can request work only by yielding a validated `DispatchIntent` to executor-owned `dispatch(intent)`; it cannot access the gateway directly.
- [ ] Bind each dispatch capability to one planned `dispatch_group_id`, route ID, policy digest, and remaining allowance. Reject cross-group/cross-route intents and undeclared operation transitions; include a malicious scripted-adapter regression.
- [ ] Add a minimal in-memory `AttemptJournal` with explicit `reserved`, `dispatching`, `succeeded`, `valid_empty`, `failed`, `timed_out`, `cancelled`, `skipped`, and `indeterminate_after_dispatch` transitions. Do not add durable storage.
- [ ] Create a journal reservation only immediately before an initial, page, redirect, or retry dispatch; assign a fresh `dispatch_id`; debit on transition to `dispatching`; and release only a definitely unused pre-dispatch reservation. Unused planner allowance is not a journal transition.
- [ ] Inject fail-closed `policy_is_active(route_id, digest)`, recompute and verify the canonical digest before dispatch, and recheck before committing candidates. Test tampered digests and revocation during the hop: accounting remains debited while candidate content is suppressed.
- [ ] Reconstruct pagination from typed cursor fields into the approved route template. Reject absolute cursor URLs, cross-origin cursors, undeclared query keys, repeated cursors, and work beyond per-route page/redirect/retry ceilings or the aggregate deadline.
- [ ] Permit retries only for explicitly retryable outcomes on idempotent methods. Debit failed dispatched work and surface possible duplicate work; never release or silently replay a dispatch that may have reached the provider.
- [ ] Enforce route-attempt, physical-dispatch, page, redirect, retry, aggregate-wall-time, and returned-result budgets independently. Apply a deterministic result cap before candidate commit and report truncation.
- [ ] Test cancellation before dispatch (reservation released), while dispatching with no definitive result (`indeterminate_after_dispatch`, debit retained), and after a definitive result (debit retained; late content suppressed if cancellation wins the commit boundary).
- [ ] Add accounting invariants: used plus released plus outstanding equals reserved, every dispatching record has a unique ID, and no counter exceeds its ceiling.

### Verify

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest -q \
  tldw_Server_API/tests/Research/test_research_discovery_executor.py \
  tldw_Server_API/tests/Research/test_research_discovery_gateway.py \
  tldw_Server_API/tests/Research/test_research_discovery_contracts.py \
  tldw_Server_API/tests/Research/test_research_discovery_planner.py
python -m compileall -q \
  tldw_Server_API/app/core/Research/discovery/executor.py \
  tldw_Server_API/tests/Research/test_research_discovery_executor.py
ruff check tldw_Server_API/app/core/Research/discovery/executor.py tldw_Server_API/tests/Research/test_research_discovery_executor.py
black --check tldw_Server_API/app/core/Research/discovery/executor.py tldw_Server_API/tests/Research/test_research_discovery_executor.py
git diff --check
```

### Commit

`feat(research): add accounted discovery executor`

## Task 5 / Stage 4B: Add Five Bounded JSON Adapters

**Goal:** Implement the simple JSON protocols for Semantic Scholar, Crossref, Zenodo, Figshare, and OSF over the executor-owned dispatch capability.

**Success Criteria:** The five adapters parse bounded offline fixtures, return inert metadata/snippets, and have no raw transport or legacy-wrapper access.

**Status:** Not Started

### Files

- Add `tldw_Server_API/app/core/Research/discovery/gateway_adapters.py`.
- Add `tldw_Server_API/tests/Research/test_research_discovery_gateway_adapters.py`.
- Add sanitized provider fixtures under the existing Research fixture convention; do not create a second framework.

### Test-first steps

- [ ] Cover success, valid empty, bounded pagination, rate-limit metadata, timeout, cancellation, malformed/schema-drift payloads, unsafe echoed errors, and inert result URLs.
- [ ] Enforce route-level maximum record count, field characters, structural depth, and parse deadline before normalized output; include deeply nested and oversized fixtures.
- [ ] Rebuild cursors only through typed fields and approved query templates; never accept an absolute next URL.
- [ ] Prove every adapter uses only executor-owned `dispatch(intent)` and that returned URLs receive zero requests.

### Verify

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest -q tldw_Server_API/tests/Research/test_research_discovery_gateway_adapters.py tldw_Server_API/tests/Research/test_research_discovery_executor.py
python -m compileall -q tldw_Server_API/app/core/Research/discovery/gateway_adapters.py tldw_Server_API/tests/Research/test_research_discovery_gateway_adapters.py
ruff check tldw_Server_API/app/core/Research/discovery/gateway_adapters.py tldw_Server_API/tests/Research/test_research_discovery_gateway_adapters.py
black --check tldw_Server_API/app/core/Research/discovery/gateway_adapters.py tldw_Server_API/tests/Research/test_research_discovery_gateway_adapters.py
git diff --check
```

### Commit

`feat(research): add bounded json discovery adapters`

## Task 6 / Stage 4C: Add the Bounded arXiv Atom Adapter

**Goal:** Parse arXiv Atom responses directly over accounted dispatches without importing the legacy wrapper that hides fetch and sleep behavior.

**Success Criteria:** arXiv fixtures cover bounded success, empty, pagination, malformed XML, excessive structure/fields/records, deadline, and safe error behavior.

**Status:** Not Started

### Test-first steps

- [ ] Add arXiv fixtures and failing adapter tests before implementation.
- [ ] Reject entity-expansion-style input and enforce structural depth, record, field, input-byte, and parse-deadline ceilings.
- [ ] Reconstruct pagination from approved numeric cursor fields; do not consume provider-supplied absolute links.
- [ ] Prove result links are inert and every physical request is mediated by executor-owned `dispatch(intent)`.

### Verify

Run the Task 5 verify commands plus the focused executor, planner, and gateway suites.

### Commit

`feat(research): add bounded arxiv discovery adapter`

## Task 7 / Stage 4D: Add the Two-Dispatch PubMed Adapter

**Goal:** Make PubMed's ESearch and conditional ESummary sequence explicit and truthfully accounted.

**Success Criteria:** ESearch and ESummary are separate intents, reservations, and dispatch IDs; an empty ESearch performs no second reservation or gateway call; and every limit/cancellation/error path is typed and deterministic.

**Status:** Not Started

### Test-first steps

- [ ] Start with failing fixtures for nonempty ESearch plus ESummary, empty ESearch, malformed ID lists, partial summaries, pagination, timeout, cancellation, and unsafe errors.
- [ ] Prove the two-step path consumes at most the planner's two-dispatch allowance while creating journal reservations only immediately before actual calls.
- [ ] Apply the XML/parser bounds and cursor rules from Task 6 to both operations.
- [ ] Do not import the legacy PubMed wrapper or hide the second call inside an SDK/default transport.

### Verify

Run the Task 5 verify commands plus the focused executor, planner, registry-reconciliation, and gateway suites.

### Commit

`feat(research): add accounted pubmed discovery adapter`

## Task 8 / Stage 4E: Prove Registry and Network Boundaries End to End

**Goal:** Close the security boundary over the complete offline V2 registry and exercise attribution/coalescing through planner, executor, and adapters.

**Success Criteria:** The seven enabled adapters exactly equal the recording-tested and statically scanned implementations; OpenAlex remains unavailable with zero calls; and no alternate transport, credential, scraping, Media, or result-dereference path exists.

**Status:** Not Started

### Files

- Add `tldw_Server_API/tests/Research/test_research_discovery_network_boundary.py`.
- Update focused executor/adapter tests only where needed for end-to-end coverage.

### Test-first steps

- [ ] Derive enabled adapter identities/modules from the registry and assert exact set equality with recording fixtures and statically scanned modules; fail if any enabled adapter is outside the allowlist.
- [ ] Add an AST test scanning every new V2 production module (`contracts`, `registry`, `planner`, `executor`, and adapters) and rejecting imports/construction of `httpx`, `aiohttp`, `requests`, `urllib`, sockets, SDK default transports, legacy Third_Party networking, and `Security.http_hop`. Only `gateway.py` may consume TASK-12971's transport API.
- [ ] Assert OpenAlex produces a typed unavailable/skipped result with zero executable attempts, allowances, journal reservations, and gateway calls; no OpenAlex adapter exists.
- [ ] Run the synthetic shared-backend aggregator end to end and prove matching, nonmatching, ambiguous attribution, and one coalesced physical request.
- [ ] Prove provider failures and result URLs make zero calls to generic fetch, Media, scraping, Playwright, cookies, secrets, or any unregistered adapter.

### Verify

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest -q \
  tldw_Server_API/tests/Research/test_research_discovery_gateway_adapters.py \
  tldw_Server_API/tests/Research/test_research_discovery_executor.py \
  tldw_Server_API/tests/Research/test_research_discovery_network_boundary.py \
  tldw_Server_API/tests/Research/test_research_discovery_gateway.py \
  tldw_Server_API/tests/Research/test_research_discovery_contracts.py \
  tldw_Server_API/tests/Research/test_research_discovery_planner.py \
  tldw_Server_API/tests/Research/test_research_discovery_registry_reconciliation.py \
  tldw_Server_API/tests/Research/test_research_discovery_legacy_execution_contract.py
python -m compileall -q tldw_Server_API/app/core/Research/discovery tldw_Server_API/tests/Research
ruff check tldw_Server_API/app/core/Research/discovery tldw_Server_API/tests/Research/test_research_discovery_gateway_adapters.py tldw_Server_API/tests/Research/test_research_discovery_executor.py tldw_Server_API/tests/Research/test_research_discovery_network_boundary.py
black --check tldw_Server_API/app/core/Research/discovery tldw_Server_API/tests/Research/test_research_discovery_gateway_adapters.py tldw_Server_API/tests/Research/test_research_discovery_executor.py tldw_Server_API/tests/Research/test_research_discovery_network_boundary.py
git diff --check
```

### Commit

`test(research): prove discovery v2 network boundaries`

## Task 9 / Stage 5: Prove Offline Compatibility and Leave Production Disabled

**Goal:** Demonstrate a reviewable foundation without claiming consumer cutover or live source delivery.

**Success Criteria:** The frozen V1 suite remains unchanged, V2 fixture projections are explainably equivalent where contracts overlap, differences are additive and documented, all V2 execution remains opt-in/offline, and focused security/quality gates pass.

**Tests:** Full Research Discovery regression, offline golden comparison, no-production-double-fetch assertion, compile, Bandit, and diff hygiene.

**Status:** Not Started

### Files

- Update `Docs/Design/2026-07-13-research-source-coverage-shared-discovery-design.md` only for material implementation decisions that differ from the approved contract.
- Add `tldw_Server_API/tests/Research/test_research_discovery_v2_compatibility.py`.
- Update TASK-12968.2 through Backlog.md with touched files, verification, known skips, and final summary.
- Do not change standalone endpoint wiring or Deep Research collection wiring.

### Test-first steps

- [ ] Define a stable client-visible compatibility projection, then compare V1 and V2 fixture executions for source ordering, normalized content, status meaning, warnings, and legacy serialization. Compare V1 logical adapter calls separately from V2 physical dispatches: PubMed's one legacy wrapper call intentionally becomes two physical requests, and OpenAlex readiness intentionally becomes credential-gated.
- [ ] Re-run the hard-coded Research broker `src_...` and `note_...` identity goldens alongside the compatibility projection.
- [ ] Add an import-boundary test proving V1 production catalog/router/service/endpoint modules do not import or construct the V2 executor, plus a construction test proving V2 requires an explicit offline/synthetic opt-in.
- [ ] Execute one real legacy service or endpoint request with a raising V2 executor/gateway tripwire and recording V1 adapters. Assert the expected legacy provider counts and zero V2 calls so the no-double-fetch claim exercises a production entry point.
- [ ] Run the complete existing Research Discovery suite plus all new foundation tests.
- [ ] Run Bandit on the touched discovery and security integration scope; fix new findings rather than suppressing them.
- [ ] Request an independent correctness/security review and resolve all important findings before completion.

### Verify

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest -q tldw_Server_API/tests/Research
python -m compileall -q tldw_Server_API/app/core/Research/discovery
ruff check \
  tldw_Server_API/app/core/Research/discovery/contracts.py \
  tldw_Server_API/app/core/Research/discovery/registry.py \
  tldw_Server_API/app/core/Research/discovery/planner.py \
  tldw_Server_API/app/core/Research/discovery/gateway.py \
  tldw_Server_API/app/core/Research/discovery/gateway_adapters.py \
  tldw_Server_API/app/core/Research/discovery/executor.py \
  tldw_Server_API/tests/Research/test_research_discovery_legacy_execution_contract.py \
  tldw_Server_API/tests/Research/test_research_discovery_contracts.py \
  tldw_Server_API/tests/Research/test_research_discovery_planner.py \
  tldw_Server_API/tests/Research/test_research_discovery_registry_reconciliation.py \
  tldw_Server_API/tests/Research/test_research_discovery_gateway.py \
  tldw_Server_API/tests/Research/test_research_discovery_gateway_adapters.py \
  tldw_Server_API/tests/Research/test_research_discovery_executor.py \
  tldw_Server_API/tests/Research/test_research_discovery_network_boundary.py \
  tldw_Server_API/tests/Research/test_research_discovery_v2_compatibility.py
black --check \
  tldw_Server_API/app/core/Research/discovery/contracts.py \
  tldw_Server_API/app/core/Research/discovery/registry.py \
  tldw_Server_API/app/core/Research/discovery/planner.py \
  tldw_Server_API/app/core/Research/discovery/gateway.py \
  tldw_Server_API/app/core/Research/discovery/gateway_adapters.py \
  tldw_Server_API/app/core/Research/discovery/executor.py \
  tldw_Server_API/tests/Research/test_research_discovery_legacy_execution_contract.py \
  tldw_Server_API/tests/Research/test_research_discovery_contracts.py \
  tldw_Server_API/tests/Research/test_research_discovery_planner.py \
  tldw_Server_API/tests/Research/test_research_discovery_registry_reconciliation.py \
  tldw_Server_API/tests/Research/test_research_discovery_gateway.py \
  tldw_Server_API/tests/Research/test_research_discovery_gateway_adapters.py \
  tldw_Server_API/tests/Research/test_research_discovery_executor.py \
  tldw_Server_API/tests/Research/test_research_discovery_network_boundary.py \
  tldw_Server_API/tests/Research/test_research_discovery_v2_compatibility.py
python -m bandit -r \
  tldw_Server_API/app/core/Research/discovery \
  -f json -o /tmp/bandit_TASK-12968.2.json
git diff --check
```

Expected: all focused tests pass, Bandit reports no new finding in touched code, V2 remains disabled in production, no real user query is double-fetched, and no result link is dereferenced.

### Commit

`test(research): verify offline discovery v2 foundation`

## Completion Boundary

Completing this plan does not mean 191 sources are shipped. It delivers a safe execution foundation for the eight existing catalog targets, with OpenAlex honestly API-key gated. TASK-12968.5 and TASK-12968.6 add the first new credentialless route families, TASK-12968.3 performs standalone cutover and large-catalog UX, and TASK-12968.4 performs new-session Deep Research integration and durable resume behavior.
