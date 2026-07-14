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
- The OpenAlex V2 route is `api_key` gated under current provider policy. It may be fixture-tested but is unavailable by default without an explicit future credential reference.
- Keep the first implementation boring: immutable values, explicit functions, injected dependencies, and an in-memory journal. Durable attempt persistence belongs to TASK-12968.4.

## Execution Preflight

- [ ] Read the official Backlog task-execution workflow, confirm TASK-12968.1, TASK-12971, and TASK-12968.7 are complete, move TASK-12968.2 to In Progress, and link this plan before the first runtime edit.
- [ ] Verify the worktree is isolated and clean apart from the approved task/plan changes.
- [ ] Re-read the TASK-12971 public transport contract and exact focused-test paths; update only the Stage 3 import/test path if its delivered public name differs.

## Shared Gate Before Every Commit

Before every stage commit, run all tests introduced in that stage and every earlier stage, the impacted existing discovery tests, `python -m compileall -q` on touched Python paths, `ruff check` and `black --check` on every touched Python file, and `git diff --check`. Fix failures before committing; do not defer these gates to Stage 5.

## Stage 1: Freeze Legacy Execution Behavior

**Goal:** Characterize what the current eight-source path actually does before adding a parallel V2 path.

**Success Criteria:** A checked-in golden fixture and focused tests freeze selection, provider invocation, ordering, warnings, source statuses, partial/all-failure behavior, and serialized response projections without network access.

**Tests:** Existing legacy selection contract plus new fake-provider execution characterization.

**Status:** Not Started

### Files

- Add `Docs/Design/research_source_inventory/research-discovery-legacy-execution-v1.json`.
- Add `tldw_Server_API/tests/Research/test_research_discovery_legacy_execution_contract.py`.
- Update `tldw_Server_API/tests/Research/test_research_broker.py` with hard-coded legacy `src_...` and `note_...` identity goldens.
- Read but do not change `tldw_Server_API/app/core/Research/discovery/catalog.py`, `router.py`, `service.py`, `models.py`, and `identity.py` unless the characterization exposes a defect that receives separate approval.

### Test-first steps

- [ ] Write tests with injected deterministic adapters for all eight current sources.
- [ ] Cover omitted, empty, explicit-source, category-only, and source/category-union selection through the real V1 service boundary.
- [ ] Pin source priority, result ordering, provider call arguments/counts, source status ordering, warning ordering, partial failure, all failure, valid-empty, malformed provider payload, and a documented stable serialization projection.
- [ ] Freeze the injected clock or omit volatile `SourceStatus.elapsed_ms`, `DiscoveryMetrics.elapsed_ms`, request IDs, and generated timestamps from the golden projection. Assert their types/ranges separately.
- [ ] Add fixed-input, hard-coded expected `src_...` source IDs and `note_...` evidence IDs through the real Research broker identity path.
- [ ] Generate the golden JSON mechanically from reviewed stable expected values, then make the test compare byte-stable canonical content.
- [ ] Run the existing discovery catalog/router/service/selection/identity suites to prove the fixture describes current behavior rather than changing it.

### Verify

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest -q \
  tldw_Server_API/tests/Research/test_research_discovery_legacy_selection_contract.py \
  tldw_Server_API/tests/Research/test_research_discovery_legacy_execution_contract.py \
  tldw_Server_API/tests/Research/test_research_discovery_catalog.py \
  tldw_Server_API/tests/Research/test_research_discovery_router.py \
  tldw_Server_API/tests/Research/test_research_discovery_service.py \
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

## Stage 2: Add Pure V2 Contracts, Registry, and Planner

**Goal:** Compile source intent into a deterministic, budgeted effective plan without performing I/O or touching V1 identities.

**Success Criteria:** Frozen V2 values model source, route, backend, policy, attempt, dispatch reservation, source predicate, provenance, budget, and outcome identity; the registry supports aliases and multiple routes; the planner deterministically coalesces compatible backend work and rejects impossible budgets.

**Tests:** Unit and property tests for construction, referential integrity, determinism, coalescing, fallback order, and physical-request budgets.

**Status:** Not Started

### Files

- Add `tldw_Server_API/app/core/Research/discovery/contracts.py`.
- Add `tldw_Server_API/app/core/Research/discovery/registry.py`.
- Add `tldw_Server_API/app/core/Research/discovery/planner.py`.
- Add `tldw_Server_API/tests/Research/test_research_discovery_contracts.py`.
- Add `tldw_Server_API/tests/Research/test_research_discovery_planner.py`.
- Add `tldw_Server_API/tests/Research/test_research_discovery_registry_reconciliation.py`.

### Test-first steps

- [ ] Start with failing constructor tests for frozen, slots-based dataclasses and validated catalog IDs, route references, policy digests, exact origins, query modes, typed source predicates, and credential requirements.
- [ ] Add an additive route-independent V2 document identity; assert existing `build_fingerprint`, `stable_result_id`, serialized V1 result IDs, and Deep Research evidence IDs remain byte-for-byte unchanged.
- [ ] Build a V2 registry for the eight existing targets without mutating `default_source_catalog()`. Record exact routes/backends and make legacy `site_hosts` descriptive only.
- [ ] Mark OpenAlex's V2 API route credentialed/unavailable by default; never inherit the stale V1 `requires_credentials=False` claim.
- [ ] Reconcile all eight existing targets (arXiv, PubMed, Semantic Scholar, Zenodo, OpenAlex, OSF, Figshare, and Crossref) against the frozen ledger's target, route, backend, credential, query-mode, and source-predicate declarations. The Crossref Metadata Search seed row resolves to the existing stable `crossref` product ID; do not create a parallel `crossref_metadata_search` runtime identity.
- [ ] Compile explicit V2 selections into stable ordered attempts with selection reasons, route fallback order, policy/catalog versions, and declared physical reservations.
- [ ] Coalesce only requests whose backend, normalized query, filters, policy, and source predicates are actually compatible. Preserve requested-target attribution separately from backend identity.
- [ ] Count PubMed ESearch plus ESummary as two physical reservations. Release an unused second reservation only when no second dispatch occurred.
- [ ] Model and report separate ceilings for route attempts, physical dispatches, per-route pages, redirects, retries, aggregate wall time, and returned results.
- [ ] Add a synthetic two-target shared-backend aggregator to tests so coalescing plus matching, nonmatching, and ambiguous source predicates are exercised even though the eight production foundation routes are direct/native.
- [ ] Add Hypothesis invariants where available: deterministic plan bytes, no negative budget, used plus released plus outstanding equals reserved, coalescing never increases physical requests, and no dimension exceeds its declared ceiling.

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

## Stage 3: Consume TASK-12971 Through a One-Hop Discovery Gateway

**Goal:** Apply discovery route policy to exactly one separately accounted physical hop using the reusable secure transport primitive.

**Success Criteria:** The discovery gateway validates a frozen route request, delegates one hop to TASK-12971, returns sanitized typed transport evidence, and never follows a redirect or performs a retry itself.

**Tests:** Unit, security, cancellation, revocation, and streaming-boundary tests with an injected fake one-hop primitive; focused integration tests against TASK-12971's local test server only.

**Status:** Not Started (TASK-12971 prerequisite delivered; TASK-12968.2 remains blocked on TASK-12968.7)

### Files

- Add `tldw_Server_API/app/core/Research/discovery/gateway.py`.
- Add `tldw_Server_API/tests/Research/test_research_discovery_gateway.py`.
- Import `HTTPHopLimits`, `NormalizedHTTPHopRequest`, `HTTPHopResponse`, `HTTPHopError`, and `request_http_hop` from `tldw_Server_API.app.core.Security.http_hop`; do not copy its transport code into Research or import its private resolver/backend seams.
- Do not call `tldw_Server_API/app/core/Security/egress.py:evaluate_url_policy`: it resolves DNS and reads ambient egress configuration. Reuse only side-effect-free normalization helpers whose inputs are explicit. TASK-12971 exclusively owns DNS-answer and connected-peer enforcement.
- Do not reuse Web Scraping `FetchRequest` defaults that permit cookies, proxies, or redirects. The discovery request contract is explicit and narrower.

### Test-first steps

- [ ] Stop if TASK-12971 is not complete or its focused security tests are not green; record the blocker instead of implementing an alternate client.
- [ ] Write failing tests for exact scheme/host/port/method/path/query enforcement, immutable policy digest binding, revocation before dispatch, query minimization, and independence from environment/config-file allowlists.
- [ ] Define `dispatch_once(...)` so one call can perform one physical hop only. Redirect responses and retryable failures return typed data to the executor; they are not followed internally.
- [ ] Prove no ambient proxy, `.netrc`, cookie, authorization, client certificate, or credential state enters the primitive request.
- [ ] Preserve TASK-12971's resolved-address, Host/SNI, connected-peer, wire-byte, decompressed-byte, header, time, and parser ceilings in the returned trace.
- [ ] Return bounded typed errors without query text, response bodies, local paths, secrets, or unsafe provider detail.

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

## Stage 4: Add Gateway-Only V2 Adapters and Executor

**Goal:** Execute the eight existing source routes offline through one injected gateway boundary with truthful per-dispatch accounting and inert results.

**Success Criteria:** All V2-enabled adapters use only the gateway, every initial/page/redirect/retry request receives a new reservation and `dispatch_id`, every budget and cancellation transition is reportable, typed partial outcomes are deterministic, and static/runtime boundary tests prove there is no alternate network or retrieval path.

**Tests:** Per-adapter fixture tests, executor state-machine tests, tripwire tests, and AST network-boundary tests.

**Status:** Not Started

### Files

- Add `tldw_Server_API/app/core/Research/discovery/gateway_adapters.py`.
- Add `tldw_Server_API/app/core/Research/discovery/executor.py`.
- Add `tldw_Server_API/tests/Research/test_research_discovery_gateway_adapters.py`.
- Add `tldw_Server_API/tests/Research/test_research_discovery_executor.py`.
- Add `tldw_Server_API/tests/Research/test_research_discovery_network_boundary.py`.
- Add sanitized provider fixtures under the existing Research test-fixture convention discovered during implementation; do not create a second fixture framework.

### Test-first steps

- [ ] Write adapter tests for OpenAlex, Semantic Scholar, Crossref, arXiv, PubMed, Zenodo, Figshare, and OSF using inert fixture URLs and a recording gateway.
- [ ] Implement V2 parsers directly over gateway responses. Do not import the legacy Third_Party search wrappers: arXiv hides fetch/sleep behavior and PubMed hides two physical calls.
- [ ] Cover success, valid empty, bounded pagination, rate-limit metadata, timeout, cancellation, malformed/schema-drift payloads, unsafe echoed errors, and missing or ambiguous attribution.
- [ ] Add a minimal in-memory `AttemptJournal` with explicit `reserved`, `dispatching`, `succeeded`, `valid_empty`, `failed`, `timed_out`, `cancelled`, `skipped`, and `indeterminate_after_dispatch` transitions. Do not add durable storage.
- [ ] Reserve before every initial request, page, redirect, and retry; revalidate the frozen route policy; assign a fresh `dispatch_id` before calling the gateway; never reuse an ID after an indeterminate dispatch.
- [ ] Reconstruct pagination from typed cursor fields into the approved route template. Reject absolute cursor URLs, cross-origin cursors, undeclared query keys, repeated cursors, and work beyond per-route page/redirect/retry ceilings or the aggregate deadline.
- [ ] Permit retries only for explicitly retryable outcomes on idempotent methods. Debit failed dispatched work and surface possible duplicate work; never release or silently replay a dispatch that may have reached the provider.
- [ ] Enforce route-attempt, physical-dispatch, page, redirect, retry, aggregate-wall-time, and returned-result budgets independently. Apply a deterministic result cap before normalization output and report truncation.
- [ ] Test cancellation before dispatch (unused reservation released), while dispatching with no definitive result (`indeterminate_after_dispatch`, debit retained), and after a definitive gateway result (debit retained, late candidate content suppressed if cancellation won the commit boundary).
- [ ] Validate aggregator source predicates before attribution. Ambiguous or nonmatching records remain unattributed to the requested target.
- [ ] Run the synthetic shared-backend aggregator through planner, executor, and normalization to prove matching, nonmatching, ambiguous attribution, and one physical coalesced request end to end.
- [ ] Prove result URLs receive zero requests and that provider failures make zero calls to generic fetch, Media, scraping, Playwright, cookies, or secrets.
- [ ] Add an AST test that fails V2 adapter imports/construction of `httpx`, `aiohttp`, `requests`, `urllib`, sockets, SDK default transports, and legacy Third_Party networking. Only `gateway.py` may consume TASK-12971's transport API.

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
python -m compileall -q \
  tldw_Server_API/app/core/Research/discovery/gateway_adapters.py \
  tldw_Server_API/app/core/Research/discovery/executor.py \
  tldw_Server_API/tests/Research/test_research_discovery_gateway_adapters.py \
  tldw_Server_API/tests/Research/test_research_discovery_executor.py \
  tldw_Server_API/tests/Research/test_research_discovery_network_boundary.py
ruff check \
  tldw_Server_API/app/core/Research/discovery/gateway_adapters.py \
  tldw_Server_API/app/core/Research/discovery/executor.py \
  tldw_Server_API/tests/Research/test_research_discovery_gateway_adapters.py \
  tldw_Server_API/tests/Research/test_research_discovery_executor.py \
  tldw_Server_API/tests/Research/test_research_discovery_network_boundary.py
black --check \
  tldw_Server_API/app/core/Research/discovery/gateway_adapters.py \
  tldw_Server_API/app/core/Research/discovery/executor.py \
  tldw_Server_API/tests/Research/test_research_discovery_gateway_adapters.py \
  tldw_Server_API/tests/Research/test_research_discovery_executor.py \
  tldw_Server_API/tests/Research/test_research_discovery_network_boundary.py
git diff --check
```

### Commit

`feat(research): execute discovery v2 through the gateway`

## Stage 5: Prove Offline Compatibility and Leave Production Disabled

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
- [ ] Add an import-boundary test proving V1 production catalog/router/service/endpoint modules do not import or construct the V2 executor, plus a construction test proving V2 requires an explicit offline/synthetic opt-in. Do not claim a no-double-fetch test merely because V2 has no production wiring.
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
