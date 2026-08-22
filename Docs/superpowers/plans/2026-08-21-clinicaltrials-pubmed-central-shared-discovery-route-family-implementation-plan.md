# ClinicalTrials.gov and PubMed Central Shared-Discovery Route Family Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add fixture-certified, shadow-only ClinicalTrials.gov and PubMed Central general-search routes, plus an identity-bearing PubMed overlay, without changing current Standalone Search, Deep Research, or foundation behavior.

**Architecture:** Extend the existing immutable discovery contracts only where ClinicalTrials.gov requires a provider-issued opaque query cursor, then add one sibling provider-family module composed over the foundation registry. ClinicalTrials.gov uses a strict two-page API-v2 adapter; PMC and the PubMed identity overlay reuse one private ESearch-to-ESummary execution seam while retaining separate parsers and exact route identities. All physical work remains gateway-bound, fixture-only, bounded, accounted, cancellation-aware, and absent from production consumers.

**Tech Stack:** Python 3.10+, frozen/slotted dataclasses, asyncio, stdlib `html.parser.HTMLParser`, existing research-discovery planner/executor/gateway/one-hop transport, pytest, Node.js inventory validator/tests, Ruff, Black, Bandit.

**Spec:** `Docs/Design/2026-07-15-clinicaltrials-pubmed-central-shared-discovery-route-family-design.md`

## Global Constraints

- Work only in the clean TASK-12968.6 worktree and branch; do not edit the dirty repository root.
- Use test-driven development for every production behavior: witness the focused RED before the minimum GREEN change.
- Add no dependency, SDK, browser, scraper, cookie/credential access, persistent cache, background job, provider-specific HTTP client, or result-link dereference.
- Make no live NCBI request. Ordinary verification is fixture-only; any optional ClinicalTrials.gov feasibility check remains opt-in and is outside this plan's required gates.
- Keep `foundation_registry()`, `foundation_readiness()`, `foundation_gateway_adapters()`, the `foundation-v2` PubMed behavior, and all existing foundation canonical plan bytes exact.
- Keep `PLANNER_VERSION = "research-discovery-planner-v2-foundation"` unchanged.
- Do not add or reorder `RoutePolicy` fields. Reuse `pagination_query_key` and discriminate opaque versus numeric pagination through the closed query-policy type.
- Freeze `tool=tldw_server` and `email=contact@tldwproject.com` as plan-visible, non-user-supplied query pairs on every new identity-bearing NCBI intent.
- Freeze ClinicalTrials.gov to `GET https://clinicaltrials.gov:443/api/v2/studies`; two pages; 50 records per page; 100 raw records; two physical dispatches; 20 seconds, 2 MiB, zero redirects, and zero retries per hop.
- Freeze PMC to ESearch then conditional ESummary at `https://eutils.ncbi.nlm.nih.gov:443`; one logical page; at most 100 IDs; two physical dispatches; 20 seconds, 2 MiB, zero redirects, and zero retries per hop.
- Freeze `RouteLimits.max_request_body_bytes=16_384` for both GET/no-body routes; each contributes exactly 40,000 ms to planned aggregate wall-time, and a plan selecting both contributes 80,000 ms before any other route.
- Leave process-wide NCBI pacing, registration proof, required notices/currentness presentation, long-query routing, API-key use, and consumer cutover to TASK-12968.3.
- Leave authenticated retrieval, cookies, browser state, and login automation outside TASK-12968.6 and blocked on TASK-13013 plus a separate approved design.
- Leave `discovery/__init__.py`, production service/endpoint/router modules, default catalog factories, and consumer adapter maps unchanged.
- Preserve dependency direction: shared `planner.py`, `executor.py`, `gateway.py`, and `gateway_adapters.py` may not import the provider-family module. Duplicate the two exact overlay version strings as private shared-module constants/literals and reconcile them in tests; imports flow family → shared only.
- Treat normalized result URLs as inert synthesized metadata; never request or resolve them.
- Preserve unrelated inventory/report repairs already present in the worktree; stage TASK-12968.6 hunks deliberately and never restore those files wholesale.
- Before each Python command, activate `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate`.
- The approved design, duplicate-ID tracking repair, prerequisite-task records, inventory freeze repair, Backlog notes, and this plan must be committed together as the pre-implementation planning checkpoint before Task 1; the implementing worker starts from that clean checkpoint.

## File Map

### Create

- `tldw_Server_API/app/core/Research/discovery/clinicaltrials_pubmed_central.py` — family-owned registry overlay, readiness overlay, exact route/adapters, strict parsers, and immutable two-entry family adapter map.
- `tldw_Server_API/tests/Research/test_research_discovery_clinicaltrials_pubmed_central.py` — constructor, planner, adapter, accounting, cancellation, partial-failure, and fixture evidence for the new family.
- `tldw_Server_API/tests/fixtures/research_discovery_gateway_adapters/clinicaltrials_success_page_1.json` — wholly synthetic valid page with continuation.
- `tldw_Server_API/tests/fixtures/research_discovery_gateway_adapters/clinicaltrials_success_page_2.json` — wholly synthetic valid terminal page.
- `tldw_Server_API/tests/fixtures/research_discovery_gateway_adapters/clinicaltrials_empty.json` — exact valid empty envelope.
- `tldw_Server_API/tests/fixtures/research_discovery_gateway_adapters/pmc_esearch_success.json` — synthetic nonempty canonical decimal-string ESearch envelope.
- `tldw_Server_API/tests/fixtures/research_discovery_gateway_adapters/pmc_esearch_empty.json` — synthetic empty canonical ESearch envelope.
- `tldw_Server_API/tests/fixtures/research_discovery_gateway_adapters/pmc_esummary_success.json` — synthetic UID-keyed ESummary records with canonical PMCID correlation.

### Modify

- `tldw_Server_API/app/core/Research/discovery/contracts.py` — closed opaque query-value policy, empty literal suffix support, query-value diagnostic redaction.
- `tldw_Server_API/app/core/Security/http_hop.py` — normalized request-target diagnostic redaction only.
- `tldw_Server_API/app/core/Research/discovery/planner.py` — named literal general queries, bounded decimal clamping, optional first-page opaque omission, exact PMC and PubMed-overlay intent shapes.
- `tldw_Server_API/app/core/Research/discovery/gateway.py` — opaque policy snapshot/value validation, empty-suffix fix, binding diagnostic redaction.
- `tldw_Server_API/app/core/Research/discovery/executor.py` — closed opaque cursor continuation, repeat detection, and unchanged reservation/accounting semantics.
- `tldw_Server_API/app/core/Research/discovery/gateway_adapters.py` — exact PubMed overlay parsing profile and the smallest private NCBI two-hop seam.
- `tldw_Server_API/tests/Research/test_research_discovery_contracts.py` — closed contracts, digest compatibility, and repr-only redaction.
- `tldw_Server_API/tests/Security/test_http_hop_contract.py` — repr-only request-target redaction and unchanged equality/wire semantics.
- `tldw_Server_API/tests/Research/test_research_discovery_planner.py` — generic opaque first-page planning, exact identity pairs, PMC two-intent planning, and foundation goldens.
- `tldw_Server_API/tests/Research/test_research_discovery_gateway.py` — strict opaque validation, encoding, request-target bound, empty suffix, and binding redaction.
- `tldw_Server_API/tests/Research/test_research_discovery_gateway_adapters.py` — post-rebase exact parsing-profile registry lock for the PubMed identity overlay and foundation profile alias.
- `tldw_Server_API/tests/Research/test_research_discovery_executor.py` — opaque continuation, exact-repeat rejection, cancellation, reservations, diagnostics, and numeric compatibility.
- `tldw_Server_API/tests/Research/test_research_discovery_pubmed_gateway_adapter.py` — unchanged foundation PubMed plus runtime-certified identity overlay.
- `tldw_Server_API/tests/Research/test_research_discovery_network_boundary.py` — exact multi-family closure configuration and runtime tripwires.
- `tldw_Server_API/tests/Research/test_research_discovery_registry_reconciliation.py` — exact new source/route/backend/adapter/readiness/inventory identities.
- `tldw_Server_API/tests/Research/test_research_discovery_v2_compatibility.py` — foundation byte locks and consumer import/factory tripwires.
- `Helper_Scripts/validate_research_source_inventory.mjs` — authoritative exact implemented-source gate for rows 0026 and 0027.
- `Helper_Scripts/tests/validate_research_source_inventory.test.mjs` — exact positive and drift cases for the new gate and summary counts.
- `Docs/Design/research_source_inventory/research-source-coverage-ledger-2026-07-13.json` — migrate rows 0026/0027 to the implemented shadow routes, preserve their source-snapshot hashes, and recompute aggregate `rows_sha256`.
- `Docs/Design/research_source_inventory/research-source-inventory-freeze-report-2026-07-13.json` — regenerate only through the authoritative validator.
- `backlog/tasks/task-12968.6 - Add-ClinicalTrials.gov-and-PubMed-Central-shared-discovery-route-family.md` — plan link, RED/GREEN evidence, verification, commit, and handoff notes through the Backlog CLI.

---

## Stage 1: Shared Opaque-Query Boundary

**Goal:** Add the smallest closed contract, planner, gateway, and executor changes required for a provider-issued query token.

**Success Criteria:** ClinicalTrials.gov-shaped first-page intents omit `pageToken`; continuations insert one exact visible-ASCII token at policy order; repeated tokens fail; secrets are absent from diagnostic repr paths; legacy fields, digests, plans, and numeric pagination remain byte/behavior exact.

**Tests:** Focused contract, planner, gateway, executor, HTTP-hop, network-boundary digest, and v2-compatibility tests after each commit.

**Status:** Complete

### Task 1: Closed opaque policy and repr-safe request contracts

**Files:**

- Modify: `tldw_Server_API/app/core/Research/discovery/contracts.py:141, 378-420`
- Modify: `tldw_Server_API/app/core/Security/http_hop.py` (`NormalizedHTTPHopRequest`)
- Test: `tldw_Server_API/tests/Research/test_research_discovery_contracts.py`
- Test: `tldw_Server_API/tests/Security/test_http_hop_contract.py`
- Test: `tldw_Server_API/tests/Research/test_research_discovery_network_boundary.py`

**Interfaces:**

- Consumes: Existing `_validate_query_value_policy_common(name, required)`, `QueryValuePolicy`, `_QUERY_VALUE_POLICY_TYPES`, `QueryPair`, `NormalizedHTTPHopRequest`, and `canonical_policy_digest()`.
- Produces: `MAX_OPAQUE_CURSOR_CHARS: Final[int] = 1_024`; `OpaqueCursorQueryValuePolicy(name: str, max_chars: int, required: bool = False)`; empty-string `LiteralTermsQueryValuePolicy.fixed_suffix`; repr-hidden `QueryPair.value` and `NormalizedHTTPHopRequest.target`.

- [x] **Step 1: Write contract RED tests**

Add explicit constructor cases and repr/equality/digest locks:

```python
@pytest.mark.parametrize("max_chars", [0, 1_025, True, "1024"])
def test_opaque_query_policy_rejects_noncanonical_bounds(max_chars: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        OpaqueCursorQueryValuePolicy("pageToken", max_chars)  # type: ignore[arg-type]


def test_query_pair_repr_hides_value_without_changing_semantics() -> None:
    pair = QueryPair("pageToken", "opaque-token-sentinel")
    assert "opaque-token-sentinel" not in repr(pair)
    assert pair == QueryPair("pageToken", "opaque-token-sentinel")
    assert hash(pair) == hash(QueryPair("pageToken", "opaque-token-sentinel"))
    assert asdict(pair) == {"name": "pageToken", "value": "opaque-token-sentinel"}


def test_empty_literal_suffix_is_contract_valid_and_digest_bound() -> None:
    policy = LiteralTermsQueryValuePolicy("query.term", "", 8, 32)
    assert policy.fixed_suffix == ""
    nonempty = LiteralTermsQueryValuePolicy("query.term", " AND FIXED", 8, 32)
    empty_route = _template_policy(
        query_value_policies=(policy,),
        allowed_query_keys=("query.term",),
    )
    nonempty_route = _template_policy(
        query_value_policies=(nonempty,),
        allowed_query_keys=("query.term",),
    )
    assert canonical_policy_digest(empty_route) != canonical_policy_digest(nonempty_route)
```

Move `LiteralTermsQueryValuePolicy("query", "", 16, 64)` out of the existing invalid-constructor tuple at `test_research_discovery_contracts.py:212-220` and into the valid frozen-policy cases; otherwise the old test contradicts the new contract. Add `OpaqueCursorQueryValuePolicy` to the exact public-contract and closed-policy assertions. In `test_http_hop_contract.py`, construct two equal requests and assert the target remains readable and wire-valid but is absent from `repr(request)`; do not assert against `repr(asdict(request))` because `asdict()` intentionally retains contract material.

- [x] **Step 2: Run the focused tests and witness RED**

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest -q \
  tldw_Server_API/tests/Research/test_research_discovery_contracts.py \
  tldw_Server_API/tests/Security/test_http_hop_contract.py
```

Expected: failures because `OpaqueCursorQueryValuePolicy` does not exist, empty suffix is rejected, and values/targets appear in repr.

- [x] **Step 3: Implement the minimum closed contract**

Use the existing constructor ordering and add no `RoutePolicy` field:

```python
MAX_OPAQUE_CURSOR_CHARS: Final[int] = 1_024


@dataclass(frozen=True, slots=True)
class OpaqueCursorQueryValuePolicy:
    name: str
    max_chars: int
    required: bool = False

    def __post_init__(self) -> None:
        _validate_query_value_policy_common(self.name, self.required)
        if type(self.max_chars) is not int or not 1 <= self.max_chars <= MAX_OPAQUE_CURSOR_CHARS:
            raise ValueError("invalid_opaque_cursor_max_chars")
```

Add the class to the closed union and type tuple. Change only these fields diagnostically:

```python
value: str = field(repr=False)  # QueryPair
target: str = field(repr=False)  # NormalizedHTTPHopRequest
```

Change literal-suffix validation to require an exact string without NUL, but not nonempty:

```python
if type(self.fixed_suffix) is not str or "\x00" in self.fixed_suffix:
    raise ValueError("invalid_literal_terms_fixed_suffix")
```

- [x] **Step 4: Refresh the changed shared-module semantic/import digests and run GREEN compatibility**

Update only the `contracts.py` and `Security/http_hop.py` import/AST entries required by `test_research_discovery_network_boundary.py`; do not add raw shared-module digests.

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest -q \
  tldw_Server_API/tests/Research/test_research_discovery_contracts.py \
  tldw_Server_API/tests/Security/test_http_hop_contract.py \
  tldw_Server_API/tests/Research/test_research_discovery_network_boundary.py \
  tldw_Server_API/tests/Research/test_research_discovery_v2_compatibility.py
```

Expected: PASS, including legacy positional `RoutePolicy` construction and exact legacy digest `8ec7b6572f32690e1425390518077742607bee40f87224c45913d7c5f54e7865`.

- [x] **Step 5: Commit the isolated contract change**

```bash
git add \
  tldw_Server_API/app/core/Research/discovery/contracts.py \
  tldw_Server_API/app/core/Security/http_hop.py \
  tldw_Server_API/tests/Research/test_research_discovery_contracts.py \
  tldw_Server_API/tests/Security/test_http_hop_contract.py \
  tldw_Server_API/tests/Research/test_research_discovery_network_boundary.py
git commit -m "feat(research): add opaque query policy and repr-safe request contracts (TASK-12968.6)"
```

### Task 2: Plan optional opaque first-page query pagination

**Files:**

- Modify: `tldw_Server_API/app/core/Research/discovery/planner.py:413-462`
- Test: `tldw_Server_API/tests/Research/test_research_discovery_planner.py`
- Test: `tldw_Server_API/tests/Research/test_research_discovery_v2_compatibility.py`
- Test: `tldw_Server_API/tests/Research/test_research_discovery_network_boundary.py`

**Interfaces:**

- Consumes: `OpaqueCursorQueryValuePolicy`, existing general-query normalization, `RoutePolicy.allowed_query_keys`, `RoutePolicy.pagination_query_key`, and `DispatchIntent`.
- Produces: `_build_typed_intents()` support for a named literal policy, empty suffix, clamped decimal value, and omission of exactly one optional opaque pagination key on page one.

- [x] **Step 1: Add a separate opaque-route planner fixture and RED tests**

Keep the existing `_typed_query_registry()` unchanged. Add `_opaque_query_registry()` with allowed keys in this order so the omitted cursor is nonterminal:

```python
(
    "query.term",
    "pageToken",
    "format",
    "pageSize",
)
```

Use policies `LiteralTermsQueryValuePolicy("query.term", "", 8, 32)`, `OpaqueCursorQueryValuePolicy("pageToken", 1024, False)`, `ExactQueryValuePolicy("format", "json")`, and `BoundedDecimalQueryValuePolicy("pageSize", 50)`. Assert the exact first-page tuple:

Define this exact test-only constructor in the planner test, and reproduce it unchanged in the gateway/executor test files when Tasks 3/4 consume it; do not import one test module from another:

```python
def _opaque_query_registry() -> tuple[DiscoveryRegistry, ReadinessOverlay]:
    limits = RouteLimits(2, 0, 0, 250, 65_536, 100, 16_384)
    policy = RoutePolicy(
        policy_version="opaque-query-policy-v1",
        origin=ExactOrigin("https", "clinical.example.test", 443),
        methods=("GET",),
        paths=("/api/v2/studies",),
        allowed_query_keys=("query.term", "pageToken", "format", "pageSize"),
        limits=limits,
        pagination_query_key="pageToken",
        query_value_policies=(
            LiteralTermsQueryValuePolicy("query.term", "", 8, 32),
            OpaqueCursorQueryValuePolicy("pageToken", 1_024, required=False),
            ExactQueryValuePolicy("format", "json"),
            BoundedDecimalQueryValuePolicy("pageSize", 50),
        ),
    )
    route = AccessRoute(
        route_id="opaque_query_search",
        backend_id="opaque_query_backend",
        adapter_id="opaque_query_adapter",
        route_kind=RouteKind.DIRECT,
        query_modes=(QueryMode.GENERAL_FREE_TEXT,),
        source_constraint=SourceConstraint.NATIVE_CORPUS,
        attribution_basis="native_response",
        credential_requirement=CredentialRequirement.NONE,
        fallback_order=0,
        max_physical_dispatches=2,
        adapter_version="opaque-v1",
        policy=policy,
    )
    registry = DiscoveryRegistry(
        catalog_version="opaque-query-catalog-v1",
        registry_version="opaque-query-registry-v1",
        sources=(
            SourceDefinition(
                catalog_source_id="opaque_query_source",
                display_name="Opaque Query Source",
                aliases=(),
                categories=("synthetic",),
                content_types=("records",),
                surfaces=("standalone_search",),
                route_references=(SourceRouteReference(route.route_id, None),),
                site_hosts=("clinical.example.test",),
                priority=10,
                catalog_version="opaque-query-catalog-v1",
            ),
        ),
        routes=(route,),
        backends=(BackendDefinition("opaque_query_backend", "Opaque Query Backend"),),
    )
    readiness = ReadinessOverlay(
        overlay_version="opaque-query-readiness-v1",
        execution_mode=ExecutionMode.SYNTHETIC,
        routes=(
            RouteReadiness(
                route.route_id,
                ReadinessState.READY,
                CredentialStatus.NOT_REQUIRED,
                "synthetic_ready",
            ),
        ),
    )
    return registry, readiness
```

Compile it everywhere with `GeneralFreeTextQuery("alpha beta")`, no filters, result limit `100`, and `BudgetCeilings(1, 2, 2, 0, 0, 500, 100)`.

```python
assert plan.dispatch_groups[0].intents[0].query_pairs == (
    QueryPair("query.term", '"alpha" AND "beta"'),
    QueryPair("format", "json"),
    QueryPair("pageSize", "50"),
)
```

Add `test_general_query_clamps_decimal_to_route_and_rule_ceilings` and parameterize invalid shapes: required opaque policy, opaque policy not named by `pagination_query_key`, and two attempted optional omissions.

- [x] **Step 2: Run planner RED tests**

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest -q \
  tldw_Server_API/tests/Research/test_research_discovery_planner.py \
  tldw_Server_API/tests/Research/test_research_discovery_v2_compatibility.py
```

Expected: the named literal/empty suffix and omitted opaque pair are rejected by current planning.

- [x] **Step 3: Generalize only the typed-general loop**

Preserve allowed-key order and use a single omission counter:

```python
omitted_opaque = 0
for name in route.policy.allowed_query_keys:
    policy = policies.get(name)
    if type(policy) is OpaqueCursorQueryValuePolicy:
        if policy.required or name != route.policy.pagination_query_key or omitted_opaque:
            raise _planning_error(f"invalid_optional_opaque_cursor_policy:{route.route_id}")
        omitted_opaque += 1
        continue
    if type(policy) is LiteralTermsQueryValuePolicy:
        if (
            literal_terms_seen
            or not 1 <= len(query.terms) <= policy.max_terms
            or any(len(term) > policy.max_term_chars for term in query.terms)
        ):
            raise _planning_error(f"invalid_literal_terms_policy:{route.route_id}")
        literal_terms_seen = True
        pairs.append(QueryPair(name, " AND ".join(f'\"{term}\"' for term in query.terms) + policy.fixed_suffix))
    elif type(policy) is BoundedDecimalQueryValuePolicy:
        pairs.append(QueryPair(name, str(min(result_limit, route.policy.limits.max_results, policy.maximum))))
```

Require exactly one literal policy and full coverage after accounting for the one permitted opaque omission. Do not change raw-string `_build_intents()` yet and do not change `PLANNER_VERSION`.

Add a RED assertion that one 33-character term fails the Clinical-shaped policy before an intent is emitted; retain the existing per-policy term-length guard for every typed general route.

- [x] **Step 4: Lock foundation bytes and run GREEN**

Assert the existing foundation request retains:

- plan digest `2e9869bc7ed6b51fe8ffe823dff2e392933e275b54bc2997e8542ba89829f403`;
- SHA-256 of canonical plan bytes `991d3a67132058625bdfef00836240cacc6b91370510c9b9d0762181310c9d46`;
- canonical byte length `10936`.

Refresh only the planner import/AST boundary digest and run the two focused files plus network-boundary tests.

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest -q \
  tldw_Server_API/tests/Research/test_research_discovery_planner.py \
  tldw_Server_API/tests/Research/test_research_discovery_v2_compatibility.py \
  tldw_Server_API/tests/Research/test_research_discovery_network_boundary.py
```

Expected: PASS with the exact foundation plan digest, canonical-byte digest, and byte length above unchanged.

- [x] **Step 5: Commit the planner change**

```bash
git add \
  tldw_Server_API/app/core/Research/discovery/planner.py \
  tldw_Server_API/tests/Research/test_research_discovery_planner.py \
  tldw_Server_API/tests/Research/test_research_discovery_v2_compatibility.py \
  tldw_Server_API/tests/Research/test_research_discovery_network_boundary.py
git commit -m "feat(research): plan optional opaque query pagination (TASK-12968.6)"
```

### Task 3: Validate and encode opaque query bindings at the gateway

**Files:**

- Modify: `tldw_Server_API/app/core/Research/discovery/gateway.py` (`_BindingSnapshot`, policy snapshot/value validation)
- Test: `tldw_Server_API/tests/Research/test_research_discovery_gateway.py`
- Test: `tldw_Server_API/tests/Research/test_research_discovery_network_boundary.py`

**Interfaces:**

- Consumes: `OpaqueCursorQueryValuePolicy`, a planned intent that may omit the optional token, and the existing one-hop request builder.
- Produces: strict exact visible-ASCII validation for present opaque pairs, correct whole-value validation for empty literal suffixes, and repr-hidden `_BindingSnapshot.query_pairs`.

- [x] **Step 1: Add table-driven gateway RED tests**

Create `_opaque_query_route_and_intent()` rather than modifying existing digest-bound fixtures. Test:

```python
def _opaque_query_route_and_intent(
    token: str | None = None,
) -> tuple[AccessRoute, DispatchIntent]:
    registry, readiness = _opaque_query_registry()
    plan = compile_discovery_plan(
        PlanningRequest(
            ("opaque_query_source",),
            GeneralFreeTextQuery("alpha beta"),
            (),
            100,
        ),
        registry=registry,
        readiness=readiness,
        budget=BudgetCeilings(1, 2, 2, 0, 0, 500, 100),
    )
    route = registry.get_route("opaque_query_search")
    intent = plan.dispatch_groups[0].intents[0]
    if token is not None:
        intent = replace(
            intent,
            query_pairs=(
                QueryPair("query.term", '"alpha" AND "beta"'),
                QueryPair("pageToken", token),
                QueryPair("format", "json"),
                QueryPair("pageSize", "50"),
            ),
        )
    return route, intent


async def _dispatch_with_token(
    token: str,
    one_hop: Callable[[NormalizedHTTPHopRequest], Awaitable[HTTPHopResponse]],
) -> DiscoveryGatewayResponse:
    route, intent = _opaque_query_route_and_intent(token)
    return await dispatch_once(
        route,
        intent,
        is_policy_active=lambda _route_id, _policy_digest: True,
        one_hop=one_hop,
    )
```

Import `Awaitable` and `Callable` from `typing`, and use the exact `_opaque_query_registry()` body from Task 2 in this test file.

```python
@pytest.mark.parametrize("token", ["", " ", "a b", "line\nbreak", "\x7f", "é", "x" * 1_025])
async def test_opaque_query_value_fails_before_one_hop(token: str) -> None:
    calls = 0
    async def one_hop(_: NormalizedHTTPHopRequest) -> HTTPHopResponse:
        nonlocal calls
        calls += 1
        raise AssertionError("must not dispatch")
    with pytest.raises(DiscoveryGatewayError):
        await _dispatch_with_token(token, one_hop)
    assert calls == 0
```

Also assert: missing optional token succeeds on page one; `!$&'()*+,;=:@/?` is preserved as one percent-encoded value; duplicate/unknown keys and a reconstructed mutated policy fail pre-hop; empty-suffix literals accept only the canonical quoted-term expression; `_BindingSnapshot` repr omits a sentinel; and a worst-case 8×32 four-byte-term query plus 1,024 reserved punctuation characters remains under `HTTPHopLimits.max_request_target_bytes == 8192`.

- [x] **Step 2: Run gateway RED tests**

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest -q tldw_Server_API/tests/Research/test_research_discovery_gateway.py
```

Expected: opaque policy reconstruction/validation is unsupported and empty suffix uses the incorrect `[:-0]` slice.

- [x] **Step 3: Extend the closed gateway snapshot and validation**

Add the exact policy type to `_snapshot_query_value_policies()` and validate without normalization:

```python
if type(policy) is OpaqueCursorQueryValuePolicy:
    return (
        type(value) is str
        and 1 <= len(value) <= policy.max_chars
        and all("!" <= character <= "~" for character in value)
    )
```

Fix the empty suffix explicitly:

```python
literal_expression = value if policy.fixed_suffix == "" else value[:-len(policy.fixed_suffix)]
```

Change only the binding diagnostic field:

```python
query_pairs: tuple[tuple[str, str], ...] = field(repr=False)
```

Do not reorder, sort, normalize, or otherwise rewrite gateway query pairs. `_build_target()` must preserve the planner/executor tuple and encode each value once.

- [x] **Step 4: Refresh gateway boundary digests and run GREEN**

Run gateway, contract, HTTP-hop, and network-boundary tests. Confirm equality, field access, canonical material, and the wire request are unchanged except for valid new opaque inputs.

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest -q \
  tldw_Server_API/tests/Research/test_research_discovery_gateway.py \
  tldw_Server_API/tests/Research/test_research_discovery_contracts.py \
  tldw_Server_API/tests/Security/test_http_hop_contract.py \
  tldw_Server_API/tests/Research/test_research_discovery_network_boundary.py
```

Expected: PASS with legacy request construction and diagnostic equality unchanged.

- [x] **Step 5: Commit the gateway change**

```bash
git add \
  tldw_Server_API/app/core/Research/discovery/gateway.py \
  tldw_Server_API/tests/Research/test_research_discovery_gateway.py \
  tldw_Server_API/tests/Research/test_research_discovery_network_boundary.py
git commit -m "feat(research): validate opaque query bindings at gateway (TASK-12968.6)"
```

### Task 4: Account opaque cursor continuations in the executor

**Files:**

- Modify: `tldw_Server_API/app/core/Research/discovery/executor.py:131, 790, 1163-1400`
- Test: `tldw_Server_API/tests/Research/test_research_discovery_executor.py`
- Test: `tldw_Server_API/tests/Research/test_research_discovery_network_boundary.py`

**Interfaces:**

- Consumes: An optional opaque policy named by `pagination_query_key`, first-page intent without the token, and the existing dispatch reservation/cancellation machinery.
- Produces: `OpaqueCursor(value: str)`; `NumericCursor | OpaqueCursor | None` accepted by `BoundDispatch` and internal dispatch/controller call sites; ordered continuation reconstruction; exact-repeat rejection; page cursor `str` accounting; repr-hidden `NumericCSVBindingValues.values` with equality/asdict/wire behavior unchanged.

- [x] **Step 1: Build an opaque paginated plan and add executor RED tests**

Add `_opaque_query_paginated_plan()` through the real planner and assert these named behaviors:

```python
def _opaque_query_paginated_plan():
    registry, readiness = _opaque_query_registry()
    plan = compile_discovery_plan(
        PlanningRequest(
            ("opaque_query_source",),
            GeneralFreeTextQuery("alpha beta"),
            (),
            100,
        ),
        registry=registry,
        readiness=readiness,
        budget=BudgetCeilings(1, 2, 2, 0, 0, 500, 100),
    )
    return registry, plan
```

Use the exact `_opaque_query_registry()` body from Task 2 locally in the executor test; do not import a test helper across modules.

- `test_opaque_query_cursor_is_inserted_in_policy_order_and_fully_accounted`;
- `test_query_cursor_type_must_match_declared_policy_before_reservation`;
- `test_opaque_query_cursor_rejects_repeat_without_imposing_order`;
- mutated empty/control/Unicode/oversized values fail before reservation;
- `test_opaque_query_cursor_cannot_mutate_bound_intent_material`;
- cancellation before continuation leaves `created=1`, `debited=1`, `released=0`, `outstanding=0`;
- token absent from cursor/effective-intent/binding/request/callback/error/outcome/usage/journal repr paths;
- deferred numeric CSV values absent from `repr(NumericCSVBindingValues(...))` while equality, `asdict()`, grounding, and existing foundation PubMed wire behavior remain exact;
- executor → gateway → `dispatch_once()` → one-hop yields two pages, two debits, and one exactly encoded continuation value.

The adapter stub must pass the cursor into the second call of the same planned intent:

```python
async def opaque_adapter(
    group: PlannedDispatchGroup,
    dispatch: BoundDispatch,
) -> DiscoveryAdapterResult:
    await dispatch(group.intents[0])
    await dispatch(group.intents[0], cursor=OpaqueCursor("second-token"))
    return DiscoveryAdapterResult(candidates=())
```

Assert exact query-pair order against `allowed_query_keys` after insertion; an opaque cursor is a `BoundDispatch` argument, never an adapter return value.

- [x] **Step 2: Run executor RED tests**

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest -q tldw_Server_API/tests/Research/test_research_discovery_executor.py
```

Expected: cursor protocol annotations and `_effective_intent()` accept only `NumericCursor`.

- [x] **Step 3: Implement the closed cursor and continuation branch**

Add beside `NumericCursor`:

```python
@dataclass(frozen=True, slots=True)
class OpaqueCursor:
    value: str = field(repr=False)

    def __post_init__(self) -> None:
        if type(self.value) is not str or not 1 <= len(self.value) <= MAX_OPAQUE_CURSOR_CHARS:
            raise ValueError("invalid_opaque_cursor")
        if any(not "!" <= character <= "~" for character in self.value):
            raise ValueError("invalid_opaque_cursor")
```

Make the existing deferred binding diagnostic-only change at the same boundary:

```python
@dataclass(frozen=True, slots=True)
class NumericCSVBindingValues:
    binding_id: str
    values: tuple[int, ...] = field(repr=False)
```

Do not change its constructor order, validation, equality, `asdict()` material, deferred-grounding bytes, or executor snapshot behavior.

Use `NumericCursor | OpaqueCursor | None` explicitly in `BoundDispatch`, `_adapter_dispatch.dispatch`, `_GroupExecutionController.__call__`, and `_effective_intent`; do not create a public provider cursor framework.

In `_effective_intent()`:

1. Find the exact query policy whose name equals `pagination_query_key`.
2. Treat the route as opaque only when its exact type is `OpaqueCursorQueryValuePolicy`.
3. Require no first-page token pair and no first-page cursor.
4. Revalidate continuation value/type/length/visible ASCII against the frozen policy.
5. Rebuild query pairs in `allowed_query_keys` order with exactly one token pair.
6. Reject exact repeated opaque values but never impose numeric ordering.
7. Preserve numeric query/body/path branches exactly.

Track `dict[int, set[int | str]]`, or keep separate integer and string sets if that produces a smaller diff; in either case, record a token only after `mark_dispatching`/physical debit at the same point numeric cursors are recorded.

- [x] **Step 4: Run GREEN plus numeric-pagination regression locks**

Run the executor file and confirm these existing tests remain green:

```text
test_initial_search_and_one_typed_page_are_independently_accounted
test_figshare_numeric_json_body_cursor_replaces_page_for_second_dispatch
test_nonrepeating_retrograde_cursor_dispatches_for_nonpath_channels
test_path_cursor_replaces_only_declared_segment_and_is_fully_accounted
test_path_cursor_requires_unique_strictly_increasing_progress
test_execution_control_post_reserve_stop_releases_unused_reservation
```

Refresh only the executor import/AST boundary digest and run network-boundary tests.

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest -q \
  tldw_Server_API/tests/Research/test_research_discovery_executor.py \
  tldw_Server_API/tests/Research/test_research_discovery_network_boundary.py
```

Expected: PASS, including every named numeric-pagination regression lock above.

- [x] **Step 5: Commit the executor change**

```bash
git add \
  tldw_Server_API/app/core/Research/discovery/executor.py \
  tldw_Server_API/tests/Research/test_research_discovery_executor.py \
  tldw_Server_API/tests/Research/test_research_discovery_network_boundary.py
git commit -m "feat(research): account opaque cursor continuations (TASK-12968.6)"
```

---

## Stage 2: Exact NCBI Identity and Reuse Seam

**Goal:** Make compliant NCBI request identity part of the immutable plan and reuse the current ESearch/ESummary sequence without changing the existing PubMed route.

**Success Criteria:** The foundation PubMed route is byte/behavior exact; the overlay is accepted only by its full identity tuple; both overlay calls include exact `tool`/`email`; its runtime path is executed through planner, executor, shared callable, gateway, and one-hop; strict documented rate envelopes are distinguished from malformed lookalikes.

**Tests:** Planner overlay goldens, PubMed adapter foundation regressions, overlay runtime success/empty/rate/malformed/cancellation, v2 compatibility, and boundary digests.

**Status:** Complete

### Task 5: Add the identity-bearing PubMed overlay and private NCBI two-hop seam

**Files:**

- Create initially: `tldw_Server_API/app/core/Research/discovery/clinicaltrials_pubmed_central.py`
- Modify: `tldw_Server_API/app/core/Research/discovery/planner.py:543-578`
- Modify: `tldw_Server_API/app/core/Research/discovery/gateway_adapters.py:57, 1337-1555, 1689`
- Test: `tldw_Server_API/tests/Research/test_research_discovery_planner.py`
- Test: `tldw_Server_API/tests/Research/test_research_discovery_pubmed_gateway_adapter.py`
- Test: `tldw_Server_API/tests/Research/test_research_discovery_clinicaltrials_pubmed_central.py`
- Test: `tldw_Server_API/tests/Research/test_research_discovery_v2_compatibility.py`
- Test: `tldw_Server_API/tests/Research/test_research_discovery_network_boundary.py`

**Interfaces:**

- Consumes: `foundation_registry()`, current exact PubMed route and adapter callable, `DeferredNumericCSVQueryBinding`, strict JSON helpers, executor dispatch callback, and gateway result types.
- Produces: frozen family constants; `clinicaltrials_pubmed_central_shadow_registry()` with only the PubMed route replaced at this stage; parsing profile `("pubmed_v2", "pubmed-v2-ncbi-identity")`; private `_validate_identity_ncbi_error_envelope(...)`; private `_execute_ncbi_esearch_summary(...)`; exact overlay planner branch and runtime evidence.

- [x] **Step 1: Add declarative registry and planner RED tests**

Define and test these constants exactly:

```python
SHADOW_CATALOG_VERSION = "research-discovery-v2-clinicaltrials-pmc-shadow"
SHADOW_REGISTRY_VERSION = "research-discovery-v2-clinicaltrials-pmc-shadow-2026-08-21"
SHADOW_READINESS_VERSION = "research-discovery-readiness-v2-clinicaltrials-pmc-shadow"
ROUTE_POLICY_VERSION = "research-discovery-route-policy-v2-clinicaltrials-pmc"
CLINICALTRIALS_GOV_ADAPTER_ID = "clinicaltrials_gov_v2"
CLINICALTRIALS_GOV_ADAPTER_VERSION = "clinicaltrials-gov-v2"
PUBMED_CENTRAL_ADAPTER_ID = "pubmed_central_v2"
PUBMED_CENTRAL_ADAPTER_VERSION = "pubmed-central-v2"
PUBMED_IDENTITY_POLICY_VERSION = "research-discovery-route-policy-v2-foundation-pubmed-ncbi-identity-2026-08-21"
PUBMED_IDENTITY_ADAPTER_VERSION = "pubmed-v2-ncbi-identity"
NCBI_TOOL = "tldw_server"
NCBI_EMAIL = "contact@tldwproject.com"
CLINICALTRIALS_FIELDS = (
    "NCTId,BriefTitle,OfficialTitle,BriefSummary,OverallStatus,Condition,"
    "InterventionName,LeadSponsorName,StudyType,StartDate,CompletionDate,HasResults"
)
```

Define the exact family parser profiles at module scope so adapters and the generalized boundary equality test share one closed mapping:

```python
_CLINICALTRIALS_PROFILE = _ParsingProfile(
    max_input_bytes=2_097_152,
    max_records=50,
    max_depth=16,
    max_nodes=50_000,
    max_string_chars=65_536,
    max_numeric_token_chars=32,
    parse_deadline_ms=500,
)
_PMC_PROFILE = _ParsingProfile(
    max_input_bytes=2_097_152,
    max_records=100,
    max_depth=16,
    max_nodes=50_000,
    max_string_chars=65_536,
    max_numeric_token_chars=32,
    parse_deadline_ms=500,
)
_FAMILY_PARSING_PROFILES = MappingProxyType(
    {
        (CLINICALTRIALS_GOV_ADAPTER_ID, CLINICALTRIALS_GOV_ADAPTER_VERSION): (
            _CLINICALTRIALS_PROFILE
        ),
        (PUBMED_CENTRAL_ADAPTER_ID, PUBMED_CENTRAL_ADAPTER_VERSION): _PMC_PROFILE,
    }
)
```

The mapping is declarative in Task 5, but readiness is not published merely because a profile exists; Tasks 6/7 must execute their fixtures before Task 7 exposes the completed readiness factory.

`clinicaltrials_pubmed_central_shadow_registry()` must reconstruct foundation sources under the shadow catalog and replace exactly `pubmed_ncbi_eutils_pubmed_direct`. Freeze the overlay keys:

```python
(
    "db", "term", "retstart", "retmax", "retmode", "sort",
    "datetype", "mindate", "maxdate", "tool", "email", "id",
)
```

with `pagination_query_key="retstart"` and `query_value_policies=()`.

Planner assertions pin ESearch order `db,term,retstart,retmax,retmode,sort,tool,email` and ESummary order `db,retmode,tool,email` followed by deferred `id`. Reject foundation backend plus only one overlay version, omitted/swapped identity, user-injected identity, or any partial identity tuple.

In `planner.py`, define private dependency-leaf constants with the same exact bytes and golden-test them against the family exports; never import `clinicaltrials_pubmed_central` from the planner:

```python
_PUBMED_IDENTITY_POLICY_VERSION = (
    "research-discovery-route-policy-v2-foundation-pubmed-ncbi-identity-2026-08-21"
)
_PUBMED_IDENTITY_ADAPTER_VERSION = "pubmed-v2-ncbi-identity"
_NCBI_TOOL = "tldw_server"
_NCBI_EMAIL = "contact@tldwproject.com"
```

- [x] **Step 2: Run planner/constructor RED tests**

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest -q \
  tldw_Server_API/tests/Research/test_research_discovery_planner.py \
  tldw_Server_API/tests/Research/test_research_discovery_clinicaltrials_pubmed_central.py \
  tldw_Server_API/tests/Research/test_research_discovery_v2_compatibility.py
```

Expected: family module/overlay branch absent.

- [x] **Step 3: Implement exact registry replacement and overlay intent branch**

Keep the current foundation branch byte-identical and classify by exact tuple:

```python
foundation_pubmed = (
    route.route_id == "pubmed_ncbi_eutils_pubmed_direct"
    and route.backend_id == "ncbi_eutils_pubmed"
    and route.adapter_id == "pubmed_v2"
    and route.adapter_version == "foundation-v2"
    and route.policy.policy_version == "research-discovery-route-policy-v2-foundation"
)
identity_pubmed = (
    route.route_id == "pubmed_ncbi_eutils_pubmed_direct"
    and route.backend_id == "ncbi_eutils_pubmed"
    and route.adapter_id == "pubmed_v2"
    and route.adapter_version == _PUBMED_IDENTITY_ADAPTER_VERSION
    and route.policy.policy_version == _PUBMED_IDENTITY_POLICY_VERSION
)
```

Only `identity_pubmed` appends `QueryPair("tool", _NCBI_TOOL)` and `QueryPair("email", _NCBI_EMAIL)` to both intents; any route that resembles PubMed but matches neither exact tuple fails planning.

- [x] **Step 4: Add shared-adapter RED tests before extracting the helper**

Through `foundation_gateway_adapters()["pubmed_v2"]`, execute a nonempty overlay plan and assert:

- exact identity pairs appear on ESearch and ESummary;
- one logical page and two physical dispatches/debits;
- normalized PubMed output matches foundation semantics;
- traces expose query keys only;
- `{"error":"API rate limit exceeded","count":"11"}` maps to the typed rate-limited outcome for the overlay;
- extra/missing keys, JSON-number count, or near-match error text remains a provider-payload failure;
- repr paths contain neither identity values nor deferred numeric values;
- foundation success, empty, diagnostics, malformed, rate, cancellation, output, and accounting fixtures remain exact.

Witness this second RED cycle before extracting the helper:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest -q \
  tldw_Server_API/tests/Research/test_research_discovery_pubmed_gateway_adapter.py \
  tldw_Server_API/tests/Research/test_research_discovery_clinicaltrials_pubmed_central.py
```

Expected: the named overlay profile, private two-hop seam, and strict identity error-envelope cases fail because the shared runtime path is not implemented yet; foundation PubMed cases remain green.

- [x] **Step 5: Extract only the private two-hop execution seam**

Register the overlay explicitly in `_PARSING_PROFILES` using a local literal/private `_PUBMED_IDENTITY_ADAPTER_VERSION = "pubmed-v2-ncbi-identity"`; `gateway_adapters.py` must not import the family module. Extract a private helper with closed callback parameters, not a public framework. Define these exact private aliases from types already present in `gateway_adapters.py`:

```python
_TrustedNCBIInputs = tuple[
    PlannedDispatchGroup,
    _ParsingProfile,
    int,
    int,
    int,
    DeferredNumericCSVQueryBinding,
]
class _TrustedNCBIInputsCallback(Protocol):
    def __call__(self, group: object) -> _TrustedNCBIInputs: ...


class _NCBIESearchIDsCallback(Protocol):
    def __call__(
        self,
        payload: Any,
        *,
        profile: _ParsingProfile,
        guard: _ParseGuard,
        retstart: int,
        retmax: int,
        binding: DeferredNumericCSVQueryBinding,
    ) -> tuple[tuple[str, int], ...]: ...


class _NCBISummaryRecordsCallback(Protocol):
    def __call__(
        self,
        payload: Any,
        *,
        expected_ids: tuple[str, ...],
        guard: _ParseGuard,
    ) -> tuple[dict[str, Any], ...]: ...


async def _execute_ncbi_esearch_summary(
    group: object,
    dispatch: BoundDispatch,
    clock: MonotonicClock,
    *,
    trusted_inputs: _TrustedNCBIInputsCallback,
    parse_esearch_ids: _NCBIESearchIDsCallback,
    parse_summary_records: _NCBISummaryRecordsCallback,
    strict_rate_envelope: bool,
) -> DiscoveryAdapterResult:
    """Execute one sealed ESearch and conditional ESummary pair."""
```

Add one pre-parser error-envelope guard without changing `_ncbi_json_root()` foundation behavior:

```python
_NCBI_RATE_COUNT_RE = re.compile(r"(?:0|[1-9][0-9]*)\Z", re.ASCII)


def _validate_identity_ncbi_error_envelope(
    payload: Any,
    profile: _ParsingProfile,
) -> None:
    root = _require_dict(payload)
    if "error" not in root:
        return
    if (
        set(root) == {"error", "count"}
        and root["error"] == "API rate limit exceeded"
        and type(root["count"]) is str
        and len(root["count"]) <= profile.max_numeric_token_chars
        and _NCBI_RATE_COUNT_RE.fullmatch(root["count"]) is not None
    ):
        raise DiscoveryAdapterError("provider_rate_limited")
    raise _PayloadInvalid
```

`trusted_inputs(group)` returns the existing six-item trusted tuple. `parse_esearch_ids(...)` receives `payload`, `_ParsingProfile`, `_ParseGuard`, `retstart`, `retmax`, and the deferred binding, and returns `(text_id, numeric_id)` pairs. `parse_summary_records(...)` receives `payload`, ordered expected text IDs, and `_ParseGuard`, and returns normalized `dict[str, Any]` records.

Generalize `_trusted_pubmed_inputs()` only by exact adapter version; its returned six-item tuple remains unchanged. Preserve every existing foundation check and classify the pair shapes as follows after the existing term/decimal type checks:

```python
summary_pairs = tuple((pair.name, pair.value) for pair in summary.query_pairs)
if (
    len(search_pairs) not in {6, 8}
    or len(summary_pairs) not in {2, 4}
    or search_pairs[0] != ("db", "pubmed")
    or search_pairs[1][0] != "term"
    or type(search_pairs[1][1]) is not str
    or not search_pairs[1][1]
    or search_pairs[2][0] != "retstart"
    or search_pairs[3][0] != "retmax"
):
    raise DiscoveryAdapterError("provider_payload_invalid")
base_search_pairs = (
    ("db", "pubmed"),
    search_pairs[1],
    search_pairs[2],
    search_pairs[3],
    ("retmode", "json"),
    ("sort", "relevance"),
)
base_summary_pairs = (("db", "pubmed"), ("retmode", "json"))
identity_pairs = (("tool", "tldw_server"), ("email", "contact@tldwproject.com"))

if group.adapter_version == "foundation-v2":
    shape_valid = search_pairs == base_search_pairs and summary_pairs == base_summary_pairs
elif group.adapter_version == _PUBMED_IDENTITY_ADAPTER_VERSION:
    shape_valid = (
        search_pairs == base_search_pairs + identity_pairs
        and summary_pairs == base_summary_pairs + identity_pairs
    )
else:
    shape_valid = False
if not shape_valid:
    raise DiscoveryAdapterError("provider_payload_invalid")
```

This replaces only the old exact `len(search_pairs) == 6` / two-summary-pair shape predicate. A missing, reordered, duplicated, partial, user-supplied, or wrong identity pair fails. Register `("pubmed_v2", "pubmed-v2-ncbi-identity")` to `_FOUNDATION_PROFILE` in shared `_PARSING_PROFILES`; keep `("pubmed_v2", "foundation-v2")` byte/behavior exact. Select `strict_rate_envelope` from the same exact version branch, never from pair presence alone.

The helper sequence is exact: dispatch/search → `_checked_response` → `_strict_json`; when `strict_rate_envelope`, invoke `_validate_identity_ncbi_error_envelope(search_payload, profile)` before the ESearch callback; short-circuit with empty candidates when IDs are empty; build one `NumericCSVBindingValues(binding.binding_id, numeric_ids)`; dispatch the conditional summary with that binding; strict-parse and prevalidate its error envelope; call the summary callback with ordered text IDs; fingerprint/deduplicate complete normalized records; conflicting same-fingerprint records fail; return candidates. It creates no retry, sleep, page loop, or third call. `_execute_pubmed_adapter()` remains an exact wrapper using `_trusted_pubmed_inputs`, `_pubmed_esearch_ids`, and `_pubmed_summary_records`; it passes `strict_rate_envelope=True` only for the exact overlay adapter version and `False` for `foundation-v2`. The family PMC wrapper passes `True`. Foundation PubMed therefore retains legacy classification.

- [x] **Step 6: Run GREEN and commit**

Refresh only the existing shared `planner.py` and `gateway_adapters.py` import/AST digest entries. Do not add the new family root to the singular boundary harness yet: Task 8 performs that harness generalization once the complete family factories and runtime shapes exist. Run planner, PubMed adapter, family, compatibility, and the still-legacy boundary harness.

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest -q \
  tldw_Server_API/tests/Research/test_research_discovery_planner.py \
  tldw_Server_API/tests/Research/test_research_discovery_pubmed_gateway_adapter.py \
  tldw_Server_API/tests/Research/test_research_discovery_clinicaltrials_pubmed_central.py \
  tldw_Server_API/tests/Research/test_research_discovery_v2_compatibility.py \
  tldw_Server_API/tests/Research/test_research_discovery_network_boundary.py
```

Expected: PASS for both byte-exact foundation PubMed and the identity-bearing overlay; the boundary file at this commit certifies only the changed shared-module digests plus its pre-existing provider family.

```bash
git add \
  tldw_Server_API/app/core/Research/discovery/clinicaltrials_pubmed_central.py \
  tldw_Server_API/app/core/Research/discovery/planner.py \
  tldw_Server_API/app/core/Research/discovery/gateway_adapters.py \
  tldw_Server_API/tests/Research/test_research_discovery_planner.py \
  tldw_Server_API/tests/Research/test_research_discovery_pubmed_gateway_adapter.py \
  tldw_Server_API/tests/Research/test_research_discovery_clinicaltrials_pubmed_central.py \
  tldw_Server_API/tests/Research/test_research_discovery_v2_compatibility.py \
  tldw_Server_API/tests/Research/test_research_discovery_network_boundary.py
git commit -m "feat(research): add NCBI identity overlay and two-hop seam (TASK-12968.6)"
```

---

## Stage 3: Fixture-Certified Provider Adapters

**Goal:** Add strict, bounded ClinicalTrials.gov and PMC adapters and expose the completed family registry/readiness/adapter factories only after runtime fixtures pass.

**Success Criteria:** ClinicalTrials.gov exercises real opaque continuation with atomic route failure; PMC executes exact ESearch/ESummary metadata discovery without EFetch; normalized records retain only approved fields and synthesized links; cancellation and partial outcomes preserve existing executor semantics.

**Tests:** New family test module, wholly synthetic fixtures, parser mutation tables, planner/adapter/accounting/cancellation/partial-outcome tests, and no-network tripwires.

**Status:** Complete

### Task 6: Add the fixture-only ClinicalTrials.gov adapter

**Files:**

- Modify: `tldw_Server_API/app/core/Research/discovery/clinicaltrials_pubmed_central.py`
- Modify: `tldw_Server_API/tests/Research/test_research_discovery_clinicaltrials_pubmed_central.py`
- Create: the three `clinicaltrials_*.json` fixtures from the File Map

**Interfaces:**

- Consumes: shared opaque cursor path, strict JSON/gateway helpers, `DiscoveryAdapterResult`, normalized candidate contracts, foundation registry composition.
- Produces: exact ClinicalTrials.gov backend/source/route; `_trusted_clinicaltrials_inputs`; `_clinicaltrials_page`; `_clinicaltrials_record`; `_LegacySummaryParser`; `_contains_url_material`; `_plain_clinical_text`; `_legacy_summary_text`; `_partial_date`; `_execute_clinicaltrials_adapter`.

- [x] **Step 1: Check in wholly synthetic valid-shape fixtures and RED constructor tests**

Use invented values only. The valid pair must have frozen `totalCount=2`, one study per page, a visible-ASCII `nextPageToken` only on page one, distinct unused-looking IDs such as `NCT90000001`/`NCT90000002`, and hostile-but-synthetic markup in the summary. The empty fixture is exactly:

```json
{"totalCount":0,"studies":[]}
```

Write `clinicaltrials_success_page_1.json` with this exact synthetic content:

```json
{
  "totalCount": 2,
  "studies": [
    {
      "protocolSection": {
        "identificationModule": {
          "nctId": "NCT90000001",
          "briefTitle": "Synthetic bounded trial one",
          "officialTitle": "Synthetic official trial title one"
        },
        "descriptionModule": {
          "briefSummary": "<p>Synthetic summary &amp; bounded details.</p>"
        },
        "statusModule": {
          "overallStatus": "RECRUITING",
          "startDateStruct": {"date": "2026-01"},
          "completionDateStruct": {"date": "2027"}
        },
        "conditionsModule": {"conditions": ["Synthetic condition"]},
        "armsInterventionsModule": {
          "interventions": [{"name": "Synthetic intervention"}]
        },
        "sponsorCollaboratorsModule": {
          "leadSponsor": {"name": "Synthetic sponsor"}
        },
        "designModule": {"studyType": "INTERVENTIONAL"}
      },
      "hasResults": false
    }
  ],
  "nextPageToken": "synthetic+token/one=="
}
```

Write `clinicaltrials_success_page_2.json` with this exact synthetic content:

```json
{
  "totalCount": 2,
  "studies": [
    {
      "protocolSection": {
        "identificationModule": {
          "nctId": "NCT90000002",
          "officialTitle": "Synthetic official-only trial title two"
        },
        "statusModule": {
          "overallStatus": "COMPLETED",
          "startDateStruct": {"date": "2025-02-28"},
          "completionDateStruct": {"date": "2026-02-28"}
        },
        "conditionsModule": {"conditions": []},
        "armsInterventionsModule": {"interventions": []},
        "sponsorCollaboratorsModule": {
          "leadSponsor": {"name": "Synthetic sponsor two"}
        },
        "designModule": {"studyType": "OBSERVATIONAL"}
      },
      "hasResults": true
    }
  ]
}
```

Assert route policy exactly:

```python
assert route.policy.allowed_query_keys == (
    "query.term", "format", "markupFormat", "fields",
    "pageSize", "countTotal", "pageToken",
)
assert route.policy.pagination_query_key == "pageToken"
assert route.policy.limits.max_pages == 2
assert route.policy.limits.max_results == 100
assert route.max_physical_dispatches == 2
```

Pin the exact source constructor: ID `clinicaltrials_gov`; display name `ClinicalTrials.gov`; site host `clinicaltrials.gov`; aliases `('clinical_trials_gov', 'clinical_trials')`; categories `('biomedical', 'clinical_trials')`; content types `('clinical_trials', 'study_records', 'summaries')`; declared surfaces `('standalone_search', 'deep_research')`; route reference `clinicaltrials_gov_studies_search_direct` with predicate `None`; priority `110`; shadow catalog version. Pin backend `BackendDefinition('clinicaltrials_gov_api_v2', 'ClinicalTrials.gov API v2')` and exact origin `https://clinicaltrials.gov:443`.

Pin the route as `direct`, query modes `('general_free_text',)`, source constraint `native_corpus`, attribution basis `native_nct_record`, credential requirement `none`, fallback order `0`, adapter ID/version `clinicaltrials_gov_v2` / `clinicaltrials-gov-v2`, origin/path/method, no body, zero retries/redirects, 20,000 ms, request-body ceiling 16,384, response ceiling 2,097,152, `format=json`, `markupFormat=legacy`, frozen projection, `pageSize=50`, `countTotal=true`, and absent first-page token. Pin its parser profile to input bytes `2_097_152`, records `50`, depth `16`, nodes `50_000`, string chars `65_536`, numeric-token chars `32`, and deadline `500` ms. Assert the compiled single-route plan has `max_wall_time_ms=40_000`; Task 7 adds the two-route `80_000` assertion after PMC exists.

- [x] **Step 2: Add parser/execution RED matrices**

Table-drive bounded mutations rather than adding one file per error. Cover:

- exact empty and two-page success; repeated/invalid token; frozen count change; page/cardinality/count biconditional failures; 50/page and 100 cumulative raw ceilings;
- a `totalCount > 100` mutation with two valid 50-record pages and a valid page-two token succeeds with exactly 100 raw records, two pages/two debits, and no third reservation/call; a small requested result limit may still make the bounded second call when page one has a valid token;
- missing/invalid/conflicting NCT ID; identical duplicate collapse only after raw accounting; conflicting duplicate atomic failure;
- brief/official title requiredness and bounds;
- wrong optional containers/scalars fail, while explicitly over-bound optional text/list/date values drop as specified;
- `PartialDate` accepts only calendar-valid `YYYY`, `YYYY-MM`, or `YYYY-MM-DD`; direct RED cases prove `0000`, `2026-13`, and `2026-02-30` return `None` so the optional field is dropped, while a present non-string date fails at the record-shape boundary;
- `HTMLParser` conversion drops tags/comments/declarations, resolves entities, rejects controls/NUL/surrogates and residual URL material, and safely drops hostile CommonMark links/images/autolinks/embedded HTML;
- a plain `https://example.org/article` with no query, fragment, or credentials is still URL material: an unsafe sole Clinical title fails, an unsafe alternate title drops when the other title remains valid, and an optional summary containing it drops before `abstract`/`snippet` construction;
- exact normalized projection, empty authors/identifiers/PDF, bounded source metadata, and synthesized `https://clinicaltrials.gov/study/{NCT_ID}` only;
- HTTP 429, timeout, redirect, malformed/non-JSON, byte/depth/node/string/deadline limits;
- page-two failure publishes no trial candidate;
- cancellation after token parsing but before continuation produces journal `1/1/0/0` and no second reservation/call.

- [x] **Step 3: Run family RED tests**

Until Task 7 exposes the complete two-adapter production family factory, use test-only composition in `test_research_discovery_clinicaltrials_pubmed_central.py`:

```python
def _clinicaltrials_test_readiness(mode: ExecutionMode) -> ReadinessOverlay:
    foundation = foundation_readiness(mode)
    return ReadinessOverlay(
        overlay_version=SHADOW_READINESS_VERSION,
        execution_mode=mode,
        routes=foundation.routes
        + (
            RouteReadiness(
                route_id="clinicaltrials_gov_studies_search_direct",
                state=ReadinessState.READY,
                credential_status=CredentialStatus.NOT_REQUIRED,
                reason=f"{mode.value}_ready",
            ),
        ),
    )


def _clinicaltrials_test_adapters(clock: MonotonicClock) -> Mapping[str, DiscoveryAdapter]:
    async def adapter(
        group: PlannedDispatchGroup,
        dispatch: BoundDispatch,
    ) -> DiscoveryAdapterResult:
        return await _execute_clinicaltrials_adapter(group, dispatch, clock)

    return MappingProxyType({CLINICALTRIALS_GOV_ADAPTER_ID: adapter})
```

This helper is test-only. Do not publish a one-entry production adapter or readiness factory; Task 7 replaces the test composition with the completed public family factories and reconciliation evidence.

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest -q tldw_Server_API/tests/Research/test_research_discovery_clinicaltrials_pubmed_central.py -k clinicaltrials
```

Expected: route/parser/adapter symbols and runtime behavior are absent.

- [x] **Step 4: Implement the exact route, parser, sanitizer, and adapter**

Use these exact internal state/signature boundaries:

```python
@dataclass(frozen=True, slots=True)
class _ClinicalTrialsPage:
    total_count: int
    records: tuple[dict[str, Any], ...]
    next_page_token: str | None


def _trusted_clinicaltrials_inputs(
    group: object,
) -> tuple[PlannedDispatchGroup, _ParsingProfile, int, int]:
    """Return exact group, profile, max input bytes, and requested page size."""


def _plain_clinical_text(
    value: Any,
    *,
    max_chars: int,
    required: bool,
) -> str | None:
    """Normalize Unicode whitespace and reject controls, markup, or URL material."""


def _legacy_summary_text(value: Any) -> str | None:
    """Return at most 16,384 inert characters, or drop unsafe optional content."""


def _contains_url_material(value: str) -> bool:
    """Reject any bounded URL/URI token in provider-supplied human text."""


def _partial_date(value: Any) -> str | None:
    """Return a valid exact partial date, drop invalid strings, and reject wrong types."""


def _clinicaltrials_record(raw: Any, *, guard: _ParseGuard) -> dict[str, Any]:
    """Normalize only the frozen study projection."""


def _clinicaltrials_page(
    payload: Any,
    *,
    guard: _ParseGuard,
    page_size: int,
) -> _ClinicalTrialsPage:
    """Validate one strict response page without applying cross-page state."""
```

`_trusted_clinicaltrials_inputs()` must reject unless the group matches the exact route/backend/adapter/version/policy, fallback `0`, empty filters, allowance `pages=2/physical=2/redirects=0/retries=0`, route limits `2/0/0/100/16_384/2_097_152/20_000`, one `SEARCH` GET intent at `/api/v2/studies`, no body/bindings, and the exact first-page ordered query tuple. Parse `pageSize` as a canonical positive decimal no greater than 50 and return `min(profile.max_input_bytes, limits.max_response_bytes)`.

Implement the summary sanitizer as data-node-only parsing:

```python
_COMMONMARK_LINK_RE = re.compile(
    r"!?\[[^\]\r\n]{0,1024}\]\([^\)\r\n]{0,4096}\)|<[^>\r\n]{0,4096}://[^>\r\n]{0,4096}>",
    re.ASCII,
)
_URL_MATERIAL_RE = re.compile(
    r"(?:https?://|ftp://|www\.|mailto:|data:|javascript:)[^\s<>\x00-\x1f]{0,4096}",
    re.IGNORECASE | re.ASCII,
)


def _contains_url_material(value: str) -> bool:
    return (
        _URL_MATERIAL_RE.search(value) is not None
        or has_unsafe_url_material(value)
    )


class _LegacySummaryParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.ignored_depth = 0

    def handle_starttag(self, tag: str, _attrs: list[tuple[str, str | None]]) -> None:
        if tag.casefold() in {"script", "style"}:
            self.ignored_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.casefold() in {"script", "style"} and self.ignored_depth:
            self.ignored_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self.ignored_depth:
            self.parts.append(data)
```

Before feeding, reject NUL/surrogates/control characters, input over 65,536 characters, `_COMMONMARK_LINK_RE.search(value)`, and `_contains_url_material(value)`. After `feed()`/`close()`, collapse Unicode whitespace, reject residual markup or `_contains_url_material(text)`, and drop rather than fail an empty/unsafe/over-16,384 optional summary. Derive `snippet` from the first 1,024 characters of the accepted normalized summary only after that post-sanitization check. `_plain_clinical_text()` uses the same control/surrogate/whitespace/any-URL checks but rejects markup outright and applies each field's required-versus-drop rule. Process brief and official titles independently as droppable candidates, then fail only when neither safe nonempty title remains; no unsafe title may survive into `source_metadata`.

Implement `_partial_date()` by exact regex branch and stdlib calendar construction:

```python
if type(value) is not str:
    raise _PayloadInvalid
try:
    if re.fullmatch(r"[0-9]{4}", value):
        date(int(value), 1, 1)
    elif re.fullmatch(r"[0-9]{4}-[0-9]{2}", value):
        year, month = map(int, value.split("-"))
        date(year, month, 1)
    elif re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", value):
        if date.fromisoformat(value).isoformat() != value:
            return None
    else:
        return None
except ValueError:
    return None
return value
```

`_clinicaltrials_record()` must use `_require_dict` at every declared container, require `NCT[0-9]{8}`, normalize brief/official titles with max 1,024/4,096 and select brief then official, and implement the spec's exact fail-versus-drop table for summary, status, conditions, intervention names, sponsor, study type, partial dates, and `hasResults`. Construct `_base_record(...)` with empty authors, only `provider_ids={"nct_id": nct_id}`, no DOI/PMID/PMCID/arXiv/PDF, provider `clinicaltrials_gov`, and URL `f"https://clinicaltrials.gov/study/{nct_id}"`; then add only the approved present `source_metadata` keys.

`_clinicaltrials_page()` must require exact-type nonnegative JSON integer `totalCount`, a studies list no larger than `page_size`, and an optional exact nonempty visible-ASCII token no longer than 1,024 characters. Return normalized records in provider order; do not deduplicate or decide terminal state inside this per-page parser.

Keep sanitizer state local per record and enforce input/output bounds before candidate construction. The request policies are exactly:

```python
(
    LiteralTermsQueryValuePolicy("query.term", "", 8, 32),
    ExactQueryValuePolicy("format", "json"),
    ExactQueryValuePolicy("markupFormat", "legacy"),
    ExactQueryValuePolicy("fields", CLINICALTRIALS_FIELDS),
    BoundedDecimalQueryValuePolicy("pageSize", 50),
    ExactQueryValuePolicy("countTotal", "true"),
    OpaqueCursorQueryValuePolicy("pageToken", 1_024, required=False),
)
```

Implement cross-page state in this exact order:

```python
async def _execute_clinicaltrials_adapter(
    group: object,
    dispatch: BoundDispatch,
    clock: MonotonicClock,
) -> DiscoveryAdapterResult:
    trusted, profile, max_input_bytes, page_size = _trusted_clinicaltrials_inputs(group)
    intent = trusted.intents[0]
    staged_by_nct: dict[str, dict[str, Any]] = {}
    frozen_total: int | None = None
    cumulative_raw = 0
    response = await dispatch(intent)

    for page_index in range(trusted.limits.max_pages):
        payload, guard = _strict_json(
            _checked_response(response),
            profile=profile,
            max_input_bytes=max_input_bytes,
            clock=clock,
        )
        page = _clinicaltrials_page(payload, guard=guard, page_size=page_size)
        if frozen_total is None:
            frozen_total = page.total_count
        elif page.total_count != frozen_total:
            raise _PayloadInvalid
        if frozen_total > cumulative_raw and not page.records:
            raise _PayloadInvalid

        cumulative_raw += len(page.records)
        if cumulative_raw > frozen_total or cumulative_raw > trusted.limits.max_results:
            raise _PayloadInvalid
        token_required = cumulative_raw < frozen_total
        if token_required != (page.next_page_token is not None):
            raise _PayloadInvalid

        for record in page.records:
            nct_id = cast(str, cast(dict[str, str], record["provider_ids"])["nct_id"])
            previous = staged_by_nct.get(nct_id)
            if previous is not None and previous != record:
                raise _PayloadInvalid
            staged_by_nct.setdefault(nct_id, record)
        guard.checkpoint()

        capacity_remains = (
            page_index + 1 < trusted.limits.max_pages
            and cumulative_raw < trusted.limits.max_results
            and page_index + 1 < trusted.allowance.physical_dispatches
        )
        if not token_required or not capacity_remains:
            break
        response = await dispatch(intent, cursor=OpaqueCursor(cast(str, page.next_page_token)))

    candidates = tuple(
        DiscoveryCandidate(
            DiscoveryOutcomeIdentity.from_fingerprint(build_fingerprint(record)).document_id,
            record,
        )
        for record in staged_by_nct.values()
    )
    return DiscoveryAdapterResult(candidates)
```

Wrap `_PayloadInvalid`, `_ParseLimitExceeded`, and `_ParseDeadlineExceeded` through `_raise_adapter_error`; preserve `DiscoveryAdapterError`; collapse unexpected `KeyError`, `TypeError`, `ValueError`, and `OverflowError` to `provider_payload_invalid`. The page/token biconditional is validated before `capacity_remains`; therefore a valid page-two token at the 100-record/two-page ceiling is discarded without constructing a third reservation or call. Candidates are constructed only after every dispatched page validates, preserving route atomicity.

- [x] **Step 5: Run GREEN, inspect fixture isolation, and commit**

Run the family tests with `-k clinicaltrials` plus executor/gateway regressions. Inspect fixtures for copied provider values, URLs, contacts, and raw source records. The generalized cross-family import/runtime/network/auth boundary is deliberately deferred to Task 8; do not add a temporary ClinicalTrials-only branch to the singular harness here.

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest -q \
  tldw_Server_API/tests/Research/test_research_discovery_clinicaltrials_pubmed_central.py \
  -k clinicaltrials
python -m pytest -q \
  tldw_Server_API/tests/Research/test_research_discovery_executor.py \
  tldw_Server_API/tests/Research/test_research_discovery_gateway.py
```

Expected: PASS with exactly two ClinicalTrials.gov dispatches at the ceiling and no third reservation; comprehensive boundary certification remains pending Task 8.

```bash
git add \
  tldw_Server_API/app/core/Research/discovery/clinicaltrials_pubmed_central.py \
  tldw_Server_API/tests/Research/test_research_discovery_clinicaltrials_pubmed_central.py \
  tldw_Server_API/tests/fixtures/research_discovery_gateway_adapters/clinicaltrials_success_page_1.json \
  tldw_Server_API/tests/fixtures/research_discovery_gateway_adapters/clinicaltrials_success_page_2.json \
  tldw_Server_API/tests/fixtures/research_discovery_gateway_adapters/clinicaltrials_empty.json
git commit -m "feat(research): add fixture-only ClinicalTrials.gov adapter (TASK-12968.6)"
```

### Task 7: Add the fixture-only PMC adapter and complete family maps/readiness

**Files:**

- Modify: `tldw_Server_API/app/core/Research/discovery/clinicaltrials_pubmed_central.py`
- Modify: `tldw_Server_API/app/core/Research/discovery/planner.py`
- Modify: `tldw_Server_API/tests/Research/test_research_discovery_clinicaltrials_pubmed_central.py`
- Create: the three `pmc_*.json` fixtures from the File Map
- Modify: `tldw_Server_API/tests/Research/test_research_discovery_network_boundary.py`

**Interfaces:**

- Consumes: `_execute_ncbi_esearch_summary`, exact family registry constants, `DeferredNumericCSVQueryBinding`, foundation readiness, gateway adapter callable contract.
- Produces: exact PMC backend/source/route; `_trusted_pubmed_central_inputs`; `_pmc_uid`; `_pmc_identifier_scalar`; `_plain_pmc_text`; `_pmc_esearch_ids`; `_pmc_article_ids`; `_pmc_record`; `_pmc_summary_records`; `_execute_pubmed_central_adapter`; `_compose_adapter_maps(*adapter_maps: Mapping[str, DiscoveryAdapter]) -> Mapping[str, DiscoveryAdapter]`; `clinicaltrials_pubmed_central_shadow_readiness(execution_mode)`; `clinicaltrials_pubmed_central_gateway_adapters(*, monotonic_clock=time.monotonic)`.

- [x] **Step 1: Add synthetic PMC fixtures and exact planner RED tests**

Use canonical string UIDs such as `"9000001"` and `"9000002"`. ESearch fixture root is `header` + `esearchresult` with string `count`, `retmax`, `retstart`, and ordered `idlist`; ESummary root is `header` + `result`, exact `uids`, and UID-keyed records whose PMCID is exactly `PMC` plus the UID.

Write `pmc_esearch_success.json` exactly:

```json
{
  "header": {"type": "esearch", "version": "0.3"},
  "esearchresult": {
    "count": "2",
    "retmax": "2",
    "retstart": "0",
    "idlist": ["9000001", "9000002"]
  }
}
```

Write `pmc_esearch_empty.json` exactly:

```json
{
  "header": {"type": "esearch", "version": "0.3"},
  "esearchresult": {
    "count": "0",
    "retmax": "0",
    "retstart": "0",
    "idlist": []
  }
}
```

Write `pmc_esummary_success.json` exactly:

```json
{
  "header": {"type": "esummary", "version": "0.3"},
  "result": {
    "uids": ["9000001", "9000002"],
    "9000001": {
      "uid": "9000001",
      "title": "Synthetic PMC metadata record one",
      "authors": [{"name": "Synthetic Author One"}],
      "articleids": [
        {"idtype": "pmcid", "value": "PMC9000001"},
        {"idtype": "doi", "value": "10.5555/synthetic.pmc.1"},
        {"idtype": "pmid", "value": "12345678"}
      ]
    },
    "9000002": {
      "uid": "9000002",
      "title": "Synthetic PMC metadata record two",
      "authors": [{"name": "Synthetic Author Two"}],
      "articleids": [
        {"idtype": "pmcid", "value": "PMC9000002"},
        {"idtype": "pmid", "value": "0"}
      ]
    }
  }
}
```

Pin the exact source constructor: ID `pubmed_central`; display name `PubMed Central`; site host `pmc.ncbi.nlm.nih.gov`; aliases `('pmc', 'pub_med_central')`; categories `('biomedical', 'open_access')`; content types `('papers', 'full_text_archive', 'biomedical_metadata')`; declared surfaces `('standalone_search', 'deep_research')`; route reference `pubmed_central_esearch_summary_direct` with predicate `None`; priority `120`; shadow catalog version. Pin backend `BackendDefinition('ncbi_eutils_pmc', 'NCBI Entrez E-utilities for PMC')` and exact execution origin `https://eutils.ncbi.nlm.nih.gov:443`.

Pin the route as `direct`, query modes `('general_free_text',)`, source constraint `native_corpus`, attribution basis `ncbi_pmc_database`, credential requirement `none`, fallback order `0`, adapter ID/version `pubmed_central_v2` / `pubmed-central-v2`, max pages `1`, max results `100`, max physical dispatches `2`, zero retries/redirects, 20,000 ms, request-body ceiling 16,384, and response ceiling 2,097,152 per hop. Its exact paths are `('/entrez/eutils/esearch.fcgi', '/entrez/eutils/esummary.fcgi')`; ordered allowlist is `('db', 'term', 'retstart', 'retmax', 'retmode', 'tool', 'email', 'id')`; `pagination_query_key='retstart'`; `query_value_policies=()`. Pin its parser profile to input bytes `2_097_152`, records `100`, depth `16`, nodes `50_000`, string chars `65_536`, numeric-token chars `32`, and deadline `500` ms. Assert the compiled single-route plan has `max_wall_time_ms=40_000`; selecting both new routes has `80_000`.

Pin the two intents:

```python
assert esearch.query_pairs == (
    QueryPair("db", "pmc"),
    QueryPair("term", '"alpha" AND "beta"'),
    QueryPair("retstart", "0"),
    QueryPair("retmax", "100"),
    QueryPair("retmode", "json"),
    QueryPair("tool", "tldw_server"),
    QueryPair("email", "contact@tldwproject.com"),
)
assert esummary.query_pairs == (
    QueryPair("db", "pmc"),
    QueryPair("retmode", "json"),
    QueryPair("tool", "tldw_server"),
    QueryPair("email", "contact@tldwproject.com"),
)
assert esummary.query_bindings == (
    DeferredNumericCSVQueryBinding("pmc_esearch_ids", "id", 100, 16),
)
```

Require the exact route/backend/adapter/version/policy tuple and reject `sort`, partial identity matches, or generic one-path planning.

Use planner-local literals/private constants for `pubmed_central_esearch_summary_direct`, `ncbi_eutils_pmc`, `pubmed_central_v2`, `pubmed-central-v2`, `research-discovery-route-policy-v2-clinicaltrials-pmc`, `tldw_server`, and `contact@tldwproject.com`; golden-test them against the public family constants. `planner.py` must not import `clinicaltrials_pubmed_central`.

Now that both provider routes exist, compile a plan selecting ClinicalTrials.gov plus PMC and assert exact aggregate `max_wall_time_ms == 80_000` (each route contributes `40_000`). This assertion belongs here, not in Task 6.

In the family module, import `_TrustedNCBIInputs` from `gateway_adapters.py`, import `MAX_PAGINATION_CURSOR` from `contracts.py`, and define the parser-local missing-value sentinel immediately after imports:

```python
_MISSING = object()
```

Do not import the shared module's private sentinel or give `_MISSING` any diagnostic representation path.

- [x] **Step 2: Add PMC parser/execution RED matrices**

Cover:

- canonical unsigned decimal-string `count`/`retstart`/`retmax`; reject JSON numbers, signs, whitespace, exponent, leading zero, and >32 characters; require returned `retstart` to equal requested `0`, returned `retmax == len(idlist) <= requested retmax`, `retstart + retmax <= count`, and forbid a positive count with an empty ID list;
- unique positive UID strings ≤16; exact ESummary UID set/keys; restore ESearch order;
- empty ESearch makes one debit and no ESummary reservation/call;
- `articleids` objects contain exactly `idtype`/`value`; reject `id` alias/extra keys; require one `pmcid == "PMC" + uid`; optional DOI must canonicalize successfully when present, including a synthetic `https://doi.org/10.5555/pmc.synthetic` form; missing/`"0"` PMID absent, otherwise positive decimal ≤16;
- numeric UID remains transport/correlation only and is absent from normalized provider identity;
- required plain title ≤4,096; ≤64 author objects with plain `name` ≤512; normalize Unicode whitespace and reject controls/NUL/surrogates, markup, and any URL/URI token in both title and author names; a plain `https://example.org/article` without query/fragment/credentials fails title/author parsing; ignore journal/date; always no abstract/snippet/PDF;
- the same `https://doi.org/10.5555/pmc.synthetic` accepted and canonicalized in a recognized DOI identifier fails if injected into PMC title or author human text;
- two distinct PMC UIDs that collapse to one canonical fingerprint/DOI but carry different normalized records fail the entire route; an exact duplicate normalized identity may collapse deterministically;
- a positive `count` greater than returned IDs is accepted as an unfetched remainder but causes no continuation, hidden reservation, or unsupported truncation-count claim;
- exact strict NCBI JSON rate envelope, malformed lookalikes, HTTP 429, timeout, cancellation before summary with journal `1/1/0/0`, and no retry/sleep/pacing;
- no EFetch/OAI/HTML/JATS/PDF/third call;
- independently successful PMC survives a malformed ClinicalTrials.gov page two with explicit partial status and exact physical accounting.

- [x] **Step 3: Run PMC RED tests**

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest -q tldw_Server_API/tests/Research/test_research_discovery_clinicaltrials_pubmed_central.py -k 'pmc or family or partial'
```

Expected: PMC planner/parser/adapter/maps/readiness symbols are absent.

- [x] **Step 4: Implement the exact PMC branch and parser callbacks**

Place the closed PMC planner branch before generic one-path typed-general handling. Use these exact callback signatures:

```python
def _trusted_pubmed_central_inputs(
    group: object,
) -> _TrustedNCBIInputs:
    """Seal the exact PMC two-intent group and expose bounded values."""


def _pmc_esearch_ids(
    payload: Any,
    *,
    profile: _ParsingProfile,
    guard: _ParseGuard,
    retstart: int,
    retmax: int,
    binding: DeferredNumericCSVQueryBinding,
) -> tuple[tuple[str, int], ...]:
    """Return canonical ordered PMC UIDs plus numeric binding values."""


def _pmc_identifier_scalar(value: Any, *, max_chars: int) -> str:
    """Validate a bounded identifier scalar without applying human URL rules."""


def _plain_pmc_text(
    value: Any,
    *,
    max_chars: int,
    required: bool,
) -> str | None:
    """Normalize human text and reject controls, markup, or any URL token."""


def _pmc_article_ids(
    raw: Any,
    expected_uid: str,
    guard: _ParseGuard,
) -> tuple[str, str | None, str | None]:
    """Return required PMCID and optional DOI/PMID."""


def _pmc_record(raw: Any, expected_uid: str, guard: _ParseGuard) -> dict[str, Any]:
    """Normalize one PMC ESummary record without retaining the numeric UID."""


def _pmc_summary_records(
    payload: Any,
    *,
    expected_ids: tuple[str, ...],
    guard: _ParseGuard,
) -> tuple[dict[str, Any], ...]:
    """Validate an exact UID-keyed result and restore ESearch ordering."""


async def _execute_pubmed_central_adapter(
    group: object,
    dispatch: BoundDispatch,
    clock: MonotonicClock,
) -> DiscoveryAdapterResult:
    return await _execute_ncbi_esearch_summary(
        group,
        dispatch,
        clock,
        trusted_inputs=_trusted_pubmed_central_inputs,
        parse_esearch_ids=_pmc_esearch_ids,
        parse_summary_records=_pmc_summary_records,
        strict_rate_envelope=True,
    )
```

`_trusted_pubmed_central_inputs(group)` must validate and return the six-item trusted tuple while sealing `db=pmc`, binding ID `pmc_esearch_ids`, both identity pairs, exact paths/operations/ordered pairs, empty bodies, one deferred binding, allowance `pages=1/physical=2/redirects=0/retries=0`, limits `1/0/0/100/16_384/2_097_152/20_000`, and the exact family parser profile. It parses requested `retstart=0` and canonical positive `retmax<=100`; binding `max_items` must equal requested `retmax` and `max_item_chars` must equal 16.

Implement ESearch arithmetic exactly:

```python
root = _ncbi_json_root(payload, "esearch")
result = _require_dict(root.get("esearchresult", _MISSING))
count = _canonical_decimal_text(result.get("count", _MISSING), profile)
returned_start = _canonical_decimal_text(
    result.get("retstart", _MISSING), profile, maximum=MAX_PAGINATION_CURSOR
)
returned = _canonical_decimal_text(result.get("retmax", _MISSING), profile, maximum=retmax)
raw_ids = _require_list(result.get("idlist", _MISSING))
if (
    returned_start != retstart
    or returned != len(raw_ids)
    or returned_start + returned > count
    or (count > 0 and returned == 0)
    or len(raw_ids) > binding.max_items
):
    raise _PayloadInvalid
ids = tuple(_pmc_uid(value, binding.max_item_chars) for value in _guarded_items(raw_ids, guard))
if len({uid for uid, _number in ids}) != len(ids):
    raise _PayloadInvalid
return ids
```

`_pmc_uid(value, max_chars)` accepts only `[1-9][0-9]{0,15}` within `max_chars` and returns `(value, int(value))`. Validate optional NCBI diagnostic lists through the existing bounded helper. A valid unfetched remainder (`count > returned_start + returned`) does not create a second logical page.

Implement identifier/record validation exactly:

```python
def _pmc_identifier_scalar(value: Any, *, max_chars: int) -> str:
    if (
        type(value) is not str
        or not 1 <= len(value) <= max_chars
        or value != value.strip()
        or any(character.isspace() for character in value)
        or any(
            unicodedata.category(character) in {"Cc", "Cf", "Cs"}
            for character in value
        )
        or "<" in value
        or ">" in value
    ):
        raise _PayloadInvalid
    return value


article_ids = _require_list(raw)
if len(article_ids) > 64:
    raise _ParseLimitExceeded
recognized: dict[str, str] = {}
for item in _guarded_items(article_ids, guard):
    identifier = _require_dict(item)
    if set(identifier) != {"idtype", "value"}:
        raise _PayloadInvalid
    idtype = _pmc_identifier_scalar(identifier["idtype"], max_chars=32)
    value = _pmc_identifier_scalar(identifier["value"], max_chars=512)
    if idtype in {"pmcid", "doi", "pmid"}:
        if idtype in recognized:
            raise _PayloadInvalid
        recognized[idtype] = value

pmcid = recognized.get("pmcid")
if pmcid != f"PMC{expected_uid}" or re.fullmatch(r"PMC[1-9][0-9]{0,15}", pmcid or "") is None:
    raise _PayloadInvalid
doi = None if "doi" not in recognized else normalize_doi(recognized["doi"])
if "doi" in recognized and doi is None:
    raise _PayloadInvalid
raw_pmid = recognized.get("pmid")
pmid = None if raw_pmid in {None, "0"} else raw_pmid
if pmid is not None and re.fullmatch(r"[1-9][0-9]{0,15}", pmid) is None:
    raise _PayloadInvalid
```

`_plain_pmc_text()` is the exact title/author sanitizer: exact string, reject controls/NUL/surrogates/markup and `_contains_url_material(value)`, collapse Unicode whitespace, require nonempty when requested, enforce the supplied bound. `_pmc_identifier_scalar()` is separate and performs no whitespace normalization and no any-URL check; it only supplies structurally safe bounded material to the exact recognized identifier grammar. `_pmc_record()` requires `uid == expected_uid`; title ≤4,096; authors list ≤64 whose members are dicts with a required `name` ≤512; and `_pmc_article_ids(...)`. It ignores journal/date/unknown citation fields and constructs `_base_record(...)` with provider `pubmed_central`, provider IDs containing PMCID plus present DOI/PMID, synthesized PMC URL, and `abstract=None`, `snippet=None`, `pdf_url=None`. It does not put `expected_uid` in normalized output. DOI values bypass the human-text detector only inside `_pmc_article_ids()` and must still pass `normalize_doi()`; raw DOI URL text is never emitted.

`_pmc_summary_records()` requires `result.uids` to be an exact string set matching `expected_ids`, requires `set(result) == {"uids", *expected_ids}`, and returns `_pmc_record(result[uid], uid, guard)` in ESearch order. The shared helper fingerprints normalized records after the complete summary validates; exact duplicates may collapse, but same-fingerprint differing records raise `_PayloadInvalid` atomically.

Call `_execute_ncbi_esearch_summary()` with that trusted-input callback, `strict_rate_envelope=True`, and the PMC-specific ESearch-ID/ESummary-record callbacks. Synthesize only:

```python
url = f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/"
```

Never infer PMCID when provider `articleids` omits or mismatches it.

Expose the family map only now:

```python
def _compose_adapter_maps(
    *adapter_maps: Mapping[str, DiscoveryAdapter],
) -> Mapping[str, DiscoveryAdapter]:
    composed: dict[str, DiscoveryAdapter] = {}
    for adapter_map in adapter_maps:
        for adapter_id, adapter in adapter_map.items():
            if adapter_id in composed:
                raise ValueError(f"duplicate_adapter_id:{adapter_id}")
            composed[adapter_id] = adapter
    return MappingProxyType(composed)


def clinicaltrials_pubmed_central_gateway_adapters(
    *,
    monotonic_clock: MonotonicClock = time.monotonic,
) -> Mapping[str, DiscoveryAdapter]:
    async def clinicaltrials_adapter(
        group: PlannedDispatchGroup,
        dispatch: BoundDispatch,
    ) -> DiscoveryAdapterResult:
        return await _execute_clinicaltrials_adapter(group, dispatch, monotonic_clock)

    async def pubmed_central_adapter(
        group: PlannedDispatchGroup,
        dispatch: BoundDispatch,
    ) -> DiscoveryAdapterResult:
        return await _execute_pubmed_central_adapter(group, dispatch, monotonic_clock)

    return _compose_adapter_maps(
        {CLINICALTRIALS_GOV_ADAPTER_ID: clinicaltrials_adapter},
        {PUBMED_CENTRAL_ADAPTER_ID: pubmed_central_adapter},
    )
```

Reject duplicate IDs before freezing; return only these two family entries. The PubMed overlay continues through `foundation_gateway_adapters()["pubmed_v2"]`.

Build readiness by reconstructing foundation entries, deliberately replacing the same-ID PubMed entry, then appending ClinicalTrials.gov and PMC only after fixture-backed runtime tests pass. Reconciliation must bind ready PubMed to both `pubmed-v2-ncbi-identity` and the full overlay policy version—not route ID alone.

- [x] **Step 5: Run GREEN and commit**

Run the complete family test module, planner, PubMed adapter, executor, registry reconciliation, and the current legacy boundary tests. Update only the shared `planner.py` import/AST digest at this commit. Do not add the new family root/config/runtime cases until Task 8 generalizes the singular harness.

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest -q \
  tldw_Server_API/tests/Research/test_research_discovery_clinicaltrials_pubmed_central.py \
  tldw_Server_API/tests/Research/test_research_discovery_planner.py \
  tldw_Server_API/tests/Research/test_research_discovery_pubmed_gateway_adapter.py \
  tldw_Server_API/tests/Research/test_research_discovery_executor.py \
  tldw_Server_API/tests/Research/test_research_discovery_registry_reconciliation.py \
  tldw_Server_API/tests/Research/test_research_discovery_network_boundary.py
```

Expected: PASS with PMC using one logical page/two physical hops, complete family readiness, byte-exact foundation regressions, and the pre-existing boundary family unchanged. Task 8 supplies the new family's boundary certification.

```bash
git add \
  tldw_Server_API/app/core/Research/discovery/clinicaltrials_pubmed_central.py \
  tldw_Server_API/app/core/Research/discovery/planner.py \
  tldw_Server_API/tests/Research/test_research_discovery_clinicaltrials_pubmed_central.py \
  tldw_Server_API/tests/fixtures/research_discovery_gateway_adapters/pmc_esearch_success.json \
  tldw_Server_API/tests/fixtures/research_discovery_gateway_adapters/pmc_esearch_empty.json \
  tldw_Server_API/tests/fixtures/research_discovery_gateway_adapters/pmc_esummary_success.json \
  tldw_Server_API/tests/Research/test_research_discovery_network_boundary.py
git commit -m "feat(research): add fixture-only PMC discovery adapter (TASK-12968.6)"
```

---

## Stage 4: Composition, Security Boundary, and Inventory Evidence

**Goal:** Prove that the shadow family is exact, isolated, fixture-executable, and authoritatively reconciled with the frozen source inventory.

**Success Criteria:** Every family/runtime shape executes under no-network/no-auth tripwires; changed module closures and digests are pinned; production consumers cannot import or invoke the family; inventory rows 0026/0027 advance only to implemented/fixture-passed/live-not-run with no surface-ready claim.

**Tests:** Network boundary, v2 compatibility, registry reconciliation, Node inventory unit tests, Python schema tests, and fixed-as-of validator gates.

**Status:** Complete

### Task 8: Generalize the boundary harness and reconcile exact inventory rows

**Files:**

- Modify: `tldw_Server_API/tests/Research/test_research_discovery_network_boundary.py`
- Modify: `tldw_Server_API/tests/Research/test_research_discovery_v2_compatibility.py`
- Modify: `tldw_Server_API/tests/Research/test_research_discovery_registry_reconciliation.py`
- Modify: `Helper_Scripts/validate_research_source_inventory.mjs`
- Modify: `Helper_Scripts/tests/validate_research_source_inventory.test.mjs`
- Modify: `Docs/Design/research_source_inventory/research-source-coverage-ledger-2026-07-13.json`
- Regenerate: `Docs/Design/research_source_inventory/research-source-inventory-freeze-report-2026-07-13.json`

**Interfaces:**

- Consumes: public family registry/readiness/adapter factories, foundation adapter map for the PubMed overlay, checked-in fixtures, inventory manifest/ledger schema, authoritative validator report generator.
- Produces: exact `_FAMILY_CONFIGS` boundary table; consumer factory/module tripwires; constructor/readiness/inventory reconciliation; exact implemented-source gate for rows 0026/0027; regenerated canonical digests/report.

- [x] **Step 1: Write boundary and compatibility RED cases**

Replace singular family constants and filename branches with these exact test-only data contracts:

```python
@dataclass(frozen=True, slots=True)
class _FamilyRuntimeCase:
    case_id: str
    source_id: str
    route_identity: tuple[str, str, str, str]
    query: str | GeneralFreeTextQuery | IdentifierLookupQuery | DateIntervalQuery
    result_limit: int
    fixture_files: tuple[str, ...]
    expected_pages: int
    expected_dispatches: int
    expect_candidates: bool = True
    fixture_transform: str | None = None


@dataclass(frozen=True, slots=True)
class _FamilyBoundaryConfig:
    module_name: str
    filename: str
    registry_factory: str
    readiness_factory: str
    adapter_factory: str
    root_modules: frozenset[str]
    local_imports: Mapping[str, frozenset[str]]
    runtime_cases: tuple[_FamilyRuntimeCase, ...]
    include_foundation_adapters: bool = False
```

Rename the current `_EXPECTED_FAMILY_LOCAL_IMPORTS` to `_BIORXIV_MEDRXIV_LOCAL_IMPORTS` without changing a member. Add this exact reviewed import closure for the new root:

```python
_CLINICALTRIALS_PMC_LOCAL_IMPORTS = MappingProxyType({
    "contracts": frozenset({
        "AccessRoute", "BackendDefinition", "BoundedDecimalQueryValuePolicy",
        "CredentialRequirement", "CredentialStatus", "DeferredNumericCSVQueryBinding",
        "DiscoveryOutcomeIdentity", "ExactOrigin", "ExactQueryValuePolicy",
        "ExecutionMode", "LiteralTermsQueryValuePolicy", "OpaqueCursorQueryValuePolicy",
        "MAX_PAGINATION_CURSOR", "OperationKind", "PlannedDispatchGroup", "QueryMode", "ReadinessOverlay",
        "ReadinessState", "RouteKind", "RouteLimits", "RoutePolicy", "RouteReadiness",
        "SourceConstraint", "SourceDefinition", "SourceRouteReference",
    }),
    "executor": frozenset({
        "BoundDispatch", "DiscoveryAdapter", "DiscoveryAdapterError",
        "DiscoveryAdapterResult", "DiscoveryCandidate", "OpaqueCursor",
    }),
    "gateway_adapters": frozenset({
        "MonotonicClock", "_ParseDeadlineExceeded", "_ParseGuard",
        "_ParseLimitExceeded", "_ParsingProfile", "_PayloadInvalid",
        "_base_record", "_canonical_decimal_text", "_checked_response",
        "_execute_ncbi_esearch_summary", "_guarded_items", "_ncbi_json_root",
        "_optional_text", "_raise_adapter_error", "_require_dict", "_require_list",
        "_required_text", "_strict_json", "_TrustedNCBIInputs",
        "_validate_ncbi_message_list",
    }),
    "identity": frozenset({"build_fingerprint", "has_unsafe_url_material", "normalize_doi"}),
    "registry": frozenset({"DiscoveryRegistry", "foundation_readiness", "foundation_registry"}),
})

_CLINICALTRIALS_PMC_IMPORTED_ATTRIBUTE_PATHS = frozenset({
    ".contracts.CredentialRequirement.NONE",
    ".contracts.CredentialStatus.NOT_REQUIRED",
    ".contracts.DiscoveryOutcomeIdentity.from_fingerprint",
    ".contracts.OperationKind.SEARCH",
    ".contracts.QueryMode.GENERAL_FREE_TEXT",
    ".contracts.ReadinessState.READY",
    ".contracts.RouteKind.DIRECT",
    ".contracts.SourceConstraint.NATIVE_CORPUS",
    "datetime.date.fromisoformat",
    "re.ASCII",
    "re.IGNORECASE",
    "re.compile",
    "re.fullmatch",
    "time.monotonic",
    "unicodedata.category",
    "unicodedata.normalize",
})
```

If implementation proves one listed private import unnecessary, remove it from both code and this allowlist; do not broaden the allowlist beyond actual imports. Configure all runtime shapes exactly:

```python
_FAMILY_CONFIGS = MappingProxyType({
    "biorxiv_medrxiv": _FamilyBoundaryConfig(
        module_name="tldw_Server_API.app.core.Research.discovery.biorxiv_medrxiv",
        filename="biorxiv_medrxiv.py",
        registry_factory="biorxiv_medrxiv_shadow_registry",
        readiness_factory="biorxiv_medrxiv_shadow_readiness",
        adapter_factory="biorxiv_medrxiv_gateway_adapters",
        root_modules=frozenset({"biorxiv_medrxiv.py"}),
        local_imports=_BIORXIV_MEDRXIV_LOCAL_IMPORTS,
        runtime_cases=(
            _FamilyRuntimeCase("biorxiv_general", "biorxiv", ("biorxiv_europe_pmc_search_aggregator", "europe_pmc_preprint_v2", "europe-pmc-preprint-v2", "research-discovery-route-policy-v2-biorxiv-medrxiv"), GeneralFreeTextQuery("bounded family discovery"), 100, ("europe_pmc_biorxiv_success.json",), 1, 1),
            _FamilyRuntimeCase("medrxiv_general", "medrxiv", ("medrxiv_europe_pmc_search_aggregator", "europe_pmc_preprint_v2", "europe-pmc-preprint-v2", "research-discovery-route-policy-v2-biorxiv-medrxiv"), GeneralFreeTextQuery("bounded family discovery"), 100, ("europe_pmc_medrxiv_success.json",), 1, 1),
            _FamilyRuntimeCase("biorxiv_lookup", "biorxiv", ("biorxiv_details_lookup_direct", "biorxiv_details_v2", "biorxiv-details-v2", "research-discovery-route-policy-v2-biorxiv-medrxiv"), IdentifierLookupQuery("10.5555/biorxiv.details.synthetic"), 30, ("biorxiv_details_doi_success.json",), 1, 1),
            _FamilyRuntimeCase("medrxiv_lookup", "medrxiv", ("medrxiv_details_lookup_direct", "biorxiv_details_v2", "biorxiv-details-v2", "research-discovery-route-policy-v2-biorxiv-medrxiv"), IdentifierLookupQuery("10.5555/medrxiv.details.synthetic"), 30, ("medrxiv_details_doi_success.json",), 1, 1),
            _FamilyRuntimeCase("biorxiv_interval", "biorxiv", ("biorxiv_details_interval_direct", "biorxiv_details_v2", "biorxiv-details-v2", "research-discovery-route-policy-v2-biorxiv-medrxiv"), DateIntervalQuery("2026-06-01", "2026-06-02", "neuroscience"), 120, ("biorxiv_details_interval_page_1.json", "biorxiv_details_interval_page_2.json"), 2, 2),
            _FamilyRuntimeCase("medrxiv_interval", "medrxiv", ("medrxiv_details_interval_direct", "biorxiv_details_v2", "biorxiv-details-v2", "research-discovery-route-policy-v2-biorxiv-medrxiv"), DateIntervalQuery("2026-06-01", "2026-06-02", "neuroscience"), 120, ("biorxiv_details_interval_page_1.json", "biorxiv_details_interval_page_2.json"), 2, 2, fixture_transform="medrxiv_interval"),
        ),
    ),
    "clinicaltrials_pubmed_central": _FamilyBoundaryConfig(
        module_name="tldw_Server_API.app.core.Research.discovery.clinicaltrials_pubmed_central",
        filename="clinicaltrials_pubmed_central.py",
        registry_factory="clinicaltrials_pubmed_central_shadow_registry",
        readiness_factory="clinicaltrials_pubmed_central_shadow_readiness",
        adapter_factory="clinicaltrials_pubmed_central_gateway_adapters",
        root_modules=frozenset({"clinicaltrials_pubmed_central.py"}),
        local_imports=_CLINICALTRIALS_PMC_LOCAL_IMPORTS,
        include_foundation_adapters=True,
        runtime_cases=(
            _FamilyRuntimeCase("clinicaltrials_nonempty", "clinicaltrials_gov", ("clinicaltrials_gov_studies_search_direct", "clinicaltrials_gov_v2", "clinicaltrials-gov-v2", "research-discovery-route-policy-v2-clinicaltrials-pmc"), GeneralFreeTextQuery("bounded family discovery"), 100, ("clinicaltrials_success_page_1.json", "clinicaltrials_success_page_2.json"), 2, 2),
            _FamilyRuntimeCase("pmc_nonempty", "pubmed_central", ("pubmed_central_esearch_summary_direct", "pubmed_central_v2", "pubmed-central-v2", "research-discovery-route-policy-v2-clinicaltrials-pmc"), GeneralFreeTextQuery("bounded family discovery"), 100, ("pmc_esearch_success.json", "pmc_esummary_success.json"), 1, 2),
            _FamilyRuntimeCase("pmc_empty", "pubmed_central", ("pubmed_central_esearch_summary_direct", "pubmed_central_v2", "pubmed-central-v2", "research-discovery-route-policy-v2-clinicaltrials-pmc"), GeneralFreeTextQuery("bounded absent discovery"), 100, ("pmc_esearch_empty.json",), 1, 1, expect_candidates=False),
            _FamilyRuntimeCase("pubmed_identity_nonempty", "pubmed", ("pubmed_ncbi_eutils_pubmed_direct", "pubmed_v2", "pubmed-v2-ncbi-identity", "research-discovery-route-policy-v2-foundation-pubmed-ncbi-identity-2026-08-21"), "bounded family discovery", 100, ("pubmed_esearch_success.json", "pubmed_esummary_success.json"), 1, 2),
        ),
    ),
})
```

Do not subtract foundation route IDs when discovering changed identities because the overlay deliberately shares one. Combine `foundation_gateway_adapters()` with the two-entry family map and reject duplicate adapter IDs.

Add `clinicaltrials_pubmed_central` to forbidden-loaded-module checks and monkeypatch all three public factories in Standalone/Deep Research compatibility tests. Keep socket, HTTP-client, browser, cookie, AuthNZ, config, DB, Media, OA, and Third_Party tripwires active during runtime fixture execution.

Derive compilation ceilings from the sealed route rather than the observed fixture length. The existing interval routes allow `4/4` but their fixtures terminate at observed `2/2`, so `_family_plan(config, case)` must use this exact split:

```python
route = registry.get_route(case.route_identity[0])
budget = BudgetCeilings(
    max_route_attempts=1,
    max_physical_dispatches=route.max_physical_dispatches,
    max_pages_per_route=route.policy.limits.max_pages,
    max_redirects=0,
    max_retries=0,
    max_wall_time_ms=route.policy.limits.timeout_ms * route.max_physical_dispatches,
    max_results=case.result_limit,
)
```

Use `case.expected_pages` and `case.expected_dispatches` only for post-execution usage/accounting assertions. Never lower a sealed planning allowance to match how early a fixture terminates.

Witness the boundary/reconciliation RED before parameterizing the singular harness:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest -q \
  tldw_Server_API/tests/Research/test_research_discovery_network_boundary.py \
  tldw_Server_API/tests/Research/test_research_discovery_v2_compatibility.py \
  tldw_Server_API/tests/Research/test_research_discovery_registry_reconciliation.py
```

Expected: the new config-driven family closure/runtime and consumer-denial cases fail against the still-singular harness.

- [x] **Step 2: Implement exact per-family closure configuration and pin digests**

Parameterize `_family_module`, `_family_plan`, fixture loading/transform, route/readiness/factory equality, scanner mutation cases, unrecorded-ready-route checks, and runtime execution over `_FAMILY_CONFIGS`. For bioRxiv/medRxiv, assert the six route identities, queries, fixture transforms, paths, pages, physical records, candidates, and accounting remain exactly equal to the pre-refactor assertions. For the new family, compare full `(route_id, adapter_id, adapter_version, policy_version)` identities so the same-ID PubMed overlay is not subtracted as foundation.

Do not reuse the old family-only profile equality unchanged for the new shadow. Split ownership and then prove combined coverage exactly:

```python
family_profile_identities = set(family_module._FAMILY_PARSING_PROFILES)
assert family_profile_identities == {
    ("clinicaltrials_gov_v2", "clinicaltrials-gov-v2"),
    ("pubmed_central_v2", "pubmed-central-v2"),
}
family_callable_ids = set(
    family_module.clinicaltrials_pubmed_central_gateway_adapters()
)
assert family_callable_ids == {"clinicaltrials_gov_v2", "pubmed_central_v2"}

overlay_profile_identity = ("pubmed_v2", "pubmed-v2-ncbi-identity")
shared_module = _adapter_module()
assert overlay_profile_identity in shared_module._PARSING_PROFILES
assert "pubmed_v2" in shared_module.foundation_gateway_adapters()

changed_ready_adapter_identities = {
    (route.adapter_id, route.adapter_version)
    for route in changed_ready_routes
}
assert changed_ready_adapter_identities == family_profile_identities | {
    overlay_profile_identity
}
assert {adapter_id for adapter_id, _version in changed_ready_adapter_identities} == (
    family_callable_ids | {"pubmed_v2"}
)
```

Here `changed_ready_routes` is the exact three-route full-identity delta: ClinicalTrials.gov, PMC, and the same-route-ID PubMed overlay. The overlay callable/profile remain shared-owned; never insert `pubmed_v2` into the two-entry family map merely to satisfy a test equality.

Reconciliation must preserve the deliberate two-level attribution vocabulary. Assert runtime routes exactly as `native_nct_record` and `ncbi_pmc_database`, while their `native_corpus` inventory candidates remain the schema-required generic `native_response`. Do not make those strings equal and do not widen `ATTRIBUTION_BASES`; this mirrors the existing explicit runtime `provider_publisher` → inventory `provider_source_field` reconciliation.

For each provider-family root, pin raw/import/semantic AST digests. For changed shared modules, pin only import/semantic AST digests, including `Security/http_hop.py`; do not add raw shared-module digests. Add `_EXPECTED_IMPORTED_ATTRIBUTE_PATHS["clinicaltrials_pubmed_central.py"] = _CLINICALTRIALS_PMC_IMPORTED_ATTRIBUTE_PATHS`; the scanner must use `.get(filename, set())` only for explicitly configured bootstrap modules, never to silently permit a family root. `_EXPECTED_GATEWAY_IMPORTS` and `_EXPECTED_HTTP_HOP_IMPORTS` give the new family an empty set; `_EXPECTED_IDENTITY_IMPORTS` gives it exactly `build_fingerprint`, `has_unsafe_url_material`, and `normalize_doi`. Enforce that only `gateway.py` imports/uses the HTTP hop.

- [x] **Step 3: Write authoritative inventory RED cases**

Add a separate exact `REQUIRED_IMPLEMENTED_SOURCES` gate, rather than weakening the existing bioRxiv/medRxiv three-route contract. Pin rows:

```javascript
const REQUIRED_IMPLEMENTED_SOURCES = Object.freeze({
  "sourclip-2026-07-13-0026": Object.freeze({
    sourceSnapshotSha256: "cbc4a8445252460ef4502924edf409c7fc8098eb6987745b83cc426bd2fc8e73",
    canonicalTarget: "clinicaltrials_gov",
    declaredSurfaces: Object.freeze(["standalone_search", "deep_research"]),
    capabilities: Object.freeze(["search", "detail", "metadata", "snippet"]),
    route: Object.freeze({
      id: "clinicaltrials_gov_studies_search_direct",
      routeKind: "direct",
      backendId: "clinicaltrials_gov_api_v2",
      queryModes: Object.freeze(["general_free_text"]),
      sourceConstraint: "native_corpus",
      sourcePredicate: null,
      attributionBasis: "native_response",
      evidenceHosts: Object.freeze(["clinicaltrials.gov"]),
    }),
    implementationState: "implemented",
    fixtureState: "passed",
    liveState: "not_run",
    certifications: Object.freeze([]),
  }),
  "sourclip-2026-07-13-0027": Object.freeze({
    sourceSnapshotSha256: "34d7fc36d4b64b2dca99c0472ad3d804c7ed9ff5a96574a8146947133913b32b",
    canonicalTarget: "pubmed_central",
    declaredSurfaces: Object.freeze(["standalone_search", "deep_research"]),
    capabilities: Object.freeze(["search", "detail", "metadata"]),
    route: Object.freeze({
      id: "pubmed_central_esearch_summary_direct",
      routeKind: "direct",
      backendId: "ncbi_eutils_pmc",
      queryModes: Object.freeze(["general_free_text"]),
      sourceConstraint: "native_corpus",
      sourcePredicate: null,
      attributionBasis: "native_response",
      evidenceHosts: Object.freeze(["www.ncbi.nlm.nih.gov"]),
    }),
    implementationState: "implemented",
    fixtureState: "passed",
    liveState: "not_run",
    certifications: Object.freeze([]),
  }),
});
```

Add `requiredImplementedSources = REQUIRED_IMPLEMENTED_SOURCES` to `validateInventoryDocuments(...)`. For each entry, find exactly one route satisfying `routeMatchesRequirement(rowCandidate, requirement.route)`, then require: resolution `mapped`; exact snapshot digest; exact one-element canonical target; exact declared surfaces/capabilities arrays; `implemented/passed/not_run`; exact empty certifications; an `implementation` evidence entry; and substantive triage. Emit state and blocker fields exactly:

```javascript
required_implemented_sources: requiredImplementedSourceStates,
required_implemented_source_blockers: requiredImplementedSourceBlockers,
```

Add `requiredImplementedSourceBlockers.length === 0` to `contractFreezeReady`. The contract gate must therefore fail when either implemented row drifts even though the generic schema and the original three-route `required_sources` gate pass. Add a positive report test and one mutation test for snapshot, target, surfaces, capabilities, every route field, each state, certifications, evidence, and report/blocker integration.

Update every unrelated synthetic/minimal `validateInventoryDocuments(...)` test option that currently neutralizes the original gate with `requiredSources: {}` to also pass `requiredImplementedSources: {}`. Those tests must retain their pre-existing isolated contract/report expectations without needing rows 0026/0027. Only the full frozen-document/default-gate tests and the dedicated implemented-source positive/mutation matrix omit that override and exercise `REQUIRED_IMPLEMENTED_SOURCES`.

Witness the inventory RED before changing the validator or ledger:

```bash
node --test Helper_Scripts/tests/validate_research_source_inventory.test.mjs
```

Expected: the new implemented-source state/blocker/report-gate assertions fail because the authoritative validator does not yet expose or enforce them.

- [x] **Step 4: Migrate only rows 0026/0027 and regenerate authoritative artifacts**

Update the ledger evidence honestly:

- ClinicalTrials.gov: bounded modified metadata projection, fixture-only adapter, no retained provider link/markup, no live/surface certification.
- PMC: metadata-only ESearch/ESummary, no abstract/snippet/EFetch/full text, fixture-only adapter, no live/surface certification.

Expected summary counts after migration:

```text
implemented=4
planned=231
fixture passed=4
fixture not_run=231
live not_run=235
inventory_delivery_ready=false
```

Remove `snippet` from row 0027's source-capability claim because this route deliberately returns metadata only. Preserve both exact source snapshot digests. After applying the two row edits, mechanically recompute canonical `rows_sha256` using the validator's exported helpers:

```bash
node --input-type=module -e 'import fs from "node:fs"; import { canonicalJson, sha256 } from "./Helper_Scripts/validate_research_source_inventory.mjs"; const file = "Docs/Design/research_source_inventory/research-source-coverage-ledger-2026-07-13.json"; const ledger = JSON.parse(fs.readFileSync(file, "utf8")); ledger.rows_sha256 = sha256(canonicalJson(ledger.rows)); fs.writeFileSync(file, `${JSON.stringify(ledger, null, 2)}\n`);'
```

Generate the report authoritatively, install that mechanical output, regenerate independently, and require byte identity:

```bash
node Helper_Scripts/validate_research_source_inventory.mjs \
  --root . --gate contract --json --as-of 2026-07-15 \
  --trusted-reviewer codex-task-12968.1-source-triage \
  --trusted-reviewer codex-task-12968.5-inventory-review \
  > /tmp/task-12968-6-inventory-report.json
cp /tmp/task-12968-6-inventory-report.json \
  Docs/Design/research_source_inventory/research-source-inventory-freeze-report-2026-07-13.json
node Helper_Scripts/validate_research_source_inventory.mjs \
  --root . --gate contract --json --as-of 2026-07-15 \
  --trusted-reviewer codex-task-12968.1-source-triage \
  --trusted-reviewer codex-task-12968.5-inventory-review \
  > /tmp/task-12968-6-inventory-report-second.json
cmp \
  Docs/Design/research_source_inventory/research-source-inventory-freeze-report-2026-07-13.json \
  /tmp/task-12968-6-inventory-report-second.json
```

Never hand-edit generated report fields.

- [x] **Step 5: Run inventory, schema, registry, compatibility, and boundary GREEN gates**

```bash
node --test Helper_Scripts/tests/validate_research_source_inventory.test.mjs
node Helper_Scripts/validate_research_source_inventory.mjs \
  --root . \
  --gate contract \
  --as-of 2026-07-15 \
  --trusted-reviewer codex-task-12968.1-source-triage \
  --trusted-reviewer codex-task-12968.5-inventory-review
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest -q \
  Helper_Scripts/tests/test_research_source_inventory_schema.py \
  tldw_Server_API/tests/Research/test_research_discovery_network_boundary.py \
  tldw_Server_API/tests/Research/test_research_discovery_registry_reconciliation.py \
  tldw_Server_API/tests/Research/test_research_discovery_v2_compatibility.py
```

Expected: all tests/gates pass; generated report remains `inventory_delivery_ready=false` because consumer/live certification is intentionally absent.

- [x] **Step 6: Commit boundary and evidence reconciliation**

```bash
git add \
  tldw_Server_API/tests/Research/test_research_discovery_network_boundary.py \
  tldw_Server_API/tests/Research/test_research_discovery_v2_compatibility.py \
  tldw_Server_API/tests/Research/test_research_discovery_registry_reconciliation.py \
  Helper_Scripts/validate_research_source_inventory.mjs \
  Helper_Scripts/tests/validate_research_source_inventory.test.mjs \
  Docs/Design/research_source_inventory/research-source-coverage-ledger-2026-07-13.json \
  Docs/Design/research_source_inventory/research-source-inventory-freeze-report-2026-07-13.json
git commit -m "test(research): seal clinicaltrials PMC shadow evidence (TASK-12968.6)"
```

---

## Stage 5: Verification, Review, and Handoff

**Goal:** Prove the complete provider-only change is safe, compatible, reviewable, and correctly tracked before any consumer work starts.

**Success Criteria:** All focused/full tests, formatting, Python 3.10 compilation, Bandit, inventory digests, and diff checks pass; independent spec/code/final reviews have no unresolved Critical/Important findings; Backlog notes contain exact evidence and commit/PR links.

**Tests:** Full Research suite, touched Security transport tests, Node validator suite, schema/contract gate, compile, Ruff, Black, Bandit, and `git diff --check`.

**Status:** In Progress

### Task 9: Run final gates, independent reviews, tracking, and PR handoff

**Files:**

- Modify through Backlog CLI: `backlog/tasks/task-12968.6 - Add-ClinicalTrials.gov-and-PubMed-Central-shared-discovery-route-family.md`
- Modify only if review finds an issue: files already listed in Tasks 1–8
- Retain unchanged as linked execution evidence: `Docs/superpowers/plans/2026-08-21-clinicaltrials-pubmed-central-shared-discovery-route-family-implementation-plan.md`.

**Interfaces:**

- Consumes: completed Tasks 1–8 and their focused RED/GREEN evidence.
- Produces: clean verification evidence, no unresolved high-priority review findings, updated TASK-12968.6 tracking, final commits, and a provider-only PR ready for the repository's human-written Change summary merge gate.

- [x] **Step 1: Run the complete functional test matrix**

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest -q tldw_Server_API/tests/Research/test_research_discovery*.py
python -m pytest -q \
  tldw_Server_API/tests/Security/test_http_hop_contract.py \
  tldw_Server_API/tests/Security/test_http_hop_streaming.py \
  tldw_Server_API/tests/Security/test_http_hop_transport.py
node --test Helper_Scripts/tests/validate_research_source_inventory.test.mjs
python -m pytest -q Helper_Scripts/tests/test_research_source_inventory_schema.py
node Helper_Scripts/validate_research_source_inventory.mjs \
  --root . \
  --gate contract \
  --as-of 2026-07-15 \
  --trusted-reviewer codex-task-12968.1-source-triage \
  --trusted-reviewer codex-task-12968.5-inventory-review
```

Record exact pass counts and elapsed times in TASK-12968.6. Do not describe `inventory_delivery_ready=false` as a failure; it is the required shadow-only state.

- [x] **Step 2: Run syntax, format, static, security, and diff gates**

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m compileall -q \
  tldw_Server_API/app/core/Research/discovery \
  tldw_Server_API/app/core/Security/http_hop.py
/Users/macbook-dev/.local/bin/python3.10 -m py_compile \
  tldw_Server_API/app/core/Research/discovery/contracts.py \
  tldw_Server_API/app/core/Research/discovery/planner.py \
  tldw_Server_API/app/core/Research/discovery/gateway.py \
  tldw_Server_API/app/core/Research/discovery/executor.py \
  tldw_Server_API/app/core/Research/discovery/gateway_adapters.py \
  tldw_Server_API/app/core/Research/discovery/clinicaltrials_pubmed_central.py \
  tldw_Server_API/app/core/Security/http_hop.py
python -m ruff check \
  tldw_Server_API/app/core/Research/discovery \
  tldw_Server_API/app/core/Security/http_hop.py \
  tldw_Server_API/tests/Research/test_research_discovery*.py \
  tldw_Server_API/tests/Security/test_http_hop_contract.py \
  Helper_Scripts/tests/test_research_source_inventory_schema.py
python -m black --check \
  tldw_Server_API/app/core/Research/discovery \
  tldw_Server_API/app/core/Security/http_hop.py \
  tldw_Server_API/tests/Research/test_research_discovery*.py \
  tldw_Server_API/tests/Security/test_http_hop_contract.py \
  Helper_Scripts/tests/test_research_source_inventory_schema.py
node --check Helper_Scripts/validate_research_source_inventory.mjs
node --check Helper_Scripts/tests/validate_research_source_inventory.test.mjs
python -m bandit -r \
  tldw_Server_API/app/core/Research/discovery/contracts.py \
  tldw_Server_API/app/core/Research/discovery/planner.py \
  tldw_Server_API/app/core/Research/discovery/gateway.py \
  tldw_Server_API/app/core/Research/discovery/executor.py \
  tldw_Server_API/app/core/Research/discovery/gateway_adapters.py \
  tldw_Server_API/app/core/Research/discovery/clinicaltrials_pubmed_central.py \
  tldw_Server_API/app/core/Security/http_hop.py \
  -f json -o /tmp/bandit_task_12968_6.json
git diff --check
```

Inspect `/tmp/bandit_task_12968_6.json`; fix every new finding in touched production code before proceeding.

- [x] **Step 3: Perform three independent review gates**

Dispatch fresh reviewers with exact scopes:

1. spec compliance: compare implementation against every design section and inventory row;
2. code/security quality: inspect parser bounds, attribution, NCBI identity, repr redaction, cancellation, and physical accounting;
3. final diff review: inspect `origin/dev...HEAD`, including tracking/generated artifacts and absence of consumer wiring.

Resolve every Critical or Important finding with a new witnessed RED/GREEN cycle. Re-run affected focused tests, then repeat Steps 1–2 after the final fix.

If a review required a repository change, commit the reviewed feature scope before the tracking-only commit:

```bash
git add \
  tldw_Server_API/app/core/Research/discovery/contracts.py \
  tldw_Server_API/app/core/Research/discovery/planner.py \
  tldw_Server_API/app/core/Research/discovery/gateway.py \
  tldw_Server_API/app/core/Research/discovery/executor.py \
  tldw_Server_API/app/core/Research/discovery/gateway_adapters.py \
  tldw_Server_API/app/core/Research/discovery/clinicaltrials_pubmed_central.py \
  tldw_Server_API/app/core/Security/http_hop.py \
  tldw_Server_API/tests/Research/test_research_discovery_contracts.py \
  tldw_Server_API/tests/Research/test_research_discovery_planner.py \
  tldw_Server_API/tests/Research/test_research_discovery_gateway.py \
  tldw_Server_API/tests/Research/test_research_discovery_gateway_adapters.py \
  tldw_Server_API/tests/Research/test_research_discovery_executor.py \
  tldw_Server_API/tests/Research/test_research_discovery_pubmed_gateway_adapter.py \
  tldw_Server_API/tests/Research/test_research_discovery_clinicaltrials_pubmed_central.py \
  tldw_Server_API/tests/Research/test_research_discovery_network_boundary.py \
  tldw_Server_API/tests/Research/test_research_discovery_registry_reconciliation.py \
  tldw_Server_API/tests/Research/test_research_discovery_v2_compatibility.py \
  tldw_Server_API/tests/Security/test_http_hop_contract.py \
  tldw_Server_API/tests/fixtures/research_discovery_gateway_adapters/clinicaltrials_success_page_1.json \
  tldw_Server_API/tests/fixtures/research_discovery_gateway_adapters/clinicaltrials_success_page_2.json \
  tldw_Server_API/tests/fixtures/research_discovery_gateway_adapters/clinicaltrials_empty.json \
  tldw_Server_API/tests/fixtures/research_discovery_gateway_adapters/pmc_esearch_success.json \
  tldw_Server_API/tests/fixtures/research_discovery_gateway_adapters/pmc_esearch_empty.json \
  tldw_Server_API/tests/fixtures/research_discovery_gateway_adapters/pmc_esummary_success.json \
  Helper_Scripts/validate_research_source_inventory.mjs \
  Helper_Scripts/tests/validate_research_source_inventory.test.mjs \
  Docs/Design/research_source_inventory/research-source-coverage-ledger-2026-07-13.json \
  Docs/Design/research_source_inventory/research-source-inventory-freeze-report-2026-07-13.json
git diff --cached --check
git commit -m "fix(research): address final clinicaltrials PMC review (TASK-12968.6)"
```

If no review required a repository change, skip this fix commit. In either case, `git status --short` must show only the pending Backlog tracking update before Step 4.

- [ ] **Step 4: Open a provider-only draft PR, then update Backlog through the official CLI**

Push the fully reviewed code/evidence commits first and open a draft PR so its real URL exists before the tracking-only commit:

```bash
git push -u origin codex/task-12968-6-clinicaltrials-pmc
gh pr create \
  --draft \
  --base dev \
  --head codex/task-12968-6-clinicaltrials-pmc \
  --title "feat(research): add ClinicalTrials.gov and PMC shadow discovery" \
  --body "Provider-only TASK-12968.6 draft. Verification and implementation rationale are linked from the Backlog task and approved design. Merge remains blocked until the human requester supplies the repository-required Change summary."
gh pr view --json url --jq .url
```

Keep the PR draft. Do not ask the CLI-generated body to stand in for the required human-written Change summary.

Use `backlog task 12968.6 --plain` to confirm the unique task, then append:

- design and plan paths;
- exact base revision and isolated branch/worktree;
- per-task commit IDs;
- focused/full test counts;
- Node validator/schema/digest results;
- Python 3.10, Ruff, Black, Bandit, and diff-check results;
- review outcomes and resolved findings;
- explicit statements: shadow-only, fixture-only PMC, no consumer wiring, no browser/cookies/credentials, no live NCBI calls;
- PR URL when created.

Do not manually edit the task file unless the official CLI is unavailable and the user explicitly approves the exception.

- [ ] **Step 5: Commit and push the final tracking-only update**

```bash
git status --short
git diff --stat origin/dev...HEAD
git diff --check
git add "backlog/tasks/task-12968.6 - Add-ClinicalTrials.gov-and-PubMed-Central-shared-discovery-route-family.md"
git commit -m "docs(research): finalize clinicaltrials PMC evidence (TASK-12968.6)"
git push
```

Confirm the draft PR now contains the tracking commit. Do not mark it ready or merge until the human requester supplies the required human-written `Change summary` explaining what changed and why these implementation choices were made.

## Final Self-Review Checklist

- [x] Every requirement in the approved design maps to one of Tasks 1–9.
- [x] Every instruction names its exact behavior, error outcome, and interface.
- [x] `OpaqueCursorQueryValuePolicy`, `OpaqueCursor`, NCBI helper callbacks, family factories, adapter IDs/versions, and inventory identities use one exact spelling everywhere.
- [x] Foundation plan/digest/version locks are explicit and unchanged.
- [x] PubMed overlay readiness is backed by a real fixture runtime path, not route-ID inheritance.
- [x] ClinicalTrials.gov fixtures are wholly synthetic and PMC verification makes no live call.
- [x] Network-boundary tests cover ClinicalTrials.gov `2/2`, PMC nonempty `1/2`, PMC empty `1/1`, and PubMed overlay `1/2`.
- [x] Inventory states stop at `implemented/passed/not_run`, certifications remain empty, and `inventory_delivery_ready=false`.
- [x] Production Search and Deep Research import no family symbol and issue zero family requests.
- [x] TASK-12968.3 remains the owner of pacing, registration proof, notices/currentness, long-query routing, and Standalone cutover; TASK-12968.4 remains consumer-only and waits for TASK-13014.
