# bioRxiv and medRxiv Shared-Discovery Route Family Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver TASK-12968.5's credentialless, shadow-only bioRxiv/medRxiv route family with truthful Europe PMC general discovery, typed official-details operations, strict gateway enforcement, fixture execution, and no production Search or Deep Research cutover.

**Architecture:** Extend the frozen discovery contracts only where the new request shapes require it: digest-bound query-value rules, one closed dynamic-path template, typed planner inputs, and executor-owned path pagination. Put the two catalog targets, six routes, two backends, two adapters, shadow registry, readiness overlay, and adapter composition in one new `biorxiv_medrxiv.py` family module. Reuse the existing parser, failure, identity, dispatch, attribution, and accounting boundaries; keep `foundation_registry()`, `foundation_gateway_adapters()`, legacy provider helpers, and current consumers unchanged.

**Tech Stack:** Python 3.10+, frozen standard-library dataclasses and enums, `datetime`, `re`, `unicodedata`, `urllib.parse`, existing Research discovery contracts/gateway/executor, pytest, sanitized JSON fixtures, Node inventory validator, Ruff, Black, Bandit, and Git diff hygiene.

---

## Global Constraints

- Work only in `/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/research-source-catalog-deep-research`; never edit the dirty root worktree.
- TASK-12968.5 is the Backlog.md unit for every file in this plan. Keep the task In Progress and link this plan before production edits.
- Use `superpowers:test-driven-development` for every behavior change: add one focused failing test, confirm the intended RED reason, make the smallest implementation pass, then refactor with the focused suite green.
- Keep all new behavior shadow-only. Do not import the family module from V1 Search, `service.py`, Deep Research, endpoints, jobs, or any production startup path.
- Do not change `foundation_registry()`, `foundation_readiness()`, or `foundation_gateway_adapters()` outputs. Existing raw-string foundation plans and foundation route-policy digests must remain byte-for-byte exact.
- Do not import or call `tldw_Server_API/app/core/Third_Party/BioRxiv.py`. Do not add browser automation, cookies, credentials, HTML site search, Cloudflare bypasses, Media ingestion, JATS/full-text/PDF retrieval, result-link dereference, or scheduled indexing.
- Every family request goes through the existing executor-owned `BoundDispatch` and `gateway.dispatch_once`; all six routes use `max_redirects=0`.
- Reuse the private bounded-parser helpers in `gateway_adapters.py` from the family module. Do not duplicate JSON parsing, status/rate-limit mapping, parser guards, candidate identity, or generic dispatch-error layers merely to make them public.
- Keep Europe PMC to one bounded first page. Omit `cursorMark`, ignore returned cursor/next-page fields, and never describe this as exhaustive historical export.
- Keep recent-count/day details shortcuts disabled. Only explicit DOI lookup and explicit date interval/category paths are executable.
- Store only distinct valid provider shapes as fixtures. Derive malformed, attribution, injection, timeout, cancellation, conflict, redirect, and rate-limit cases by fixture mutation or scripted gateway responses.
- Live and surface certification remain `not_run`; TASK-12968.3 owns Standalone Search cutover and TASK-12968.4 owns Deep Research cutover.

## Frozen Bounds

These are reviewed contract values, not implementation-time guesses. Constructor, gateway, parser, and exhaustion tests must pin them:

- General query: at most 16 Unicode alphanumeric terms, at most 64 characters per term.
- Category: at most 128 canonical characters from the approved bounded-text grammar.
- DOI path: registrant `10.` plus 4-9 digits; one suffix of at most 128 decoded characters.
- Date interval: two real canonical ISO dates, inclusive, at most 366 days.
- Normalized metadata: title 4,096 characters; abstract/snippet 65,536; at most 1,024 authors of at most 512 characters each; provider/record identifiers 128 characters.
- Family JSON profile: 2,097,152 input bytes; 120 records; depth 16; 50,000 nodes; 65,536 characters per JSON string; 32 characters per numeric token; 500 ms cooperative parse deadline.
- Europe PMC routes: `max_pages=1`, `max_redirects=0`, `max_retries=0`, `timeout_ms=20_000`, `max_response_bytes=2_097_152`, `max_results=100`, `max_physical_dispatches=1`.
- Details DOI routes: `max_pages=1`, `max_redirects=0`, `max_retries=0`, `timeout_ms=20_000`, `max_response_bytes=2_097_152`, `max_results=30`, `max_physical_dispatches=1`.
- Details interval/category routes: `max_pages=4`, `max_redirects=0`, `max_retries=0`, `timeout_ms=20_000`, `max_response_bytes=2_097_152`, `max_results=120`, `max_physical_dispatches=4`.

Do not silently raise these values. Any future expansion requires a policy-version/digest change and fresh boundary tests.

## Shared Gate Before Every Commit

Before each stage commit:

- [ ] Run all tests added in the current stage plus the earlier focused stages.
- [ ] Run `python -m compileall -q` on every touched Python source/test path.
- [ ] Run exact-scope `ruff check` and `black --check` on touched Python files.
- [ ] Run `git diff --check` and inspect `git status --short` for unrelated changes.
- [ ] Run Bandit on touched production Python before any commit that claims a security boundary complete.
- [ ] Ask an independent reviewer to inspect the stage diff; resolve all Critical/Important findings and re-run the affected tests.

## Task 1 / Stage 1: Add Digest-Bound Query and Dynamic-Path Policy Primitives

**Goal:** Represent and independently enforce the exact Europe PMC query values and official-details path shapes without widening existing foundation policy.

**Success Criteria:** Family policies can freeze exact, bounded-decimal, literal-term-plus-suffix, and bounded-text query values plus one closed path template; the gateway rejects forged values and non-canonical paths; omitted additive fields preserve every foundation digest.

**Tests:** Constructor/digest tests in `test_research_discovery_contracts.py`, gateway request-rejection tests in `test_research_discovery_gateway.py`, and exact foundation compatibility in `test_research_discovery_v2_compatibility.py`.

**Status:** Complete

### Files

- Modify `tldw_Server_API/app/core/Research/discovery/contracts.py`.
- Modify `tldw_Server_API/app/core/Research/discovery/gateway.py`.
- Modify `tldw_Server_API/tests/Research/test_research_discovery_contracts.py`.
- Modify `tldw_Server_API/tests/Research/test_research_discovery_gateway.py`.
- Modify `tldw_Server_API/tests/Research/test_research_discovery_v2_compatibility.py` only for additive compatibility assertions.

### Contract shape

Use closed typed rules rather than a regex/stringly policy bag. Equivalent naming is acceptable only if the behavior and digest projection stay exact:

```python
class PathSlotKind(str, Enum):
    DATE = "date"
    UINT = "uint"
    DOI_REGISTRANT = "doi_registrant"
    DOI_SUFFIX = "doi_suffix"


@dataclass(frozen=True, slots=True)
class PathSlot:
    kind: PathSlotKind
    max_chars: int


@dataclass(frozen=True, slots=True)
class PathTemplate:
    segments: tuple[str | PathSlot, ...]
    pagination_segment_index: int | None = None


@dataclass(frozen=True, slots=True)
class ExactQueryValuePolicy:
    name: str
    value: str
    required: bool = True


@dataclass(frozen=True, slots=True)
class BoundedDecimalQueryValuePolicy:
    name: str
    maximum: int
    required: bool = True


@dataclass(frozen=True, slots=True)
class LiteralTermsQueryValuePolicy:
    name: str
    fixed_suffix: str
    max_terms: int
    max_term_chars: int
    required: bool = True


@dataclass(frozen=True, slots=True)
class BoundedTextQueryValuePolicy:
    name: str
    max_chars: int
    required: bool = False


QueryValuePolicy = (
    ExactQueryValuePolicy
    | BoundedDecimalQueryValuePolicy
    | LiteralTermsQueryValuePolicy
    | BoundedTextQueryValuePolicy
)
```

Add optional `path_template: PathTemplate | None = None` and `query_value_policies: tuple[QueryValuePolicy, ...] = ()` to `RoutePolicy`.

### Test-first steps

- [ ] Write RED constructor tests proving path slots accept only exact enum instances and positive bounds; template literals are visible ASCII, non-empty, slash-free segments; one template has an exact segment count; and `pagination_segment_index` points to a `UINT` slot.
- [ ] Write RED tests proving `RoutePolicy` accepts exactly one path channel: non-empty `paths` or one `path_template`, never both or neither. Adjust `AccessRoute`'s required physical-dispatch calculation so one template counts as one initial intent and subsequent pages still count normally.
- [ ] Treat a template's `pagination_segment_index` as the path pagination channel and prove query, JSON-body, and path pagination channels are pairwise mutually exclusive at contract construction.
- [ ] Write RED tests proving query-value policy names are unique, are a complete set of `allowed_query_keys` whenever the new tuple is non-empty, and have strict exact runtime types. An optional `category` policy remains declared but may be absent from one intent.
- [ ] Write RED digest tests proving template literals, ordered slot kind/bounds, pagination index, and each query rule participate in `canonical_policy_digest`; changing any one changes the digest.
- [ ] Pin the current foundation policy digests before implementation and assert they remain exact. Recompute the complete current all-foundation raw-string baseline from this exact fixture and assert plan digest `2e9869bc7ed6b51fe8ffe823dff2e392933e275b54bc2997e8542ba89829f403`, `sha256(canonical_plan_bytes)=991d3a67132058625bdfef00836240cacc6b91370510c9b9d0762181310c9d46`, and byte length `10936`:

```python
registry = foundation_registry()
request = PlanningRequest(
    source_ids=(
        "openalex",
        "semantic_scholar",
        "crossref",
        "arxiv",
        "pubmed",
        "zenodo",
        "figshare",
        "osf",
    ),
    query="  Causal   Inference  ",
    filters=(),
    result_limit=25,
)
readiness = foundation_readiness(ExecutionMode.OFFLINE_FIXTURE)
budget = BudgetCeilings(16, 20, 1, 0, 0, 500_000, 100)
plan = compile_discovery_plan(
    request,
    registry=registry,
    readiness=readiness,
    budget=budget,
)
```
- [ ] Implement the smallest validated dataclasses and conditional digest projection. Add new digest keys only when `path_template` is not `None` or `query_value_policies` is non-empty.
- [ ] Write RED gateway tests for missing/changed `format=json`, changed `resultType=core`, non-canonical/negative/oversized decimal values, missing fixed suffix, injected Europe PMC operators, too many/long terms, and invalid category text.
- [ ] Implement one query-policy validator in `gateway.py`. For literal terms, require `"term" AND "term"` where every inner term is canonical NFKC Unicode alphanumeric text within the frozen bounds and the whole value ends in the exact immutable suffix. For bounded text, require canonical NFKC text, single internal spaces, no leading/trailing whitespace, and only alphanumeric characters plus space, hyphen, ampersand, and slash.
- [ ] Write RED gateway path tests for malformed `%` escapes, non-ASCII raw paths, wrong segment count/literals, empty segments, encoded slash/backslash/percent, controls, dot segments, double encoding, invalid/reversed/over-366-day dates, overflow/non-canonical unsigned cursors, malformed registrants, unsafe DOI suffixes, and non-canonical percent encoding.
- [ ] Implement one canonical path validator: split the raw ASCII path before decoding; strict-decode dynamic segments once; reject `/`, `\\`, `%`, controls, empty/dot segments, and a second encoded form; full-match the slot grammar; validate the first two date slots as an inclusive interval of at most 366 days; render dynamic segments with one canonical RFC 3986 segment encoder; and require byte-for-byte equality with `DispatchIntent.path`.
- [ ] Route `_snapshot_binding(...)` through both new validators before policy activity or `one_hop`. Preserve exact-path behavior for foundation routes.
- [ ] Add mutation tests using `object.__setattr__`/`dataclasses.replace` to prove a self-consistent forged intent/policy cannot bypass value or path enforcement.

### Focused RED/GREEN command

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest -q \
  tldw_Server_API/tests/Research/test_research_discovery_contracts.py \
  tldw_Server_API/tests/Research/test_research_discovery_gateway.py \
  tldw_Server_API/tests/Research/test_research_discovery_v2_compatibility.py
```

### Commit

`feat(research): add digest-bound discovery request policies`

### Completion evidence

- Implementation: `9f33dc647d2cde850d84b9f37881b69bbd06e65e`
- Review hardening: `584075be1cb427b24db19d56380ae76f9517d802`
- Focused final suite: 338 passed with four pre-existing warnings.
- Exact foundation plan digest, canonical-byte SHA-256, length, and all eight foundation policy digests remained pinned.
- Compileall, Ruff, Black, diff hygiene, and Bandit passed; Bandit reported zero findings.
- Independent specification review and post-fix code-quality review both approved Stage 1 with no open Critical or Important findings.

## Task 2 / Stage 2: Add Typed Planner Selection and Accounted Path Pagination

**Goal:** Compile exact typed general/DOI/interval requests into only compatible routes and let the sealed executor advance a validated details cursor in the path.

**Success Criteria:** Typed values select the correct mode; incompatible routes yield `query_mode_not_supported`; raw-string foundation output remains exact; all cursor mutations pass through normal gateway/accounting checks.

**Tests:** Typed planner behavior in `test_research_discovery_planner.py`, cursor/accounting behavior in `test_research_discovery_executor.py`, and foundation-byte compatibility.

**Status:** Complete

### Files

- Modify `tldw_Server_API/app/core/Research/discovery/contracts.py` for `SkippedCode.QUERY_MODE_NOT_SUPPORTED` and one stable non-sensitive reason constant only.
- Modify `tldw_Server_API/app/core/Research/discovery/planner.py`.
- Modify `tldw_Server_API/app/core/Research/discovery/executor.py`.
- Modify `tldw_Server_API/tests/Research/test_research_discovery_planner.py`.
- Modify `tldw_Server_API/tests/Research/test_research_discovery_executor.py`.
- Modify `tldw_Server_API/tests/Research/test_research_discovery_v2_compatibility.py`.

### Typed request shape

Keep the public types in `planner.py`, where `PlanningRequest` already lives:

```python
@dataclass(frozen=True, slots=True)
class GeneralFreeTextQuery:
    text: str


@dataclass(frozen=True, slots=True)
class IdentifierLookupQuery:
    doi: str


@dataclass(frozen=True, slots=True)
class DateIntervalQuery:
    start_date: str
    end_date: str
    category: str | None = None


PlanningQuery = str | GeneralFreeTextQuery | IdentifierLookupQuery | DateIntervalQuery
```

### Test-first steps

- [ ] Write RED construction tests accepting only the exact built-in `str` or exact three typed dataclasses. Reject subclasses, duck types, empty text, invalid Unicode/control content, invalid DOI shape, invalid calendar dates, reversed/over-366-day intervals, empty/invalid categories, and non-empty `filters` on typed requests.
- [ ] Add an internal normalized query context with the selected `QueryMode` and canonical display/storage value. Preserve the existing `_normalize_query(str)` and `_build_intents(...)` raw-string branch byte-for-byte.
- [ ] Write RED route-selection tests: `GeneralFreeTextQuery` selects only `GENERAL_FREE_TEXT`; `IdentifierLookupQuery` only `IDENTIFIER_LOOKUP`; an interval without category only `DATE_INTERVAL`; an interval with category only `CATEGORY_BROWSE`; raw strings remain `STRUCTURED_QUERY`.
- [ ] Add `SkippedCode.QUERY_MODE_NOT_SUPPORTED` plus exact reason `query_mode_not_supported`, and emit one stable skipped outcome for each explicitly selected incompatible source/route instead of silently dropping it. Mode mismatch must be evaluated before readiness/credential handling, independently of egress, and no mode field is added to plans, groups, or intents.
- [ ] Extend executor plan validation so the new code is accepted only as exact `SkippedStatus.SKIPPED` with the exact stable reason on any otherwise valid route; preserve existing credential/not-ready semantics and reject forged combinations before adapters, journals, ID factories, or gateway effects.
- [ ] Write RED canonical Europe PMC intent tests. Tokenize the general query into a bounded sequence of NFKC Unicode alphanumeric terms, quote each term, join with literal `AND`, append the exact route-owned publisher suffix, and reject input with no terms. Do not copy punctuation or provider syntax.
- [ ] Write RED details-intent tests rendering these exact concrete paths:

```text
/details/{server}/{start_date}/{end_date}/0/json
/details/{server}/{doi_registrant}/{doi_suffix}/na/json
```

The DOI has exactly one slash; registrant full-matches `10.` plus 4-9 digits; suffix begins alphanumeric, is bounded, contains no slash/backslash/percent/control, and is encoded once with the shared canonical path renderer.
- [ ] Write RED executor tests for a route with no pagination channel: one initial `SEARCH` dispatch succeeds only when `max_pages == 1`, but any supplied cursor or multi-page zero-channel route is rejected. This covers Europe PMC and DOI lookup without weakening foundation routes.
- [ ] Write RED executor tests for path pagination: first dispatch extracts the initial numeric path cursor, a later `NumericCursor` replaces exactly the declared template segment, repeated/retrograde/overflow cursors fail, and query/body/path channels are mutually exclusive.
- [ ] Implement path cursor replacement inside `_GroupExecutionController._effective_intent(...)` with `dataclasses.replace`. Keep every derived intent under `_snapshot_binding`, policy activity, journal reservation/debit, gateway dispatch, deadline, cancellation, and page accounting.
- [ ] Record a cursor in `_seen_cursors` only when it is a real integer; a valid one-page zero-channel search must never insert `None` into cursor lineage.
- [ ] Add a malicious adapter regression proving it cannot change any other path segment, path literal, date, server, DOI, query value, policy digest, or cursor channel.
- [ ] Re-run exact canonical foundation-plan tests to prove raw-string behavior, `PLANNER_VERSION`, ordering, IDs, skipped semantics, digests, and serialized bytes did not change.

### Focused RED/GREEN command

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest -q \
  tldw_Server_API/tests/Research/test_research_discovery_contracts.py \
  tldw_Server_API/tests/Research/test_research_discovery_planner.py \
  tldw_Server_API/tests/Research/test_research_discovery_gateway.py \
  tldw_Server_API/tests/Research/test_research_discovery_executor.py \
  tldw_Server_API/tests/Research/test_research_discovery_v2_compatibility.py
```

### Commit

`feat(research): add typed discovery query planning`

### Completion evidence

- Implementation: `293ef476d0540fd4fa2d10b64e7dba50fcded57d`
- Review hardening: `794fb20b255c735fd1eb56be561feee056939066`
- Focused final suite: 702 passed with four pre-existing warnings.
- The unchanged foundation oracle retained its exact plan digest, canonical-byte SHA-256/length, and eight route-policy digests.
- Compileall, Ruff, Black, diff hygiene, and Bandit passed; Bandit reported zero findings.
- Independent specification review and post-fix code-quality review approved Stage 2 with no open Critical or Important findings.

## Task 3 / Stage 3: Build the Shadow Family and Europe PMC General Discovery

**Goal:** Add the two stable catalog targets and source-constrained one-page general search without affecting the eight-source foundation.

**Success Criteria:** A shadow registry has two additional sources, six declared routes, and two physical backends; both Europe PMC routes produce bounded attributable metadata through the shared gateway; unknown/mismatched publishers are never force-attributed.

**Tests:** New focused family suite plus foundation registry/compatibility checks.

**Status:** Complete

### Files

- Add `tldw_Server_API/app/core/Research/discovery/biorxiv_medrxiv.py`.
- Add `tldw_Server_API/tests/Research/test_research_discovery_biorxiv_medrxiv.py`.
- Add `tldw_Server_API/tests/fixtures/research_discovery_gateway_adapters/europe_pmc_biorxiv_success.json`.
- Add `tldw_Server_API/tests/fixtures/research_discovery_gateway_adapters/europe_pmc_medrxiv_success.json`.
- Add `tldw_Server_API/tests/fixtures/research_discovery_gateway_adapters/europe_pmc_empty.json`.
- Modify `tldw_Server_API/tests/Research/test_research_discovery_registry_reconciliation.py` only for the additive shadow registry.
- Modify `tldw_Server_API/tests/Research/test_research_discovery_v2_compatibility.py` only for foundation non-regression.

### Family factories and immutable identities

Implement explicit names so no caller can confuse shadow and foundation scope:

```python
def biorxiv_medrxiv_shadow_registry() -> DiscoveryRegistry: ...
def biorxiv_medrxiv_shadow_readiness(execution_mode: ExecutionMode) -> ReadinessOverlay: ...
def biorxiv_medrxiv_gateway_adapters(
    *, monotonic_clock: MonotonicClock = time.monotonic,
) -> Mapping[str, DiscoveryAdapter]: ...
```

Freeze distinct constants for the family catalog, registry, readiness, policy, and adapter versions. Rebuild foundation source values with `replace(source, catalog_version=SHADOW_CATALOG_VERSION)` before concatenating; never mutate foundation objects.

### Test-first steps

- [x] Write RED registry tests for exact stable source IDs `biorxiv`/`medrxiv`, aliases, priorities after the foundation set, site hosts, categories/content types, intended surfaces, ordered route references, six route IDs, two backend IDs, and exact `https:443` origins.
- [x] Assert the general routes use `RouteKind.AGGREGATOR`, `SourceConstraint.PROVIDER_SOURCE_FILTER`, `QueryMode.GENERAL_FREE_TEXT`, adapter `europe_pmc_preprint_v2`, one page, zero redirects, no credentials, and exact `source_platform` predicates.
- [x] Assert the details routes are declared now but use the correct direct kinds, query modes, server-specific literal templates, shared details backend, and explicit `DISABLED` readiness reason until Stage 4 supplies the adapter. Stage 4 changes those entries to `READY`; no ready route may lack an adapter.
- [x] Prove foundation and shadow catalog/registry/readiness versions differ, all source values match their enclosing catalog version, foundation values are unchanged, duplicate IDs are rejected, and composing adapter maps rejects duplicate adapter IDs.
- [x] Write RED request tests for exact Europe PMC path `/europepmc/webservices/rest/search`; required keys `query`, `format`, `resultType`, `pageSize`; exact `format=json`; exact `resultType=core`; `pageSize=min(result_limit, route.max_results)`; publisher-specific fixed suffix; no `cursorMark`.
- [x] Add one bounded family parse profile keyed by exact `(adapter_id, adapter_version)`. Reuse `_ParsingProfile`, `_strict_json`, `_checked_response`, `_require_dict`, `_require_list`, `_required_text`, `_optional_text`, `_base_record`, `_raise_adapter_error`, `build_fingerprint`, `DiscoveryOutcomeIdentity`, `DiscoveryAdapterError`, and `BoundDispatch` rather than copying their behavior.
- [x] Write RED success and valid-empty tests using only the three checked-in fixtures. Accept exact Europe PMC core result shape, top-level `source == "PPR"`, and exact known `bookOrReportDetails.publisher` values.
- [x] Normalize a validated known publisher into `source_platform`; let the existing exact `SourcePredicate` decide bioRxiv versus medRxiv. Missing/unknown attribution is ambiguous; a different known publisher is a non-match.
- [x] Retain only bounded title, authors, abstract/snippet, date/year, DOI, PPR identifier, derived source platform, provider name/IDs, and synthesized inert links. Synthesize only `https://doi.org/{doi}` or `https://europepmc.org/article/PPR/{id}` from strict identifiers; leave `pdf_url=None`.
- [x] Drop provider URLs, `nextCursorMark`, `nextPageUrl`, HTML, grants, annotations, references, full-text/JATS links, and unknown fields. Convert bounded title/abstract HTML fragments to plain text without I/O.
- [x] Add parameterized mutations for malformed JSON/UTF-8/schema, HTML bounds, too many records, missing/unknown/mismatched source/publisher, bad DOI/PPR IDs, duplicate/conflicting identities, hostile next-page URLs, rate limit with/without valid retry metadata, timeout, cancellation, and one-source partial failure.
- [x] Prove the adapter invokes `dispatch(intent)` exactly once, uses no cursor, ignores provider continuation fields, performs zero result-link requests, and emits only existing typed errors/outcomes.

### Focused RED/GREEN command

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest -q \
  tldw_Server_API/tests/Research/test_research_discovery_biorxiv_medrxiv.py \
  tldw_Server_API/tests/Research/test_research_discovery_registry_reconciliation.py \
  tldw_Server_API/tests/Research/test_research_discovery_planner.py \
  tldw_Server_API/tests/Research/test_research_discovery_gateway.py \
  tldw_Server_API/tests/Research/test_research_discovery_executor.py \
  tldw_Server_API/tests/Research/test_research_discovery_v2_compatibility.py
```

### Commit

`feat(research): add Europe PMC preprint discovery routes`

- Implementation: `b6d416cf4dfbcb4109d6668c17c34ed3f35492eb`.
- Exact six-suite gate: 798 passed with four pre-existing warnings; final family suite: 159 passed with four pre-existing warnings.
- Ruff, Black, Python byte-compilation, fixture JSON validation, diff hygiene, and Bandit passed; Bandit reported zero findings and zero errors.
- Independent specification and code-quality/security reviews found no Critical, Important, or actionable Minor findings.
- The family remains shadow-only: Europe PMC general routes are ready, the four official-details routes remain disabled, and production Search/Deep Research consumers are unchanged.

## Task 4 / Stage 4: Implement Strict Official Details Lookup and Interval Adapters

**Goal:** Add truthful DOI, date-interval, and category operations over the official bioRxiv/medRxiv details service with response-to-request binding and response-derived path pagination.

**Success Criteria:** DOI and interval routes validate the exact requested server/identifier/window/category/cursor, retain only bounded metadata, paginate atomically, and reject drift or force-attribution.

**Tests:** Expand the focused family suite with distinct valid details fixtures and derived failure cases.

**Status:** In Progress

### Files

- Modify `tldw_Server_API/app/core/Research/discovery/biorxiv_medrxiv.py`.
- Modify `tldw_Server_API/tests/Research/test_research_discovery_biorxiv_medrxiv.py`.
- Add `tldw_Server_API/tests/fixtures/research_discovery_gateway_adapters/biorxiv_details_doi_success.json`.
- Add `tldw_Server_API/tests/fixtures/research_discovery_gateway_adapters/medrxiv_details_doi_success.json`.
- Add `tldw_Server_API/tests/fixtures/research_discovery_gateway_adapters/biorxiv_details_interval_page_1.json`.
- Add `tldw_Server_API/tests/fixtures/research_discovery_gateway_adapters/biorxiv_details_interval_page_2.json`.
- Add `tldw_Server_API/tests/fixtures/research_discovery_gateway_adapters/biorxiv_details_interval_empty.json`.

### Test-first steps

- [ ] Write RED DOI success tests for both server routes and exact concrete paths with two DOI segments. Reject an encoded DOI slash as one segment and all multi-slash identifiers.
- [ ] Write RED interval/category success tests for exact dates, optional canonical category query, initial cursor `0`, and page two requested with `NumericCursor(current + count)` rather than a hard-coded page size.
- [ ] Validate HTTP/MIME through `_checked_response` and JSON through `_strict_json`. Require `messages` to contain one usable status record with success status, expected exact mixed-case server label (`bioRxiv` or `medRxiv`), and bounded canonical numeric strings or integers.
- [ ] DOI lookup must bind every collection item to the exact canonical requested DOI. Missing/different DOI or server is malformed/ambiguous and commits no partial candidates.
- [ ] Interval responses must echo the exact requested interval, optional category, and current cursor; require `count == len(collection)`, `cursor + count <= total`, dates inside the inclusive interval, and category equality after canonical normalization.
- [ ] Treat a zero-count page with remaining `total`, repeated/non-progress cursor, decreasing/inconsistent total, or inconsistent termination as malformed. The route attempt remains atomic across pages.
- [ ] Normalize bounded title, authors, abstract/snippet, DOI, date, numeric version, license, category, publication linkage, provider IDs, source platform, and synthesized DOI landing URL. Drop raw provider URLs, `jatsxml`, full-text/PDF fields, payload echoes, and unknown fields; keep `pdf_url=None`.
- [ ] For multiple records with one DOI, retain the highest numeric version deterministically. If the same version has conflicting retained metadata, fail the route instead of choosing one.
- [ ] Add derived tests for application-level error under HTTP 200; string/integer count variants; requested DOI, interval, category, cursor, count, result date, result category, server, and total mismatches; invalid versions; empty interval; malformed JSON/schema; rate limit; timeout; cancellation; and later-page failure suppressing earlier-page candidates.
- [ ] Change the four details readiness entries from the Stage 3 fixture-pending `DISABLED` state to `READY` only after their adapter tests pass, then assert every ready shadow route resolves to exactly one registered adapter.
- [ ] Before enabling the medRxiv interval route, derive its valid payload deterministically from the checked-in bioRxiv interval fixture by changing only the exact response server label and medRxiv-specific item fields, then execute the real medRxiv interval plan end-to-end and assert its path, response binding, attribution, candidate, and accounting. Do not mark it ready based only on the bioRxiv route.
- [ ] Add link tripwires proving DOI/JATS/full-text/PDF/provider links are never dereferenced and no Media/legacy helper is imported.
- [ ] Execute both details modes through `execute_discovery_plan(...)` and assert physical dispatch/page counts, unique dispatch IDs, cursor lineage, cancellation behavior, valid-empty outcomes, and independent partial failure beside another selected source.

### Focused RED/GREEN command

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest -q \
  tldw_Server_API/tests/Research/test_research_discovery_biorxiv_medrxiv.py \
  tldw_Server_API/tests/Research/test_research_discovery_executor.py \
  tldw_Server_API/tests/Research/test_research_discovery_gateway.py \
  tldw_Server_API/tests/Research/test_research_discovery_planner.py
```

### Commit

`feat(research): add bioRxiv details discovery routes`

## Task 5 / Stage 5: Close the Boundary, Reconcile Inventory, and Verify

**Goal:** Prove the new family has no alternate effect path, update the authoritative source ledger from the invalid native-search assumption to the implemented shadow routes, and finish with reproducible full-suite/security evidence.

**Success Criteria:** Static/runtime tripwires cover the family module and ready adapters; rows 0021/0022 describe Europe PMC plus official details accurately; the authoritative report regenerates exactly with no schema/count drift; focused/full tests, style, syntax, Bandit, diff hygiene, and independent review are clean; TASK-12968.5 is truthfully finalized.

**Tests:** Network-boundary suite, inventory semantic tests, Python schema tests, registry reconciliation, legacy compatibility, full Research regression, static quality gates, and Bandit.

**Status:** Not Started

### Files

- Modify `tldw_Server_API/tests/Research/test_research_discovery_network_boundary.py`.
- Modify `Helper_Scripts/validate_research_source_inventory.mjs`.
- Modify `Helper_Scripts/tests/validate_research_source_inventory.test.mjs`.
- Modify `Docs/Design/research_source_inventory/research-source-coverage-ledger-2026-07-13.json`.
- Regenerate `Docs/Design/research_source_inventory/research-source-inventory-freeze-report-2026-07-13.json` through the authoritative validator.
- Update this plan's stage statuses/evidence as work completes.
- Update `backlog/tasks/task-12968.5 - Add-bioRxiv-and-medRxiv-shared-discovery-route-family.md` through the official Backlog CLI only, except the known narrow duplicate-final-summary-marker repair if the CLI reproduces it.
- Update `.superpowers/sdd/progress.md` only if the active subagent workflow requires it.
- Do not modify `Docs/Design/research_source_inventory/sourclip-research-sources-2026-07-13.json` or the JSON schema unless an actual validator/schema defect is first demonstrated and separately reviewed.

### Test-first steps

- [ ] Add a separate family root/fixture/digest set to the static discovery closure scan; do not add family adapters to the frozen foundation `_RECORDED_FIXTURES`. Enumerate ready family route `(adapter_id, adapter_version)` identities at runtime and require exact equality with `biorxiv_medrxiv_gateway_adapters()`.
- [ ] Extend runtime tripwires across all six routes: no direct socket/http/urllib client, alternate hop, subprocess, browser, cookie, credential, config, AuthNZ, DB, Web Scraping, Media, OA, legacy Third_Party provider, or result-link access.
- [ ] Refresh reviewed raw and AST digests only after the final code is stable. Keep the test's closed import/gateway/identity expectations explicit; do not weaken it to a wildcard scan.
- [ ] Add fixture accounting for the distinct family success fixtures and prove every ready route crosses the executor-owned gateway with a physical debit. Details page two must be separately accounted.
- [ ] Add the family module to the compatibility suite's forbidden production-consumer import set and patch its factories with raising tripwires during a real legacy endpoint request; assert zero family factory, V2 executor, and V2 gateway calls.
- [ ] Write RED inventory semantic tests for the corrected required-source contract before changing production validator constants.
- [ ] Replace required native site-search routes with `biorxiv_europe_pmc_search_aggregator` and `medrxiv_europe_pmc_search_aggregator`; require `RouteKind.AGGREGATOR`, backend `europe_pmc_rest_api`, `GENERAL_FREE_TEXT`, provider-source constraint, exact publisher predicate, and Europe PMC evidence host.
- [ ] Replace each single stale bounded route requirement with separate lookup and interval requirements using backend `biorxiv_details_api`; allow only identifier lookup on lookup routes and date/category modes on interval routes. Do not require or advertise `RECENT_FEED`.
- [ ] Update ledger rows `sourclip-2026-07-13-0021` and `...0022`: retain canonical targets and both declared surfaces; remove native site-search and recent-shortcut candidates from `route_candidates`; add the implemented six route candidates; record dated robots/Cloudflare/recent-shortcut evidence in resolution/evidence fields; keep live state `not_run`, certifications empty, and surface claims absent.
- [ ] Set `implementation_state=implemented` and `fixture_state=passed` only after Stages 3-4 are green. Keep `live_state=not_run`, and do not imply inventory delivery/surface certification.
- [ ] Assert the derived counts become exactly `233 planned / 2 implemented` and `233 not_run / 2 passed` for fixtures while all 235 live states remain `not_run`; resolution counts remain `191 mapped / 35 credentialed` with all other counts unchanged.
- [ ] Recompute the ledger `rows_sha256` using the validator's existing `canonicalJson(...)` and `sha256(...)`; do not hand-guess derived hashes.
- [ ] Run the authoritative validator with the frozen as-of date and trusted reviewer, capture JSON, and update the checked-in report only from that exact result:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
node Helper_Scripts/validate_research_source_inventory.mjs \
  --root . \
  --gate contract \
  --as-of 2026-07-13 \
  --trusted-reviewer codex-task-12968.1-source-triage \
  --json
```

- [ ] Assert manifest/ledger row counts and resolution counts do not drift, errors remain empty, required-source mappings are satisfied, and regenerated report bytes match the checked-in report.

### Focused RED/GREEN command

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest -q \
  tldw_Server_API/tests/Research/test_research_discovery_biorxiv_medrxiv.py \
  tldw_Server_API/tests/Research/test_research_discovery_network_boundary.py \
  tldw_Server_API/tests/Research/test_research_discovery_registry_reconciliation.py \
  tldw_Server_API/tests/Research/test_research_discovery_v2_compatibility.py \
  tldw_Server_API/tests/Research/test_research_discovery_legacy_selection_contract.py
node --test Helper_Scripts/tests/validate_research_source_inventory.test.mjs
python -m pytest -q Helper_Scripts/tests/test_research_source_inventory_schema.py
```

### Final verification and task-finalization steps

- [ ] Run the focused family/security matrix:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest -q \
  tldw_Server_API/tests/Research/test_research_discovery_contracts.py \
  tldw_Server_API/tests/Research/test_research_discovery_planner.py \
  tldw_Server_API/tests/Research/test_research_discovery_gateway.py \
  tldw_Server_API/tests/Research/test_research_discovery_executor.py \
  tldw_Server_API/tests/Research/test_research_discovery_biorxiv_medrxiv.py \
  tldw_Server_API/tests/Research/test_research_discovery_registry_reconciliation.py \
  tldw_Server_API/tests/Research/test_research_discovery_network_boundary.py \
  tldw_Server_API/tests/Research/test_research_discovery_v2_compatibility.py \
  tldw_Server_API/tests/Research/test_research_discovery_legacy_execution_contract.py \
  tldw_Server_API/tests/Research/test_research_discovery_legacy_selection_contract.py
```

- [ ] Run the complete Research suite:

```bash
python -m pytest -q tldw_Server_API/tests/Research
```

- [ ] Run authoritative inventory gates:

```bash
node --test Helper_Scripts/tests/validate_research_source_inventory.test.mjs
python -m pytest -q Helper_Scripts/tests/test_research_source_inventory_schema.py
node Helper_Scripts/validate_research_source_inventory.mjs \
  --root . \
  --gate contract \
  --as-of 2026-07-13 \
  --trusted-reviewer codex-task-12968.1-source-triage \
  --json
```

- [ ] Run compile/style checks on the exact touched Python set:

```bash
python -m compileall -q \
  tldw_Server_API/app/core/Research/discovery/contracts.py \
  tldw_Server_API/app/core/Research/discovery/planner.py \
  tldw_Server_API/app/core/Research/discovery/gateway.py \
  tldw_Server_API/app/core/Research/discovery/executor.py \
  tldw_Server_API/app/core/Research/discovery/biorxiv_medrxiv.py \
  tldw_Server_API/tests/Research/test_research_discovery_contracts.py \
  tldw_Server_API/tests/Research/test_research_discovery_planner.py \
  tldw_Server_API/tests/Research/test_research_discovery_gateway.py \
  tldw_Server_API/tests/Research/test_research_discovery_executor.py \
  tldw_Server_API/tests/Research/test_research_discovery_biorxiv_medrxiv.py \
  tldw_Server_API/tests/Research/test_research_discovery_registry_reconciliation.py \
  tldw_Server_API/tests/Research/test_research_discovery_network_boundary.py \
  tldw_Server_API/tests/Research/test_research_discovery_v2_compatibility.py
ruff check \
  tldw_Server_API/app/core/Research/discovery/contracts.py \
  tldw_Server_API/app/core/Research/discovery/planner.py \
  tldw_Server_API/app/core/Research/discovery/gateway.py \
  tldw_Server_API/app/core/Research/discovery/executor.py \
  tldw_Server_API/app/core/Research/discovery/biorxiv_medrxiv.py \
  tldw_Server_API/tests/Research/test_research_discovery_contracts.py \
  tldw_Server_API/tests/Research/test_research_discovery_planner.py \
  tldw_Server_API/tests/Research/test_research_discovery_gateway.py \
  tldw_Server_API/tests/Research/test_research_discovery_executor.py \
  tldw_Server_API/tests/Research/test_research_discovery_biorxiv_medrxiv.py \
  tldw_Server_API/tests/Research/test_research_discovery_registry_reconciliation.py \
  tldw_Server_API/tests/Research/test_research_discovery_network_boundary.py \
  tldw_Server_API/tests/Research/test_research_discovery_v2_compatibility.py
black --check \
  tldw_Server_API/app/core/Research/discovery/contracts.py \
  tldw_Server_API/app/core/Research/discovery/planner.py \
  tldw_Server_API/app/core/Research/discovery/gateway.py \
  tldw_Server_API/app/core/Research/discovery/executor.py \
  tldw_Server_API/app/core/Research/discovery/biorxiv_medrxiv.py \
  tldw_Server_API/tests/Research/test_research_discovery_contracts.py \
  tldw_Server_API/tests/Research/test_research_discovery_planner.py \
  tldw_Server_API/tests/Research/test_research_discovery_gateway.py \
  tldw_Server_API/tests/Research/test_research_discovery_executor.py \
  tldw_Server_API/tests/Research/test_research_discovery_biorxiv_medrxiv.py \
  tldw_Server_API/tests/Research/test_research_discovery_registry_reconciliation.py \
  tldw_Server_API/tests/Research/test_research_discovery_network_boundary.py \
  tldw_Server_API/tests/Research/test_research_discovery_v2_compatibility.py
```

- [ ] Parse every touched Python file with Python 3.10 grammar (or the repository's existing AST grammar helper) and record the exact command/result.
- [ ] Run Bandit over the touched production scope and inspect the JSON rather than trusting exit status alone:

```bash
python -m bandit -r \
  tldw_Server_API/app/core/Research/discovery \
  -f json -o /tmp/bandit_TASK-12968.5.json
```

- [ ] Run `git diff --check`, inspect `git diff --stat`, inspect the complete diff, and prove no production Search/Deep Research import or request count changed.
- [ ] Request independent correctness, security, and minimality reviews. Fix Critical/Important findings RED-first; re-run the affected and full gates. Keep no speculative abstraction solely for hypothetical providers.
- [ ] Update the task with touched files, fixture/live distinction, exact test counts, inventory report outcome, Bandit outcome, review result, known skips, local commit hashes, and explicit statements that surface cutover/authenticated scraping remain deferred.
- [ ] Check all acceptance criteria/DoD only after evidence exists. Add one final summary, repair any duplicated CLI summary markers, validate task parsing, and commit the final reviewable unit.

### Commits

`docs(research): reconcile bioRxiv discovery inventory`

`test(research): verify bioRxiv medRxiv shadow discovery`

## Completion Boundary

Completing this plan means bioRxiv and medRxiv are implemented and fixture-certified in the shared discovery shadow pipeline with two truthful capability families: bounded Europe PMC general discovery and explicit official-details operations. It does not make either target available in production Search or Deep Research, does not certify live/surface readiness, does not enable native site search or recent shortcuts, and does not begin authenticated scraping. Those remain separately authorized work.
