# bioRxiv and medRxiv Shared-Discovery Route Family

**Task:** TASK-12968.5

**Parent:** TASK-12968

**Status:** Approved design correction (2026-07-15)

**Execution scope:** Shadow registry and fixture execution only

## Goal

Add stable `biorxiv` and `medrxiv` catalog targets to the shared discovery pipeline without claiming that a bounded metadata feed is complete historical free-text search. The route family must support credentialless general discovery and the useful structured capabilities of the official details API while preserving the existing foundation and all production consumer behavior.

This document narrows and corrects the bioRxiv/medRxiv assumptions in the parent design. The parent architecture, gateway boundary, provenance model, consumer-cutover ownership, and authenticated-scraping deferral remain unchanged.

## Provider Evidence and Design Correction

The originally proposed anonymous native site-search adapter is not viable for phase one:

- The [official bioRxiv/medRxiv API](https://api.biorxiv.org/) provides metadata by explicit date range, category, DOI, and documented recent shortcuts. It does not expose arbitrary free-text search.
- The human [site-search documentation](https://www.biorxiv.org/content/search-tips) confirms free-text search exists, but the route is an HTML product surface rather than a supported API.
- [medRxiv robots.txt](https://www.medrxiv.org/robots.txt) disallows `/search/` and declares a seven-second crawl delay.
- A credentialless plain-HTTP probe of bioRxiv `/search/` on 2026-07-15 received a Cloudflare challenge.
- Live probes of the details API's recent-count/day shorthands returned stale records or HTTP 500. Explicit date intervals and DOI lookup worked. The shorthand is therefore not enabled merely because it is documented.
- The [Europe PMC REST API](https://europepmc.org/RestfulWebService) supports programmatic free-text search. Live credentialless probes on 2026-07-15 returned bounded preprint results for both `SRC:PPR AND PUBLISHER:"bioRxiv"` and `SRC:PPR AND PUBLISHER:"medRxiv"`, with exact publisher metadata on each result.
- Crossref exact container filters returned zero for both sources in live probes. Semantic Scholar venue filtering returned relevant records, but its bounded relevance endpoint was anonymously rate-limited during the same review. Neither is selected as the primary phase-one route.

Consequently:

1. Native HTML site search is recorded as policy-blocked/technically infeasible for this phase and is not an enabled route.
2. Europe PMC supplies source-constrained general free-text discovery.
3. The official details API supplies separate typed interval, category, and DOI operations.
4. Recent-count/day shortcuts remain disabled until a later live review proves stable semantics.

## Scope

This task includes:

- two catalog targets, with Standalone Search and Deep Research declared as intended surfaces;
- a Europe PMC general-search aggregator route for each target;
- official details lookup and explicit-interval routes for each target;
- typed query-mode selection so incompatible routes are never attempted together;
- bounded validated path-template and numeric path-cursor support for the details API;
- strict normalization, attribution, inert links, request accounting, fixtures, and shadow execution;
- ledger reconciliation and live feasibility notes;
- compatibility proofs showing no change to foundation plans, legacy selection, Search, or Deep Research.

This task does not include:

- production Search or Deep Research cutover;
- local metadata/full-text indexing or scheduled backfills;
- arbitrary HTML retrieval, browser automation, cookies, credentials, or Cloudflare bypasses;
- result-link dereference, Media ingestion, JATS retrieval, PDF retrieval, or full-text retention;
- surface-ready or live-certified claims;
- a fallback from one query mode to a semantically different mode.

## Stable Runtime Identities

### Catalog targets

| Source ID | Display name | Site host | Declared surfaces |
| --- | --- | --- | --- |
| `biorxiv` | bioRxiv | `biorxiv.org` | `standalone_search`, `deep_research` |
| `medrxiv` | medRxiv | `medrxiv.org` | `standalone_search`, `deep_research` |

### Physical backends

| Backend ID | Physical service | Exact origin |
| --- | --- | --- |
| `europe_pmc_rest_api` | Europe PMC REST API | `https://www.ebi.ac.uk:443` |
| `biorxiv_details_api` | Shared bioRxiv/medRxiv details service | `https://api.biorxiv.org:443` |

bioRxiv and medRxiv do not receive separate details backend IDs. A backend identifies one physical external service, while source and route identities capture the server-specific request and attribution semantics.

### Routes

| Route ID | Target | Kind | Query modes | Backend | Adapter | Shadow readiness |
| --- | --- | --- | --- | --- | --- | --- |
| `biorxiv_europe_pmc_search_aggregator` | bioRxiv | `aggregator` | `general_free_text` | `europe_pmc_rest_api` | `europe_pmc_preprint_v2` | fixture-ready after this task |
| `medrxiv_europe_pmc_search_aggregator` | medRxiv | `aggregator` | `general_free_text` | `europe_pmc_rest_api` | `europe_pmc_preprint_v2` | fixture-ready after this task |
| `biorxiv_details_lookup_direct` | bioRxiv | `direct` | `identifier_lookup` | `biorxiv_details_api` | `biorxiv_details_v2` | fixture-ready after this task |
| `medrxiv_details_lookup_direct` | medRxiv | `direct` | `identifier_lookup` | `biorxiv_details_api` | `biorxiv_details_v2` | fixture-ready after this task |
| `biorxiv_details_interval_direct` | bioRxiv | `direct` | `date_interval`, `category_browse` | `biorxiv_details_api` | `biorxiv_details_v2` | fixture-ready after this task |
| `medrxiv_details_interval_direct` | medRxiv | `direct` | `date_interval`, `category_browse` | `biorxiv_details_api` | `biorxiv_details_v2` | fixture-ready after this task |

Native site-search and recent-shortcut outcomes remain ledger evidence, not viable runtime route candidates. Those rejected outcomes must not make the otherwise usable general or structured routes unavailable.

## Typed Query Contract

The current route `query_modes` field is descriptive, while the foundation planner executes every ready route for a selected source. That is insufficient once one source has semantically different routes.

The planner gains typed query values while preserving raw-string foundation compatibility:

- Existing raw string requests keep the current `structured_query` interpretation and must produce byte-for-byte equivalent foundation plans.
- `GeneralFreeTextQuery(text)` selects only `general_free_text` routes.
- `IdentifierLookupQuery(doi)` selects only `identifier_lookup` routes.
- `DateIntervalQuery(start_date, end_date, category=None)` selects `date_interval`; a non-empty category selects `category_browse`.
- No phase-one `RecentFeedQuery` is executable for this family.

The query value is validated before route resolution. Invalid dates, reversed intervals, unsupported identifier shapes, empty categories, incompatible filters, and mixed query modes fail during pure planning. A raw string has the existing `structured_query` mode; because the new routes do not declare that mode, a raw-string request cannot accidentally execute them. A general query never triggers details scanning, and an interval/DOI request never triggers Europe PMC. Incompatible routes receive a stable `query_mode_not_supported` skipped code rather than disappearing or executing with the wrong shape.

The request accepts either an exact built-in string or one of the three exact typed query dataclasses; subclasses and duck-typed substitutes are rejected. Query mode is planner selection semantics, not an egress capability, so no mode field or reserved user-filter namespace is added to `DiscoveryPlan`, dispatch groups, or intents. The chosen route plus its concrete query/path shape records the operation, while the gateway policy independently enforces every egress-relevant value. Existing raw-string foundation plans therefore remain byte-for-byte unchanged.

The executed plan remains immutable. Its existing normalized query/filter fields, concrete route intent, policy digest, and pagination contract capture the operation without changing the foundation projection. A plan-mode field is deferred unless query modes later receive different authorization, credential, retention, quota, or network privileges.

## Europe PMC General Search

### Request shape

- Method: `GET`
- Path: `/europepmc/webservices/rest/search`
- Fixed query values: `format=json` and `resultType=core`
- Bounded query value: `pageSize=min(result_limit, route.max_results)`
- Pagination: one page only (`max_pages=1`)
- Redirects: disabled
- Cookies and credentials: absent

The planner constructs the `query` parameter from literalized user terms plus an immutable suffix:

- bioRxiv: `SRC:PPR AND PUBLISHER:"bioRxiv"`
- medRxiv: `SRC:PPR AND PUBLISHER:"medRxiv"`

User text is normalized into a bounded sequence of Unicode alphanumeric terms. The canonical provider expression quotes each term and joins them with literal `AND` operators before the fixed suffix. Punctuation and provider-query syntax are never copied into the outbound value; a query with no alphanumeric terms fails planning.

The route policy contains digest-bound query-value rules for every permitted key. Exact rules require `format=json` and `resultType=core`; a bounded-decimal rule caps `pageSize` at the route result ceiling; and a literal-terms-with-suffix rule fixes the complete bioRxiv or medRxiv suffix and the term count/length grammar. `cursorMark` is omitted because Europe PMC defines omission as the first page. The gateway independently validates the canonical expression and every fixed/bounded value before dispatch. A self-consistent reconstructed plan therefore cannot remove the source constraint, inject Europe PMC field syntax, enlarge the page, change the response shape, or request a continuation.

### Attribution

The adapter exposes a derived source platform only when all of the following hold:

- the record's top-level `source` is exactly `PPR`;
- `bookOrReportDetails.publisher` is exactly `bioRxiv` or `medRxiv`, case-normalized;
- the publisher matches the logical target's exact source predicate.

The normalizer exposes a safe derived `source_platform` field only after the `PPR` and publisher checks. The existing exact `SourcePredicate` operates only on that derived field; no Boolean predicate tree is added. Missing or unknown raw attribution is dropped as ambiguous, and a different known publisher is dropped as a non-match. If no attributable record remains, the route has the existing valid-empty outcome rather than a forced attribution.

### One-page bound

Europe PMC permits up to 1,000 results per page. This route deliberately asks for only the plan's bounded result ceiling and never sends `cursorMark` or requests a continuation page. Returned `nextCursorMark` and `nextPageUrl` are ignored and dropped. The route is described as bounded first-page discovery, not a complete export of all matching Europe PMC records. Real pagination coverage for this task is provided by the details interval routes.

### Retention

The adapter retains only bounded normalized metadata: title, authors, abstract/snippet, publication date/year, DOI, Europe PMC preprint ID, source platform, and inert validated DOI/Europe PMC links. Links are synthesized only from validated DOI or `PPR` identifiers onto exact `doi.org` or `europepmc.org` hosts; provider-supplied URLs never reach the generic URL sanitizer or `pdf_url`. The adapter drops raw HTML, `nextPageUrl`, arbitrary full-text URLs, grants, annotations, references, and unknown fields. Provider HTML fragments in title or abstract are converted to bounded plain text without fetching anything.

## Official Details Routes

### Request shapes

The details service requires dynamic path segments rather than a query-string cursor:

- Explicit interval: `/details/{server}/{start_date}/{end_date}/{cursor}/json`
- DOI lookup: `/details/{server}/{doi_registrant}/{doi_suffix}/na/json`
- Optional interval category: exact `category` query parameter

`server` is a literal template segment fixed by the route as `biorxiv` or `medrxiv`; it is never taken from user text. Each route separately freezes the exact provider response label (`bioRxiv` or `medRxiv`) rather than comparing that mixed-case field to the lowercase path literal. Dates must be real calendar dates in canonical ISO `YYYY-MM-DD` form, ordered inclusively, with a maximum 366-day span. Categories are separately encoded query values containing only bounded letters, numbers, spaces, hyphens, ampersands, and slashes.

The interval route policy carries an optional bounded-text query-value rule for `category`, so the gateway rechecks that grammar and size independently of planning.

The DOI must be strict ASCII with exactly one slash, a full-match registrant `10.` followed by four through nine digits, and a bounded path-safe suffix beginning with an alphanumeric character. This validation does not infer source identity from the DOI. The provider does not accept an encoded DOI slash as one segment, so unsupported multi-slash identifier shapes fail closed rather than changing path structure. The looser search-oriented DOI normalizer is not used at this path boundary.

### Bounded path templates

Existing `DispatchIntent` remains unchanged. `RoutePolicy` gains an optional closed path template whose only slot kinds are `date`, `uint`, `doi_registrant`, and `doi_suffix`, plus an optional numeric pagination segment index. Template literals, ordered slot kinds/bounds, and the pagination index participate in the immutable policy digest. Exact-path and template policies are mutually exclusive. Foundation policies omit this structure and keep their existing digests and canonical plan bytes.

The planner validates typed values and renders one canonical concrete `DispatchIntent.path`. The gateway splits that raw ASCII path before decoding, requires the exact template segment count, strict-decodes each dynamic segment once, and rejects malformed escapes, controls, empty values, slash/backslash/percent after decoding, dot segments, and any second encoded form. It full-match validates dates, bounded unsigned cursors, and strict DOI parts, re-renders every segment with the one canonical encoder, and requires byte-for-byte equality with the supplied path before transport. No later layer may normalize the checked target.

Interval pagination reuses the sealed `NumericCursor`. The executor replaces exactly the policy-declared path segment, tracks initial and subsequent cursors, and sends every mutation through the full gateway validation and accounting path. Query, JSON-body, and path pagination channels are mutually exclusive. The adapter derives the next cursor from the response's validated current cursor and count; it does not assume a fixed page size. DOI lookup is single-page and has no cursor mutation.

### Response validation and attribution

The adapter validates HTTP status and the application-level `messages[0].status`. Numeric metadata may arrive as bounded canonical numeric strings or integers. The response must have the expected collection shape.

Every retained item must include an exact `server` value matching the requested route. API hostname and DOI prefix are never used for source attribution. A missing or mismatched server fails attribution.

DOI lookup requires every retained item to have the exact requested canonical DOI. Interval responses must echo the exact requested interval, category, and current cursor; `count` must equal collection length; `cursor + count` must not exceed `total`; every retained date must fall inside the inclusive interval; and category-filtered results must match the requested normalized category. A zero-count page with remaining total, a repeated cursor, or inconsistent termination is malformed. These checks bind provider content to the typed request before normalization or attribution.

The adapter retains bounded title, authors, abstract/snippet, DOI, date, version, license, category, publication linkage, and inert links synthesized from the validated DOI onto exact allowlisted hosts. It drops `jatsxml`, raw provider payloads, provider-supplied landing/full-text URLs, and unknown fields; `pdf_url` remains unset.

When one DOI has multiple versions, the logical attempt retains the highest validated numeric version deterministically before canonical DOI deduplication. Same-version conflicting metadata is schema drift and fails the logical attempt rather than silently choosing a record.

## Planner, Coalescing, and Failure Semantics

- Only routes compatible with the typed query mode enter the plan.
- The two Europe PMC routes have different immutable publisher clauses and route IDs, so the existing coalescing key cannot merge or cross-attribute them; no family-specific coalescer is added.
- Details routes share one physical backend for fairness and accounting, but different server paths remain distinct physical requests.
- No general route falls back to a recent scan, and no structured route falls back to general search.
- Pagination is atomic per logical route attempt. A malformed later page contributes no partial records from that route.
- Failure of one selected source does not discard independently successful sources. Partial failure is reported through the existing logical outcomes and usage accounting.
- Cancellation stops further pages and retries through the existing cooperative executor boundary.
- All six routes set `max_redirects=0`; relative, absolute, same-origin, and cross-origin redirects are terminal provider failures.

## Shadow Composition and Compatibility

`foundation_registry()` and `foundation_gateway_adapters()` remain the exact eight-source/seven-adapter compatibility oracle.

One family module owns the routes, backends, adapters, and immutable shadow composition. It:

- starts from the foundation registry without mutating it;
- appends the two source definitions, six enabled routes, and two physical backend definitions;
- composes the two family adapters while rejecting duplicate adapter IDs;
- supplies an explicit offline-fixture or synthetic readiness overlay;
- is not imported by current Search or Deep Research consumers.

The composed registry uses task-specific, distinct catalog and registry versions. Because each `SourceDefinition.catalog_version` must equal the enclosing registry catalog version, the factory reconstructs foundation source values with `dataclasses.replace(..., catalog_version=shadow_catalog_version)` before tuple concatenation. It does not reuse a foundation version for different contents and does not mutate the objects returned by `foundation_registry()`. The readiness overlay also receives a distinct shadow version.

Foundation plans, recorded fixtures, adapter identities, legacy source selection, and production network behavior remain unchanged. TASK-12968.3 owns standalone consumer cutover; TASK-12968.4 owns Deep Research cutover.

## Inventory Reconciliation

Frozen rows `sourclip-2026-07-13-0021` and `sourclip-2026-07-13-0022` retain their canonical targets and declared surfaces, with these corrections:

- infeasible native site-search candidates are removed from the viable `route_candidates` array; their 2026-07-15 robots/Cloudflare evidence and revisit rule remain in the row's resolution/evidence fields;
- Europe PMC aggregator candidates become the planned general-query routes;
- direct lookup and interval candidates use the shared `biorxiv_details_api` backend identity;
- unreliable recent shortcuts are removed from the viable details candidates and remain documented in row evidence; no inventory-schema extension is introduced merely to attach runtime readiness to a rejected candidate;
- implementation and fixture state may advance in this task, but live/surface certification remains not run and empty;
- generated manifest/digest artifacts are refreshed only through the authoritative inventory tooling.

## Test and Evidence Matrix

Sanitized fixtures and deterministic tests cover:

### Europe PMC

- known non-empty bioRxiv and medRxiv success;
- valid empty result;
- rate limit with and without valid retry metadata;
- timeout and cancellation;
- malformed JSON/schema drift;
- forged/missing fixed query values, provider-language injection, and oversized page size;
- ignored/hostile `nextPageUrl`;
- missing, unknown, or mismatched `source`/publisher attribution;
- HTML-to-text normalization and field-size limits;
- one aggregator failure alongside another source's success.

### Details API

- known DOI lookup for both sources;
- explicit interval and category success;
- valid empty interval;
- page two using response-derived cursor/count;
- string/integer message count variants;
- application-level error inside HTTP 200;
- requested DOI, echoed interval/category/cursor, collection count, result date, and result category mismatches;
- multiple DOI versions and same-version conflict;
- missing/mismatched server attribution;
- malformed DOI/date/path inputs, encoded separators, and traversal attempts;
- rate limit, timeout, cancellation, malformed JSON, and partial failure;
- proof that JATS/full-text links are dropped and never dereferenced.

### Compatibility and security

- foundation registry, adapter factory, and canonical plans remain exact;
- raw-string planning retains current behavior;
- each typed mode selects only compatible family routes;
- every enabled family request crosses the shared gateway and is physically accounted;
- no direct networking imports, client construction, browser, cookies, credentials, Media calls, or legacy helper calls enter the family;
- network-boundary closure includes the new family modules and their reviewed digests;
- legacy Search and Deep Research execute zero additional requests.

Only distinct valid provider shapes are stored as fixtures. Error, attribution, cursor, injection, cancellation, and conflict cases are derived with parameterized fixture mutation or gateway stubs rather than one checked-in file per branch.

Live feasibility evidence records the exact official endpoints, bounded synthetic queries, observation date, HTTP/application outcome, returned attribution fields, and zero result-link dereferences. It does not claim surface certification.

## Rollout and Revisit Rules

1. Implement contracts and pure-planner selection with foundation compatibility tests.
2. Add and fixture-certify Europe PMC general routes.
3. Add and fixture-certify details lookup/interval routes.
4. Reconcile the shadow registry and inventory rows.
5. Run focused/full Research tests, formatting/linting, Python 3.10 syntax checks, Bandit, network-boundary checks, and diff hygiene.
6. Keep all new routes out of production consumers until TASK-12968.3 and TASK-12968.4 perform their own surface certification.

Native site search may be reconsidered only if robots policy permits automated search, the route works without cookies/browser state, provider terms are reviewed, and a bounded HTML adapter passes the gateway/security gates. Recent shortcuts may be enabled only after repeated live probes demonstrate current, monotonic semantics. A local metadata or full-text index requires a separate persistence, backfill, freshness, and operations design.
