# ClinicalTrials.gov and PubMed Central Shared-Discovery Route Family

**Task:** TASK-12968.6

**Parent:** TASK-12968

**Status:** Requester-approved design; 2026-08-21 independent architecture and security/provider re-reviews cleared; implementation-plan ready

**Execution scope:** Shadow registry and fixture execution only

## Goal

Add stable `clinicaltrials_gov` and `pubmed_central` targets to the shared research-discovery pipeline through credentialless, source-native APIs. The family must provide useful general discovery, strict source attribution, bounded physical work, and inert metadata links without changing current Standalone Search or Deep Research behavior.

This document narrows TASK-12968's approved program architecture for the second provider-family slice. The shared gateway, immutable plans, executor accounting, no-dereference boundary, consumer-cutover ownership, and authenticated-retrieval deferral remain unchanged.

## Decision Summary

The implementation will:

1. add one shadow-only family module alongside `biorxiv_medrxiv.py`;
2. use ClinicalTrials.gov API v2 `GET /api/v2/studies` for direct general trial discovery;
3. extend the existing `RoutePolicy.pagination_query_key` channel with one closed opaque query-policy and cursor type for ClinicalTrials.gov's `pageToken`/`nextPageToken` continuation;
4. use NCBI E-utilities ESearch followed by conditional ESummary with `db=pmc` for PubMed Central general discovery, with the exact public product identity `tool=tldw_server` and `email=contact@tldwproject.com` sealed into both intents;
5. reuse the existing PubMed two-dispatch machinery through the smallest private parameterization, preserving the current foundation route exactly while adding one identity-bearing PubMed shadow-overlay version for later production cutover;
6. keep PMC continuation to one bounded ESearch/ESummary page in this slice while validating the returned pagination envelope;
7. update the frozen inventory and shadow artifacts without importing the family from any production consumer;
8. redact query values, deferred numeric binding values, and built request targets from diagnostic `repr` paths while retaining their deterministic plan/digest material; and
9. leave process-wide NCBI pacing, proof that the frozen identity was registered, required notices, long-query routing, and surface certification to TASK-12968.3.

The family adds no dependency, SDK, browser, scraper, persistent cache, background job, or provider-specific HTTP client.

## Provider Evidence and Constraints

### ClinicalTrials.gov

The [modern ClinicalTrials.gov API](https://clinicaltrials.gov/data-api/api) exposes an OpenAPI-described v2 JSON service. The [official migration guide](https://clinicaltrials.gov/data-api/about-api/api-migration) identifies `/api/v2/studies` as the modern search endpoint, maps the classic search expression to `query.term`, and documents paginated responses with a maximum provider page size of 1,000.

The route uses opaque continuation tokens returned by the provider. They are not numeric offsets and must not be coerced, ordered, normalized, interpreted, or copied into any location except the policy-declared `pageToken` query value for the same exact route. The local 1,024-character printable-ASCII token ceiling in this design is a fail-closed application limit, not a provider guarantee; rejecting a future otherwise valid token is reported as provider-contract drift.

ClinicalTrials.gov study records are structured documents, not publications. The adapter therefore does not invent authors, publication dates, DOIs, or PDFs. It retains a bounded discovery projection: NCT identifier, title, plain-text brief summary, status, conditions, interventions, study type, sponsor, relevant study dates, results availability, and an inert synthesized study link.

The ClinicalTrials.gov terms gate any later user-visible distribution on source attribution, the provider processing date, disclosure of modifications, and a currentness policy. This task checks in synthetic response shapes only. Any bounded observation records local observation time separately from `/api/v2/version.dataTimestamp` and retains no study values. TASK-12968.3 owns user-visible presentation, refresh policy, and currentness enforcement.

### PubMed Central

The [NCBI E-utilities parameter reference](https://www.ncbi.nlm.nih.gov/books/NBK25499/) defines ESearch pagination through `retstart` and `retmax` and JSON output through `retmode=json`. PubMed Central discovery uses ESearch and ESummary with the database fixed to `pmc`; it does not substitute PMC OAI date feeds, OA package retrieval, HTML search, or article dereference for general search.

The [NCBI usage guidance](https://www.ncbi.nlm.nih.gov/books/NBK25497/) limits unkeyed clients to three requests per second and raises the usual ceiling for approved API-key use. It asks programmatic clients to identify themselves with `tool` and `email`, and requires the NCBI disclaimer/copyright notice to be evident to software users. The public, non-user identity is frozen as `tool=tldw_server` and `email=contact@tldwproject.com`, matching the repository's public project metadata. Both values are declared query pairs in every new identity-bearing ESearch and ESummary intent; they are not credentials and no user input can override them. TASK-12968.3 must record NCBI registration evidence for this exact pair before enabling live use. This shadow task validates bounded per-execution request accounting and typed rate-limit failures but does not implement process-wide pacing, an API-key route, or a user-visible notice.

Implementation and ordinary verification for PMC are fixture-only. No new live E-utilities call is allowed in TASK-12968.6. The historical 2026-07-15 feasibility observation remains a non-certifying schema note; it does not authorize a repeat or retain provider records. TASK-12968.3 owns registration evidence for the already frozen non-user identity, the shared per-origin limiter, required notice, bounded long-query policy, and production enablement before any later live or user-facing use.

ESummary for `db=pmc` provides citation metadata and identifiers but not a guaranteed abstract. This slice always leaves `abstract` and `snippet` unset, ignores any uncontracted summary-like field, and does not add EFetch as an adapter-internal third call.

## Scope

This task includes:

- two canonical catalog targets with Standalone Search and Deep Research declared as intended surfaces;
- one direct ClinicalTrials.gov general-query route;
- one direct PubMed Central general-query route;
- exact backend, route, adapter, source, and policy identities;
- one shadow-only identity-bearing replacement of the existing foundation PubMed route, leaving `foundation_registry()` and every identityless foundation plan byte unchanged;
- a closed opaque query-cursor contract used only when declared by route policy;
- minimal planner generalization for a literal-term policy named `query.term`, an empty immutable suffix, and one optional opaque pagination query pair;
- strict bounded JSON parsing, normalization, attribution, deduplication, and inert links;
- diagnostic redaction for query values, deferred numeric CSV binding values, gateway binding pairs, and normalized HTTP targets without changing equality, hashing, canonical plan bytes, or policy digests;
- gateway-only physical requests, request accounting, cancellation, and partial-failure behavior;
- sanitized fixtures, deterministic tests, inventory reconciliation, the existing bounded feasibility notes, and no new live PMC call; and
- compatibility proofs for the foundation registry, existing PubMed adapter, legacy selection, and both production consumers.

This task does not include:

- Standalone Search or Deep Research cutover;
- surface-ready or live-certified claims;
- a process-wide NCBI limiter, API-key route, NCBI registration operation, or user notice;
- PMC EFetch, abstracts, full text, JATS, PDFs, OA packages, OAI harvesting, or HTML retrieval;
- ClinicalTrials.gov documents, attachments, raw markup, complete study records, or bulk download;
- result-link dereference, Media ingestion, persistence, caching, or scheduled backfills;
- cookies, credentials, browser state, login automation, or authenticated scraping; or
- a generic provider DSL or speculative cursor types for unrelated APIs.

## Stable Runtime Identities

### Catalog targets

| Source ID | Display name | Site host | Declared surfaces |
| --- | --- | --- | --- |
| `clinicaltrials_gov` | ClinicalTrials.gov | `clinicaltrials.gov` | `standalone_search`, `deep_research` |
| `pubmed_central` | PubMed Central | `pmc.ncbi.nlm.nih.gov` | `standalone_search`, `deep_research` |

The exact `SourceDefinition` values are frozen:

| Source ID | Aliases | Categories | Content types | Route reference | Priority |
| --- | --- | --- | --- | --- | --- |
| `clinicaltrials_gov` | `clinical_trials_gov`, `clinical_trials` | `biomedical`, `clinical_trials` | `clinical_trials`, `study_records`, `summaries` | `clinicaltrials_gov_studies_search_direct`, predicate `None` | `110` |
| `pubmed_central` | `pmc`, `pub_med_central` | `biomedical`, `open_access` | `papers`, `full_text_archive`, `biomedical_metadata` | `pubmed_central_esearch_summary_direct`, predicate `None` | `120` |

Both definitions use catalog version `research-discovery-v2-clinicaltrials-pmc-shadow` and the exact surface tuple `('standalone_search', 'deep_research')`.

### Physical backends

| Backend ID | Physical service | Exact origin |
| --- | --- | --- |
| `clinicaltrials_gov_api_v2` | ClinicalTrials.gov API v2 | `https://clinicaltrials.gov:443` |
| `ncbi_eutils_pmc` | NCBI Entrez E-utilities for PMC | `https://eutils.ncbi.nlm.nih.gov:443` |

The PMC site host is a catalog identity and inert result-link host. All PMC network execution in this task uses only the E-utilities origin.

### Routes and adapters

| Route ID | Target | Kind | Query modes | Backend | Adapter | Shadow readiness |
| --- | --- | --- | --- | --- | --- | --- |
| `clinicaltrials_gov_studies_search_direct` | ClinicalTrials.gov | `direct` | `general_free_text` | `clinicaltrials_gov_api_v2` | `clinicaltrials_gov_v2` | fixture-ready after this task |
| `pubmed_central_esearch_summary_direct` | PubMed Central | `direct` | `general_free_text` | `ncbi_eutils_pmc` | `pubmed_central_v2` | fixture-ready after this task |

Both new routes freeze `source_constraint=native_corpus`, `credential_requirement=none`, and `fallback_order=0`. ClinicalTrials.gov freezes `attribution_basis=native_nct_record`; PMC freezes `attribution_basis=ncbi_pmc_database`. Each route allows two physical dispatches. The family registry also replaces only `pubmed_ncbi_eutils_pubmed_direct` with a shadow-overlay copy that retains its source, backend, adapter ID, route kind, query modes, source constraint, attribution basis, credential requirement, fallback order, limits, and physical allowance while changing only its adapter/policy version and exact NCBI identity query shape. The overlay policy freezes `allowed_query_keys=('db', 'term', 'retstart', 'retmax', 'retmode', 'sort', 'datetype', 'mindate', 'maxdate', 'tool', 'email', 'id')`, `pagination_query_key='retstart'`, and `query_value_policies=()`; this full ordered tuple is canonical-policy-digest material.

Identifier lookup, ClinicalTrials.gov bulk downloads, PMC OAI/date feeds, OA package lookup, EFetch, and full-text retrieval remain inventory capabilities or future route candidates. They are not enabled general-search routes and cannot satisfy this task by proxy.

### Version identities

| Contract | Frozen value |
| --- | --- |
| Shadow catalog | `research-discovery-v2-clinicaltrials-pmc-shadow` |
| Shadow registry | `research-discovery-v2-clinicaltrials-pmc-shadow-2026-08-21` |
| Shadow readiness | `research-discovery-readiness-v2-clinicaltrials-pmc-shadow` |
| Route policy | `research-discovery-route-policy-v2-clinicaltrials-pmc` |
| ClinicalTrials.gov adapter | `clinicaltrials-gov-v2` |
| PubMed Central adapter | `pubmed-central-v2` |
| Identity-bearing PubMed overlay policy | `research-discovery-route-policy-v2-foundation-pubmed-ncbi-identity-2026-08-21` |
| Identity-bearing PubMed overlay adapter | `pubmed-v2-ncbi-identity` |
| Shared planner | existing `research-discovery-planner-v2-foundation` (unchanged) |
| NCBI JSON envelope | existing `0.3` (unchanged) |

Changing one of these values after fixtures or policy snapshots are committed requires an explicit contract migration rather than an in-place rename.

## Shadow Composition

`clinicaltrials_pubmed_central.py` owns the family registry additions, readiness overlay, adapters, strict family parsers, and immutable adapter map. It follows the established bioRxiv/medRxiv composition pattern with one explicit NCBI compliance overlay:

- begin with a fresh `foundation_registry()` value;
- reconstruct foundation sources with the family shadow catalog version;
- replace exactly the foundation PubMed `AccessRoute` in this family registry with the frozen identity-bearing overlay version;
- append two source definitions, two routes, and two backend definitions;
- reconstruct the foundation readiness tuple, replace the PubMed entry deliberately for the exact overlay route, and append fixture-ready family route entries only after their corresponding runtime fixtures pass;
- compose only the two family adapters while rejecting duplicate adapter IDs; and
- expose no default import or consumer wiring.

The family receives distinct catalog, registry, readiness, route-policy, and adapter versions. `foundation_registry()`, `foundation_readiness()`, the existing `foundation-v2` PubMed adapter version, and every foundation canonical plan remain unchanged. Because `RouteReadiness` carries only a route ID, the family readiness constructor must not accidentally inherit PubMed readiness by ID: it explicitly replaces that entry and a reconciliation test binds the ready ID to the family registry's exact `pubmed-v2-ncbi-identity` adapter version and overlay policy version. The planner and shared PubMed callable accept the new overlay only through its exact route/backend/adapter/policy-version tuple; partial identity matches fail closed. A later consumer must select the overlay registry and may not enable the identityless foundation PubMed route.

## General Query Contract

Both routes use the existing exact `GeneralFreeTextQuery` planning type. User input is normalized into a bounded sequence of Unicode alphanumeric terms before provider rendering. Raw provider syntax, punctuation, field operators, and control characters are never copied outbound.

`LiteralTermsQueryValuePolicy` already accepts any syntactically valid query-policy name. The hard-coded `name == "query"` restriction exists only in `_build_typed_intents`; this task removes that planner restriction. The policy constructor separately gains one narrow capability: `fixed_suffix` may be the empty string. The gateway treats the literal expression as the whole value when the suffix is empty rather than applying Python's `value[:-0]` slice. Existing nonempty-suffix policies retain identical constructor order, digest material, intents, and plan bytes.

The shared literal renderer still permits exactly one literal-term policy for a general route and emits normalized quoted terms joined by literal ` AND `. A `BoundedDecimalQueryValuePolicy` used as a typed result/page-size field renders `min(request.result_limit, route.policy.limits.max_results, policy.maximum)`; this is pinned with compatibility tests so existing routes are byte-identical while ClinicalTrials.gov can request 50 records per page under a 100-result route ceiling. The ClinicalTrials.gov policy further narrows accepted input to at most eight terms of 32 Unicode-alphanumeric characters each.

For a generic one-path `general_free_text` route, every allowed query key must be covered by one closed policy. The planner may omit exactly one first-page query pair only when all of the following are true:

- it is `OpaqueCursorQueryValuePolicy(required=False)`;
- its name equals `RoutePolicy.pagination_query_key`; and
- the first-page intent contains no query pair with that name.

Every other missing, optional, duplicate, unknown, or uncovered key fails planning. Query pairs remain in `allowed_query_keys` order with the one first-page cursor gap.

PMC is the only two-path typed-general exception in this task. A closed planner branch matches the exact tuple `pubmed_central_esearch_summary_direct` / `ncbi_eutils_pmc` / `pubmed_central_v2` / `pubmed-central-v2`; any partial identity match or policy-shape drift fails planning. It emits, in order:

1. `SEARCH /entrez/eutils/esearch.fcgi` with query pairs `db=pmc`, `term=<quoted literal terms>`, `retstart=0`, `retmax=<bounded limit>`, `retmode=json`, `tool=tldw_server`, and `email=contact@tldwproject.com`;
2. `CONDITIONAL_SUMMARY /entrez/eutils/esummary.fcgi` with query pairs `db=pmc`, `retmode=json`, `tool=tldw_server`, and `email=contact@tldwproject.com`, plus exactly one deferred numeric CSV binding `pmc_esearch_ids` for query key `id`, at most 100 items, and at most 16 decimal characters per item.

`sort` is intentionally absent for PMC because the official E-utilities reference says supported sort values vary by database and does not establish `relevance` for `db=pmc`. Provider-returned ordering is treated as opaque. The legacy identityless PubMed planner branch remains byte-exact. One additional closed branch recognizes only the identity-bearing PubMed overlay version and appends the same frozen `tool`/`email` pair to both of its existing ESearch and ESummary shapes. Full first-page plan tuples for ClinicalTrials.gov, PMC, and the PubMed overlay, including the absence of `pageToken`, are golden fixtures. No general provider DSL or configurable intent builder is introduced.

## Closed Opaque Query Pagination

### Why a new cursor type is required

The current executor supports only bounded nonnegative numeric cursors. Treating a ClinicalTrials.gov token as an integer, URL, arbitrary intent mutation, or adapter-owned direct request would either fail valid continuation or bypass the sealed plan and gateway. The correct extension is one closed opaque query-cursor channel.

### Contract

Add:

- `OpaqueCursorQueryValuePolicy(name, max_chars, required=False)` to the closed query-policy union;
- `OpaqueCursor(value)` to the closed dispatch cursor union, with `value` excluded from `repr`.

For this route, `max_chars` is 1,024. An opaque token must be an exact nonempty ASCII string containing only visible bytes `0x21` through `0x7e`. Whitespace, controls, DEL, Unicode, empty values, and oversized values fail closed. The token is never Unicode-normalized, decoded, parsed, sorted, logged, persisted as result metadata, or used as a URL/path. This local ceiling is deliberately narrower than the earlier draft so the worst-case percent-encoded token, worst-case eight-by-32 four-byte Unicode query, and every frozen query pair fit the existing 8,192-byte normalized request-target ceiling. A construction test proves that aggregate bound; a contract-valid route input must never fall through to a generic target-size rejection.

Do not add a second route-policy field. The route reuses `RoutePolicy.pagination_query_key="pageToken"`; the closed policy type for that key discriminates opaque from numeric query pagination. Existing numeric query, integer JSON-body, and numeric path pagination channels remain mutually exclusive through the existing constructor shape and digest material.

The first intent must omit `pageToken`. On continuation, the executor:

1. accepts only an exact `OpaqueCursor` for an opaque policy;
2. inserts the absent policy-declared query pair at its `allowed_query_keys` position, or replaces exactly one already-bound pair on a later continuation;
3. rejects numeric/opaque type mismatches;
4. tracks exact seen tokens and rejects repetition;
5. submits the reconstructed intent through the normal policy snapshot, digest, gateway validation, journal reservation, physical accounting, timeout, and cancellation path.

Opaque tokens have no ordering relationship, so only exact-repeat progress checks apply. Existing numeric query/body/path pagination retains its current type checks and monotonic path-cursor behavior.

The gateway independently validates the opaque policy and then percent-encodes the exact token as a query value for the already-bound exact origin and path. Provider punctuation therefore cannot add a query key, alter the path, change the host, or create a fragment.

The new query-value policy participates in the policy digest and immutable policy snapshot. `RoutePolicy` gains no field, so its full positional construction and all legacy digest inputs stay unchanged; policies that do not use the new query-policy type retain exact digest bytes. `QueryPair.value`, `NumericCSVBindingValues.values`, the gateway binding's raw query-pair field, and `NormalizedHTTPHopRequest.target` become `repr=False`; this changes diagnostic representation only, not equality, hashing, canonical serialization, dispatch-group identity, deferred grounding, or wire construction. Gateway traces retain query keys only. Tests prove the token is absent from cursor/effective-intent/binding/request `repr`, public traces, callback diagnostics, outcomes, usage, attempt-journal state, and exception text; identity-bearing NCBI execution also proves deferred numeric IDs are absent from binding diagnostics while `asdict()` and exact wire grounding remain unchanged. The sole raw-token assertion is inside a one-hop transport stub that proves the exact token is one percent-encoded wire query value.

## Exact Bounded Contracts

| Contract | ClinicalTrials.gov | PubMed Central |
| --- | --- | --- |
| `RouteLimits.max_pages` | `2` | `1` |
| Request page size | `pageSize <= 50` | `retmax <= 100` |
| `RouteLimits.max_results` | `100` | `100` |
| `AccessRoute.max_physical_dispatches` | `2` | `2` |
| `RouteLimits.timeout_ms` | `20_000` per physical hop | `20_000` per physical hop |
| `RouteLimits.max_response_bytes` | `2_097_152` per response | `2_097_152` per response |
| `RouteLimits.max_request_body_bytes` | `16_384` (no body is sent) | `16_384` (no body is sent) |
| Redirects / retries | `0 / 0` | `0 / 0` |
| Parser `max_input_bytes` | `2_097_152` | `2_097_152` |
| Parser `max_records` | `50` per page | `100` |
| Parser depth / nodes | `16 / 50_000` | `16 / 50_000` |
| Parser max string / numeric token | `65_536 / 32` chars | `65_536 / 32` chars |
| Parser deadline | `500 ms` per response | `500 ms` per response |

The 50-record ClinicalTrials.gov page cap is intentionally lower than its 100-result route ceiling so a two-page fixture exercises real continuation. The PMC two-intent route has one logical result page but two separately reserved and accounted physical requests. Each route therefore contributes 40,000 ms to planned aggregate wall-time allowance, and a plan selecting both contributes 80,000 ms before other routes. Actual execution also remains subject to the plan-wide deadline. No new group-level timeout is introduced. Zero retries and redirects mean every physical dispatch is visible in the sealed plan allowance.

## ClinicalTrials.gov Route

### Request shape

- Method: `GET`
- Path: `/api/v2/studies`
- Ordered policy keys: `query.term`, `format`, `markupFormat`, `fields`, `pageSize`, `countTotal`, `pageToken`
- General expression: `query.term=<quoted literal terms>`
- Fixed response format: `format=json`
- Fixed markup representation: `markupFormat=legacy`
- Fixed field projection: `fields=NCTId,BriefTitle,OfficialTitle,BriefSummary,OverallStatus,Condition,InterventionName,LeadSponsorName,StudyType,StartDate,CompletionDate,HasResults`
- Bounded page size: `pageSize=min(request.result_limit,100,50)`
- Fixed total-count request: `countTotal=true`
- Optional continuation: `pageToken`, absent on page one
- Redirects: disabled
- Credentials, cookies, and custom user headers: absent

The route declares `pagination_query_key='pageToken'` and the exact ordered policies `LiteralTermsQueryValuePolicy('query.term', '', 8, 32)`, `ExactQueryValuePolicy('format', 'json')`, `ExactQueryValuePolicy('markupFormat', 'legacy')`, `ExactQueryValuePolicy('fields', <frozen projection>)`, `BoundedDecimalQueryValuePolicy('pageSize', 50)`, `ExactQueryValuePolicy('countTotal', 'true')`, and `OpaqueCursorQueryValuePolicy('pageToken', 1_024, required=False)`.

The exact first-page plan is pinned byte-for-byte, including `format=json`, `markupFormat=legacy`, and the absence of `pageToken`. The field projection is frozen in both route policy and adapter validation. A bounded historical observation on 2026-07-15 confirmed only the expected schema modules, `totalCount`, `studies`, and opaque-token shape; it retains no real study values or raw response and is non-certifying. The planner cannot widen the request, and provider-supplied links or document fields are never retained.

The adapter follows `nextPageToken` only while another page, another physical dispatch, and raw-result capacity remain. A token returned after any ceiling is discarded and causes no reservation or hidden request. `DiscoveryAdapterResult` currently carries candidates only, so TASK-12968.6 makes no claim that an unused token increments the executor's existing truncated-candidate counter or exposes an explicit `more_available` flag; TASK-12968.3 may add a separately reviewed surface signal if product requirements need one.

### Response validation

Each page must be one strict JSON object containing a nonnegative JSON-integer `totalCount`, a `studies` array no larger than the requested `pageSize`, and an optional `nextPageToken`. Because the request fixes `countTotal=true`, a missing or invalid count is schema drift. The first valid `totalCount` is frozen; every later page must repeat the same value. Cumulative raw studies are counted before deduplication and may not exceed the frozen count or the 100-result ceiling. Token/count consistency is biconditional: `nextPageToken` is required while cumulative raw studies remain below `totalCount`, even when a local ceiling means it will be discarded, and it must be absent exactly when cumulative raw studies equal `totalCount`.

Each retained study must contain one canonical identifier matching `NCT[0-9]{8}` and one nonempty bounded brief or official title. Duplicate NCT IDs with identical normalized records collapse deterministically only after raw-page accounting; conflicting records for one NCT ID are schema drift.

Nested objects and arrays are subject to the exact parser profile above. The adapter validates only the frozen projection and drops unknown fields. A valid first-page empty result is exactly `totalCount=0`, `studies=[]`, and no token. Positive-count empty pages, premature no-token terminal pages, token-after-total pages, a page larger than `pageSize`, changed counts, malformed tokens, missing required envelopes, wrong scalar types, invalid identifiers, oversized required fields, and inconsistent duplicate records fail as provider payload errors.

Provider HTTP 429 and retry metadata use the existing typed rate-limit path. Timeouts, cancellation, non-JSON responses, unexpected redirects, 4xx/5xx responses, and parse-deadline exhaustion use existing gateway/adapter outcomes.

### Normalized record

ClinicalTrials.gov documents `BriefTitle` and `OfficialTitle` as plain text and `BriefSummary` as markup. The route requests `markupFormat=legacy`. Titles use bounded Unicode-whitespace normalization and reject controls, NUL, surrogates, markup, and any URL/URI token. Only `BriefSummary` uses a bounded standard-library legacy-markup data-node parser that converts character references, drops tags/comments/declarations, rejects controls/NUL/surrogates, collapses Unicode whitespace, and rejects any residual URL/URI token. A family-local bounded detector recognizes case-insensitive `http://`, `https://`, `ftp://`, `www.`, `mailto:`, `data:`, and `javascript:` material before provider text is retained; the existing unsafe-URL helper remains defense-in-depth but is not treated as an all-URL detector. CommonMark links, images, autolinks, embedded HTML, entity edge cases, plain safe-looking HTTPS URLs, and controls are explicit hostile fixtures; an unsafe optional summary is dropped rather than copied.

The exact ClinicalTrials.gov input rules are:

| Source path | Expected shape and bound | Requiredness | Invalid/over-bound behavior |
| --- | --- | --- | --- |
| `protocolSection.identificationModule.nctId` | string `NCT[0-9]{8}` | required | fail route |
| `protocolSection.identificationModule.briefTitle` | plain string, max 1,024 | at least one title | fail if it is the only title; otherwise drop |
| `protocolSection.identificationModule.officialTitle` | plain string, max 4,096 | at least one title | fail if it is the only title; otherwise drop |
| `protocolSection.descriptionModule.briefSummary` | legacy-markup string, sanitized max 16,384; derived snippet max 1,024 | optional | drop field on unsafe/over-bound content; fail on wrong container/scalar type |
| `protocolSection.statusModule.overallStatus` | string, max 256 | optional | drop over-bound; fail wrong type |
| `protocolSection.conditionsModule.conditions` | list, max 64 strings of 512 | optional | drop whole field when over-bound; fail malformed member/container |
| `protocolSection.armsInterventionsModule.interventions[].name` | list, max 64 names of 512 | optional | drop whole field when over-bound; fail malformed member/container |
| `protocolSection.sponsorCollaboratorsModule.leadSponsor.name` | string, max 1,024 | optional | drop over-bound; fail wrong type |
| `protocolSection.designModule.studyType` | string, max 256 | optional | drop over-bound; fail wrong type |
| `protocolSection.statusModule.startDateStruct.date` and `.completionDateStruct.date` | ClinicalTrials.gov `PartialDate`: calendar-valid `YYYY`, `YYYY-MM`, or `YYYY-MM-DD` | optional | drop invalid value; fail wrong container/scalar type |
| top-level `hasResults` | JSON boolean | optional | fail wrong type |

The exact normalized record retains:

- canonical NCT ID and `provider_ids.nct_id`;
- top-level `title` chosen as brief title then official title;
- sanitized bounded brief summary as `abstract` and a bounded derived `snippet` when present;
- `authors=()`, `doi=None`, `pmid=None`, `pmcid=None`, `arxiv_id=None`, and `pdf_url=None`;
- `provider='clinicaltrials_gov'` and a `source_metadata` mapping with only bounded `brief_title`, `official_title`, `overall_status`, `conditions`, `interventions`, `lead_sponsor`, `study_type`, `start_date`, `completion_date`, and `has_results` keys when present; and
- `https://clinicaltrials.gov/study/{NCT_ID}` synthesized solely from the validated identifier.

Raw markup, arbitrary URLs, contacts, locations, documents, references, and unknown modules are dropped. Checked-in response fixtures are entirely synthetic shape fixtures with no copied study values, summaries, or real NCT IDs. Historical notes retain schema facts only. Any optional live check records local observation time separately from the provider's `/api/v2/version.dataTimestamp`, never substitutes one for the other, retains no study payload, and records that the observed output was modified into a bounded projection.

The fingerprint uses the validated normalized record and stable NCT provider identity. Source attribution is route-constrained to `clinicaltrials_gov`; no provider field can force another logical source.

## PubMed Central Route

### Planned physical work

The route plans exactly two immutable intents:

1. ESearch: `GET /entrez/eutils/esearch.fcgi`
2. Conditional ESummary: `GET /entrez/eutils/esummary.fcgi`

Its exact `RoutePolicy` has method `GET`, paths `('/entrez/eutils/esearch.fcgi', '/entrez/eutils/esummary.fcgi')`, allowed query keys `('db', 'term', 'retstart', 'retmax', 'retmode', 'tool', 'email', 'id')`, and `pagination_query_key='retstart'`. It has no generic `query_value_policies` because one total-coverage policy set cannot represent two different conditional path shapes; the exact typed planner branch and private NCBI adapter validator independently freeze every value and binding instead.

ESearch fixes `db=pmc`, `retmode=json`, `retstart=0`, `retmax=min(request.result_limit,100)`, `tool=tldw_server`, and `email=contact@tldwproject.com`. ESummary fixes `db=pmc`, `retmode=json`, and the same identity pair; its `id` value is grounded only from the validated positive numeric ESearch UID list through the exact `pmc_esearch_ids` deferred numeric CSV binding.

A valid empty ESearch result performs no ESummary reservation or physical request. Nonempty execution accounts for both calls separately. No SDK, EFetch, hidden retry, hidden page, or third call is permitted.

This task keeps PMC to one bounded result page, matching the existing PubMed foundation contract. It validates `count`, `retstart`, `retmax`, ID cardinality, and whether additional results exist but does not continue to another ESearch/ESummary pair. An unfetched remainder causes no hidden request and no unsupported truncation-count claim. Real continuation in the family is exercised by ClinicalTrials.gov. Repeated two-dispatch pagination is deferred until a consumer requirement justifies a reviewed executor contract for reusable conditional intents.

### Minimal NCBI reuse

The existing PubMed adapter already owns strict NCBI root/version checks, diagnostic-list handling, numeric UID validation, conditional CSV grounding, response checking, parse guards, and two-dispatch accounting. Extract the smallest private ESearch/ESummary execution helper parameterized by:

- adapter and route identity;
- exact database value;
- deferred binding identity and item bounds;
- ESearch-ID and ESummary-record normalization callbacks; and
- an exact envelope/error validator selected by adapter version.

The existing `foundation-v2` PubMed wrapper remains one caller with unchanged values, error classification, output, and plan bytes. The `pubmed-v2-ncbi-identity` overlay is an exact second PubMed version with only the declared identity pairs and strict documented rate envelope added; the parsing-profile table explicitly registers this version rather than relying on a permissive fallback. The family PMC wrapper is the third closed call shape with `db=pmc`. The helper may share dispatch sequencing and strict JSON/envelope utilities, but PubMed and PMC retain separate identifier and record parsers. No public adapter framework, class hierarchy, provider registry, or transport abstraction is added.

All existing PubMed fixtures and plan bytes are regression locks for the extraction.

### Response validation and normalization

Before canonical-envelope validation of either ESearch or ESummary, the PMC and identity-bearing PubMed overlay versions recognize only the documented exact-key top-level NCBI rate envelope `{"error":"API rate limit exceeded","count":"<canonical decimal>"}` and emit the existing typed rate-limited outcome even when HTTP status alone is not 429. Malformed lookalikes remain provider-payload failures. The unchanged `foundation-v2` PubMed version retains its legacy error classification exactly.

Otherwise ESearch requires the canonical NCBI JSON header/root and a consistent `esearchresult`. `count`, `retstart`, and `retmax` must be canonical unsigned ASCII decimal strings: zero is exactly `"0"`, positive values have no leading zero, and signs, whitespace, exponent notation, JSON numbers, and over-32-character tokens fail closed. Their parsed values must agree with the requested page and returned `idlist`. Bounded diagnostic lists remain nonfatal only in the already accepted shapes; fatal `ERROR` envelopes and malformed diagnostics fail closed. Returned PMC UIDs must be unique canonical positive decimal strings within the 16-character binding limit.

ESummary requires `uids` to contain exactly the expected ESearch UID set and one UID-keyed record per expected value, with no extra result keys. The adapter restores ESearch ordering rather than trusting provider object order. The ESearch value is the numeric form of the PMC database's PMCID and is used only for transport binding and response correlation.

`articleids` must be a list of at most 64 typed objects. Each object has exactly one bounded `idtype` string and one bounded `value` string; an `id` alias or duplicate/unknown object key is schema drift. These two scalars use a separate identifier validator: exact nonempty strings within their 32/512 bounds, with no whitespace, controls, NUL, surrogates, or angle-bracket markup, but without the human-text any-URL rejection. The PMC parser recognizes only one each of `idtype=pmcid`, `idtype=doi`, and `idtype=pmid`; duplicate or conflicting recognized types fail. Unknown structurally valid identifier types are ignored. A retained record requires a provider-supplied canonical `pmcid` equal to `"PMC" + expected_esearch_uid`; a missing or mismatched value fails, and the adapter never invents a missing PMCID. DOI is optional and canonicalized through the existing DOI validator, including an accepted DOI URL form; it is never copied as raw provider text. A missing PMID or exact sentinel `"0"` is treated as absent; any other present PMID must be a canonical positive decimal of at most 16 characters.

The exact PMC record rules are:

| Raw ESummary field | Expected shape and bound | Requiredness | Invalid/over-bound behavior |
| --- | --- | --- | --- |
| `uid` | canonical positive decimal, max 16, equal expected ESearch value | required | fail route |
| `title` | plain string with normalized Unicode whitespace, max 4,096; controls/markup/URL material forbidden | required | fail route |
| `authors` | list of at most 64 objects, each `name` plain string max 512 with controls/markup/URL material forbidden | optional | fail malformed/over-bound list or member |
| `articleids` | list of at most 64 objects; `idtype` max 32, value max 512 | required | fail malformed/over-bound list or recognized identifier |

Other ESummary citation fields, including journal and date strings, are ignored in this slice rather than retained under an underspecified grammar.

The exact normalized projection contains:

- title;
- bounded author names;
- top-level canonical required `pmcid`, optional canonical `doi` and `pmid`, and `arxiv_id=None`;
- `provider='pubmed_central'` and `provider_ids.pmcid` plus optional `provider_ids.doi` / `provider_ids.pmid`; the numeric ESearch value is not emitted as a second identity;
- `https://pmc.ncbi.nlm.nih.gov/articles/{PMCID}/` synthesized from the validated PMCID; and
- `abstract=None`, `snippet=None`, and `pdf_url=None`.

Missing or mismatched PMCID, conflicting article identifiers, duplicate UIDs, partial summaries, unexpected extra records, oversized values, invalid identifiers, and same-identity conflicting metadata fail the logical route rather than inventing or force-attributing data.

The route is source-constrained to `pubmed_central`. The shared NCBI origin and numeric UID do not authorize attribution to PubMed, GenBank, or another NCBI database.

## Planning, Coalescing, and Compatibility

- Only `GeneralFreeTextQuery` selects these family routes.
- Raw-string legacy planning remains on the existing `structured_query` path and cannot accidentally execute the family.
- The ClinicalTrials.gov and PMC routes have different backends, adapters, policies, and intent shapes, so they never coalesce.
- The PMC route does not coalesce with PubMed because its database, backend identity, route identity, source constraint, and normalized semantics differ.
- Coalescing remains wholly owned by the existing exact dispatch-group key; no family-specific merge logic is added.
- A failure in one selected family route does not discard an independently successful route. Existing logical outcomes report partial failure.
- A malformed later ClinicalTrials.gov page contributes no partial records from that logical route.
- Cancellation prevents subsequent pages and conditional calls through the existing executor boundary.
- Foundation registry values, the `foundation-v2` PubMed behavior, canonical foundation plans, and legacy selection fixtures remain exact; the identity-bearing PubMed overlay has separate adapter and policy versions and separate golden plans.
- Current Standalone Search and Deep Research import no family symbol and issue zero additional network requests.

## Egress, Retention, and Security

Every physical call crosses the shared research-discovery gateway and the connected-peer-verified one-hop transport. Both routes declare exact HTTPS origins, exact paths, allowed methods, query keys, closed planner/adapter value validation, zero redirects, timeouts, byte ceilings, result ceilings, page ceilings, retry ceilings, and physical-dispatch allowances.

The family code may not import an HTTP client, DNS/socket library, browser tool, legacy provider helper, Media ingestion function, scraper, cookie store, credential store, or result fetcher.

`test_research_discovery_network_boundary.py` is currently singular and hard-coded to `biorxiv_medrxiv.py`; merely adding a digest would not certify this family. This task generalizes that test to an exact per-family configuration map. For each family it declares the module filename, allowed root/import closure, runtime adapter/route fixtures, expected logical-page count, expected physical-dispatch count, and consumer-denial expectations. The scanner applies its AST import rules to every configured family module, pins raw and semantic-AST digests for each family root, pins semantic/import digests for changed shared closure modules, updates exact closure/import allowlists, and runs both ClinicalTrials.gov and PMC fixture plans under the no-network runtime tripwire. Raw shared-module digests are deliberately avoided because comment/format churn is not executable policy. Production consumers remain forbidden from importing either family.

Only normalized metadata and synthesized inert links leave the adapter. Pagination tokens, raw payloads, provider query URLs, provider-supplied links, contacts, locations, attachments, documents, markup, and unknown fields are discarded. Every retained human-facing ClinicalTrials.gov or PMC string passes the bounded family-local any-URL detector after normalization/sanitization; DOI/PMID/PMCID values follow their separate identifier grammars and never pass through this human-text rule. No returned URL is requested, resolved, or used to alter attribution. Against the implementation-base revision, explicit import/AST/runtime tests deny `tldw_Server_API.app.core.Web_Scraping`, `http.cookiejar`, browser-cookie modules, cookie managers, credential stores, and cookie/authorization header construction. They do not claim knowledge of unknown future TASK-13013 APIs.

## Rate, Timeout, Cancellation, and Failure Semantics

- Exact route limits bound pages, physical dispatches, response bytes, results, redirects, retries, and per-hop time; the plan allowance and plan-wide execution controller bound aggregate wall-clock execution.
- ClinicalTrials.gov HTTP 429 and either NCBI HTTP 429 or the documented JSON error envelope become typed rate-limited outcomes; the adapter does not sleep or retry.
- This is a bounded per-execution request/dispatch policy, not a process-wide rate limiter. Global per-origin NCBI pacing is a production cutover requirement owned by TASK-12968.3.
- A ClinicalTrials.gov page token consumes a new page and physical-dispatch allowance only immediately before the gateway call.
- ClinicalTrials.gov intentionally uses its fixed 100-record raw route ceiling rather than adding a new dynamic group field. A small user `result_limit` can therefore cause one bounded second request when the provider returns a token; tests pin this at no more than two calls/100 raw records, and global result truncation remains executor-owned.
- PMC ESummary receives its own reservation only after a successful nonempty ESearch.
- Cancellation immediately after a ClinicalTrials.gov token is parsed but before continuation dispatch creates no second reservation: journal `created=1`, `debited=1`, `released=0`, `outstanding=0`, one page, and zero continuation calls.
- Cancellation after nonempty PMC ESearch parsing but before ESummary dispatch has the same exact journal counts and zero summary calls.
- Existing executor reservation-race tests continue to prove that cancellation after a reservation but before a physical debit releases that reservation; this family does not weaken those shared invariants.
- Timeout and cancellation preserve existing attempt-journal and usage semantics.
- A failed/malformed family route is atomic; candidates from that route are not published.
- Independently successful routes remain available with explicit partial status.
- Attribution mismatches are dropped or fail closed according to the exact provider contract; they are never force-labeled as the selected source.

## Inventory Reconciliation

Frozen rows `sourclip-2026-07-13-0026` and `sourclip-2026-07-13-0027` retain stable canonical targets and both declared surfaces.

The route-identity migration is exact:

| Inventory row | Old route / mode | New route / mode |
| --- | --- | --- |
| `sourclip-2026-07-13-0026` | `clinicaltrials_gov_clinicaltrials_gov_api_v2_direct` / `structured_query` | `clinicaltrials_gov_studies_search_direct` / `general_free_text` |
| `sourclip-2026-07-13-0027` | `pubmed_central_ncbi_eutils_pmc_direct` / `structured_query` | `pubmed_central_esearch_summary_direct` / `general_free_text` |

The ledger updates must:

- identify `clinicaltrials_gov_api_v2` and `ncbi_eutils_pmc` as the planned physical backends;
- identify the two exact general-query route candidates;
- keep identifier, bulk, date/OAI, OA-package, EFetch, HTML, and full-text capabilities distinct from general search;
- remove the `snippet` capability claim from the PMC row because this exact implemented route is metadata-only and always emits `snippet=None`;
- record implementation and fixture feasibility without claiming Standalone Search or Deep Research readiness;
- retain official evidence references and bounded live-observation notes; and
- set implementation/fixture state only to the exact shadow evidence achieved, leave surface readiness false, and regenerate report content and raw-byte digests only through the authoritative inventory tools.

The authoritative inventory contract must also extend `REQUIRED_SOURCES` (or an equally authoritative exact gate) for both rows. It pins the unchanged `source_snapshot_sha256`, exact route/backend/query mode, `implementation_state=implemented`, `fixture_state=passed`, `live_state=not_run`, empty `certifications`, unchanged declared surfaces with no surface-ready claim, and regenerated ledger/report digests. Generic schema validity alone is insufficient.

Mapped, implemented, fixture-ready, live-observed, and surface-ready remain distinct states. This task advances only the states it actually proves.

## Test and Evidence Matrix

### Opaque cursor contract

- exact optional first-page omission and continuation insertion;
- numeric/opaque type mismatch;
- empty, whitespace, control, Unicode, DEL, and oversized token rejection;
- a worst-case eight-by-32 four-byte Unicode query plus 1,024 reserved-character token stays within the existing 8,192-byte encoded-target cap;
- provider punctuation remains one percent-encoded query value;
- repeated-token loop rejection;
- cursor mutation cannot alter another key, method, path, origin, policy digest, or intent;
- opaque policies affect their digest while all foundation digests remain exact;
- `RoutePolicy` positional construction and legacy digest material remain unchanged;
- token absence from public trace/outcome/usage/journal/error text and from cursor/effective-intent/binding/request/callback-diagnostic `repr`;
- one isolated wire-stub proof that punctuation remains one percent-encoded query value; and
- numeric query/body/path cursor behavior remains unchanged.

### ClinicalTrials.gov

- known bounded nonempty success;
- exact valid empty result (`totalCount=0`, empty studies, no token);
- page-two continuation through `nextPageToken`;
- 50-record page and 100-result/two-page/two-dispatch ceilings with no hidden continuation;
- frozen `totalCount`, per-page raw cardinality, cumulative pre-deduplication counting, positive-count empty rejection, and the token/count biconditional;
- HTTP 429, timeout, cancellation, redirect, and partial failure;
- malformed JSON, missing envelopes, invalid nested types, parser limits, and schema drift;
- invalid/repeated/conflicting NCT IDs;
- exact `markupFormat=legacy`; hostile CommonMark links/images/autolinks, legacy/embedded HTML, entities, controls, URLs, and unknown fields are reduced to bounded plain text or dropped;
- required field projection and query grammar cannot be widened or injected; and
- synthesized inert link, stable identity, source attribution, entirely synthetic fixtures, and distinct observation-time/provider-`dataTimestamp` evidence semantics.

### PubMed Central

- nonempty ESearch plus ordered ESummary success;
- valid empty ESearch with no second reservation/call;
- canonical decimal-string `count`/`retstart`/`retmax` envelope validation;
- HTTP and documented JSON rate-limit envelopes, timeout, cancellation, malformed JSON, and partial failure;
- exact frozen `tool`/`email` on both intents, no `sort`, and user override rejection;
- malformed, duplicate, noncanonical, or oversized UIDs;
- missing/extra/partial/reordered summaries;
- numeric ESearch/ESummary correlation plus provider-supplied `pmcid == "PMC" + uid`, typed DOI/PMID handling, and exact title/author bounds;
- missing/conflicting article identifiers and same-identity conflicts;
- exact `db=pmc` on both intents and no PubMed cross-attribution;
- no EFetch, OAI, HTML, JATS, PDF, or hidden third call;
- unchanged `foundation-v2` PubMed plan/output/error/accounting plus exact identity-bearing overlay plans;
- a nonempty overlay fixture executed through the executor, shared PubMed callable, gateway, and one-hop transport, proving successful normalized output, exact identity pairs on both calls, one logical page/two physical dispatches, strict rate-envelope versus malformed-lookalike classification, query-key-only traces, and redacted diagnostic representations; and
- no live NCBI call in TASK-12968.6.

### Registry, compatibility, and security

- stable source, route, backend, adapter, catalog, registry, readiness, and policy identities;
- exact source constructor values, route constructor values, inventory rows, and generated digests reconcile;
- only compatible typed query modes plan family routes;
- no unsafe cross-source coalescing or forced attribution;
- every family dispatch crosses the gateway and is physically accounted;
- exact cancellation races prove no pre-continuation/pre-summary reservation and no token retention;
- ClinicalTrials.gov page-one success followed by malformed page two publishes no trial candidate while an independently successful nonempty PMC route retains its candidates and explicit partial status;
- the per-family network/import closure includes every new or changed module and executes all three shadow call shapes under runtime tripwires with separate page/dispatch expectations (ClinicalTrials.gov `2/2`, nonempty PMC `1/2`, empty PMC `1/1`, and nonempty identity-bearing PubMed overlay `1/2`);
- foundation and legacy selection behavior remain exact; and
- production Standalone Search and Deep Research execute no family request.

Only distinct valid provider shapes become checked-in fixtures, and every ClinicalTrials.gov response fixture is synthetic. Error, injection, token, attribution, duplicate, cancellation, timeout, and partial-failure cases should be generated through bounded fixture mutation or gateway stubs rather than one fixture file per branch.

Any optional ClinicalTrials.gov feasibility check is opt-in, credentialless, excluded from ordinary CI, and capped at two calls: `/api/v2/version` plus one `/api/v2/studies` page. It records local observation time separately from provider `dataTimestamp`, endpoints, request count, status, schema/attribution shape, projection/modification note, zero retained study values or raw bodies, and zero result-link dereferences. PMC implementation evidence is fixture-only; no E-utilities feasibility call is permitted in this task. Historical observations are schema feasibility notes, not surface certification.

## Verification Gates

Before the implementation PR is ready for review:

- each production behavior is introduced through a witnessed RED test followed by the minimum GREEN change;
- focused family, contract, planner, executor, gateway, adapter, registry, inventory, and network-boundary tests pass;
- the complete Research test suite passes;
- Python compilation and Python 3.10 syntax compatibility checks pass;
- repository-configured Ruff/Black or equivalent touched-scope formatting checks pass;
- Bandit reports no new findings in touched production Python;
- inventory validators and digest checks pass;
- `git diff --check` and self-review are clean; and
- independent spec-compliance, code-quality, and final reviews have no unresolved Critical or Important findings.

## Rollout and Cutover Gates

1. Implement diagnostic redaction plus the closed cursor and literal-term policy changes with foundation compatibility tests.
2. Add the exact identity-bearing PubMed shadow overlay while preserving `foundation-v2` plans and behavior.
3. Add and fixture-certify ClinicalTrials.gov in the shadow family.
4. Add and fixture-certify PMC through the parameterized NCBI two-dispatch seam.
5. Reconcile exact constructor contracts, inventory rows, generated artifacts, and bounded schema notes.
6. Complete the verification and review gates, then merge this provider-only PR before TASK-12968.3 implementation begins.

TASK-12968.3 separately owns:

- process-wide NCBI anonymous pacing;
- evidence that the frozen non-user NCBI `tool`/`email` pair is registered, plus the required disclaimer/copyright notice;
- ClinicalTrials.gov attribution, data-processed date, modification disclosure, and currentness/refresh presentation;
- bounded long-query GET/POST policy;
- Standalone Search consumer wiring and live surface certification;
- canonical selection, effective-plan preview, partial outcomes, and large-catalog controls.

TASK-12968.4 separately owns new-session Deep Research wiring and waits for merged TASK-12968.3 and TASK-13014. Authenticated retrieval remains outside TASK-12968 and waits for TASK-13013 plus a separate approved design.

## Alternatives Rejected

### One-page ClinicalTrials.gov only

Rejected because it would ignore the provider's actual continuation contract and would not honestly satisfy TASK-12968.6 pagination coverage.

### Adapter-owned URL or intent reconstruction

Rejected because accepting a raw continuation URL or arbitrary provider string as a new intent would bypass the immutable plan, exact route policy, and sealed gateway boundary.

### Reuse bounded human-text policy for page tokens

Rejected because the current text grammar intentionally excludes token punctuation and normalizes human text. Opaque provider tokens require exact bounded bytes and no semantic normalization.

### Add a second opaque-pagination field to `RoutePolicy`

Rejected as unnecessary. `pagination_query_key` already identifies the physical query channel, and the closed query-policy type safely distinguishes numeric from opaque cursor values. A second field would expand positional construction, digest snapshots, and mutual-exclusion logic without representing a second physical capability.

### Duplicate the PubMed two-dispatch adapter

Rejected because it would copy security-sensitive ESearch/ESummary validation and accounting. A small private parameterization reuses proven behavior while keeping provider-specific normalization separate.

### Inject NCBI identity inside the gateway

Rejected because gateway-side `tool`/`email` mutation would make the wire request differ from the policy snapshot, immutable plan, coalescing identity, execution trace, and adapter validation. The exact non-user identity must be visible in the route allowlist and both planned intents before hashing and dispatch.

### Fully generic NCBI or provider framework

Rejected as speculative. Two E-utilities databases justify one private helper, not a public DSL, class hierarchy, plugin system, or configuration-driven parser.

### Add EFetch for PMC abstracts

Rejected because it creates a hidden third physical operation, increases rate and retention scope, and is unnecessary for honest metadata discovery. Any abstract/full-text route requires its own planned operation and review.

### Add the providers directly to Search or Deep Research

Rejected because the approved rollout forbids a PR from both adding a provider family and migrating a consumer contract.
