# ClinicalTrials.gov and PubMed Central Shared-Discovery Route Family

**Task:** TASK-12968.6

**Parent:** TASK-12968

**Status:** Requester-approved design; independent spec review approved (2026-07-15)

**Execution scope:** Shadow registry and fixture execution only

## Goal

Add stable `clinicaltrials_gov` and `pubmed_central` targets to the shared research-discovery pipeline through credentialless, source-native APIs. The family must provide useful general discovery, strict source attribution, bounded physical work, and inert metadata links without changing current Standalone Search or Deep Research behavior.

This document narrows TASK-12968's approved program architecture for the second provider-family slice. The shared gateway, immutable plans, executor accounting, no-dereference boundary, consumer-cutover ownership, and authenticated-retrieval deferral remain unchanged.

## Decision Summary

The implementation will:

1. add one shadow-only family module alongside `biorxiv_medrxiv.py`;
2. use ClinicalTrials.gov API v2 `GET /api/v2/studies` for direct general trial discovery;
3. add one closed, policy-bound opaque query-cursor channel for ClinicalTrials.gov's `pageToken`/`nextPageToken` continuation;
4. use NCBI E-utilities ESearch followed by conditional ESummary with `db=pmc` for PubMed Central general discovery;
5. reuse the existing PubMed two-dispatch machinery through the smallest private parameterization that preserves all PubMed behavior;
6. keep PMC continuation to one bounded ESearch/ESummary page in this slice while validating the returned pagination envelope;
7. update the frozen inventory and shadow artifacts without importing the family from any production consumer; and
8. leave global NCBI pacing, product identity/notice, long-query routing, and surface certification to TASK-12968.3.

The family adds no dependency, SDK, browser, scraper, persistent cache, background job, or provider-specific HTTP client.

## Provider Evidence and Constraints

### ClinicalTrials.gov

The [modern ClinicalTrials.gov API](https://clinicaltrials.gov/data-api/api) exposes an OpenAPI-described v2 JSON service. The [official migration guide](https://clinicaltrials.gov/data-api/about-api/api-migration) identifies `/api/v2/studies` as the modern search endpoint, maps the classic search expression to `query.term`, and documents paginated responses with a maximum provider page size of 1,000.

The route uses opaque continuation tokens returned by the provider. They are not numeric offsets and must not be coerced, ordered, normalized, interpreted, or copied into any location except the policy-declared `pageToken` query value for the same exact route.

ClinicalTrials.gov study records are structured documents, not publications. The adapter therefore does not invent authors, publication dates, DOIs, or PDFs. It retains a bounded discovery projection: NCT identifier, title, brief summary, status, conditions, interventions, study type, sponsor, relevant study dates, results availability, and an inert synthesized study link.

### PubMed Central

The [NCBI E-utilities parameter reference](https://www.ncbi.nlm.nih.gov/books/NBK25499/) defines ESearch pagination through `retstart` and `retmax` and JSON output through `retmode=json`. PubMed Central discovery uses ESearch and ESummary with the database fixed to `pmc`; it does not substitute PMC OAI date feeds, OA package retrieval, HTML search, or article dereference for general search.

The [NCBI usage guidance](https://www.ncbi.nlm.nih.gov/books/NBK25497/) limits unkeyed clients to three requests per second and raises the usual ceiling for approved API-key use. This shadow task validates request accounting and typed rate-limit failures but does not implement process-wide pacing or a credentialed route. TASK-12968.3 owns the shared per-origin limiter, product `tool`/`email` identity and notice, bounded long-query policy, and production enablement.

ESummary for `db=pmc` provides citation metadata and identifiers but not a guaranteed abstract. The adapter leaves `abstract` and `snippet` unset when the provider does not supply them; it does not add EFetch as an adapter-internal third call.

## Scope

This task includes:

- two canonical catalog targets with Standalone Search and Deep Research declared as intended surfaces;
- one direct ClinicalTrials.gov general-query route;
- one direct PubMed Central general-query route;
- exact backend, route, adapter, source, and policy identities;
- a closed opaque query-cursor contract used only when declared by route policy;
- minimal literal-term policy generalization for the `query.term` key and an empty immutable suffix;
- strict bounded JSON parsing, normalization, attribution, deduplication, and inert links;
- gateway-only physical requests, request accounting, cancellation, and partial-failure behavior;
- sanitized fixtures, deterministic tests, inventory reconciliation, and bounded live feasibility notes; and
- compatibility proofs for the foundation registry, existing PubMed adapter, legacy selection, and both production consumers.

This task does not include:

- Standalone Search or Deep Research cutover;
- surface-ready or live-certified claims;
- a process-wide NCBI limiter, API-key route, product contact configuration, or user notice;
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

Identifier lookup, ClinicalTrials.gov bulk downloads, PMC OAI/date feeds, OA package lookup, EFetch, and full-text retrieval remain inventory capabilities or future route candidates. They are not enabled general-search routes and cannot satisfy this task by proxy.

## Shadow Composition

`clinicaltrials_pubmed_central.py` owns the family registry additions, readiness overlay, adapters, strict family parsers, and immutable adapter map. It follows the established bioRxiv/medRxiv composition pattern:

- begin with a fresh `foundation_registry()` value;
- reconstruct foundation sources with the family shadow catalog version;
- append two source definitions, two routes, and two backend definitions;
- append fixture-ready family route entries to a family-only readiness overlay;
- compose only the two family adapters while rejecting duplicate adapter IDs; and
- expose no default import or consumer wiring.

The family receives distinct catalog, registry, readiness, route-policy, and adapter versions. Foundation factories and their values remain unchanged.

## General Query Contract

Both routes use the existing exact `GeneralFreeTextQuery` planning type. User input is normalized into a bounded sequence of Unicode alphanumeric terms before provider rendering. Raw provider syntax, punctuation, field operators, and control characters are never copied outbound.

`LiteralTermsQueryValuePolicy` gains two narrow capabilities:

1. its policy name may be any already-allowed query key instead of the hard-coded name `query`; and
2. `fixed_suffix` may be empty.

The planner still permits exactly one literal-term policy for a general route and emits quoted terms joined by literal `AND`. The gateway independently reconstructs the same grammar. Its validator must treat the literal expression as the whole value when the suffix is empty rather than applying Python's `value[:-0]` slice. Existing nonempty-suffix routes retain identical policy content, digests, intents, and plan bytes.

The ClinicalTrials.gov policy binds the literal expression to `query.term`. The PMC route keeps its `term` value in the explicit E-utilities intent builder because it is a two-intent route rather than a generic one-intent typed route.

## Closed Opaque Query Pagination

### Why a new cursor type is required

The current executor supports only bounded nonnegative numeric cursors. Treating a ClinicalTrials.gov token as an integer, URL, arbitrary intent mutation, or adapter-owned direct request would either fail valid continuation or bypass the sealed plan and gateway. The correct extension is one closed opaque query-cursor channel.

### Contract

Add:

- `OpaqueCursorQueryValuePolicy(name, max_chars, required=False)` to the closed query-policy union;
- `OpaqueCursor(value)` to the closed dispatch cursor union.

For this route, `max_chars` is 2,048. An opaque token must be an exact nonempty ASCII string containing only visible bytes `0x21` through `0x7e`. Whitespace, controls, DEL, Unicode, empty values, and oversized values fail closed. The token is never Unicode-normalized, decoded, parsed, sorted, logged, persisted as result metadata, or used as a URL/path.

Append `opaque_pagination_query_key: str | None = None` to `RoutePolicy` so existing positional construction remains valid. The route declares `opaque_pagination_query_key="pageToken"` and binds that key to the optional opaque-cursor policy. Existing numeric query, integer JSON-body, numeric path, and opaque query pagination channels are mutually exclusive. The first intent must omit `pageToken`. On continuation, the executor:

1. accepts only an exact `OpaqueCursor` for an opaque policy;
2. appends the absent policy-declared query pair, or replaces exactly one previously bound pair;
3. rejects numeric/opaque type mismatches;
4. tracks exact seen tokens and rejects repetition;
5. submits the reconstructed intent through the normal policy snapshot, digest, gateway validation, journal reservation, physical accounting, timeout, and cancellation path.

Opaque tokens have no ordering relationship, so only exact-repeat progress checks apply. Existing numeric query/body/path pagination retains its current type checks and monotonic path-cursor behavior.

The gateway independently validates the opaque policy and then percent-encodes the exact token as a query value for the already-bound exact origin and path. Provider punctuation therefore cannot add a query key, alter the path, change the host, or create a fragment.

The new query-value policy participates in the policy digest and immutable policy snapshot. The new route-policy channel is added to canonical digest material only when non-`None`; policies that do not use it keep their existing digest bytes. Contract tests pin full positional construction and the existing legacy digest oracle.

## ClinicalTrials.gov Route

### Request shape

- Method: `GET`
- Path: `/api/v2/studies`
- General expression: `query.term`
- Bounded page size: `pageSize`
- Optional continuation: `pageToken`
- Fixed total-count request: `countTotal=true`
- Fixed response format: JSON
- Fixed allowlisted `fields` value: `NCTId,BriefTitle,OfficialTitle,BriefSummary,OverallStatus,Condition,InterventionName,LeadSponsorName,StudyType,StartDate,CompletionDate,HasResults`
- Redirects: disabled
- Credentials, cookies, and custom user headers: absent

The exact field projection is frozen in the route policy and adapter validation. A bounded live probe on 2026-07-15 confirmed that this projection returns the expected nested modules, `totalCount`, `studies`, and an opaque `nextPageToken`. The planner cannot widen it, and provider-supplied links or document fields are never retained.

The route uses a bounded first-page size no greater than the task's route result ceiling and at most the provider's documented maximum. `max_pages` and the executor result ceiling bound total continuation. The adapter follows `nextPageToken` only while another page and more candidates are permitted; a token returned after a ceiling produces the existing truncation accounting rather than hidden work.

### Response validation

The adapter requires one strict JSON object containing a nonnegative integer `totalCount`, a bounded `studies` array, and an optional `nextPageToken`. Because the request fixes `countTotal=true`, a missing or invalid count is schema drift. Cumulative returned records may not exceed that count, and a continuation is invalid after the count is exhausted. Each retained study must contain one canonical `NCT` identifier matching `NCT` plus eight ASCII digits and one nonempty bounded title. Duplicate NCT IDs with identical normalized records collapse deterministically; conflicting records for one NCT ID are schema drift.

Nested objects and arrays are subject to the shared byte, record, depth, node, string, numeric-token, and parse-deadline guards. The adapter validates only the frozen projection and drops unknown fields. A valid empty `studies` array without a continuation token succeeds. Empty data with a continuation token, malformed tokens, missing required envelopes, wrong scalar types, invalid identifiers, oversized fields, and inconsistent duplicate records fail as provider payload errors.

Provider HTTP 429 and retry metadata use the existing typed rate-limit path. Timeouts, cancellation, non-JSON responses, unexpected redirects, 4xx/5xx responses, and parse-deadline exhaustion use existing gateway/adapter outcomes.

### Normalized record

The adapter retains:

- canonical NCT ID and `provider_ids.nct_id`;
- brief title, with official title as bounded supplemental metadata when present;
- bounded brief summary as `abstract` and `snippet` when present;
- overall status, conditions, intervention names, sponsor, study type, relevant start/completion dates, and results availability as source-specific metadata; and
- `https://clinicaltrials.gov/study/{NCT_ID}` synthesized solely from the validated identifier.

`authors`, DOI, PMID, PMCID, arXiv ID, publication date, and `pdf_url` remain absent unless the provider contract supplies a semantically correct field in a later reviewed route. Raw markup, arbitrary URLs, contacts, locations, documents, references, and unknown modules are dropped.

The fingerprint uses the validated normalized record and stable NCT provider identity. Source attribution is route-constrained to `clinicaltrials_gov`; no provider field can force another logical source.

## PubMed Central Route

### Planned physical work

The route plans exactly two immutable intents:

1. ESearch: `GET /entrez/eutils/esearch.fcgi`
2. Conditional ESummary: `GET /entrez/eutils/esummary.fcgi`

ESearch fixes `db=pmc`, `retmode=json`, `sort=relevance`, `retstart=0`, and a bounded `retmax`. ESummary fixes `db=pmc` and `retmode=json`; its `id` value is grounded only from the validated positive numeric ESearch UID list through the existing deferred numeric CSV binding.

A valid empty ESearch result performs no ESummary reservation or physical request. Nonempty execution accounts for both calls separately. No SDK, EFetch, hidden retry, hidden page, or third call is permitted.

This task keeps PMC to one bounded result page, matching the existing PubMed foundation contract. It validates `count`, `retstart`, `retmax`, ID cardinality, and the existence of additional results but does not continue to another ESearch/ESummary pair. Real continuation in the family is exercised by ClinicalTrials.gov. Repeated two-dispatch pagination is deferred until a consumer requirement justifies a reviewed executor contract for reusable conditional intents.

### Minimal NCBI reuse

The existing PubMed adapter already owns strict NCBI root/version checks, diagnostic-list handling, numeric UID validation, conditional CSV grounding, response checking, parse guards, and two-dispatch accounting. Extract the smallest private ESearch/ESummary helper parameterized by:

- adapter and route identity;
- exact database value;
- deferred binding identity and item bounds; and
- a record-normalization callback.

The existing `pubmed_v2` wrapper becomes one caller with unchanged values and output. The family PMC wrapper becomes the second caller with `db=pmc`. Provider-specific record parsing remains separate. No public adapter framework, class hierarchy, provider registry, or transport abstraction is added.

All existing PubMed fixtures and plan bytes are regression locks for the extraction.

### Response validation and normalization

ESearch requires the canonical NCBI JSON header/root and a consistent `esearchresult`. Bounded diagnostic lists remain nonfatal only in the already accepted shapes; fatal `ERROR` envelopes and malformed diagnostics fail closed. Returned UIDs must be unique canonical positive decimal strings within binding limits.

ESummary requires `uids` to contain exactly the expected UID set and one record per UID. The adapter restores ESearch ordering rather than trusting provider object order. It parses only bounded citation metadata and `articleids`.

Each retained record requires a canonical PMCID from provider metadata. It may additionally retain a canonical DOI and PMID when present. The normalized projection contains:

- title;
- bounded author names;
- journal/source and publication metadata when present;
- `provider_ids.pmc_uid`, `provider_ids.pmcid`, and optional DOI/PMID;
- `https://pmc.ncbi.nlm.nih.gov/articles/{PMCID}/` synthesized from the validated PMCID; and
- `abstract=None`, `snippet=None`, and `pdf_url=None`.

Missing PMCID, conflicting article identifiers, duplicate UIDs, partial summaries, unexpected extra records, oversized values, invalid dates/identifiers, and same-identity conflicting metadata fail the logical route rather than inventing or force-attributing data.

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
- Foundation registry values, adapters, canonical plans, and legacy selection fixtures remain exact.
- Current Standalone Search and Deep Research import no family symbol and issue zero additional network requests.

## Egress, Retention, and Security

Every physical call crosses the shared research-discovery gateway and the connected-peer-verified one-hop transport. Both routes declare exact HTTPS origins, exact paths, allowed methods, query keys, typed value policies, zero redirects, timeouts, byte ceilings, result ceilings, page ceilings, retry ceilings, and physical-dispatch allowances.

The family code may not import an HTTP client, DNS/socket library, browser tool, legacy provider helper, Media ingestion function, scraper, cookie store, credential store, or result fetcher. Network-boundary closure tests and reviewed module digests expand to include the new family and any shared helper changes.

Only normalized metadata and synthesized inert links leave the adapter. Pagination tokens, raw payloads, provider query URLs, provider-supplied links, contacts, locations, attachments, documents, markup, and unknown fields are discarded. No returned URL is requested, resolved, or used to alter attribution.

## Rate, Timeout, Cancellation, and Failure Semantics

- Route limits bound pages, physical dispatches, response bytes, results, redirects, retries, and wall-clock execution.
- Upstream 429 responses become typed rate-limited outcomes; the adapter does not sleep.
- Global per-origin NCBI pacing is a production cutover requirement owned by TASK-12968.3.
- A ClinicalTrials.gov page token consumes a new page and physical-dispatch allowance only immediately before the gateway call.
- PMC ESummary receives its own reservation only after a successful nonempty ESearch.
- Timeout and cancellation preserve existing attempt-journal and usage semantics.
- A failed/malformed family route is atomic; candidates from that route are not published.
- Independently successful routes remain available with explicit partial status.
- Attribution mismatches are dropped or fail closed according to the exact provider contract; they are never force-labeled as the selected source.

## Inventory Reconciliation

Frozen rows `sourclip-2026-07-13-0026` and `sourclip-2026-07-13-0027` retain stable canonical targets and both declared surfaces.

The ledger updates must:

- identify `clinicaltrials_gov_api_v2` and `ncbi_eutils_pmc` as the planned physical backends;
- identify the two exact general-query route candidates;
- keep identifier, bulk, date/OAI, OA-package, EFetch, HTML, and full-text capabilities distinct from general search;
- record implementation and fixture feasibility without claiming Standalone Search or Deep Research readiness;
- retain official evidence references and bounded live-observation notes; and
- regenerate report content and raw-byte digests only through the authoritative inventory tools.

Mapped, implemented, fixture-ready, live-observed, and surface-ready remain distinct states. This task advances only the states it actually proves.

## Test and Evidence Matrix

### Opaque cursor contract

- exact optional first-page omission and continuation insertion;
- numeric/opaque type mismatch;
- empty, whitespace, control, Unicode, DEL, and oversized token rejection;
- provider punctuation remains one percent-encoded query value;
- repeated-token loop rejection;
- cursor mutation cannot alter another key, method, path, origin, policy digest, or intent;
- opaque policies affect their digest while all foundation digests remain exact; and
- the optional route-policy channel preserves positional construction and is omitted from legacy digest material; and
- numeric query/body/path cursor behavior remains unchanged.

### ClinicalTrials.gov

- known bounded nonempty success;
- valid empty result;
- page-two continuation through `nextPageToken`;
- result and page ceilings with truthful truncation;
- HTTP 429, timeout, cancellation, redirect, and partial failure;
- malformed JSON, missing envelopes, invalid nested types, parser limits, and schema drift;
- invalid/repeated/conflicting NCT IDs;
- hostile provider URLs and unknown fields are dropped;
- required field projection and query grammar cannot be widened or injected; and
- synthesized inert link, stable identity, and source attribution.

### PubMed Central

- nonempty ESearch plus ordered ESummary success;
- valid empty ESearch with no second reservation/call;
- `count`/`retstart`/`retmax` pagination-envelope validation;
- rate-limit envelopes, timeout, cancellation, malformed JSON, and partial failure;
- malformed, duplicate, noncanonical, or oversized UIDs;
- missing/extra/partial/reordered summaries;
- PMCID, DOI, PMID, author, title, journal, and date bounds;
- missing/conflicting article identifiers and same-identity conflicts;
- exact `db=pmc` on both intents and no PubMed cross-attribution;
- no EFetch, OAI, HTML, JATS, PDF, or hidden third call; and
- unchanged PubMed adapter output and accounting.

### Registry, compatibility, and security

- stable source, route, backend, adapter, catalog, registry, readiness, and policy identities;
- inventory rows and generated digests reconcile;
- only compatible typed query modes plan family routes;
- no unsafe cross-source coalescing or forced attribution;
- every family dispatch crosses the gateway and is physically accounted;
- network/import closure includes every new or changed module;
- foundation and legacy selection behavior remain exact; and
- production Standalone Search and Deep Research execute no family request.

Only distinct valid provider shapes become checked-in fixtures. Error, injection, token, attribution, duplicate, cancellation, timeout, and partial-failure cases should be generated through bounded fixture mutation or gateway stubs rather than one fixture file per branch.

Live feasibility checks are opt-in, bounded, credentialless, and excluded from ordinary CI. They record observation time, endpoint, request count, status, returned identifier/attribution shape, and zero result-link dereferences. They are feasibility evidence, not surface certification.

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

1. Implement the closed cursor and literal-term policy changes with foundation compatibility tests.
2. Add and fixture-certify ClinicalTrials.gov in the shadow family.
3. Add and fixture-certify PMC through the parameterized NCBI two-dispatch seam.
4. Reconcile the family registry, inventory rows, generated artifacts, and bounded live notes.
5. Complete the verification and review gates.
6. Merge this provider-only PR before TASK-12968.3 implementation begins.

TASK-12968.3 separately owns:

- process-wide NCBI anonymous pacing;
- product identity/contact and required user-facing notices;
- bounded long-query GET/POST policy;
- Standalone Search consumer wiring and live surface certification;
- canonical selection, effective-plan preview, partial outcomes, and large-catalog controls.

TASK-12968.4 separately owns new-session Deep Research wiring and waits for merged TASK-12968.3 and TASK-12970. Authenticated retrieval remains outside TASK-12968 and waits for TASK-12969 plus a separate approved design.

## Alternatives Rejected

### One-page ClinicalTrials.gov only

Rejected because it would ignore the provider's actual continuation contract and would not honestly satisfy TASK-12968.6 pagination coverage.

### Adapter-owned URL or intent reconstruction

Rejected because accepting a raw continuation URL or arbitrary provider string as a new intent would bypass the immutable plan, exact route policy, and sealed gateway boundary.

### Reuse bounded human-text policy for page tokens

Rejected because the current text grammar intentionally excludes token punctuation and normalizes human text. Opaque provider tokens require exact bounded bytes and no semantic normalization.

### Duplicate the PubMed two-dispatch adapter

Rejected because it would copy security-sensitive ESearch/ESummary validation and accounting. A small private parameterization reuses proven behavior while keeping provider-specific normalization separate.

### Fully generic NCBI or provider framework

Rejected as speculative. Two E-utilities databases justify one private helper, not a public DSL, class hierarchy, plugin system, or configuration-driven parser.

### Add EFetch for PMC abstracts

Rejected because it creates a hidden third physical operation, increases rate and retention scope, and is unnecessary for honest metadata discovery. Any abstract/full-text route requires its own planned operation and review.

### Add the providers directly to Search or Deep Research

Rejected because the approved rollout forbids a PR from both adding a provider family and migrating a consumer contract.
