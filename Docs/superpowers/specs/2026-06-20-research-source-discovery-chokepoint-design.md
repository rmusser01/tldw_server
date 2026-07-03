# Research Source Discovery Chokepoint Design

Date: 2026-06-20
Backlog task: TASK-2336
Boundary revision task: TASK-12082
Reference directory: https://www.sourclip.com/resources/research-sources

## Summary

Add a shared research discovery chokepoint for open research graph sources. The new design consolidates the repo's scattered paper/source searches behind one internal module while preserving existing provider-specific endpoints as compatibility surfaces.

The staged design focuses on open research graph discovery: OpenAlex, Semantic Scholar, Crossref, arXiv, PubMed, Zenodo, Figshare, OSF, and Unpaywall-style open-access resolution. Standalone search and Deep Research both consume the same normalized discovery service. Users can review search results, resolve open-access/full-text candidates, and hand approved selections to the existing Media ingestion surface. Deep Research uses the same service during collection and asks Media to ingest only after a source review checkpoint is approved. Phase 1 is implemented; the next plan should target the Phase 2 Media-owned ingest handoff unless the human requester explicitly asks for a broader multi-phase implementation plan.

Sourclip is used as inspiration and seed material for a curated local catalog. It is not a runtime dependency.

## Goals

- Create one chokepoint module for research/media source discovery instead of continuing to grow independent provider paths.
- Keep user-facing "source" concepts separate from implementation-facing provider adapters.
- Support standalone search and Deep Research through the same backend service and normalized result contract.
- Start with API-backed open research graph providers, using official/public APIs where available.
- Allow configurable per-source fallback site search, disabled by default unless explicitly enabled.
- Attach open-access/full-text candidates during discovery, then revalidate legality, URL policy, content type, size, and duplicate status at ingest time.
- Make Deep Research ingest review-gated and idempotent.
- Keep Media as the sole public ingestion owner. Discovery resolves selected candidates; Media owns duplicate handling, egress checks, extraction, persistence, quotas, and response outcomes.
- Preserve existing `/api/v1/paper-search/*` endpoints while allowing safe internal delegation over time.

## Non-Goals

- Do not implement every source from the Sourclip directory in the first slice.
- Do not make Sourclip scraping part of runtime behavior.
- Do not remove or break existing provider-specific paper-search endpoints.
- Do not treat web fallback results as official provider data.
- Do not auto-ingest sources during Deep Research collection.
- Do not add a public research-owned ingest endpoint such as `POST /api/v1/research/discovery/ingest`.
- Do not duplicate Media ingestion options, extraction logic, duplicate checks, quota checks, or persistence behavior in Research Discovery.
- Do not build a general third-party connector/plugin marketplace in this slice.

## Current Repo Context

The repo already has several relevant pieces:

- `/api/v1/research/websearch` and core web search engines.
- `/api/v1/research/runs` and the Jobs-backed Deep Research lifecycle.
- `tldw_Server_API/app/core/Research/providers/` with local, web, academic, synthesis provider layers.
- `ResearchBroker` with `local`, `academic`, and `web` collection lanes.
- `/api/v1/paper-search/*` provider-specific endpoints.
- `tldw_Server_API/app/core/Third_Party/` adapters for arXiv, BioRxiv, ChemRxiv, Crossref, EarthArXiv, Figshare, HAL, IACR, IEEE Xplore, OpenAlex, OSF, PMC, PubMed, RePEc, Semantic Scholar, Springer Nature, Unpaywall, viXra, Zenodo, and related sources.

The new work should reuse those pieces and move shared source discovery behavior into a single internal service boundary.

## Architecture

### ResearchSourceCatalog

`ResearchSourceCatalog` is a curated local catalog seeded from Sourclip-style categories. Each entry describes a user-facing source, not necessarily a one-to-one adapter.

Each source entry includes:

- `source_id`
- display name
- category and optional subcategory
- supported content types, such as paper, preprint, dataset, repository record, media, webpage
- access level, such as public API, credentialed API, site search, manual, disabled
- capabilities:
  - `searchable`
  - `full_text_resolvable`
  - `ingestable`
  - `requires_credentials`
  - `fallback_search_allowed`
  - `rate_limited`
- default discovery mode: `api`, `site_search`, `manual`, or `disabled`
- source priority for deterministic ranking
- known provider adapter or aggregator mapping
- optional site-search host constraints
- terms/trust notes suitable for user-facing display
- catalog version

Catalog entries can map to direct source APIs, aggregator-backed providers, or fallback search. For example, a source may be searched through its own API, OpenAlex/Crossref metadata, or opt-in site search depending on the catalog entry and effective configuration.

### ResearchDiscoveryService

`ResearchDiscoveryService` is the shared chokepoint. It accepts a query plus selected source ids or categories and returns normalized, deduped, provenance-rich results.

Responsibilities:

- validate and resolve selected sources/categories through the catalog
- enforce caps before execution:
  - max selected sources
  - max results per source
  - total result cap
  - timeout per source
  - total timeout
  - bounded concurrency
  - per-provider rate limits where configured
- reject over-cap requests with a clear validation error instead of silently truncating category-expanded source selections
- route each source to an adapter or fallback search implementation
- normalize provider-specific payloads into one result contract
- preserve sanitized provider metadata and adapter/catalog versions
- dedupe overlapping records
- rank deterministically
- attach OA/full-text candidates
- return source-level statuses and user-facing warnings
- persist a user-owned discovery snapshot for later Media handoff

### Source Router And Provider Adapters

`ResearchSourceRouter` maps catalog source ids to implementation paths. Provider adapters remain small and provider-specific. The first slice should wrap existing `Third_Party` functions where practical instead of creating new endpoint-to-endpoint HTTP calls.

Initial provider set:

- OpenAlex
- Semantic Scholar
- Crossref
- arXiv
- PubMed
- Zenodo
- Figshare
- OSF
- Unpaywall-style OA resolution

Existing `/api/v1/paper-search/*` endpoints remain provider-specific compatibility wrappers. They may delegate to the chokepoint only when their existing response contract naturally matches the normalized discovery contract.

### ResearchOAResolver

`ResearchOAResolver` is a distinct step from search and ingest. It attaches candidate full-text/PDF URLs and access metadata from provider fields and DOI-based OA lookup.

Discovery-time OA candidates are advisory. Media ingest must revalidate every candidate before download.

### ResearchDiscoverySelectionResolver

`ResearchDiscoverySelectionResolver` is the only Research-side handoff component for ingestable selections. It resolves server-owned discovery snapshot references into structured source descriptors for Media. It does not download content, parse files, extract text, check duplicates, persist Media DB records, or return ingest outcomes.

Responsibilities:

- validate snapshot integrity and user scoping for the authenticated user context supplied by Media
- locate selected `{ result_id, candidate_id }` pairs in the server-owned snapshot
- revalidate source fingerprint and selected OA/full-text candidate identity
- return a bounded `MediaSourceCandidate` descriptor with normalized identifiers, candidate type, safe display URL or resolver reference, provenance, access/license hints, and safe metadata
- preserve enough resolver/provider provenance for Media to re-resolve signed or expiring candidate URLs when needed
- reject unsupported candidate types before Media attempts ingestion

### Media-Owned Ingestion Handoff

Standalone discovery ingest is not a Research API. Approved discovery selections are submitted through the existing Media ingestion surface, primarily `/api/v1/media/add` for the synchronous Phase 2 slice.

Media responsibilities:

- authenticate, rate-limit, and enforce quotas for the ingest request
- call `ResearchDiscoverySelectionResolver` with authenticated user context only to resolve discovery references into source descriptors
- check duplicate DOI, PMID, PMCID, arXiv id, provider id, canonical URL, and content fingerprint before downloading large files
- enforce centralized egress/SSRF policy
- validate URL scheme and host
- validate content type and file size
- validate access/license hints where available
- process PDFs through the existing PDF/media pipeline
- process HTML candidates through the existing web/context extraction pipeline, storing extracted content and metadata rather than full raw HTML snapshots
- persist Media DB references and per-item ingest outcomes
- return partial success/failure results without exposing sensitive provider details

If `/api/v1/media/add` cannot accept discovery selection references cleanly without damaging its existing contract, the implementation plan must stop and review the Media API shape. That pressure is not a reason to create a parallel research-owned ingest endpoint.

## Normalized Result Contract

The normalized discovery result should include:

- stable `result_id`
- canonical `fingerprint`
- `primary_source_id`
- `primary_provider`
- `discovery_mode`: `api`, `aggregator`, or `site_search`
- title
- authors
- abstract or snippet
- DOI
- PMID or PMCID
- arXiv id
- provider ids
- canonical URL
- published and updated dates
- source category
- OA/full-text candidates
- access/license hints
- Media ingest eligibility and recommended ingest candidate id when available
- dedupe confidence
- ranking signals
- source trust/provenance labels
- warnings
- merged provenance entries
- safe provider metadata
- adapter version
- catalog version

Identity rules:

- `fingerprint` is deterministic across runs where stable identifiers are available. It is derived from DOI, PMID/PMCID, arXiv id, provider ids, canonical URL, and normalized title/author/date fallback data using the dedupe priority below.
- `result_id` is stable within a persisted discovery snapshot or Deep Research source artifact. It is derived from the canonical fingerprint plus source/provider context and is not treated as a globally permanent identifier.
- Standalone search responses include a `discovery_id` that identifies the persisted result snapshot owned by the requesting user.
- Deep Research collection persists the same normalized records in run artifacts and checkpoint payloads, scoped by `session_id` and checkpoint id.
- Media ingest never relies on client-resubmitted metadata as authority. It loads the server-side discovery snapshot or Deep Research artifact through the resolver, then revalidates source identity and full-text candidates before download.

Merged provenance entry shape:

- `source_id`
- `provider`
- `discovery_mode`
- provider ids observed for this source
- URL or landing page observed for this source
- source-specific score or rank when available
- source status and warnings
- safe provider metadata
- adapter version

When dedupe merges multiple provider records into one normalized result, `primary_source_id` and `primary_provider` identify the record selected for display/ranking, while `merged_provenance[]` preserves every contributing provider/source record. Provider-specific scores remain scoped to provenance entries and are not promoted into a universal relevance score.

OA/full-text candidate shape:

- `candidate_id`
- candidate type, such as `pdf`, `html_full_text`, `repository_file`, `landing_page`, or `metadata_only`
- safe URL or display URL, when the candidate URL is safe to expose
- opaque resolver reference, when the raw candidate URL is signed, expiring, token-bearing, or otherwise sensitive
- `url_redacted` flag
- `requires_reresolution` flag
- provider or resolver provenance
- access status and license hints where available
- content type hint where available
- rank within the result
- confidence
- warnings

Candidate identity rules:

- `candidate_id` is stable within a persisted discovery snapshot or Deep Research source artifact.
- `candidate_id` is derived from result fingerprint, candidate type, resolver/provider provenance, and either a normalized safe candidate URL or an opaque resolver reference. It must not be derived from raw secret-bearing URL material.
- Signed, expiring, token-bearing, or otherwise secret-bearing candidate URLs must be sanitized before API response, persistence, and logs. Responses and snapshots should expose a safe display URL or opaque resolver reference, not the raw sensitive URL. The stored candidate should keep enough resolver/provider provenance to re-resolve the URL at ingest time.
- Media handoff selections must identify both `result_id` and `candidate_id`; the system should not silently choose among multiple full-text candidates.
- A result may expose `recommended_candidate_id` for UI convenience, but the Media ingest request/checkpoint approval still records the explicit candidate selected.
- Phase 2 ingest eligibility is limited to `pdf` and `html_full_text` candidates. `landing_page`, `metadata_only`, `unknown`, and generic external search results are not ingestable in Phase 2.
- A later phase may evaluate broader page-like candidates, but only through existing Media extraction paths, with no raw full-HTML storage and no new Research-owned ingestion path.

Deduplication priority:

1. DOI
2. PMID or PMCID
3. arXiv id
4. provider ids
5. canonical URL
6. normalized title plus author/date hints

Ranking in the first slice should be deterministic and honest. It should use source priority, exact identifier matches, title match quality, date, OA availability, and dedupe confidence. Provider scores should not be merged into a universal relevance score unless an explicit reranking stage is enabled later.

## API Surface

### `GET /api/v1/research/sources`

Lists catalog sources, categories, capability flags, whether a source is configured, and whether fallback search is enabled or configurable.

The response includes catalog version and source-level capability metadata so clients do not infer behavior from source names.

### `POST /api/v1/research/discovery/search`

Searches selected sources or categories.

Input shape:

- query
- `source_ids`
- categories
- date filters
- result limits
- fallback policy
- optional provider/source overrides bounded by the catalog capability model

If category expansion or explicit source selection exceeds configured caps, the endpoint returns a validation error that includes the configured limit and selected count. It should not silently drop sources, because silent truncation would make source coverage and later citations misleading.

Output shape:

- `discovery_id`
- normalized deduped results
- source-level statuses
- warnings
- effective config
- catalog version
- timing and count metrics

The server persists the normalized result snapshot for the authenticated user with a bounded retention period. Persisted snapshots are the only standalone Media handoff model; client-signed result payloads and stateless recomputation remain out of scope.

### Media Ingestion Handoff

There is no `POST /api/v1/research/discovery/ingest` endpoint in this design.

Standalone discovery ingest uses the existing Media ingestion surface, primarily `/api/v1/media/add` for the synchronous Phase 2 slice. The Media request may include discovery selection references such as:

- `discovery_id`
- selected ingest candidates as `{ result_id, candidate_id }`
- optional Media-owned target collection/tags/keywords fields, if those are already supported by Media ingest

Media calls the internal `ResearchDiscoverySelectionResolver`, resolves the selections from the server-owned snapshot, and then continues through the existing Media duplicate, policy, extraction, and persistence path.

The Media endpoint verifies that the `discovery_id` belongs to the current user, is still within retention, and contains the requested `{ result_id, candidate_id }` pairs. Idempotency is keyed by owner user id, discovery id or Deep Research session id, normalized fingerprint, and selected candidate id, but Media owns the duplicate decision and returned Media DB references.

The Phase 2 synchronous handoff must use conservative default bounds. Implementations may make these values configurable through Media-owned config, but raising them beyond these defaults requires explicit design review:

- max 5 selected candidates per request
- max 45 seconds per candidate
- max 180 seconds total request time
- max 50 MiB per PDF candidate
- max 10 MiB per HTML candidate response before extraction
- accepted MIME/content types limited to `application/pdf`, `text/html`, and `application/xhtml+xml`
- over-cap requests reject with validation errors instead of silent truncation or background enqueueing
- per-item response includes `created`, `duplicate_existing`, `unsupported`, `policy_blocked`, `timeout`, and `failed` outcomes

The discovery handoff must not expose a second set of parser, chunking, metadata override, HTML extraction, or persistence options. Any such options remain Media-owned and must use the existing Media request/config model.

## Standalone Search Flow

1. Client calls `GET /api/v1/research/sources`.
2. Client submits `POST /api/v1/research/discovery/search`.
3. `ResearchDiscoveryService` resolves source selections through the catalog.
4. The router calls provider adapters or allowed fallback search.
5. Results are normalized, deduped, ranked, and enriched with OA candidates.
6. API persists a user-owned discovery snapshot and returns `discovery_id`, provenance-rich metadata, and Media ingest eligibility.
7. User selects results and submits the `discovery_id` plus selected `{ result_id, candidate_id }` pairs through the existing Media ingestion surface.
8. Media calls `ResearchDiscoverySelectionResolver` to resolve the selection from the user-owned discovery snapshot.
9. Media revalidates identifiers, policy, URLs, content, size, access hints, and duplicates.
10. Approved PDFs are processed through the existing PDF/media pipeline; approved HTML full-text candidates are processed through the existing web/context extraction pipeline.
11. Media persists successful items to Media DB and returns per-item outcomes.

## Deep Research Flow

Run creation may include `provider_overrides.discovery` with source/category selection and fallback policy.

Collection phase:

1. Deep Research calls `ResearchDiscoveryService` for each focus area.
2. It writes normalized sources, evidence notes, OA candidates, discovery warnings, effective config, and catalog version into existing artifacts.
3. It presents deduped sources and Media ingest eligibility in the source review checkpoint.

After checkpoint approval:

1. The approved `{ result_id, candidate_id }` ingest selections are resolved from the run's persisted source artifacts and submitted to a separate idempotent Media ingest job/phase.
2. The ingest job uses Media ingestion services and calls `ResearchDiscoverySelectionResolver` only for discovery artifact/source descriptor resolution.
3. Partial ingest failures are recorded as warnings.
4. Synthesis proceeds with metadata citations and Media DB references where ingest succeeded.

Deep Research should hard-fail only when policy/validation blocks the run or no usable evidence remains after collection and approved ingest outcomes are considered.

## Fallback Site Search

Fallback site search is optional per source and disabled by default.

Rules:

- only enabled by catalog/admin/user configuration where policy allows
- uses existing web search engines with bounded `site:` or host constraints
- obeys egress allowlists, SSRF protections, provider timeouts, and rate limits
- marks results as `discovery_mode=site_search`
- does not scrape or store full raw HTML; ingestable HTML candidates must go through the existing Media web/context extraction pipeline
- never presents fallback results as official API results

## Error Handling

Discovery is partial-failure tolerant.

Per-source status values:

- `ok`
- `partial`
- `rate_limited`
- `credentials_missing`
- `provider_not_configured`
- `policy_blocked`
- `provider_error`
- `timeout`

The search endpoint hard-fails when:

- request validation fails
- all selected sources are disabled or policy-blocked
- every selected source fails and zero results are available

Provider errors are sanitized. User-facing warnings should be clear but not leak secrets, configured URLs with embedded credentials, or raw upstream response bodies. Operator logs may include sanitized provider/status context.

## Security And Operations

Discovery and Media ingestion handoff must respect API terms, configured rate limits, and egress policy.

Operational controls:

- bounded concurrency
- bounded retries for transient failures only
- short-lived caching where provider terms allow it
- persisted discovery snapshots with bounded retention for standalone search-to-ingest handoff
- no retry storms across broad source selections
- credentialed sources disabled unless configured
- source-level timing and status metrics
- catalog/config version persistence
- sanitized API responses, persistence, and logging for candidate URLs that may contain credentials, signatures, or expiry tokens

Media ingest controls:

- duplicate checks before large downloads
- centralized URL policy checks
- content-type checks
- file-size caps
- license/access hint checks where available
- idempotency keyed by discovery id or run/session id plus normalized source fingerprint and selected candidate id
- existing PDF/media extraction path for PDF candidates
- existing web/context extraction path for HTML full-text candidates

Metrics and logs should cover:

- selected source count
- per-source latency
- per-source result counts
- dedupe count
- OA candidate count
- fallback usage
- ingest attempts
- ingest successes and failures
- policy blocks

## Testing Strategy

Testing should prioritize the chokepoint contract.

### Phase 1 Tests

Phase 1 unit tests:

- catalog id/category/capability validation
- catalog versioning
- disabled and credential-required source behavior
- source/category selection resolution
- caps, timeouts, and fallback policy enforcement
- partial failures and warning separation
- dedupe priority and deterministic ranking
- OA candidate attachment
- sanitized provenance metadata
- signed/token-bearing OA URL sanitization for API responses
- signed/token-bearing OA URL sanitization for persisted discovery snapshots
- signed/token-bearing OA URL exclusion from logs
- signed/token-bearing OA URL exclusion from `candidate_id` derivation
- opaque resolver reference retention for later ingest re-resolution
- source router behavior with fake adapters

Phase 1 provider adapter tests:

- mocked HTTP/client functions for first-slice sources
- reuse and extend existing sanitizer tests for OpenAlex, Semantic Scholar, Crossref, PubMed, arXiv, Zenodo, Figshare, OSF, and Unpaywall-style lookup

Phase 1 integration tests:

- `GET /api/v1/research/sources`
- `POST /api/v1/research/discovery/search`
- persisted discovery snapshot creation and ownership checks
- over-cap source/category selection validation
- fallback site search disabled-by-default behavior

### Later-Phase Tests

Standalone Media ingest handoff tests:

- discovery selection resolution from a user-owned snapshot
- URL revalidation in Media
- duplicate checks in Media
- policy-blocked URLs
- unsupported content types
- size caps
- max selection, per-item timeout, and total timeout enforcement
- partial ingest failures
- `landing_page`, `metadata_only`, `unknown`, and generic external search result candidates rejected as non-ingestable for Phase 2
- HTML full-text candidates routed through the existing web/context extraction pipeline, not stored as full raw HTML
- discovery handoff does not expose duplicate parser/chunking/HTML extraction/persistence knobs
- successful routing into existing Media DB ingestion helpers through the Media API

Later-phase integration tests:

- existing Media ingestion surface accepts discovery selection references without adding a research-owned ingest endpoint
- Deep Research source review checkpoint includes normalized results and Media ingest eligibility
- checkpoint approval triggers an idempotent Media ingest phase
- synthesis proceeds with metadata and Media DB references when available

## Rollout Plan

Phase 1: catalog, source router, discovery chokepoint, standalone search API, metadata and OA candidates only.

Phase 2: review-gated standalone ingest through the existing Media ingestion surface, using existing PDF/media helpers and the existing web/context extraction pipeline for HTML full-text candidates.

Phase 3: Deep Research collection uses the discovery chokepoint and source review checkpoint exposes Media ingest eligibility.

Phase 4: checkpoint-approved Deep Research Media ingest job before synthesis.

Phase 5: existing compatibility endpoints delegate to the chokepoint where safe, and fallback site search is enabled source-by-source.

## Implementation Planning Scope

Phase 1 is implemented. The next implementation plan should cover Phase 2 only by default: review-gated standalone Media ingest handoff for discovery-selected `pdf` and `html_full_text` candidates, using persisted discovery snapshots and existing Media extraction/persistence paths. Deep Research integration, compatibility wrapper delegation, and fallback site-search rollout should remain later phases unless the human requester asks for a larger plan.

## Acceptance Criteria

### Phase 1 Acceptance Criteria

- A single internal discovery service is the shared path for new standalone discovery.
- Phase 1 supports the approved open research graph provider set through API-backed adapters where available.
- Source catalog entries expose capabilities instead of requiring clients to guess behavior.
- Discovery responses preserve normalized metadata, source provenance, adapter/catalog versions, warnings, OA candidates, and persisted `discovery_id` snapshots.
- OA candidate URLs exposed in API responses and snapshots are safe URLs/display URLs or opaque resolver references; raw signed/token-bearing URLs are not exposed.
- Fallback site search remains disabled by default and opt-in where configured.

### Phase 2 Acceptance Criteria

- No public research-owned ingest endpoint, router, or action service is added.
- Discovery adds only an internal selection resolver that returns bounded source descriptors and does not download, parse, dedupe, persist, or return ingest outcomes.
- Standalone discovery ingest is submitted through the existing Media ingestion surface.
- Media owns public request authorization, quota/rate-limit checks, duplicate decisions, egress checks, content-type/size checks, extraction, persistence, and per-item outcomes.
- Foreign-user, expired, missing, tampered, and unsupported discovery snapshot selections are rejected before any download.
- Phase 2 accepts only `pdf` and `html_full_text` candidates.
- HTML full-text candidates use the existing Media web/context extraction pipeline and do not store full raw HTML.
- Synchronous handoff enforces the Phase 2 caps for candidate count, per-item timeout, total timeout, download size, and MIME/content type.
- Over-cap requests fail validation instead of silently truncating selections or enqueueing background work.
- Tests prove the discovery handoff does not expose duplicate parser, chunking, HTML extraction, metadata override, or persistence options.

### Overall Design Acceptance Criteria

- A single internal discovery service is the shared path for new standalone discovery and Deep Research source collection.
- The first slice supports the approved open research graph provider set through API-backed adapters where available.
- Source catalog entries expose capabilities instead of requiring UI or Deep Research code to guess behavior.
- Discovery responses preserve normalized metadata, source provenance, adapter/catalog versions, warnings, and OA candidates.
- Ingest is explicit, revalidated, idempotent, Media-owned, and review-gated for Deep Research.
- Fallback site search is opt-in and provenance-labeled.
- Existing provider-specific endpoints remain compatible.
