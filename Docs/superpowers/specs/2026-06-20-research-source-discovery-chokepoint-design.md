# Research Source Discovery Chokepoint Design

Date: 2026-06-20
Backlog task: TASK-2336
Reference directory: https://www.sourclip.com/resources/research-sources

## Summary

Add a shared research discovery chokepoint for open research graph sources. The new design consolidates the repo's scattered paper/source searches behind one internal module while preserving existing provider-specific endpoints as compatibility surfaces.

The staged design focuses on open research graph discovery: OpenAlex, Semantic Scholar, Crossref, arXiv, PubMed, Zenodo, Figshare, OSF, and Unpaywall-style open-access resolution. Standalone search and Deep Research both consume the same normalized discovery service. Users can review search results, resolve open-access/full-text candidates, and ingest approved items into Media DB. Deep Research uses the same service during collection and performs ingestion only after a source review checkpoint is approved. The first implementation plan should target Phase 1 unless the human requester explicitly asks for a multi-phase implementation plan.

Sourclip is used as inspiration and seed material for a curated local catalog. It is not a runtime dependency.

## Goals

- Create one chokepoint module for research/media source discovery instead of continuing to grow independent provider paths.
- Keep user-facing "source" concepts separate from implementation-facing provider adapters.
- Support standalone search and Deep Research through the same backend service and normalized result contract.
- Start with API-backed open research graph providers, using official/public APIs where available.
- Allow configurable per-source fallback site search, disabled by default unless explicitly enabled.
- Attach open-access/full-text candidates during discovery, then revalidate legality, URL policy, content type, size, and duplicate status at ingest time.
- Make Deep Research ingest review-gated and idempotent.
- Preserve existing `/api/v1/paper-search/*` endpoints while allowing safe internal delegation over time.

## Non-Goals

- Do not implement every source from the Sourclip directory in the first slice.
- Do not make Sourclip scraping part of runtime behavior.
- Do not remove or break existing provider-specific paper-search endpoints.
- Do not treat web fallback results as official provider data.
- Do not auto-ingest sources during Deep Research collection.
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
- persist a user-owned discovery snapshot for standalone ingest handoff

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

Discovery-time OA candidates are advisory. Ingest must revalidate every candidate before download.

### ResearchIngestActionService

`ResearchIngestActionService` takes selected normalized result ids and performs explicit ingest.

Responsibilities:

- revalidate source fingerprint and selected OA/full-text candidate
- check duplicate DOI, PMID, PMCID, arXiv id, provider id, and canonical URL before downloading large files
- enforce centralized egress/SSRF policy
- validate URL scheme and host
- validate content type and file size
- validate access/license hints where available
- download/process approved items through existing paper/media ingestion helpers
- persist Media DB references and per-item ingest outcomes
- return partial success/failure results without exposing sensitive provider details

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
- ingest eligibility and recommended ingest candidate id when available
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
- Ingest never relies on client-resubmitted metadata as authority. It loads the server-side discovery snapshot or Deep Research artifact, then revalidates source identity and full-text candidates before download.

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
- candidate type, such as `pdf`, `html_full_text`, `repository_file`, or `landing_page`
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
- Ingest selections must identify both `result_id` and `candidate_id`; the service should not silently choose among multiple full-text candidates.
- A result may expose `recommended_candidate_id` for UI convenience, but the ingest request/checkpoint approval still records the explicit candidate selected.

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

The server persists the normalized result snapshot for the authenticated user with a bounded retention period. The first implementation should use this persisted snapshot as the only standalone ingest handoff model. Client-signed result payloads and stateless recomputation are out of scope for the first slice.

### `POST /api/v1/research/discovery/ingest`

Ingests selected approved results.

Input shape:

- `discovery_id`
- selected ingest candidates as `{ result_id, candidate_id }`
- optional target collection/tags/keywords
- per-request ingest limits

Output shape:

- per-item ingest status
- Media DB references for successes
- per-item warnings/failures
- aggregate counts

The endpoint performs revalidation and does not trust discovery-time download URLs blindly.

The ingest endpoint verifies that the `discovery_id` belongs to the current user, is still within retention, and contains the requested `{ result_id, candidate_id }` pairs. Idempotency is keyed by owner user id, discovery id or Deep Research session id, normalized fingerprint, and selected candidate id.

## Standalone Search Flow

1. Client calls `GET /api/v1/research/sources`.
2. Client submits `POST /api/v1/research/discovery/search`.
3. `ResearchDiscoveryService` resolves source selections through the catalog.
4. The router calls provider adapters or allowed fallback search.
5. Results are normalized, deduped, ranked, and enriched with OA candidates.
6. API persists a user-owned discovery snapshot and returns `discovery_id`, provenance-rich metadata, and ingest eligibility.
7. User selects results and calls `POST /api/v1/research/discovery/ingest` with `discovery_id` and selected `{ result_id, candidate_id }` pairs.
8. Ingest service revalidates identifiers, policy, URLs, content, size, access hints, and duplicates.
9. Approved files are processed through existing paper/media ingestion helpers and persisted to Media DB.

## Deep Research Flow

Run creation may include `provider_overrides.discovery` with source/category selection and fallback policy.

Collection phase:

1. Deep Research calls `ResearchDiscoveryService` for each focus area.
2. It writes normalized sources, evidence notes, OA candidates, discovery warnings, effective config, and catalog version into existing artifacts.
3. It presents deduped sources and ingest eligibility in the source review checkpoint.

After checkpoint approval:

1. The approved `{ result_id, candidate_id }` ingest selections are resolved from the run's persisted source artifacts and submitted to a separate idempotent ingest job/phase.
2. The ingest job uses `ResearchIngestActionService`.
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
- does not scrape full text unless the source is explicitly ingestable and ingest-time policy permits it
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

Discovery and ingest must respect API terms, configured rate limits, and egress policy.

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

Ingest controls:

- duplicate checks before large downloads
- centralized URL policy checks
- content-type checks
- file-size caps
- license/access hint checks where available
- idempotency keyed by discovery id or run/session id plus normalized source fingerprint and selected candidate id

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

Unit tests:

- catalog id/category/capability validation
- catalog versioning
- disabled and credential-required source behavior
- source/category selection resolution
- caps, timeouts, and fallback policy enforcement
- partial failures and warning separation
- dedupe priority and deterministic ranking
- OA candidate attachment
- sanitized provenance metadata
- source router behavior with fake adapters

Provider adapter tests:

- mocked HTTP/client functions for first-slice sources
- reuse and extend existing sanitizer tests for OpenAlex, Semantic Scholar, Crossref, PubMed, arXiv, Zenodo, Figshare, OSF, and Unpaywall-style lookup

Ingest tests:

- URL revalidation
- duplicate checks
- policy-blocked URLs
- unsupported content types
- size caps
- partial ingest failures
- successful routing into existing Media DB ingestion helpers

Integration tests:

- `GET /api/v1/research/sources`
- `POST /api/v1/research/discovery/search`
- `POST /api/v1/research/discovery/ingest`
- Deep Research source review checkpoint includes normalized results and ingest eligibility
- checkpoint approval triggers an idempotent ingest phase
- synthesis proceeds with metadata and Media DB references when available

## Rollout Plan

Phase 1: catalog, source router, discovery chokepoint, standalone search API, metadata and OA candidates only.

Phase 2: review-gated standalone ingest using existing paper/media helpers.

Phase 3: Deep Research collection uses the discovery chokepoint and source review checkpoint exposes ingest eligibility.

Phase 4: checkpoint-approved Deep Research ingest job before synthesis.

Phase 5: existing compatibility endpoints delegate to the chokepoint where safe, and fallback site search is enabled source-by-source.

## Implementation Planning Scope

The next implementation plan should cover Phase 1 only by default: catalog, source router, discovery chokepoint, standalone search API, normalized metadata, OA candidate discovery, persisted discovery snapshots, and tests for that surface. Standalone ingest, Deep Research integration, compatibility wrapper delegation, and fallback site-search rollout should remain later phases unless the human requester asks for a larger plan.

## Acceptance Criteria

### Phase 1 Acceptance Criteria

- A single internal discovery service is the shared path for new standalone discovery.
- Phase 1 supports the approved open research graph provider set through API-backed adapters where available.
- Source catalog entries expose capabilities instead of requiring clients to guess behavior.
- Discovery responses preserve normalized metadata, source provenance, adapter/catalog versions, warnings, OA candidates, and persisted `discovery_id` snapshots.
- OA candidate URLs exposed in API responses and snapshots are safe URLs/display URLs or opaque resolver references; raw signed/token-bearing URLs are not exposed.
- Fallback site search remains disabled by default and opt-in where configured.

### Overall Design Acceptance Criteria

- A single internal discovery service is the shared path for new standalone discovery and Deep Research source collection.
- The first slice supports the approved open research graph provider set through API-backed adapters where available.
- Source catalog entries expose capabilities instead of requiring UI or Deep Research code to guess behavior.
- Discovery responses preserve normalized metadata, source provenance, adapter/catalog versions, warnings, and OA candidates.
- Ingest is explicit, revalidated, idempotent, and review-gated for Deep Research.
- Fallback site search is opt-in and provenance-labeled.
- Existing provider-specific endpoints remain compatible.
