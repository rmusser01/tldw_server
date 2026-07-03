# Standalone MCP Docs Stage 4B Bounded Source Discovery Design

Date: 2026-07-03
Status: Draft for user review
Backlog: TASK-12121
Related:
- Docs/superpowers/specs/2026-06-30-standalone-mcp-docs-catalog-design.md
- Docs/superpowers/specs/2026-06-30-standalone-mcp-docs-url-acquisition-design.md
- Docs/superpowers/specs/2026-07-01-standalone-mcp-docs-stage4a-sync-source-design.md
- Docs/superpowers/plans/2026-07-02-standalone-mcp-docs-stage4a-source-sync-implementation-plan.md

## Summary

Stage 4B adds explicit, bounded source discovery to the standalone MCP docs
corpus. The new discovery slice lets an agent inspect a sitemap or one approved
HTML seed page, see candidate document URLs, and then optionally register or
ingest a capped set of accepted pages into the existing SQLite + FTS5 corpus.

This is not a broad crawler. Stage 4A already added a source registry and
`docs.sync_source` for known local and URL page sources. Stage 4B fills the
deferred gap: how a standalone MCP server safely turns a known website entry
point into a small document collection without requiring a `tldw_server`
instance, browser automation, or mandatory scraping dependencies.

## Goals

- Add a bounded discovery service for explicit sitemap URLs and explicit HTML
  seed pages.
- Expose a `docs.discover_source` MCP tool with dry-run and apply modes.
- Let dry-run return normalized, redacted, policy-filtered candidate URLs
  without mutating the corpus.
- Let apply register a refreshable `url_sitemap` source, ingest accepted page
  candidates through `DocsAcquisitionService.ingest_url()`, or both.
- Extend `docs.sync_source` to refresh registered `url_sitemap` sources when
  `sitemap_sync_enabled=true`.
- Reuse Stage 2 source policy, DNS/IP guards, redirect handling, extraction,
  content limits, query redaction, and fake transport/resolver seams.
- Reuse Stage 4A source records and source-document links so discovered pages
  are visible through `docs.list(kind="sources")` and refreshable through
  `docs.sync_source`.
- Keep web scraping optional: when installed, `beautifulsoup4` is preferred for
  HTML link extraction and `trafilatura` remains preferred for page body
  extraction through the Stage 2 ingestion path. Both remain lazy optional
  imports, not baseline dependencies.
- Preserve locked-down deployment profiles where URL acquisition and discovery
  are unavailable unless explicitly configured.
- Keep `mcp_unified.docs` independent from `tldw_Server_API` runtime imports.

## Non-Goals

- No broad recursive crawler.
- No automatic site-wide crawl from an arbitrary domain.
- No browser automation, Playwright, browser profile access, cookies,
  JavaScript rendering, login/session reuse, or stealth scraping behavior.
- No robots.txt auto-discovery unless a standalone robots checker is added in
  the same implementation with fail-closed tests. Until then,
  `respect_robots=true` continues to fail closed.
- No mandatory `beautifulsoup4`, `trafilatura`, requests, httpx, aiohttp, or
  Playwright dependency.
- No embeddings, vector index, reranking, answer generation, Media DB bridge,
  ChromaDB bridge, Jobs/Scheduler worker, or `tldw_server` RAG service coupling.
- No persisted query-bearing refreshable source unless
  `persist_url_query_strings=true`.
- No background recurring discovery.

## Approaches Considered

### Recommended: Explicit Discovery Tool Plus Existing Ingestion

Add `docs.discover_source` as the only new Stage 4B tool. It fetches an
explicit sitemap URL or HTML seed page under the existing Stage 2 URL policy,
returns candidates in dry-run mode, and in apply mode calls existing ingestion
paths for accepted pages. Registered sitemap sources are then refreshed through
the existing `docs.sync_source` tool. This keeps arbitrary seed discovery out
of `docs.sync_source` while still making approved sitemap sources refreshable.

Trade-off: this adds one new tool contract and one new `url_sitemap` branch in
`docs.sync_source`, but it avoids overloading `docs.sync_source` with arbitrary
seed URLs.

### Minimal: Only Enable `url_sitemap` In `docs.sync_source`

This would implement Stage 4A's deferred sitemap sync path by adding a narrow
sitemap registration mechanism and sitemap handling inside `docs.sync_source`.
It is smaller, but it does not solve HTML page link discovery and gives agents
less control over candidate review before ingestion.

### Rejected: General Crawler Or Host Scraping Bridge

A general crawler, browser scraper, or direct `tldw_server` scraping bridge
would ingest more pages with less prompting, but it violates the standalone
package boundary and creates avoidable security, privacy, and dependency risk.

## Current Baseline

The merged Stage 4A branch includes:

- `DocsSettings` with web acquisition, source sync, sitemap sync, and query
  persistence settings.
- `DocsAcquisitionService.ingest_url()` for one approved URL.
- `URLFetcher` with policy checks, resolver/transport seams, manual redirects,
  SSRF defenses, content-type checks, and byte limits.
- Optional lazy extraction through `trafilatura` and BeautifulSoup, with stdlib
  fallback extraction.
- `docs_sources`, `docs_source_documents`, and `docs_sync_runs` tables.
- `docs.list(kind="sources")`.
- `docs.sync_source` for `local_file`, `local_directory`, and `url_page`.
- A disabled `url_sitemap` source type that returns `sitemap_sync_disabled`
  unless explicitly enabled.

Stage 4B should build on those seams. It should not introduce a second HTTP
stack or an alternate ingestion path.

## Settings

Add discovery-specific settings to `DocsSettings`.

- `enable_source_discovery`: default `false`. Tool is not advertised unless
  both this and `enable_web_acquisition` are true.
- `max_discovery_pages`: default `25`. Hard cap for accepted page candidates
  and ingested pages.
- `max_discovery_depth`: default `1`. Stage 4B.1 supports only explicit
  sitemap entries and one-hop HTML links from a seed page; higher values are
  rejected until a later crawler slice exists.
- `max_discovery_sitemaps`: default `3`. Only needed if sitemap index support
  is implemented; otherwise keep sitemap index support deferred.
- `discovery_apply_default`: default `register`. Allowed values:
  `register`, `ingest`, or `register_and_ingest`.
- `discovery_same_origin_only`: default `true`; Stage 4B should treat `false`
  as unsupported unless the URL is also under an explicit allowed URL prefix.

Profile behavior:

- `locked_down`: discovery is disabled by default. If enabled, it may fetch
  only explicit `allowed_url_prefixes`; domain-only allow rules are not enough.
- `local_first`: discovery may use configured `preapproved_domains` and
  `allowed_url_prefixes`; unknown public domains return `approval_required`.
- `online_capable`: same as `local_first`, plus unknown public domains only
  when `allow_arbitrary_public_domains=true`.

`sitemap_sync_enabled` remains default `false`. Stage 4B may allow sitemap
source registration and initial ingestion when source discovery is enabled, but
repeated refresh through `docs.sync_source` still returns
`sitemap_sync_disabled` until this flag is true. Registration responses should
include that warning when they create a sitemap source that is not yet
syncable.

## Tool Contract

### `docs.discover_source`

`docs.discover_source` is an ingestion-category tool because apply mode writes
sources and documents. It is advertised only when source discovery and web
acquisition are enabled.

Input schema:

- `url`: required string. Explicit sitemap URL or HTML seed page URL.
- `kind`: `auto`, `sitemap`, or `page_links`; default `auto`.
- `mode`: `dry_run` or `apply`; default `dry_run`.
- `apply_action`: `register`, `ingest`, or `register_and_ingest`; default from
  settings.
- `max_pages`: optional positive integer capped by `max_discovery_pages`.
- `max_depth`: optional positive integer; Stage 4B.1 accepts only `1`.
- `collections`: optional array of collection names applied to ingested pages.
- `keywords`: optional array of keywords applied to ingested pages.
- `title`: optional display title for the registered discovery source.
- `include_seed`: optional boolean; default `false` for page-link discovery.
  When true, the seed page can be ingested along with discovered links.

Selector rules:

- `url` must pass the existing Stage 2 source policy before any network I/O.
- For `kind=auto`, a URL whose path ends in `.xml` or whose fetched content
  type is XML is treated as `sitemap`; otherwise it is `page_links`.
- `mode=dry_run` performs network fetches but must not mutate the docs store.
- `mode=apply` may create or update a `url_sitemap` source and/or ingest
  accepted candidates through `DocsAcquisitionService.ingest_url()`.
- `apply_action=ingest` without `register` still creates normal `url_page`
  sources for each ingested page because `ingest_url()` already owns that path.
- `apply_action=register` creates a refreshable `url_sitemap` source for
  sitemap discovery and does not ingest candidate pages.
- `apply_action=register` is invalid for `page_links` unless `include_seed=true`;
  in that narrow case it may register the seed as a `url_page` source. It must
  not persist a discovered-link graph.

Response shape:

```json
{
  "status": "completed",
  "reason_code": "ok",
  "mode": "dry_run",
  "kind": "sitemap",
  "source": {
    "id": null,
    "source_type": "url_sitemap",
    "canonical_uri": "https://example.com/sitemap.xml",
    "display_uri": "https://example.com/sitemap.xml"
  },
  "counts": {
    "accepted": 12,
    "duplicates": 1,
    "denied": 0,
    "skipped": 4,
    "ingested": 0,
    "failed": 0
  },
  "candidates": [
    {
      "url": "https://example.com/docs/install",
      "display_url": "https://example.com/docs/install",
      "safe_argument_hash": "ab12...",
      "status": "accepted",
      "reason_code": "ok",
      "source_kind": "sitemap",
      "parent_url": "https://example.com/sitemap.xml"
    }
  ],
  "warnings": []
}
```

Candidate `url`, `display_url`, and `parent_url` fields are public display
values, not privileged fetch logs. They must use redacted URL forms whenever
query strings are present. `safe_argument_hash` is the stable correlation value
for a query-bearing candidate. Apply-mode responses include `source.id`,
per-candidate ingest status, document IDs when ingestion succeeds, and bounded
warnings. Raw query-bearing fetch URLs must not appear in public tool responses
even when `persist_url_query_strings=true`; a future privileged diagnostic path
would be required for that. Stage 4B does not add that diagnostic path.

## Discovery Semantics

### Sitemap Discovery

Sitemap discovery handles an explicit sitemap URL. Stage 4B.1 should support
`urlset` sitemaps with `<url><loc>...</loc></url>` entries.

Rules:

- Fetch the sitemap URL through `URLFetcher`.
- Require an XML-ish content type or a `.xml` path when `kind=sitemap` is
  explicitly requested; return `sitemap_content_type_denied` otherwise.
- Reject bodies containing `DOCTYPE` or `ENTITY` before XML parsing.
- Use stdlib XML parsing configured with no external entity resolution.
- Ignore `lastmod`, `changefreq`, and `priority` for ingestion decisions in
  Stage 4B. Store them only as candidate metadata if they are cheap to retain.
- Enforce `max_pages` while collecting `<loc>` entries and before page fetches.
- Normalize, dedupe, and policy-check each URL before ingestion.
- Require same-origin with the sitemap URL, unless the target URL is under an
  explicit allowed URL prefix.
- Skip query-bearing candidates unless `persist_url_query_strings=true`.
- Do not fetch candidate pages in dry-run mode.
- In apply mode, fetch candidate pages only by calling `ingest_url()` so Stage 2
  SSRF, redirect, content type, extraction, and source population behavior is
  reused.

Sitemap index support is optional for Stage 4B.1. If implemented, it must be
bounded by `max_discovery_sitemaps`, require same-origin or allowed prefixes for
child sitemap URLs, and apply the same XML `DOCTYPE`/`ENTITY` rejection to every
child sitemap. Otherwise return `sitemap_index_unsupported`.

### Page-Link Discovery

Page-link discovery handles one explicit HTML seed page and extracts links from
that page only.

Rules:

- Fetch the seed URL through `URLFetcher`.
- Accept only HTML content types for link discovery.
- Prefer BeautifulSoup for link extraction when `beautifulsoup4` is installed,
  imported lazily inside the extractor. Fall back to a stdlib `html.parser`
  extractor when BeautifulSoup is unavailable.
- Normalize relative links against the final fetched seed URL.
- Drop fragments before dedupe.
- Skip unsupported schemes such as `mailto:`, `tel:`, `data:`, `file:`, and
  JavaScript URLs.
- Skip links with `rel=nofollow` in Stage 4B.1 if the extractor exposes the
  attribute cheaply; otherwise do not attempt full robots semantics.
- Require same-origin with the seed URL unless the target URL is under an
  explicit allowed URL prefix.
- Enforce path-prefix boundaries when an allowed URL prefix matched the seed.
- Enforce `max_pages` before candidate page fetches.
- Do not recursively fetch discovered pages in Stage 4B.1. `max_depth` exists
  to make the boundary explicit, not to implement a crawler in this slice.

When `include_seed=true` and `mode=apply`, the seed page is ingested through
`ingest_url()` before discovered link candidates, and it counts against the
effective page cap.

## Data Model

Use the Stage 4A tables first.

For sitemap discovery:

- `docs_sources.source_type = "url_sitemap"` stores the sitemap source.
- `docs_sources.canonical_uri` is the normalized sitemap URL.
- `docs_sources.source_url` stores the fetch-capable sitemap URL only when
  query persistence policy allows it.
- `docs_sources.metadata_json` stores discovery defaults:
  `default_keywords`, `default_collections`, `discovery_kind`,
  `same_origin_only`, and bounded path prefixes if provided.
- Each ingested page remains a normal `url_page` source from `ingest_url()`.
- The sitemap source may link to ingested page documents through
  `docs_source_documents` using each page canonical URL as `source_item_uri`.

For page-link discovery:

- A separate `url_sitemap` source should not be created.
- If apply mode registers without candidate ingestion, it may create or update a
  `url_page` source for the seed page only when `include_seed=true`.
- If link candidates are ingested, each candidate becomes a normal `url_page`
  source through `ingest_url()`.
- Candidate link sets are not persisted as a crawl graph in Stage 4B. Page-link
  discovery is a one-shot helper for creating small collections from an
  explicit seed page.

Stage 4B should not add a broad crawler state table. If audit history is needed
for apply mode, store bounded discovery summaries in the parent source
metadata or add a narrowly scoped `docs_discovery_runs` table only after the
implementation proves `docs_sync_runs` cannot represent the result. Dry-run
must remain non-mutating.

## Sitemap Source Sync

Stage 4B should make registered `url_sitemap` sources refreshable through
`docs.sync_source` when `sitemap_sync_enabled=true`.

Rules:

- `docs.sync_source` for `url_sitemap` reuses the same sitemap parser,
  candidate normalization, source policy, same-origin checks, query redaction,
  and page caps as `docs.discover_source`.
- Dry-run re-fetches the sitemap and reports candidate changes without mutating
  documents, chunks, source links, source rows, or sync-run rows.
- Apply mode ingests accepted current candidates through
  `DocsAcquisitionService.ingest_url()` and links successful documents to the
  sitemap source with the page URL as `source_item_uri`.
- Sitemap source defaults in `docs_sources.metadata_json`, including
  `default_keywords` and `default_collections`, are passed into each page
  ingestion call and merged with any existing user organization using the Stage
  4A sync-aware merge semantics.
- Active sitemap links absent from the current sitemap are reported as
  `missing` under `stale_policy=report`.
- Under `stale_policy=tombstone`, apply mode tombstones sitemap source links
  using the Stage 4A source-link tombstone path. It must not hard-delete
  documents.
- `max_pages`, `max_documents`, and `max_sync_run_items` all cap work before
  page fetches.
- Repeated sitemap sync does not recurse into discovered HTML links. Each page
  is still a single `url_page` source if it was ingested.

This keeps recurring refresh tied to explicit sitemap membership while leaving
recursive crawl refresh for a later, separately reviewed feature.

## Source Policy And Security

Discovery must inherit Stage 2 URL policy and Stage 4A redaction rules:

- Policy evaluation happens before every network request.
- Resolver and transport seams must be injectable and tested with fake objects.
- Private, loopback, link-local, multicast, unspecified, and reserved IP ranges
  remain denied.
- Redirects are manual and every redirect target must pass policy, DNS, and IP
  checks.
- No fetch happens for `approval_required` results.
- URL credentials remain denied.
- Query strings are redacted in display/logging/tool responses and are not
  persisted in refreshable sources unless `persist_url_query_strings=true`.
- Candidate pages beyond caps are counted as skipped and never fetched.
- Candidate result arrays are bounded by `max_discovery_pages` and
  `max_sync_run_items` style limits.
- Discovery must never import `tldw_Server_API`, Media DB, ChromaDB, RAG, or
  host scraping modules from `mcp_unified.docs`.

If `respect_robots=true`, discovery must fail closed with `robots_unavailable`
before fetching sitemap or page content until a standalone robots checker is
implemented with deterministic tests.

## Error Handling And Reason Codes

Add stable reason codes where missing:

- `source_discovery_disabled`
- `source_discovery_request_invalid`
- `source_discovery_kind_unsupported`
- `source_discovery_limit_exceeded`
- `source_discovery_no_candidates`
- `sitemap_content_type_denied`
- `sitemap_fetch_failed`
- `sitemap_parse_failed`
- `sitemap_index_unsupported`
- `sitemap_xml_forbidden_doctype`
- `sitemap_xml_forbidden_entity`
- `sitemap_sync_disabled`
- `candidate_out_of_scope`
- `candidate_query_not_persisted`
- `candidate_duplicate`
- `page_link_content_type_denied`
- `page_link_registration_unsupported`
- `robots_unavailable`

Apply mode should return `partial` when at least one candidate ingests and at
least one accepted candidate fails. It should return `failed` only when no
candidate can be processed after discovery succeeds.

## MCP And Permission Behavior

- `docs.discover_source` is category `ingestion` with `readOnlyHint=false`.
- `docs.status` reports source discovery availability, effective caps,
  supported discovery kinds, and disabled reason.
- Stale clients that call `docs.discover_source` while disabled receive
  `source_discovery_disabled`.
- Scope handling uses the existing `AccessScope` object and store-level owner
  and profile filters.
- Host `DocsModule` only forwards the tool and settings. It must not add
  scraping, RAG, Media DB, Jobs, or Scheduler business logic.

## Testing Strategy

Unit tests:

- settings parse source discovery flags and caps;
- `docs.status` reports discovery disabled by default and enabled when
  configured;
- provider advertises `docs.discover_source` only when discovery and web
  acquisition are enabled;
- stale disabled tool call returns `source_discovery_disabled`;
- sitemap XML parser rejects `DOCTYPE` and `ENTITY`;
- sitemap parser enforces `max_pages` before page fetches;
- sitemap parser rejects or skips out-of-scope candidates;
- page-link extractor normalizes relative links, drops fragments, skips
  unsupported schemes, and dedupes candidates;
- page-link extractor prefers BeautifulSoup when installed and falls back to
  stdlib parsing without making `beautifulsoup4` mandatory;
- query-bearing candidates are skipped unless query persistence is enabled;
- query-bearing candidates never expose raw query strings in public tool
  response fields;
- dry-run does not mutate source, document, chunk, keyword, collection, or run
  tables.

Integration tests:

- dry-run sitemap discovery with fake resolver/transport returns accepted
  candidates and no documents;
- apply sitemap discovery with fake sitemap and fake page bodies ingests
  bounded pages and creates searchable FTS5 chunks;
- sitemap discovery and sitemap sync apply registered default collections and
  keywords to ingested pages without replacing user-added metadata;
- apply register mode creates a `url_sitemap` source without ingesting pages;
- `docs.sync_source` refreshes a registered `url_sitemap` source when
  `sitemap_sync_enabled=true`;
- `docs.sync_source` reports or tombstones stale sitemap source links according
  to `stale_policy`;
- page-link discovery ingests one-hop same-origin links and applies requested
  collections/keywords;
- `docs.sync_source` can refresh `url_page` sources created by discovery;
- host `DocsModule` exposes the tool without importing Media/RAG services.

Boundary and security tests:

- no live internet;
- no top-level imports of BeautifulSoup, trafilatura, requests, httpx, aiohttp,
  Playwright, ChromaDB, RAG, or `tldw_Server_API` from `mcp_unified.docs`;
- no network call when source policy returns approval-required;
- redirect-to-private denial still applies during discovery;
- respect-robots fail-closed behavior applies before sitemap/page fetch;
- tool responses never leak query strings by default.

## Implementation Slices

1. Settings, status, and tool advertisement for disabled/enabled discovery.
2. Candidate models and pure URL normalization/dedupe helpers.
3. Safe sitemap fetch and parser for explicit `urlset` sitemaps.
4. Safe one-hop HTML link extraction from explicit seed pages.
5. `DocsSourceDiscoveryService.discover_source()` dry-run path.
6. Apply register mode for `url_sitemap` sources.
7. Apply ingest/register-and-ingest mode using `DocsAcquisitionService`.
8. `docs.sync_source` support for registered `url_sitemap` sources.
9. Host `DocsModule` forwarding and import-boundary tests.

If the implementation needs to split into multiple PRs, stop after slice 6.
That delivers reviewable discovery and source registration while deferring
bulk page ingestion and repeated sitemap sync to follow-ups.

## Open Decisions For Implementation Planning

- Whether Stage 4B.1 should support sitemap indexes or explicitly return
  `sitemap_index_unsupported`.
- Whether apply mode should persist bounded discovery run summaries in
  `docs_sync_runs.metadata_json` or add a small `docs_discovery_runs` table.
- Whether page-link extraction should expose additional BeautifulSoup-derived
  anchor metadata beyond URL and rel filtering.

Recommended defaults:

- Defer sitemap indexes unless the first implementation remains small.
- Avoid a new discovery-runs table until there is a concrete audit UI/API need.
- Prefer BeautifulSoup when installed for link extraction, with a stdlib parser
  fallback so the standalone package still works without optional web extras.
