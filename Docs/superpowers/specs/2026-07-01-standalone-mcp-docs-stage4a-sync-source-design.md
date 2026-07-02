# Standalone MCP Docs Stage 4A Bounded Source Sync Design

Date: 2026-07-01
Status: Draft for user review
Backlog: TASK-12091
Related:
- Docs/superpowers/specs/2026-06-30-standalone-mcp-docs-catalog-design.md
- Docs/superpowers/specs/2026-06-30-standalone-mcp-docs-url-acquisition-design.md
- Docs/superpowers/plans/2026-07-01-standalone-mcp-docs-stage3-server-mounting-plan.md

## Summary

Stage 4A adds bounded source refresh to the standalone MCP docs corpus. The
new `docs.sync_source` tool refreshes sources that the corpus already knows
about, such as a trusted local file, trusted local directory, approved URL
page, or explicitly configured sitemap source. It is not a crawler, browser
automation layer, embedding pipeline, reranker, job worker, or
`tldw_server` Media/RAG bridge.

The main architectural change is to make source identity explicit. Stage 1 and
Stage 2 store provenance on each document, but `docs.list(kind="sources")`
still returns only a Stage 1 warning. Stage 4A introduces scoped source records
and sync run records so agents can see what can be refreshed, run a dry-run,
apply a bounded refresh, and receive stable item-level results.

## Goals

- Add a `docs_sources` registry for local and URL-backed sources.
- Populate source records from `docs.import_path` and `docs.ingest_url`.
- Expose `docs.list(kind="sources")` with real source records.
- Add `docs.sync_source` for scoped, bounded refresh of existing sources.
- Support dry-run and apply modes with deterministic item-level results.
- Reuse Stage 1 local path scope checks for local sync.
- Reuse Stage 2 source policy, DNS/IP guards, redirect handling, extraction,
  and content limits for URL sync.
- Keep stale/missing handling conservative by default: report first, tombstone
  only when explicitly requested.
- Keep baseline standalone installs dependency-light and FTS5-only.
- Preserve the `mcp_unified.docs` import boundary: no `tldw_Server_API`
  imports, no eager optional web/browser imports.

## Non-Goals

- No arbitrary URL or path crawling from `docs.sync_source`.
- No broad recursive link discovery.
- No Playwright, browser profile, cookies, session cloning, or authenticated
  web capture in this slice.
- No embeddings, vector indexes, reranking, or answer generation.
- No Jobs or Scheduler implementation. A host can wrap sync later.
- No Media DB, ChromaDB, RAG service, AuthNZ, or full `tldw_server` runtime
  dependency in `mcp_unified.docs`.
- No hard delete of missing documents by default.
- No background daemon or automatic recurring sync.

## Current Baseline

The current Stage 1-3 stacked branch already has:

- `DocsSettings` with trusted roots and web acquisition policy settings.
- `DocsImportService.import_path()` for trusted local file/tree import.
- `DocsAcquisitionService.ingest_url()` for one approved URL.
- `DocsCatalogStore.upsert_document()` with scoped documents, FTS5 chunks,
  collections, and keywords.
- `DocsMCPToolProvider` with `docs.search`, `docs.get`, `docs.context`,
  `docs.list`, `docs.import_path`, `docs.ingest_url`, and management tools.

The gap is source lifecycle. `docs_documents` stores `source_path`,
`source_url`, and `metadata_json`, but there is no first-class source table,
no sync run table, and `docs.list(kind="sources")` returns a
`sources_not_populated_in_stage1` warning.

## Source Model

Stage 4A should add explicit, scoped source records.

### `docs_sources`

Each source belongs to an owner/profile scope and has one stable identity.

Recommended fields:

- `id`
- `owner_scope`
- `profile_scope`
- `source_type`: `local_file`, `local_directory`, `url_page`, or
  `url_sitemap`
- `canonical_uri`: unique source identifier within the scope
- `display_name`
- `source_path`
- `source_url`
- `redacted_source_url`
- `policy_profile`
- `sync_enabled`
- `last_sync_status`
- `last_sync_started_at`
- `last_sync_completed_at`
- `last_error_code`
- `metadata_json`
- `created_at`
- `updated_at`

The unique key should be `(owner_scope, profile_scope, canonical_uri)`.

Canonical URI rules:

- Local files and directories use normalized `file://` URIs derived from the
  resolved trusted-root path.
- URL pages use the Stage 2 normalized canonical URL after redirects. A URL
  source is not created until policy allows a fetch and the fetch produces a
  canonical, fetch-capable URL.
- Sitemaps use the normalized sitemap URL.
- `source_url` stores the fetch target for URL-backed sources and may include
  query parameters needed to refresh the source.
- `redacted_source_url` stores the display/logging form. Tool results, source
  lists, errors, warnings, and sync-run metadata must use redacted URL forms
  unless the caller explicitly requests full source metadata through a future
  privileged diagnostic path. Stage 4A does not add that diagnostic path.
- Query strings may be part of `canonical_uri` and `source_url` for fetch
  identity, but logs, errors, audit summaries, and source list display must use
  redacted forms.

### `docs_source_documents`

A source can map to many documents, and a document can be reachable from more
than one source.

Recommended fields:

- `source_id`
- `document_id`
- `source_item_uri`
- `status`: `active`, `missing`, `tombstoned`, or `failed`
- `last_seen_at`
- `last_hash`
- `last_error_code`
- `metadata_json`

The unique key should be `(source_id, source_item_uri)`.

### `docs_sync_runs`

Sync runs record tool-level outcomes without requiring a background job system.

Recommended fields:

- `id`
- `owner_scope`
- `profile_scope`
- `source_id`
- `mode`: `dry_run` or `apply`
- `status`: `completed`, `partial`, `skipped`, `denied`, or `failed`
- `started_at`
- `completed_at`
- `requested_limits_json`
- `counts_json`
- `warnings_json`
- `error_code`
- `metadata_json`

Item-level results can live in `metadata_json` initially. If implementation
proves that large run histories need separate rows, a later slice can split
them into `docs_sync_run_items`.

## Tool Contract

### `docs.list(kind="sources")`

Returns scoped source records:

```json
{
  "sources": [
    {
      "id": 1,
      "source_type": "local_directory",
      "canonical_uri": "file:///repo/docs/",
      "display_name": "Project docs",
      "sync_enabled": true,
      "last_sync_status": "completed",
      "last_sync_completed_at": "2026-07-01T12:00:00Z",
      "document_count": 12,
      "metadata": {}
    }
  ]
}
```

The list must apply store-level owner/profile scope filters before returning
records.

### `docs.sync_source`

`docs.sync_source` is a write-capable ingestion tool.

Input schema:

- `source_id`: integer, preferred source selector.
- `source_uri`: string, alternative selector for `canonical_uri`.
- `mode`: `dry_run` or `apply`; default `dry_run`.
- `max_documents`: positive integer; default from settings.
- `max_pages`: positive integer for URL/sitemap sources; default from settings.
- `stale_policy`: `report` or `tombstone`; default `report`.
- `force`: boolean; default false. When false, unchanged content is counted
  without rewriting chunks.

Selector rules:

- Exactly one of `source_id` or `source_uri` must be supplied.
- The source must exist in the active access scope.
- The source must have `sync_enabled=true`.
- `docs.sync_source` must not create a new arbitrary source from an untracked
  path or URL. New sources are created by `docs.import_path` or
  `docs.ingest_url`.

Response shape:

```json
{
  "status": "completed",
  "reason_code": "ok",
  "mode": "apply",
  "source": {
    "id": 1,
    "source_type": "local_directory",
    "canonical_uri": "file:///repo/docs/"
  },
  "counts": {
    "created": 1,
    "updated": 2,
    "unchanged": 8,
    "missing": 1,
    "tombstoned": 0,
    "failed": 0,
    "skipped": 0
  },
  "items": [
    {
      "source_item_uri": "file:///repo/docs/index.md",
      "status": "updated",
      "document_id": 42,
      "reason_code": "ok"
    }
  ],
  "warnings": []
}
```

For `dry_run`, no documents, chunks, collection memberships, keywords, source
links, source status fields, source timestamps, or sync run records are
mutated. Dry-run audit persistence is deferred to a host wrapper or later
diagnostic feature so the Stage 4A standalone dry-run path is strictly
read-only.

## Metadata Preservation

Sync must not erase user organization.

- `docs.import_path` and `docs.ingest_url` should record their initial
  collection and keyword arguments as source defaults in `docs_sources`
  metadata.
- In apply mode, sync merges source-default collections and keywords with the
  document's existing collections and keywords.
- Sync must not call `DocsCatalogStore.upsert_document()` with empty keyword or
  collection arguments for an existing document unless the source explicitly has
  an empty replacement policy. Stage 4A does not add replacement policy.
- User-applied keywords and collection memberships survive unchanged content,
  updated content, missing-source report mode, and tombstone mode.
- If a future source wants authoritative replacement semantics, it needs a new
  explicit source metadata policy and separate tests.

## Local Source Sync

Local file and directory sync use Stage 1 import rules.

Local file sync:

1. Resolve the source path.
2. Enforce trusted-root path scope and symlink escape checks.
3. If the file is missing, return `missing`.
4. If the suffix is unsupported, return `failed` with
   `unsupported_import_format`.
5. Parse, chunk, hash, and compare with the linked document.
6. In apply mode, merge existing user collections/keywords with source-default
   collections/keywords.
7. In apply mode, upsert the document only when content or metadata changed,
   unless `force=true`.

Local directory sync:

1. Resolve the directory source path.
2. Enforce trusted-root path scope.
3. Enumerate supported files deterministically.
4. Enforce `max_documents` before parsing all candidates.
5. Merge existing user collections/keywords with source-default
   collections/keywords for each active document.
6. Upsert active files through the same parser/chunker path as import.
7. Compare enumerated item URIs with previously linked source items.
8. Report previously linked items that are no longer present as `missing`.
9. Only set source-item status to `tombstoned` when
   `stale_policy="tombstone"` and `mode="apply"`.

Directory sync should not recurse outside the source root even if symlinks
point elsewhere. Existing Stage 1 path checks remain the authority.

## URL Page Sync

URL page sync refreshes exactly one existing `url_page` source.

Rules:

- Requires `settings.enable_web_acquisition=true`.
- Reuses Stage 2 `SourcePolicy`, `URLFetcher`, and extraction service.
- Re-evaluates policy before every fetch.
- Does not fetch when policy returns `approval_required` or `denied`.
- Re-runs DNS/IP and redirect checks.
- Reuses content-type and body-size limits.
- Uses the final canonical URL as the source item URI.
- Does not follow page links.

If the refreshed final URL differs from the stored source canonical URI because
of redirects, the implementation should update source metadata with the final
URL in apply mode, but it must not silently widen policy. The redirected target
must still pass policy.

## Sitemap Source Sync

Sitemap support is allowed in Stage 4A only for explicit `url_sitemap` sources.
It is still not a crawler.

Rules:

- A sitemap source must already exist and must have a canonical sitemap URL.
- Fetch the sitemap URL through the same Stage 2 URL fetcher and policy checks.
- Parse XML with stdlib XML tooling configured to avoid external entity
  expansion.
- Reject sitemap bodies containing `DOCTYPE` or `ENTITY` declarations before
  XML parsing.
- Accept only `<url><loc>...</loc></url>` entries in Stage 4A.
- Ignore sitemap index recursion unless a later task explicitly adds it.
- Every discovered URL must pass the source policy and must be same-origin or
  under an explicitly allowed URL prefix for the source.
- Enforce `max_pages` while collecting `<loc>` entries and before page fetches;
  entries beyond the limit are counted as skipped without being fetched.
- Cap stored run item details at `max_sync_run_items`; additional item summaries
  are counted but not persisted into run metadata.
- Fetch each accepted page with Stage 2 guards.
- No JavaScript rendering, no cookies, no browser extraction.

If `respect_robots=true` and no standalone robots checker is available, sitemap
sync must fail closed with `robots_unavailable` before fetching page content.

## Stale And Tombstone Semantics

Default stale behavior is report-only.

- `stale_policy="report"`: missing items are returned in the response and run
  record, but existing documents remain searchable.
- `stale_policy="tombstone"`: apply mode marks the source item as
  `tombstoned`.
- Add `lifecycle_status` to `docs_documents` with default `active`.
- Add `preserve_on_source_tombstone` to `docs_documents` with default false.
  Stage 4A does not expose a public tool to set it; the field exists so future
  manual-preservation flows have a defined search rule.
- A document is hidden from default search only when all source links for that
  document are tombstoned or missing and `preserve_on_source_tombstone=false`.
- Legacy rows without source links remain `active` after migration.
- Hard delete is out of scope.

## Settings

Extend `DocsSettings` with source-sync limits:

- `enable_source_sync`: default true.
- `max_sync_documents`: default 500.
- `max_sync_pages`: default 25.
- `max_sync_run_items`: default 500.
- `default_stale_policy`: `report`.
- `sitemap_sync_enabled`: default false unless explicitly enabled.

Profile behavior:

- `locked_down`: local source sync can run for trusted roots; URL and sitemap
  sync return `capability_disabled` unless web acquisition is explicitly
  enabled with narrow URL prefixes.
- `local_first`: local sync is enabled; URL page sync is allowed only for
  preapproved domains or allowed URL prefixes; sitemap sync requires explicit
  `sitemap_sync_enabled=true`.
- `online_capable`: same as `local_first`, with unknown public domains allowed
  only when `allow_arbitrary_public_domains=true`; sitemap still requires
  `sitemap_sync_enabled=true`.

## Reason Codes

Stable reason codes should include:

- `ok`
- `source_not_found`
- `source_selector_invalid`
- `source_scope_denied`
- `source_sync_disabled`
- `source_sync_unsupported_type`
- `source_sync_limit_exceeded`
- `path_scope_denied`
- `import_path_not_found`
- `unsupported_import_format`
- `web_acquisition_disabled`
- `approval_required`
- `source_policy_denied`
- `fetch_failed`
- `extract_empty`
- `sitemap_sync_disabled`
- `sitemap_fetch_failed`
- `sitemap_parse_failed`
- `sitemap_url_out_of_scope`
- `sitemap_xml_forbidden_doctype`
- `sitemap_xml_forbidden_entity`
- `robots_unavailable`
- `stale_reported`
- `tombstoned`

Tool responses should expose reason codes at both run and item levels.

## MCP And Permission Behavior

- `docs.sync_source` is category `ingestion` and `readOnlyHint=false`.
- Listing sources is read-only.
- Sync must use the same access scope object as other docs tools.
- Store methods must enforce scope internally; callers cannot bypass scope by
  omitting filters.
- If a host later wraps sync in Jobs or Scheduler, the host must pass the same
  scope into the standalone service.

## Host Integration

The built-in `tldw_server` MCP server should expose the same tool through the
existing `DocsModule` shim and docs host adapter.

Stage 4A host integration remains thin:

- translate module config into `DocsSettings`;
- translate request context metadata into `AccessScope`;
- do not bridge Media DB or RAG stores;
- do not import host services from `mcp_unified.docs`;
- do not start background workers.

Jobs/Scheduler integration is a later host-specific wrapper. The standalone
service should be synchronous and deterministic in Stage 4A.

## Testing Strategy

Unit tests:

- source schema migration and legacy row compatibility;
- source creation from local import;
- source creation from URL ingest;
- URL source identity stores a fetch-capable `source_url` while source lists
  and run metadata expose only redacted URL forms;
- source listing with owner/profile scope enforcement;
- sync selector validation;
- dry-run does not mutate documents, source links, source timestamps,
  statuses, or sync-run rows;
- sync preserves existing user-applied collections and keywords while merging
  source defaults;
- local file sync created/updated/unchanged behavior;
- local directory sync deterministic enumeration and max-document limit;
- missing local file report mode;
- tombstone mode marks source item only in apply mode;
- URL sync no-fetch-before-approval;
- URL sync denial for source policy and private redirect cases;
- sitemap disabled failure;
- sitemap parsing with safe XML settings;
- sitemap `DOCTYPE` and `ENTITY` declaration rejection;
- sitemap `max_pages` enforcement before page fetches;
- sitemap out-of-scope URL rejection;
- source sync status reporting in `docs.status`.

Integration tests:

- import local directory, list source, dry-run sync, apply sync, search updated
  content;
- ingest approved URL, list source, sync source with fake transport;
- sitemap source sync with fake sitemap and fake page transport;
- `DocsModule` exposes `docs.sync_source` through the host shim without Media
  DB/RAG imports.

Boundary tests:

- `mcp_unified.docs` imports no `tldw_Server_API` modules;
- baseline docs package imports without Playwright, trafilatura,
  BeautifulSoup4, requests, httpx, aiohttp, ChromaDB, or RAG services;
- URL/sitemap sync tests use fake resolver/transport and never require live
  internet.

Security tests:

- local symlink escape denial;
- unsupported local suffix failure;
- URL scheme denial;
- localhost/private/link-local DNS denial;
- redirect-to-private denial;
- oversized response denial;
- unsupported content type denial;
- sitemap XML external entity is not resolved;
- sitemap XML `DOCTYPE` or `ENTITY` declarations are rejected before parser
  construction;
- no fetch before approval-required result.

## Implementation Slices

Recommended Stage 4A implementation sequence:

1. Source registry schema and store helpers.
2. Populate sources from `docs.import_path` and `docs.ingest_url`.
3. `docs.list(kind="sources")` and `docs.status` source-sync capability.
4. Local file/directory `docs.sync_source`.
5. URL page `docs.sync_source`.
6. Optional explicit sitemap source sync.
7. Host shim registration and boundary tests.

If the work needs to be split into multiple PRs, stop after slice 4. Local
source sync is independently useful and proves the source model before adding
more URL risk.

## Implementation Planning Defaults

- The first Stage 4A implementation plan should include slices 1 through 5:
  source registry, source population, source list/status, local sync, and URL
  page sync.
- Explicit sitemap source sync should be planned as the final task in the same
  plan and can be deferred to Stage 4A.2 if the first PR needs to stay smaller.
- Store run item details in `docs_sync_runs.metadata_json` in Stage 4A, bounded
  by `max_sync_run_items`; do not add `docs_sync_run_items` until result sizes
  justify another table.
- Add `docs_documents.lifecycle_status` in Stage 4A so tombstone behavior has
  clear search semantics while preserving legacy active rows.
