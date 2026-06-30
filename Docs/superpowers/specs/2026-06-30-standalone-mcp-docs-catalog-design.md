# Standalone MCP Document Corpus And RAG Tools Design

Date: 2026-06-30
Status: Draft for spec review
Backlog: TASK-12071
Related: Docs/superpowers/specs/2026-05-26-mcp-unified-standalone-library-gateway-design.md

## Summary

Build a standalone-first MCP document corpus that gives agents local SQLite + FTS5
retrieval across ingested documents. The feature is not only a Context7 clone
for coding references. It is a general document and collection retrieval module:
users and agents can import local docs, scrape approved pages, assign keywords,
group documents into collections, and ask the MCP server for bounded RAG context
packs with citations.

Context7-style library resolution remains useful for compatibility, but it is a
view over the corpus rather than the core storage model. Documents do not need
to belong to a package, library, or version. Package/version fields are optional
metadata for collections or sources that happen to represent package docs.

The default owner of docs state is the standalone MCP server. The built-in
`tldw_server` MCP server should expose the same tools by mounting the same
runtime-neutral module through host adapters. The docs package must not import
`tldw_Server_API`.

## Goals

- Provide a standalone MCP docs module backed by local SQLite + FTS5.
- Support general documents, optional collections, optional keywords, and
  source provenance.
- Make documents first-class records that can exist outside any collection.
- Expose agent-friendly retrieval primitives, especially a bounded
  `docs.context` tool for RAG context composition.
- Support local import from configured path scopes.
- Support single-page URL ingestion behind source policy, approval, egress
  safety checks, and audit records.
- Provide Context7-compatible aliases for package-like collections without
  requiring all content to use library/version semantics.
- Keep optional embeddings, reranking, Playwright extraction, and crawling as
  extension points rather than v1 requirements.
- Let `tldw_server` mount the same module and optionally bridge existing Media
  or RAG stores later without changing the standalone data model.

## Non-Goals

- No generated answer tool in v1. Retrieval and context composition are the
  reliable primitives; the calling agent can generate final answers.
- No mandatory package/library/version hierarchy.
- No dependency on `tldw_server`, Media DB, ChromaDB, RAG services, or AuthNZ in
  the standalone package.
- No broad web crawler in v1. Bounded crawl and sitemap sync belong to a later
  stage using the same source policy.
- No mandatory browser automation for URL ingestion in v1.
- No redistribution/export workflow for scraped third-party docs in v1.
- No arbitrary filesystem import outside configured trusted roots.
- No URL fetch before required approval is granted.

## Approved Approach

Use a document and collection-first docs corpus with a standalone SQLite store.

The module exposes canonical `docs.*` tools and compatibility aliases:

- Canonical tools are the supported long-term API.
- Context7 aliases route into the same services and are limited to
  package-like collection metadata when available.
- General RAG behavior lives in `docs.search`, `docs.get`, and `docs.context`.

The first implementation slice should prove the standalone substrate before
adding richer sync or host-specific bridges.

## Current Repo Fit

The repository already has useful adjacent capabilities:

- MCP Unified module lifecycle, tool definitions, RBAC, write-tool handling,
  idempotency, and audit concepts.
- Standalone MCP extraction planning with adapter seams and package boundaries.
- Existing web scraping and media ingestion code in `tldw_server`.
- Existing knowledge/media MCP tools and documentation ingestion playbooks.

This design intentionally does not make the new docs corpus depend on those
host features. Instead:

- standalone runtime uses the docs module directly with local SQLite;
- `tldw_server` provides adapters for auth, path scopes, approval, audit, and
  optional host-provided sources;
- future adapters can bridge Media/RAG content into the docs query surface.

## Package Boundary

The runtime-neutral package should live under the standalone MCP package
boundary, for example:

```text
mcp_unified/
  docs/
    models.py
    settings.py
    errors.py
    store/
      sqlite.py
      schema.sql
      migrations.py
      fts.py
    importers/
      base.py
      markdown.py
      plaintext.py
      html.py
    acquisition/
      fetcher.py
      policy.py
      approvals.py
      url_normalization.py
      egress_guard.py
    retrieval/
      search.py
      context.py
      citations.py
      aliases.py
    mcp_module.py
```

Host-specific integration belongs outside that package. For `tldw_server`, use
adapters under its tree, for example:

```text
tldw_Server_API/app/core/MCP_unified/adapters/docs/
  authnz.py
  approval.py
  audit.py
  path_scope.py
  source_bridge.py
```

Boundary requirements:

- `mcp_unified.docs` must not import `tldw_Server_API`.
- Host adapters may import both the standalone package and `tldw_Server_API`.
- Import-boundary tests must enforce this.
- Standalone examples should use the local SQLite store as the default state
  owner.

## Core Architecture

The module is composed of focused services:

- `DocsCatalogStore`: owns SQLite connection setup, migrations, FTS5 tables,
  transactional writes, and query helpers.
- `DocsImportService`: parses local documents into normalized documents,
  sections, chunks, keywords, and optional collection membership.
- `DocsAcquisitionService`: performs approved single-page URL ingestion and
  later source sync.
- `DocsSourcePolicy`: evaluates source profiles, domain policy, approval
  requirements, egress safety, robots behavior, limits, and reason codes.
- `DocsRetrievalService`: resolves aliases, searches FTS5, loads documents,
  builds bounded context packs, and formats citations.
- `DocsMCPModule`: exposes canonical MCP tools plus Context7-compatible aliases.

Discovery and execution authority stay separate:

- read tools are visible according to profile/RBAC policy;
- write/acquisition tools require explicit write permission, source policy, and
  approval when configured;
- source catalogs and aliases never grant execution authority by themselves.

## Data Model

The schema is document-first.

### Documents

`docs_documents` stores first-class content records.

Important fields:

- `id`
- `owner_scope` and `profile_scope` when the host supports multi-user or
  profile-specific data
- `title`
- `description`
- `document_type` such as `markdown`, `text`, `html`, `pdf`, `docx`, or `other`
- `language`
- `source_id`
- `canonical_uri`
- `source_url`
- `source_path`
- `content_hash`
- `raw_content_hash`
- `license_hint`
- `created_at`, `updated_at`, `fetched_at`, `indexed_at`
- `metadata_json`

A document can exist without any collection membership.

### Collections

`docs_collections` stores arbitrary groups of documents.

Examples:

- project docs
- scraped site pages
- research corpora
- package docs
- manuals
- imported note bundles
- policy handbooks

Important fields:

- `id`
- `name`
- `description`
- `collection_type`
- `owner_scope`
- `visibility`
- `default_sort`
- `metadata_json`
- optional package-like metadata such as ecosystem, package name, version label,
  release label, or upstream docs URL

`docs_collection_members` is many-to-many:

- `collection_id`
- `document_id`
- `member_path`
- `order_index`
- `role`
- `added_at`

Package/library/version semantics are metadata on collections or sources, not a
required hierarchy.

### Keywords

`docs_keywords` stores tags, topics, entities, or user-defined labels.

`docs_document_keywords` and `docs_chunk_keywords` associate keywords with
documents or retrieval chunks.

Keywords are optional, but import tools should allow assigning them during
ingestion. Later enrichment can add extracted entities or tags without changing
the model.

### Sources

`docs_sources` stores provenance and sync policy:

- local file
- local directory
- URL
- upload/import bundle
- crawler seed
- sitemap
- package docs source
- host-provided source

Important fields:

- `id`
- `source_type`
- `display_name`
- `source_uri`
- `canonical_source_key`
- `approval_state`
- `policy_profile`
- `owner_scope`
- `last_fetch_status`
- `last_fetch_at`
- `etag`
- `last_modified`
- `content_hash`
- `sync_enabled`
- `metadata_json`

### Sections And Chunks

`docs_sections` stores a best-effort structure index:

- heading text
- heading path
- slug or anchor
- parent section
- order index
- start/end char offsets when available
- start/end line numbers when available
- offset precision marker: `exact`, `line`, `anchor`, or `unknown`

`docs_chunks` stores retrieval units:

- document id
- section id
- chunk text
- content hash
- token or char estimate
- rank hints
- citation anchor
- offset precision
- metadata JSON

Markdown and text imports should provide line/char offsets where practical. HTML
imports may only provide heading anchors and best-effort extracted section
boundaries.

### Aliases

`docs_aliases` maps names to targets:

- collection
- document
- source
- keyword
- package-like collection metadata

Alias resolution must return ambiguity metadata instead of silently picking a
low-confidence match.

### Audit

`docs_fetch_audit` and general docs audit records should capture:

- requested URL/path/source
- approval requirement and approval id
- policy decision
- redirect chain summary
- status code and content type
- size limit decisions
- private-IP or unsafe redirect denials
- created/updated/unchanged/staged result
- safe argument hash
- actor/profile metadata when available

## Storage And Indexing

SQLite + FTS5 is the required v1 retrieval backend.

Indexes should cover:

- documents: title, description, source URI/path, metadata text
- collections: name, description, metadata text
- keywords: keyword text and aliases
- sections: heading path and heading text
- chunks: chunk text and citation text

FTS5 should work without embeddings. Optional embedding support is an adapter:

- embeddings are not required for v1;
- if configured, retrieval can rerank FTS candidates or support vector search;
- if unavailable, tools degrade to FTS-only behavior without changing response
  contracts.

Writes should be transactional:

- import source;
- upsert document;
- replace sections/chunks for changed content;
- update FTS rows;
- update collection memberships and keywords;
- append audit summary.

## Tool Surface

### Read Tools

`docs.search`

Searches the corpus with optional filters:

- query
- collection ids or aliases
- keyword ids or names
- source ids or source type
- document type
- URL/path prefix
- date ranges
- owner/profile scope when exposed by the host
- package-like metadata when present

Returns ranked snippets with citation anchors and source metadata.

`docs.get`

Retrieves a document, section, or chunk by URI/id. Supports modes:

- `metadata`
- `snippet`
- `section`
- `full`
- `chunk`
- `chunk_with_neighbors`

`docs.context`

Builds a bounded RAG context pack for agents. Inputs include:

- query
- filters
- max chunks
- max documents
- max characters or token estimate
- dedupe strategy
- citation style

Output includes:

- ranked chunks
- citations
- source metadata
- omitted result counts
- budget usage
- dedupe decisions
- retrieval warnings

`docs.resolve`

Resolves names or aliases across collections, documents, sources, keywords, and
package-like docsets.

`docs.list`

Lists documents, collections, keywords, or sources with pagination.

`docs.status`

Reports store health, migration version, FTS status, source policy mode, counts,
and optional embedding availability.

### Write And Acquisition Tools

`docs.import_path`

Imports files or directories under configured trusted roots. It supports:

- target collection ids or names
- keyword assignment
- staged import when no collection is provided
- include/exclude globs within allowed roots
- idempotency by source path and content hash

`docs.ingest_url`

Fetches and ingests one approved URL. It supports:

- explicit target collection
- staged source when no collection is selected
- keyword assignment
- title override
- source policy profile
- idempotency by canonical URL and content hash

The tool must evaluate approval and egress policy before making a network
request.

`docs.collections`

Lists, creates, updates, or modifies collection membership when write tools are
enabled. Read-only deployments may expose list behavior through `docs.list`
instead.

`docs.keywords`

Lists and applies keywords when write tools are enabled. Read-only deployments
may expose list behavior through `docs.list`.

`docs.sync_source`

Declared as a future v1.1 write tool for bounded crawl, sitemap refresh, and
source re-sync. It should not be operational in v1 unless the implementation
also provides the required crawler safety tests.

### Context7-Compatible Aliases

`resolve-library-id`

Routes to `docs.resolve` and prefers collection aliases or package-like
metadata. It returns compatible library-id style results when a package-like
collection exists.

`get-library-docs`

Routes to `docs.search`, `docs.get`, or `docs.context` depending on the
arguments. It operates against package-like collections when possible. If the
corpus only has general documents with no package-like metadata, it should
return a clear no-match result rather than pretending every collection is a
library.

Compatibility aliases are useful for existing clients and prompts, but
documentation should teach the canonical `docs.*` tools first.

## Source Policy And URL Safety

Source profiles:

- `locked_down`: no arbitrary URL ingestion; only preconfigured local paths or
  preapproved sources.
- `local_first`: local imports allowed; arbitrary URL ingestion returns
  `source_approval_required` unless the domain/source is preapproved.
- `online_capable`: approved domains can be fetched directly; unknown domains
  require approval.

URL ingestion must enforce:

- allowed schemes: `http` and `https` only;
- DNS resolution before connect;
- deny loopback, link-local, multicast, private, and otherwise blocked address
  ranges;
- repeat DNS/private-address validation after redirects;
- maximum redirect count;
- maximum body size;
- content type allowlist;
- request timeout;
- per-domain rate limits;
- configured user-agent;
- robots policy according to source profile;
- no cookies or credentials by default;
- no local file URL handling;
- audit for every deny, approval-required, and fetch result.

Approval-required responses are first-class outcomes:

```json
{
  "status": "approval_required",
  "reason_code": "source_approval_required",
  "url": "https://example.com/docs/page",
  "domain": "example.com",
  "requested_scope": "single_url",
  "safe_arguments_hash": "..."
}
```

The tool must not partially fetch content before returning this result.

## Data Flow

### Local Import

1. `docs.import_path` receives a configured source id or a path under an
   approved root.
2. Path-scope adapter canonicalizes the path and rejects escapes or symlink
   traversal outside scope.
3. Import service parses Markdown, MDX, text, and static HTML in v1.
4. Normalizer creates documents, sections, chunks, keywords, optional
   collection membership, content hashes, and FTS rows.
5. Store writes happen transactionally.
6. Audit/import summary reports `created`, `updated`, `unchanged`, `staged`, or
   `denied`.

### Single-Page URL Ingestion

1. `docs.ingest_url` receives a URL, optional collection, and optional keywords.
2. Source policy evaluates domain/source profile.
3. If approval is required, the tool returns `approval_required` without
   fetching.
4. Fetcher enforces DNS, redirect, private-IP, timeout, size, content-type,
   robots, and rate-limit controls.
5. Static extractor normalizes HTML, Markdown, or text.
6. Store upserts by canonical source URL and content hash.
7. The result reports `created`, `updated`, `unchanged`, `staged`, or `denied`.

### Retrieval

1. Client calls `docs.search`, `docs.resolve`, or a compatibility alias.
2. Retrieval service searches FTS5, applies filters, and optionally reranks
   through configured adapters.
3. Client calls `docs.get` for specific records or `docs.context` for a bounded
   RAG pack.
4. Responses include source provenance and citation anchors with offset
   precision.

## Error Handling

Use stable machine-readable reason codes:

- `source_approval_required`
- `source_domain_denied`
- `egress_private_address_denied`
- `redirect_policy_denied`
- `content_type_denied`
- `content_too_large`
- `path_scope_denied`
- `document_not_found`
- `collection_not_found`
- `unsupported_import_format`
- `index_unavailable`
- `alias_ambiguous`
- `context_budget_exceeded`
- `staged_source_requires_assignment`
- `embedding_backend_unavailable`

Read tools should return partial results plus warnings when optional features
such as embeddings are unavailable. Acquisition tools should fail closed when
policy, approval, egress, or path-scope checks are unavailable.

## Rollout Plan

### Stage 1: Standalone Docs Substrate

- Runtime-neutral package.
- SQLite + FTS5 schema and migrations.
- Document/collection/keyword/source/audit model.
- Markdown, MDX, text, and static HTML import.
- `docs.search`, `docs.get`, `docs.context`, `docs.resolve`, `docs.list`, and
  `docs.status`.
- Context7-compatible read aliases.
- Import-boundary test proving no `tldw_Server_API` imports.

### Stage 2: Acquisition

- `docs.import_path` with path-scope enforcement.
- `docs.ingest_url` with approval-required flow and egress policy.
- Static HTTP fetcher.
- URL canonicalization, redirect handling, content-type allowlists, body limits,
  and content hash dedupe.
- Locked-down, local-first, and online-capable source profiles.

### Stage 3: Server Mounting

- Standalone MCP server enables the docs module by default with local SQLite
  state.
- Built-in `tldw_server` MCP server mounts the same module through host
  adapters.
- `tldw_server` host bridge for Media/RAG content remains optional and
  separately planned.

### Stage 4: Richer Sync And Retrieval

- `docs.sync_source` for bounded crawl and sitemap refresh.
- Optional Playwright/browser extraction extra.
- Optional embedding and reranking adapters.
- Jobs or scheduler integration where a host provides it.

## Testing Strategy

Unit tests:

- schema migrations
- FTS5 indexing and ranking
- document without collection
- collection membership
- keyword assignment and filtering
- alias resolution and ambiguity
- section/chunk offset precision
- context budget enforcement
- path-scope canonicalization
- URL policy decisions
- redirect and private-IP blocking
- content-type and body-size denials
- idempotent ingest
- Context7 alias routing

Integration tests:

- local import to search to get to context
- staged document import without collection
- collection and keyword-filtered retrieval
- approved URL ingest to search
- approval-required URL flow
- locked-down URL denial
- standalone server tool discovery and tool call flow
- `tldw_server` module mounting without Media DB/RAG dependency

Boundary tests:

- standalone docs package imports no `tldw_Server_API` modules
- host adapters live outside the standalone docs package
- optional embedding/browser extras are not required for baseline FTS behavior

Security tests:

- URL scheme denial
- localhost/private/link-local DNS denial
- redirect-to-private denial
- oversized response denial
- unsupported content type denial
- no fetch before approval-required result
- path traversal and symlink escape denial for local import

## Acceptance Criteria

- A fresh standalone MCP server can ingest local Markdown, MDX, text, and static
  HTML into SQLite, search with FTS5, retrieve cited chunks, and return bounded
  `docs.context` packs.
- Documents can exist outside any collection.
- Collections and keywords are optional metadata for grouping and filtering.
- Context7-compatible aliases work for package-like collections without forcing
  all documents into library/version semantics.
- `docs.ingest_url` never fetches before approval when policy requires
  approval.
- URL ingest blocks private/link-local/loopback targets, disallowed schemes,
  unsafe redirects, unsupported content types, and oversized bodies.
- Read tools degrade cleanly to FTS5 when optional embedding/rerank adapters are
  unavailable.
- The built-in `tldw_server` MCP server exposes the same docs tools through
  host adapters without making the standalone package depend on `tldw_server`.
- Tests prove import-boundary cleanliness, FTS retrieval, idempotent ingestion,
  policy denials, approval-required flow, collection/keyword filtering, and
  alias routing.

## Implementation Planning Notes

The first implementation plan should start with Stage 1 only. Stage 1 should
avoid URL networking, browser extraction, embeddings, and `tldw_server` Media/RAG
bridging. That keeps the first slice small enough to prove the data model, MCP
tool contracts, FTS retrieval, and package boundary before adding acquisition
risk.
