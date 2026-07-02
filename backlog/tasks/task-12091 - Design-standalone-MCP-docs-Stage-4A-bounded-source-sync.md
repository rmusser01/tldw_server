---
id: TASK-12091
title: Design standalone MCP docs Stage 4A bounded source sync
status: In Progress
assignee: []
created_date: ''
updated_date: '2026-07-02 03:44'
labels:
  - mcp
  - docs
  - design
dependencies: []
documentation:
  - Docs/superpowers/specs/2026-06-30-standalone-mcp-docs-catalog-design.md
  - >-
    Docs/superpowers/specs/2026-07-01-standalone-mcp-docs-stage4a-sync-source-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Write the Stage 4A design/spec for bounded source refresh in the standalone MCP docs corpus. Scope: docs.sync_source for already-known local and URL sources, conservative stale handling, source-policy reuse, optional sitemap refresh under strict same-origin limits, no arbitrary crawler/browser/embeddings/reranking/Media-RAG bridge, and no tldw_server imports in mcp_unified.docs.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Stage 4A design spec defines docs.sync_source semantics, inputs, outputs, status values, and reason codes.
- [ ] #2 Spec keeps baseline standalone installs dependency-light and preserves the no tldw_Server_API import boundary for mcp_unified.docs.
- [ ] #3 Spec defines bounded local trusted-root refresh and URL/source refresh behavior using existing Stage 1/2 services.
- [ ] #4 Spec explicitly excludes broad crawling, Playwright/browser extraction, embeddings/reranking, Jobs/Scheduler implementation, and Media/RAG host bridges from Stage 4A.
- [ ] #5 Spec covers stale/missing document handling, idempotency/dedupe, audit/provenance updates, and security tests.
- [ ] #6 Spec self-review and whitespace verification are recorded before final summary.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Created `Docs/superpowers/specs/2026-07-01-standalone-mcp-docs-stage4a-sync-source-design.md` as the Stage 4A bounded `docs.sync_source` design spec.
- Grounded the design in the current Stage 1-3 stacked branch seams: `DocsSettings`, `DocsImportService.import_path`, `DocsAcquisitionService.ingest_url`, `DocsCatalogStore.upsert_document`, and `DocsMCPToolProvider`.
- Self-review pass removed ambiguity around tombstone/search behavior and replaced open decisions with explicit implementation planning defaults.
- Review follow-up addressed the identified spec risks: sync now preserves existing collections/keywords while merging source defaults; dry-run is strictly read-only and writes no sync-run rows; URL sources separate fetch-capable `source_url` from redacted display/logging; tombstone preservation uses a concrete `preserve_on_source_tombstone` column; sitemap sync rejects `DOCTYPE`/`ENTITY`, enforces `max_pages` before page fetches, and caps stored run item details.
- Second review follow-up tightened Stage 4A around four implementation risks: query-bearing URL refresh sources now require explicit `persist_url_query_strings` opt-in; metadata preservation now requires a sync-aware transactional upsert/merge helper instead of relying on replacement-oriented `upsert_document()`; report-mode missing items no longer persist missing source-link status or hide documents; optional sitemap sync now requires a narrow sitemap source registration path or remains deferred/disabled.
- Verification after second review follow-up: `git diff --check` passed; stale wording scan for persisted missing-source hiding and old sitemap assumptions returned no matches; positive scan confirmed `persist_url_query_strings`, `upsert_document_for_sync()`, missing-status, and sitemap-registration wording; `backlog task view TASK-12091 --plain` renders cleanly.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Pending user approval before finalizing this design task.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
