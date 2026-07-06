---
id: TASK-12872
title: Design standalone MCP docs Stage 4B bounded source discovery
status: Done
assignee: []
created_date: ''
updated_date: 2026-07-03 19:26
labels:
- mcp
- docs
- design
dependencies: []
documentation:
- Docs/superpowers/specs/2026-06-30-standalone-mcp-docs-catalog-design.md
- Docs/superpowers/specs/2026-06-30-standalone-mcp-docs-url-acquisition-design.md
- Docs/superpowers/specs/2026-07-01-standalone-mcp-docs-stage4a-sync-source-design.md
- Docs/superpowers/specs/2026-07-03-standalone-mcp-docs-stage4b-source-discovery-design.md
priority: high
modified_files:
- Docs/superpowers/specs/2026-07-03-standalone-mcp-docs-stage4b-source-discovery-design.md
- backlog/tasks/task-12121 - Design-standalone-MCP-docs-Stage-4B-bounded-source-discovery.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Write the Stage 4B design/spec for bounded source discovery in the standalone MCP docs corpus. Scope: sitemap and tightly bounded same-origin page discovery that registers candidate URL sources and feeds approved pages into the existing docs.ingest_url/docs.sync_source flow. Keep scraping optional and lazy-loaded with beautifulsoup4/trafilatura, preserve locked-down profiles, avoid browser automation and broad crawling, and keep mcp_unified.docs independent from tldw_Server_API runtime imports.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Stage 4B design defines source discovery goals, non-goals, and explicit boundaries versus Stage 4A sync_source.
- [x] #2 Spec defines tool contracts for bounded sitemap/page discovery and how accepted candidates become docs sources or ingested documents.
- [x] #3 Spec preserves optional web scraping dependencies and locked-down deployment profiles where URL acquisition/discovery can be disabled.
- [x] #4 Spec defines source policy, same-origin, URL normalization, robots/sitemap, caps, redaction, dedupe, and SSRF/privacy constraints.
- [x] #5 Spec keeps standalone MCP independent from tldw_Server_API runtime imports and excludes embeddings/vector/reranking/Media-RAG bridges.
- [x] #6 Spec includes implementation/testing strategy with fake transport/resolver tests and no live internet.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Inspect merged standalone MCP docs Stage 2 and Stage 4A specs/code paths.
2. Create the Stage 4B bounded source discovery design spec.
3. Self-review for source-sync gaps, optional dependency boundaries, locked-down profile behavior, and host import boundaries.
4. Run documentation verification and hand the draft to the user for review before implementation planning.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Design review follow-up completed before implementation planning. Issues found and addressed: clarified optional BeautifulSoup/trafilatura usage; tightened query-bearing candidate response redaction with safe_argument_hash; clarified sitemap registration versus sitemap_sync_enabled repeated refresh; required sitemap source default keywords/collections to flow into page ingestion while preserving user-added organization; added tests for optional BeautifulSoup fallback, query redaction, and metadata preservation.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Stage 4B bounded source discovery design is approved for implementation planning. The spec defines docs.discover_source, bounded sitemap/page-link discovery, url_sitemap refresh through docs.sync_source, optional BeautifulSoup/trafilatura behavior, locked-down profile gates, query redaction, source metadata propagation, fake transport/resolver tests, and standalone import-boundary constraints. Verification was documentation-focused: git diff --check passed, ASCII scan passed, and Backlog rendering passed. Bandit was skipped because this design task changed only documentation and Backlog metadata.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
