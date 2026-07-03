---
id: TASK-12121
title: Design standalone MCP docs Stage 4B bounded source discovery
status: In Progress
labels:
- mcp
- docs
- design
priority: high
documentation:
- Docs/superpowers/specs/2026-06-30-standalone-mcp-docs-catalog-design.md
- Docs/superpowers/specs/2026-06-30-standalone-mcp-docs-url-acquisition-design.md
- Docs/superpowers/specs/2026-07-01-standalone-mcp-docs-stage4a-sync-source-design.md
- Docs/superpowers/specs/2026-07-03-standalone-mcp-docs-stage4b-source-discovery-design.md
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
- [ ] #1 Stage 4B design defines source discovery goals, non-goals, and explicit boundaries versus Stage 4A sync_source.
- [ ] #2 Spec defines tool contracts for bounded sitemap/page discovery and how accepted candidates become docs sources or ingested documents.
- [ ] #3 Spec preserves optional web scraping dependencies and locked-down deployment profiles where URL acquisition/discovery can be disabled.
- [ ] #4 Spec defines source policy, same-origin, URL normalization, robots/sitemap, caps, redaction, dedupe, and SSRF/privacy constraints.
- [ ] #5 Spec keeps standalone MCP independent from tldw_Server_API runtime imports and excludes embeddings/vector/reranking/Media-RAG bridges.
- [ ] #6 Spec includes implementation/testing strategy with fake transport/resolver tests and no live internet.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Inspect merged standalone MCP docs Stage 2 and Stage 4A specs/code paths.
2. Create the Stage 4B bounded source discovery design spec.
3. Self-review for source-sync gaps, optional dependency boundaries, locked-down profile behavior, and host import boundaries.
4. Run documentation verification and hand the draft to the user for review before implementation planning.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

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
