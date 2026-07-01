---
id: TASK-12071
title: Design standalone MCP document corpus and Context7-compatible RAG tools
status: Done
labels:
- mcp
- docs
- design
documentation:
- Docs/superpowers/specs/2026-06-30-standalone-mcp-docs-catalog-design.md
modified_files:
- Docs/superpowers/specs/2026-06-30-standalone-mcp-docs-catalog-design.md
- backlog/tasks/task-12071 - Design-standalone-MCP-document-corpus-and-Context7-compatible-RAG-tools.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Write a design spec for a standalone-first MCP document corpus that supports document/collection-first SQLite FTS5 retrieval, optional URL acquisition with approval policy, Context7-compatible aliases, and tldw_server mounting through host adapters.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Design captures standalone-first package boundary with no tldw_server dependency.
- [x] #2 Design uses document/collection/keyword-first model rather than library/version constraints.
- [x] #3 Design defines canonical docs.* MCP tools plus Context7-compatible aliases.
- [x] #4 Design covers URL/source policy, approval flow, egress protections, testing, rollout, and tldw_server host integration.
- [x] #5 Spec is written under Docs/superpowers/specs and committed.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Reviewed the draft before implementation planning and tightened the spec around:
- Stage 1 versus full-program acceptance criteria.
- Local `docs.import_path` belonging to Stage 1, while Stage 2 is optional URL acquisition only.
- Baseline standalone install not requiring the web-scraping pipeline.
- Existing tldw_server scraping code being a reference/copy-adapt source or host-adapter implementation, not a standalone runtime dependency.
- Store-level owner/profile scope enforcement.
- Current tldw_server module-loader shim requirements.
- Config-backed standalone source profiles and future policy storage.
- Explicit collection/keyword read and write tools.
- Context7 alias authorization through canonical docs.* operations.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Closeout update after Stage 1 and Stage 2 implementation: the standalone MCP docs catalog design is approved for staged implementation, and the first two implementation slices are complete (`TASK-12074` Stage 1 and `TASK-12078` Stage 2). Updated the spec status from draft to approved-for-staged-implementation. Verification for this closeout is documentation-only: `git diff --check` will be run before commit; Bandit not applicable except as already recorded in implementation tasks.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Design spec completed and approved for staged implementation. The design captures a standalone-first docs corpus with document/collection/keyword-first SQLite+FTS5 retrieval, canonical docs.* tools plus Context7-compatible aliases, optional URL acquisition with policy/egress protections, and tldw_server host mounting boundaries. Stage 1 and Stage 2 follow-on implementation tasks are complete; Stage 3 server mounting planning remains the next separate slice.
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
