---
id: TASK-12076
title: Design standalone MCP docs Stage 2 URL acquisition
status: Done
assignee: []
created_date: ''
updated_date: '2026-07-01 00:17'
labels: []
dependencies: []
documentation:
  - Docs/superpowers/specs/2026-06-30-standalone-mcp-docs-catalog-design.md
  - >-
    Docs/superpowers/specs/2026-06-30-standalone-mcp-docs-url-acquisition-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Write the Stage 2 design/spec for optional single-page URL acquisition for the standalone MCP docs corpus. Scope: docs.ingest_url with rich single-page extraction, config-driven source profiles, approval-required flow, SSRF/redirect/body/content-type safety, optional lazy rich extractors, no crawler/browser/cookies, and no tldw_server imports in mcp_unified.docs.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Spec captures approved single-page URL acquisition scope and exclusions.
- [x] #2 Spec records reviewed risks: optional dependency boundary, SSRF/redirect validation, approval semantics, extraction fallback/status, tool discovery, and fake-transport testing.
- [x] #3 Spec is committed and ready for user review before implementation planning.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Wrote the Stage 2 single-page URL acquisition design and completed self-review for placeholders, contradictions, scope, and ambiguity. The spec keeps web acquisition optional, excludes crawler/browser/cookies, requires config-driven approval semantics, fake transport/resolver tests, lazy optional extractors, and import-boundary protection for the standalone package.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Stage 2 URL acquisition spec written for user review at Docs/superpowers/specs/2026-06-30-standalone-mcp-docs-url-acquisition-design.md. Scope is optional single-page docs.ingest_url with approval-required flow, redirect-aware SSRF controls, body/content-type limits, lazy rich extraction fallback, no live-internet tests, and no tldw_server runtime imports.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Spec written under Docs/superpowers/specs/.
- [x] #2 Spec self-review completed for placeholders, contradictions, scope, and ambiguity.
- [x] #3 Backlog task updated with final summary.
- [x] #4 Spec committed to git.
<!-- DOD:END -->
