---
id: TASK-2344
title: Design MCP filesystem SQLite lock backend
status: In Progress
labels:
- mcp
- filesystem
- design
priority: medium
modified_files:
- Docs/superpowers/specs/2026-06-09-mcp-filesystem-sqlite-lock-backend-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Write and review the package-level design spec for a persistent/shared SQLite filesystem lock backend behind the existing MCP filesystem lock manager seam. Keep tldw_server integration minimal and preserve standalone mcp_unified packaging boundaries.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Spec documents the package-level extraction, tldw_server compatibility shim, and no-framework constraint.
- [ ] #2 Spec defines SQLite lock acquisition, renewal, release, validation, expiry, and conflict semantics without read-then-write races.
- [ ] #3 Spec covers config validation, dependency/package metadata, standalone artifact gate updates, docs, and validation commands.
- [ ] #4 Spec records scope limits: host-local/shared-file SQLite coordination only, not distributed locking.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/specs/2026-06-09-mcp-filesystem-sqlite-lock-backend-design.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Draft spec written with package-level extraction, minimal tldw_server host integration, SQLite lock semantics, config validation, packaging/artifact gates, documentation, and validation commands. Awaiting user review before implementation planning.
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
