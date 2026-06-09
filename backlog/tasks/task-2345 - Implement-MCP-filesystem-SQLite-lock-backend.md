---
id: TASK-2345
title: Implement MCP filesystem SQLite lock backend
status: In Progress
labels:
- mcp
- filesystem
- implementation
priority: medium
references:
- Docs/superpowers/specs/2026-06-09-mcp-filesystem-sqlite-lock-backend-design.md
modified_files:
- Docs/superpowers/plans/2026-06-09-mcp-filesystem-sqlite-lock-backend-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the approved package-level SQLite filesystem lock backend for MCP. Preserve existing tldw_server behavior by default, keep host integration minimal, and prove standalone mcp_unified packaging remains correct.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Package-level mcp_unified.filesystem_locks module provides memory and SQLite lock managers with matching semantics.
- [ ] #2 tldw_server filesystem module consumes the package-level factory through a compatibility shim without broad host abstractions.
- [ ] #3 SQLite backend uses SQLAlchemy Core, lazy optional imports, and atomic conditional acquire/renew behavior.
- [ ] #4 Standalone packaging, README/USER_GUIDE, and artifact boundary tests cover the new package.
- [ ] #5 Focused filesystem, package-boundary, py_compile, Bandit, and diff hygiene validation are recorded.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-06-09-mcp-filesystem-sqlite-lock-backend-implementation-plan.md
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
