---
id: TASK-12029
title: Design RPG rules-pack attachment and retrieval-backed lookup
status: To Do
created_date: 2026-06-25 15:08
labels:
- design
- rpg
- ttrpg
- rag
- backend
priority: high
references:
- TASK-12017
- TASK-12026
- TASK-12028
documentation:
- tldw_Server_API/app/core/RPG/README.md
- Docs/superpowers/specs/2026-06-25-rpg-campaign-session-runtime-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Design the next RPG runtime feature: attaching user-provided rules-pack references to campaigns/sessions and using existing retrieval/RAG infrastructure to augment RPG rules lookup and context building without copying long-form rules prose into RPG tables. Scope is design only; implementation should follow in separate tasks after approval.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Design defines how campaigns and sessions attach, list, update, and remove user rules-pack references without duplicating rules prose into RPG tables
- [ ] #2 Design specifies how RPG rules lookup blends built-in citation-only references with user-provided retrieval/RAG results, including citation/attribution fields and ranking/fallback behavior
- [ ] #3 Design specifies how the session context builder includes retrieved user rules snippets within existing context bounds and diagnostics
- [ ] #4 Design covers REST and MCP surface changes, AuthNZ privileges, idempotency/concurrency behavior for attachment writes, and failure modes
- [ ] #5 Design documents licensing/privacy constraints for user-provided rules content and makes clear that bundled adapters remain mechanics-metadata/citation-only
- [ ] #6 Design includes a test and verification plan for repository/service/API/MCP behavior, retrieval mocking, privilege catalog sync, Bandit, and focused regression coverage
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
['Inspect existing RPG runtime and RAG/retrieval APIs to identify the safest integration boundary', 'Propose design options for rules-pack reference storage and retrieval execution', 'Write an approved design spec under Docs/superpowers/specs/', 'Self-review the spec for scope creep, ambiguous retrieval semantics, and licensing overclaims', 'After approval, create a separate implementation plan/task sequence']
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
