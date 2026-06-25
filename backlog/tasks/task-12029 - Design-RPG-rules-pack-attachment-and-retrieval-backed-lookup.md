---
id: TASK-12029
title: Design RPG rules-pack attachment and retrieval-backed lookup
status: In Progress
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
- Docs/superpowers/specs/2026-06-25-rpg-rules-pack-attachment-retrieval-design.md
updated_date: 2026-06-25 22:49
modified_files:
- Docs/superpowers/specs/2026-06-25-rpg-rules-pack-attachment-retrieval-design.md
- backlog/tasks/task-12029 - Design-RPG-rules-pack-attachment-and-retrieval-backed-lookup.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Design the next RPG runtime feature: attaching user-provided rules-pack references to campaigns/sessions and using existing retrieval/RAG infrastructure to augment RPG rules lookup and context building without copying long-form rules prose into RPG tables. Scope is design only; implementation should follow in separate tasks after approval.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Design defines how campaigns and sessions attach, list, update, and remove user rules-pack references without duplicating rules prose into RPG tables
- [x] #2 Design specifies how RPG rules lookup blends built-in citation-only references with user-provided retrieval/RAG results, including citation/attribution fields and ranking/fallback behavior
- [x] #3 Design specifies how the session context builder includes retrieved user rules snippets within existing context bounds and diagnostics
- [x] #4 Design covers REST and MCP surface changes, AuthNZ privileges, idempotency/concurrency behavior for attachment writes, and failure modes
- [x] #5 Design documents licensing/privacy constraints for user-provided rules content and makes clear that bundled adapters remain mechanics-metadata/citation-only
- [x] #6 Design includes a test and verification plan for repository/service/API/MCP behavior, retrieval mocking, privilege catalog sync, Bandit, and focused regression coverage
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
['Inspect existing RPG runtime and RAG/retrieval APIs to identify the safest integration boundary', 'Propose design options for rules-pack reference storage and retrieval execution', 'Write an approved design spec under Docs/superpowers/specs/', 'Self-review the spec for scope creep, ambiguous retrieval semantics, and licensing overclaims', 'After approval, create a separate implementation plan/task sequence']
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
2026-06-25: Brainstorming complete. Approved design direction: hybrid direct media/media-collection references now with registry-compatible schema later; sessions copy campaign refs at creation; rules lookup is snippet/citation-first with opt-in generated answer mode; misses fall back only to bundled citation-only references; no broad RAG/web fallback.
2026-06-25: Wrote approved design spec at Docs/superpowers/specs/2026-06-25-rpg-rules-pack-attachment-retrieval-design.md. Self-review checked for placeholders, contradictory scope, ambiguous retrieval behavior, and licensing overclaims. `git diff --check` passed. Bandit is not applicable because this step changed only Markdown/backlog task metadata, not Python code.
2026-06-25: Post-review amendment tightened the spec around async service boundaries, media.read requirements for attached-source dereference/retrieval, live collection readiness semantics, server-owned ref timestamps, answer-mode quota/governance, and authorization regression tests.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
