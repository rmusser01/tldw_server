---
id: TASK-13950
title: 'Architecture review 2026-10-25: remediation roadmap'
status: To Do
labels:
- architecture-review
- architecture
- epic
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Umbrella for the 2026-10-25 architecture review of dev (findings re-verified on dca11342de). Subtasks TASK-13950.1-TASK-13950.48 carry one finding each; the phased order, dependencies and exit criteria live in Docs/superpowers/plans/2026-10-25-architecture-remediation-roadmap.md.

Root causes: (1) the 16 lint ratchets cap imports and SQL but never size, so god files and functions keep growing (ChaChaNotes_DB.py 17,199 lines on 2026-02-28 -> 47,874; unified_rag_pipeline 5,589 -> 7,063 lines and 233 params); (2) tests pin module structure (26,131 monkeypatch.setattr, 5,266 against endpoint modules), so code cannot move without seams first; (3) SQLite/Postgres split by 2,890 inline branches, which hides tenant-isolation gaps from SQLite CI; (4) copied tables instead of derived ones (17 provider tables, 6 alias tables, 7 placeholder rewriters, 8 InputError classes).

Owner decisions (2026-10-25): moderation applies to every LLM surface but is optional and policy-configured; keywords and tags are case-insensitive in every store; Scheduler folds into Jobs (TASK-13950.48; the owner corrected on 2026-10-25 that no fold work was in progress); admin consoles consolidate onto tldw-admin (/admin-ui); the mlx /chat/completions rejection is a confirmed bug.

Review and dry run (2026-10-25): Moderation decision as applied in .27: global [Moderation] enabled = False disables the default policy on every surface, while explicit per-user overrides and Guardian supervised-account overlays still enable it, as they do today. Subtasks carry the parent in their description because backlog-py cannot set parent_task_id.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Phase 0 (bugs, security fixes, growth guards) subtasks .1-.11 are Done
- [ ] #2 Phase 1 (cycle-closing edges) subtasks .12-.17 are Done and the layering baseline (function-level imports included, 311 entries at review) has no foundation-upward, LLM_Calls -> Chat/AuthNZ, core -> api.schemas or DB_Management -> domain-logic entries; both SCC measures (module-level 100, with function-level imports 106 at review) are recorded in this task's notes
- [ ] #3 Phase 2 (injection seams) subtasks .18-.21 are Done
- [ ] #4 Phase 3 (hotspot decomposition) subtasks .22-.30 are Done
- [ ] #5 Phase 4 (consolidation) subtasks .31-.48 are Done or explicitly descoped with a recorded reason
<!-- AC:END -->

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
