---
id: TASK-13134
title: Normalize duplicate Backlog task identities from standalone HTML closeout
status: Done
created_date: 2026-08-24 17:49
labels:
- backlog
- data-integrity
priority: Low
references:
- backlog/tasks/task-12115 - Add-first-class-standalone-HTML-JS-presentation-generation.md
- backlog/tasks/task-13135 - Make-chat-focus-mode-truly-fullscreen.md
- backlog/tasks/task-13125 - Normalize-Scheduled-Tasks-Phase-4D-Backlog-identities.md
- backlog/tasks/task-13126 - Design-Scheduled-Tasks-Phase-4D-Agent-Task-execution.md
- backlog/tasks/task-13128 - Plan-Scheduled-Tasks-Phase-4D-prerequisite-and-feasibility-implementation.md
- https://github.com/rmusser01/tldw_server/pull/2809
- https://github.com/rmusser01/tldw_server/pull/2578
- https://github.com/rmusser01/tldw_server/pull/2814
- https://github.com/rmusser01/tldw_server/pull/2825
- https://github.com/rmusser01/tldw_server/pull/2826
- legacy:TASK-13116 (standalone housekeeping and Scheduled Tasks predecessor; both
  retired)
- 'interim:TASK-13125/TASK-13126 (standalone-closeout interim allocation superseded
  by PR #2826)'
modified_files:
- backlog/tasks/task-12115 - Add-first-class-standalone-HTML-JS-presentation-generation.md
- backlog/tasks/task-13134 - Normalize-duplicate-Backlog-task-identities-from-standalone-HTML-closeout.md
- backlog/tasks/task-13135 - Make-chat-focus-mode-truly-fullscreen.md
updated_date: 2026-08-27 02:54
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Normalize the task-identity collisions exposed by the standalone HTML closeout without rewriting history: keep the standalone HTML rollout canonical as TASK-12115, preserve the historical chat-focus record as TASK-13135, and keep this normalization record as TASK-13134. Respect Scheduled Tasks PR #2826's final canonical mapping: TASK-13125 is its normalization record, TASK-13126 is its Phase 4D design, and the prerequisite record formerly using TASK-13116 is now TASK-13128. Do not recreate retired TASK-13116. TASK-13134 and TASK-13135 replace the interim IDs selected before the latest dev rebase allocated TASK-13125 and TASK-13126.
<!-- SECTION:DESCRIPTION:END -->
## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The standalone-closeout duplicate TASK-12115 identity, the historical housekeeping TASK-13116 identity, and the rebase-time interim TASK-13125/TASK-13126 identities no longer collide with any active, completed, or archived Backlog frontmatter.
- [x] #2 The standalone HTML rollout remains canonical as TASK-12115 with PR #2809 and merge evidence intact.
- [x] #3 The historical chat-focus work is canonical as TASK-13135, retains PR #2578 evidence, and records its legacy TASK-12115 identity.
- [x] #4 This normalization record is canonical as TASK-13134; Scheduled Tasks TASK-13125, TASK-13126, and TASK-13128 remain unchanged from PR #2826; and no canonical TASK-13116 record is recreated.
- [x] #5 Backlog MCP task_view resolves TASK-12115, TASK-13125, TASK-13126, TASK-13128, TASK-13134, and TASK-13135 to one canonical record each; broad task_search results contain exactly one result whose frontmatter ID equals each active scoped query; and repository frontmatter contains zero exact TASK-13116 IDs.
- [x] #6 No acceptance criteria, implementation history, or completion evidence is lost from either completed work record.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Inventory every in-scope collision against latest dev. 2. Preserve standalone HTML as TASK-12115. 3. Move the historical chat-focus record to TASK-13135 and this normalization record to TASK-13134 using the user-approved narrow path/frontmatter-ID exception because Backlog MCP/CLI cannot rename IDs. 4. Update semantic links through Backlog MCP while preserving Scheduled Tasks PR #2826's canonical TASK-13125, TASK-13126, and TASK-13128 records and leaving retired TASK-13116 unallocated. 5. Verify exact filesystem uniqueness, active task_view resolution, broad-search exact-ID filtering, Markdown formatting, and PR review state.
<!-- SECTION:PLAN:END -->
## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Created during the standalone HTML rollout closeout and implemented as an independent Backlog-only review unit.

Final standalone-closeout identity map: standalone HTML remains TASK-12115; this housekeeping record is TASK-13134; historical chat-focus work is TASK-13135.

The first normalization pass used TASK-13125 and TASK-13126 when they were free. Latest dev then merged PR #2826, allocating those IDs to the Scheduled Tasks normalization and Phase 4D design. PR #2826 also moved its prerequisite record from TASK-13116 to TASK-13128. The current repository therefore has no canonical TASK-13116 frontmatter record, and this PR deliberately does not recreate one. The standalone-closeout records were rebased and remapped to TASK-13134/TASK-13135.

Backlog MCP and CLI expose semantic editing but no task-ID/path rename or delete operation. The user explicitly approved apply_patch for only the path/frontmatter ID moves; all semantic changes use the official Backlog MCP.

Qodo review correction: task_search searches the complete Markdown body and is intentionally non-unique when records cross-reference one another. The verified invariant is one exact frontmatter-ID match within broad search results, not one total search row.

PR: https://github.com/rmusser01/tldw_server/pull/2825 (target: dev).

The earlier MCP task_view response for TASK-13116 referenced the pre-PR-#2826 path and was not accepted as current canonical evidence. Final verification uses repository frontmatter as the identity authority: active IDs must each have exactly one record, while retired TASK-13116 must have zero exact frontmatter records.

2026-08-27 latest-dev correction verification: PR #2826's commit history confirms the Scheduled Tasks prerequisite record was renamed from TASK-13116 to TASK-13128. Repository-wide frontmatter counts across active, completed, and archived storage are exactly one for TASK-12115, TASK-13125, TASK-13126, TASK-13128, TASK-13134, and TASK-13135, and zero for retired TASK-13116. Backlog task_view resolves each active ID to its canonical current path. Broad task_search returned 100 rows for TASK-12115, TASK-13125, TASK-13126, TASK-13128, and TASK-13134 and 92 rows for TASK-13135; each result set contained exactly one record whose reported frontmatter ID equaled the active query. Stale standalone-closeout interim paths are absent. The Scheduled Tasks TASK-13125, TASK-13126, and TASK-13128 files are byte-for-byte unchanged from origin/dev. git diff --check passes. Product tests and Bandit are not applicable because this correction changes only Backlog Markdown.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Normalized every standalone-closeout identity collision against latest dev without rewriting completed work. Standalone HTML remains TASK-12115; the historical chat-focus record is preserved as TASK-13135 with PR #2578 evidence; and this governing record is TASK-13134. Scheduled Tasks PR #2826 retains its canonical TASK-13125 normalization, TASK-13126 design, and TASK-13128 prerequisite records; the retired TASK-13116 identity is not recreated. Active scoped IDs resolve to one canonical path and one exact frontmatter match in broad search, retired TASK-13116 has no exact frontmatter record, stale standalone-closeout interim paths are absent, and the diff remains Backlog Markdown only.
<!-- SECTION:FINAL_SUMMARY:END -->
## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 The duplicate identity is normalized through the official Backlog workflow.
- [x] #2 Both affected task records link to each other or to the normalization record.
- [x] #3 Backlog MCP resolution is verified after the change.
- [x] #4 The Backlog-only diff is isolated from product code and ready for independent review.
<!-- DOD:END -->
