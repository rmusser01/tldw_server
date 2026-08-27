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
- backlog/tasks/task-13116 - Plan-Scheduled-Tasks-Phase-4D-prerequisite-and-feasibility-implementation.md
- backlog/tasks/task-13125 - Normalize-Scheduled-Tasks-Phase-4D-Backlog-identities.md
- backlog/tasks/task-13126 - Design-Scheduled-Tasks-Phase-4D-Agent-Task-execution.md
- https://github.com/rmusser01/tldw_server/pull/2809
- https://github.com/rmusser01/tldw_server/pull/2578
- https://github.com/rmusser01/tldw_server/pull/2814
- https://github.com/rmusser01/tldw_server/pull/2825
- legacy:TASK-13116 (standalone HTML identity housekeeping)
- interim:TASK-13125/TASK-13126 (superseded by latest-dev allocation)
modified_files:
- backlog/tasks/task-12115 - Add-first-class-standalone-HTML-JS-presentation-generation.md
- backlog/tasks/task-13134 - Normalize-duplicate-Backlog-task-identities-from-standalone-HTML-closeout.md
- backlog/tasks/task-13135 - Make-chat-focus-mode-truly-fullscreen.md
updated_date: 2026-08-27 02:24
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Normalize the task-identity collisions exposed by the standalone HTML closeout without rewriting history: keep the standalone HTML rollout canonical as TASK-12115, preserve the historical chat-focus record as TASK-13135, and keep this normalization record as TASK-13134. Leave the legitimate Scheduled Tasks TASK-13116, TASK-13125, and TASK-13126 records untouched. TASK-13134 and TASK-13135 replace the interim IDs selected before the latest dev rebase allocated TASK-13125 and TASK-13126.
<!-- SECTION:DESCRIPTION:END -->
## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The standalone-closeout TASK-12115/TASK-13116 collisions and the rebase-time TASK-13125/TASK-13126 collisions no longer exist in active, completed, or archived Backlog storage.
- [x] #2 The standalone HTML rollout remains canonical as TASK-12115 with PR #2809 and merge evidence intact.
- [x] #3 The historical chat-focus work is canonical as TASK-13135, retains PR #2578 evidence, and records its legacy TASK-12115 identity.
- [x] #4 This normalization record is canonical as TASK-13134, and the legitimate Scheduled Tasks TASK-13116, TASK-13125, and TASK-13126 records remain unchanged.
- [x] #5 Backlog MCP task_view resolves TASK-12115, TASK-13116, TASK-13125, TASK-13126, TASK-13134, and TASK-13135 to one canonical record each; broad task_search results contain exactly one result whose frontmatter ID equals each scoped query.
- [x] #6 No acceptance criteria, implementation history, or completion evidence is lost from either completed work record.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Inventory every in-scope collision against latest dev. 2. Preserve standalone HTML as TASK-12115. 3. Move the historical chat-focus record to TASK-13135 and this normalization record to TASK-13134 using the user-approved narrow path/frontmatter-ID exception because Backlog MCP/CLI cannot rename IDs. 4. Update all semantic links through Backlog MCP while leaving Scheduled Tasks TASK-13116, TASK-13125, and TASK-13126 unchanged. 5. Verify exact filesystem uniqueness, task_view resolution, broad-search exact-ID filtering, Markdown formatting, and PR review state.
<!-- SECTION:PLAN:END -->
## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Created during the standalone HTML rollout closeout and implemented as an independent Backlog-only review unit.

Final identity map: standalone HTML remains TASK-12115; this housekeeping record is TASK-13134; historical chat-focus work is TASK-13135. Scheduled Tasks TASK-13116, TASK-13125, and TASK-13126 remain untouched.

The first normalization pass used TASK-13125 and TASK-13126 when they were free. Latest dev then merged PR #2826, allocating those IDs to Scheduled Tasks. The PR was rebased and this record/chat history were remapped to the next unused IDs, TASK-13134/TASK-13135, before merge.

Backlog MCP and CLI expose semantic editing but no task-ID/path rename or delete operation. The user explicitly approved apply_patch for only the path/frontmatter ID moves; all semantic changes use the official Backlog MCP.

Qodo review correction: task_search searches the complete Markdown body and is intentionally non-unique when records cross-reference one another. The verified invariant is one exact frontmatter-ID match within broad search results, not one total search row.

PR: https://github.com/rmusser01/tldw_server/pull/2825 (target: dev).
Post-rebase live matrix: task_view returned the canonical paths for TASK-12115, TASK-13116, TASK-13125, TASK-13126, TASK-13134, and TASK-13135. Corresponding task_search result counts were capped at 100 for the first five queries and 92 for TASK-13135; every set contained exactly one record whose frontmatter ID equaled the query. Filesystem frontmatter counts were also one each. The Qodo search-wording finding was corrected, re-reviewed as Bugs (0), and its inline thread resolved.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Normalized every standalone-closeout identity collision against latest dev without rewriting completed work. Standalone HTML remains TASK-12115; the historical chat-focus record is preserved as TASK-13135 with PR #2578 evidence; this governing record is TASK-13134; and Scheduled Tasks TASK-13116, TASK-13125, and TASK-13126 remain untouched. task_view resolves each scoped ID to one canonical path, broad task_search results contain exactly one matching frontmatter ID per query, stale interim paths are absent, and the diff remains Backlog Markdown only.
<!-- SECTION:FINAL_SUMMARY:END -->
## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 The duplicate identity is normalized through the official Backlog workflow.
- [x] #2 Both affected task records link to each other or to the normalization record.
- [x] #3 Backlog MCP resolution is verified after the change.
- [x] #4 The Backlog-only diff is isolated from product code and ready for independent review.
<!-- DOD:END -->
