---
id: TASK-13125
title: Normalize duplicate Backlog task identities from standalone HTML closeout
status: Done
created_date: 2026-08-24 17:49
labels:
- backlog
- data-integrity
priority: Low
references:
- backlog/tasks/task-12115 - Add-first-class-standalone-HTML-JS-presentation-generation.md
- backlog/tasks/task-13126 - Make-chat-focus-mode-truly-fullscreen.md
- backlog/tasks/task-13116 - Plan-Scheduled-Tasks-Phase-4D-prerequisite-and-feasibility-implementation.md
- https://github.com/rmusser01/tldw_server/pull/2809
- https://github.com/rmusser01/tldw_server/pull/2578
- https://github.com/rmusser01/tldw_server/pull/2814
- legacy:TASK-13116 (standalone HTML identity housekeeping)
- https://github.com/rmusser01/tldw_server/pull/2825
modified_files:
- backlog/tasks/task-12115 - Add-first-class-standalone-HTML-JS-presentation-generation.md
- backlog/tasks/task-13125 - Normalize-duplicate-Backlog-task-identities-from-standalone-HTML-closeout.md
- backlog/tasks/task-13126 - Make-chat-focus-mode-truly-fullscreen.md
updated_date: 2026-08-27 01:54
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Normalize the two task-identity collisions exposed by the standalone HTML closeout without rewriting history: keep the standalone HTML rollout canonical as TASK-12115, move the historical chat-focus record to TASK-13126, and move this housekeeping record away from the legitimate Scheduled Tasks TASK-13116 to TASK-13125. Preserve both completed work records, their pull-request evidence, and deterministic Backlog resolution.
<!-- SECTION:DESCRIPTION:END -->
## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The in-scope TASK-12115 and TASK-13116 collisions no longer exist in active, completed, or archived Backlog storage.
- [x] #2 The standalone HTML rollout remains canonical as TASK-12115 with PR #2809 and merge evidence intact.
- [x] #3 The historical chat-focus work is canonical as TASK-13126, retains PR #2578 evidence, and records its legacy TASK-12115 identity.
- [x] #4 This normalization record is canonical as TASK-13125, and the legitimate Scheduled Tasks TASK-13116 record remains unchanged.
- [x] #5 Backlog MCP task_view resolves TASK-12115, TASK-13116, TASK-13125, and TASK-13126 to one canonical record each; broad task_search results contain exactly one result whose frontmatter ID equals each scoped query.
- [x] #6 No acceptance criteria, implementation history, or completion evidence is lost from either completed work record.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Inventory the two in-scope collisions against latest dev. 2. Move the historical chat-focus record to TASK-13126 and the housekeeping record to TASK-13125 using the user-approved narrow file-path/ID exception because Backlog MCP/CLI cannot rename task IDs. 3. Update titles, references, legacy notes, and completion evidence through Backlog MCP. 4. Verify exact ID uniqueness, deterministic MCP resolution, clean Markdown, and an isolated Backlog-only diff.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Created during the standalone HTML rollout closeout and implemented as an independent Backlog-only review unit.

Identity map: standalone HTML remains TASK-12115; this housekeeping record moved from the accidental TASK-13116 identity to TASK-13125; historical chat-focus work moved from the duplicate TASK-12115 identity to TASK-13126. The legitimate Scheduled Tasks TASK-13116 record is deliberately untouched.

Backlog MCP and CLI expose semantic editing but no task-ID/path rename or delete operation. The user explicitly approved apply_patch for only the two path/frontmatter ID moves; all semantic changes use the official Backlog MCP.

Verification: exact filesystem ID counts are one each for TASK-12115, TASK-13116, TASK-13125, and TASK-13126; Backlog task_view resolved each exact ID to one canonical record; broad full-text task_search results contain cross-references but exactly one result whose frontmatter ID equals each scoped query; stale historical paths and stale TASK-13116 housekeeping wording are absent; end-of-file-fixer and trailing-whitespace hooks pass on all three resulting records; git diff --check passes. Bandit and product tests were not run because this change touches Backlog Markdown only.

PR: https://github.com/rmusser01/tldw_server/pull/2825 (target: dev). Merge remains gated on the requester’s human-written Change summary.

Qodo review correction: task_search searches the complete Markdown body and is intentionally non-unique when records cross-reference one another. The verified invariant is one exact frontmatter-ID match within broad search results, not one total search row.

Post-Qodo live matrix: task_view returned the canonical TASK-12115, TASK-13116, TASK-13125, and TASK-13126 paths. Each corresponding task_search query reached the configured 100-result cap because cross-references are indexed, and each result set contained exactly one record whose frontmatter ID equaled the query.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Normalized the two Backlog identity collisions exposed by the standalone HTML closeout without rewriting completed work. Standalone HTML remains TASK-12115; the original chat-focus record is preserved as TASK-13126 with PR #2578 evidence and a legacy-ID note; this housekeeping record is now TASK-13125; and the legitimate Scheduled Tasks TASK-13116 record remains untouched. Backlog task_view resolves each of the four exact IDs to one canonical record; broad task_search results preserve cross-references while containing one exact frontmatter-ID match for each scoped query; stale paths and deferred references are removed, and the diff contains Backlog Markdown only.
<!-- SECTION:FINAL_SUMMARY:END -->
## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 The duplicate identity is normalized through the official Backlog workflow.
- [x] #2 Both affected task records link to each other or to the normalization record.
- [x] #3 Backlog MCP resolution is verified after the change.
- [x] #4 The Backlog-only diff is isolated from product code and ready for independent review.
<!-- DOD:END -->
