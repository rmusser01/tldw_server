---
id: TASK-418.9
title: Plan WebUI study safety specialized implementation
status: Done
labels:
- ux
- design
- webui
- extension
- planning
- study
- safety
- specialized
priority: High
parent_task_id: TASK-418
documentation:
- Docs/superpowers/specs/2026-05-17-webui-extension-ux-remediation-program-design.md
- Docs/superpowers/plans/2026-05-17-webui-extension-ux-remediation-implementation-plan.md
modified_files:
- Docs/superpowers/plans/2026-05-17-webui-study-safety-specialized-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Documentation-only child implementation plan for the approved WebUI/extension UX remediation program Task 11B. Scope maps F2 support, F9 support, F15 support, F18 support, and F19 into reviewable work for evaluations, study, safety, review, data, chunking, kanban, and VN route identity, readiness, and classification without product code changes in this task.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] Created the documentation-only Task 11B implementation plan at `Docs/superpowers/plans/2026-05-17-webui-study-safety-specialized-implementation-plan.md`.
- [x] Covered `/evaluations`, `/flashcards`, `/quiz`, `/moderation-playground`, `/content-review`, `/claims-review`, `/data-tables`, `/chunking-playground`, `/kanban`, `/vn-assets`, and `/vn-play`.
- [x] Mapped `F2 support`, `F9 support`, `F15 support`, `F18 support`, and `F19` into concrete implementation tasks.
- [x] Included route inventory, route ownership, frontend-only versus backend-gated scope, non-goals, file structure, implementation tasks, acceptance criteria, and verification commands.
- [x] Explicitly split implementation into sub-slices so evaluations, study, safety, data/chunking, kanban, and VN work do not become one broad PR.
- [x] Kept this task limited to Markdown planning artifacts with no product frontend or backend code changes.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Created the Task 11B child implementation plan as a continuation of `TASK-418`.
- Cross-checked current route ownership before writing the plan:
  - `/evaluations` uses `EvaluationsPlaygroundPage`, re-exporting `EvaluationsPage`, and currently lacks a route boundary in shared and extension wrappers.
  - `/flashcards` uses `FlashcardsWorkspace` and `FlashcardsManager` with Study, Manage, Import / Export, Templates, and Scheduler modes.
  - `/quiz` uses `QuizWorkspace` and `QuizPlayground` with Take Quiz, Generate, Create, Manage, and Results modes.
  - `/moderation-playground` uses `ModerationPlaygroundShell` with Policy and Settings, Blocklist Studio, User Overrides, Test Sandbox, and Advanced modes.
  - `/content-review` uses `ContentReviewPage`.
  - `/claims-review` redirects to `/content-review`.
  - `/data-tables` uses `DataTablesPage` with My Tables and Create Table modes.
  - `/chunking-playground` uses `ChunkingPlayground` with Single, Compare, Templates, and Capabilities modes.
  - `/kanban` uses `KanbanPlayground`.
  - `/vn-assets` and `/vn-play` are Next pages that dynamically import VN workbench components.
- Added explicit implementation guidance for route classification, route boundaries, study modes, safety/admin states, claims aliasing, advanced-tool framing, labs visibility, VN readiness, and browser verification.
- Bandit was not run because this task touched only Markdown planning and Backlog task files.
- Verification performed for the plan artifact:
  - `rg -n "T[O]D[O]|T[B]D|F[I]XME|\\.\\.\\.|\\bm[a]ybe\\b|\\bpr[o]bably\\b|\\bshould c[o]nsider\\b" Docs/superpowers/plans/2026-05-17-webui-study-safety-specialized-implementation-plan.md`
  - `rg -n "[[:blank:]]$|[^\\x00-\\x7F]" Docs/superpowers/plans/2026-05-17-webui-study-safety-specialized-implementation-plan.md`
  - `git diff --check -- Docs/superpowers/plans/2026-05-17-webui-study-safety-specialized-implementation-plan.md`
  - `node -e` required-route, finding, file, and test coverage check
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Created the Task 11B study, safety, and specialized tools implementation plan. The plan preserves current route ownership, avoids backend changes, and turns the audit findings into reviewable sub-slices for evaluations, flashcards, quiz, moderation, content review, data tables, chunking, kanban, VN assets, and VN play.
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
