---
id: TASK-530.6
title: Implement Skills test-run semantics and execution-risk disclosure
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-21 16:47'
labels:
  - skills
  - webui
  - ux
  - safe-operations
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/2423'
parent_task_id: TASK-530
priority: high
ordinal: 530.6
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the first Safe Operations slice for /skills. Rename misleading passive preview language to explicit test-run language, disclose model/tool execution risk before running a skill, and render execution errors with alert semantics. Keep dry-run backend support, import review, delete/versioning, and permission metadata panels out of scope for this PR.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The Skills manager row action and modal title use Test run language instead of passive Preview language.
- [x] #2 The modal primary action is named Run test.
- [x] #3 The modal explains before execution that skill rendering uses the supplied arguments and fork-mode skills may call configured models and allowed tools.
- [x] #4 Execution errors render with alert semantics so they are discoverable to assistive technology.
- [x] #5 Focused SkillPreview and Skills manager tests cover the copy, action naming, risk disclosure, and error alert behavior.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/Plans/IMPLEMENTATION_PLAN_skills_test_run_semantics_TASK_530_6.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Added focused SkillPreview coverage for test-run title/action copy, execution-risk disclosure, executeSkill argument forwarding, and alert-based error rendering.
- Added SkillsManager coverage that the row play action is labeled as a test run instead of Preview.
- Updated SkillPreview to use Test run / Run test language, disclose fork-mode model/tool execution risk before execution, render execution failures with AntD Alert semantics, and replace deprecated Modal destroyOnClose with destroyOnHidden.
- Updated the SkillsManager row action tooltip and accessible name to Test run semantics.
- Kept backend dry-run support, import review, delete/versioning, permission metadata panels, and bulk actions out of scope.
- PR review follow-up: verified CodeRabbit duplicate-execution finding, added a synchronous pending guard, disabled the argument input and Run test button while execution is pending, and added a regression test.
- PR review follow-up: verified Gemini Alert/message and Modal/destroyOnClose suggestions against installed AntD 6.2.1; title and destroyOnHidden are the current non-deprecated APIs, so no code change was made for those threads.
- Verification: focused Skills Vitest files pass: 27 tests; git diff --check passes.
- Optional UI typecheck was run with NODE_OPTIONS=--max-old-space-size=8192 and fails on existing unrelated Notes, ScheduledTasks, background, and voice-cloning TypeScript errors; no Skills diagnostics were reported.
- Bandit is not applicable for this frontend-only TypeScript/TSX and markdown slice.
- Known skip/blocker: no backend dry-run semantics are claimed or changed in this task.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the first Safe Operations slice for Skills test runs and PR review follow-up: row and modal actions now use explicit Test run / Run test language, the modal discloses fork-mode model/tool execution risk before execution, execution failures render as accessible alerts, and pending test runs cannot be duplicated through repeated actions. Focused Skills manager and SkillPreview tests cover the updated workflow. Bandit is not applicable because this task only changes frontend TypeScript/TSX and task/plan markdown. Optional UI typecheck currently fails on unrelated repo baseline errors outside Skills.
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
