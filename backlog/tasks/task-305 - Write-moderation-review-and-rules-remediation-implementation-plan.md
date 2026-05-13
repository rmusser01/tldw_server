---
id: TASK-305
title: Write moderation review and rules remediation implementation plan
status: Done
assignee: []
created_date: '2026-05-12 15:37'
updated_date: '2026-05-12 15:57'
labels:
  - moderation
  - webui
  - implementation-plan
dependencies:
  - TASK-303
documentation:
  - >-
    Docs/superpowers/specs/2026-05-12-moderation-review-rules-remediation-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create the detailed implementation plan for the approved moderation design. The plan should break the work into reviewable stages covering route split, rules hardening, accessibility/responsive fixes, backend review contract and event capture, review queue MVP, audit/recovery, power-user workflows, and regression fixtures. This task is for the implementation plan artifact only; it should not implement code changes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Implementation plan exists under Docs/superpowers/plans and references the approved moderation design spec.
- [x] #2 Plan maps each stage to exact frontend/backend/docs/test files or file families to touch.
- [x] #3 Plan identifies stage dependencies and separates route/rules hardening work from backend review contract and frontend review queue work.
- [x] #4 Plan includes concrete test and verification commands for frontend, backend, route, accessibility/responsive, and E2E coverage.
- [x] #5 Plan review loop is completed or blockers are documented.
- [x] #6 Only the new plan and associated Backlog task changes are included in the planning commit.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Created implementation plan at Docs/superpowers/plans/2026-05-12-moderation-review-rules-remediation-implementation-plan.md. Local self-review found and fixed a Stage 1 dependency issue: the plan now creates the ModerationReview index export before route wrappers import it, and explicitly updates the existing rules shell visible copy to Content Rules. Plan-document-reviewer subagent dispatch is documented as blocked by current collaboration/tool rules because the user has not explicitly requested delegated agent work.

Verification for planning slice: staged set was limited to Docs/superpowers/plans/2026-05-12-moderation-review-rules-remediation-implementation-plan.md and backlog/tasks/task-305 - Write-moderation-review-and-rules-remediation-implementation-plan.md. `git diff --cached --check` passed with no whitespace errors. Bandit was not run because this slice changes only Markdown planning/task files, not backend code.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added a staged implementation plan for the approved moderation review and rules remediation design. The plan separates route/naming work, rules hardening, accessibility/responsive fixes, backend review contract/event capture, review queue MVP, audit/undo, power-user controls, and fixtures/docs verification. It documents exact file targets, dependencies, verification commands, CDP/Playwright browser checks, and the subagent-review limitation for this planning-only slice.

Staged verification confirmed that only the new plan artifact and TASK-305 were included in the planning commit scope.
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
