---
id: TASK-327
title: Plan VN script authoring graph API implementation
status: Done
assignee: []
created_date: '2026-05-14 01:37'
updated_date: '2026-05-14 01:52'
labels:
  - vn
  - scripts
  - plan
  - api
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1391'
  - 'https://github.com/rmusser01/tldw_server/pull/1641'
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create a detailed implementation plan from the approved VN script authoring graph API design. The plan must be backend-only, TDD-oriented, and scoped to a future implementation PR that adds computed graph/outline APIs for stored drafts, supplied draft previews, and published versions without WebUI changes or graph persistence.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Implementation plan is written under Docs/superpowers/plans and references the approved graph design spec.
- [x] #2 Plan decomposes the work into bite-sized backend tasks with exact files, tests, commands, and expected results.
- [x] #3 Plan preserves the design constraints: no model calls, no mutation, no graph persistence, conservative static edge semantics, encoded stable IDs, bracket JSON paths, limits, content hashes, and pinned published-version context.
- [x] #4 Plan includes final verification commands for VN script tests, compile checks, Bandit on touched Python scope, docs hygiene, and git diff checks.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Write `Docs/superpowers/plans/2026-05-14-vn-script-authoring-graph-api-implementation-plan.md`.
2. Review the plan against the approved design and current VN script files.
3. Patch any gaps found during review.
4. Run markdown/diff hygiene checks and record verification.
<!-- SECTION:PLAN:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Planning is being done on branch `codex/vn-script-authoring-graph-design` after the design spec commit.
- Created `Docs/superpowers/plans/2026-05-14-vn-script-authoring-graph-api-implementation-plan.md`.
- Review pass corrected the label ID encoding helper example and the published-version service test shape to match current `publish_script()` return data.
<!-- SECTION:NOTES:END -->

## Verification

<!-- SECTION:VERIFICATION:BEGIN -->
- Reviewed the plan against `Docs/superpowers/specs/2026-05-14-vn-script-authoring-graph-design.md` and current VN script service/API behavior.
- `git diff --check` -> passed.
- Bandit skipped because this task only changes Markdown plan/task files.
<!-- SECTION:VERIFICATION:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Wrote the backend-only VN script authoring graph API implementation plan. The plan breaks implementation into pure graph builder, service methods, API schemas/endpoints, capability/docs, and final verification tasks, with TDD steps and exact commands for focused pytest, compileall, Bandit, and diff hygiene.
<!-- SECTION:FINAL_SUMMARY:END -->
