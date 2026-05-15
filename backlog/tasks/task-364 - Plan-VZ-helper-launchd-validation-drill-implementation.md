---
id: TASK-364
title: Plan VZ helper launchd validation drill implementation
status: Done
assignee: []
created_date: '2026-05-15 03:29'
labels:
  - Sandbox
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1442'
documentation:
  - Docs/superpowers/specs/2026-05-15-vz-helper-launchd-validation-drill-design.md
  - Docs/superpowers/plans/2026-05-15-vz-helper-launchd-validation-drill.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Write the implementation plan for the reviewed VZ helper launchd validation drill design. The plan should be execution-ready for a future PR, scoped to a single focused slice that adds an explicit operator-owned `vz-helperctl.py launchd-drill`, portable tests, and docs while preserving direct-helper smoke defaults and avoiding automatic launchd installation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Plan starts from the reviewed launchd validation drill spec and maps exact files to modify or create.
- [x] #2 Plan decomposes implementation into TDD-sized tasks with concrete tests, commands, and expected outcomes.
- [x] #3 Plan preserves the design constraints: isolated drill label by default, pre-bootstrap loaded-service guard, drill-owned bootout cleanup only, launchd-mode PID-file expectations, and external-helper smoke semantics.
- [x] #4 Plan includes verification, docs updates, Bandit guidance, and PR handoff expectations.
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->

## Notes

<!-- SECTION:NOTES:BEGIN -->
- Verification: `git diff --check` passed for the docs-only planning slice.
- Bandit: skipped for this planning slice because it only adds a plan document and Backlog task record. The implementation plan requires Bandit for the future Python changes.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added `Docs/superpowers/plans/2026-05-15-vz-helper-launchd-validation-drill.md`, an execution-ready implementation plan for the reviewed launchd validation drill design. The plan breaks the future implementation into TDD-sized tasks for helperctl defaults, loaded-service guards, drill orchestration, external-helper VZ smoke, CLI output, docs, verification, Bandit, optional real-host validation, and PR handoff.
<!-- SECTION:FINAL_SUMMARY:END -->
