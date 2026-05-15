---
id: TASK-367
title: Implement VZ helper launchd validation drill
status: In Progress
assignee: []
created_date: '2026-05-15 03:37'
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
Implement the reviewed and planned VZ helper launchd validation drill. Add an explicit operator-owned `vz-helperctl.py launchd-drill` command with isolated default labels, pre-bootstrap loaded-service guard, drill-owned bootout cleanup, launchd-mode helper readiness without requiring a helperctl pid file, optional external-helper VZ Linux smoke, portable tests, operator docs, and verification. Preserve the default direct-helper smoke path and do not add automatic launchd installation or scheduled workflow integration.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 `vz-helperctl.py launchd-drill` exists and uses isolated drill labels/private plist paths by default while supporting explicit label/path overrides.
- [ ] #2 The drill refuses pre-existing loaded launchd services, only bootouts targets it bootstrapped, preserves primary failures when cleanup also fails, and treats missing helperctl pid files as valid in launchd mode when helper ping/protocol are healthy.
- [ ] #3 Optional VZ Linux smoke runs against the launchd-managed socket without starting a second helper; default direct-helper smoke behavior remains unchanged.
- [ ] #4 Portable helperctl tests cover defaults, loaded-service guard, sequencing, cleanup, CLI output, JSON shape, and external-helper smoke command construction.
- [ ] #5 Operator docs and host-gated policy document the drill, expected skips, cleanup behavior, and manual/host-gated validation boundaries.
- [ ] #6 Focused helperctl tests, `git diff --check`, and Bandit on touched Python code are run or documented with explicit host-gated skips.
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->

## Notes

<!-- SECTION:NOTES:BEGIN -->
- Implementation follows `Docs/superpowers/plans/2026-05-15-vz-helper-launchd-validation-drill.md`.
<!-- SECTION:NOTES:END -->
