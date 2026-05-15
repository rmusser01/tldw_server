---
id: TASK-367
title: Implement VZ helper launchd validation drill
status: Done
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
- [x] #1 `vz-helperctl.py launchd-drill` exists and uses isolated drill labels/private plist paths by default while supporting explicit label/path overrides.
- [x] #2 The drill refuses pre-existing loaded launchd services, only bootouts targets it bootstrapped, preserves primary failures when cleanup also fails, and treats missing helperctl pid files as valid in launchd mode when helper ping/protocol are healthy.
- [x] #3 Optional VZ Linux smoke runs against the launchd-managed socket without starting a second helper; default direct-helper smoke behavior remains unchanged.
- [x] #4 Portable helperctl tests cover defaults, loaded-service guard, sequencing, cleanup, CLI output, JSON shape, and external-helper smoke command construction.
- [x] #5 Operator docs and host-gated policy document the drill, expected skips, cleanup behavior, and manual/host-gated validation boundaries.
- [x] #6 Focused helperctl tests, `git diff --check`, and Bandit on touched Python code are run or documented with explicit host-gated skips.
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
- Implementation follows `Docs/superpowers/plans/2026-05-15-vz-helper-launchd-validation-drill.md`.
- Review process: each implementation/docs task received spec-compliance and code-quality/doc-quality review; Task 4 and Task 5 review findings were fixed and re-approved.
- Verification: `python -m pytest tools/macos-vz-helper/Tests/test_vz_helperctl.py -q` passed with 123 passed and 1 skipped.
- Verification: `git diff --check` passed.
- Verification: `python -m bandit -r tools/macos-vz-helper/scripts/vz-helperctl.py -f json -o /tmp/bandit_vz_launchd_drill.json` completed with 0 errors and 0 findings.
- Host-gated real launchd/VM smoke: not run in this final portable verification pass; documented as an explicit prepared-host/manual validation path.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the opt-in VZ helper `launchd-drill` workflow. The helperctl CLI now has isolated launchd drill defaults, pre-bootstrap loaded-service protection, drill-owned cleanup, launchd-managed helper readiness checks, optional external-helper `vz_linux` host smoke through the launchd-managed socket, JSON/human output, and portable unit coverage. Operator docs and host-gated policy now describe cleanup, expected skips, manual LaunchAgent validation boundaries, and that the default direct-helper smoke path remains unchanged.
<!-- SECTION:FINAL_SUMMARY:END -->
