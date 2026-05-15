---
id: TASK-360
title: Design VZ helper launchd validation drill
status: Done
assignee: []
created_date: '2026-05-15 03:15'
labels:
  - Sandbox
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1442'
documentation:
  - Docs/Sandbox/macos-runtime-operator-notes.md
  - Docs/Sandbox/vz-linux-host-gated-ci-acceptance-policy.md
  - Docs/superpowers/plans/2026-05-13-vz-helper-launchd-operator.md
  - Docs/superpowers/specs/2026-05-15-vz-helper-launchd-validation-drill-design.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create the design/spec for an explicit operator-owned launchd validation drill for the macOS VZ helper. The design must build on the existing `vz-helperctl.py launchd` commands already merged on dev, keep the default direct-helper smoke path unchanged, preserve explicit opt-in/no hidden install semantics, and define safety, cleanup, logging, and host-gated validation boundaries for a future implementation PR.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Spec documents the drill scope, lifecycle sequence, safety boundaries, cleanup behavior, and non-goals.
- [x] #2 Spec explains why the drill is separate from the default direct-helper smoke path and how it reuses existing launchd commands.
- [x] #3 Spec defines portable validation expectations and host-gated/manual validation expectations without requiring scheduled CI reboot or automatic launchd installation.
- [x] #4 Spec references the sandbox tracker and relevant operator docs so future implementers can find the broader roadmap context.
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
- Design scope approved by the user before writing the spec.
- This is a docs-only brainstorming/spec slice. No runtime code changes are included.
- Verification: `git diff --check` passed.
- Bandit: skipped because the slice only adds docs and a Backlog task record.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added the approved launchd validation drill design in `Docs/superpowers/specs/2026-05-15-vz-helper-launchd-validation-drill-design.md`. The spec defines an explicit operator-owned drill on top of the existing `vz-helperctl.py launchd` commands, keeps the default direct-helper smoke path unchanged, calls out the existing shell smoke ownership boundary, and captures safety, cleanup, logging, portable tests, and host-gated/manual validation expectations for the future implementation slice.
<!-- SECTION:FINAL_SUMMARY:END -->
