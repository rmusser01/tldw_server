---
id: TASK-2362
title: Reconcile sandbox tracker docs after VZ lifecycle drills
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-16 13:49'
labels:
  - sandbox
  - docs
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1442'
  - Docs/Sandbox/sandbox-runtime-capability-inventory.md
  - Docs/Sandbox/vz-linux-prepared-host-evidence.md
  - Docs/Sandbox/vz-linux-host-gated-ci-acceptance-policy.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Refresh sandbox tracking documentation after the merged VZ lifecycle and host reboot drill slices so contributors can distinguish completed explicit manual/operator-gated drills from remaining default-smoke, scheduled-CI, and cross-runtime gaps. Keep this docs/backlog-only and do not change runtime behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Capability inventory no longer implies launchd or host reboot drill procedures are undefined; it identifies them as explicit manual/operator-gated paths with remaining scheduled/default coverage gaps.
- [x] #2 Prepared-host evidence tracker distinguishes skipped manual drills from genuinely uncovered lifecycle gaps.
- [x] #3 Backlog task records verification and final status for this docs-only reconciliation.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Verified current dev already marks launchd drill design/plan/implementation tasks as Done and includes completed host reboot drill tasks. Updated only the remaining docs wording that blurred completed explicit operator drills with still-open default/scheduled/cross-runtime gaps.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Updated the sandbox capability inventory, prepared-host evidence tracker, and macOS operator notes so completed VZ launchd/host-reboot/lifecycle drill work is not described as undefined. The docs now distinguish explicit manual/operator-gated drills from remaining default-smoke, scheduled-CI, prepared-host evidence, mutating repair, and broader helper crash gaps. Verification: git diff --check exited 0; targeted stale-phrase rg search returned no matches; targeted replacement-wording rg search found the expected updated lines. Bandit was not run because this reconciliation touched only Markdown/Backlog documentation.
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
