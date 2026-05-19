---
id: TASK-432
title: Design VZ Linux lifecycle drill gap contract
status: Done
assignee: []
created_date: ''
updated_date: '2026-05-18 19:56'
labels:
  - sandbox
  - vz_linux
  - design
  - host-gated
dependencies: []
references:
  - Docs/superpowers/specs/2026-05-02-sandbox-module-roadmap-design.md
  - Docs/Sandbox/vz-linux-prepared-host-evidence.md
  - Docs/superpowers/specs/2026-05-18-vz-linux-lifecycle-drill-gaps-design.md
  - tldw_Server_API/tests/Infrastructure/test_vz_linux_host_gated_workflow.py
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Write the approved design/spec for the remaining vz_linux lifecycle drill gaps: stale socket, stuck boot/readiness, guest-agent mismatch, and host reboot/manual boundaries. Keep the slice documentation/test-focused and preserve manual/host-gated execution only.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Spec defines stale socket, stuck boot/readiness, guest-agent mismatch, and host reboot/manual lifecycle drill contracts.
- [x] #2 Spec preserves manual/host-gated execution only and explicitly excludes PR/push triggers, scheduled destructive drills, automatic host reboot, broad VM termination, and repair-default mutation.
- [x] #3 Evidence tracker, host-gated acceptance policy, or operator notes point contributors at the lifecycle drill contract where relevant.
- [x] #4 Focused doc-contract tests cover the spec anchors and manual-only boundaries.
- [x] #5 Verification records focused pytest, diff hygiene, and Bandit for touched Python test scope.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Approved design direction: create a docs/spec slice first, then use it to guide later manual drill implementation. No runtime behavior or workflow triggers should change in this slice.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

Implemented approved design/spec slice:
- Added Docs/superpowers/specs/2026-05-18-vz-linux-lifecycle-drill-gaps-design.md.
- Linked the drill contract from the prepared-host evidence tracker and sandbox roadmap.
- Added focused doc-contract tests for drill anchors and manual-only boundaries.
- No runtime behavior or workflow triggers changed.

PR review follow-up:
- Added roadmap and lifecycle drill spec path constants to the doc-contract tests.
- Added targeted doc existence assertions before normalized text reads.
- Wrapped long lifecycle drill spec assertions for PEP 8/readability.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Designed the remaining vz_linux lifecycle drill gap contract for stale socket, stuck boot/readiness, guest-agent mismatch, and host reboot/manual boundaries. The spec keeps future work split into narrow manual/host-gated slices and explicitly excludes PR/push triggers, scheduled destructive drills, automatic host reboot, broad VM termination, and repair-default mutation.

Verification:
- /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Infrastructure/test_vz_linux_host_gated_workflow.py -q
- git diff --check
- /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -q tldw_Server_API/tests/Infrastructure/test_vz_linux_host_gated_workflow.py

Known skips: no real VZ VM smoke or lifecycle drill was run in this design-only slice; future implementation slices must add targeted host-independent and manual host-gated coverage.
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
