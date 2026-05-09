---
id: TASK-153
title: Add manual VZ Linux helper restart recovery drill
status: Done
assignee: []
created_date: '2026-05-09 04:26'
updated_date: '2026-05-09 05:05'
labels:
  - sandbox
  - vz_linux
  - host-gated
  - recovery
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/1397'
  - .github/workflows/vz-linux-host-gated.yml
  - tools/vz-linux-image/scripts/run-host-e2e-smoke.sh
  - tools/macos-vz-helper/scripts/vz-helperctl.py
  - tldw_Server_API/tests/sandbox/test_vz_linux_real_host_e2e.py
documentation:
  - Docs/Sandbox/sandbox-architecture-doctrine.md
  - Docs/Sandbox/sandbox-runtime-capability-inventory.md
  - Docs/Sandbox/vz-linux-host-gated-ci-acceptance-policy.md
  - Docs/Sandbox/macos-runtime-operator-notes.md
  - Docs/superpowers/specs/2026-05-02-sandbox-module-roadmap-design.md
  - >-
    Docs/superpowers/plans/2026-05-09-vz-linux-helper-restart-recovery-drill-implementation-plan.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add the next narrow host-gated recovery slice for vz_linux: a manual-only drill that validates session recovery behavior when the macOS VZ helper is stopped and restarted after a session VM has been created. This follows PR #1397's stale-VM termination drill and the sandbox roadmap's Phase 1/Phase 4 recovery goals. Scope must stay operator-first and host-gated; do not add host reboot automation, launchd bootstrap/install behavior, networking changes, or broad repair generalization in this task.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A design spec is added for the manual helper restart recovery drill and reviewed for scope risks before implementation.
- [x] #2 The drill remains opt-in through the existing failure-drill path and is never enabled by default for scheduled or normal host smoke runs.
- [x] #3 The planned drill validates that a same-session command after helper stop/start does not reuse stale control state and can provision or recover cleanly.
- [x] #4 The plan defines how helper lifecycle ownership is handled without adding host reboot automation, launchd bootstrap/install behavior, networking changes, or broad repair generalization.
- [x] #5 Focused host-independent tests and host-gated verification expectations are specified for the implementation slice.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Design-first slice. Write a focused helper restart recovery drill spec, review it for lifecycle ownership and host-gated safety risks, record verification, then ask for human review before implementation planning.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Created design spec: Docs/superpowers/specs/2026-05-09-vz-linux-helper-restart-recovery-drill-design.md. Key design decision: keep the lower-level smoke script as helper lifecycle owner and add an explicit restart lease/pid-file for manual failure drills instead of refactoring all smoke startup through vz-helperctl in this PR.

Design self-review completed. Reviewed the spec for lifecycle ownership, private pid-file cleanup, direct pytest safety, default scheduled CI behavior, and scope creep. The design intentionally avoids host reboot automation, launchd bootstrap/install, networking changes, and broad repair generalization. Bandit is not applicable for this design-only checkpoint.

Design hardening review found implementation-risk gaps and patched the spec before planning: pid-file lease must live under the private socket directory, use lstat/regular-file and positive-PID validation, verify process command matches the helper before signaling, bound helper-stop waits, avoid arbitrary socket deletion, add wrapper pass-through coverage for vz-helperctl smoke --include-failure-drills, and document the old-VM/orphan assumption as a separate follow-up risk.

Implementation plan added at Docs/superpowers/plans/2026-05-09-vz-linux-helper-restart-recovery-drill-implementation-plan.md. Pre-execution plan review patched three implementation hazards before coding: require ProcessLookupError for replacement helper cleanup checks, include the signal import for SIGTERM, and validate env strings before constructing Path objects so missing helper lease env cannot collapse to the current directory.

Implementation completed: smoke script now grants an opt-in restart lease only for manual failure drills, helperctl forwards --include-failure-drills, real-host pytest validates private PID-file restart ownership and adds a helper restart recovery drill, and operator docs now distinguish this manual helper restart coverage from host reboot, launchd, networking, and broad repair gaps.

Verification after rebase on origin/dev: focused pytest suite passed with 108 passed, 2 skipped, 5 deselected; bash -n passed for run-host-e2e-smoke.sh; py_compile passed for vz-helperctl.py and test_vz_linux_real_host_e2e.py; host-gated failure-drill selection skipped 2 tests because TLDW_SANDBOX_VZ_LINUX_E2E was not set; Bandit report /tmp/bandit_vz_helper_restart_recovery.json had zero findings after skipping pytest assert noise with -s B101; git diff --check was clean.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added a manual-only VZ Linux helper restart recovery drill. The smoke harness now exposes a private restart lease only for opt-in failure drills, helperctl forwards --include-failure-drills, pytest validates restart PID ownership and same-session stale VM replacement after helper restart, and docs clarify this covers the smoke-owned helper restart path but not host reboot, launchd, networking, or broad repair automation. Host-independent verification passed; real VZ execution remains host-gated and skipped locally without TLDW_SANDBOX_VZ_LINUX_E2E.
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
