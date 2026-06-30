---
id: TASK-145
title: Add manual opt-in VZ Linux host failure drills
status: Done
assignee: []
created_date: '2026-05-09 03:50'
updated_date: '2026-05-09 04:10'
labels:
  - sandbox
  - vz_linux
  - host-gated
  - recovery
dependencies: []
references:
  - .github/workflows/vz-linux-host-gated.yml
  - tools/vz-linux-image/scripts/run-host-e2e-smoke.sh
  - tldw_Server_API/tests/sandbox/test_vz_linux_real_host_e2e.py
  - tools/vz-linux-image/tests/test_host_e2e_smoke_script.py
documentation:
  - Docs/Sandbox/vz-linux-host-gated-ci-acceptance-policy.md
  - tldw_Server_API/app/core/Sandbox/README.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add a focused host-gated recovery drill path for real vz_linux execution on prepared Apple Silicon runners. The drill must stay manual opt-in so normal nightly/manual smoke remains stable while operators can explicitly validate stale-session recovery behavior after the baseline real execution smoke passes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The host E2E smoke script exposes a manual opt-in failure-drill flag that is disabled by default.
- [x] #2 The GitHub host-gated workflow exposes a workflow_dispatch input for failure drills and does not enable them for scheduled runs by default.
- [x] #3 A real-host pytest drill verifies a session VM can be invalidated through the helper and that the next same-session command provisions a fresh VM and completes without reusing stale control state.
- [x] #4 The failure drill is guarded by a dedicated marker or equivalent so normal host smoke can run without the drill.
- [x] #5 Host-gated policy or sandbox docs describe the manual-only failure-drill contract and safety constraints.
- [x] #6 Focused tests validate script/workflow wiring without requiring a real VZ host.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Spec: Docs/superpowers/specs/2026-05-09-vz-linux-host-failure-drills-design.md
Plan: Docs/superpowers/plans/2026-05-09-vz-linux-host-failure-drills-implementation-plan.md
Implementation approach: add a manual-only failure-drill script flag and workflow input, a dedicated pytest marker for stale session VM recovery, focused host-independent wiring tests, and policy docs.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Verification: python -m pytest tools/vz-linux-image/tests/test_host_e2e_smoke_script.py -q -> 9 passed, 1 skipped; python -m pytest tldw_Server_API/tests/Infrastructure/test_vz_linux_host_gated_workflow.py -q -> 13 passed; python -m pytest tldw_Server_API/tests/sandbox/test_vz_linux_real_host_e2e.py -m vz_linux_host_failure_drill -q -rs -> 1 skipped because TLDW_SANDBOX_VZ_LINUX_E2E is not set locally; git diff --check -> clean; Bandit on touched real-host test with B101 skipped -> 0 results.

Follow-up PR review pass: verified current Qodo findings on brittle helper terminate assertion and workflow_dispatch input access before patching.

Review fixes completed: the failure drill now verifies VM health before and after helper termination and only skips if helper termination cannot invalidate a still-healthy VM; workflow contract test now validates workflow_dispatch/input shape before indexing.

Review verification: python -m pytest tldw_Server_API/tests/Infrastructure/test_vz_linux_host_gated_workflow.py -q -> 13 passed; python -m pytest tools/vz-linux-image/tests/test_host_e2e_smoke_script.py -q -> 9 passed, 1 skipped; python -m pytest tldw_Server_API/tests/sandbox/test_vz_linux_real_host_e2e.py -m vz_linux_host_failure_drill -q -rs -> 1 skipped because TLDW_SANDBOX_VZ_LINUX_E2E is not set locally; git diff --check -> clean; Bandit on touched tests with B101 skipped -> 0 results.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added manual opt-in VZ Linux host failure drills. The host smoke script and GitHub workflow now expose failure-drill wiring disabled by default, the real-host pytest module has a dedicated stale session VM replacement drill, docs describe the manual-only safety contract, and focused verification passed with the real VZ drill skipped locally due missing TLDW_SANDBOX_VZ_LINUX_E2E opt-in.

PR review follow-up hardened the stale-session drill and workflow contract test. The drill now treats helper terminate false results as operational state to verify with VM health instead of a hard assertion, while still requiring the VM to be unhealthy/missing before testing fresh provisioning. The workflow test now fails with explicit shape assertions instead of a KeyError if workflow_dispatch inputs are missing.
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
