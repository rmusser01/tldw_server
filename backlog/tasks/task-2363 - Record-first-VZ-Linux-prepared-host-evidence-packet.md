---
id: TASK-2363
title: Record first VZ Linux prepared-host evidence packet
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-16 14:00'
labels:
  - sandbox
  - docs
  - vz_linux
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1442'
  - Docs/Sandbox/vz-linux-prepared-host-evidence.md
  - Docs/Sandbox/vz-linux-host-gated-ci-acceptance-policy.md
  - Docs/Sandbox/macos-runtime-operator-notes.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Run the prepared-host VZ Linux smoke path where available and update the prepared-host evidence tracker with the actual host facts, commands, results, expected skips, blockers, and residual gaps. Scope is evidence/documentation only unless the smoke exposes a minimal setup/doc issue that must be corrected to record accurate evidence.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Prepared-host smoke commands and environment are inspected from current docs/scripts before running anything destructive.
- [x] #2 A real VZ Linux smoke is run if the prepared host bundle/helper prerequisites are available, or the operator-setup blocker is recorded precisely if they are not.
- [x] #3 The prepared-host evidence tracker records a dated evidence packet with commands, results, expected skips, artifacts/log pointers, and residual gaps.
- [x] #4 Verification results and any Bandit skip rationale are recorded in the Backlog task.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Inspected current operator docs and run-host-e2e-smoke.sh before execution. Used local evidence dir /var/folders/p_/x47tgtn57cv43r7yxxn40tyh0000gn/T/tldw-vz-evidence-20260616-065631, canonical Debian Bookworm arm64 bundle, repo entitlements, and project Python 3.11. Default smoke only; manual failure, launchd, stale-socket, host reboot, and boot-fault drills were intentionally skipped and recorded as residual evidence gaps.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Recorded the first local prepared-host VZ Linux evidence packet. The run used the canonical Debian Bookworm arm64 bundle, built and ad hoc signed the macOS VZ helper with the checked-in virtualization entitlement, ran helper daemon smoke, ran real vz_linux ephemeral execution, same-session reuse, and recovery diagnostics/dry-run repair smoke, and retained artifact/log pointers without pasting raw serial logs. Verification: real smoke via vz-helperctl.py smoke completed with helper daemon smoke 2 passed and vz_linux host smoke 3 passed, 11 deselected; git diff --check exited 0; stale prepared-host wording rg returned no matches; replacement evidence rg found expected lines; python -m pytest tldw_Server_API/tests/Infrastructure/test_vz_linux_host_gated_workflow.py -q passed with 18 passed. Bandit was not run because only Markdown/Backlog docs were intentionally edited.
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
