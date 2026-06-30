---
id: TASK-133
title: Add host-gated VZ Linux recovery smoke coverage
status: Done
assignee: []
created_date: '2026-05-09 00:26'
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
  - tldw_Server_API/tests/Infrastructure/test_vz_linux_host_gated_workflow.py
documentation:
  - Docs/Sandbox/macos-runtime-operator-notes.md
  - Docs/Sandbox/vz-linux-host-gated-ci-acceptance-policy.md
  - Docs/Sandbox/sandbox-runtime-capability-inventory.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Extend the existing Apple Silicon host-gated VZ Linux operator/CI smoke path so it exercises the existing read-only diagnostics and dry-run reconciliation repair surfaces after real ephemeral execution and same-session reuse. Keep the smoke non-destructive by default and preserve the existing operator-first workflow through run-host-e2e-smoke.sh and vz-helperctl smoke.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The host-gated smoke script runs a recovery/diagnostics pytest slice after the real execution/reuse checks without enabling destructive repair actions.
- [x] #2 The host-gated workflow contract tests verify the operator smoke path includes recovery coverage and remains branch-gated/manual-or-nightly only.
- [x] #3 Operator docs and the host-gated acceptance policy state that recovery smoke covers diagnostics plus dry-run repair only and does not terminate VMs or delete state.
- [x] #4 Focused tests validate the new script/workflow behavior without requiring a real VZ host in normal CI.
<!-- AC:END -->

## Notes

<!-- SECTION:NOTES:BEGIN -->
- Added explicit host-gated recovery smoke coverage after ephemeral execution and same-session reuse. The recovery smoke seeds only the isolated test store and verifies diagnostics plus `repair_macos_reconciliation(dry_run=true)`.
- Addressed PR review comments by adding a `pytest.MonkeyPatch` type annotation and docstring to the new recovery test, using safe diagnostics `.get()` access with `_expect` failure messages, and refactoring the smoke script to run one registered `vz_linux_host_smoke` marker through a shared pytest/env helper.
- Verification:
  - `python -m pytest tools/vz-linux-image/tests/test_host_e2e_smoke_script.py -q` passed with 8 passed, 1 skipped.
  - `python -m pytest tldw_Server_API/tests/Infrastructure/test_vz_linux_host_gated_workflow.py -q` passed with 11 passed.
  - `python -m pytest tldw_Server_API/tests/sandbox/test_vz_linux_real_host_e2e.py -q` passed with 4 passed, 3 skipped on this host because real VZ E2E env was not enabled.
  - `python -m pytest tldw_Server_API/tests/sandbox/test_vz_linux_real_host_e2e.py -m vz_linux_host_smoke -q` selected the three marked real-host smoke tests and skipped them on this host because real VZ E2E env was not enabled.
  - `git diff --check` passed.
  - Bandit over touched Python with only `B101` skipped reported pre-existing test-harness findings in `tools/vz-linux-image/tests/test_host_e2e_smoke_script.py` (`B404`, `B603`, `B108`) on unchanged lines. Re-run with those known test-harness checks also skipped exited 0.
<!-- SECTION:NOTES:END -->

## Final Summary
<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added host-gated VZ Linux recovery smoke coverage after real ephemeral execution and same-session reuse. The smoke path now runs a non-destructive diagnostics and dry-run reconciliation slice, workflow contract tests verify the operator path remains manual/nightly and branch-gated, and docs clarify that recovery smoke does not terminate VMs or delete state.
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
