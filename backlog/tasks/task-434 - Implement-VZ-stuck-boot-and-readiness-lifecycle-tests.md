---
id: TASK-434
title: Implement VZ stuck boot and readiness lifecycle tests
status: Done
labels:
- sandbox
- vz_linux
- lifecycle
- hardening
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the next slice from the VZ Linux lifecycle drill gaps design: host-independent coverage for stuck boot and stuck guest-readiness behavior. Verify helper and Python runner cleanup paths do not leave reusable VM/session state after boot/readiness failures, preserve manual/host-gated boundaries, and update operator/evidence docs as needed without adding workflow trigger expansion.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Host-independent helper or runner tests cover boot-driver failure cleanup without requiring a real VZ VM.
- [x] #2 Host-independent helper or runner tests cover guest-readiness timeout/failure cleanup without marking a session VM reusable.
- [x] #3 Diagnostics or evidence docs identify the stuck boot/readiness contract and expected evidence without exposing raw serial logs through API output.
- [x] #4 Implementation reuses existing helper/runner cleanup primitives rather than adding broad repair automation.
- [x] #5 Focused verification, git diff hygiene, and Bandit for touched Python scope are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented the host-independent stuck boot/readiness slice:

- Added Swift helper coverage for boot-driver failure cleanup before a VM reaches readiness.
- Added Swift service-level coverage for boot-driver failure and guest-readiness failure through the same `HelperService.createVM` path used by real execution.
- Added Python runner coverage for a session-mode readiness failure after a stale/unhealthy reuse candidate is rejected. The test verifies the stale session-control row is deleted, no replacement row is stored, guest execution is skipped, helper termination is not attempted without a returned `vm_id`, and active runner maps are clear.
- Updated operator/evidence docs and doc-contract tests to require stable reason/error evidence, session-control outcome, helper stdout/stderr paths, serial-log pointers only, and no raw serial log exposure through diagnostics/docs.

Verification:

- `swift test --filter 'createVMRemovesBootingRecordWhenBootDriverFails|helperServiceCreateVMClearsRegistryWhenBootDriverFails|helperServiceCreateVMClearsRegistryWhenReadinessFails'` passed.
- `swift test` passed: 91 tests. Existing warning: SwiftPM reports `Tests/test_vz_helperctl.py` as an unhandled file.
- `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/sandbox/test_vz_linux_runner.py -q` passed: 28 tests.
- `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Infrastructure/test_vz_linux_host_gated_workflow.py -q` passed: 18 tests.
- `git diff --check` passed.
- `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -q tldw_Server_API/app/core/Sandbox/runners/vz_linux_runner.py` passed.
- Bandit on touched test files reports the existing pytest `assert` and hardcoded temp-path baseline; rerunning with `B101,B108` skipped passed.
- PR review follow-up: changed the new stuck boot/readiness doc-contract test to use `_normalized_existing_text(...)` for both docs, preserving this module's targeted missing-file failure messages.
- PR review verification: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Infrastructure/test_vz_linux_host_gated_workflow.py -q` passed; Bandit on that test file passed; `git diff --check` passed.

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed host-independent stuck boot/readiness lifecycle coverage and documentation. The slice adds Swift helper/service cleanup tests, a Python session-mode readiness failure cleanup test, and evidence/doc-contract updates for stable reason codes and artifact-pointer-only diagnostics.
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
