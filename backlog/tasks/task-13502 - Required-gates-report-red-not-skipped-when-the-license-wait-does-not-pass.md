---
id: TASK-13502
title: Required gates report red, not skipped, when the license wait does not pass
status: Done
labels:
- ci
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
GitHub counts a skipped required job as satisfied. Each required workflow has its own await_license job, and when it is cancelled (runner starvation: 'The job was not acquired by Runner of type hosted even after multiple attempts', seen on PR #3188 on 2026-10-05) or reports a negative verdict, the gate job of security-required, coverage-required, e2e-required, frontend-required and container-build-check is skipped and reads as passed. Only backend-required already turns this case red. Seen live: PR #3188's container-build-check showed 'skipping' with nothing built. Same class as the change-detection fix in TASK-13462 (owner: 'fix item 2').
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Every required gate job runs and fails with a clear error when its license wait was cancelled, failed or reported a negative verdict on a pull_request event
- [x] #2 A passing license wait and the workflow_run admission path behave exactly as before
- [x] #3 Contract tests pin the new arm and refusal step for all six gates, and reverting any of them fails a test
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Each required workflow has its own await_license job. When it was cancelled (runner starvation), failed or reported a negative verdict, every gate but backend-required was skipped, and GitHub counts a skipped required job as satisfied. Seen live on PR #3188 on 2026-10-05: container-build-check showed 'skipping' with nothing built.

Fix: security-required, coverage-required, e2e-required, frontend-required and container-build-check get backend-required's third if arm and its 'Require the license audit to have passed' refusal as their first step. coverage-required and e2e-required now need admission and await_license themselves (they previously keyed only on the changes job), so their checkout ref follows the admitted head like every other admission-guarded job.

Unchanged: the pull_request path with a passing license wait, and the workflow_run path (admission declined still skips).

Tests: truth table over every (event, admission, license wait, changes, cancelled) row now expects 'license' red for all gates; per-gate refusal pins; contract pins in test_license_first_workflow_contracts.py (LICENSE_RED_GATES) and test_required_workflow_contracts.py. Removing the red arm from any one gate fails 2-4 tests. tests/CI + concurrency policy: 768 passed, 0 failed. actionlint clean.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
All six required gates now report red instead of skipped when their license wait does not pass.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
