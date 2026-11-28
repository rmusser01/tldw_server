---
id: TASK-15495
title: Address open CodeQL alerts reported on v0.1.47 main
status: In Progress
labels:
- security
- codeql
references:
- Docs/Evidence/v0.1.47-codeql-follow-up.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Review the 15 open CodeQL findings on v0.1.47 main against current source. Patch confirmed problems and preserve evidence for any test-only findings or false positives. Keep the release tag and dev reconciliation PR intact.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Account for all 15 findings with code changes or trace-based rationale.
- [x] #2 Run focused regression checks and compare the hosted CodeQL results for the remediation PR.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Initial inventory: 2705, 2735–2747, 2759. Both VN idempotency helpers preserve the existing caller contract and recovery protocol. No new ADR is needed for replacing their random source.
Trace review of 15 findings is in Docs/Evidence/v0.1.47-codeql-follow-up.md. Changes address 2705, 2735, 2742, 2743 and 2747. The other ten findings stay open for reviewer confirmation, including the three OAuth/gateway fingerprints already tracked by TASK-13950.102. Verified 437 VN tests, 7 Admin Teams tests, 286 focused Python tests, and 27 loopback transport tests with a temporary curl dependency. Bandit reported no medium or high findings in the touched Python files. Broader local Chroma and MIME tests had failures, recorded in the evidence. Hosted CodeQL is pending.
Post-rebase rerun: 437 VN tests, 7 Teams tests, and 483 focused Python tests passed (1 skipped). With the supported Chroma 1.5.9 SDK, all 84 Chroma and DB-path tests passed. Nine upload MIME tests still fail when puremagic 1.30 falls back to the claimed extension. The path-containment finding stays open pending review.
PR #3467 is based on dev 15793ad67a. Hosted Python CodeQL run 43613731426 passed on d96bfb0533 with SARIF upload disabled and tests excluded. The release SARIF traces support individual dispositions in the evidence table. The HTML validation failure was reproduced and fixed. The rerun passed 567 focused Python tests with one skip, and 105 additional backup and download tests passed. Bandit found no medium or high severity issues in the touched production Python files.
Local CodeQL 2.27.4 ran the eight alert queries on the updated branch. The VN randomness, TLS fixture, and webhook log assertion results disappeared. Teams and the keyed model-inventory fingerprint still trigger the queries and have trace-backed false-positive evidence. The results introduced no new rule and file pairs compared with the release SARIF. All 437 VN and seven Teams tests passed on the updated source.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Accounted for all 15 release alerts in the evidence table, repaired VN randomness and the unkeyed model cache fingerprint, tightened HTML content validation, and hardened the Teams link and test fixtures. Verified the scoped CodeQL rerun, focused tests, Bandit, and pre-commit. PR #3467 is the review and integration record.
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
