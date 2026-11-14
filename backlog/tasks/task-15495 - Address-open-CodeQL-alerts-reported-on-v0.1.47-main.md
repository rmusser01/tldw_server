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
- [ ] #2 Run focused regression checks and compare the hosted CodeQL results for the remediation PR.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Initial inventory: 2705, 2735–2747, 2759. Both VN idempotency helpers preserve the existing caller contract and recovery protocol. No new ADR is needed for replacing their random source.
Trace review of 15 findings is in Docs/Evidence/v0.1.47-codeql-follow-up.md. Changes address 2705, 2735, 2742, 2743 and 2747. The other ten findings stay open for reviewer confirmation, including the three OAuth/gateway fingerprints already tracked by TASK-13950.102. Verified 437 VN tests, 7 Admin Teams tests, 286 focused Python tests, and 27 loopback transport tests with a temporary curl dependency. Bandit reported no medium or high findings in the touched Python files. Broader local Chroma and MIME tests had failures, recorded in the evidence. Hosted CodeQL is pending.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
