---
id: TASK-15572
title: Tier server development CI and require full qualification before publication
status: In Progress
labels:
- ci
- release
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Reduce CI fan-out for explicitly mapped low-risk development changes while retaining nightly regression and requiring full same-commit server qualification before publication.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Existing required PR gates stay unchanged; opt-in selects bounded smoke only for mapped same-repository dev changes and uncertain inputs retain broad regression.
- [ ] #2 Nightly runs the broad Linux regression suite at the dispatched dev SHA, deduplicates successful or active runs for that SHA and retries failed attempts.
- [ ] #3 Main promotion and server package and image publication depend on full cross-platform and package checks on their own event SHA, with missing and skipped results rejected.
- [x] #4 Targeted routing and publication contracts pass, and rollout, live qualification, protection and rollback steps are documented.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR required: yes. ADR path: Docs/ADR/096-tiered-development-and-release-ci.md. The workflow changes the durable PR, nightly, and release gating policy and preserves ADR-063 merge queue and existing license admission. Implementation order: record ADR and rollout, prove routing and verdict contracts, reuse the existing matrices and package checks for release callers, then verify focused tests and workflow lint. Keep opt-in off until scheduled and release qualification are proven.
Local implementation on codex/server-tiered-ci. All 350 focused workflow and Backlog contracts, 25 backend smoke tests and six mocked dispatcher cases passed. Actionlint and targeted Ruff passed. Bandit reported no issues in new Python files. The wheel and sdist passed make pypi-check, and the wheel imported in a clean local install without runtime dependencies. All six baseline full matrix definitions, runner types, timeouts and test steps remain intact, with main PR jobs selecting the merge SHA. Pending GitHub qualification includes a normal associated PR, the scheduled nightly at its dev SHA, a non-publishing release validation with dependency installation, and verification of strict branch protection before enabling light CI.
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
