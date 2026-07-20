---
id: TASK-12977
title: Bootstrap base-controlled frontend license gate
status: In Progress
labels:
- licensing
- security
- ci
- frontend
priority: high
documentation:
- Docs/superpowers/specs/2026-07-20-base-controlled-frontend-license-gate-design.md
references:
- https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows
- https://docs.github.com/en/actions/reference/security/secure-use
- https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets/available-rules-for-rulesets
modified_files:
- Docs/superpowers/specs/2026-07-20-base-controlled-frontend-license-gate-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Replace the PR-controlled temporary frontend contribution gate with a base-controlled GitHub Actions trust root. Design and implement a read-only pull_request_target workflow on the default branch, NUL-safe changed-path classification, and required-status rulesets for dev and main, while ensuring the workflow never executes or checks out untrusted pull-request code.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A reviewed design documents why the pull_request gate is insufficient and defines the default-branch trust root, two-PR rollout, ruleset bootstrap, and rollback behavior.
- [ ] #2 The gate workflow runs from the default branch with contents: read, never checks out or executes pull-request code, and evaluates immutable pull-request identity and changed paths.
- [ ] #3 Changed paths are transported and parsed NUL-safely with adversarial filename coverage and exact protected/governance/API boundaries.
- [ ] #4 The bootstrap workflow lands on main before the dev licensing cutoff and the gate status becomes required on both dev and main without weakening existing rules.
- [ ] #5 Owner-authored cutoff changes pass while external protected or governance changes fail, and all workflow, security, and focused tests pass.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Review of Task 4 in TASK-12976 proved that a pull_request workflow is PR-controlled and that line-oriented changed paths are not exact. Live GitHub inspection found dev unprotected, ruleset 5653432 active only on the default branch without required checks, and rmusser01 as the only direct collaborator. The approved replacement uses a default-branch pull_request_target workflow, NUL-safe metadata-only diffing, a distinct trusted commit-status context, and staged ruleset activation. No implementation or external ruleset mutation proceeds until the written spec is reviewed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

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
