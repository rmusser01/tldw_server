---
id: TASK-15693
title: Keep dev PR CI to basic sanity checks
status: In Progress
labels:
- ci
- release
dependencies:
- TASK-15572
references:
- Docs/ADR/106-dev-pr-sanity-only.md
- Docs/Development/CI_REQUIRED_GATES.md
- https://github.com/rmusser01/tldw_server/pull/3540
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Limit PRs targeting dev to cheap syntax and static checks and retain full CI, browser, coverage, and image validation for the existing main, release, scheduled, and manual routes. Reduce the cost of strict current-base reruns.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 No dev PR builds a Docker image or starts browser, coverage, or broad test matrices.
- [ ] #2 Required dev contexts and trusted license admission keep reporting while backend, frontend and security validation use bounded static checks.
- [ ] #3 Main, release, nightly and manual validation retain their original workloads and triggers.
- [ ] #4 Coverage includes dev code and docs routing, failure handling, and the impact of strict current-base branch protection.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR required: yes. Follow up on ADR-096 with a record of the dev-only sanity policy. ADR-063 still governs current-base queueing. Related: TASK-13462 and PR #3540.
Kept the existing required context names while skipping Docker, coverage and browser workloads on dev PRs. Backend and frontend sanity check the test merge commit. The separate trusted license status and static security checks still gate dev. Strict GitHub rules force a fresh sanity result whenever the queue rebases; scope and tradeoffs are recorded in ADR-106.
Local result: 507 targeted workflow contracts passed, plus actionlint, the dev backend sanity commands, and Backlog ID validation. Bandit found zero new issues in touched tests. Hosted status checks and merge are pending.
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
