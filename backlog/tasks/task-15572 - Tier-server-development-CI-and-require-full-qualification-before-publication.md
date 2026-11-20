---
id: TASK-15572
title: Tier server development CI and require full qualification before publication
status: In Progress
labels:
- ci
- release
references:
- https://github.com/rmusser01/tldw_server/pull/3508
- https://github.com/rmusser01/tldw_server/pull/3540
- Docs/ADR/096-tiered-development-and-release-ci.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Use focused CI for every dev PR and require full cross-platform CI before merging into main or publishing server artifacts. Retain nightly broad Linux regression on dev.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Nightly runs the broad Linux regression suite at the dispatched dev SHA, deduplicates successful or active runs for that SHA and retries failed attempts.
- [ ] #2 Main promotion and server package and image publication depend on full cross-platform and package checks on their own event SHA, with missing and skipped results rejected.
- [x] #3 Targeted routing and publication contracts pass, and rollout, live qualification, protection and rollback steps are documented.
- [ ] #4 Backend changes on dev PRs use the bounded smoke job by default for same-repository and fork PRs. No broad regression matrix runs on dev PRs. Existing path-scoped required gates and license admission remain enforced.
- [ ] #5 Main protection requires the observed Main PR release qualification GitHub Actions check at the current base, alongside the existing license check.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR required: yes. ADR path: Docs/ADR/096-tiered-development-and-release-ci.md. The workflow changes the durable PR, nightly, and release gating policy and preserves ADR-063 merge queue and existing license admission. Implementation order: record ADR and rollout, prove routing and verdict contracts, reuse the existing matrices and package checks for release callers, then verify focused tests and workflow lint. Keep opt-in off until scheduled and release qualification are proven.
Local implementation on codex/server-tiered-ci. All 350 focused workflow and Backlog contracts, 25 backend smoke tests and six mocked dispatcher cases passed. Actionlint and targeted Ruff passed. Bandit reported no issues in new Python files. The wheel and sdist passed make pypi-check, and the wheel imported in a clean local install without runtime dependencies. All six baseline full matrix definitions, runner types, timeouts and test steps remain intact, with main PR jobs selecting the merge SHA. Pending GitHub qualification includes a normal associated PR, the scheduled nightly at its dev SHA, a non-publishing release validation with dependency installation, and verification of strict branch protection before enabling light CI.
Maintainer correction: the target branch sets the policy for all PRs. ADR required: yes. ADR path: Docs/ADR/096-tiered-development-and-release-ci.md. Revise the existing proposed ADR to keep dev fast by default and require full CI on main, preserving ADR-063 and the existing dev gates.
The dev queue keeps its seven protected workflows and the trusted license check, including the backend-required unit and startup smoke on rebased heads. Supplemental CI uses smoke or a backend no-op on normal dev PRs. Nightly and release select the full Linux matrix; main PRs also select the macOS/Windows matrices and package check. The Linux qualification job provisions PortAudio before installing the wheel and runtime dependencies.
Branch-policy correction verified locally: 691 dev and 683 main workflow contracts passed, with one runner-only test skipped in each. All 25 backend smoke tests passed. Actionlint, Ruff and shard coverage passed on both candidates, and Bandit found no new findings in the touched Python tests. Hosted merge and branch protection verification are pending.
Hosted main qualification found two test prerequisites. The macOS Python 3.13 tooling shard now reuses the Python 3.12 Compose setup. The Windows upgrade contract now locates Bash and launches its isolated Python fixtures through the shell. The original hosted failures were job 136523705785 and job 136523706411. Local targeted suites passed 221 and 131 tests. Ruff, Actionlint, and Bandit passed. Hosted requalification is pending.
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
