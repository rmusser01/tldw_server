---
id: TASK-13264
title: >-
  Prepare 0.1.44 release with all changes since v0.1.43 and land prior closure
  records
status: In Progress
assignee: []
created_date: '2026-09-27 15:16'
updated_date: '2026-09-27 16:50'
labels: []
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/3027'
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
User authorized both remaining workstream items and preparation of a new release. Freeze dev at 9668e1454b0b28b7a4de13e1a35496fa0b368c42; carry local closure commit 62a2b70e; include PR2978 and all merged changes since v0.1.43. Prepare a reviewable release candidate without publishing until final release approval.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Prior release closure records included in the candidate
- [x] #2 Exhaustive change inventory from v0.1.43 to frozen dev
- [x] #3 0.1.44 metadata, notes and protected-source licensing records consistent
- [x] #4 Required focused checks and packaging verification pass
- [x] #5 Release PR and remaining publication approval gates documented
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-09-27-release-0.1.44-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Frozen range verified: 258 commits and 34 first-parent merged PRs since v0.1.43, including synchronization PR2971. Closure commit62a2b70e carried as6414a4a830. Candidate proposes release date2026-09-27 and Countdown2028-09-27T12:00:00Z; final legal/publication approval pending.

Protected source: `3d6cbf2757a320b3286a1d14e1e755c288e70ab3`.
Protected manifest SHA-256: `1739e25a6dd4bdcfc5d7385366e4b0d9c620f99bf4ae95d3b049cfba37d8a58e`.
Candidate contracts: 89 passed (four warnings), including strict MkDocs and protected-source checkout equality. CI-matching OpenAPI fingerprint passes. Local 0.1.44 wheel/sdist pass Twine and backend-only contents checks; final README metadata refresh is being rebuilt. Ruff, compilation and whitespace checks pass. Bandit main.py has zero findings/errors; licensing tests retain 82 baseline B101 assertion findings, unchanged severity/type counts. Main publication, new PR-specific summary/waiver and proposed legal-date approval remain pending. Full UAT remains separately tracked.
Final corrected README/docs/licensing matrix: 33 passed, four warnings; strict MkDocs and protected source equality pass. Final rebuilt 0.1.44 wheel and sdist pass Twine and backend-only contents verification. Independent release preparation review requested. Candidate remains unpublished.
Draft release PR3027: https://github.com/rmusser01/tldw_server/pull/3027 . Candidate commit 10ca6d2570. Independent reviewer found no preparation blockers: exact 258-commit/34-PR inventory, 7,370 protected-file hashes/path set and source equality, prior grants immutable, closure records intact, PR2978 tree equal approved head, versions/docs aligned, migration prerequisites/native-fork limits explicit. This review covers release preparation and closure, not a new full audit of all merged features. Required remote CI remains pending. Final merge/publication and legal dates need approval; this PR requires a human-owned summary or explicit waiver. Closure records land in main with this release and in dev with subsequent approved synchronization.

MCP portability failure reproduced locally and in installed CI artifacts. Existing TASK-13264 Stage 3 covers the fix. Broader verification also exposed two stale license-first workflow contract expectations from the same merged ordering change; update them to enforce the new wait gate and absent redundant workflow_run triggers. Root-venv exec-worker checks cannot import the uninstalled standalone package in isolated mode, so verify the full protocols using clean installed artifacts.
Release CI blocker reproduced: MCP portable installed wheel/sdist suites fail only test_rc_workflow_runs_installed_stdio_contracts_on_linux_and_windows. It asserts needs == admission, but merged license-first ordering now correctly requires [admission, await_license]. Linux wheel/sdist evidence each: 380 passed, one stale contract failed; Windows each: 374 passed, six platform skips, one same stale contract failed. Local selected test reproduces the exact assertion. Fix plan: require both dependencies and the reusable licensing wait gate, retain platform matrix and protocol assertions, rerun protocol/workflow and installed-artifact checks, compare scoped Bandit baseline, then push to PR3027. No production code or protected frontend changes are needed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

CI repair validated: generic licensing contracts plus MCP workflow contract, 15 passed. Clean portable-gate wheel and sdist each 381 protocol tests passed; both official SDK stdio smokes passed. Root shared environment has a legacy installed MCP package lacking protocol_validation in isolated mode, so direct root exec-worker failures are environmental; clean installed protocol suites cover those tests successfully. Generic contract now enforces both reusable gates, absent duplicate triggers, original dependencies/immutable checkouts, and the backend negative-verdict first-step exit before checkout. Bandit finding type/severity/confidence baselines unchanged: stdio 132/132, generic 168/168, no errors. Ruff and whitespace checks pass. Protected source unchanged.

Reviewed candidate repairs committed at3d6cbf2757a320b3286a1d14e1e755c288e70ab3. Protected-source manifest refreshed over7370 files. Rebuilt wheel/sdist pass Twine and backend-only boundary checks;114 frontend controller/selector/store/macro tests and16 VN tests pass, with2 Chromium account qualifications. Final remote CI remains pending.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:END -->

<!-- SECTION:FINAL_SUMMARY:END -->

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
