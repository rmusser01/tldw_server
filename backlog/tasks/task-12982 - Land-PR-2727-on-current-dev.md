---
id: TASK-12982
title: 'Land PR #2727 on current dev'
status: In Progress
assignee: []
created_date: '2026-07-22 03:45'
updated_date: '2026-07-22 05:45'
labels:
  - integration
  - release
  - licensing
  - ci
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/2727'
  - TASK-12963
documentation:
  - Docs/superpowers/specs/2026-07-21-pr-2727-landing-private-pilot-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Integrate the merged frontend licensing cutoff into PR #2727, revalidate the exact head, satisfy review and human-authorship requirements, and merge the provider credential runtime into dev without disturbing the user-owned dirty worktree.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Current dev is integrated into PR #2727 without losing its reviewed feature commits or the pre-existing user-owned worktree changes.
- [ ] #2 Fresh exact-head required CI and frontend-license trusted checks pass, with reproduced failures fixed rather than bypassed.
- [ ] #3 The requester supplies the required human-written Change summary, PR #2727 is marked ready, and it merges into dev.
- [ ] #4 Landing evidence records integration parents, exact-head gates, reproduced failure dispositions, review results, and the final merge commit.
- [ ] #5 The actual merge commit is verified to contain the validated PR head and current protected dev tip, with merged licensing metadata and trusted-policy files present, before any deployment task begins.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-07-21-pr-2727-current-dev-landing-implementation-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
2026-07-21: During design finalization, a separate owner process committed the previously dirty follow-up as 7d76bdfcc0, merged protected dev 8ed612c7e0 in conflict-free merge 0e8eadc55f (first parent 7d76bdfcc0, second parent 8ed612c7e0), recorded post-merge validation in 6065c64ab4, and pushed that exact PR head. Both original PR head e8bcc4c8b and protected dev are ancestors. Fresh exact-head CI is in progress; no non-green context is waived.

Execution baseline reconfirmed: PR head 6065c64ab4 contains original head e8bcc4c8b and protected dev 8ed612c7e0 through merge 0e8eadc55f (parents 7d76bdfcc0 and 8ed612c7e0). Local work descends from that head. The index is empty; protected concurrent out-of-scope dirty paths are the tracked TASK-12963 and TASK-2234 Backlog records, plus untracked server-ux-smoke.pid and the two named watchlist templates; all remain unstaged and excluded from this task. PR #2727 remains draft and mergeable/unstable at base 8ed612c7e0; frontend-license-policy/trusted/dev passes. Current non-passing checks are actionlint and backend-required failures, a Windows Python 3.12 Full Suite failure, and one canceled Windows research-websearch shard. Known current-head corrections are actionlint SC2155 and OpenAPI fingerprint drift.

2026-07-22: Exact-head 6065c64ab4 CI drained with 782 terminal checks: 774 successful, 3 intentionally skipped, 2 cancelled, and 3 failed. The Windows research-websearch timeout is a confirmed intermittent pre-existing baseline with identical 65-minute failures on unrelated main, sync-main, and Dependabot heads and successful controls in 7:23-8:48; the evidence and investigation direction are recorded in existing TASK-2234. The two deterministic failures are corrected locally: frontend-license-gate now assigns fetched SHAs before marking them readonly, resolving both SC2155 findings while preserving fail-closed command status, and the canonical OpenAPI fingerprint now matches Python 3.12 SHA 9a07fa34479c3fd6fcff06026295123117fee8d40dacb7c1537ecc21dbf7a4b1. The fingerprint drift is solely incoming dev commit 99fdd189c5 OpenAPI licensing/contact metadata; path and schema counts remain 1,999 and 2,909, so frontend type regeneration is not warranted. Verification passed: Python 3.12 canonical drift check; 54 CI workflow/license-gate contract tests, including the refreshed trusted-script hash and SC2155 regression; 75 Python 3.12 OpenAPI-contract tests; Ruff and py_compile on the touched Python test; workflow YAML parse; extracted Bash syntax check; git diff --check; and two independent read-only reviews with no actionable findings. A fresh origin/dev fetch still resolves to 8ed612c7e0, which is already an ancestor of local HEAD, so no additional merge is required before the next push. No production Python changed. Bandit on the touched test reproduced one unchanged B105 false positive for the literal GitHub token expression; with the existing test-only B101 and that unchanged B105 baseline excluded, it reported zero findings and zero errors. PR #2727 remains draft pending fresh exact-head CI and the requester-authored Change summary. The unrelated PID and both watchlist templates remain untouched and unstaged.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
