---
id: TASK-13112
title: Rebase PR 2808 recipient shared data plane onto latest dev
status: Done
assignee: []
created_date: '2026-08-23 04:10'
updated_date: '2026-08-23 04:34'
labels:
  - workspaces
  - sharing
  - rebase
  - frontend
  - backend
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/2808'
  - TASK-12020.40
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Rebase codex/research-workspace-power-user-uat onto the latest origin/dev, resolve integration conflicts without regressing current dev or the canonical recipient shared-workspace data plane, verify the affected backend/frontend contracts, and update PR #2808.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Branch history is rebased onto the latest fetched origin/dev.
- [x] #2 All conflicts preserve current dev behavior and the canonical recipient shared-workspace contract.
- [x] #3 Affected frontend/backend tests and repository integrity checks pass or documented unrelated baselines remain.
- [x] #4 PR #2808 is updated with force-with-lease and its merge state is verified.
- [x] #5 The unrelated untracked watchlist templates remain untouched and excluded.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-08-23-pr-2808-dev-rebase.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Rebased 45 unique commits onto fetched origin/dev and resolved auth_deps.py, the generated OpenAPI fingerprint, and background-proxy.ts conflicts by composing both sides' contracts. Regenerated the OpenAPI fingerprint to 2014 paths / 2967 schemas (sha256 0a8f59ff75e7...). Updated the refresh-coalescing test fixture for the guarded tldwRefreshRotation storage contract from dev. Verification: background proxy 108/108; request scope 15/15; shared recipient UI/domain 78/78; CDP runner 102/102; backend auth/workspace 120/120; source-only TypeScript pass; targeted ESLint 0 errors (existing no-explicit-any warnings only); OpenAPI drift pass; Bandit 0 findings; git diff --check pass. Package-wide TypeScript retains unrelated existing errors across multiple modules, and Prettier baseline fails on both unchanged dev source and the pre-rebase branch test. The two unrelated untracked watchlist templates remain untouched.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
PR #2808 was rebased onto the latest dev and force-pushed with lease at ed0ef25a20. Conflict resolutions preserve dev's Service Prompt/request-auth safeguards alongside the recipient shared-workspace authorization, request coalescing, and expected-status contracts. The canonical OpenAPI fingerprint was regenerated, and the stale refresh fixture was aligned with guarded rotation-record storage. GitHub reports the branch MERGEABLE; the PR remains draft because the required human-authored Change summary has not yet been supplied. Focused verification passed: frontend 108/108, 15/15, 78/78, and 102/102; backend 120/120; source-only TypeScript; OpenAPI drift; targeted ESLint with zero errors; Bandit with zero findings; and git diff checks. Package-wide TypeScript and Prettier retain documented unrelated/pre-existing baselines. Untracked watchlist templates were not staged or modified.
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
