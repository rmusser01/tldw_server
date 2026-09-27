# VN Command Recovery PR Review

Task: TASK-13385. PR: https://github.com/rmusser01/tldw_server/pull/3028.
Requester authorized protected rebases, scoped review fixes and a gated normal
merge. Preserve the human-written Change summary verbatim and unrelated work.

## Stage 1: Protected Rebase
**Goal**: Rebase the owned PR branch onto latest dev without changing prior patches.
**Success Criteria**: Clean tracked checkout and matching remote ownership before
rebase; conflict-free rebase and all five patches unchanged in range-diff; explicit
expected-head lease on publication. Exclude local preview link and archive13379.
**Tests**: Range-diff, unrelated base-file equality, 97 VN/fetch-client tests,
frontend typecheck, scoped lint, diff checks; unchanged VN Python Bandit baseline.
**Status**: In Progress

## Stage 2: Current-Head Review
**Goal**: Address all actionable findings and obtain complete exact-head review.
**Success Criteria**: Explain Qodo's human-gate finding inline, distinguish design
approval from merge authorization, preserve requester summary; validate any new
scoped findings with regression tests before fixing. No backend scope expansion.
**Tests**: Exact base/head coverage in completed Qodo and CodeRabbit full reviews,
all review comments/threads checked, scoped regressions and relevant suites.
**Status**: Not Started

## Stage 3: Gated Merge
**Goal**: Merge normally only after current-head review and live dev gates pass.
**Success Criteria**: Fresh head/base/rules/summary check; backend-required,
security-required, coverage-required, frontend-required, e2e-required,
container-build-check and frontend-license-policy/trusted/dev all pass on exact
head. Full match-head normal merge, verified merge commit, task finalized and only
this completed plan removed. Preserve checkout and chat; stop own follow-up.
**Tests**: Live GitHub rules/checks, merge API verification and tracked diff check.
**Status**: Not Started
