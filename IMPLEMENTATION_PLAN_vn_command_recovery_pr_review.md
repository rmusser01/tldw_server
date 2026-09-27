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
**Status**: Complete

Published 581979a4a87543aedd73ac0f11667c2dc253e73b onto dev
35d6dd90d4c3b703a753efdbd926e30af4f9eac5 with an explicit lease protecting
5de2ed11671f593968a86aef24d3be0422488f75. All five prior patches unchanged;
97 tests, typecheck, scoped lint and diff checks pass. Bandit baseline has zero
findings/errors; it does not scan touched TypeScript.

## Stage 2: Current-Head Review

**Goal**: Address all actionable findings and obtain complete exact-head review.
**Success Criteria**: Explain Qodo's human-gate finding inline, distinguish design
approval from merge authorization, preserve requester summary; validate any new
scoped findings with regression tests before fixing. No backend scope expansion.
**Tests**: Exact base/head coverage in completed Qodo and CodeRabbit full reviews,
all review comments/threads checked, scoped regressions and relevant suites.
**Status**: In Progress

Qodo completed the exact-head full-diff reassessment at issuecomment-5858115068
with no production defects and accepted the human-gate clarification. Its minor
tracking hygiene feedback was addressed in 804c0745c0. CodeRabbit completed the
full review of 804c0745c0 at 17:46:08 UTC with three scoped findings:
failed discard hides unreadable controls, a retired-plan link, and heading spacing.
The failed-discard workbench regression failed before the minimal hook fix because
the warned confirmation checkbox disappeared. With the fix, the journal and
generation lock survive removal denial, confirmation resets, and explicit retry
succeeds without generation or cancellation after storage access is restored.
The retired documentation link was removed through official Backlog mutation;
heading spacing is corrected here. All 98 VN/frontend fetch-client tests pass
(8.97s), typecheck and scoped ESLint pass, and unchanged Python VN Bandit has zero
findings/errors (not a TypeScript scan). Any changed head still requires complete
reassessment; no backend scope expansion is authorized.

Qodo reassessment of 70953ae10c (issuecomment-5858353005) identified an empty-string
journal being mistaken for absence. Storage and rendered workbench regressions
failed before the one-line null-only absence check: storage did not throw and
Start was enabled (2 failed, 10 passed, 2.15s). The initial UI wait timed out;
the corrected direct disabled-state assertion supplies genuine regression proof.
Malformed data stays intact until warned explicit discard; a missing key remains
valid. All 101 VN/frontend fetch-client tests pass (13.11s), typecheck, scoped
ESLint and diff checks pass. Complete new-head hosted review remains required.

## Stage 3: Gated Merge

**Goal**: Merge normally only after current-head review and live dev gates pass.
**Success Criteria**: Fresh head/base/rules/summary check; backend-required,
security-required, coverage-required, frontend-required, e2e-required,
container-build-check and frontend-license-policy/trusted/dev all pass on exact
head. Full match-head normal merge, verified merge commit, task finalized and only
this completed plan removed. Preserve checkout and chat; stop own follow-up.
**Tests**: Live GitHub rules/checks, merge API verification and tracked diff check.
**Status**: In Progress

Exact 804c0745c0 E2E failed before tests in unchanged AuthNZ bootstrap. Isolated
30.19.0/30.20.0 SQLGlot comparison reproduces rejection of identical canonical SQL
because standalone AUTOINCREMENT rendering changed. A separate backend-fix
decision is pending; shared environments and dependency policy are unchanged.
No merge attempted; all live current-head gates remain prerequisites.
