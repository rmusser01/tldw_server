# VN Command Recovery PR Review

Task: TASK-13385. PR: https://github.com/rmusser01/tldw_server/pull/3028.
Requester authorized protected rebases, scoped review fixes and a gated normal
merge. Preserve the human-written Change summary verbatim and unrelated work.

## Stage 1: Protected Rebase

**Goal**: Rebase the owned PR branch onto latest dev without changing prior patches.
**Success Criteria**: Clean tracked checkout and matching remote ownership before
rebase; conflict-free rebase and all prior patches unchanged in range-diff; explicit
expected-head lease on publication. Exclude local preview link and archive13379.
**Tests**: Range-diff, unrelated base-file equality, 116 VN/fetch/shared-auth tests,
frontend typecheck, scoped lint, diff checks; unchanged VN Python Bandit baseline.
**Status**: Complete

Published 581979a4a87543aedd73ac0f11667c2dc253e73b onto dev
35d6dd90d4c3b703a753efdbd926e30af4f9eac5 with an explicit lease protecting
5de2ed11671f593968a86aef24d3be0422488f75. All five prior patches unchanged;
97 tests, typecheck, scoped lint and diff checks pass. Bandit baseline has zero
findings/errors; it does not scan touched TypeScript.

Dev advanced to `a3d52f30b0d21b8528d426d16e06c4a013414807` through the
independently merged AuthNZ compatibility fix in PR #3030. The latest authorized
rebase from clean owned `98be158e87903b27afa4a98c2552bc08729e176d` is conflict-free;
all ten prior patches are unchanged in range-diff. Fresh 116 VN/fetch/shared-auth
tests pass (8.79s), frontend typecheck and scoped lint pass. The read-only AuthNZ
guard suite passed 123 tests (375.07s, exit 0) using isolated SQLGlot 30.20.0, including
the actual startup DDL and fail-closed controls. Slow session cleanup was sampled
in Python garbage collection, not a network wait. The process exited normally
before a bounded stop attempt reached it; no process was terminated. Direct
startup-DDL acceptance and off-id AUTOINCREMENT rejection checks also exited 0.
The unchanged VN Python Bandit
baseline has zero findings/errors over 9064 lines and does not scan TypeScript.
Publication uses an explicit lease on the full original owned head.

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

Qodo's complete deep review of 5953de6951 reported zero bugs and the shared-auth
architecture finding in discussion 4116389900. The hook's profile-first policy is
moved into shared `services/tldw/verified-principal.ts`, using the existing caller
transport. No cached identity, new auth state, or change to `getCurrentUser()` is
introduced. Seven identity behavior checks passed before extraction; all 116
VN/frontend fetch-client/shared-auth checks pass after extraction (9.38s), as do
typecheck and scoped lint. The shared-file lint command required the installed
ESLint 9.39.2 binary with the existing frontend config and UI working directory;
the cached bunx 10.11.0/config-base setup failures are not source findings.
Bandit on unchanged Python VN baseline remains clear and does not scan TypeScript.
Complete exact-new-head hosted review and all required gates remain prerequisites.

CodeRabbit discussion 4116478050 flagged the final summary's outdated 97-test
checkpoint. Official Backlog mutation updates it to the latest 116 passing tests
and makes new rebased-head reviews explicitly pending. Historical test/review
evidence is retained, including Qodo's completed 98be158e87 reassessment.

## Stage 3: Gated Merge

**Goal**: Merge normally only after current-head review and live dev gates pass.
**Success Criteria**: Fresh head/base/rules/summary check; backend-required,
security-required, coverage-required, frontend-required, e2e-required,
container-build-check and frontend-license-policy/trusted/dev all pass on exact
head. Full match-head normal merge, verified merge commit, task finalized and only
this completed plan removed. Preserve checkout and chat; stop own follow-up.
**Tests**: Live GitHub rules/checks, merge API verification and tracked diff check.
**Status**: In Progress

Exact 804c0745c0, 5953de6951 and 98be158e87 E2E failed before tests in unchanged
AuthNZ bootstrap. Isolated
30.19.0/30.20.0 SQLGlot comparison reproduces rejection of identical canonical SQL
because standalone AUTOINCREMENT rendering changed. PR #3030 independently
integrated the backend fix into dev with all its required gates passed. The
frontend PR is rebased onto that external integration; no separate-fix decision
is still needed. Shared environments and dependency policy are unchanged.
No merge attempted; all live current-head gates remain prerequisites.
