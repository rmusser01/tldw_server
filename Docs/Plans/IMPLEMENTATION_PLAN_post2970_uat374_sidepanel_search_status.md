# UAT374: truthful sidepanel search status

Task: TASK13260.277.22. Root approved this bounded design on 2026-09-20.

Change only Sidebar presentation/local search state and its tests. Keep the reviewed UAT366 ownership guards and the existing server hook, cache keys, and scoped transport. Distinguish current-query loading, source failure/unavailability, successful empty, and matches. Preserve valid partial matches with a concise failure notice. Retry preserves input and current owner; blocked for revoked owner, unresolved debounce, or an unavailable server connection. Bind local results to owner and query so pending searches cannot show old-query matches.

## Stage 1: Causal real QueryClient tests
**Goal**: Reproduce false empty and stale results through the real hook.
**Success Criteria**: Current Sidebar fails loading/error/retry/query-fencing regressions; existing ownership controls still pass.
**Tests**: Held local/server reads, HTTP failure, success empty/match, partial local matches with server error, Retry, query changes, owner changes, offline/unverified guards, same-owner cache.
**Status**: Complete

## Stage 2: Narrow Sidebar repair
**Goal**: Present source status truthfully and retry within existing scope.
**Success Criteria**: Empty appears only after both current searches succeed; failed/old-query server cache stays hidden; partial valid results remain; current owner checks guard all retries and completions.
**Tests**: Stage 1 causal tests plus coordinator and server-history ownership/transport suites.
**Status**: Complete

## Stage 3: Checks and frozen review handoff
**Goal**: Deliver an exact UAT374 diff distinct from the existing UAT366 changes.
**Success Criteria**: Focused/regression tests pass, matched lint/types add no diagnostics, source freeze and independent review handoff recorded before native acceptance.
**Tests**: Vitest, scoped ESLint, matched TypeScript checks, whitespace/self-review. Bandit is inapplicable to TS-only production/test edits.
**Status**: In Progress

## Scope and evidence

- Pre374 Sidebar and ownership test captured at `/private/tmp/uat374/` before implementation.
- No server hook, backend, runtime, frozen candidate, git, or global tracker changes.
- Root owns independent review and native acceptance; task stays In Progress until those gates.
- Causal first run: 12 failed / 3 existing ownership controls passed (`/private/tmp/uat374/red.log`); narrow repair passed 18 tests including coordinator coverage. Additional real disconnection/cache case failed 1 / passed 18 before its display guard (`/private/tmp/uat374/offline-red.log`).
- Final related run: 7 suites / 139 tests passed (`/private/tmp/uat374/final-tests.log`), including 19 Sidebar real-hook/QueryClient tests, route ownership, resume, account privacy, server history and scoped transport.
- Exact UAT374-only source/test diffs: `/private/tmp/uat374/sidebar-isolated.diff` and `/private/tmp/uat374/tests-isolated.diff`. These compare against the pre374 UAT366 candidate, not HEAD.
- Scoped ESLint: 0 errors / 0 warnings before and after (`/private/tmp/uat374/lint-comparison.json`). Matched frontend TypeScript: 93 baseline / 93 candidate diagnostics, zero added, removed or touched diagnostics (`/private/tmp/uat374/type-comparison.json`).
- Bandit is not applicable: the only production/test edits are TSX; no Python changed. Whitespace check passed. Source frozen for root review; native status acceptance remains outstanding.
