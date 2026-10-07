# UAT354 — Disconnect completion investigation

Task: TASK-13260.277.4. PR: https://github.com/rmusser01/tldw_server/pull/2979.

The frozen run cleared the manual API key and gated other private tabs but left the Disconnect spinner visible for minutes. Trace the actual credential service and mounted Settings controls before choosing a repair. Further native UAT is paused by the requester; deterministic causal tests are allowed.

## Stage 1: Reproduce at the real service/form boundary
**Goal**: Identify whether credential clearing, configuration reload, or control lifecycle retains the spinner.
**Success Criteria**: A causal regression or explicit bounded non-reproduction, with the old observation preserved.
**Tests**: Real TldwAuth/TldwApiClient and WebUI storage with actual Ant Design Settings controls; device/session manual keys and cookie logout, pending/failure controls.
**Status**: Complete

## Stage 2: Repair a demonstrated cause
**Goal**: Apply the smallest shared fix if a cause is demonstrated.
**Success Criteria**: Original failing case passes, credential isolation remains intact; no speculative timeout or forced-success UI.
**Tests**: Focused service/form regressions and neighboring authentication tests; scoped lint/types.
**Status**: Complete

## Stage 3: Record reviewable disposition
**Goal**: Commit engineering findings and update tracker/task/PR without claiming unrun native acceptance.
**Success Criteria**: Cause, evidence, limitations and pending native acceptance are explicit.
**Tests**: Diff review and relevant verification; Bandit applicability stated.
**Status**: Complete

## Bounded diagnostic result — 2026-09-22

60 checks pass across5 suites; final new integration4/4, lint0findings, types354existing/0added. Real service and mounted form do not reproduce the original stall. The historical icon-only observation cannot distinguish pending logout from stale animation. Keep the finding open and do not implement an unsupported timeout or mark native acceptance complete. Stage2 awaits a causal failure. No Python production changes; Bandit is not applicable to this TypeScript-only diagnostic. Private harness errors and final logs are under /tmp/uat354-*.


## Causal motion repair — 2026-09-22

The original pg-single-354 snapshot already contains "Logged out successfully" while the loading icon remains. The frozen Settings handler emits that toast after both logout and config reload, then clears loading in finally. A pending auth operation therefore does not explain that snapshot.

Ant 6.2.1 retains its default loading icon during CSSMotion leave until a transition-end event; no fallback deadline is supplied. Motion-enabled integration tests using the real auth/storage services reproduce this: the missing-event control fails while the delivered-event control succeeds (1 failed, 5 passed). Disconnect now reuses the existing Common/Button, as the other logout actions already do; its icon follows loading directly and retains native disabled/aria-busy behavior.

Verification: 6 focused and 79 neighboring tests pass; scoped ESLint has 0 findings; shared UI TypeScript has the same 354 baseline diagnostics, 0 added/removed. Independent review finds no material issue. Bandit is inapplicable to the TypeScript-only change. Exact historical event suppression is unobserved; native AC2 stays pending during the requested PR-first pause. Evidence /tmp/uat354-motion-{red-control,green,neighbors}.log and /tmp/uat354-motion-root-cause.md. No global motion or auth policy changes.
