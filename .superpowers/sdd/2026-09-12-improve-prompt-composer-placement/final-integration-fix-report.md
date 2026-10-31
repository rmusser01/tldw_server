# Final integration fix wave — TASK-12984.2 / TASK-12984.3

## Scope

This single bounded wave addresses the four Important findings in `final-integration-review.md`. No dependency, schema, route, visual-placement, or server-idempotency scope was added.

## Corrections

1. Direct recipe mutations now use the existing token-bound provisional delivery receipt already used by Overlay extension dispatch. Both direct and extension receipts remain guarded through transport plus local success/error settlement and are acknowledged only in `prompt-sync`'s final settlement. A same-owner Pull cannot consume the running operation's guard and admit a second PUT.
2. Starting either system-prompt improvement action clears the older recipe Undo immediately. Success, pending cancel, and failure retain the compiled recipe draft as the current value; only a successful newer improvement creates its own one-step Undo.
3. The capabilities route again authenticates before running `rbac_rate_limit("prompts.capabilities")`, while FastAPI dependency caching preserves the same principal used for authorization metadata.
4. Maintained shell-wiring tests now assert the approved topology: one WebUI action immediately before external Send, one shared extension action supplied to legacy/v1/v3/v5 send clusters, and no toolbar/Quick Chat ownership.

## RED evidence

- Real direct/extension active-mutation interleavings initially failed **3/4**, demonstrating a second PUT while the first operation was held in transport or local settlement. Removing awaited error settlement then failed both direct and extension compensation cases.
- Eighteen real system-modal recipe-to-Improve/Review cases failed on stale Undo across undefined, empty, and custom overrides and success, pending-cancel, and failure outcomes.
- The existing capability catalog limiter test returned 200 rather than the required 429.
- The maintained shell-wiring test failed **3/7** because it required the removed toolbar-owned topology.

## GREEN evidence

- Persistence/transport/owner/builder/caller matrix: **24 files / 1,728 tests passed**, randomized seed 12984. The six new direct/extension transport/local-success/local-error interleavings passed.
- Root aggregate focused UI/persistence gate: **5 files / 195 tests passed**.
- Cumulative composer/recipe/system matrix: **10 files / 360 tests passed**.
- Track A state/diff/localization plus refreshed shell wiring: **7 files / 69 tests passed**.
- Capability/auth gate: **114 tests passed**; the catalog rate limiter returns 429 again.
- Extension TypeScript compile passed. Scoped ESLint reported no errors; maintained-format files passed their scoped formatter checks; two legacy files retain verified baseline-only formatting differences. `git diff --check` passed.
- Bandit on the touched Python production endpoint: **0 findings, 0 errors** (`/tmp/bandit_task12984_final_fix.json`). The remaining TypeScript changes are outside Bandit's scope.

## Preserved contracts and known limits

Exact-owner recovery, all-owner quarantine/Forget, compensation ordering, background no-replay, restart behavior, v1 behavior, current-draft/active-model request isolation, recipe identity, exact Undo, system template identity, approved compact placement, Quick Chat exclusion, and extension/Web ownership remain intact. Live-backend browser smoke remains environment-gated; full browser suites will be rerun after the requested rebase. The task records remain In Progress pending the one scoped independent re-review.
