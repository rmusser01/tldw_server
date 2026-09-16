# UAT159 first-run request auth gating — author report

Task: TASK-12918 (reopened). Agent: retry031_repair. Approved scope: three production files, three existing test files. Ready for independent review; native acceptance remains root-owned. No runtime/browser, backend/auth primitive, tracker, staging or commit changes.

## Cause and fix

_app correctly distinguishes authResolved from isAuthenticated. A blank-key single-user configuration resolves with isAuthenticated=false, but still renders Home and FirstRunGate. Home's existing bypass hides the overlay only: FirstRunGate called useFirstRunCheck before its conditional render, and that hook always dispatched GET /api/v1/persona/profiles.

The existing authenticated shell decision now participates in FirstRunGate.bypass. The gate passes enabled:!bypass to useFirstRunCheck. A disabled hook sends no request and resets setup/resume/loading state. Adding enabled to the effect dependencies invokes the existing cleanup when auth or bypass changes, so a prior request cannot publish a late response. Re-enabling checks again. Final self-review exposed apiSend GET coalescing across owners: a new enabled invocation could join the old pending promise. The hook now uses the existing per-call coalesce:false option for this profile GET. Existing effect cleanup still rejects the old invocation's response; the new invocation obtains a fresh request. No apiSend implementation change. The hook keeps enabled=true by default for compatibility with its other caller.

No raw API-key gate or duplicate authentication logic was added. Existing _app rules continue to recognize quickstart cookie sessions and validate hosted multi-user cookie sessions without persisted access tokens. Login, settings and setup retain their existing routing branches. Home, Research Workspace, admin and character-onboarding bypasses now avoid their unused profile probe.

The retained native receipt is output/playwright/cycle5-repair-verification-2026-09-16/native-image118-157/persona-preauth-response.json (429, character_chat.default). Root/source013 separately traced anonymous identity-scope denial and authenticated200; this change leaves that fail-closed backend policy intact. No native acceptance is claimed by these synthetic tests.

## Owned files

Production:

- apps/tldw-frontend/pages/_app.tsx
- apps/packages/ui/src/components/PersonaGarden/FirstRunGate.tsx
- apps/packages/ui/src/hooks/useFirstRunCheck.ts

Tests:

- apps/tldw-frontend/__tests__/app/app-layout.test.tsx
- apps/packages/ui/src/hooks/__tests__/useFirstRunCheck.test.tsx
- apps/packages/ui/src/components/PersonaGarden/__tests__/FirstRunGate.test.tsx

## Evidence and coverage

- red.log: **7 failures / 69 passing controls** before production changes. Actual App→FirstRunGate→useFirstRunCheck tests expose blank-key Home/media requests, authenticated bypassed Home/Research requests, and missing re-check after logout/re-entry. Two hook tests expose dispatch while disabled and missing disabled-state reset.
- red-coalescing.log: real App→Gate→Hook→apiSend→mock request-core reproduces the cross-enable request reuse (1 failure / 13 passing app controls). This strengthened test replaces the earlier apiSend mock; no production API helper was changed.
- final-tests.log: **110 tests / 5 suites PASS**, no skips, at 2026-09-16 13:38 local. App70, hook8, gate7, unchanged composer notices12, unchanged apiSend13. Existing app tests retain their layout mocks, while the new bounded group opts into the real Gate/Hook/apiSend and mocks only the request-core transport at that boundary. Positive auth controls cover manual/runtime keys, quickstart single-user cookies, hosted multi-user cookies with no access token, rejected hosted auth, and deferred bootstrap. Setup/login/settings remain accessible without profile reads. Late old responses after logout/re-entry do not replace the new result; disabling clears an in-progress resume state. Existing error/dismiss/optional-personalization controls remain. Stable authenticated rerenders and auth refreshes retain exactly one request; enabled remount gets one new request and disabled remount gets none. apiSend's normal coalescing and its explicit opt-out controls pass unchanged.
- green.log records the first passing new behaviors plus five older fixture expectation failures: those tests expected a gate-active route without supplying authenticated credentials. Their fixtures now explicitly provide an env key; original route/callback assertions are preserved. This is a fixture alignment, not five additional product defects.
- final-eslint.log/eslint-baseline.json/eslint-summary.json: **0 errors / 0 warnings** across all six paths in both HEAD and current.
- final-diff-check.log: scoped git diff --check exits0.
- typecheck.log/typecheck-comparison.json: full compiler exit2 with **90 baseline / 90 current diagnostics, none added or removed**. This is baseline equivalence, not a clean compiler result.
- bandit.json/log: invoked with the project virtualenv over three production TS/TSX files; 0 findings and **3 Python AST parse errors**. Bandit did not analyze TypeScript, so this is not TypeScript security assurance.

## Exact commands

From apps/tldw-frontend, using its normal Vitest configuration (no private alias required):

```sh
bunx vitest run __tests__/app/app-layout.test.tsx ../packages/ui/src/hooks/__tests__/useFirstRunCheck.test.tsx ../packages/ui/src/components/PersonaGarden/__tests__/FirstRunGate.test.tsx ../packages/ui/src/components/Option/Playground/__tests__/PlaygroundComposerNotices.first-run.test.tsx ../packages/ui/src/services/__tests__/api-send.test.ts > ../../.tmp/uat159-first-run-auth-20260916/final-tests.log 2>&1
bun run typecheck > ../../.tmp/uat159-first-run-auth-20260916/typecheck.log 2>&1
```

From repo root:

```sh
source .venv/bin/activate
python .tmp/uat159-first-run-auth-20260916/verify-static.py
```

verify-static.py invokes the repository ESLint binary/config for owned-paths.json; compares HEAD text via --stdin against current text; runs scoped git diff --check; invokes python -m bandit over exactly the three production paths. Compiler comparison strips only diagnostic line/column positions and compares diagnostic multisets with .tmp/fresh-uat-recovery-20260916/typecheck-current.log.

## Caller inventory and limits

1. FirstRunGate is used only by _app. This is the observed Home request path and is now explicitly gated by existing auth and bypass decisions.
2. ChatFirstRunNudge in PlaygroundComposerNotices independently calls useFirstRunCheck() with default enabled=true. Home does not mount that composer. Its eligibility decision is unchanged, and its existing12tests pass. The shared hook's read also opts out of coalescing for this caller. We do not claim universal Chat/extension preauth suppression. Root explicitly limited this change until a distinct no-key composer failure is proven; connectivity readiness alone must not be mistaken for authentication because offline bypass can mark the connection ready.

Request-count tradeoff: two simultaneously mounted consumers now make two independent profile reads rather than sharing one pending GET; an enabled remount also makes a fresh read. The current hook has no persistent success cache. There is no timer/retry loop, and unchanged rerenders/auth refreshes do not refetch. This scoped opt-out prevents old-response reuse across a disable/re-enable transition without modifying the general coalescer.

No generic request cache, auth state provider, backoff or backend policy change is part of this repair. Root independent review and fresh native blank-key/post-auth checks remain required. owned-manifest.json and review-snapshot/ retain exact source/test/evidence hashes and copies at handoff.
