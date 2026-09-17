# TASK13260.171 / UAT230 — Settings same-origin OpenAPI guard

**Implementation frozen for independent review.** Native Settings acceptance remains parent-owned and pending. Baseline snapshots were captured on `e1ccad4be7cf5b5c1b4c1e3b7405741a3ad19f0d`; final patch/manifest were captured on `5cfac7bbb54da83fb10c9d7eeb3f4f298d3734c2`. The intervening commit retains UAT225 documentation, task and native evidence only and changes none of the four owned source/test paths. There are four owned paths (two production, two tests). Source manifest SHA256 `9a41995dec568803117d3604acf8cbf3a75c287ac42af62dfd1b05ccd9be3671`.

## Cause and smallest repair

Normal authenticated multi-user Settings directly fetched `<serverUrl>/openapi.json` to discover four optional Billing GET routes. Quickstart same-origin Next routing proxies `/health` and `/api/...`, not `/openapi.json`. The shared API client already deliberately skips discovery for precisely that deployment/URL combination, but Settings did not use its guard. Native evidence retained by the parent shows two404 during ordinary Settings/login; it does not show a Notes/graph failure or a retry loop.

Export the existing `isQuickstartWebUiSameOriginServerUrl` predicate and call it in Settings' probe precondition. The predicate reads only deployment/browser URL inputs, not mutable client configuration. The component continues to use its explicit displayed server target. No new abstraction, singleton request, proxy route or configuration change was added.

The effect still clears Billing availability, checks effective login/multi-user mode, owns its five-second timeout/AbortController, rejects late results using the existing cancellation flag, and enables Billing only when all four GET routes are advertised. Direct-backend discovery works in quickstart and advanced modes; same-origin advanced discovery is preserved. Genuine404 leaves Billing hidden without loading Billing endpoints.

## Owned paths

- `apps/packages/ui/src/services/tldw/TldwApiClient.ts`: export the existing predicate only; its body and getOpenAPISpec behavior are unchanged.
- `apps/packages/ui/src/components/Option/Settings/tldw.tsx`: import the predicate and extend the existing precondition only.
- `.../Settings/__tests__/tldw.cookie-logout.test.tsx`: actual mounted Settings regressions using the real URL predicate via partial module passthrough. The Form fixture forwards submit to the real save handler for the stale-target control. Existing assertions remain; direct-backend Billing positive now runs in both deployment modes.
- `.../Settings/__tests__/tldw.form-lifecycle.test.tsx`: passthrough real module exports while preserving its client double. No assertion change.

Baseline copies are retained under `baseline/`; one extra unchanged auth-mode test snapshot is historical context and is not owned by the patch. `owned.patch` and `review-snapshot/` contain exactly the four changed paths.

## Causal evidence and controls

- `causal-red.log`: **2 FAIL /5 PASS /24 not-selected**. Both mounted same-origin cases (with and without trailing slash) observed the real unwanted fetch. Connection testing still succeeded. This is a transport-boundary assertion, not a CSS/string mirror.
- `expanded-red.log`: **2 FAIL /8 PASS /23 not-selected**, adding direct-backend quickstart/advanced and saved-server transition controls before production edits.
- `green.log`: **87 PASS across6 files, zero skipped**,19.47s, on final source. Tests cover same-origin quickstart suppression; current authenticated Connection usability; all-four-route direct-backend discovery; same-origin advanced positive; real404 absence; five-second abort; logout/unmount cleanup; and old-server advertised response ignored after a normal save changes the target. Existing actual-form/auth/timeout/tab/client connection suites remain positive.

Vitest reports `skipped` for nonselected cases when `-t` is used in the causal runs; these were selections, not disabled tests. The final six-file run has no such exclusions. Fetches are controlled responses; no browser, native API, provider or runtime action was performed.

## Verification

`commands.json` contains exact commands. From `apps/tldw-frontend`, the final run uses `node node_modules/vitest/vitest.mjs run --config vitest.config.ts` with:

1. `../packages/ui/src/components/Option/Settings/__tests__/tldw.cookie-logout.test.tsx`
2. `../packages/ui/src/components/Option/Settings/__tests__/tldw.form-lifecycle.test.tsx`
3. `../packages/ui/src/components/Option/Settings/__tests__/tldw.timeouts.form.test.tsx`
4. `../packages/ui/src/components/Option/Settings/__tests__/tldw-auth-mode.form.test.tsx`
5. `../packages/ui/src/components/Option/Settings/__tests__/tldw-settings-tabs.test.tsx`
6. `../packages/ui/src/services/__tests__/tldw-api-client.connection-sync.test.ts`

- Original ESLint receipt was invalid for source validation: all four files were ignored as outside the config base path. Its four warnings did not represent parsed source. The ignored-scope results and original helper are preserved with `-ignored-scope-original` names. Corrected independent ESLint parses all four actual logical paths using the existing frontend config, repository cwd and explicit Next rootDir: **0 errors /565 baseline warnings /565 current warnings**, exact normalized diagnostic equality and zero added/removed. No rules were disabled. The author repeated this corrected lint with `lint230.mjs`; results are retained in `eslint-*.json` and `lint-scoped.log`.
- Full frontend TypeScript: **90 baseline /90 current**, zero added/removed, exact normalized diagnostic equality. `validate230.mjs` uses the actual tsconfig/compiler and overrides only the four owned files in memory for baseline; no production restoration or HMR change occurs.
- Owned diff whitespace check: exit0.
- Bandit attempted from activated project `.venv`: zero findings but **four TypeScript/TSX parse failures**. This is an unsupported-language limitation, not Python security evidence.
- Frozen source hashes match after tests/static checks; `freeze-check.json` records all four.

Original native404 is parent-retained `.tmp/uat225-native-20260917/console-from-line43.redacted.txt`; it is not copied or relabeled as final acceptance. Independent review and repeat native Settings interaction remain required before UAT230 closure.

## Independent review

The source review is clear with independent **87 PASS /6 files /0 skipped**, full compiler **90/90 unchanged**, and corrected parsed ESLint **0 errors /565 unchanged warnings**. Report `.tmp/uat230-independent-20260917/REVIEW230.md` SHA256 `95cf578e38527750408e3239e5fff3ce1c367a171567077f057722b95031efaf`. The initial ignored-file lint attempt is not counted as a passed source check. Production and test hashes remain frozen; this correction changes private evidence only.
