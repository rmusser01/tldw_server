# UAT154 / TASK13260.92 — Settings test storage fixtures

## Result

61 tests across7 files passed, zero skips,32.59s. The36 existing auth/lifecycle cases now pass alongside25 timeout/request controls. Independent review remains pending. This is a test-fixture repair; no native/product behavior claim.

## Cause and minimal repair

The original3 fixtures used extension Plasmo Storage under jsdom without extension storage or its opt-in localStorage fallback. Real useSettingsLoginStatus reads therefore did not see seeded auth. Prototype get/watch spies also target the WebUI class-method adapter rather than Plasmo instance methods. This produced28 failures against unchanged pre-152 production1ea5402c83, with exactly the same28 failure headings as the expanded post-152 run.

Each fixture now resolves @plasmohq/storage to the existing Next WebUI shim, following established shared-UI test patterns. Its real get/set/watch/unwatch and same-tab/cross-tab storage bridge execute. No hook/auth mocks or assertions were weakened, no helper or production implementation added. Removing the five inserted adapter lines from each file leaves every original byte unchanged:77 expect call sites preserved. The existing extension-style boolean-watch control still overrides watch's return value and tests cleanup; owner ABA, rotation, drafts, auth mode, logout and billing cases remain unchanged.

## Exact owned test paths

- apps/packages/ui/src/components/Option/Settings/__tests__/tldw-auth-mode.form.test.tsx (5 cases)
- apps/packages/ui/src/components/Option/Settings/__tests__/tldw.form-lifecycle.test.tsx (7 cases)
- apps/packages/ui/src/components/Option/Settings/__tests__/tldw.cookie-logout.test.tsx (24 cases)
- Official CLI task13260.92 record only.

Frozen152/153 source, new timeout test and task91 were not edited; explicit hash audit retained.

## Verification

Working directory: /Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui

```sh
./node_modules/.bin/vitest run src/components/Option/Settings/__tests__/tldw-auth-mode.form.test.tsx src/components/Option/Settings/__tests__/tldw.form-lifecycle.test.tsx src/components/Option/Settings/__tests__/tldw.cookie-logout.test.tsx src/components/Option/Settings/__tests__/tldw.timeouts.form.test.tsx src/components/Option/Settings/__tests__/tldw-settings-tabs.test.tsx src/services/tldw/__tests__/request-core.refresh-timeout.test.ts src/services/__tests__/tldw-settings-storage.test.ts --maxWorkers=1 > /private/tmp/cycle5-uat154-green.log 2>&1
```

From repository root:

```sh
apps/tldw-frontend/node_modules/.bin/eslint --config apps/tldw-frontend/eslint.config.mjs apps/packages/ui/src/components/Option/Settings/__tests__/tldw-auth-mode.form.test.tsx apps/packages/ui/src/components/Option/Settings/__tests__/tldw.form-lifecycle.test.tsx apps/packages/ui/src/components/Option/Settings/__tests__/tldw.cookie-logout.test.tsx -f json > /private/tmp/cycle5-uat154-eslint-final.json
```

ESLint0 errors/0 warnings, unchanged baseline. Config emits the existing root Pages-directory advisory. Vitest retains jsdom CSS parser/Node localStorage diagnostics and a Flashcards transfer cleanup advisory in the first-connection timeout control; none is hidden or converted into a pass. Parent owns compiler comparison. Bandit N/A: TypeScript test-only, no Python/production edits. No browser/runtime/inference/staging/commit.

## Evidence

- Baseline RED: /private/tmp/cycle5-uat152-existing-baseline.log (28failed/8passed).
- Baseline configuration: /private/tmp/cycle5-uat152-baseline.config.mts plus baseline-tldw.tsx and baseline-TldwTimeoutSettings.tsx.
- Exact failure heading equality: /private/tmp/cycle5-uat152-baseline-comparison.json.
- Original diagnosis: /private/tmp/cycle5-uat154-readonly-fixture-diagnosis.md.
- Final GREEN: /private/tmp/cycle5-uat154-green.log.
- Byte-preservation: /private/tmp/cycle5-uat154-assertion-preservation.json.
- Lint: /private/tmp/cycle5-uat154-eslint-{baseline,final}.json and -eslint-comparison.json.
- Freeze: /private/tmp/cycle5-uat154-code-freeze.json and -owned-manifest.json.
- Prior freeze audit: /private/tmp/cycle5-uat154-prior-freeze-audit.json.
