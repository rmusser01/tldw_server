# TASK13260.37 / UAT096 — independent review clear

No material issue found in the bounded Settings Form lifecycle change frozen at2026-09-15T22:31:14.991Z. All three owned-file hashes and both retained original-probe hashes match the frozen manifest; evidence `/private/tmp/uat096-independent-hashes.json`.

## Assessment

The only production change replaces initial `form.setFieldsValue` calls with accepted component-state values, then applies those values in an effect after `initializing` becomes false. The skeleton branch contains no Form, while the settled branch mounts the same owned Form instance. This corrects the actual Ant Form connection lifetime without timers or an auth-state workaround.

The existing `configLoadGeneration` check still precedes staging in both configuration branches; unmount still invalidates pending loads. Ordinary loading, login-status refreshes and storage account transitions do not replace the staged object, so they do not reapply old values over the mounted user's draft. Logout's explicit reload still stages the cleared/retained-target configuration as before. Failed logout and failed config loading retain their existing behavior. No auth transport, credential policy or account identity logic changed.

The new lifecycle suite uses the actual Ant Form/fields/context and actual Next WebUI storage aliases. Its configuration/logout/network services remain controlled. The normal already-mounted logout passing on old source is correctly documented as a control, not represented as the RED reproduction; native application-shell logout scheduling remains parent-owned.

## Fresh independent evidence

From `apps/tldw-frontend`, existing Vitest with `--maxWorkers=1 --no-file-parallelism`:

- **57 passed /5 suites**, `/private/tmp/uat096-independent-tests.log`: real form lifecycle7, auth-mode form5, cookie/logout24, connection review16, settings tabs5. Includes pending cookie/manual disconnect, storage A→B→A draft preservation, delayed old-owner load after replacement and genuine logout failure.
- Original unchanged `/private/tmp/uat096-baseline.config.ts` loading unchanged `/private/tmp/uat096-settings-before.tsx`: **1 expected failure /6 filtered**, `/private/tmp/uat096-independent-baseline-red.log`. Exact failure is `Instance created by useForm is not connected to any Form element`; visible loaded values still succeed. This reproduces the causal warning against original source.
- Independent ESLint from repository root covers both changed source/test files: **0 errors,33 baseline warnings,0 added warning signatures**, `/private/tmp/uat096-independent-eslint.json`. No ignored/outside-base-path file. The existing Next pages-directory advisory remains in the stderr log.
- Scoped `git diff --check`: passed.

No repository files, browser/runtime, commits or global docs changed during this review. Only private review reports/logs were written. Full compiler is deliberately parent-owned after the concurrent Chat/title scopes freeze; the existing90-diagnostic baseline is not a clean typecheck claim. Native logout acceptance is not inferred from unit tests.
