# TASK13260.37 / UAT096 — Settings Form hydration

Status: production/tests frozen for independent review at **2026-09-15T22:31:14.991Z**. Exact source, test and task hashes plus original replay artifacts are in `/private/tmp/uat096-frozen-manifest.json`. No commit, browser/runtime action, auth/Chat production change or global document edit was performed.

## Diagnosis and minimal correction

The native normal logout at 22:13:43 UTC succeeded and showed Login Required; its retained console additionally emitted the Ant Form not-connected warning. Evidence: `/private/tmp/uat093-admin-logout.txt` and `.playwright-cli/console-2026-09-15T22-05-15-136Z.log` lines 33–35. The nearby unauthenticated warnings are separate and expected after logout.

`TldwSettings` creates its Form instance before configuration loading, but the initializing branch renders only a skeleton. Both loaded-config branches called `form.setFieldsValue` before the first Form committed. Installed `@rc-component/form` 1.6.2 explicitly calls `warningUnhooked()` from `setFieldsValue`, checking connection in a subsequent timer. A delayed React commit allows that timer to run while no Form owns the instance. The existing cookie/logout fixture mocked Form and could not catch this lifetime.

Only production file changed: `apps/packages/ui/src/components/Option/Settings/tldw.tsx`.

- Stage accepted configuration fields in component state.
- Apply them in an effect only after initializing is false and the owning Form has committed.
- Preserve the existing configuration-load generation guard, skeleton, field names, auth handlers, current draft behavior, and cookie/manual logout semantics.

The actual normal mounted logout is a passing control even before this correction. The automated RED reproduces the concrete pre-mount configuration lifetime and exact same warning; it does not reproduce the complete native application shell's logout scheduling. Parent native retest is therefore still required before claiming UAT096 closed.

## Tests and RED → GREEN

New test: `apps/packages/ui/src/components/Option/Settings/__tests__/tldw.form-lifecycle.test.tsx`.

It uses actual TldwSettings, ConnectionSettings, Ant Form/fields, Ant App/message context, login-status hook and canonical web storage. Configuration/logout services and external health/discovery are controlled fixtures; no server or browser runtime is contacted. The hydration case deliberately holds the async React commit across the real warning timer and spies on console.error before the general test suppression layer. It asserts both correctly loaded visible fields and absence of the actual warning.

Seven controls:

1. Deferred initial configuration does not access an unconnected Form.
2. Normal multi-user logout reaches the real Login Required form with its target retained.
3. Pending cookie logout clears the form when complete.
4. Pending manual-key disconnect clears the form when complete.
5. Same-owner mounted field draft survives A→B→A storage transitions.
6. A late old-account initial load cannot overwrite the replacement owner's edited form.
7. Genuine logout failure before credential removal leaves current form/login visible and displays the existing failure message.

Evidence:

- `/private/tmp/uat096-form-lifecycle-final-red.log`: **1 failed / 6 passed** before production correction, exact warning failure.
- `/private/tmp/uat096-form-final-tests.log`: **57 passed in 5 suites** after correction: 7 new real-owner cases, 24 cookie/logout cases, 5 real auth-mode cases, 16 connection review cases and 5 tab cases.
- `/private/tmp/uat096-baseline-replay-red.log`: **1 failed / 6 filtered**, replaying the unchanged permanent case against the exact original source. The original source and replay config are `/private/tmp/uat096-settings-before.tsx` and `/private/tmp/uat096-baseline.config.ts`.

These runs overlap; filtered cases are not counted as passes. Initial exploratory invocations under the shared package's default Vitest config lacked the web Plasmo storage shim, so canonical login fixtures failed there. That was a harness mismatch, corrected by using the existing web config, not an auth finding or a product workaround. The real-form suite retains jsdom CSS parser notices and the existing Node localStorage advisory; they are not native browser results.

Final focused command from `apps/tldw-frontend`:

```sh
./node_modules/.bin/vitest run ../packages/ui/src/components/Option/Settings/__tests__/tldw.form-lifecycle.test.tsx ../packages/ui/src/components/Option/Settings/__tests__/tldw.cookie-logout.test.tsx ../packages/ui/src/components/Option/Settings/__tests__/tldw-auth-mode.form.test.tsx ../packages/ui/src/components/Option/Settings/__tests__/tldw-review-comments.test.tsx ../packages/ui/src/components/Option/Settings/__tests__/tldw-settings-tabs.test.tsx --maxWorkers=1 --no-file-parallelism
```

Original-source replay from the same directory:

```sh
./node_modules/.bin/vitest run --config /private/tmp/uat096-baseline.config.ts --maxWorkers=1 --no-file-parallelism -t 'loads the actual form'
```

## Static checks and limits

Root-scoped ESLint covers both owned source/test files: **0 errors, 33 unchanged existing warnings, 0 added**. Exact baseline/current/comparison JSON: `/private/tmp/uat096-eslint-baseline.json`, `/private/tmp/uat096-eslint-current.json`, `/private/tmp/uat096-eslint-comparison.json`. New test has zero warnings. Root command:

```sh
apps/tldw-frontend/node_modules/.bin/eslint --config apps/tldw-frontend/eslint.config.mjs apps/packages/ui/src/components/Option/Settings/tldw.tsx apps/packages/ui/src/components/Option/Settings/__tests__/tldw.form-lifecycle.test.tsx -f json
```

Scoped diff-check passes. Bandit is not applicable to this TypeScript-only scope. Full TypeScript was not run during concurrent adjacent Chat work; the parent owns the stable combined compiler comparison. The known 90-diagnostic baseline is not a clean compiler claim.

Independent review and parent native normal logout retest remain pending. No permissions, session policy, transport behavior or expected post-logout 401 responses were altered. The Backlog task remains In Progress with these limits recorded.
