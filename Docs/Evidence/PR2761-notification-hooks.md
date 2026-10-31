# PR2761 notification hook repairs

Tracking: TASK-12116; release parent TASK-13013.3. Bounded follow-up on candidate 6150040801, September 10, 2026.

## Demonstrated failures

- `useNotificationCount`: removing the artificial second rerender from the existing account-switch test made it fail: expected 9, received 0. The effect only changed a ref, so it never requested the render needed to lift its barrier. Original suite with that regression: 1 failed, 4 passed.
- `useAntdNotification`: a frozen context API threw `TypeError: Cannot add property success, object is not extensible`; a mutable API's existing method was replaced for other consumers; a partial static fallback lacked normalized severity methods. Initial regression suite: 3 failed, 1 passed.
- Installed Plasmo's hook accepts `{ key, instance }`, not `{ key, area, serde }`. Its dynamic-key value is retained until asynchronous loading completes. Clearing the old ref barrier alone would therefore be insufficient to guarantee account isolation.

Initial RED commands, from `apps/tldw-frontend`:

```sh
node_modules/.bin/vitest run ../packages/ui/src/hooks/__tests__/useNotificationCount.test.tsx --maxWorkers=1
node_modules/.bin/vitest run ../packages/ui/src/hooks/__tests__/useAntdNotification.test.tsx --maxWorkers=1
```

## Changes

The count hook uses the existing `createSafeStorage({ area: "local" })` helper and supported hook instance option for config and selector subscriptions. Dynamic notification records are read and watched with their subscription identity attached to the result. A count is visible only for the current authenticated scope and subscription. Scope changes immediately render zero; the current read/watch result schedules recovery. Cleanup unregisters the exact callback map and ignores superseded reads; watch updates win over older pending reads. A fresh subscription identity prevents reuse of an earlier A snapshot during A → B → A transitions. Missing values and failed reads resolve to zero; later watch updates can recover. The hook no longer uses write-through initializer callbacks.

The Ant Design hook returns a memoized adapter. It normalizes deprecated `message` to `title`, supplies missing severity methods through `open`, preserves existing method receivers and `destroy`, and refreshes when the provider API changes. It never mutates the shared API. Static compatibility setup remains in the existing WebUI and extension application entrypoints; the hook itself normalizes its fallback directly.

No storage schema, public hook return type, global lint configuration, release metadata, or package changes.

## Verification

Final GREEN command, from `apps/tldw-frontend`: **4 files, 25 tests passed**.

```sh
node_modules/.bin/vitest run \
  ../packages/ui/src/hooks/__tests__/useNotificationCount.test.tsx \
  __tests__/hooks/useNotificationCount.storage.test.tsx \
  ../packages/ui/src/hooks/__tests__/useAntdNotification.test.tsx \
  ../packages/ui/src/utils/__tests__/antd-notification-compat.test.ts \
  --maxWorkers=1
```

The 16 count unit cases cover normal and StrictMode recovery, delayed data, account-selector lag, unresolved credentials, missing/invalid records, read/watch ordering, superseded A → B → A reads, unsubscribe, read failure, and later recovery. Two additional cases exercise real WebUI storage subscriptions through account changes, deletion, and updates, and verify mount does not write shared storage. Four adapter tests and three existing normalization tests cover the notification compatibility surface.

Focused lint command, from repository root: **exit 0, zero file diagnostics**.

```sh
apps/tldw-frontend/node_modules/.bin/eslint \
  --config apps/tldw-frontend/eslint.config.mjs \
  apps/packages/ui/src/hooks/useNotificationCount.ts \
  apps/packages/ui/src/hooks/useAntdNotification.ts \
  apps/packages/ui/src/hooks/__tests__/useNotificationCount.test.tsx \
  apps/tldw-frontend/__tests__/hooks/useNotificationCount.storage.test.tsx \
  apps/packages/ui/src/hooks/__tests__/useAntdNotification.test.tsx \
  --rule 'react-hooks/refs:error' \
  --rule 'react-hooks/immutability:error' \
  --rule 'react-hooks/set-state-in-effect:error' \
  --rule 'react-hooks/preserve-manual-memoization:error'
```

Two temporary focused configurations extend the existing WebUI tsconfig with incremental output disabled, the five touched source/test files, and existing ambient declarations. The second removes the two Plasmo aliases to check against the installed extension package declarations. Both commands returned **exit 0, zero diagnostics**:

```sh
apps/tldw-frontend/node_modules/.bin/tsc -p /tmp/pr2761-notification-hooks-tsconfig.json --noEmit --incremental false
apps/tldw-frontend/node_modules/.bin/tsc -p /tmp/pr2761-notification-hooks-extension-tsconfig.json --noEmit --incremental false
```

`git diff --check` passed. No full build, install, container operation, commit, or push was performed.

## Limits

Vitest printed the existing Node experimental localStorage warning. ESLint printed its existing repository-root pages-directory notice. Neither produced test failures or file diagnostics.

The actual browser-extension runtime was not launched; extension verification is source/API inspection plus installed-package type checking. The storage integration suite above uses the WebUI Vitest configuration and real WebUI storage shim.

Bandit was invoked through the project virtual environment on both touched TypeScript files, writing `/tmp/bandit_pr2761_notification_hooks.json`. Its JSON reports two Python AST syntax errors because it cannot parse TypeScript. This is not a successful security scan. No Python code was changed.

Shared-package collection parity was also verified after placing the WebUI-only integration test under `apps/tldw-frontend/__tests__/hooks/`. From `apps/packages/ui`, the unchanged shared Vitest configuration passed **3 files, 23 tests**:

```sh
../../tldw-frontend/node_modules/.bin/vitest run \
  src/hooks/__tests__/useNotificationCount.test.tsx \
  src/hooks/__tests__/useAntdNotification.test.tsx \
  src/utils/__tests__/antd-notification-compat.test.ts \
  --maxWorkers=1
```

After the test move, the WebUI selection again passed 25 tests; both focused typechecks and the lint command above again exited 0.
