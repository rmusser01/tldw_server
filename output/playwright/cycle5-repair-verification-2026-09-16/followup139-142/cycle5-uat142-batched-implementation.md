# UAT142 / TASK13260.81 — batched authority correction

Frozen for independent rereview. Current cumulative139/141/142 SHA256 manifest: `/private/tmp/cycle5-uat142-batched-owned-manifest.json`. Earlier manifests, RED/GREEN logs and reports remain historical and unchanged.

## Root cause and correction

A pending View already checked the route page generation after awaiting mark-read. That generation observed committed scope/verification changes, but a provider-observed A→B→A event sequence could collapse into a final render of A. The old continuation then appeared current.

NotificationLifecycleProvider now records a narrow authority revision inside its existing synchronous scope-change and credential-removal handlers. It exposes a captured predicate checking that revision; the function is explicitly omitted from the runtime snapshot. View checks this captured authority alongside its existing page generation before navigation. There is no new token store, event listener, transport, schema, broad cancellation framework or mutation replay.

This revision is distinct from the provider's request generation: normal reconnect and same-principal token refresh may restart notification reads without changing authority. Scope comparison reuses the existing notification scope builder, including canonical principal decoding. Same-owner refresh is deliberately valid.

## Permanent RED / GREEN

Added actual Provider→Route controls dispatch real config events in a single React batch while mark-read awaits:

- server A→B→A: RED stale navigation before correction;
- principal Alice→Bob→Alice: RED stale navigation before correction;
- same config event: connected View remains valid;
- genuine distinct JWT values for the same decoded subject: View remains valid;
- credentials removed/restored: pending View stays cancelled.

RED: **2 failed / 3 passed / 12 intentionally unselected**, `/private/tmp/cycle5-uat142-batched-red.log`.

Final: **150 tests / 6 suites passed**, no skips, `/private/tmp/cycle5-uat142-batched-green.log`. Existing139 no-private-dispatch,141 successful freshness, committed owner/outage/unmount and credential rotation controls remain green. Scoped ESLint **0 errors / 0 warnings**, `/private/tmp/cycle5-uat142-batched-lint.log` (empty, exit0); owned git diff --check clean.

## Commands

From apps/tldw-frontend, installed local binaries:

```sh
node_modules/.bin/vitest run __tests__/components/notification-connectivity.integration.test.tsx -t 'synchronously observed authority' --maxWorkers=1 > /private/tmp/cycle5-uat142-batched-red.log 2>&1
node_modules/.bin/vitest run __tests__/components/notification-connectivity.integration.test.tsx __tests__/components/notification-lifecycle-provider.test.tsx __tests__/pages/notifications.test.tsx __tests__/components/notification-rotation.integration.test.tsx ../packages/ui/src/components/Layouts/__tests__/ChatHeader.notifications.test.tsx ../packages/ui/src/services/__tests__/notification-lifecycle.test.ts --maxWorkers=2 > /private/tmp/cycle5-uat142-batched-green.log 2>&1
node_modules/.bin/eslint components/notifications/NotificationLifecycleProvider.tsx components/notifications/NotificationsRoute.tsx __tests__/components/notification-connectivity.integration.test.tsx __tests__/components/notification-lifecycle-provider.test.tsx __tests__/pages/notifications.test.tsx > /private/tmp/cycle5-uat142-batched-lint.log 2>&1
```

## Scope and limits

Incremental files: notification provider and route; existing actual integration test; isolated route mock gains the required capture predicate; official task81 only. Cumulative manifest includes stable earlier notification paths/task records so root can verify the complete unit. No further source/test/task writes planned.

No browser/runtime/API/inference, staging or commits. Tests use actual components and canonical scope helper with controlled config/auth/API fixtures; not a native authentication or server acceptance claim. Root owns native/compiler checks. Existing Node localStorage advisory retained. Python Bandit does not apply to these TypeScript changes.
