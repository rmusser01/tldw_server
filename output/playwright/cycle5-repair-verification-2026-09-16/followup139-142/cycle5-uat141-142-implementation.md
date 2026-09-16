# UAT141 / TASK13260.80 and UAT142 / TASK13260.81

Implementation frozen for independent rereview. Original139 evidence remains immutable. Current complete notification scope hashes are in `/private/tmp/cycle5-uat141-142-owned-manifest.json` (2 production, 3 tests, tasks78/80/81). The incremental141/142 edits affect only NotificationsRoute.tsx, notification-connectivity.integration.test.tsx, and official tasks80/81.

## UAT141: successful inbox freshness

Permanent actual-provider/actual-route tests reproduced two failures: cold verified-core with every notification read failing still displayed a success timestamp, and a later failure replaced a five-minute-old successful timestamp with “Just now”. The route now records a nullable inboxLoadedAt only after both inbox list requests succeed, clears it on scope change, and renders it only for its loaded scope. Lifecycle initialization/error times no longer drive the claim. Cached notifications remain available through a failed refresh. No provider field, schema, service, or new framework was needed.

RED: 2 failed / 4 intentionally unselected in `/private/tmp/cycle5-uat141-red.log`. Interim GREEN: 40 tests / 2 suites in `/private/tmp/cycle5-uat141-green.log`.

## UAT142: View continuation ownership

Permanent actual-provider/actual-route tests hold mark-read in flight, then transition through outage, owner switch, unmount, outage/recovery and owner A→B→A. All five stale actions originally navigated; the otherwise-identical connected control passed. View now captures the existing page generation before awaiting mark-read and checks it before navigation. Existing cleanup also invalidates that generation on unmount, preserving scope/outage invalidation. The normal connected action still routes to its same-origin source and sends mark-read exactly once; invalidated actions do not replay or navigate.

RED: 5 failed / 1 connected positive passed / 6 intentionally unselected in `/private/tmp/cycle5-uat142-red.log`.

## Final verification

**145 tests / 6 suites passed, no skips**, `/private/tmp/cycle5-uat141-142-green.log`. Scoped ESLint **0 errors / 0 warnings**, `/private/tmp/cycle5-uat141-142-lint.log` (empty, exit0). Owned-path git diff --check clean.

Working directory: apps/tldw-frontend. Installed local binaries only:

```sh
node_modules/.bin/vitest run __tests__/components/notification-connectivity.integration.test.tsx -t 'successful inbox' --maxWorkers=1 > /private/tmp/cycle5-uat141-red.log 2>&1
node_modules/.bin/vitest run __tests__/components/notification-connectivity.integration.test.tsx -t 'View authority' --maxWorkers=1 > /private/tmp/cycle5-uat142-red.log 2>&1
node_modules/.bin/vitest run __tests__/components/notification-connectivity.integration.test.tsx __tests__/components/notification-lifecycle-provider.test.tsx __tests__/pages/notifications.test.tsx __tests__/components/notification-rotation.integration.test.tsx ../packages/ui/src/components/Layouts/__tests__/ChatHeader.notifications.test.tsx ../packages/ui/src/services/__tests__/notification-lifecycle.test.ts --maxWorkers=2 > /private/tmp/cycle5-uat141-142-green.log 2>&1
node_modules/.bin/eslint components/notifications/NotificationLifecycleProvider.tsx components/notifications/NotificationsRoute.tsx __tests__/components/notification-connectivity.integration.test.tsx __tests__/components/notification-lifecycle-provider.test.tsx __tests__/pages/notifications.test.tsx > /private/tmp/cycle5-uat141-142-lint.log 2>&1
```

Both units used official CLI tracking before repository edits. No browser/runtime/API/inference actions, global tracker/plan changes, staging or commits. Tests use real provider/route components with controlled API/connection boundaries; no native acceptance claim. Native and root combined compiler checks remain pending. Existing Node localStorage warnings remain. Python Bandit is not applicable to TypeScript-only changes.
