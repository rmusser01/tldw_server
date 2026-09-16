# UAT139 / TASK13260.78 implementation

## Result
Notification outages now present reconnecting guidance while verified connection remains a separate, required request prerequisite. Authentication failures and missing multi-user credentials retain sign-in guidance; terminal notification permission failures retain unavailable guidance. No private inbox requests occur on an initially network-unverified or missing-credentials render.

Production changes are limited to the web NotificationLifecycleProvider and NotificationsRoute. The derived connectionVerified field is omitted from the runtime snapshot. Both lifecycle start and synchronous projection distinguish unreachable from auth; callback dependencies cover errorKind changes while verification remains false. Route reads, handlers and controls require verification and nonterminal state. Connection loss aborts inbox reads, clears retry work and invalidates pending response generations without replaying mutations. Cold outage omits the misleading last-updated timestamp.

## Verification
- RED: 4 expected failures in actual-provider/actual-route integration, including two inbox calls before missing-credential state reached the child; cycle5-uat139-red.log.
- Initial focused GREEN: 109 tests / 4 suites; cycle5-uat139-green.log.
- Final GREEN: **137 tests / 6 suites**, no skips; cycle5-uat139-final-green.log.
- Scoped ESLint: **0 errors / 0 warnings**; cycle5-uat139-lint.log (empty, exit 0).
- git diff --check on owned paths: exit 0.
- Additional permanent controls cover disabled cached item/preferences actions, delayed preferences across outage, fresh preference load after recovery, and delayed mutation result after reconnect with no replay. Existing credential rotation, scope, terminal auth/permission, retry, header, and classifier suites pass.

The expanded test run initially used the old Preferences button label after opening that panel; two test-only selectors were corrected to its existing Hide Preferences label before the final passing run. No product adjustment was made for that test harness issue.

## Exact commands
Working directory: apps/tldw-frontend.

```sh
node_modules/.bin/vitest run __tests__/components/notification-connectivity.integration.test.tsx --maxWorkers=2 > /private/tmp/cycle5-uat139-red.log 2>&1
node_modules/.bin/vitest run __tests__/components/notification-connectivity.integration.test.tsx __tests__/components/notification-lifecycle-provider.test.tsx __tests__/pages/notifications.test.tsx __tests__/components/notification-rotation.integration.test.tsx ../packages/ui/src/components/Layouts/__tests__/ChatHeader.notifications.test.tsx ../packages/ui/src/services/__tests__/notification-lifecycle.test.ts --maxWorkers=2 > /private/tmp/cycle5-uat139-final-green.log 2>&1
node_modules/.bin/eslint components/notifications/NotificationLifecycleProvider.tsx components/notifications/NotificationsRoute.tsx __tests__/components/notification-connectivity.integration.test.tsx __tests__/components/notification-lifecycle-provider.test.tsx __tests__/pages/notifications.test.tsx > /private/tmp/cycle5-uat139-lint.log 2>&1
```

## Freeze and limits
Base: 3c30685611a26d0495dafabd38f594fee93a1655. Freeze: 2026-09-16T15:18:49.016Z. Six owned paths and SHA256 values: /private/tmp/cycle5-uat139-owned-manifest.json (2 production, 3 tests, official task78).

Only installed local Vitest used. No dependencies installed, no shared locales/source, runtime/browser/API/inference actions, staging or commits. Bandit is not applicable to these TypeScript-only changes. Native recheck, independent review and root combined compiler validation remain pending. Tests exercise real provider/route components with API, connection and credential boundaries mocked; they do not claim native network proof. Node localStorage experimental warnings are emitted by the existing test environment.
