# Final independent review — UAT139 / UAT141 / UAT142

## Verdict

**Clear for the scoped implementation. No remaining actionable finding.** Original UAT139 request-gating/presentation, UAT141 truthful freshness, and UAT142 delayed View ownership corrections pass the reviewed boundaries. Native acceptance remains separate and pending controller verification.

Reviewed working HEAD baseline: 7c7f4093df; cumulative repair source base in the author manifest: 3c30685611a26d0495dafabd38f594fee93a1655. Frozen scope: `/private/tmp/cycle5-uat142-batched-owned-manifest.json`, 2026-09-16T15:36:00.641Z. All eight frozen path hashes match; audit `/private/tmp/cycle5-uat142-batched-final-independent-hashes.json`.

## Prior ABA finding resolved

`NotificationLifecycleProvider.tsx:173–175,431–449,492–507` captures a synchronous authority revision. Actual observed server/principal changes and removed credentials invalidate it even when React batches A→B→A into final rendered A. `NotificationsRoute.tsx:712–719` captures that predicate before mark-read and requires it plus the existing route generation after the await.

The revision is distinct from request generation and excludes ordinary same-scope restarts. Canonical scope comparison remains the existing helper; distinct tokens for the same decoded subject do not create a new authority. The predicate is excluded from the runtime snapshot. Existing page cleanup still handles committed scope changes, verification loss and unmount.

No new event listener, token store, transport, automatic mutation replay or broad lifecycle abstraction was introduced.

## Independent tests

- **150 tests / 6 suites PASS**, no skips: `/private/tmp/cycle5-uat142-batched-final-independent-tests.log`.
- **Original batched private fixture/config replayed unchanged: 2/2 PASS**, 17 unrelated author cases intentionally deselected: `/private/tmp/cycle5-uat142-batched-final-original-probes.log`. This exact negative previously navigated incorrectly; same-scope positive remains valid.
- **Original139 private fixture/config replayed unchanged: 3/3 PASS**, 17 unrelated author cases intentionally deselected: `/private/tmp/cycle5-uat142-final-original-139-probes.log`. Cold all-reads-failed timestamp remains absent, delayed outage View cancels, normal connected View routes.

Permanent actual-provider/actual-route controls include server A→B→A, Alice→Bob→Alice, same-config event, genuinely different token strings for the same decoded principal, and removed/restored credentials. Changed/removal authorities cancel; same-owner controls route. Every case sends mark-read once only. Existing committed owner/outage/recovery/unmount, notification transport rotation, no-initial-private-dispatch and terminal auth/permission controls remain green.

Freshness still starts null, changes only after both current inbox reads succeed, retains the prior successful time through failure, and clears on owner change. The independent five-minute-old success/failure control passes. The provider's initialization/error timestamp no longer supplies the inbox success claim.

## Exact commands

Working directory: `apps/tldw-frontend`; installed local Vitest only.

```sh
node_modules/.bin/vitest run __tests__/components/notification-connectivity.integration.test.tsx __tests__/components/notification-lifecycle-provider.test.tsx __tests__/pages/notifications.test.tsx __tests__/components/notification-rotation.integration.test.tsx ../packages/ui/src/components/Layouts/__tests__/ChatHeader.notifications.test.tsx ../packages/ui/src/services/__tests__/notification-lifecycle.test.ts --maxWorkers=2
node_modules/.bin/vitest run --config /private/tmp/cycle5-uat142-batched-private.config.ts __tests__/components/notification-connectivity.integration.test.tsx -t 'independent-batched:' --maxWorkers=1 --no-file-parallelism
node_modules/.bin/vitest run --config /private/tmp/cycle5-uat139-private.config.ts __tests__/components/notification-connectivity.integration.test.tsx -t 'independent:' --maxWorkers=1 --no-file-parallelism
```

## Limits

Controlled component tests use actual provider, route and scope builder with synthetic credentials and controlled API/connection boundaries. They do not prove native auth/network timing or server access. No new launched-browser/runtime/API/inference work, repository/task edits, staging or commits. Original failing probes/reports remain preserved; this report supersedes their finding status only for the new frozen bytes. Whole compiler and the six unrelated Form-suite failures remain root-owned and are not claimed resolved here. Existing Node localStorage advisory appears in test logs. No independent lint rerun was necessary for this read-only recheck; author scoped lint reports 0/0. Native acceptance is not inferred from these automated results.
