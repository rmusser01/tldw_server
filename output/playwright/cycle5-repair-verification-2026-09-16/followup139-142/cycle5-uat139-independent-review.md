# UAT139 independent review

Reviewed base: 3c30685611a26d0495dafabd38f594fee93a1655.
Frozen implementation: 2026-09-16T15:18:49.016Z.
Outcome: two bounded gaps in the requested late-action / truthful-timestamp controls; not yet clear for those boundaries. Both paths predate this patch, so these are incomplete coverage of the stated repair constraints, not newly introduced regressions or new native observations.

## Findings

### P2 — Pending View still navigates after verification is lost

`apps/tldw-frontend/components/notifications/NotificationsRoute.tsx:708–716`

The View button awaits `handleMarkRead`, then uses the captured notification link without checking page generation or current request eligibility. The inner handler correctly discards a late mutation response, but returns normally. Thus a View click before an outage continues routing after the connection has become unverified and the rendered View button is disabled. Private real-provider/real-route probe: one mark-read dispatch is held; verification is lost; resolving the old dispatch still calls router.push('/media/71'). The otherwise identical connected control routes correctly.

Capture/check the page action generation around this await (including unmount/authority invalidation as applicable) before navigating; retain normal View navigation and do not replay the mutation. This probe establishes stale navigation after outage, not a demonstrated foreign-account API dispatch or server-side access breach.

### P3 — Failed first notification reads still fabricate a last-success timestamp

`apps/tldw-frontend/components/notifications/NotificationsRoute.tsx:550–552`, with `NotificationLifecycleProvider.tsx:75–80,202–206`

The added connectionVerified condition only hides the timestamp when the shared connection is unverified. A verified core connection with all notification reads failing still renders “Last updated before the connection was lost (Just now)”. The provider's updatedAt records initialization/state/error time, including a fresh Date.now() on failure; it does not prove notification data was ever read successfully. This can also make warm failure timestamps appear fresher than the actual successful data.

Omit that success wording unless backed by a real successful notification update, or use a distinct nullable success timestamp that errors cannot advance. Preserve the cold-outage omission already covered by the author.

## Independent evidence

- All six current source/test/task hashes match `/private/tmp/cycle5-uat139-owned-manifest.json`; audit at `/private/tmp/cycle5-uat139-independent-hashes.json`.
- Installed local Vitest: **137 tests / 6 suites PASS**, no skips. `/private/tmp/cycle5-uat139-independent-tests.log`.
- Bounded private probes use a read-only Vite load override of the existing actual-provider/route test. **2 failures / 1 passing connected-View control**; no repository source/test edits.
  - `/private/tmp/cycle5-uat139-private-tests.txt`
  - `/private/tmp/cycle5-uat139-private.config.ts`
  - `/private/tmp/cycle5-uat139-independent-probes.log`
- Production source inspection shows both timestamp semantics and post-await navigation existed at the base. The patch adds useful verification gating but leaves these two branches reachable.

## Positive review findings

The derived connectionVerified field is excluded from the runtime snapshot, so it cannot become a stale cached authorization flag. Synchronous context projection handles missing configured multi-user credentials before the child's first effect. Network-unverified state uses reconnecting presentation while request permission remains false; errorKind transitions update the projection even while verification remains false. Real integration controls demonstrate no initial private reads, no credential-missing early dispatch and successful same-scope recovery. Existing terminal auth/permission and actual canonical-rotation/scope controls remain green.

Route reads and ordinary mutation handlers check request eligibility; outage invalidates generations, aborts inbox reads and clears retries. The added late-preference and late-mark-read controls pass, with no automatic mutation replay. The separate View navigation continuation is the exception above.

## Commands

From `apps/tldw-frontend`:

```sh
node_modules/.bin/vitest run __tests__/components/notification-connectivity.integration.test.tsx __tests__/components/notification-lifecycle-provider.test.tsx __tests__/pages/notifications.test.tsx __tests__/components/notification-rotation.integration.test.tsx ../packages/ui/src/components/Layouts/__tests__/ChatHeader.notifications.test.tsx ../packages/ui/src/services/__tests__/notification-lifecycle.test.ts --maxWorkers=2
node_modules/.bin/vitest run --config /private/tmp/cycle5-uat139-private.config.ts __tests__/components/notification-connectivity.integration.test.tsx -t 'independent:' --maxWorkers=1 --no-file-parallelism
```

## Limits

No browser/runtime/API/inference action, repository/task edit, staging or commit. Controlled API/connection fixtures establish mounted component behavior, not native network timing or live access control. Existing rotation suite exercises its more concrete credential/storage/transport boundaries. No full compiler or linter rerun in this read-only review; author reports scoped lint 0/0 and root owns combined compiler validation. UAT140/013/137/138 excluded.
