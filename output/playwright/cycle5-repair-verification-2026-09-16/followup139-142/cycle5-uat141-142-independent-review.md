# UAT139 / UAT141 / UAT142 independent rereview

Reviewed frozen scope: `/private/tmp/cycle5-uat141-142-owned-manifest.json`, 2026-09-16T15:27:45.836Z; base 3c30685611a26d0495dafabd38f594fee93a1655.

## Outcome

UAT139 request gating remains sound in the reviewed controls. UAT141 resolves the prior freshness finding. UAT142 resolves the original delayed-outage finding and committed owner/unmount/recovery transitions, but one **P2 batched authority A→B→A gap remains**. This is within existing TASK13260.81, not a new native-discovered issue.

## Remaining P2: View generation misses observed authority changes collapsed into one render

`apps/tldw-frontend/components/notifications/NotificationsRoute.tsx:710–716` captures/checks `pageGenerationRef`, whose authority invalidation depends on the scope-key effect at approximately lines310–333.

Actual-provider/actual-route private reproduction:

1. Connected A inbox contains a source link; click View and hold mark-read pending.
2. Inside one React batch, change the effective-config fixture to server B and dispatch the real `tldw:config-updated` event, then restore server A and dispatch the same event.
3. The provider observes both changes and invalidates its own work, but React commits only final scope A. The route's scope effect sees no A→B change; canRequest also remains true.
4. Resolve the old mark-read. The View continuation passes its page-generation check and calls router.push('/media/71').

The otherwise identical same-scope configuration event passes its normal navigation control. Exactly one mark-read occurred in each case; this is stale navigation, not a demonstrated foreign API read or mutation replay. The permanent owner-returned test uses two separate rerenders, so it does not exercise this batching boundary.

Bind the pending View continuation to authority transitions already observed by the provider, including an intermediate authority that never becomes the final rendered scope. Preserve the legitimate same-owner event control. A current scope string alone cannot detect this ABA sequence.

Private evidence (read-only Vite loader; no repo edits):

- `/private/tmp/cycle5-uat142-batched-private-tests.txt`
- `/private/tmp/cycle5-uat142-batched-private.config.ts`
- `/private/tmp/cycle5-uat142-batched-independent-probes.log`: **1 expected failure / 1 positive control passed**, 12 unrelated tests intentionally deselected.

## Resolved findings and retained behavior

- Original cold verified-core/all-notification-reads-failed timestamp probe now passes unchanged. `inboxLoadedAt` starts null and changes only after both current inbox requests succeed. Failure handlers do not advance it; owner change clears it; rendering is scoped to the loaded owner.
- Permanent five-minute-old successful inbox → failed refresh control passes and keeps cached items. New-owner cold failure neither exposes the previous items nor claims their timestamp.
- Original delayed View/outage probe now passes unchanged; connected View still routes normally.
- Permanent controls cover committed owner switch, unmount, outage/recovery, and committed A→B→A, without mutation replay. Cleanup invalidates continuations on unmount.
- Retained139 derived verification remains separate from runtime snapshots. Initial network-unverified/missing-credential renders dispatch no private work; network→auth presentation and same-scope recovery controls pass. Cached action disablement, preference/mark-read delayed outcomes, terminal auth/permission and canonical rotation controls remain green.

## Independent verification

- Eight frozen source/test/task SHA256 values match; `/private/tmp/cycle5-uat141-142-independent-hashes.json`.
- Permanent installed-local six-suite command: **145/145 PASS**, no skips; `/private/tmp/cycle5-uat141-142-independent-tests.log`.
- Original three private probes/config were rerun **unchanged**: **3/3 PASS**, twelve author cases intentionally unselected; `/private/tmp/cycle5-uat141-142-independent-original-probes.log`. Original RED evidence and fixtures remain intact at `/private/tmp/cycle5-uat139-private-tests.txt`, `-private.config.ts`, and `-independent-probes.log`.

Commands from `apps/tldw-frontend`:

```sh
node_modules/.bin/vitest run __tests__/components/notification-connectivity.integration.test.tsx __tests__/components/notification-lifecycle-provider.test.tsx __tests__/pages/notifications.test.tsx __tests__/components/notification-rotation.integration.test.tsx ../packages/ui/src/components/Layouts/__tests__/ChatHeader.notifications.test.tsx ../packages/ui/src/services/__tests__/notification-lifecycle.test.ts --maxWorkers=2
node_modules/.bin/vitest run --config /private/tmp/cycle5-uat139-private.config.ts __tests__/components/notification-connectivity.integration.test.tsx -t 'independent:' --maxWorkers=1 --no-file-parallelism
node_modules/.bin/vitest run --config /private/tmp/cycle5-uat142-batched-private.config.ts __tests__/components/notification-connectivity.integration.test.tsx -t 'independent-batched:' --maxWorkers=1 --no-file-parallelism
```

## Limits

No source/task edits, browser/runtime/API/inference actions, staging or commits. These tests mount actual provider and route with controlled config/connection/API boundaries; they do not establish native acceptance or actual server access. Full compiler and native checks remain controller-owned. Prior reports/manifests/probes preserved. Review excludes UAT013/137/138/140 and other working-tree changes.
