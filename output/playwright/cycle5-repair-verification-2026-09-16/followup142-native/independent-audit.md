# UAT142 independent native acceptance audit

**Verdict: TASK-13260.81 acceptance is supported.** The retained native controls confirm that a late successful mark-read response cannot redirect after route departure or actual Disconnect, while the current authorized View action still marks read and navigates. Existing permanent tests cover the additional owner/verification boundaries. No remaining UAT142 defect is demonstrated by this evidence.

Reviewed 2026-09-16. This pass inspected retained scripts, receipts, fixture provenance, source hashes and screenshots only. No browser/runtime/product changes, inference, or git operations.

## Native controls

| Control | Real backend evidence | Continuation outcome |
| --- | --- | --- |
| Route departure (`bounded-control3`) | Notification3; POST mark-read `ids:[3]`;200 `updated:1`; response held for2012ms using `route.fetch()` then `route.fulfill({response})` | Actual Notes click changes `/notifications` to `/notes` before release. Fulfillment succeeds; final URL stays `/notes` after502ms. Screenshot shows actual Notes UI. |
| Current-authority positive | Notification4; actual POST `ids:[4]`;200 `updated:1`; no response interception in this control | View reaches `/prompts`, matching the fixture link. Total recorded action interval2658ms. |
| Authority removal (`authority-control`) | Notification5; POST `ids:[5]`;200 `updated:1`; response held for622ms | Real Settings Disconnect occurs116ms after the response is held. Before release, the inbox displays “Sign in again to view notifications.” Fulfillment succeeds; `/notifications` and sign-in state persist after503ms. Screenshot corroborates this state. |

The scripts delay delivery of actual server responses. They do not invent a success body, alter application source, force credential storage, or synthesize an authority event. The authority control clicks the real Disconnect button in a prepared Settings tab. That also backgrounds the inbox; the permanent component tests separately isolate owner/verification invalidation. These are controlled response-delay browser checks, not naturally occurring latency measurements.

The three persisted fixtures use the existing `CollectionsDatabase.create_user_notification` API, check that the resolved database is inside the owned profile, and record `fixture_only:true`. Their IDs3/4/5, titles and `/prompts` links match the native scripts and response requests. This does not test reminder scheduling. Mark-read completes at the backend before navigation/Disconnect; the contract being verified is cancellation of a stale navigation continuation, not rollback of a valid earlier write.

## Earlier harness attempts and limits

- `held.txt` records an earlier actual200 response for id1, but `attempt1-limit.txt` then reports `Route is already handled!`. This attempt is not counted as successful late-delivery acceptance. `delay-real-response.js` preserves the earlier harness.
- `bounded-control.txt` reports `ReferenceError: setTimeout is not defined` for the earlier `bounded-control.js`. It is not a product failure or a passing control.
- `bounded-control3.txt` independently records successful delivery and the final Notes URL. It does not rely on either failed attempt.
- Native negative observations cover approximately500ms after successful fulfillment. They are corroborated by permanent async ownership regressions; they are not an indefinite navigation observation.
- No clean-console claim: the authority receipt reports0 errors/1 warning, and other receipts reference console entries outside this retained set. Auth/Settings snapshots and portal diagnostics were excluded entirely, avoiding credential-bearing content.
- The signed-out screenshot still contains “Loading notifications...” beneath the sign-in message. This audit verifies stale navigation cancellation; it does not claim all inbox loading presentation is resolved.

## Automated evidence and source identity

The prior final independent review and logs in `../followup139-142/` record **150 tests/6 suites passing**, plus the original batched authority probes2/2 and earlier boundary probes3/3. Permanent actual-provider/route controls cover pending View across verification loss, owner switch, unmount, outage/recovery, server/principal A→B→A, credential removal/restoration, same configuration events and same-principal token rotation. The latter valid-authority controls preserve navigation. Those suites were read, not rerun in this artifact audit.

All five production/test file hashes in the frozen `cycle5-uat142-batched-owned-manifest.json` still match current files:

```text
37bba2f0e3c05fd09c981a9568f121ccf1d38fc6c7d65d77877669f0355fc141  NotificationLifecycleProvider.tsx
2c66d004e46d98c87523e57b35b3ff13577c75c15f142bb724540914f56262a9  NotificationsRoute.tsx
5ae30fd3f787474b3b5c63a69f9ff87ac9f0f517f5c6008e897ad6ccc6bc9206  notification-connectivity.integration.test.tsx
45d17393e1a4e56f6ec3e108a554cfcbea63e6f6703bed5f6f2fbe0a6643bdfc  notification-lifecycle-provider.test.tsx
f0810dffd4592d4619db4500eb06c668defe651e868e7ed1261cbb8f7f2a05af  notifications.test.tsx
```

AC1 is supported by permanent boundary regressions and native route/Disconnect negatives. AC2 is supported by the permanent valid-authority controls and native200/update/navigation positive. Both can be checked and the task marked Done within this bounded scope.

## Retention and security

`retention-manifest.json` enumerates the exact19 copied files and generated audit/security artifacts, source/retained SHA256 values and normalization policy. All copies are byte-identical. Known credentials from six local sources were read only for comparison;26 distinct values and credential patterns produced zero matches, without printing any values. Both screenshots were visually inspected and contain no visible credentials. Browser profiles, private logs and all Settings/auth snapshots are excluded. Bandit on the three copied fixture scripts reports zero findings/errors; full machine output is `bandit.json`.
