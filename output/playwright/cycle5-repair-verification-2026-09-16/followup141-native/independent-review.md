# UAT141 independent native audit

**PASS for both native timestamp acceptance cases in TASK-13260.80.** Cold failed reads show no fabricated last-success label; later failed reads retain a timestamp consistent with the previous genuine successful inbox read. This complements the retained permanent regression/review evidence; no test suite was rerun here.

## Scope and source

Read-only inspection of the recorded Playwright actions, final41-event receipt, UI snapshots and warm-failed.png. Parent records live source as HEAD `ec28c34b7c` plus separately frozen UAT164 guidance changes; this is not claimed to be a clean full-HEAD run. Current NotificationsRoute.tsx SHA256 is `2c66d004e46d98c87523e57b35b3ff13577c75c15f142bb724540914f56262a9`, exactly matching the prior final notification freeze in followup139-142/cycle5-uat142-batched-owned-manifest.json.

The actual cold-control.js installs an abort route for `**/api/v1/notifications**` **before** navigating to /notifications. It records real aborts/responses and uses the real UI. Subsequent scripts remove/reapply that route and click actual Refresh/Try again. No fabricated successful response, product-state injection, clock replacement, or application timestamp mutation appears in the inspected actions. Test phase/event variables are observer bookkeeping only.

## AC1 — cold failure

- Notification reads begin aborting at **22:09:15.673 UTC**. Both full active and snoozed reads abort, along with lifecycle probes and retries.
- cold-and-recovery.txt captures the cold page at **22:10:07.289**, after16 aborted reads and **zero successful notification responses** in that phase.
- Its actual main text, also supported by cold-settled.txt, shows “Notifications are reconnecting”, “Failed to fetch” and “No notifications yet.” It contains **no “Last updated” label**.
- The observation therefore supports cold failure without inventing a successful inbox-read time. It does not depend on interpreting a loading-only snapshot.

## AC2 — success followed by later failure

Removing the fault and clicking real Refresh produces real HTTP200 bodies:

| Read | Recorded UTC time | Result |
| --- | --- | --- |
| Snoozed inbox, limit100 | 22:10:07.315 | zero items |
| Active inbox, limit100 | 22:10:07.317 | five items, IDs5/4/3/2/1 |

warm-success.txt then shows the populated list with “Last updated before the connection was lost (Just now).” The notification lifecycle is still reconnecting at this point; successful route list reads are nevertheless real, making this a legitimate success timestamp.

The notification abort route is reapplied at **22:10:13.857** (warm-restored-events.txt). There are no intervening successful notification reads before the later failed Refresh. Background probes abort at22:10:15,22:10:45 and22:11:15.

warm-failed-refresh.txt records an actual Refresh click and both full list requests aborting at **22:11:26.666**. The captured main text still says:

> Last updated before the connection was lost (1 minute ago).

The elapsed interval from the active-list successful body receipt to this capture is **79,349ms**. This supports the rounded “1 minute ago” display; advancing the timestamp to this failed attempt would instead make it newly fresh. Additional route read retries remain aborted through22:11:44.691; warm-failed.txt and the independently inspected warm-failed.png retain the same one-minute label, reconnecting/error feedback and all five prior notifications. The screenshot clearly shows “Failed to fetch” with the retained list, rather than an empty or newly successful state.

## Recovery control

final-recovery-start.txt removes the fault and clicks real Try again and Refresh. final-events.txt records genuine HTTP200 unread-count and probe responses at22:11:54.829/.841, a stream response at22:11:54.866, and both full inbox responses at22:11:54.874 (five active, zero snoozed). final-recovered.txt shows “Notifications are active”, the same populated list, and no stale-connection timestamp or error message. This proves the controlled outage was removed and the UI recovered.

## Limits

- Observer response timestamps are recorded after response-body reading (except the stream); they are evidence of successful reads, not an exact inspection of the application's internal inboxLoadedAt value. The79-second interval and minute-level UI independently support the required freshness behavior without claiming millisecond equality.
- This is a notification-only transport-abort control. It does not establish whole-server outage behavior or accept UAT138/139 by itself.
- Stream HTTP200 is recorded, but no raw SSE frames or event-delivery guarantee is claimed.
- The parent identifies the cold page as newly opened from about:blank. The inspected cold script independently establishes interception before navigation and absence of successful notification responses; the packet is not a continuous recording of all browser activity.
- Existing synthetic notification fixtures are real returned API rows; no notification creation or other external mutation was performed by this audit. Only this requested private report was written. No browser, service, product, tracker, or git action was performed.

## Exact original evidence SHA256

| File | SHA256 |
| --- | --- |
| cold-control.js | `e5036269209c3394f030e9bb999e4a0d939269de089daeb05b5d3ad342b7d9f2` |
| cold-settled.txt | `be2ba3e7378262c7b22f9499e3db98c6754e38209c751e8bc11271d84b129d0c` |
| cold-and-recovery.txt | `d71be6df2315feec9f4d72715c02e4a5614b0ff77b6b552922fe247a9afa28b6` |
| warm-restored-events.txt | `c28d69977be88b9ed1e17084ac047a5a9334304885612923ddd5ade18977099b` |
| warm-failed-refresh.txt | `8b64531a89767ff7e2a252438a6af83750c899b78b4989eddc733e633831d286` |
| warm-failed.txt | `b930212dd07817f9321231656036a87457a0389d8d781d9f769315dba290ae5a` |
| warm-failed.png | `b14f854196e89242917558b51fbd8e03f119a568d840dc2e4655897174e6f8fc` |
| final-events.txt | `7b09c25b7ad6668e13c234f6659e4f930d147426cb5e923d87d7740411981ad0` |
| final-recovered.txt | `42599ffaef1827d1857856ca50535fd5d35a3f3f39800ac0456a88990498f882` |
