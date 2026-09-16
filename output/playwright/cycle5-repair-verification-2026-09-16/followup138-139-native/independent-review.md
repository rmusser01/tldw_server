# Independent UAT138 / UAT139 native audit

**PASS for both targeted native outage/recovery checks.** Media shows contextual outage feedback without an observed runtime overlay or captured page error, then returns the original Rowan item/content. Notifications uses reconnecting guidance during the outage and recovers to active without authentication reentry. This supplies the pending native evidence for TASK-13260.77/.78 alongside their previously retained regression/review results; no suites were rerun here.

## Real outage and source identity

- api-stopped.json records SIGTERM of owned sqlite-single API PID25020 at **2026-09-16T22:16:16.753Z**.
- source-and-restoration.json records the existing recovery launcher restarting the same profile at **22:21:05.864Z**, PID72891, exec session83022. runtime-recovered.json reports **API200/WebUI200**, expected configured Gemma advertised and no inference requested.
- Recorded source HEAD is `588d6c3cef7bb29ec9a4fef9f726b65f9330fb5f`, before the separate UAT165 candidate. The following current source hashes independently match that recorded commit and the prior relevant reviewed freezes:

| Source | SHA256 |
| --- | --- |
| apps/packages/ui/src/components/Review/hooks/useMediaSearch.ts | `316dc3845a374d6e169d74cd1d95e5a96345eac7668edf6c801523721c7d7f86` |
| apps/tldw-frontend/components/notifications/NotificationsRoute.tsx | `2c66d004e46d98c87523e57b35b3ff13577c75c15f142bb724540914f56262a9` |
| apps/tldw-frontend/components/notifications/NotificationLifecycleProvider.tsx | `37bba2f0e3c05fd09c981a9568f121ccf1d38fc6c7d65d77877669f0355fc141` |

This is scoped source verification, not a claim that the current entire worktree is clean. No runtime action was performed by this reviewer; shutdown/restart ownership and command details come from the retained parent receipts.

## UAT138 — Media

media-offline-before.txt, media-offline-fresh.txt, media-offline-events.txt and the independently inspected media-offline.png show the normal application shell with:

> Can't reach your tldw server right now

The supporting text says the server settings are saved but Media cannot reach the server. Retry connection and Health & diagnostics remain available. The offline main-text capture at **22:20:46.266** contains this contextual state and no runtime-error overlay.

The Media monitor records **zero pageerror events** through its final capture. Expected transport failure logs and proxy HTTP500 responses are present, including the fixed “Media search request failed.” warning. The screenshot shows no Next runtime overlay. This supports the bounded no-overlay/no-captured-unhandled-error acceptance; it is expressly not a clean-console claim.

**Harness limitation:** media-retry-offline.txt records `Ref e344 not found in the current page snapshot`. That click was not applied and is not credited as successful Retry-button coverage. Media recovered automatically after the backend returned; no successful manual Media Retry is needed to explain the observed recovery.

Genuine Media response bodies become HTTP200 at **22:21:17.517** (capabilities), with list reads at **22:21:17.525/.572** returning the original **media1, rowan-community-library**, document/completed, created **21:22:07.154**, updated **21:28:46.992**. media-restored.txt shows that card without a new login or reload action in the recorded sequence.

The subsequent real card click in media-selected-restored.txt navigates to `/media?id=1`. GET `/api/v1/media/1` returns200 at **22:22:12.432**, and the final snapshot at22:22:13.605 shows its original content. The returned text exactly equals the retained original Rowan input after removing its trailing file newline: Mara Chen, 7 December2026, east entrance, Friday19:30, plus the distinct Cedar facts. The independently inspected media-recovered.png visibly confirms the selected card, chapter and full source content. The data was not recreated to recover the view.

## UAT139 — Notifications

The notification observer also includes older UAT141 route-abort entries. This audit isolates entries **after the actual22:16:16.753 shutdown**; those earlier synthetic faults are not counted as this outage.

Actual unread-count responses return proxy HTTP500 at **22:16:24.847**,22:16:54.854 and22:17:24.855. notifications-offline.txt/png and the22:21:02.311 main-text receipt show **“Notifications are reconnecting”**, retained notification rows, and automatic-recovery guidance. They do not ask the configured user to sign in. notifications-try-again-offline.txt records the real Try again click during the outage.

After the same profile restarts, genuine response bodies arrive at **22:21:15.816** (unread count), **22:21:15.818** (active list5 and snoozed list0), and22:21:15.848 (probe). A notification-stream HTTP200 follows at22:21:15.894. notifications-restored.txt shows **“Notifications are active”** before the later manual Refresh. The actual Refresh recorded in notifications-refresh-restored.txt produces further list200 responses at **22:21:57.590/.631**; final main text at22:21:58.775 remains active with the same five fixtures.

Recovery requires no recorded auth or Settings action. The parent explicitly confirms the same saved credentials and no auth reentry/reload; credentials were not exposed or re-read by this audit. Genuine missing/invalid-auth sign-in behavior was not newly exercised here and remains covered by the prior permanent UAT139 controls.

## Limits

- Expected failed-resource errors and request warnings occur throughout the real outage; Media's observer includes unrelated buddy/storage/profile network failures. None should be relabeled as a clean console. No other console-message category was found in that captured Media set.
- Media's installed monitor listens for pageerror; the reused notification monitor primarily records notification responses, so absence of notification pageerror entries alone is not equivalent to dedicated notification error instrumentation.
- Screenshots and captured page text establish visible states at their recorded checkpoints, not continuous video. The recovered Media item was deliberately selected again; uninterrupted preservation of a previously open inspector selection is not claimed.
- Stream HTTP200 is observed, not raw SSE frames or a new event-delivery test. No model inference, browser/service action, product/task edit or git mutation was performed by this audit. Only this requested report was written.

## Key original evidence SHA256

| File | SHA256 |
| --- | --- |
| api-stopped.json | `2fad09ab4f56c2d448614fdca5fed0d4304cedf90de65108be7a3246519cd28b` |
| runtime-recovered.json | `c393e5f6714f640902e55a7672d11d6c5afa86bb1edfd678881a198c6375c4bf` |
| source-and-restoration.json | `5f533025a27be4789eb17c16bff2ed001cde674f642e3477a8de788c9b23bb43` |
| media-final-events.txt | `b134a9a4d64574311d76aa37e696fe44f747e40ddc2e4f9f82ce06e10aaa5864` |
| notifications-final-events.txt | `c4daa63d0a91cd6284519c04a385166cb60026252f09b237994bf86b7167a89f` |
| media-offline.png | `e47870f1d90c92ea67e96047a031651691dde6189d2d802f0ad3580eccf3f7c2` |
| notifications-offline.png | `03a4f5e9c2ecbf09e8e8653af296f7903365f79f318aed9936f335ca50a06fd0` |
| media-recovered.png | `c7cfb546c16f970c0b020af7646a27edbd7de7f8d2a191db3cb56584496240f3` |
| media-retry-offline.txt | `b583df31486c2e081b133e9b173554518059e57bf3067f6758f18c453adf8d37` |
