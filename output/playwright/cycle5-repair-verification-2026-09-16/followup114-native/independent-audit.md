# UAT114 independent native evidence audit

**Verdict: retained evidence supports the bounded native acceptance result.** Six same-origin application tabs remain usable, hidden tabs release notification streams, and the reactivated tab catches up to a notification created during its hidden interval. No remaining UAT114 failure is demonstrated by these receipts. The original connection-exhaustion explanation remains an inference.

Reviewed 2026-09-16; read-only artifact/source inspection and local parsing only. No browser/runtime/product changes or inference calls. This audit file is the sole reviewer output. Task: TASK-13260.54.

## Verified observations

- `visibility-control.txt` reads both ordinary blank pages in one command: index0 hidden, index1 visible. The retained measurement reads `document.visibilityState`; it does not override it or dispatch visibility events.
- `six-tabs-receipt.txt` contains one blank tab plus six actual `http://127.0.0.1:18580/` application tabs. Index6 is visible; indices0–5 are hidden. Its 406 network events are an exact prefix of the final receipt's 1,170 events.
- Stream18 belongs to tab1: HTTP200, then `net::ERR_ABORTED` at **20:39:22.450Z**. Tab1 has no subsequent notification request starts until reactivation at **20:42:52.719Z**. Tabs2–5 open no notification streams in the retained interval.
- Stream94 belongs to tab6: HTTP200 and remains open while tab6 is the visible tab. At **20:42:52.719Z** it aborts, coincident with tab1's resumed unread-count request356. The final visibility snapshot shows tab1 visible and tabs2–6 hidden.
- On tab1's return: unread-count356 returns200 in18ms; latest-notification360 returns200 in21ms; stream361 starts with **`after=1`** and receives200 in11ms. Stream361 is the only notification stream lacking a terminal event at the final receipt, **20:43:15.484Z**.
- Prompt project creation317 and prompt creation318 on tab6 each return **201 and finish in16ms**, at20:42:01Z, while the six application tabs remain open. This directly establishes ordinary write success for this run.
- `notification-created.json` records fixture notification **id1**, title **“UAT114 hidden-tab catch-up”**, created at **20:42:37.384693Z**, between tab1's hidden-stream abort and its reactivation. The reviewed fixture script uses `CollectionsDatabase.create_user_notification`, verifies the database is inside the owned runtime profile, and marks its receipt `fixture_only: true`. This validates a real persisted notification fixture; it does not validate reminder scheduling or another notification producer.
- Visually inspected both screenshots: `reactivated-catchup.png` shows bell badge1 and “1 unread notification”; `caught-up-inbox.png` shows Unread:1 and the exact matching notification title/message in the actual Notifications page.

## Boundaries and auxiliary observations

- The evidence demonstrates release/resume and successful ordinary requests. It does **not** measure browser socket limits, prove HTTP connection exhaustion as the original cause, or reproduce a pre-fix stalled run under this fresh browser configuration.
- Visibility is sampled at the retained snapshots, not logged continuously at every transition. Stream cancellation/resumed reads and final states corroborate the parent's tab-switch sequence. The observer in `install-monitor.js` subscribes to Playwright request/response/finished/failed events; it does not intercept requests or change visibility APIs.
- The final receipt retains an early unread-count **request1** start with no response or terminal event. Later unread-count requests succeed. This is an incomplete recorded request lifecycle, not sufficient evidence of a persistent stall; consequently the report claims only stream361 is the sole **notification stream** still open, not that every other request has a complete lifecycle.
- Auxiliary `/buddies`, `/buddies/attachment`, and health polling continue in hidden tabs. Post-reactivation hidden buddy requests in the retained receipt finish200. This bounds the claim to notification lifecycle work; it does not establish a new failure or total background-network silence.
- No full auth/account-isolation regression rerun was part of this receipt audit; the task's earlier preservation-test evidence remains separate. Screenshots establish rendered outcomes, not every intermediate transport response body.

## Source identity

Observed HEAD: `ac35a27da654c48b6ab3a020c0bbd44b52d7ec92`. The six working source/test files from frozen UAT159 all match `.tmp/uat159-first-run-auth-20260916/owned-manifest.json` (SHA256 `ea324859e8e271efd49ef576a5bfad7c7a51f8cd2ed40b1eb98a833ff5337bdb`). Its older recorded HEAD is manifest provenance, not the current audit HEAD.

No diff against HEAD in the inspected UAT114 provider, its lifecycle test, or shared notification service. Provider lines249 and416–425 gate hidden startup and stop/restart work on natural visibility changes. SHA256:

```text
37bba2f0e3c05fd09c981a9568f121ccf1d38fc6c7d65d77877669f0355fc141  apps/tldw-frontend/components/notifications/NotificationLifecycleProvider.tsx
45d17393e1a4e56f6ec3e108a554cfcbea63e6f6703bed5f6f2fbe0a6643bdfc  apps/tldw-frontend/__tests__/components/notification-lifecycle-provider.test.tsx
16f91a0b05767b9ab813fac6ae74b4b186e702aa2fd7288944d196d13e26f1d1  apps/packages/ui/src/services/notifications.ts
```

## Audited evidence hashes

SHA256; paths relative to this directory:

```text
1ed9b571f5dd8f0621e3ab004a45ddebcb407812c686fd043eb615ff7dd6966d  visibility-control.txt
ad12c6653138ba60b5759aa2288d1050e49bd7d6485ef4eec8abd9b4f6318e2b  six-tabs-receipt.txt
b8d1f1a5fd178857b1c88d4f41c4c83667a7176b149d833c3ebe124848d79b54  catchup-final-receipt.txt
85fdcec61e41751ba13be125eb5a3abe18412837ef3a13d068c9234b367fd0cb  notification-created.json
2bef21f2dabe9e47c6ccce94b153c15cb2991545df74cb476fa85c90144bc18e  reactivated-catchup.png
7bcd2a82098094d694bbe50ee61b1e53f61cd815befa9dac8f3ffb54fc508878  caught-up-inbox.png
2ab49ba1b72197ed6c56593fe80b4a71f51eadb88b43fcaaca0036e0be325fb4  install-monitor.js
4c9482fa207cb949fd36f8cf0c2bf93b080b10e82d70a666b4de6d731655ff95  create-notification.py
```
