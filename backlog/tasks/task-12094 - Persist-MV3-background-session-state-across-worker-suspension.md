---
id: TASK-12094
title: Persist MV3 background session state across worker suspension
status: Done
labels:
- bug
- high
- extension
- mv3
- packages-ui
documentation:
- apps/FRONTEND_AUDIT.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
**Severity: High (feature breaks under normal MV3 lifecycle).** From the 2026-07-02 frontend audit (finding H2), independently flagged by two reviewers.

Chrome suspends an idle Manifest V3 service worker (~30s). `apps/packages/ui/src/entries/background.ts` keeps critical state only in `main()`-closure Maps with no `chrome.storage.session` rehydration, and drives long-running polls from detached `setTimeout` loops (up to ~10 min, `:1716`) that give Chrome no reason to keep the worker alive:

- `ingestSessions` (`:444`) — context-menu "Send to tldw" ingest. On suspend mid-poll, an ingest the server completed never emits `media-ingest-ready` → sidepanel stuck on "Queued for processing"; `cancel`/`retry` then return "Ingest session not found" (`:1490,1792`) → permanently unrecoverable from the UI.
- `pendingAuthReplay` (`:445`) + `queueAuthRecovery` (`:695-706`) — the user-facing "ingest will retry automatically" after a 401 never fires because the replay set is empty on wake (`replayPendingAuthSessions:1805`).
- `quickIngestModalSessions` (`:446`, incl. abort controllers) — quick-ingest batch orphaned; progress UI hangs; cancel can't find the session to abort in-flight uploads.

The model-warmup path already uses `chrome.alarms` correctly (`shared/background-init.ts:54`) — use it as the template.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Ingest, quick-ingest, and pending-auth-replay session state is persisted (e.g. `chrome.storage.session` or `local`) and rehydrated on `onStartup`/worker restart.
- [ ] #2 Long-running polls are driven by `chrome.alarms` (or an equivalent worker-survivable mechanism) rather than detached `setTimeout` loops.
- [ ] #3 After a simulated worker suspension mid-ingest, the session is recoverable: status/ready messages still deliver, and cancel/retry find the session.
- [ ] #4 The 401 "will retry automatically" path actually replays the queued ingest after credentials are updated, across a worker restart.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
