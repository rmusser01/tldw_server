---
id: TASK-13183
title: Restore Buddy live-stream readiness after Strict Mode remount
status: Done
assignee:
  - '@codex'
created_date: '2026-09-05 14:55'
updated_date: '2026-09-05 16:36'
labels: []
dependencies: []
references:
  - Docs/Reviews/MIGU_BUDDY_UAT_2026_09_05.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Live Migu UAT: Start succeeds, but Send repeatedly reports Persona live stream failed to connect and no WebSocket is created. usePersonaLiveControl cleanup sets mountedRef false and never resets it on effect setup; the WebUI enables React Strict Mode. A direct authenticated backend WebSocket connects and returns a tool_plan.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The real development WebUI can send Buddy text after Strict Mode effect remounts.
- [x] #2 Actual unmount still cancels pending handshakes and releases sockets; failed sends retain the unsent draft.
- [x] #3 Stale startup or overlapping list responses cannot overwrite a newer started/focused session or the latest loading/error state.
- [x] #4 A successful persistent session Start resolving after unmount returns the backend result without stale UI writes or terminating a reusable user session; stale sends remain cancelled.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce the missing WebSocket in a StrictMode hook regression and cover cancellation around pending session/config/handshake work.
2. Restore mount readiness on effect setup and fence asynchronous sends/connections by mount generation while retaining socket and timer cleanup.
3. Reproduce and fence stale startup/overlapping list responses by mount generation and request order, preserving newer Start/Focus results.
4. Run the focused hook Vitest tests and touched-file checks, record red/green evidence; parent will perform real WebUI UAT.
5. Qodo review follow-up: inspect backend create/resume and ownership contract, reproduce successful pending Start incorrectly rejecting on unmount, preserve its successful persistent result without stale UI writes, and cover shared-session StrictMode resume without destructive cleanup.
ADR required: no
ADR path: N/A
Reason: routine bug fix preserving existing persistent session ownership, resume, hook lifecycle and stream contracts.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Restored mounted readiness on every lifecycle setup. A mount-generation fence cancels session creation, configuration lookup, and send continuations from a discarded mount; stale connection rejections cannot clear a newer connection promise or publish its error. Actual cleanup still rejects pending handshakes, clears timers, detaches handlers, and closes sockets.

Changed apps/packages/ui/src/hooks/usePersonaLiveControl.tsx and its focused hook test file. No ADR required: routine preservation of the existing lifecycle/stream contract.

Red evidence: original hook produced zero sockets in the StrictMode send case (2 failed, 13 passed). Readiness-only fix exposed a discarded session replacing sess-current with sess-discarded (1 failed, 14 passed). Final focused Vitest: 16 passed, covering StrictMode send, stale session completion, configuration resolving after unmount, connection timeout/error, pending handshake cleanup, open-then-unmount before send, retry identifiers, and open-socket release. Touched-file ESLint: no findings; root invocation reports a pages-directory configuration notice. git diff --check clean. Modified test sections formatted using shared UI conventions; unrelated formatter churn removed. Bandit not applicable to TS/TSX-only scope. Existing Node localStorage experimental warning remains.

Real development-WebUI UAT and final task completion remain with coordinating agent; no full suite or commit performed.

Review follow-up: reproduced the remaining startup/list race before changing production code. Five regressions failed: a discarded StrictMode list erased a new Start; earlier same-mount lists erased Start/Focus; older list success/failure ended the latest request loading state. Reload now fences sessions/focus/error/loading writes by mount generation and request sequence. Successful Start/Focus invalidates earlier snapshots and clears their obsolete loading state. Final focused hook suite: 21 passed; touched-file ESLint has no findings (existing root pages-directory notice only); git diff --check clean. Coordinating agent notified that final browser UAT can proceed. This follow-up changes only the frontend hook/tests and task record.

Coordinated final validation: 265 focused frontend tests, 54 backend tests, production Bandit0 findings, scoped frontend ESLint0 errors (warnings documented), unchanged Python lint baseline, real browser evidence and limitations recorded in Docs/Reviews/MIGU_BUDDY_UAT_2026_09_05.md. Repository-wide typechecking remains limited by80 diagnostics across6 unchanged unrelated files; no full suite run.

Qodo PR2884 follow-up: Start now returns the successful backend session result even if the requesting mount has ended, while guarding only local focus/state writes. sendText retains its independent generation fence and cancels stale sends. Automatic stop was deliberately rejected: create_or_resume_live_session can return an existing nonterminal user-owned session; its response has no exclusive mount ownership or created/resumed provenance, and its idempotency key is overwritten even when resuming. Closing that session could terminate work resumed by another mount/client. Persistent sessions remain listable/resumable across component lifetimes.

Red: the new pending-Start/unmount regression failed with promise rejected STREAM_CONNECT_ERROR (1 failed, 22 passed). Green: bun x vitest run ../packages/ui/src/hooks/__tests__/usePersonaLiveControl.test.tsx from apps/tldw-frontend passed 23 tests, including discarded StrictMode send returning either a different session or the same session used by the current mount. Existing backend contract tests create_resume_compatible_reuses_active_session and stop_marks_closed_and_clears_focus passed (2 passed, 39 deselected). Scoped ESLint run from repository root with apps/tldw-frontend/eslint.config.mjs passed without findings; existing root pages-directory configuration notice remains. An initial frontend-directory ESLint invocation ignored shared files; it was replaced by the effective root run. git diff --check clean. Bandit not applicable to this TS/TSX-only follow-up. No additional browser UAT or full suite run; no backend contract changes. ADR not required: preserves existing persistent ownership/resume behavior.
Rebase provenance: originally TASK-13177, created before review. Latest dev independently allocated13177 to the Docs Design recovery task in PR2885. Buddy record renumbered to13183 during the already-authorized task collision closeout; the dev record is preserved.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Strict Mode readiness and stale session-list races repaired. Final clean Chromium page sends Buddy text and receives the real backend plan. Lifecycle/retry regressions pass; visible outcome handling remains separate TASK-13180.

Qodo lifecycle follow-up preserves successful Start outcomes after unmount and shared persistent sessions while keeping stale sends cancelled; 23 focused hook tests and 2 backend ownership contract tests passed.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
