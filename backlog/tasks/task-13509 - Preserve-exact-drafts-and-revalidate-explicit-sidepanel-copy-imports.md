---
id: TASK-13509
title: Preserve exact drafts and revalidate explicit sidepanel copy imports
status: Done
labels:
- chat
- ux
- handoff
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Bounded Stage4 receiver correction under approved master: preserve all unsent whitespace, keep a whitespace-only destination draft in conflict review, reread the existing short-lived package before explicit Insert or Replace, and prevent a delayed choice from applying after route departure/unmount or overwriting text edited during its read. Reuse existing package reader and conflict UI; no transfer fields or server/account authority added.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Insert preserves both source and current destination text exactly, including whitespace-only drafts; nonempty destination content requires explicit review before consumption.
- [x] #2 Insert and Replace revalidate existing package freshness and retire delayed work on route departure/unmount; a changed destination draft remains intact and reviewable.
- [x] #3 Rapid activation issues one acceptance read; pending actions stay understandable; existing context, submission and route regressions pass.
- [x] #4 Relevant checks, shared types, extension compile, no-new lint and both independent reviews are recorded; owner/provenance and native full journeys remain explicit parent gates.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ParentTASK13457. Master Docs/Design/2026-10-04-chat-conversation-ux-remediation-design.md, bounded original receiver contract. Start85f4527bee; dev94854ca current and integrated292ea526d1. Official dedupe searches for handoff draft import and sidepanel import returned none; broad handoff search inspected prior Done units and active parent continuity work. Existing Media acceptance flow is an inspected pattern. ADR required:no, existing fields/reader/conflict actions retained. Bandit N/A TS/docs unless scope changes. Plan Docs/Plans/IMPLEMENTATION_PLAN_TASK_13509_chat_copy_import.md. Source/destination account ownership, unknown provenance, cold draft readiness and cross-origin URL-fragment authority stay in separate parent continuity work.
Completed TASK13509: exact whitespace and whitespace-only conflict, fresh explicit Insert/Replace, one pending choice with Cancel, route/hash/unmount retirement, real form revision detecting batched edit/Undo, and retryable actual-reader storage errors through opt-in throwOnReadError. Initial/default reader contract retained; no transfer/authority change. Four review checks fail before correction and 46 relevant checks pass after; final13-suite158 checks pass. Logs /private/tmp/chat-copy-import-review-red.log, /private/tmp/chat-copy-import-review-green.log, /private/tmp/chat-copy-import-final-regressions.log, /private/tmp/chat-copy-import-final-types.log, /private/tmp/chat-copy-import-final-extension.log, /private/tmp/chat-copy-import-final-lint.log. WebUI types, extension entrypoint/config compile, no added lint, English four-key official locale sync/mirror and diff pass. Both independent source reviews clear; reviewers did not execute checks. Bandit N/A TS/TSX/JSON/docs. Master checkpoint updated; own completed plan removed. PR3147 persistence and PR3139 fresh send readiness inspected as independent overlap references, not integrated. Dev94854ca unchanged and included292ea526d1. Cold readiness, initial auto-import, ownership/provenance, source retirement, cross-origin authority and native/browser/full program gates remain parent work. Actual hook/router/useSimpleForm/reader use declared storage IO doubles; no full native composition/durability claim.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Exact drafts survive explicit copy conflict resolution; held choices cannot overwrite edits/Undo or a retired route. A storage read failure retains Retry and the current draft/URL. Final158 relevant checks and all bounded client gates pass; both independent reviews clear. Parent account/readiness/native/continuity gates remain open.
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
