---
id: TASK-13553
title: Keep edited message text until its current save is acknowledged
status: Done
labels:
- frontend
- chat
- ux
- bug
dependencies:
- TASK-13549
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Repair the shared editor acknowledgement root identified during CM01 saved action review. EditMessageForm closes before awaiting Save or Save & Send; multiple actual parent callbacks discard the Promise. Preserve exact edited text on rejection or skipped/retired action and prevent an old save from closing newer input or another editor. This is a bounded prerequisite for accepted native graph saved-action support, not that feature implementation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Save and Save & Send retain the editor and exact text while pending or after rejection/unconfirmed outcome; only the current successful operation closes it.
- [x] #2 Late outcomes cannot close or overwrite newer text, edit/undo revisions, cancelled/reopened editors, replaced message identity or unmounted editors; duplicate dispatch and composing Enter remain controlled.
- [x] #3 All shared callers propagate actual operation outcomes; real Playground and compact editor failure paths preserve authored text and canonical saved rows.
- [x] #4 Both client types, affected positive tests, authored lint/format, two independent source reviews and honest evidence/task/design records pass.
- [x] #5 Editing or deleting visible rows keeps undispatched local RAG diagnostics out of provider history using the existing eligible-row projection; unsupported resend remains a verified refusal until native branch support.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR required:no. Existing ADR049/059, accepted D7 CM01 design and all-ten remediation govern. This is UI acknowledgement/outcome propagation, not a new write protocol or persistence authority. Native regeneration/branch admission remains separate required work. Latest fetched dev97ea9cd is included in committed0d145a7e5a. Unique ID checked across fresh origin/dev, all52openPR files and both local backlogs; searched existing editor tasks.
Red baseline:16 editor cases,14 fail and2 controls pass; three old mutation tests fail because a removed mock API and unsupported legacy resend assumptions precede two genuine eligible-history projection assertions. Add the existing exclusion helper at the shared edit/delete history projection only after those assertions are reproduced with current I/O fixtures; no new protocol or history truncation.
Final root repair and limits are recorded in Docs/Design/2026-10-04-chat-conversation-ux-remediation-design.md, TASK13553 checkpoint. Positive final matrix886/17, actual idle hook subset309/3, both types0, authored21-file lint0errors/0added with21baseline warnings, owned formatting, exact English mirrors and diff pass. Two independent source reviewers clear; delayed ACK ABA counterexample fixed before qualification. Exact disposable backend readback equals prior2rows. Initial native failure was a mixed stale preview callback; rebuilt chunk/source map returns actual Promise, but retained IAB data-URL error page blocks fresh native confirmation. This limitation stays open; no native success or new window/tab claim. Bandit N/A for TS/TSX/JSON/Markdown-only authored scope. All4 bounded plan stages complete; own completed plan retired with final copy /private/tmp/IMPLEMENTATION_PLAN_chat_edit_ack_13553_completed.md. Saved native Save/Regenerate/Edit-and-Send, complete D7/scoped consumption/roleplay/Syncv2 and native/package/physical/AT/participant gates remain required.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Repaired shared editor acknowledgement, exact draft retention and every reachable result-propagating caller; current H1/draft fences and live stable-row merge prevent retired saves from publishing over a newer view. Existing history projection excludes local diagnostics. Qualified886positive checks, both client types, authored static/locale/format and two source reviews. Fresh native editor outcome remains explicitly unverified; all-ten parent remains In Progress.
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
