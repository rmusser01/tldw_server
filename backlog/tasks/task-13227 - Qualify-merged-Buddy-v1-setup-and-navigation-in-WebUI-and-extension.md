---
id: TASK-13227
title: Qualify merged Buddy v1 setup and navigation in WebUI and extension
status: In Progress
created_date: 2026-09-09 04:32
assignee:
- '@codex'
priority: high
references:
- TASK-13226
- TASK-13211
- TASK-13202
documentation:
- Docs/superpowers/plans/2026-09-09-buddy-v1-live-qualification.md
updated_date: 2026-09-09 05:41
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Verify the merged independent Buddy and Persona user journeys with disposable fresh and upgraded profiles in the real WebUI and packaged extension, and revisit the recorded visual-load loop on current dev.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Fresh and upgraded disposable profiles exercise independent Buddy selection, explicit conversation/workspace attachment, Persona defaults, and Static/Dynamic presentation through the real user interface.
- [x] #2 A conversation reply remains correctly targeted across navigation and the workspace inbox preserves explicit reply scope and queued result semantics.
- [x] #3 The recorded artwork-loading incident is investigated on current dev with request/lifecycle evidence and its outcome is reconciled with TASK-13211 without inventing a fix.
- [x] #4 Source revisions, runtime/profile isolation, screenshots, exact checks and remaining human voice limitations are recorded; any reproduced defects receive scoped regression verification.
- [x] #5 The model catalog remains available with the shipped blank optional max-token limits, preserving unset limits and valid configured values.
- [x] #6 Saved chat failures are presented as readable summary/hint text in the Buddy conversation instead of exposing the internal encoded error envelope.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. ADR path: backlog/decisions/005-independent-buddy-bindings-and-work-ownership.md. Reason: qualification of existing behavior and routine repairs only. 1. Pin current dev and launch an isolated authenticated server plus WebUI with fresh data; record provenance and resolved storage. 2. Drive first-use Buddy/Persona setup, attachment, motion settings, navigation and scoped replies through the real UI; qualify the packaged extension where available. 3. Exercise repeated state/navigation cycles and record bounded artwork/session requests to investigate TASK-13211; add a focused failing regression only for a reproduced mechanism. 4. Verify upgrade behavior using disposable prior-schema fixtures and existing targeted tests; preserve screenshots/request receipts and report precise coverage limits. No physical microphone capture without a coordinated user start; no full local sweep.
5. Reproduced first-use blocker: get_configured_providers int-converts shipped blank *_max_tokens options and swallows the ValueError into an empty catalog. Add a failing focused test, preserve optional unset semantics, and verify catalog plus real Buddy replies with the original disposable configuration.
6. Reproduced Buddy transcript presentation defect: a controlled standard Chat timeout saves the encoded error envelope; Buddy history renders its raw JSON. Reuse existing chat error decoding for readable summary/hint, preserve ordinary transcript text, and add a focused regression.
7. Independent review found a second sink for the same saved error: read-aloud queues raw persisted assistant content. Share assistant summary/hint presentation between transcript and speech, preserve the conversation-name prefix and ordinary/user text, and add a failing activity-to-speech regression before rebuilding the corrected Chrome artifact.
<!-- SECTION:PLAN:END -->
## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Fresh WebUI selected Pixel Migu without Persona, attached New Research workspace, retained the visible Buddy on Watchlists, opened scoped interaction there, saved Static mode, and verified Cancel preserves None before saving Research Assistant for future conversations. Provider catalog failure traced to blank optional *_max_tokens in shipped defaults; scoped repair added to acceptance before implementation.
Qualification reconciled in Docs/Reviews/2026-09-09-buddy-v1-qualification.md with curated receipts/source hashes. Fresh WebUI: independent Pixel Migu, None/Research Assistant workspace defaults and inheritance, Static/Dynamic artwork, exact conversation Buddy reply completed after Watchlists-to-Workspace navigation, selected workspace conversation increased from 5 to 7 messages while sibling stayed at 5, and one acknowledged result left the sibling unread. Original blank-config catalog defect repaired with 5 focused/adjacent tests; assistant-only readable error rendering repaired with 8 component/decoder tests and quoted-user-envelope preservation. Backend 51 passed/1 unavailable-PostgreSQL skip; frontend artwork/lifecycle 81 passed. Bandit and touched regression formatting passed; unchanged source lint debt recorded. Final Chrome production build, shared-token sync, manifest targets and ZIP integrity passed. TASK-13211 investigation reconciled without speculative repair. AC2 covers typed reply and result receipt/acknowledgement semantics; real audible queue playback remains unqualified. AC1 remains open: native Chrome permissions unavailable, Terminal explicitly prohibited by Computer Use, upgraded-profile WebUI journey and native Chatbook interaction incomplete. No full suite or human-voice pass claimed. ADR-005 remains the governing contract.
Before finalization, independent review identified that a newly arriving saved assistant error would still be read aloud as its internal JSON envelope. AC6 reopened until transcript and speech share the assistant-only presentation and the activity-to-speech regression passes. Previously retained Chrome artifact qualifies the visual-only fix; a new build is required for this final correction.
Final review correction complete: private assistant-only formatter is shared by visual history and read-aloud after its authorized transcript read, preserving the conversation-name prefix. Activity-to-speech regression failed first with raw JSON/detail, then final component/decoder gate passed9/9; independent re-review has no remaining findings. Final Chrome build includes this source and passed in43s; ZIP SHA2561375a51f13cb851ca3e12c277b0da2b7e6da19f510038382fe3c6d6c778048e8. Final catalog gate passed5/5 and Bandit reported no findings; raw final logs and curated manifest retained. Original catalog RED and81-test artwork raw logs were not separately retained, so their worker summaries are labeled accordingly. Owned disposable WebUI/backend/mock processes stopped; profiles and artifacts retained. AC1/native acceptance remains open.
Published draft PR https://github.com/rmusser01/tldw_server/pull/2934 against dev with the reviewed fixes and qualified evidence. Native/upgrade acceptance and the human-written summary gate for any future merge remain open; no merge performed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
