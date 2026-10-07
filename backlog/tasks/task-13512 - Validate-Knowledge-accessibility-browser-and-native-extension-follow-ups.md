---
id: TASK-13512
title: Validate Knowledge accessibility browser and native extension follow-ups
status: In Progress
labels:
- ux
- accessibility
- knowledge-followup
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Extend PR3196 evidence with available Safari, screen-reader and actual extension-control checks across first-use, multi-source review and saved-content workflows. Prepare real participant and mobile-device sessions; record unavailable hardware and participants honestly.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Exercise available real browser and native extension controls with retained evidence
- [ ] #2 Perform available screen-reader checks and fix observed accessibility issues
- [x] #3 Prepare first-time and power-user task protocol and explicitly record participant/device coverage
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Related TASK-13453 and Docs/Reviews/KNOWLEDGE_UX_REMEDIATION_2026_10_04.md. User availability for human testers and iOS hardware requested asynchronously. ADR required:no; validation and bounded UI fixes use existing platform conventions.
Native Chrome for Testing exposes real extension toolbar/context-menu capture actions. Save to Clipper did not open its sidebar. Traced launchWebClipperFromContextMenu: it awaits capture before requesting the gesture-restricted Chrome sidebar; openSidepanel also ignores the returned open Promise. Use the existing helper, open during the original context-menu callback, and observe acknowledgment/failure. Verify ordering and native capture on the final bundle; do not substitute a simulated onClicked event.
Available desktop browser and native menu controls exercised. Original actual Chrome menu capture failure reproduced; shared gesture/acknowledgment fix and all sibling callers have regression coverage. Final frontend affected set: 152 passing tests; both clients types/builds pass. Participant protocol and detailed device/VoiceOver/native-final limits retained in Docs/Reviews/KNOWLEDGE_FOLLOWUP_RESULTS_2026_10_06.md. No participant, iOS keyboard or spoken VoiceOver session was completed.
Published bounded follow-up fixes/evidence as draft PR3205 against latest dev: https://github.com/rmusser01/tldw_server/pull/3205 . The separate human Change summary, unresolved device/participant qualifications, independent provenance design and existing CI integration remain explicit in the PR/report. Disposable task runtimes/browser profiles are stopped; managed worktree retained for review.
Native follow-up attempted in a newly isolated Chrome profile through cua-driver launch_app with focus guard. Owned article window was available on current Space, but branded Chrome did not load the flag-supplied extension (manifest expects background.js, debug targets had only built-in service workers). A second profile-bound new-window launch opened New Tab and left the owned browser active despite transient launch response active:false. AXShowMenu reported dispatch but post-snapshot showed no menu. Stopped this path after these three failures and closed only the owned Chrome instance/fixture. No final native capture success, VoiceOver, mobile-device or participant qualification claimed. Headless WebUI canonical history verification continues separately; foreground permission remains pending.
Cleanup verification: cmd+Q was acknowledged by the driver but windows remained, so stopped only the two recorded isolated-profile browser PIDs after verifying their exact --user-data-dir argument; no primary/user Chrome process was targeted. The owned native fixture exited. The background-driver native-menu result remains unqualified; test setup/driver behavior is not asserted as a Knowledge product defect.
Canonical source-history browser checks additionally proved native Tab focus/name and inert retained URLs, explicit deleted-history restoration, actual JSON export, and Research save/reopen with original question and full current note snapshot. Sanitized evidence is committed under Docs/Reviews/artifacts/knowledge-followups-20261006. Headless/native owned browsers and runtimes are stopped; disposable profiles/private fixture helpers/screenshots, generated build/Playwright outputs and four owned dependency symlinks are removed. Operator checkout and local model9099 remain untouched. Final native extension capture, VoiceOver/Safari/iOS/mobile keyboard and actual participant sessions remain unqualified; earlier pending foreground/VoiceOver and device/tester questions are not answered by design approval.
2026-10-07: requester granted foreground/VoiceOver checks, then explicitly directed CDP after native app access stalled/refused. Continue browser operation through CDP only. Isolated Chrome production extension installed/enabled; actual Switch to Sidebar button opens the real panel. Actual content-script capture, runtime-contract handoff and Clipper Save clip UI produced clip f2637dd7-096c-41d0-a3f7-7e9ba1b8b65c, canonical Note5d0d9205-008f-4307-8775-f0362d82c2a7 version1; owner-scoped GET200 confirms saved, and Notes UI reopens the snapshot. This entry is controlled programmatic capture/handoff, not final native context-menu evidence; no injected onClicked callback. Actual Ask chat reached the chat route but hit React useReducer dispatcher failure in the current build, now under dependency-identity investigation. VoiceOver spoken output/Safari/iOS/mobile/real participant sessions remain unqualified. No VoiceOver setting changed; hardware and participant availability questions remain unanswered. Report Docs/Reviews/KNOWLEDGE_BURNDOWN_2026_10_07.md.
Additional actual CDP UX defect: selecting a specific Note before typing/pressing Ask immediately hides the first-use hero and shows No results found. Independent root-cause review proves updateSetting(include_note_ids or sources) clears a possible error via SET_ERROR(null), whose shared reducer unconditionally marks hasSearched=true; no RAG query is needed. Minimal authorized repair under this validation task: preserve existing hasSearched when clearing a null error, while retaining true for actual failed/completed zero-result searches. Regression first in existing scope-handoff behavior tests; preserve cancellation/owner/trust/error semantics. No layout guard or new feature/ADR. Final browser artifact will include this one-line shared fix.
Final production artifact qualification supersedes the earlier React investigation: WXT native deduplication and matching React/ReactDOM18.3.1 pins collapse all5 bundle receipts to one runtime pair. Actual CDP source selection retains Ask Your Library with no premature No results found; one-line shared reducer regression red3 to green224 related tests. Saved Note reopens fully settled at version1. Actual Notes-only Ask sends canonical UUID5d0d9205-008f-4307-8775-f0362d82c2a7, web fallback false, receives HTTP200 and one source. Explicit local Custom OpenAI generation returns 24 November 2026 with one mapped citation and visible original capture URL/date; no page errors. Direct panel URL plus existing runtime-contract handoff opens Clipper and Ask chat without the former dispatcher crash; a local chat analysis attempt ended Stream completion failed and is not counted as successful chat generation. Capture entry remains programmatic, not native menu attestation. The saved Note editor still labels its manual editing state Origin: Typed manually despite a captured source; record this wording for a bounded provenance follow-up rather than infer capture identity from a user-editable tag. Native menu, spoken VoiceOver, Safari/iOS/mobile and human participant sessions remain unqualified. Final receipts and report will retain these limits.
Final isolated validation cleanup is verified in the burndown report: browser closed through CDP; owned runtime/article stopped; private browser/API profiles removed. No VoiceOver setting changed. Canonical capture/source evidence is retained in sanitized committed artifacts, with all native/device/participant and direct-panel qualifications still open.
Final CDP runtime/state repairs and qualification evidence published in draft PR3211: https://github.com/rmusser01/tldw_server/pull/3211 . Canonical Note/scoped Ask/source-preview pass; native menu/panel, stream, wording, spoken/device and participant limits stay open. Disposable assets cleaned up and report evidence retained.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
The original sidebar gesture/acknowledgment fix is merged in PR3205. Final CDP programmatic capture/save/reopen and exact Note-scoped Knowledge Ask pass with a cited local-model answer; React route crash and premature source-selection empty state are repaired in this burndown branch. Keep In Progress for actual native menu/panel qualification, direct-panel streaming outcome, capture-origin wording, spoken VoiceOver and real Safari/iOS/device/participant sessions. No native-menu or human evidence is inferred from CDP.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
