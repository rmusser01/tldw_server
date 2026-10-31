---
id: TASK-15178
title: Use owned native history for Character Quick Chat and continuation
status: Done
labels:
- chat
- ux
- character
- webui
- extension
dependencies:
- TASK-15177
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue the approved all-ten chat UX program by replacing this popup caller of legacy complete-v2 with the existing native history/create/turn recovery protocol under an independent popup controller and verified account. Preserve full-chat source and popup text across delayed work, close, failure and saved continuation; keep existing expert popup capability, native H1 qualification and legacy-review rules. No parallel chat framework or automatic rewrite of historical rows.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Quick Chat has an independent verified popup/account lifetime and synchronous attempt guard; close, replacement, unmount and account/connection retirement cannot publish stale IDs, responses or busy state, and close does not implicitly hard-delete saved conversations.
- [x] #2 New popup turns reuse scoped native creation/admission/settlement and durable existing recovery with canonical roles and one owned parent path; unknown outcomes preserve exact input and cannot blindly replay create or append; explicit provider/model choices are preserved.
- [x] #3 Known-session Open full chat uses qualified saved H1 intent without eagerly changing full Core/assistant/transcript. Nonempty unsent text has exact acknowledged preservation and explicit copy/continue semantics; failed retention, copy, qualification or declined navigation preserves popup and destination drafts. Undispatched fresh entry is distinct from uncertain creation.
- [x] #4 Meaningful red/green popup/native/Manager/destination regressions, relevant client gates, no-new touched lint, both independent reviews and exact runtime/storage/native/legacy limits are recorded against latest dev.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Approved master design Docs/Design/2026-10-04-chat-conversation-ux-remediation-design.md. Current checkpoint87bf9aea71 integrates dev7226596cca in34f678d9b1; Library29 suites902 passed. Official search quick chat/native popup found no duplicate bounded unit; refreshed open PR paths /private/tmp/chat-quick-open-pr-paths.json and local/dev paths highest13504. Proposed reuse: one optional private draftHandle seam in sendNativeHistoryCharacter skips global Core bind and supplies existing create recovery; popup-local controller/callbacks. Both authorized independent reviewers are assessing coupling/preconditions. Legacy complete-v2 historical role/multi-root behavior remains under explicit H1 review, not inferred from popup RAM. No Python implementation planned; Bandit N/A unless scope changes.
2026-10-05 in progress: shared private native preparation guard has 2 real-controller red/green cases; popup lifetime/settings/retention/handoff regressions 22 passed using declared native IO adapters. Cold retained discovery attaches existing settled creation facts (1 red/green); settled Review requires exact Copy then switch, with 4 real-H1/composer/storage cases proving destination draft and metadata, clipboard denial, newer input and source address ABA. Both independent reviewers confirmed immutable unique fresh private storage is the correct minimal path; native H1 remains authority. Core suites and client gates running; WebUI caught card AbortController typo corrected before rerun. No checkpoint or completion claim; real native two-turn runtime and final reviews still required. Scope includes useComposerText/ChatDraftRecovery settled continuation, no Python implementation and Bandit N/A.
2026-10-06 final source review corrections: real-registry Temporary listing no longer reads native creation records; text-only Review refuses a source attachment before retention/clipboard/H1; existing private bound Retry settles only exact original input for the accepted chat; differing retained next input uses explicit Copy and switch with preserved destination metadata. Cancelled prepared creation is not server uncertainty. Actual red3/green10 native-H1 cases and red2/green67 composer-popup cases; visible Copy label red/green9. Both independent reviewers now clear bounded source; they ran no checks. Latest dev94854ca3db integrated in292ea526d1 (CI-only), CI contracts153 pass, client types and no-new touched lint pass. Opt-in native HTTP proof passes:2 turns,4 canonical rows,1 root, second-context2 members, exact cold NEXT input; owned local mock auth corrected after documented import/model/auth fixture failures. Account/protocol/SSE and profile/bookmark IO are adapters; no packaged/native-focus/VoiceOver/progressive-timing claim. Runtime3748 app-file comparison differs only unrelated notes.py/web_clipper.py. Existing shared sanitizer trims outer send whitespace; record as open parent-program data-fidelity improvement. Full39-file relevant merged matrix still running.
Final current-owner qualification and acknowledged-copy corrections have207 passing checks across4 affected suites. Both independent source reviewers clear them. Actual browser recovery confirms the source exact draft, retained popup opening and non-destructive failed explicit Review; native browser successful generation remains open. Broader39-file matrix originally1175pass/2 owner-fixture failures/1 opt-in skip; correcting all owned-default fixtures to the same a*64 hash gives99/100 in the full Manager run, with only the existing5s import-completion deadline failing. The same import behavior passes focused in2.48s with all assertions. After two full deadline failures and a focused pass, reviewed adjacent import/Manager patterns: use the existing30s per-renderer deadline for this completion case, without weakening assertions. Final full Manager and client gates are being checked. No Python implementation; Bandit N/A.
Final checkpoint:39-suite relevant matrix retained33 unchanged passing suites and the opt-in skip; reruns cover all100 Manager behaviors and207 checks in4 changed suites. Aggregate1180 passed /1 opt-in skipped across39 files, not one all-green original run. Owned opt-in HTTP test passes separately. Final WebUI types, extension entrypoint/config compile, no-new touched lint, official English locale agreement and diff checks pass. Dev94854ca3db remains current and is integrated in292ea526d1; latest-dev CI contracts153 pass. Both authorized reviewers clear the final current-owner read-error guard and acknowledged-vs-unknown creation copy. Unique completed native Quick Chat plan removed per project guidance; master design retains evidence and limits. Next tracked status unit TASK13506 addresses header mode/Library false empty; no whole-program completion or native browser success is claimed.
Verified source and closure committed locally as fb0cf6ca79. No PR/push/merge requested. The separately tracked TASK13506 status corrections are excluded from this checkpoint.
Renumbered from TASK-13505 to TASK-15178 on 2026-10-31 (TASK-13960): TASK-13505 now names only "Present extracted web content without raw ingestion metadata in reading" (backlog/tasks/task-13505 - Present-extracted-web-content-without-raw-ingestion-metadata-in-reading.md). Before 2026-10-31 both tasks used TASK-13505; Docs/Development/Backlog_Task_Id_Renumbering.md lists every renumbered id.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Quick Chat now reuses the existing owned native create/turn/H1 recovery protocol under a private popup lifetime and unique durable draft. Creation/turn uncertainty cannot blindly replay; saved continuation qualifies the exact session and requires explicit Copy for unsent text while preserving destination drafts. Current native read failures remain honest and recoverable, and acknowledged creation differs from unknown outcome. Canonical two-turn HTTP and cold-input proof pass with declared principal/HTTP/SSE/bookmark adapters.1180 relevant checks pass in combined matrix/reruns with1 opt-in skip; the owned opt-in test passes separately; both client gates, no-new lint/locales/diff and two reviews pass against current dev. Native browser generation, physical storage, packaged extension/assistive focus, existing outer-whitespace normalization and broader staged enhancements remain documented parent-program gates. No Python implementation; Bandit N/A.
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
