---
id: TASK-13664
title: Qualify character journeys against original native roleplay admission and completion
  contracts
status: Done
labels:
- chat
- testing
- ux
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Actual7case owned runner proves desktop/tablet/mobile setup passes but four character assertions target the wrong lifecycle: blockedmodel is refused before trackedsetup, and richroleplay saves via roleplay/messages plus complete-v2 rather than generic chat/completions. Verify original roleplay admission, completion receipts, owner/digest/revision and canonical cold persistence. Investigate real providererror recovery before correcting assertions; never force richcharacter flow through generic chat or bypass model/owner guards.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Real blocked-model lifecycle refuses before owner/input/provider mutations and retains draft with actionable recovery; assertions match public supported setup.
- [x] #2 Character success and provider failure journeys observe original roleplay requests/receipts and preserve identity, richcontext, canonical IDs/parents/bookmark and cold equality.
- [x] #3 Actual targeted native cases pass or unresolved roots remain explicit; static gates and independent source review recorded before closure.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Parent TASK13662 navigation remainsInProgress. Actual headless7case run3pass4fail0skip7.4min, private report /private/tmp/chat-pr3229-character-13662-results.json. API evidence: /chats/<id>/roleplay/messages201 then complete-v2 success200/failure401; generic endpoint waits never match. Independent review confirms unusable model seeded before tracked selection conflicts with native preflight refusal. ADR assessment:no new rule for fixture correction; existing nativeH1/roleplay admission and error contracts govern; any durable product change requires separate design/ADR assessment.
Actual corrected character-create journey1pass0skip7.3s: original H1 capture plus roleplay/messages201 and complete-v2 nonstream200; existing strict production parsers bind input/snapshot/settings/history fence and result IDs. ExactBEEP, canonicalparents/pair, ownerbookmark, explicitreopen/coldfullsavedresponse equality andoriginalcardidentityall execute. Feedbacktest setup reassessment: first mountedNativeOpening create mutatesoriginalsource so notification correctlysuppressed (0calls); initialclassifierattemptdidnotchangeit; warmfirstsuccess+rerender lostnativeview. Stop after3attempts, sourcefeedback REVERTED, no validfeedbackREDclaimed. Investigating alreadybound housefixture via independentreadonlyreview; no sourceguard weakening. Provider401 matchescurrent sanitizednested detail.error and accepted_unsentknownappend, not generic502/unknowninput.
Valid already-bound feedback RED then GREEN; all five connected suites 694pass0skip each native/WebUI before storage-contract fix13665. Source compilers for both production dependency trees now0, real lint baseline initially found only unused test import which is removed. Same-source16-delta Bun production154page build/token/assets/budget passes recorded privately; automatic build gates against older .next excluded. Current native7case run6pass1fail0skip: provider401 receipt/recovery retained, expected notification absent. After three attempts at same boundary reassessing with private diagnostic artifact and independent reviewer; no guard weakening or retry/cold claim. Live Actor Scene preparation guard appears inappropriate for saved-roleplay path but actual boolean evidence still pending. TASK13662 and broader master remain In Progress.
Valid scene-retirement RED: supplied original scene receipt retires during actual mocked postappend401; recovery exact accepted_unsent while notification0 (first two isolation attempts excluded). Minimal full-page feedback guard now uses original captured resume draft/session + H1 fence + native lease + original owner/view, no live scene dependence. Focused6cases pass current/retired-scene feedback and history/account/draft/unmount negative controls; two original recoveries dismissed only in fixture teardown after assertions to prevent crosscase contamination. Actual browser pre-fix diagnostic: saved=true, requestedScene=true, preparationCurrent=false, draftCurrent=false, historyCurrent=false, sessionCurrent=true,mounted=true,status401. Therefore original H1 also retires in real first-send lifecycle; current unitfix alone cannot close browser issue. Tracing automatic loader vs session restore vs explicit intent, preserving all guards and original data.
Fresh final feedback guard native and WebUI five connected suites699pass0skip each; both scoped production dependency compilers0diagnostics, no wholeproject/testtyping claim. Independent bounded guard review clear; review artifact Docs/Reviews/CHAT_SAVED_ROLEPLAY_FEEDBACK_GUARD_13664_2026_10_13.md. Immutable original9conversations28rows deepEqual baselineNotRewritten and declaredsynthetic4rows verify unchanged after diagnostic runs. Provider browser retry/cold remains unqualified; lifecycle diagnostic build traces exact load/restore/intent next. Bandit N/A TS/Markdown scope.
Lifecycle trace resolves attribution: both invalidations have ready=false/native=false during request-own first native binding; later automatic loader ticks ready=true/mode ordinary/deliberate=false without invalidation. No loader or restore fix warranted. First-native sender delegates to saved-roleplay after accepted bind but raw provider error returns to caller retaining original fresh Scene/H1 guard. New mounted first-bind RED verifies one create/append/complete401, accepted native draft plus exact accepted_unsent receipt, notification0. Design: preserve accepted-source authority across delegated failure without claiming no admission; reuse transient guarded-error pattern and safe classifier across full/compact/popup callers. Existing typed no-admission refusals must stay unchanged. Independent design review pending.
Original accepted-source root implemented as transient NativeHistoryRoleplaySendError only around postbind delegated saved-roleplay await; original message/cause retained, existing typed refusal/settings instances rethrown unchanged. Caller full currentness uses captured accepted guard even when original fresh Scene/H1 retired; classifier uses guarded roleplay provenance rather than late ambient metadata. Compact fresh catch uses same classifier and stale-error veto. First native full5cases (current/history/account/draft/unmount) + compact3(success/providerfailure/retiredfailure) now8pass0fail; all exactacceptedinput/receipt/noextraappend invariants remain. Initial script matched earlier separate resume delegate, corrected by inspected onefile hash-checked idempotent edit; excluded failed runs. Auto-review rejected non-idempotent rerun; safer verified correction succeeded, no bypass. Popup3RED exposes raw HTTP401 and stale error before/duringreview; same bounded guard/check-after-await fix nowunderverification. Temporary browser diagnostic instrumentation removed before final nativejourney qualification.
Root source checkpoint:7connected native and WebUI suites748pass0skip each; isolated19delta productionbuild154pages exit0; actualdist token sync0 and budgets590.2KB shared/869.0KB heaviest pass. Firstbind test journalcleanup moved tofinally per independentminorreview. Actualrootpreview nowU6RUXbj2-NEmmxWkcO7-4 on19769. Stop/reassess asset verifier after3mis-scoped walker failures: traced JSONC isnotstrictJSON; stat follows danglingBunlink; servingdist/node_modules expectedexternalized symlinks. No sourcehash mismatch, no missingstaticasset fact established. Research repository budget/build/smoke implementations; budget parses build-manifest ratherthan recursively traversingtracedsource. Standalone deployment/extensionpackage qualifications remainseparateopen. No failed walker credited.
Final bounded closure:7actual native character journeys pass0skip0flake23.95s on root buildU6RUXbj2-NEmmxWkcO7-4; original provider failure/retry/cold detailssummary all execute, exact original operation/input/append/intent persists;2complete-v2 attempts1append0generic completions. Native/WebUI7suites748pass0skip each; both4root sourcecompilers0;12ESLint files0errors261unchangedwarnings0new0ignored. Real-output token/budget pass; emitted969JSON files1128JS/CSSrefs0missing after3walkerfailures reassessed (standalone/dependency traces notqualified). Source19delta hashesmatch; immutable9convos28rows and declaredsynthetic4rows deepEqual. Independent bounded source review clear, fixtureteardownminor addressed. Bandit N/A TS/test/Markdown. Review Docs/Reviews/CHAT_SAVED_ROLEPLAY_FEEDBACK_GUARD_13664_2026_10_13.md. PR https://github.com/rmusser01/tldw_server/pull/3229 remainsunmerged; wholePRreview/rebase/allrequiredchecks and broadermaster/cockpit/H2/packaging/AT/participant remainopen.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed provider feedback after a request-owned first native binding by carrying its accepted source with raw delegated errors, preserving typed refusals and exact uncertain-reply recovery. Qualified full/compact/popup guards and actual character admission, provider failure, original retry and cold recovery. Seven actual cases and both748-test profiles pass; scoped types, unchanged lint baseline, source-matched production gates and original-data preservation pass. Final whole-PR and broader program gates remain open.
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
