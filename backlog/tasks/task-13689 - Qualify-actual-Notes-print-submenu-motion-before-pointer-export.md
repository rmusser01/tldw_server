---
id: TASK-13689
title: Qualify actual Notes print submenu motion before pointer export
status: Done
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Current a34 UX CI reaches visible Print flyout but the ordinary click sees entering motion instability then a hidden retained leaf; 19 other UX cases pass. One unchanged local NE04 passes, so there is no established production print closure fix. Qualify actual flyout motion and ordinary pointer entry in the shared NotesPage driver, preserving known-defect print oracle and evidence. Both callers must continue to work.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Wait on observable flyout motion using bounded actual browser state; no force, retries, fixed sleeps, disabled motion or oracle relaxation
- [x] #2 Qualify NE04 and derived note shared caller using actual server and record limitations
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR assessment: no new ADR; test driver readiness contract only, existing Notes production print defect #3117 and oracle remain unchanged. Evidence: /private/tmp/chat-pr3229-a34-ux-regression-job.log and private CI video; actual flyout visible then unstable then hidden. Source @rc-component/menu uses delayed entering motion. Driver has two callers: UX notes-p0 and workflows/notes-studio-derived-note.
Driver NE04 now passes3 repeatEach zero-retry cases and second derived-note caller; original oracle remains. Broadened full UX20 on reused runtime is invalid qualification: empty-library precondition got4 notes and native extension fell back to an existing unqualified build. Preserve this invalid report, do not claim full pass or infer production roots. Re-run using supported isolated live-tier runner with fresh backend and newly built native extension. All contexts headless.
Supported isolated runner first local launch failed Turbopack node_modules symlink outside filesystem root after current native extension production build35.1s. Second launch uses already-qualified private artifact with owned Bun dependencies: all97 production hashes match;314 current tracked E2E/config files copied without transforms; new native extension build supplied explicitly; fresh backend remains managed-worktree current source. Preserve first failure artifact; no product-source change for local environment.
Final6b133421 CI NE04 fails during Print hover before click: visible/stable item twice hit-tests textarea, then hidden224retries until120s. Prior fullcfd and local20cases pass. Existing no-running-animation poll can accept closing/closed pointer-disabled popup, not proof of entering readiness. Reopen root investigation: inspect actual video/timeline and reproduce with pointer/open/hit-state diagnostics; only if movement reproducibly closes a genuinely ready menu change product behavior. No retries/force/dispatch/oracle weakening. Independent reviewerB identifies insufficient readiness but leaves cause of closure unqualified. Finalhead original log /private/tmp/chat-pr3229-6b1334-notes-chat.log and actual failure artifact retained. ADR assessment unchanged: driver-readiness fix requires no new architecture rule.
Animation-only predicate reproduced5behavioral failures (closingpointer-none, editorinterception, geometrychange,zero-size and missing hitcheck) while2animation negatives pass; final DOM helper7pass. Parent and leaf readiness now checkactualstationarypositivegeometry/hitownership with existingtwoRAFs/animationguard, Exportremainsaria-expanded, thenoriginalhover+oneclick. No retries/force/reopen/disabledmotion. Normal+CPU8throttled oldpathbothpass, so no claim of confirmed productionmenuclosure. CompletefinalrealUX20passes3.7m withstrictisolation afterTASK13699. Boundedsecondderivedcaller initiallyfailsbeforeprint atnotecreation with onlyGETkeywords andzeroPOST/PUT; existingwarmBackendOnce explicitlydocuments coldOpenAPIstall#3135. Applysameactualschema/modelwarmup beforeUI inunchangedderivedworkflow; retainoriginalnotecreate/derive/printoraclesand60sdeadline. Finalboundedqualificationpending.
FinalwarmderivedcallerPASS14.4s, original60s/15sexpectdeadline, exactcreatePOST/body+Cornellderive+ordinaryPrint assertionsretained. SameexistingrealOpenAPI/modelpreflightasNotesP0 avoidsconcurrentcoldschemasetup; no specificproductionSaveguardclaimed. FullsupportedfreshUX20PASS218.67s withfinalisolation+loopbackguardsthroughTASK13699,healthbefore/after/stoppedtrue,0skips/retries. WebUI193files2378casesPASS14existingSkips177.69s; finalrunner/pointer43PASS. Pointer/runner5fileslint0errors0warnings; types/hookspass. Bothindependentworking-sourcereviewsclear, finalexactcommitreviewandnewheadCImustprecedemerge. Excludefull20browserpasswithfailedanalyticsisolation, initialderivedcoldfailureandnormal/CPU8diagnosticsfromfinalcertification.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Strengthened shared print-driver readiness using actual stationary pointer-hit ownership and open parent state before ordinary hover/click; seven red-first cases pass. Complete fresh Notes/chat UX20 and derivedcaller1 pass with strict isolation and original deadlines after real preflight. No proven production closure claim, forced interaction, retries or oracle relaxation. Final published-head CI remains merge gate.
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
