---
id: TASK-13629
title: Integrate current remote dev and restore PR3229 CI admission
status: In Progress
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Remote dev advanced with PR3230 LLM quota enforcement and PR3229 is now conflicting; GitHub cannot launch its pull_request gates. Integrate pinned current remote dev, resolve overlapping quota/profile/chat contracts without losing source ownership or recovery guards, qualify affected paths and restore mergeability.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Remote dev79aa6f00a4509919ce78c11cab9b3be40025866a is integrated and conflicts resolved semantically
- [x] #2 Affected backend quota/chat/profile tests, client types and canonical API contract qualify without new security findings
- [ ] #3 Independent conflict review and normal hooks pass, PR workflows can launch; final review and merge remain gated
- [x] #4 Protected Roleplay Resume fresh generation respects all3LLM quota caps before provider/persistence, while completed replay remains available
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR required: no new rule; integrate accepted remote changes while retaining ADR067 accounting and existing H1 ownership/admission contracts, ADR059 task format, proposedADR069 contract package ownership. MERGE_QUEUE unset, PR autoMerge null. Fresh PR state CONFLICTING/DIRTY with only license/bot checks; no required-gate green claim. Plan Docs/Plans/IMPLEMENTATION_PLAN_chat_remote_dev_13629.md. PR https://github.com/rmusser01/tldw_server/pull/3229.
Only explicit merge conflict was canonical API fingerprint. Ran full shared generator under pinned exporter with supportedPython3.12/Bun1.3.2/Node26; retained2129paths3301schemas and regenerated hashc8ee398cb33d5d9d6288e77ecb3f6c3072899ad294741296fb47e9c3168461c3. Staged fingerprint resolves index conflict; no handpicked parent artifact. Auto-merged chat, character and UserProfiles overlap require ordering/security review and tests.
Initial7backend integration suites142passed5officialfixture-skips. Backendreview found protected Resume returns before newlegacy LLM quota guard: added6actual endpoint controls for3caps and3completed replay cases before production fix. Frontend canonical API review clear. New active task13610 collision identified separately; historical duplicates outsidethismerge are not scope.
Protected Resume quota RED3failed3passed: all three capped requests incorrectly generated and saved200; completed replay controls passed. Added shared enforce_llm_quota after original prepare/ownership/Sync validation and completed replay return, before limiter/credential/provider work. No new quota mechanism or replay charge. GREEN qualification next.
Resolved the one new active tracking collision through backlog-py own canonical frontmatter editor + atomicwriter routines (CLI has no ID flag for edit): local reviewed retention upload TASK13610 is nowTASK13630, with identical full body/criteria/checks/other metadata before provenance note. RemoteMessages API TASK13610 unchanged. Historical unrelated duplicate IDs are untouched. Canonical normalize check passes.
Wider8suite final run194passed5Postgres skips exits0, no shutdown abort (faulthandler only diagnostic). Isolated green6assertions hadpostteardown libc++recursive_mutex abort134; not counted as successful process. Production Bandit3overlap files0findings0scanerrors; test Bandit excludingintentionalB101 also0. Canonical fingerprint+exactpaths driftcheck passes. Black whole-file check exposes preexisting legacy formatting; formatted onlynew6controls withprojectmode and proved fullASTunchanged; Ruffnewtestfilepasses. Types/docs/hooks/reviewfollowup next.
Combined docs+strict source preservation+sharedAPIartifact suites60passed; public/private boundary passes. Both normal client source typechecks pass. Backend/API bounded followup clear after quota gap fix; scope review Docs/Reviews/CHAT_REMOTE_DEV_INTEGRATION_13629_2026_10_13.md. Normal hooks and publication/admission readback next; final wholePRreview/rebase/merge not done.
Normal exact-file hooks pass on77imported/localresolved paths without bypass. Git index has no unmerged paths; source diff check clean. Publishing integration whileTask remainsInProgress untilmergeability/required-workflow launch readback. Final whole-PR review/rebase/merge and remaining issues are not complete.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
