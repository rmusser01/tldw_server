---
id: TASK-13699
title: Keep live-tier relative backend state in its disposable profile
status: Done
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The complete print-driver UX20 browser run passes but final isolation guard rejects Analytics.db initialized at managed worktree Databases. Backend cwd is repoRoot and lazy feedback analytics uses a relative default. Execute owned auth/backend in the disposable profile with repository modules explicitly on PYTHONPATH; retain absolute configured paths and existing strict log guards.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Command regression rejects backend/auth cwd at repository root and preserves explicit module import path
- [x] #2 Full fresh Notes/chat UX20 passes including final isolation and loopback guards
- [x] #3 Both print-driver callers, connected runner contracts, hooks and independent review qualify without disabling analytics or weakening guards
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR required:no; existing disposable live-tier isolation contract applies, no production persistence change. Stages: reproduce actual relative-path escape and red command contract; minimal cwd/PYTHONPATH repair; full real qualification and final review. Prior run20cases pass but runner failed and is excluded from certification. All98 production hashes unchanged; original9conversations28rows remain protected. Do not delete or overwrite existing ignored Analytics.db.
CommandregressionRED1fail35pass→GREEN36; pointer7plusrunner36=43PASS. FinalfullfreshUX20PASS218.67s withstrictrepository-stateandloopbackguards,healthbefore/after/stoppedtrue. BoundedactualderivedcallerPASS14.4s afterexistingrealOpenAPI/modelpreflight atoriginal60sdeadline, allcreate/derive/printassertionsretained. Initialcoldcallerfailedbeforedriver/noPOST andisnotclaimedpassed. FullownedWebUI193files2378casesPASS14existingSkips177.69s beforecwd-onlychange; finalconnected43pass. Configuredtypesandlint0errors0warnings/hooksPASS; TS/JS-onlyruntimefixnoPythonproductionsecuritydelta. Bothworking-sourceindependentreviewersclear; finalexactcommitreviewandGitHubchecksremainlandinggates. Production98hashesunchanged; original9conv28rowsdeepEqual.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Run owned auth/backend from disposable profile directory, with current repository modules explicitly on PYTHONPATH. Lazy relative analytics state stays in profile; existing analytics and strict guards unchanged. Full fresh UX20 and actual derived print caller pass with unchanged deadlines; connected43cases/type/lint/hooks/review qualify. Final exact-head GitHub checks precede merge.
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
