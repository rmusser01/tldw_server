---
id: TASK-13598
title: Clarify character compatibility selection feedback
status: Done
labels:
- chat
- ux
- copy
dependencies:
- TASK-13596
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Current successful saved Roleplay still shows Character context: Selected — not verified because the chip describes only client compatibility prediction. Clarify that this chip is selection/setup feedback, preserve excluded/override warnings and actions, and avoid implying persisted setup or actual prompt inclusion. Reuse the shared hook and existing copy mechanism; no new saved-binding authority.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Included and blended character/persona chips describe client selection plainly without unverified/save/prompt-use claims.
- [x] #2 Excluded and prompt-override feedback and existing corrective actions remain intact; targeted behavior checks and normal application type/lint checks pass.
- [x] #3 Current owned WebUI copy is verified in the single existing headless tab; exact original9/28 preservation and broader acceptance limits stay recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR required:no; existing ADR049/050/059 and parent NN/G plan govern. Root is ambiguous selection-copy projection after an otherwise qualified saved flow; no persistence defect inferred. Two authorized reviewers inspect read-only source. Before code: map callers/locales/tests; causal existing behavior fixture expectations before smallest production text-only fix. Bandit N/A TS/MD-only. No new browser, profile reset, generated/publicH2/cold-binding authority or physical AT/extension acceptance.
Causal normal WebUI RED9 failed/11passed/184 intentionally filtered-out, including all three owned UAT355 positives. Four shared fallback/key changes and four canonical English entries applied; enum/branch/action behavior unchanged. House locale dry-run writes only English playground mirror, but normal sync also catches up nine pre-existing canonical history-recovery keys absent from that mirror. Initial four-only mirror assertion correctly stopped qualification; inspect confirms sourceAdds4/mirrorAdds13 and no modified/removed prior entries. Keep canonical generator output and include HistorySelectionReview behavior checks; record this bounded English mirror parity scope, not a new history feature or physical extension acceptance.
Normal connected nine-file357/357 passes. Independent source reviews bothCLEAR; reviewer identified synthetic persona-blended input is not current derivation, so remove that unreachable case before final checks, retain reachable character-source action. Normal WebUI type command hit Node default4GiB heap (exit134), not a TypeScript diagnostic. Actual showConfig470files/zero old generated-owned-build files rules out stale artifact fan-in. Retry same normal project with8GiB Node allowance; no tsconfig include/exclude, suppression or source logic change.
Final reachable hook15/15 after removing exactly one synthetic persona-blended case; preceding normal connected nine-file357/357 covers unchanged production. Both normal application type projects0 with8GiB Node allowance and identical configs after explicit default4GiB resource diagnostic. Four exact-filename source/test lint baseline0/final0 errors,39baseline/39current warnings,no introduced findings; genuine coverage asserted. House English locale generator sourceAdds4/mirrorAdds13/prior entries exact and final dry-run no writes. Two source reviewers CLEAR. Original profile normal production build and one-tab current copy/preservation follow; no physical extension claim.
Current actual newmanifest tyt0kJ9ulPTjWhuB47qr3/Next83607/critical19767, oneexistingownedtab. Character setup: Selected visible/no oldchip,3articles/blankcomposer, exact prior d9d... canonicalrows and immutable original9/28 unchanged. Same profile/no auth-key reset/ambient19764; oldparentgone guard refused before signal then exactreadyPID/cwd/manifest/listener verified owned orphan before graceful retirement, noforcedkill. Normalbuild/token/budgetall0 shared593.4/600/chat795.6/900. Review Docs/Reviews/CHAT_SELECTION_FEEDBACK_13598_2026_10_12.md records boundaries/diagnostics; both independent source/currentUI boundedCLEAR. Bandit N/A TS/JSON/docs; normalexact-ownedhooks/localcommit follow; masterwhole-ten/publicH2 and physical acceptance open.
Both final independent currentUI reviews now explicitlyCLEAR. Eleven initial exact-owned normal repository hooks0, no bypass; all3AC/all6DOD and ownthree stages complete. Only owncompleted plan retiresprivately. Finalten-file metadata hooks and exact-owned localcommit follow; masterbroadercriteria remain open.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Four shared copy/key changes distinguish client selection from save/prompt use, preserving warning/actions and meaningful cross-tab controls. Normal connected/affected tests, two normaltypes, matchedlint, house locale parity and current one-tab build/UI/canonical preservation qualify. Explicit defaultheap/managerdiagnostics and broader limits recorded; Bandit N/A. Exact-ownedhooks/localcommit follow.
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
