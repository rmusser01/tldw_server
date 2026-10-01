---
id: TASK-13408
title: Finalize Chat Workspace live UAT and latest-dev PR gates
status: In Progress
assignee: []
created_date: '2026-10-01 18:39'
updated_date: '2026-10-01 18:46'
labels: []
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/3071'
documentation:
  - Docs/superpowers/reviews/chat-workspace/2026-10-01-latest-dev-no-mock-uat.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue reviewed TASK-13398 Chat Workspace fixes and PR3071. Latest upstream ec86ba4 introduces a different task with ID13398; this unique finalization record preserves both histories and owns remaining integration/UAT gates.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Complete live Browse/staging/external-link and mobile recovery/composer keyboard UAT plus fresh owner-target visual proof without mocks.
- [ ] #2 Integrate latest dev FastAPI0.142.1 and verify backend contracts and original data/tab/stash preservation.
- [ ] #3 Verify production Turbopack token-sync and unchanged bundle budgets, scoped tests/security; push reviewed fixes to PR3071 without merging or inventing the human Change summary.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Official CLI create failed Maximum call stack size exceeded without a created record. Existing TASK-13398.12.5 and .7 already associate source/UI and upstream integration changes. Functional owner/account/backend matrix passes 3 real Gemma turns/6 canonical rows and0 autosends; mobile real recovery-overflow identified and minimal transcript containment fix now GREEN (native rerun pending). Bundle ID helper decoupling retains exact behavior, actual production539.9KBshared/842.3KBmermaid pass unchangedlimits.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

Native final acceptance PASS: native-preview-mobile-repair-acceptance.json has actual memo/web HTTP200 previews, native Stage/Unstage/two-source Clear retaining real nonempty typed draft, actual external Example Domain title/content, actual mobile keyboard composer650-738/Send746-790 within390x844/nooverflow,0completiondispatch. Fresh owner-uat/fresh-visual-final.json/png captures originalA Ready/loaded canonical transcript and preserved draft with0completiondispatch; parent visually inspected mobile, sidecar and memo frames. Stale cachedLoadingframe excluded. Production final current source copied/hashesbound into matched isolated snapshot; Turbopack build0/token-sync0/unchangedbudgetsPASS539.9KBshared,842.3KBheaviest. Direct checkoutbuild failed solely node_modules symlink outsideTurbopackroot; no repo configchanged. FullTSC0; focusedmobileRED2/19->GREEN158. Independent9pathreviewpending. FastAPI0.142.1 installed with coherentexistingOTelSDK/exporter1.45.0; latest-devmerge/APIpreservation/backendgatesnext.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->
<!-- SECTION:FINAL_SUMMARY:END -->

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
