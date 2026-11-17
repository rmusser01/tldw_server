---
id: TASK-13596
title: Validate latest production chat runtime and owned user journeys
status: Done
labels:
- chat
- ux
- verification
dependencies:
- TASK-13595
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue parent chat UX stage5 against the current branch including freshly fetched dev. Refresh the owned production WebUI build with its normal profile/token/budget pipeline, keep one existing owned headless browser tab and the same sealed server profile, preserve the immutable original nine conversations/twenty-eight rows, and walk current new conversation, reopen and character-card flows. No new visible browser windows or ambient19764 interaction. Do not infer public H2 or packaged extension/physical AT acceptance.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Current production webpack/profile/token/budget pipeline passes from the branch including latest fetched dev, with no unrelated tracked changes or stale build assertion.
- [x] #2 Only the verified owned services and single existing headless tab are used; immutable original9 conversations/28 canonical rows stay exact before/after and ambient19764 remains untouched.
- [x] #3 Current normal setup/reopen/character journeys are exercised or specifically bounded by evidence; found root problems are corrected or tracked without declaring broader H2/physical/all-ten acceptance.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR required:no; existing ADR049/050/059 and approved NN/G/H2 parent plans govern. Runtime verification/evidence only; any source correction must first have its own bounded tracked scope and causal control. Use existing build-with-profile and own sealed runtime/preservation managers; Node26, one owned headless tab, no profile reset or original transcript mutation. Root sole runner/browser owner; original two reviewers only inspect supplied evidence/source and create no browser.
Current Node26 production profile/webpack/token/budget pipeline exits0; shared596.3KB/600, chat798.6KB/900. No tracked source/config diff after build. Original immutable9/28 preservation BEFORE deepEqual=true, baselineRewritten=false. Existing owned CLI session/tab reachable; no new browser. Verified old Next62134→manager62132 and API29631→manager29607 plus actual listeners before direct child SIGTERM; wait actual exits before manager notification/new startup. Avoid house five-second kill fallback during API grace; no forced kill. Stage1 complete, owned refresh in progress.
Runtime diagnostic: forced production build-profile changed deployment to quickstart and baked health/API rewrites to default8000, while original sealed runtime uses development/advanced and direct19767. This was root review harness setup error, not product/source finding. Original packaged-production build/token/budgets remain real but not qualified for this runtime. Restore original development profile through same normal Next production webpack/token/budget pipeline in separate ignored dist, with explicit profile assertions; no server/profile/key reset. Server-settings recovery opens a modal (no URL navigation), so waitForURL diagnostic excluded. Owned originals untouched; API27579 remains ready.
Final current1173e3e4ad includes fresh dev a744577. Correct original development/advanced normal Next production webpack/token/budget commands0,593.3/600 and795.6/900. Actual manifest LR5UsmKYlcp2mxLQ9_2cg matches Next78083/API27579 and critical19767; same sealed profile/no auth-key reset, forced kill or new tab/window; ambient19764 untouched. Ordinary creation/reopen, distinct prior-character reload and new post-fix greeting→protected Roleplay201/complete-v2 200→cold exact3row preservation pass; original9/28 BEFORE/AFTER deepEqualtrue immutable baseline unchanged. Recoverdrafts bounds + keyboard/pointer opening and current screenshot clear prior scrolled-position candidate. Two final independent bounded runtime CLEAR. Selection-copy ambiguity tracked TASK13598, no persistence failure. Review Docs/Reviews/CHAT_CURRENT_RUNTIME_13596_2026_10_12.md records excluded profile/modal/accessible-name diagnostics and explicit limits; no whole-ten/publicH2/first-install/real-model/Compact/installedextension/PG/AT/study acceptance. Bandit N/A runtime/doc-only; normal exact-owned hooks and local commit follow.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Current configured Full WebUI production artifact and actual ordinary/reopen/character happy paths qualify with unchanged original9/28. Same owned one-tab profile, critical19767 and current manifest verified; ambiguity tracked13598. Full review records evidence, diagnostics and remaining broad acceptance. Runtime/doc-only Bandit N/A; exact-owned hooks/local commit follow.
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
