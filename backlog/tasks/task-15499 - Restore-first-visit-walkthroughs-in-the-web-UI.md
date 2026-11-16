---
id: TASK-15499
title: Restore first-visit walkthroughs in the web UI
status: In Progress
labels:
- frontend
- tutorials
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Offer the existing first-visit prompt on eligible web pages and add Settings and Collections walkthroughs.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Fresh web sessions see the first-visit prompt on eligible pages and can start a tour.
- [x] #2 Settings and Collections tours work on desktop and mobile.
- [x] #3 Notes offers a short optional create/edit/save walkthrough and never starts by itself.
- [x] #4 Organization members serialize PostgreSQL timestamps without changing the string response contract.
- [x] #5 Research Workspace keeps its page-owned first-visit guide.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR assessment: no new decision. Reuse the registered tutorials, prompt, and runner in the existing web shell.
Red: web Chat and Settings prompt tests and three registry cases reproduced absence. Green: 124 focused WebLayout, prompt lifecycle, registry, locale, Settings focus and Joyride runner tests passed. A single browser-level Settings prompt covers its subroutes. Page-owned Notes and Research guides remain in place.
Code review caught early prompts on loading pages and trailing-slash URLs. Added failing cases first, then deferred new prompts until their first target mounts and honored page-owned guides using canonical routes. Final connected focused Vitest suite: 126 passed.
Signed-in hosted UAT: Media/Prompts first anchors absent in deployed variants, Knowledge skips unavailable secondary targets, handheld Settings spotlight can leave view, and generic offers hide the task. Follow-up aligns tours to actual stable controls and shows task descriptions before Start.
Candidate verification: 119 web-shell, registry, prompt, locale and Joyride tests passed, plus 12 offline demo unit tests. Browser fixtures opened the corrected Media and Prompts tours and kept the first Settings target visible at 320 px. Beta UAT verified Not now survives reload and Flashcards remains in Help.
Signed-in Team UAT reproduced HTTP 500. The production response model rejects a synthetic PostgreSQL datetime at added_at; retain the API string result with explicit normalization.
Signed-in beta UAT identified the Notes auto-tour and PostgreSQL timestamp failure. The fix preserves first-visit dismissal and Notes Help replay. Targeted frontend checks and all three schema regression cases passed.
Public PR #3473. Signed-in owner UAT also exposed an invitation list 500 after recovering the member list. The repository uses naive UTC parameters for PostgreSQL TIMESTAMP in list, create, and expiry cleanup. Eight focused backend cases passed.
PR #3473 CI reproduced the mobile parity guard failure against an unchanged dev source. dev already removed the decorative Composer pill in 157b04a3ac. Remove the stale source assertion; the sticky composer integration test covers the intended absence.
Verified all three local Playground CI suites (52 composer, 15 device matrix, and 63 accessibility tests), plus four sticky-composer integration tests.
Local WebUI gate reproduction found the lazy TutorialPrompt default import breaking command palette and loader focus mocks that already match the shared layout named export. Use the same named export in WebLayout.
All 20 layout integration tests passed with Node 20 after matching the named export. The remaining local WebUI failures also reproduce on untouched dev and the affected suites passed in Linux required frontend CI job 132613158739. Its independent shared-UI shard failure is the already-fixed stale Composer pill assertion.
After rebasing onto 6ec6241b0e, the real-server UX smoke exposed the Chat first-visit offer covering the mobile MCP dialog footer. The GitHub failure screenshot confirms the overlay received the Close press. Keep modal dialogs above tour offers and defer new offers while a modal is visible.
Browser-verified the modal fix at 390 × 844 and 390 × 600 with the real MCP dialog. An existing prompt no longer intercepts Close before or after 150 tool rows arrive mid-press. A new prompt waits for the modal to close. 98 focused Node 20 tests and WebUI typecheck passed.
Required frontend shards on 1d63a854a5 caught two fixture regressions on clean Linux: the WebLayout chat-scroll partial Ant Design mock omits theme and the real notification dependencies, and Quick Chat still expects no Settings Health walkthrough. Both replayed clean on the base. Updating fixtures while preserving the new tours and dialog layering.
Finished inspecting all eight frontend unit shards. Four failed because the layout fixture omitted real Ant Design exports, two Quick Chat fixtures treated Settings Health as a route without a tour, and the Notes fixture referenced its removed sorting step. Reproduced the layout regression with Ant Design deduplicated and kept its real tour rendering. Verified 78 WebUI layout tests, 100 Quick Chat and prompt tests, and 66 tutorial and Notes tests (overlapping focused sets). Production Python Bandit scan is clean.
Required Critical E2E on 82d8bdfb49 caught the tablet overflow assertion during the first-visit notification entrance. The document width passed and the Playwright trace captured the toast partially outside the viewport before the screenshot showed it fully inside. Updating the Character Chat assertion to retry both existing layout checks after the entrance. On the same head, unit shard 7 passed all 979 assertions but failed from a post-teardown Joyride timer; the exact 31-file shard and eight repeats of the real Joyride suite passed locally.
Chromium reproduced the toast crossing the 768 px viewport during entry. The revised assertion passed after entry and still rejected persistent overflow. The required backend job separately reported changed-file mypy errors in the Team schema, invite repo and its fixture, and the test-quality ratchet flagged an assert inside the PostgreSQL fake. Fixing the schema/SQLite return types and checking the captured cutoff in the test.
Verified 74 AuthNZ tests, the repository test-quality ratchet, all four changed Python files under mypy, and Bandit on the two production Python files. Browser verification used the same overflow helper as CI and preserved the no-overflow guard.
Linux frontend shard 7 repeated the same post-teardown real Joyride Popper callback on 6516c4c4de after all 979 assertions passed. The fixture needs to cancel its pending animation frames and timeouts after unmounting. Keeping the live positioning and scroll assertions during the tests themselves.
A focused teardown probe showed that the original fixture let test-owned timeouts and animation frames escape, and the corrected fixture canceled both. All 37 focused prompt and runner tests, the exact 979-test deterministic shard, and shared UI lint passed locally.
Public PR #3473 CI, 2026-11-15: E2E UX Regression run 43777339267 / job 133011460380 found a light-theme axe contrast violation when the first-visit prompt was covered by the open Chat model picker (32 of 33 passed). Defer or temporarily close the offer while its picker or a modal is open, and reoffer it on close without persisting a dismissal. A per-offer guard rejects stale actions and close callbacks. Added focused cases for both opening orders and extended the existing held-open axe ratchet to check the real offer in both themes and its return after Escape. Both focused tests failed before the fix. Afterward all 38 focused UI tests, the real-browser axe case, strict shared typecheck, WebUI test-file lint and the 6,241-file shared lint passed. Accessibility baselines are unchanged.
Full local browser coverage also reproduced the visible Chat offer intercepting pointer events on the model chip at 1280×720. Move the Chat offer below its toolbar and above the composer. All 26 focused tests and five real-browser Chat accessibility / Notes layout and onboarding tests then passed, including a real pointer click, axe scans in both themes, and the offer returning after Escape. The separate local Notes placeholder contrast result reproduced on the pre-fix commit; the previous GitHub run passed all three Notes accessibility cases.
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
