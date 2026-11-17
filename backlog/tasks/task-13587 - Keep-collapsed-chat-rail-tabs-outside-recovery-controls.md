---
id: TASK-13587
title: Keep collapsed chat rail tabs outside recovery controls
status: Done
labels:
- chat
- ux
- accessibility
dependencies:
- TASK-13586
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fix the confirmed shared cockpit layout root: the absolute collapsed runtime tab covers560px² of Recover drafts at1280x900 and intercepts its right-end pointer. Reserve desktop space only for visible restore tabs, retain expanded/focus/mobile behavior and existing transcript/composer ownership, and qualify real geometry/pointer routing against integrated latest dev590e7e5. This follows the approved all-ten root remediation queue.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Collapsed desktop restore tabs have dedicated space and no longer intercept conversation recovery or composer controls.
- [x] #2 Existing rail callbacks, mounted controls, focus/mobile behavior and canonical conversation/draft contents remain preserved in connected and actual browser checks.
- [x] #3 Both normal clients, current production build/tokens/budgets, introduced static checks, two bounded reviews and normal owned hooks pass; packaged/physical AT/study and broader queue limits remain explicit.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR assessment: no new durable architecture rule; existing shared cockpit layout and ADR049/050/059 govern. Actual prior-build hit-test560px² overlap and right-end Recover drafts pointer resolves Restore runtime sidechannel: /private/tmp/chat-runtime-overlap-13585-investigation.log. Latest fetched dev590e7e5 verified ancestor at normal merge7de55a36fa; highest task13586 across local/dev/open PR scan. Root owns source/runners and one existing headless tab; no visible window or ambient19764 access.
Six-suite first39PASS/5FAIL; all five identical failures reproduce at exact committed HEAD7de55a36fa and changed shell restores byte-exact. Sticky fixture omits current pure settings exports: use partial actual module while keeping only declared sync/patch IO doubles. Context count follows current selected-source label; mobile source guard follows existing non-shrinking scroll wrapper. Original assertions remain. No new class-string mirror test; actual native RED/green qualifies the layout correction. Final connected run follows.
Final six connected44/44, both normal client types and current production build/tokens/budgets593.3KB shared600 and795.6KB chat900 exit0. Four TS static0/0 introduced0; both approved independent source reviews CLEAR. Actual fresh-bundle1280/1024/1023/390/320 overlap0 and all15 recovery hits reach correct target; formerly obstructed85% click opens Retained drafts and closes without applying. Expanded/Focus gutters0; mobile gutters0 and desktop tabs hidden. Actual mounted composer identity, original transcript and blank input remain through both rail callbacks/Focus/return. One owned headless tab, no visible window. Visual1280/320 screenshot inspection and no horizontal overflow qualify viewport checks only. Nine canonical responses28 rows deep-equal before/after, baseline untouched. Evidence /private/tmp/chat-rail-overlap-13587-*. Bandit N/A TS/Markdown only. Fresh dev now19d213e936da37a908b30e79d1e4ff7a4c3839d1 after frozen590e7e5 qualification; next bounded integration remains required. Master13458/all-ten, publicH2/quota/adoption/D7/package/physicalAT/study stay open. Normal exact-owned hooks follow.
Normal exact-owned seven-file repository hooks pass without bypass. All three bounded stages,3AC/6DOD complete; only own plan retires to /private/tmp/IMPLEMENTATION_PLAN_chat_rail_overlap_13587_completed.md. Broader master remains open; next integration targets dev19d213e936.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed the shared collapsed-rail root with two conditional desktop gutter classes. Actual560px² interception becomes zero overlap/fifteen correct hit targets across five widths; formerly obstructed recovery click opens Retained drafts. Rail/Focus transitions preserve the same composer, transcript and blank draft; nine canonical responses28 rows unchanged. Connected44/44, both types, current build/tokens/budgets, introduced lint0, two source reviews and normal exact-owned hooks pass. Three inherited fixtures repaired against proven exact-HEAD failures. TS-only Bandit N/A. This bounded correction is complete; newer dev integration, publicH2/quota/adoption/D7 and full surface/package/physicalAT/study acceptance remain open.
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
