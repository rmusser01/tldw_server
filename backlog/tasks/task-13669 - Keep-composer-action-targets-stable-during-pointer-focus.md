---
id: TASK-13669
title: Keep composer action targets stable during pointer focus
status: Done
labels:
- chat
- ux
- bug
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Actual Pro cockpit native trace shows Tools moves16px between pointerdown and pointerup when composer focus expands the empty textarea; click lands on DIV and Tools never opens. Preserve idle collapse but expand only on textarea entry, retaining expanded geometry through intra-composer action focus. Qualify installed browser pointer and keyboard Tools opening after real picker closure. ADR assessment: transient layout/focus behavior only; existing ADR049 ownership unchanged; no new durable architecture rule.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Actual toolbar pointer focus preserves textarea geometry and Tools target throughout one click
- [x] #2 Textarea focus still expands and stays expanded through intra-composer action focus until focus leaves
- [x] #3 Native/WebUI behavioral controls, current artifact browser and original-record preservation qualify the fix
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
PR https://github.com/rmusser01/tldw_server/pull/3229. Native trace: pointerdown Tools top756.89/bottom784.89, focusin same; pointerup top772.89/bottom800.89 and targetDIV, clickDIV. Earlier conjectures about missing onClick/hover forwarding are excluded: clean actual AntD works and onClick/ref forwarded. Global highest id across fetched origin/dev, all open PR files and local13668; reserve13669.
Broader14-suite controls expose obsolete fixture APIs: first29 failures include public QueryClient.getQueryState omission and old selector-only message-store; second6 failures isolated voice/pinned presentation fakes. Stop extending piecemeal fake APIs after three contract attempts. Reassess existing image/openui/roleplay fixtures: they reuse actual Zustand store APIs and actual useSimpleForm. Adopt that established pattern for these presentation fixtures, preserving original six assertions, stable declared sources and no claimed native ownership. Production pointer fix independently source-reviewed clear; four-root types0, full13-file actual-root lint0errors/198 unchangedwarnings/0new; current pointer artifact build pending.
Final14 connected suites405 tests pass each native/WebUI0skips. Four production roots types0 both; actual root lint13 TS0errors/198 unchangedwarnings/0new/0ignored; current cockpit/imported-helper types0. Source review clear. Artifact fMvRRh02jPiKTqhZPNMyD/154pages/44 matching deltas/969JSON1129refs0missing, actual tokens/budgets590.3KB/869.0KB. Six browser cases pass0retries/skips/flakes13.4s plus final explicitpointer y<=1px and Enter/Escape/focus run pass. Original9/28 and existing4rowfixture deeply equal, immutable baseline not rewritten. Design/plan and review Docs/Reviews/CHAT_COMPOSER_REQUIRED_CATALOG_POINTER_13668_13669_2026_10_14.md. BanditN/A TS/MD-only. Normal hooks/commit pending, TASK13656 early-stop and13668AC3 open; no wholePR/merge claim.
Final native six-case run adds exact actual pointer y<=1px plus keyboard Enter open/Escape close/opener return; all6pass0retries/skips/flakes. Normal hooks22 exactownedfiles passed including canonical tasks/security keys/test guards. Retire this own completed plan privately; TASK13668AC3 and13656 early-stop stay open. Current source commit will exclude the still-in-progress cockpit E2E file/task13656 and its unresolved early-stop assertion; no failing test changes committed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed the actual Pro lost-click root: toolbar focus no longer expands an idle textarea and moves its action target; input expansion stays stable through internal action focus and resets outside. Real hook/textarea regressions and all affected caller fixtures complete. Both runtime profiles14suites405 tests pass0skip; four-root types0; lint13TS0new; actual artifact/tokens/budgets/assets pass; six native browser cases pass0retry/skip/flake, including pointer y<=1px and Enter/Escape/focus. Original9 conversations28rows and existing4rowfixture unchanged. Normal22-file hooks passed. Bandit N/A TS/Markdown scope. Final whole-PR review/streaming recovery/rebase/CI/merge remain separate.
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
