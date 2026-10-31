---
id: TASK-15180
title: Keep chat model selection and composer semantics accessible at narrow sizes
status: Done
labels:
- chat
- ux
- accessibility
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Bounded Stage 4 client repair on latest dev: use existing Ant Design popup placement and viewport constraints, name search/sort controls, correct menu and textarea ARIA, and preserve Escape/focus behavior. No new popup or settings framework. Native geometry, VoiceOver, themes and packaged-extension journeys remain parent verification gates.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The shared model popup uses native viewport width/height limits and scrollable content with reachable controls; long model labels retain their full accessible name.
- [x] #2 Model trigger advertises the actual menu; both shared catalog and fallback controls have persistent search/sort names and Escape returns focus without losing the draft.
- [x] #3 The message textarea exposes its native multiline textbox semantics without unsupported aria-expanded; existing slash/mention and input behavior remain covered.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Approved master: Docs/Design/2026-10-04-chat-conversation-ux-remediation-design.md; parent TASK-13457. Baseline dev94854ca3db6ec3eaa0c27b4e9537bdd47d222611 integrated292ea526d1; starting HEAD8887a1cf23. Dedupe reviewed model dropdown/responsive work plus67 open PR path inventories (/private/tmp/chat-responsive-open-pr-paths.json); no source overlap, max PR task13461, local13506. Live IAB exposes unnamed sort and menu-role popup. Its viewport/screenshot APIs currently report0x0 even after reset/presentation; no native geometry claim. Use existing Dropdown/CSS/locale tooling and exact draft preservation. Implementation plan will be Docs/Plans/IMPLEMENTATION_PLAN_TASK_13507_chat_accessible_model_picker.md; Bandit N/A (TS/JSON/docs only).
Final verification: nine suites182 passed; actual Form departure/Escape/held returns6 focused passed; real Dropdown13 passed; Favorite Enter/Space add/remove regrouping passed. Logs /private/tmp/chat-accessible-model-final-regressions.log, chat-accessible-model-form-focus-held.log, chat-accessible-model-trigger-toggle-red.log and -green.log, chat-accessible-model-favorite-focus-red.log and -green.log. Final WebUI types, extension entrypoint/config compile, no-new touched ESLint, official English mirror agreement and diff checks pass. Two authorized independent source reviews clear final source; reviewers inspected source/logs and did not run tests. Remote dev94854ca remains integrated292ea526d1. Bandit N/A: TS/JSON/docs only, no Python implementation.
Completed the three-stage plan; removed only this unit's own temporary plan per repository guidance. Durable evidence and explicit parent limitations are in Docs/Design/2026-10-04-chat-conversation-ux-remediation-design.md. Browser AX names/open-focus/Escape verified, native geometry unavailable with0x0 IAB; native touch/zoom/themes/VoiceOver/package and complete journeys remain TASK-13457/13458. Form harness uses real callback/Dropdown/Tooltip and declared state/IO/other-control doubles, not a packaged shell. No whole-program completion.
Renumbered from TASK-13507 to TASK-15180 on 2026-10-31 (TASK-13960): TASK-13507 now names only "Use singular saved-item wording for one-item ingest results" (backlog/tasks/task-13507 - Use-singular-saved-item-wording-for-one-item-ingest-results.md). Before 2026-10-31 both tasks used TASK-13507; Docs/Development/Backlog_Task_Id_Renumbering.md lists every renumbered id.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Existing popup is viewport bounded and accessible; Search/Sort names, correct menu/multiline semantics and full model names are retained. Keyboard/pointer focus lifecycle repairs cover departure, nested Sort, Favorite regrouping, trigger close and actual Form external return without stealing outside focus. Final nine-suite182 checks, types, extension entrypoint compilation, locale mirror, no-new lint and diff checks pass; both independent reviews clear. Native geometry/assistive/package and remaining all-ten stage gates stay open; no new framework or dependency.
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
