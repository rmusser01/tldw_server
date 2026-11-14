---
id: TASK-13544
title: Keep the conversation visible when the chat composer expands
status: Done
labels:
- chat-ux
- webui
- extension
- layout
dependencies:
- TASK-13458
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The live latest-dev /chat character walkthrough at 1280x720 saves the correct greeting, but expanded Customize chat and setup notices reduce the transcript to 52px and scroll the greeting above its visible region. Bound the shared composer using existing CSS scrolling so a novice can read the opening and a power user can reach every control without changing messages, privacy, or history authority.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Trace full-page WebUI/extension and compact siblings and record causal native layout evidence before editing production code.
- [x] #2 Expanded, ordinary, and focused full chat retain readable timeline space with the composer and all existing controls reachable through scrolling and keyboard; floating transcript actions remain visible.
- [x] #3 Record native before/after dimensions and screenshots for desktop and compact widths, including greeting, roleplay follow-up, reload, and composer controls without claiming packaged-extension or assistive-technology results.
- [x] #4 Both client types, relevant existing behavior suites, effective touched-file lint, formatting and security scope assessment pass; the two authorized independent reviewers clear the repair.
- [x] #5 Update the parent audit/task with verified behavior and remaining D7/protocol/acceptance limits, then commit the bounded repair.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Duplicate searches composer height and timeline found no matching repair. Fresh task-ID maximum: local13543, origin/dev13530, open PR13526. Native proof /private/tmp/chat-native-timeline-collapse-13544-before.{json,jpg}; saved greeting independently matches API /private/tmp/chat-native-walkthrough-13543-greeting-receipt.json. Existing browser tab2/19769 only. ADR assessment: no new ADR; this is a bounded CSS correction to the accepted shared conversation UX, with ADR-049/050 history/lifetime and ADR-059 task tracking unchanged. Design/plan: Docs/Design/IMPLEMENTATION_PLAN_chat_visible_timeline_13544.md, accepted parent Docs/Design/2026-10-04-chat-conversation-ux-remediation-design.md. No protocol or persistence edits are authorized by this layout task.
Native keyboard check found the active /model row outside the slash-menu scrollport (scrollTop 0). Add active-option scrolling in the existing shared slash/mentions components. Both independent reviewers found default content-box observation mismatches occupied dock border-box accounting; cover padding-only delivery before changing the observation option. Broader matrix was stopped after a render loop in an unchanged cockpit-controls H1 test mock that recreates its fence each render; stabilize only that fixture and preserve production fences. Stopped run is not passing evidence.
Native320px Focus keyboard check exposed a remaining source root: keepComposerBottomInView aligns the full tall Form bottom on input/button focus, placing focused Send at y=-82 and capped dock scrollTop507.5. One production Form caller and four helper call paths traced. Replace manual whole-Form scrolling with existing native nearest scrolling of the current focused descendant; delayed calls ignore focus outside the composer. Preserve sticky guard and keyboard timings. Record before artifact chat-native-keyboard-visibility-13544-before; native evidence overrides prior source-clear review.
Completed TASK13544 shared dock/focus/menu repair against fresh origin/dev2c5f19d0328360d45bce99b8bae75ce252cc312f included in HEAD. Native1280x720 timeline52.22px before,217.60px after;60percent occupied dock cap with border-box observation and Strict Mode/retired callback protection. Native320 Focus old Send y=-82; current focused-descendant nearest scrolling keeps input/Improve/Send visible and preserves exact unsent trailing-space text;27 additional controls reachable by Tab. Shared compact debug390/320 loads the saved three-row Mira chat, cap432 of720px with448px actual padding, last slash option inside all clips;390 input-to-Send traversal visible. Preview outer-page scrolling is recorded and not packaged-extension proof. Temporary sticky preference restored false and viewport reset; full saved greeting is readable after scroll-to-opening with blank draft. Fresh ordinary4 and Mira5 UUID chats and final layout Mira3 survive actual server reload; original ordinary4/Mira5 hashes unchanged, receipt /private/tmp/chat-native-timeline-13544-integrity.json. Before has more character notices than after; not a matched notice-count experiment. Deterministic provider tests lifecycle, not live roleplay quality. Final500/500 in22suites/no skips; both types0;16 effective lint files147/145/0 added;15 owned format regions and diff checks pass. Final logs /private/tmp/chat-visible-timeline-13544-verified-{matrix,web-types,extension-types,static}.log. Both authorized source reviewers clear; textareaRef dependency included. Stale H1/actor receipt/effective-state/lazy action fixtures corrected without weakening production guards; interrupted and setup-failed runs excluded. Bandit N/A TS/TSX/docs/tasks only, no Python repo edits. ADR049/050/059 unchanged. Parent accepted design checkpoint contains native artifacts, NN/G implications and all limits. Four bounded plan stages complete; owned temporary plan retired with this repair. Old inaccessible discard dialog is gone, dismissal/positive acceptance unknown; TASK13540 AC4 remains open. TASK13458/all-ten program remain In Progress for packaged/MV3/docking/handoffs, physical/200percent zoom/VoiceOver, full theme/variant/large-history/session and participant journeys, remaining D7P1/P4/P5/settlement/provenance/lore/Syncv2. No new Chrome window/tab, dependency, persistence or authority behavior introduced.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Bound the shared complete composer dock with existing CSS scrolling, observe its occupied border box, reveal active slash/mention rows and the currently focused control instead of the full Form bottom. Native desktop/full320/compact390/320 checks,500 tests, both types, zero-added effective lint, owned formatting and two independent source reviews pass. Preserve history/privacy/owner behavior; detailed native evidence and unrun parent/D7 gates are recorded in the accepted design. Commit this bounded source/test/task unit; no PR, push or dev merge.
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
