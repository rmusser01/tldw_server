---
id: TASK-485
title: Fix /chat rails regression coverage and sidepanel handoff target
status: In Progress
labels:
- webui
- chat
- sidepanel
- ux
priority: High
documentation:
- Docs/superpowers/specs/2026-05-31-chat-siderail-collapse-design.md
modified_files:
- Docs/superpowers/specs/2026-05-31-chat-siderail-collapse-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
PR 1 from the fresh /chat UX re-evaluation: preserve restored /chat cockpit rails and fix the directly connected sidepanel chat handoff so visible sidepanel actions target WebUI /chat with draft/context handoff instead of /options.html#/chat.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Stage 0/1: add failing regression coverage for /chat desktop rails and sidepanel handoff route; trace existing model selector/handoff controls; implement the minimal route/visible-action changes so sidepanel Continue in WebUI opens /chat?handoff=<id>; verify with focused tests and browser smoke.

Design addendum 2026-05-31: collapsed /chat siderails must disappear from layout and leave same-side edge-mounted expand buttons. Left chat rail collapse releases left width and shows a left-edge expand button. Right artifact rail collapse releases right width and shows a right-edge expand button when an artifact is available. Both collapsed states must keep chat/composer vertically anchored and visibly recoverable from both edges.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Design spec committed and reviewed through the brainstorming spec-review loop. Initial review flagged ambiguous 768-1023px behavior and shared OptionLayout scope. Spec was updated to scope edge-mounted expand buttons to lg-and-wider /chat side-rail behavior, preserve md/tablet behavior, and require /chat/Playground-scoped layout changes. Re-review status: Approved.

Follow-up spec review clarified two planning details before implementation: the right-edge expand button is only visible when an active artifact exists, and browser verification must include layout measurements for chat width, chat-shell top stability, and composer bottom docking.
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
