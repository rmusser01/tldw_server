---
id: TASK-546
title: Design explicit sidepanel chat WebUI handoff
status: In Progress
labels:
- chat
- extension
- ux
priority: Medium
references:
- TASK-531
- TASK-534
documentation:
- Docs/superpowers/specs/2026-05-29-sidepanel-chat-webui-handoff-design.md
modified_files:
- Docs/superpowers/specs/2026-05-29-sidepanel-chat-webui-handoff-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Document the approved explicit sidepanel-to-WebUI /chat handoff design: route-only launch remains default, while a separate Continue in WebUI action transfers the current sidepanel draft plus visible page context through a one-time short-lived handoff token.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/specs/2026-05-29-sidepanel-chat-webui-handoff-design.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Design direction approved in brainstorming: separate Continue in WebUI action, ephemeral token, composer prefill without auto-send, visible already-captured page context only, one-time short-lived consume, sidepanel draft preserved. Local spec review added non-overwrite behavior for existing WebUI drafts, role-play query merge rules, unguessable IDs, payload bounds, and malformed package rejection.
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
