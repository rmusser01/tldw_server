---
id: TASK-491
title: Design OpenUI dynamic chat rendering support
status: In Progress
references:
- https://github.com/pewdiepie-archdaemon/odysseus/pull/151
documentation:
- Docs/superpowers/specs/2026-06-01-openui-dynamic-chat-rendering-design.md
modified_files:
- Docs/superpowers/specs/2026-06-01-openui-dynamic-chat-rendering-design.md
- backlog/tasks/task-491 - Design-OpenUI-dynamic-chat-rendering-support.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create an approved design spec for supporting OpenUI as the first renderer in a broader dynamic UI/artifact system across shared chat surfaces, including /chat, extension sidepanel chat, and workspace chat. Scope is design only; implementation planning follows after spec review and user approval.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Design spec written and reviewed for OpenUI as the first renderer in a shared dynamic UI chat rendering layer. V1 decisions: temporary /chat composer/request mode, frontend-tagged metadata preserved through existing persistence, final render before streaming preview, source fallback outside enabled surfaces, and validated OpenUI action round-trips as normal user messages with provenance metadata.
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
