---
id: TASK-551
title: Import sidepanel chat handoff in WebUI chat
status: In Progress
labels:
- chat
- extension
- implementation
priority: Medium
references:
- TASK-546
- TASK-547
- TASK-548
- TASK-549
- TASK-550
documentation:
- Docs/superpowers/specs/2026-05-29-sidepanel-chat-webui-handoff-design.md
- Docs/superpowers/plans/2026-05-29-sidepanel-chat-webui-handoff-implementation.md
modified_files:
- apps/packages/ui/src/components/Option/Playground/SidepanelImportedContextBanner.tsx
- apps/packages/ui/src/components/Option/Playground/hooks/useSidepanelChatHandoffImport.ts
- apps/packages/ui/src/components/Option/Playground/PlaygroundForm.tsx
- apps/packages/ui/src/components/Option/Playground/hooks/usePlaygroundSubmit.ts
- apps/packages/ui/src/components/Option/Playground/__tests__/sidepanel-chat-handoff-import.test.tsx
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Task 3 from the sidepanel chat WebUI handoff plan: make `/chat` read sidepanel handoff ids, prefill or resolve draft conflicts, render imported page context, include imported context in the next request's model message, clear/consume the handoff safely, and cover the import flow with focused tests.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

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
