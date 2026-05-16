---
id: TASK-393
title: Implement main /chat rail information architecture and action hierarchy
status: In Progress
assignee: []
created_date: '2026-05-16 00:30'
labels:
  - webui
  - chat
  - ux
  - frontend
dependencies:
  - TASK-391
documentation:
  - >-
    Docs/superpowers/specs/2026-05-15-main-chat-cockpit-maturity-roadmap-design.md
  - >-
    Docs/superpowers/plans/2026-05-15-chat-cockpit-rail-ia-action-hierarchy-plan.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement PR 2 of the post-merge main /chat cockpit maturity roadmap: reorganize the main WebUI /chat cockpit rails into predictable work surfaces without changing sidepanel/sidebar behavior. Preserve existing controls and shared handlers while improving first-time comprehension and returning-user scan efficiency.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Main /chat left rail groups context stack, prompt management, search/RAG sources, files/media, and session persistence with clear headings and compact states.
- [ ] #2 Main /chat right rail groups runtime state, model route/settings, assistant/persona, tools/MCP, and recovery controls with clear action hierarchy.
- [ ] #3 Existing rail controls, shared handlers, keyboard-accessible names, focus behavior, and focus-mode behavior are preserved.
- [ ] #4 First-time users can identify where to change prompt, persona/character, model, context, and tools without opening unrelated surfaces.
- [ ] #5 Focused Vitest coverage and real-server /chat Playwright proof are updated for the reorganized rail IA without mocked backend routes.
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Created the PR 2 implementation plan for rail information architecture and action hierarchy. Scope is the main WebUI /chat cockpit only. The plan preserves existing shared state/handlers, keeps focus mode and mobile rail tabs intact, excludes sidepanel/sidebar and model selector redesign work, and requires TDD plus real-server Playwright proof.
<!-- SECTION:NOTES:END -->
