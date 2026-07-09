---
id: TASK-12928
title: Update WebUI and extension sidepanel default items
status: Done
assignee: []
created_date: ''
updated_date: '2026-07-09 02:51'
labels: []
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Set the default sidepanel navigation items for the WebUI/browser extension to the requested ordered set: Quick Ingest, Chat, Prompts, Characters, Chat Dictionaries, World Books, Notes, Knowledge QA, Media, Document Workspace, Research Workspace, Kanban, Watchlists.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Default sidepanel shortcuts match the requested ordered list.
- [x] #2 Legacy and previous default persisted selections migrate to the requested list.
- [x] #3 Custom shortcut selections preserve saved order.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Updated shared UI settings default sidebar selection, preserved custom order during normalization, rendered sidebar shortcuts in saved order, and renamed the English Kanban shortcut label.

Verification: packages UI Vitest sidebar/header/settings tests passed; frontend route registry Vitest test passed; frontend typecheck passed; git diff --check passed. Bandit not applicable because touched runtime files are TypeScript/JSON and Backlog metadata only.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Updated WebUI/browser extension sidepanel defaults to the requested ordered set and added regression coverage for default migration, custom order preservation, sidebar render order, and route registry guard expectations.
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
