---
id: TASK-2265
title: Implement chat Mermaid card artifact rail support
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-07 00:06'
labels:
  - chat
  - mermaid
  - webui
  - artifacts
  - implementation
dependencies: []
references:
  - TASK-2264
  - Docs/superpowers/specs/2026-06-06-chat-mermaid-card-artifact-rail-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add opt-in chat artifact rail actions for assistant Mermaid diagram blocks, preserving shared Markdown defaults and QuickChat behavior while opening main chat diagrams as existing diagram artifacts.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Assistant Mermaid diagram blocks can open the existing chat artifact rail as diagram artifacts when artifact actions are enabled.
- [x] #2 Markdown defaults Mermaid artifact actions off so QuickChat and shared fallback surfaces keep render-only behavior.
- [x] #3 Main chat and compact chat assistant Mermaid messages opt into artifact actions and pass context-aware ids for jump-to-source.
- [x] #4 Tests cover artifact payload, source anchoring, opt-in forwarding, main-chat opt-in, and QuickChat non-opt-in behavior.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/specs/2026-06-06-chat-mermaid-card-artifact-rail-design.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented TDD red/green flow for Mermaid chat cards. Red run failed on missing artifact action, prop forwarding, and context-aware ids. Green run passed the focused Vitest suite. TypeScript full UI check was attempted with an 8GB heap and failed on an unrelated existing TaskActivityNotice i18next count type error outside the touched files. Bandit is not applicable because no Python code changed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented opt-in Mermaid artifact rail actions for main chat while preserving shared Markdown defaults and QuickChat render-only behavior.
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
