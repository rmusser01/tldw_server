---
id: TASK-463.1
title: Gate Research Workspace grounded chat on queryable sources
status: Done
labels:
- research-workspace
- workspace
- chat
- source-status
priority: high
parent_task_id: TASK-463
references:
- Docs/superpowers/specs/2026-05-23-research-workspace-hard-replacement-roadmap-design.md
documentation:
- Docs/superpowers/plans/2026-05-24-research-workspace-queryable-chat-guard-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the Phase A chat guardrail so grounded/RAG chat uses only queryable selected Research Workspace sources. Users with selected but still-processing or failed sources should see why grounded mode is unavailable while general chat remains usable.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 RAG/grounded chat mode is enabled only when at least one selected effective source is queryable/ready.
- [x] #2 Selected processing or failed sources are visible in the composer context but do not populate RAG media ids.
- [x] #3 The chat input explains when selected sources are not queryable yet and keeps general chat available.
- [x] #4 Focused tests cover selected ready, processing, and failed source combinations.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented the Phase A grounded-chat guardrail in ChatPane by splitting visible source scope from queryable RAG scope. The composer now shows selected processing/failed sources with status labels while only ready/queryable selected sources can enable RAG mode or populate RAG media ids. When no selected source is queryable, RAG mode remains disabled, the input uses general-chat copy, and the user sees concise guidance that they can keep chatting generally while extraction/indexing finishes. Mixed ready/non-queryable selections keep RAG enabled only for queryable media and explain that non-queryable selected sources are excluded. The referenced task plan file is not present in this checkout, so implementation followed the Backlog acceptance criteria and the roadmap spec.

Verification: focused ChatPane/store/route Vitest suite passed 111 tests across 9 files. git diff --check passed. UI TypeScript with 8 GB heap still fails only on unrelated baseline errors in CharacterListContent.design-system.test.tsx and sidepanel-flashcards.test.tsx. Bandit skipped because this slice touched frontend TypeScript/tests and Backlog metadata only.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Research Workspace chat now gates grounded/RAG mode on queryable selected sources while preserving visible selected-source context for processing and failed sources. General chat remains available when selected sources are not queryable, and focused tests cover ready, processing, failed, and mixed source selections without adding any /workspace-playground route behavior.
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
