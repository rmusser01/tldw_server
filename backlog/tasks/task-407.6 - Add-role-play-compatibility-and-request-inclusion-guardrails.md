---
id: TASK-407.6
title: Add role-play compatibility and request-inclusion guardrails
status: To Do
labels:
- chat
- ux
- roleplay
- stage-6
parent_task_id: TASK-407
documentation:
- Docs/superpowers/plans/2026-05-17-main-chat-role-play-preset-remediation-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Stage 6 implementation for the main /chat role-play preset plan: add compatibility tests and user-visible guardrails for cases where selected character/persona context is excluded from the outgoing request.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Compatibility helper mirrors request inclusion/exclusion axes from usePlaygroundRawPreview.
- [ ] #2 UI does not claim character/persona context will be sent when compare, image, docs/search, document context, selected knowledge, or file-retrieval RAG paths exclude it.
- [ ] #3 Guardrails cover character-vs-persona behavior and shared-component compatibility.
- [ ] #4 Focused Stage 6 tests, frontend checks, and browser verification are recorded.
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
