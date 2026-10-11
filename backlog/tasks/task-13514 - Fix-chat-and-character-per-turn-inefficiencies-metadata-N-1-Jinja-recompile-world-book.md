---
id: TASK-13514
title: Fix chat and character per-turn inefficiencies (metadata N+1, Jinja recompile,
  world book)
status: To Do
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Batch 2. Plan: Docs/Plans/2026-10-06-perf-batch-2-chat-turn-path-implementation-plan.md. Batch message-metadata fetch + hoist per-call DDL (chat_service.py:4251, ChaChaNotes_DB.py:25451); Jinja template cache + identity short-circuit (chat_service.py:4771); default-character cache; token estimate computed once; overlap-trim hashing; moderation post-boundary only; bounded continuation walk; WorldBookService reuse + entry cache; character tail window + alias cache; streamed tool-args list buffer (streaming_utils.py:1460).
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
