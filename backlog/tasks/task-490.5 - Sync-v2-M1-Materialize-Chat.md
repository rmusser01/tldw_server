---
id: TASK-490.5
title: 'Sync v2 M1: Materialize Chat'
status: To Do
assignee:
- '@Codex'
labels:
- sync
- sync-v2
- m1
- chat
- backend
priority: high
parent_task_id: TASK-490
documentation:
- Docs/superpowers/specs/2026-05-23-chatbook-sync-v2-roadmap-prd-design.md
- Docs/superpowers/plans/2026-05-23-chatbook-sync-v2-m1-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Chat domain materialization for chat.conversation metadata and chat.message append/tombstone behavior through DB_Management-owned ChaChaNotes helpers, including whole-object conversation conflicts and stable-message-ID dedupe/conflicts.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 chat.conversation envelopes create/update/tombstone usable server chat metadata.
- [ ] #2 chat.message envelopes append messages, dedupe same stable ID and payload_hash, and preserve both versions plus a conflict for same stable ID with different payload_hash.
- [ ] #3 Message tombstones soft-delete messages without deleting conversations unless a conversation tombstone exists.
- [ ] #4 Chat materializer and ChaChaNotes helper tests pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-05-23-chatbook-sync-v2-m1-implementation-plan.md#task-5-implement-chat-materialization
<!-- SECTION:PLAN:END -->

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
