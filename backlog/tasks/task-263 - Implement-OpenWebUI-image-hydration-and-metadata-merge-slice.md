---
id: TASK-263
title: Implement OpenWebUI image hydration and metadata merge slice
status: Done
assignee: []
created_date: '2026-05-11 16:02'
updated_date: '2026-05-11 16:12'
labels:
  - chatbooks
  - openwebui
  - implementation
dependencies:
  - TASK-262
references:
  - >-
    Docs/superpowers/plans/2026-05-11-openwebui-attachment-hydration-implementation-plan.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Stage 3 of the OpenWebUI attachment hydration implementation plan: append safe hydrated images to existing chat messages, merge hydration status into OpenWebUI message metadata without losing original import metadata, and make image hydration retry-idempotent.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Post-insert message image append helper appends after existing image positions and enforces the existing message-image byte cap
- [x] #2 Hydration metadata updates deep-merge under extra.openwebui_import.hydration while preserving original OpenWebUI import fields
- [x] #3 Image hydration byte-sniffs PNG/JPEG/GIF/WebP, rejects oversized or unsupported bytes, and records structured statuses
- [x] #4 Retrying the same source key does not duplicate message_images rows
- [x] #5 Focused pytest, diff checks, and Bandit for touched backend code are recorded
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Follow Stage 3 of the implementation plan locally because subagent execution is blocked by account quota. Use TDD: extend hydration service tests for image/metadata behavior, verify red, add a ChaCha append_message_image helper plus facade export, implement image hydration and deep metadata merge in openwebui_hydration.py, run focused pytest/regression/diff/Bandit checks, then update task/plan and commit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented Stage 3 image hydration locally after subagent quota blocked delegation: added append_message_image with commit=False support, deep OpenWebUI hydration metadata merge, byte-sniffed image hydration, idempotent source-key handling, and rollback on metadata update failure.

Verification: pytest focused image/metadata slice passed 6 selected tests; full OpenWebUI Chatbooks regression set passed 35 tests; git diff --check passed; Bandit JSON at /tmp/bandit_openwebui_image_hydration.json had 0 results and 0 errors.

Documentation for this slice is the implementation plan/task tracking update; user-facing hydration docs remain in the later docs stage of the plan.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Stage 3 complete: OpenWebUI image attachments can now be hydrated into ChaCha message_images while preserving original import metadata, enforcing image byte limits/signature checks, avoiding duplicate hydration on retry, and rolling back image insertion if metadata recording fails.
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
