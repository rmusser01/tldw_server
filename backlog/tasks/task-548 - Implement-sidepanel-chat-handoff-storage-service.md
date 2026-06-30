---
id: TASK-548
title: Implement sidepanel chat handoff storage service
status: Done
assignee: []
created_date: ''
updated_date: 2026-05-29 06:27
labels:
- chat
- extension
- implementation
dependencies: []
references:
- TASK-546
- TASK-547
documentation:
- Docs/superpowers/specs/2026-05-29-sidepanel-chat-webui-handoff-design.md
- Docs/superpowers/plans/2026-05-29-sidepanel-chat-webui-handoff-implementation.md
priority: medium
modified_files:
- apps/packages/ui/src/services/sidepanel-chat-handoff.ts
- apps/packages/ui/src/services/__tests__/sidepanel-chat-handoff.test.ts
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Task 1 from the sidepanel chat WebUI handoff plan: create a fail-closed extension-local handoff storage service with validation, payload bounds, one-time consume, route merge helpers, message-for-model composition, and focused Vitest coverage.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Storage service creates bounded sidepanel chat handoff packages and verifies read-back before returning the saved package.
- [x] #2 Storage failures, failed read-back, expired records, malformed records, and mismatched key/body ids fail closed without exposing handoff ids.
- [x] #3 Read, consume, cleanup, route merge, URL leakage, serialized storage values, metadata bounds, and message-for-model helpers have focused regression coverage.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented sidepanel-chat-handoff.ts and focused service tests. Spec review passed after serialized Plasmo getAll values were normalized before validation. Code-quality review passed after adding key/body id validation, page metadata/route bounds, route leakage assertions, and timer cleanup. Verification: bun run test src/services/__tests__/sidepanel-chat-handoff.test.ts passed with 11 tests; git diff --check d73fded19a..HEAD passed. Bandit is not applicable to TypeScript-only service/test changes; package-wide TypeScript check previously hit Node heap OOM and remains a known environment limitation for this slice.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Task 1 is complete. The sidepanel chat handoff storage service now stores short-lived, bounded, one-time handoff packages in local extension storage, validates read-back, tolerates serialized Plasmo `getAll()` values, removes expired/malformed/mismatched records, avoids URL payload leakage, and composes imported page context for the next model request. Focused Vitest coverage passes with 11 tests.
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
