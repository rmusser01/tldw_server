---
id: TASK-12127
title: 'Task 3: Chat macros storage, sync, and settings'
status: Done
labels:
- chat-macros
- task-3
modified_files:
- tldw_Server_API/app/core/Chat_Macros/storage.py
- tldw_Server_API/app/core/Chat_Macros/settings.py
- tldw_Server_API/app/core/Chat_Macros/output_profiles.py
- tldw_Server_API/app/core/Chat_Macros/service.py
- tldw_Server_API/tests/Chat_Macros/unit/test_macro_storage.py
- tldw_Server_API/tests/Chat_Macros/unit/test_macro_service.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement file-backed chat macro storage, service registry sync, settings, and output profile resolution for Chat Macros v1 Task 3.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented Task 3 chat macro storage/service foundation. Added file-backed user macro CRUD with conservative names, basename-only supporting files, symlink rejection, byte caps, and canonical digesting. Added settings/output profile helpers and ChatMacrosService for built-in wrapup listing, per-user disable state, user macro create/update/delete, validation, cloning, registry sync, core-command collisions, and bounded output profile overrides. Verification: red run failed on missing storage/output profile modules; new storage/service tests passed; existing parser/repository regressions passed; Bandit results: [].
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
