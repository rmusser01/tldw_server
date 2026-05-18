---
id: TASK-429
title: Add Character Chat DB health and recovery release gate
status: To Do
labels:
- chat
- characters
- database
- recovery
- release-gate
priority: High
references:
- Docs/Product/WebUI/Character_Chat_Roleplay_First_Class_PRD_2026_05_18.md
- TASK-428
documentation:
- Docs/Product/WebUI/Character_Chat_Roleplay_First_Class_PRD_2026_05_18.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Backend-focused release dependency for first-class Character Chat GA: detect corrupt per-user ChaChaNotes/chat databases at startup or diagnostics, identify affected DB and failure reason, provide a documented recovery/doctor path, and avoid silent data mutation. This is linked from Character Chat Phase 0 as the R11 dependency owner.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Startup or diagnostics identifies the affected per-user chat DB and failure reason when ChaChaNotes/chat DB integrity fails.
- [ ] #2 A documented doctor/recovery path covers backup, SQLite integrity_check or recover, validation, and restore.
- [ ] #3 Where safe, one corrupt per-user chat DB does not prevent setup, diagnostics, or recovery UI from loading.
- [ ] #4 User-facing recovery copy avoids implying data was silently changed.
- [ ] #5 Character Chat GA release notes link this task as resolved or explicitly release-blocking.
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
