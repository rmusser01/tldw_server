---
id: TASK-429
title: Add Character Chat DB health and recovery release gate
status: Done
assignee: []
created_date: ''
updated_date: '2026-05-19 04:00'
labels:
  - chat
  - characters
  - database
  - recovery
  - release-gate
dependencies: []
references:
  - Docs/Product/WebUI/Character_Chat_Roleplay_First_Class_PRD_2026_05_18.md
  - TASK-428
  - Docs/RELEASE_NOTES.md#unreleased
documentation:
  - Docs/Product/WebUI/Character_Chat_Roleplay_First_Class_PRD_2026_05_18.md
  - Docs/Operations/ChaChaNotes_DB_Recovery.md
  - Docs/RELEASE_NOTES.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Backend-focused release dependency for first-class Character Chat GA: detect corrupt per-user ChaChaNotes/chat databases at startup or diagnostics, identify affected DB and failure reason, provide a documented recovery/doctor path, and avoid silent data mutation. This is linked from Character Chat Phase 0 as the R11 dependency owner.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Startup or diagnostics identifies the affected per-user chat DB and failure reason when ChaChaNotes/chat DB integrity fails.
- [x] #2 A documented doctor/recovery path covers backup, SQLite integrity_check or recover, validation, and restore.
- [x] #3 Where safe, one corrupt per-user chat DB does not prevent setup, diagnostics, or recovery UI from loading.
- [x] #4 User-facing recovery copy avoids implying data was silently changed.
- [x] #5 Character Chat GA release notes link this task as resolved or explicitly release-blocking.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-05-19-character-chat-db-health-release-gate-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Extended ChaChaNotes init health with sanitized sqlite_corruption last_failure metadata and recovery docs pointer.
- Startup warm-up remains fail-open; corrupt DB warm-up records degraded health and does not escape an exception.
- Release notes traceability: Docs/RELEASE_NOTES.md#unreleased now links TASK-429 and PR #1862 as resolved for the Character Chat GA R11 backend recovery gate.
- Verification: python -m pytest tldw_Server_API/tests/API_Deps/test_chacha_notes_db_deps_error_mapping.py tldw_Server_API/tests/Health/test_readiness_health_sanitizers.py tldw_Server_API/tests/Services/test_startup_chacha_warmup.py -q (19 passed, after activating the project virtualenv).
- Security: source .venv/bin/activate && python -m bandit -r tldw_Server_API/app/api/v1/API_Deps/ChaCha_Notes_DB_Deps.py tldw_Server_API/app/services/startup_chacha_warmup.py -f json -o /tmp/bandit_character_chat_db_health.json (reported zero findings).
- Diff hygiene: git diff --check passed.
- Known blockers/skips: no known blockers for this backend release gate; no browser/UI test was run because this slice only changes backend health/recovery and documentation.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the Character Chat DB health release gate by extending the existing ChaChaNotes health snapshot with sanitized corruption details, recovery metadata, and no automatic data repair. Added tests covering corrupt DB diagnostics, aggregate /api/v1/health exposure, startup warm-up fail-open behavior, and path sanitization. Added the ChaChaNotes recovery guide, linked the R11 PRD backend gate, and added a draft release-note entry in Docs/RELEASE_NOTES.md#unreleased that marks TASK-429/PR #1862 as resolved for Character Chat GA. Verification: focused pytest passed (19 tests), Bandit passed with zero findings on touched production code using the command recorded in Implementation Notes, and git diff --check passed.
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
