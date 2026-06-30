---
id: TASK-2418
title: Harden Persona module review findings
status: Done
assignee: []
created_date: 2026-06-23 18:23
updated_date: 2026-06-24 04:24
labels:
- persona
- security
- review
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Verify and address validated review findings in the current Persona core module code. Scope: native visual import cleanup, visual candidate terminal review state, Persona Live focus/idempotency consistency, visual export asset bounds, and connection-template log redaction.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Validated findings have focused regression tests.
- [x] #2 Native visual import failures clean up partial packs/assets.
- [x] #3 Visual candidate accept/reject terminal state is enforced.
- [x] #4 Persona Live focus/idempotency consistency is hardened or invalidated with evidence.
- [x] #5 Visual export enforces asset size bounds before loading/writing large files.
- [x] #6 Connection template render errors do not log raw sensitive template values.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Verified findings and fixed validated issues.

Red verification: focused regression run failed before fixes for native import cleanup, candidate terminal review state, stale live focus reconciliation, oversized export preflight, and template log redaction.

Post-fix verification:
- `python -m pytest tldw_Server_API/tests/Persona/test_persona_visual_portability.py::test_export_pack_rejects_oversized_asset_before_archive_validation tldw_Server_API/tests/Persona/test_persona_visual_portability.py::test_import_commit_native_archive_cleans_up_failed_manifest_update tldw_Server_API/tests/Persona/test_persona_visuals_api.py::test_review_generated_candidate_terminal_status_cannot_be_changed tldw_Server_API/tests/Persona/test_persona_live_control_api.py::test_live_session_focus_reconciles_stale_focused_rows_after_missed_snapshot tldw_Server_API/tests/Persona/test_persona_connection_helpers.py::test_render_template_value_logs_warning_for_invalid_format -q --tb=short` passed 5 tests.
- `python -m pytest tldw_Server_API/tests/Persona/test_persona_visual_portability.py tldw_Server_API/tests/Persona/test_persona_visuals_api.py tldw_Server_API/tests/Persona/test_persona_live_control_api.py tldw_Server_API/tests/Persona/test_persona_connection_helpers.py -q --tb=short` passed 127 tests.
- `git diff --check` passed.
- `python -m bandit -r tldw_Server_API/app/core/Persona/visual_portability/importer.py tldw_Server_API/app/core/Persona/visual_portability/exporter.py tldw_Server_API/app/core/Persona/visual_service.py tldw_Server_API/app/core/Persona/live_control.py tldw_Server_API/app/core/Persona/connections.py tldw_Server_API/app/core/DB_Management/chacha/persona_state_store.py tldw_Server_API/app/api/v1/endpoints/persona.py -f json -o /tmp/bandit_persona_review_hardening.json` exited 0.

Additional live-control verification: added a concurrent create-new idempotency regression that failed before the lock (`len(set(created)) == 2`) and passed after serializing create/resume with the process-local live-session mutation lock.

Final current-code verification:
- `python -m pytest tldw_Server_API/tests/Persona/test_persona_live_control_api.py::test_live_session_create_idempotency_serializes_concurrent_requests -q --tb=short` passed 1 test.
- `python -m pytest tldw_Server_API/tests/Persona/test_persona_visual_portability.py tldw_Server_API/tests/Persona/test_persona_visuals_api.py tldw_Server_API/tests/Persona/test_persona_live_control_api.py tldw_Server_API/tests/Persona/test_persona_connection_helpers.py -q --tb=short` passed 128 tests.
- `git diff --check` passed.
- Bandit over touched backend Persona/API/DB files exited 0.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Rebased PR #2455 onto latest origin/dev, removing the unrelated inherited Claims commit from the PR diff. Addressed all validated review comments by hardening Persona Live focus conflict handling with retries, locking, and bounded reconciliation; adding the requested reconciliation comment and docstrings; using a Persona-specific export exception for new archive limit failures; and annotating/expanding regression tests. Verified with targeted tests, the 129-test focused Persona suite, git diff --check, and Bandit.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
- [x] #7 Focused tests pass.
- [x] #8 Bandit runs on touched Python files.
- [x] #9 Backlog task records verification and final summary.
<!-- DOD:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
PR #2455 follow-up: rebased onto latest origin/dev and validating review comments for remediation (focus conflict handling, exporter domain exception, missing docstrings, test type annotations, and focus reconciliation comment).
PR #2455 review follow-up completed. Addressed validated comments: rebased the branch onto latest origin/dev and dropped the unrelated inherited Claims commit from the PR diff; added bounded optimistic-concurrency retries for Persona Live preference updates; serialized focus mutations with the live-session mutation lock; reconciled focused rows in bounded passes and reverts target focus before surfacing unreconciled conflicts; added an explanatory focus reconciliation comment; changed new export limit failures to PersonaVisualPackExportError; added missing docstrings; annotated new live-control regression tests; added a regression for conflicted focused-row cleanup. Verification: targeted review tests passed 4 tests; focused Persona suite passed 129 tests; git diff --check passed; Bandit over touched Persona/API/DB paths exited 0 with only existing nosec warnings in persona_state_store.py.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
