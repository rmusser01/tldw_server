---
id: TASK-551
title: Stabilize full-suite backend flakes blocking Evaluations alert PR
status: In Progress
labels:
- ci
- tests
- backend
- flake
- TASK-45.44.5.2-followup
priority: High
references:
- https://github.com/rmusser01/tldw_server/pull/2134
modified_files:
- tldw_Server_API/app/services/admin_system_ops_service.py
- tldw_Server_API/app/api/v1/API_Deps/Audit_DB_Deps.py
- tldw_Server_API/tests/Admin/test_admin_invitations.py
- tldw_Server_API/tests/Audio/test_audio_transcriptions_hotwords.py
- tldw_Server_API/tests/Audit/test_audit_db_deps.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
PR #2129 review-fix CI exposed unrelated full-suite backend flakes on Windows/macOS after the EvaluationsPage alert migration. Stabilize the failing tests or underlying behavior without broadening the UI migration logic.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Windows/macOS full-suite failure causes are addressed or explicitly documented as unrelated baseline if not safely fixable.
- [x] #2 Focused tests for the failing Admin, Audio, and Audit cases pass locally.
- [ ] #3 PR #2134 check rollup is rerun after fixes or status is documented if remote CI remains pending.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- PR #2129 full-suite Windows log showed unrelated backend failures after the EvaluationsPage alert migration: `TestListInvitations::test_list_sorted_newest_first`, `test_audio_transcriptions_sanitizes_heartbeat_jobs_failure_log`, and `test_schedule_service_stop_clears_flag_on_failure`.
- Invitation listing now preserves newest-first behavior even when two invitations share the same `created_at` timestamp by using creation order as a tie-breaker.
- Audio heartbeat log coverage now waits briefly for the background heartbeat task to emit the sanitized failure log, avoiding scheduler timing races on slower runners.
- Audit service stop cleanup now treats `LookupError` from `service.stop()` as a handled stop failure and the async cleanup test waits with a bounded timeout for background cleanup.
- Opened follow-up PR #2134 from `codex/task-549-full-suite-flakes` after rebasing onto `origin/dev`; the branch diff now contains only this stabilization work on top of the merged Evaluations alert PR.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Opened PR #2134 from `codex/task-549-full-suite-flakes` after rebasing onto `origin/dev`. The branch isolates the full-suite flake fixes exposed by PR #2129: deterministic invitation ordering for tied `created_at` timestamps, less race-prone audio heartbeat failure-log coverage, handled audit stop `LookupError` cleanup, and a bounded audit cleanup wait in tests. Review feedback on PR #2134 has been addressed by using reverse + stable sort for invitation ties and moving the audio heartbeat polling inside the `TestClient` lifetime. Verification passed locally: Admin invitation suite 32 tests, Audio transcription hotword suite 23 tests, Audit DB deps suite 17 tests, `py_compile` on touched Python files, `git diff --check`, and Bandit on touched backend code with 0 findings. Remote PR check rollup is pending rerun after push.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
