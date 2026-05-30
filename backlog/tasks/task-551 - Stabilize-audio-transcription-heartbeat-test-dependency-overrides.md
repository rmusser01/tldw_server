---
id: TASK-551
title: Stabilize audio transcription heartbeat test dependency overrides
status: In Progress
assignee: []
created_date: ''
updated_date: '2026-05-30 06:45'
labels:
  - ci
  - tests
  - audio
  - pr-2133
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fix the PR #2133 full-suite Audio test failure where the heartbeat create_task failure test leaks its asyncio.create_task monkeypatch into AuthNZ dependency initialization before the endpoint exercises the heartbeat branch.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The hotwords audio test helper bypasses AuthNZ DB/principal dependencies for endpoint-level tests.
- [x] #2 The failing heartbeat task startup regression test passes locally.
- [x] #3 Existing design-system dictionary verification remains unchanged.
- [ ] #4 PR #2133 CI no longer fails in the Audio full-suite module for this test.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
CI evidence from PR #2133 Full Suite Ubuntu/Python 3.11 showed `test_audio_transcriptions_sanitizes_heartbeat_task_start_failure_log` failing in the Audio module because the test-injected `asyncio.create_task` error reached AuthNZ database pool initialization. Updated `_setup_stubbed_audio_app` to override `get_auth_principal` and `get_db_transaction`, matching the existing retention/redaction audio endpoint test helper so the test remains focused on the endpoint heartbeat branch.

Local verification:
- `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest -q tldw_Server_API/tests/Audio/test_audio_transcriptions_hotwords.py::test_audio_transcriptions_sanitizes_heartbeat_task_start_failure_log` passed.
- `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest -q tldw_Server_API/tests/Audio/test_audio_transcriptions_hotwords.py` passed: 23 tests.
- `bunx vitest run src/components/Option/Dictionaries/__tests__/DictionaryVersionHistoryModal.design-system.test.tsx` passed: 2 tests.
- `git diff --check` passed.
- Bandit ran on the touched test file; remaining findings are existing low-severity test-file assert usage, with no B106 or medium/high findings after removing the token_type literal.

Follow-up Windows full-suite Audio artifact for PR #2133 showed `test_audio_transcriptions_sanitizes_heartbeat_jobs_failure_log` racing the heartbeat task on Windows. Replaced the 10ms sleep in the mocked jobs-start increment with an event-gated wait that is released by the failing heartbeat callback, so the test deterministically observes the sanitized heartbeat log before request cleanup can cancel the task.

Additional local verification:
- `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest -q tldw_Server_API/tests/Audio/test_audio_transcriptions_hotwords.py::test_audio_transcriptions_sanitizes_heartbeat_jobs_failure_log` passed.
- `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest -q tldw_Server_API/tests/Audio/test_audio_transcriptions_hotwords.py` passed: 23 tests.
- Bandit ran on touched Admin/Audio files; remaining findings are low-severity test assert usage only, with no B106 or medium/high findings.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
