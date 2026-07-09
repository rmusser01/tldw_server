---
id: TASK-12924
title: Prevent TTS memory monitor failure from breaking app shutdown
status: Done
labels:
- bug
- tts
- shutdown
priority: medium
modified_files:
- tldw_Server_API/app/core/TTS/tts_resource_manager.py
- tldw_Server_API/tests/TTS/test_tts_resource_manager.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Investigate and fix TTSInsufficientMemoryError escaping from the TTS resource manager memory monitor during app shutdown.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 TTS memory monitor cleanup failure does not propagate into app shutdown.
- [ ] #2 A focused regression test covers the monitor task failure path.
- [ ] #3 Touched backend tests pass.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added a focused regression test for stop_monitoring after a background cleanup memory error. Updated MemoryMonitor._monitor_loop so TTSInsufficientMemoryError from best-effort cleanup is logged and does not break shutdown. Verification: red test failed before the fix; focused test passed after; tldw_Server_API/tests/TTS/test_tts_resource_manager.py and tldw_Server_API/tests/Services/test_shutdown_resource_cleanup.py passed 51 tests; tldw_Server_API/tests/TTS_NEW/unit/service/test_tts_resource_manager_stress.py passed 3 tests; git diff --check passed; Bandit results empty for tts_resource_manager.py; restarted backend and /api/v1/health returned status ok.
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
