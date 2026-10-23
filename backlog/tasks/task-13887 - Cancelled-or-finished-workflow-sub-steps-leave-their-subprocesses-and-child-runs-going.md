---
id: TASK-13887
title: Cancelled or finished workflow sub-steps leave their subprocesses and child
  runs going
status: In Progress
labels:
- workflows
- backend
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while planning TASK-13863, which makes a failed map or parallel step cancel its other sub-steps and fixed media_ingest's poll. Still open: (1) tts (adapters/audio/tts.py near line 235) and multi_voice_tts (adapters/audio/multi_voice_tts.py, 7 sites) start ffmpeg with asyncio.create_subprocess_exec and kill it only on TimeoutError from wait_for(proc.communicate()), so a cancel (the engine's step timeout, or a failed sibling) leaves ffmpeg running to its end without its timeout; a probe showed the child alive after the cancel and after asyncio.run closed the loop. run_checked_async in subprocess_utils already kills and reaps on cancel. (2) workflow_call with wait: true (usable inside parallel, which does not check parallelizable): a cancelled child run is left running for the orphan reaper, because the engine has no CancelledError handler; its own timeout_seconds already does this. (3) media_ingest returns as soon as the first file appears in its step directory, without waiting for yt-dlp to exit.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A cancelled tts or multi_voice_tts sub-step kills the ffmpeg process it started, with a test
- [ ] #2 A cancelled workflow_call child run ends in a terminal status instead of staying running, with a test
- [ ] #3 media_ingest waits for yt-dlp to exit, or terminates it, before it returns, with a test
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
