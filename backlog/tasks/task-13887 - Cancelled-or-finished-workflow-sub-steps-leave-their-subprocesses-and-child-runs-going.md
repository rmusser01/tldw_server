---
id: TASK-13887
title: Cancelled or finished workflow sub-steps leave their subprocesses and child
  runs going
status: Done
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
- [x] #1 A cancelled tts or multi_voice_tts sub-step kills the ffmpeg process it started, with a test
- [x] #2 media_ingest waits for yt-dlp to exit, or terminates it, before it returns, with a test
- [x] #3 workflow_call's cancel cleanup moves to TASK-13919, which repairs the adapter: on dev it creates no child run to leave running (it imports a module that does not exist, Workflows.workflows_db, and passes status= to create_run, which has no such parameter), recorded with the probe
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
AC 2 reworded (premise wrong, measured on e06eeec68c while planning; orchestration.py is unchanged on 617094d014 (the plan measured e06eeec68c; no file this plan touches changed between them)): the real adapter, with no injected module, returns {'error': 'workflow_call_error', 'result': None} for wait true and false, because import tldw_Server_API.app.core.Workflows.workflows_db fails ('No module named ...'), and WorkflowsDatabase.create_run has no status parameter. The module has never existed (git log --all). No child run is ever created, so there is none to leave running and no RED test is possible. TASK-13919 repairs workflow_call and carries the requirement (its AC 2).
AC 1 RED on 617094d014 (the plan measured e06eeec68c; no file this plan touches changed between them): test_audio_subprocess_cancellation.py gave 3 failed: ffmpeg still running after the cancel, for multi_voice_tts._normalize_audio and run_tts_adapter's normalization; and 8 direct asyncio spawn sites (tts.py:235, multi_voice_tts.py:64, :153, :192, :267, :304, :398, :488). Fix: all 8 call run_checked_async(cmd, timeout=..., check=False), which kills and reaps on cancel; subprocess.TimeoutExpired joins each module's noncritical tuple, so a timeout returns what dev returned. tts.py drops cwd=out_dir (both paths in its command are absolute). The AST guard also flags from-imports of the spawn functions and loop.subprocess_exec/subprocess_shell. GREEN: 140 passed with test_audio_adapters.py, test_tts_workflow_normalization.py, Watchlists/test_audio_briefing_workflow.py and TTS/unit/test_audio_converter_sanitizer.py; test_system_tts_wrapper_converts_requested_format now patches subprocess_utils.asyncio. No leftover fake ffmpeg (pgrep '^sleep 30$' empty).
AC 3 RED on 617094d014 (the plan measured e06eeec68c; no file this plan touches changed between them): -k 'media_ingest or download' gave 3 failed, 2 passed: ['clip.mp4.part'] == ['clip.mp4'] (returned at yt-dlp's .part file with yt-dlp running); [('timeout', None)] == [('download_failed', 1)] (a failed yt-dlp waited out timeout_seconds); and, through the engine, ('succeeded', None) == ('failed', '') (the run succeeded at the .part file). Fix: SubprocessTask keeps its Popen (proc); the poll waits for proc.poll(), so media_ingest returns only after yt-dlp exits; an exit the run's cancel caused returns cancelled; an exit with an error or with no file is download_failed with returncode. GREEN: 138 passed with test_media_adapters.py, test_orphan_requeue_unit.py, CI/test_async_blocking_call_guard.py and Web_Scraping/test_phase3_preflight_external_tools.py.
Ruling (implementation): the engine-level budget test now loads the adapter registry before the timed run (assert flow.get_adapter('media_ingest') is ingest.run_media_ingest_adapter). Its first RED run, at load average 37, failed for the wrong reason: the registry's one-off imports on the engine's first get_adapter took 5.2 s (the plan measured 1.1-1.5 s), past the step's 3 s budget, so dev's run also timed out and the child was killed before writing its .part file (assert [] == ['clip.mp4.part'], and AttributeError on .proc in the whole set). With the registry loaded first, RED at load 43-61 gave exactly the stated failures, and GREEN passes. A test-only change: the budget no longer depends on machine load. Separately, one GREEN run at load 53 timed out TASK-13863's test_map_cancels_the_other_items_when_one_raises in its 5 s wait_for during the same one-off imports (first test in the file); it passed alone and on the re-run (138 passed), and its path patches get_adapter, so this plan's code is not on it: load.
Download budget: owner Decision 1, approved 2026-10-23 with the plan's recommendation: keep the wait for yt-dlp and the 300 s default step budget, and document how to raise it (the step-level timeout_seconds, beside type; the WebUI editor has no control for it, so set it in the definition's JSON). The step-level timeout_seconds (default 300), shared by the step's sources, bounds the wait; a longer download is stopped and fails the step with step_timeout. Pinned by test_a_download_that_outlasts_the_steps_timeout_fails_the_step.
Docs: Usage_Quotas.md's 'A workflow' bullet and Docs/Design/Workflows.md's media_ingest bullet updated (no Published mirror); tests/Docs 212 passed. Bandit -ll on tts.py, multi_voice_tts.py, ingest.py and subprocess_utils.py: 0 results. Follow-ups: TASK-13919 (workflow_call), TASK-13920 (terminate_process on the event loop). Plan: Docs/superpowers/plans/2026-10-22-workflow-substep-cleanup-and-detached-chacha.md.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
A cancelled tts or multi_voice_tts step now kills and reaps the ffmpeg (or ffprobe, espeak) it started: all eight call sites go through run_checked_async, and an AST guard keeps direct asyncio spawns out of the workflow adapters. media_ingest now returns only after yt-dlp has exited, within the step's timeout_seconds, so a cancel at any point of a download stops it; a download that outlasts the budget fails the step with step_timeout, a failed download is reported as download_failed, and a download killed by the run's cancel reports cancelled. workflow_call creates no child run on dev at all (it imports a missing module), so its cancel requirement moved to TASK-13919, which repairs the adapter.
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
