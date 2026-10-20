---
id: TASK-13863
title: Workflow map and parallel leave sibling sub-steps running after one fails
status: Done
labels:
- workflows
- quotas
- backend
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
run_map_adapter and run_parallel_adapter (Workflows/adapters/control/flow.py) use asyncio.gather without return_exceptions for map, and for parallel with fail_fast. That raises the first error but does not cancel the other sub-steps: those already started, and those queued behind the concurrency limit, run on after the engine has marked the step and the run failed (probe in the TASK-13638 plan review: map items 1-5 finished after the step raised). Since TASK-13638 every content, NLP and bibliography sub-step raises on an LLM quota refusal, so a map over flashcard_generate reaches this. The siblings' own LLM calls are refused before reaching a provider, but non-LLM sub-steps complete with their side effects.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 map and parallel run their sub-steps in an asyncio.TaskGroup, or cancel the remaining tasks, so the first raised error cancels the siblings, with a test that records which siblings finished after the step raised
- [x] #2 parallel with fail_fast off keeps collecting per-step errors, apart from an LLM quota refusal, which fails the step (TASK-13638)
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Plan: Docs/superpowers/plans/2026-10-20-workflow-map-parallel-cancel-siblings.md. One helper in control/flow.py gathers the sub-steps as tasks and, on the first exception or its own cancellation, cancels and awaits the rest, then re-raises. Not TaskGroup: ruff targets py39 and flags BaseExceptionGroup as F821. A cancelled media_ingest terminates its yt-dlp download. Measured on dev: map items 1-5 and parallel(fail_fast) steps 1-3 finished after the step raised; through the engine with on_failure, the five siblings wrote events and artifacts to the run after step_failed.
RED on origin/dev: 8 failed, 1 passed. map: items [1, 2, 3, 4, 5] ran after the step raised; parallel(fail_fast), refusal and HTTP 500: [1, 2, 3]; parallel without fail_fast: the step waited for its siblings (the test's 5 s bound expired); error collection: an HTTPException(500) left None in results, and so did a sub-step's own CancelledError; media_ingest: terminate_process never called after a cancel; engine with on_failure: map_item_finished events after step_failed. The pass is the step-timeout guard, which dev already met. GREEN: 10 passed (9 new and the existing map integration test), three runs. Mutants: cancel-without-await fails 4 tests, dropping the CancelledError clause 1, dropping media_ingest's terminate 1. tests/Workflows: 1730 passed, 6 skipped. Test-quality ratchet exit 0; ratchet test + tests/Docs 213 passed; ruff: nothing new (only the I001s ingest.py and the map integration test already have on dev); bandit -ll: clean.
Decision 1, recommended option: fail_fast keeps today's meaning (a noncritical sub-step error is recorded, not raised). Not TaskGroup: ruff targets py39 and reports BaseExceptionGroup as F821; the helper gathers tasks and cancels the rest on the first exception, as AC1 allows. Follow-ups: TASK-13886 (map runs a sub-step twice on a noncritical error); TASK-13887 (ffmpeg in tts and multi_voice_tts, workflow_call child runs, media_ingest's early return).
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
map and parallel run their sub-steps through one helper. On the first exception, or its own cancellation, it cancels the other sub-steps, running or queued, waits for them to stop and re-raises, so no sub-step writes to the run after the step fails. A cancelled media_ingest terminates the download it started. parallel without fail_fast still collects sub-step errors, now including those outside the noncritical tuple and a sub-step's own CancelledError, which dev dropped as None; an LLM quota refusal fails the step and cancels the rest. ffmpeg started by the TTS adapters and workflow_call child runs still outlive a cancel (TASK-13887). Usage_Quotas.md and the workflow design note say so.
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
