---
id: TASK-13863
title: Workflow map and parallel leave sibling sub-steps running after one fails
status: In Progress
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
- [ ] #1 map and parallel run their sub-steps in an asyncio.TaskGroup, or cancel the remaining tasks, so the first raised error cancels the siblings, with a test that records which siblings finished after the step raised
- [ ] #2 parallel with fail_fast off keeps collecting per-step errors, apart from an LLM quota refusal, which fails the step (TASK-13638)
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Plan: Docs/superpowers/plans/2026-10-20-workflow-map-parallel-cancel-siblings.md. One helper in control/flow.py gathers the sub-steps as tasks and, on the first exception or its own cancellation, cancels and awaits the rest, then re-raises. Not TaskGroup: ruff targets py39 and flags BaseExceptionGroup as F821. A cancelled media_ingest terminates its yt-dlp download. Measured on dev: map items 1-5 and parallel(fail_fast) steps 1-3 finished after the step raised; through the engine with on_failure, the five siblings wrote events and artifacts to the run after step_failed.
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
