---
id: TASK-482
title: Plan Watchlists durable audio artifact projection implementation
status: Done
labels:
- watchlists
- plan
- audio
priority: high
documentation:
- Docs/superpowers/plans/2026-05-22-watchlists-durable-audio-artifact-projection-implementation-plan.md
modified_files:
- Docs/superpowers/plans/2026-05-22-watchlists-durable-audio-artifact-projection-implementation-plan.md
- backlog/tasks/task-482 - Plan-Watchlists-durable-audio-artifact-projection-implementation.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create the implementation plan for Watchlists durable audio artifact projection from the approved spec. The plan should break the work into test-first stages covering Workflows correlation metadata, request IDs/retry semantics, projection helper, /runs audio read-repair, frontend durable artifact graph, and verification.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] Implementation plan covers Workflows run metadata, definition metadata propagation, audio_request_id propagation, projection helper, lazy read-repair, retry stale-state handling, frontend graph rendering, optional proactive projection, and verification.
- [x] Plan includes test-first steps, exact file paths, expected failing/passing commands, and commit boundaries.
- [x] Plan was locally reviewed for correlation metadata, retry/request identity, target-user Collections DB routing, and whitespace hygiene.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Write a test-first implementation plan from the durable audio projection spec. The plan covers Workflows run metadata, audio_request_id propagation, artifact metadata tagging, projection helper, /runs audio lazy read-repair, retry stale-state handling, frontend durable graph rendering, optional proactive projection, and verification.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Created the durable audio artifact projection implementation plan at `Docs/superpowers/plans/2026-05-22-watchlists-durable-audio-artifact-projection-implementation-plan.md`.

Review hardening added:
- Durable Workflows run metadata is distinct from Workflow definition metadata used by adapter context.
- `audio_request_id` must not be read from user/job `output_prefs`.
- Trigger tests should cover stale user-supplied request IDs.
- `retry-audio` stale-state metadata updates need target-user Collections DB routing.

Additional pre-execution review added:
- Clarified that implementation tasks 1-7 are the MVP durable `/watchlists` UX; only proactive projection is the deferrable follow-up.
- Required `/runs/{run_id}/audio` to resolve Workflows DB through the same factory path used by Scheduler/Workflows API, not by manually constructing a target-user SQLite path.
- Required synchronous Watchlists/Collections/Workflows DB projection helpers to be called from async endpoints through `run_in_threadpool`.
- Required Workflow run metadata extraction to support run `metadata_json`, definition `metadata`, and legacy inputs.
- Required download URL generation to avoid unsupported `target_user_id` query params.

Verification: `git diff --check` passed for the plan changes. Bandit was not run because this task only adds documentation and Backlog task metadata. The writing-plans reviewer subagent was not dispatched because this session requires explicit user permission before using subagents; local plan review was performed instead.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implementation plan is complete. It is ready for execution as a staged PR sequence beginning with Workflows correlation metadata and request identity, followed by projection/read-repair, retry hardening, frontend rendering, and optional proactive projection only if a safe worker path is verified.
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
