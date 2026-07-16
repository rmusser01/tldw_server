---
id: TASK-12970
title: Add cooperative cancellation and partial-result finalization to Jobs
status: To Do
labels:
- jobs
- cancellation
- reliability
- research
priority: High
references:
- TASK-12964
- TASK-12968.4
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add a Jobs-level cooperative cancellation primitive for processing jobs so workers retain their lease, observe cancellation at bounded checkpoints, and can finalize cancellation with a sanitized partial result. Resolve the current race where JobManager marks a processing job terminally cancelled while its worker may continue persisting. This cross-cutting primitive is required by the Deep Research shared-discovery bridge and the separate Phase 2B HTML Media handoff design.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Queued jobs cancel and sanitize immediately, while processing jobs transition to a non-terminal cancellation-requested state without releasing their active lease.
- [ ] #2 Workers can observe cancellation at bounded checkpoints and atomically finalize cancelled with a bounded sanitized partial result and usage/accounting metadata.
- [ ] #3 Lease recovery terminalizes abandoned cancellation requests using the latest durable checkpoint without allowing the abandoned worker to persist after lease loss.
- [ ] #4 Cancellation, completion, failure, retry, and lease-recovery transitions are race-tested so exactly one terminal transition wins and terminal results remain immutable.
- [ ] #5 Existing Jobs consumers retain characterized cancellation behavior or explicitly migrate to the cooperative contract; API status and admin controls clearly distinguish requested from terminal cancellation.
- [ ] #6 Focused unit, integration, concurrency/property tests, database migration tests where needed, lint/format checks, Bandit, and diff hygiene pass with documented operational rollout.
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
