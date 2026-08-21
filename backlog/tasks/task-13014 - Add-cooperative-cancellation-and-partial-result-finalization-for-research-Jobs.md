---
id: TASK-13014
title: Add cooperative cancellation and partial-result finalization for research Jobs
status: To Do
assignee: []
created_date: '2026-08-21 19:37'
labels:
  - jobs
  - cancellation
  - reliability
  - research
  - discovery
dependencies: []
references:
  - TASK-12964
  - TASK-12968.4
priority: high
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

<!-- SECTION:NOTES:BEGIN -->
Created as the unique research-discovery replacement for the ambiguous active TASK-12970 record. The superseded discovery record is archived at `backlog/archive/tasks/task-12970 - Add-cooperative-cancellation-and-partial-result-finalization-to-Jobs.md` after every discovery-specific reference and dependency was migrated; the unrelated Web_Scraping parent and children retain TASK-12970.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
