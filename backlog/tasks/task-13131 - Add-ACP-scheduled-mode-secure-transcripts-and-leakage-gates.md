---
id: TASK-13131
title: Add ACP scheduled-mode secure transcripts and leakage gates
status: To Do
created_date: 2026-08-24 17:39
dependencies:
- TASK-13129
labels:
- scheduled-tasks
- phase-4d
- security
- acp
- privacy
priority: High
references:
- Docs/superpowers/specs/2026-08-24-scheduled-tasks-phase4d-agent-task-execution-design.md
- Docs/superpowers/plans/2026-08-24-scheduled-tasks-phase4d-feasibility-gate-implementation-plan.md
updated_date: 2026-08-24 17:55
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add an ACP scheduled execution mode whose prompt and secure output storage is tenant-scoped and protected, references opaque prompt_ref values in ordinary records, and cannot leak prompt sentinels through ordinary ACP detail, events, artifacts, fork, export, bootstrap, search, logs, errors, audit, or backups. Preserve ordinary interactive ACP for standalone Agent Tasks.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Scheduled-mode prompt events store prompt_ref and protected content without copying plaintext into ordinary ACP message fields.
- [ ] #2 Secure reads enforce the Phase 4D secure-output and prompt-reveal permissions plus required step-up, including prompt-echo representations.
- [ ] #3 Sentinel tests cover ordinary detail, events, artifacts, fork, export, bootstrap, search, logs, errors, audit, and backup/retention paths.
- [ ] #4 Failure, deletion, retention, and key-outage behavior fails closed without claiming deletion outside verified scope.
- [ ] #5 Ordinary interactive ACP and standalone Agent Tasks keep their existing transcript behavior and APIs.
- [ ] #6 Secure-transcript installation/migration, key rotation or outage, retention cleanup, health, and upgrade compatibility fail closed and publish bounded operator evidence.
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
