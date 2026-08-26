---
id: TASK-13128
title: Plan Scheduled Tasks Phase 4D prerequisite and feasibility implementation
status: Done
created_date: 2026-08-24 17:22
dependencies:
- TASK-13126
labels:
- scheduled-tasks
- phase-4d
- implementation-plan
- agent-task
- api-first
priority: high
references:
- Docs/superpowers/specs/2026-08-24-scheduled-tasks-phase4d-agent-task-execution-design.md
- tldw_Server_API/app/core/Scheduled_Tasks/agent_task_jobs.py
- tldw_Server_API/app/core/DB_Management/Scheduled_Tasks_DB.py
- tldw_Server_API/tests/Notifications/test_agent_task_jobs_consumer.py
- TASK-13127
- TASK-13129
- TASK-13130
- TASK-13131
- TASK-13132
- TASK-13133
documentation:
- Docs/superpowers/plans/2026-08-24-scheduled-tasks-phase4d-prerequisite-implementation-plan.md
- Docs/superpowers/plans/2026-08-24-scheduled-tasks-phase4d-feasibility-gate-implementation-plan.md
modified_files:
- Docs/superpowers/plans/2026-08-24-scheduled-tasks-phase4d-prerequisite-implementation-plan.md
- Docs/superpowers/plans/2026-08-24-scheduled-tasks-phase4d-feasibility-gate-implementation-plan.md
- backlog/tasks/task-13128 - Plan-Scheduled-Tasks-Phase-4D-prerequisite-and-feasibility-implementation.md
- backlog/tasks/task-13129 - Implement-Scheduled-Tasks-Phase-4D.0F-execution-feasibility-gate.md
- backlog/tasks/task-13130 - Add-scheduled-execution-isolation-attestation-and-hostile-runtime-proof.md
- backlog/tasks/task-13131 - Add-ACP-scheduled-mode-secure-transcripts-and-leakage-gates.md
- backlog/tasks/task-13132 - Add-ACP-dispatch-recovery-and-monotonic-execution-evidence.md
- backlog/tasks/task-13133 - Add-scheduled-execution-identity-credentials-and-pre-action-mediation.md
updated_date: 2026-08-24 19:48
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Write the first two executable implementation plans derived from the approved Phase 4D Agent Task execution design: the narrow TASK-13127 missing-definition prerequisite fix and the Phase 4D.0F deployment-class execution feasibility/certification gate. Keep later execution, migration, and client implementation out of these plans except as explicit gated follow-on work.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The TASK-13127 plan identifies exact current files, contracts, failing test behavior, red-green steps, focused regression commands, Bandit scope, and a reviewable commit boundary.
- [x] #2 The Phase 4D.0F plan defines exact evidence-producing tasks for isolation attestation, hostile-agent probes, scheduled transcript leakage, adapter idempotency, terminal/cancellation evidence, credential mediation readiness, operational health, and the certification ADR.
- [x] #3 Both plans preserve the API-first boundary and explicitly avoid reducing Watchlists or standalone Agent Tasks.
- [x] #4 Plan tasks use exact file paths, interfaces, test commands, expected failures/results, and small independently reviewable commits without placeholders.
- [x] #5 A self-review maps every approved prerequisite and 4D.0F requirement to a plan task and records any later-phase deferral explicitly.
- [x] #6 User reviews and approves the implementation plans before execution begins.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Map current TASK-13127 consumer/storage/test behavior and write its narrow test-first implementation plan.
2. Map current sandbox, ACP transcript, Jobs/adapter, credential, cancellation, health, and ADR patterns relevant to 4D.0F.
3. Write the separate Phase 4D.0F evidence and certification implementation plan.
4. Self-review both plans against the approved spec, verify documentation structure, and present them for user approval.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Planning review mapped current implementation evidence before defining work. TASK-13127 will preserve the Scheduled Tasks run foreign-key invariant, preflight the owner-scoped definition, and return skipped with run_id=null/reason=definition_missing; no invalid run, notification, or definition audit is fabricated. Phase 4D.0F is a separate API-first certification gate, not execution implementation. Current ACP/Sandbox/MCP primitives are partial: ordinary ACP stores raw prompts, ACP sandbox creation passes no idempotency key and persists no dispatch token, cancellation has no per-attempt monotonic journal, and MCP credential brokering does not provide scheduled subject/grant/attestation binding. The plan therefore expects draft_only or unsupported, never forced certification. Four exact dependency tasks (TASK-13130 through TASK-13133) were pre-created so later security work remains independently reviewable.

Final self-review against rebased origin/dev (2c6553c4ed) addressed these plan defects before approval: certification is necessary but not sufficient for execution; capability state is versioned and includes evidence/recovery/freshness; direct Run Now, scheduler arming/fire, and stale-Job worker admission fail closed; unsupported creation/duplicate routes cannot bypass capability guidance; evidence manifests omit argument values and bind the full isolation profile; worktree venv commands resolve the shared repository venv; ADR-040 updates the canonical index; tests follow the existing Notifications test layout; final Backlog evidence is committed rather than left dirty; and each security dependency has an operational fail-closed acceptance criterion. Documentation-only planning verification passed path, heading, fenced-block, placeholder, scope, and git whitespace checks. Bandit was not applicable because this task changes only Markdown planning/task records. User approval was received before implementation began.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Approved implementation plans were produced for the TASK-13127 missing-definition prerequisite and the Phase 4D.0F execution feasibility gate. The plans are based on origin/dev 2c6553c4ed, include exact TDD, API/admission, evidence, ADR, security, and verification steps, and preserve Watchlists plus standalone Agent Tasks. Self-review corrected certification/readiness conflation, direct-API bypasses, evidence-manifest leakage, operational acceptance gaps, ADR indexing, test placement, and worktree environment commands. Execution proceeds with TASK-13127 first; TASK-13129 remains dependent on its merge.
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
