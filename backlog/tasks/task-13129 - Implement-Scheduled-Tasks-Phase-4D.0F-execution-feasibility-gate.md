---
id: TASK-13129
title: Implement Scheduled Tasks Phase 4D.0F execution feasibility gate
status: Done
created_date: 2026-08-24 17:33
dependencies:
- TASK-13127
labels:
- scheduled-tasks
- phase-4d
- agent-task
- certification
- security
- api-first
priority: High
references:
- Docs/superpowers/specs/2026-08-24-scheduled-tasks-phase4d-agent-task-execution-design.md
- Docs/ADR/010-sandbox-vz-runtime-ownership.md
- Docs/ADR/026-security-outbound-egress-and-ssrf-policy.md
- TASK-13130
- TASK-13131
- TASK-13132
- TASK-13133
documentation:
- Docs/superpowers/plans/2026-08-24-scheduled-tasks-phase4d-feasibility-gate-implementation-plan.md
- Docs/ADR/041-scheduled-agent-execution-feasibility.md
- Docs/Development/Scheduled_Agent_Execution_Certification.md
updated_date: 2026-08-27 03:22
modified_files:
- Docs/ADR/041-scheduled-agent-execution-feasibility.md
- Docs/ADR/README.md
- Docs/Development/Scheduled_Agent_Execution_Certification.md
- Docs/Evidence/Scheduled_Agent_Execution/2026-08-24-phase4d0f-baseline.json
- Docs/Evidence/Scheduled_Agent_Execution/2026-08-24-phase4d0f-baseline.md
- Docs/superpowers/plans/2026-08-24-scheduled-tasks-phase4d-feasibility-gate-implementation-plan.md
- Helper_Scripts/Testing-related/scheduled_agent_execution_certification.py
- backlog/tasks/task-13129 - Implement-Scheduled-Tasks-Phase-4D.0F-execution-feasibility-gate.md
- backlog/tasks/task-13130 - Add-scheduled-execution-isolation-attestation-and-hostile-runtime-proof.md
- backlog/tasks/task-13131 - Add-ACP-scheduled-mode-secure-transcripts-and-leakage-gates.md
- backlog/tasks/task-13132 - Add-ACP-dispatch-recovery-and-monotonic-execution-evidence.md
- backlog/tasks/task-13133 - Add-scheduled-execution-identity-credentials-and-pre-action-mediation.md
- tldw_Server_API/app/api/v1/endpoints/scheduled_tasks_control_plane.py
- tldw_Server_API/app/api/v1/schemas/scheduled_tasks_automation_schemas.py
- tldw_Server_API/app/core/Scheduled_Tasks/agent_task_jobs.py
- tldw_Server_API/app/core/Scheduled_Tasks/execution_certification.py
- tldw_Server_API/app/services/scheduled_task_automation_scheduler.py
- tldw_Server_API/app/services/scheduled_task_automation_service.py
- tldw_Server_API/tests/Helper_Scripts/test_scheduled_agent_execution_certification.py
- tldw_Server_API/tests/Notifications/test_agent_task_jobs_consumer.py
- tldw_Server_API/tests/Notifications/test_automation_definition_feed.py
- tldw_Server_API/tests/Notifications/test_scheduled_task_automation_api.py
- tldw_Server_API/tests/Notifications/test_scheduled_task_execution_certification.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the Phase 4D.0F proof and fail-closed API/admission gate. Produce reproducible evidence for each configured deployment class, publish the ADR and current outcome, and keep Scheduled Tasks Agent automation creation unavailable on unsupported deployments and execution unavailable everywhere until both certification and the later execution stack are independently ready. This task does not implement Phase 4D execution.
<!-- SECTION:DESCRIPTION:END -->
## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A typed certification model defines deployment-class identity, the seven required evidence domains, freshness, stable reason codes, and the exact certified, draft_only, and unsupported outcome rules.
- [x] #2 Reproducible manifests and probes cover isolation attestation, hostile boundary attempts, scheduled transcript leakage, exact dispatch-token recovery, monotonic terminal/approval/effect/cancellation evidence, brokered credentials and pre-action mediation, and operational installation/upgrade/health/fail-closed behavior.
- [x] #3 The current repository baseline is evaluated honestly: statically ineligible runtimes are unsupported, eligible runtimes with missing or failed proof are draft_only, and no production resolver or artifact path can certify mocked, self-asserted, unsigned, stale, or wrong-subject evidence.
- [x] #4 The Scheduled Tasks capabilities API reports a versioned Agent automation certification outcome, evidence identity/source/freshness, bounded reasons, and recovery without exposing raw evidence, paths, prompts, credentials, host details, or argument values.
- [x] #5 Certification is necessary but not sufficient: Agent execute/run_now remain disabled, direct Run Now refuses, the scheduler does not arm or enqueue Agent definitions, and already-queued Agent Jobs cannot reach an executor until the later execution stack is independently ready; no environment flag can bypass this gate.
- [x] #6 Unsupported deployments visibly refuse Agent preview-create, definition-create, and duplicate through the same typed API contract while existing definitions remain inspectable and safely manageable.
- [x] #7 The ADR and operator guide document the evidence commands, trust boundary, retention/freshness, current outcomes, and the exact follow-on dependency tasks required before any deployment class can become certified.
- [x] #8 No Scheduled Tasks Agent execution implementation, automatic migration activation, Watchlists behavior, standalone Agent Tasks behavior, or frontend execution control is added by this task.
- [x] #9 Focused Scheduled Tasks, ACP, Sandbox, capability API, admission, helper, static-analysis, and Bandit gates pass, with host-gated skips and evidence limitations recorded explicitly.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Execute Docs/superpowers/plans/2026-08-24-scheduled-tasks-phase4d-feasibility-gate-implementation-plan.md only after user approval. Keep the certification/evidence change separate from TASK-13127 and all later Phase 4D execution work.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Baseline dev SHA: 2306c1939f3b460f9c62da8ae83a1aa47c02ee0d. ADR-040 is occupied by synchronized moodboards; ADR-041 is reserved for this task. Reusable evidence sources: Sandbox runtime capabilities/operator evidence/operator status; ACP sandbox bridge/runner and session persistence/admin service; MCP managed credential broker; existing ACP certification smoke helper. Confirmed limitations: ordinary ACP stores raw prompts; ACP sandbox creation has no stable dispatch token/idempotency binding; cancellation lacks the required ordered per-attempt evidence journal; MCP credential brokering is partial and not Scheduled Tasks grant/action-token binding; no deployment class is currently certified. Statically eligible runtimes must remain draft_only until all seven server-verified domains pass; ineligible runtimes are unsupported.

Focused pre-change baseline on the recorded SHA: 171 passed, 0 failed, 0 skipped, and 19 warnings in 78.28 seconds across Scheduled Tasks automation API, ACP sandbox/session/runner, Sandbox runtime capabilities/operator evidence, and MCP slot-status tests. This validates reusable primitives only and is not Phase 4D.0F certification evidence. Two unrelated untracked Watchlists templates appeared in the isolated worktree and are intentionally excluded from this task.

Stage 1 TDD evidence: the new pure-domain suite first failed during collection because execution_certification.py did not exist. The implemented immutable domain now covers canonical deployment/profile identity, the seven closed requirements, exact subject/freshness/verification checks, authoritative bundle receipt matching, unsupported boundary outcomes, bounded reason codes, a repository-characterization-only current resolver, and the independent execution-stack conjunction. GREEN rerun: 31 passed, 0 failed, 2 warnings in 7.97 seconds. Ruff and compileall passed. Bandit report /tmp/bandit_task_13129_stage1.json contains 0 findings and 0 errors across 510 lines. git diff --check passed.
Stage 2/3 evidence: the sanitized helper now has 22 passing tests and exact JSON/Markdown pair validation, including a typed runtime-eligibility appendix. The API-first gate regression matrix passed 102 tests with 11 warnings; Ruff, compileall, and Bandit passed with zero findings. Agent execution remains disabled at capability discovery, direct Run Now, scheduler arm/fire/rearm, and queued-job worker admission; Recurring Questions remain unchanged.

Stage 4 decision evidence: ADR-041 accepts that no current deployment class is certified. Baseline evidence `sha256:1df8024b73472ea0a02a323fbad0d2f864d8b5f604611cb01bf49478f60a5874` for deployment class `sha256:76a1074c303c74cd6db3f6823f391133e44437a0da019f99f5b02b95b2cb3337` is `draft_only`, contains exactly seven missing repository-characterization domains and seven value-free command templates, and passes exact pair/prohibited-content validation. TASK-13130 through TASK-13133 now own the four bounded domain slices; `operational_fail_closed` is a cross-cutting exit criterion for this gate and all four dependencies.
Final verification: the 12-file cross-module matrix passed 275 tests with 0 failures, 0 skips, and 19 warnings in 93.04 seconds. Compileall and Ruff passed. Bandit `/tmp/bandit_task_13129.json` reports 0 findings and 0 errors across 4,965 production lines. Artifact-pair validation and both prohibited-content scans passed. Branch-wide diff review found no frontend, Watchlists, standalone Agent Tasks, migration activation, or execution-stack implementation.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the API-first Phase 4D.0F Scheduled Agent execution feasibility gate. Added the immutable seven-domain certification model, sanitized evidence generator and baseline, versioned capability projection, and consistent fail-closed Run Now, scheduler, and worker admission. ADR-041 records that no current deployment class is certified: eligible runtimes remain `draft_only`, ineligible runtimes are `unsupported`, and execution remains unavailable until a separate reviewed stack is ready. Attached the accepted decision and baseline to TASK-13130 through TASK-13133 without starting those dependency slices. Verification completed with 275 passing tests, compileall, Ruff, zero Bandit findings, artifact validation, redaction scans, and branch-scope review.
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
