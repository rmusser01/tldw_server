---
id: TASK-13129
title: Implement Scheduled Tasks Phase 4D.0F execution feasibility gate
status: To Do
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
- Docs/ADR/040-scheduled-agent-execution-feasibility.md
- Docs/Development/Scheduled_Agent_Execution_Certification.md
updated_date: 2026-08-24 17:58
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the Phase 4D.0F proof and fail-closed API/admission gate. Produce reproducible evidence for each configured deployment class, publish the ADR and current outcome, and keep Scheduled Tasks Agent automation creation unavailable on unsupported deployments and execution unavailable everywhere until both certification and the later execution stack are independently ready. This task does not implement Phase 4D execution.
<!-- SECTION:DESCRIPTION:END -->
## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A typed certification model defines deployment-class identity, the seven required evidence domains, freshness, stable reason codes, and the exact certified, draft_only, and unsupported outcome rules.
- [ ] #2 Reproducible manifests and probes cover isolation attestation, hostile boundary attempts, scheduled transcript leakage, exact dispatch-token recovery, monotonic terminal/approval/effect/cancellation evidence, brokered credentials and pre-action mediation, and operational installation/upgrade/health/fail-closed behavior.
- [ ] #3 The current repository baseline is evaluated honestly: statically ineligible runtimes are unsupported, eligible runtimes with missing or failed proof are draft_only, and no production resolver or artifact path can certify mocked, self-asserted, unsigned, stale, or wrong-subject evidence.
- [ ] #4 The Scheduled Tasks capabilities API reports a versioned Agent automation certification outcome, evidence identity/source/freshness, bounded reasons, and recovery without exposing raw evidence, paths, prompts, credentials, host details, or argument values.
- [ ] #5 Certification is necessary but not sufficient: Agent execute/run_now remain disabled, direct Run Now refuses, the scheduler does not arm or enqueue Agent definitions, and already-queued Agent Jobs cannot reach an executor until the later execution stack is independently ready; no environment flag can bypass this gate.
- [ ] #6 Unsupported deployments visibly refuse Agent preview-create, definition-create, and duplicate through the same typed API contract while existing definitions remain inspectable and safely manageable.
- [ ] #7 The ADR and operator guide document the evidence commands, trust boundary, retention/freshness, current outcomes, and the exact follow-on dependency tasks required before any deployment class can become certified.
- [ ] #8 No Scheduled Tasks Agent execution implementation, automatic migration activation, Watchlists behavior, standalone Agent Tasks behavior, or frontend execution control is added by this task.
- [ ] #9 Focused Scheduled Tasks, ACP, Sandbox, capability API, admission, helper, static-analysis, and Bandit gates pass, with host-gated skips and evidence limitations recorded explicitly.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Execute Docs/superpowers/plans/2026-08-24-scheduled-tasks-phase4d-feasibility-gate-implementation-plan.md only after user approval. Keep the certification/evidence change separate from TASK-13127 and all later Phase 4D execution work.
<!-- SECTION:PLAN:END -->

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
