---
id: TASK-13129
title: Implement Scheduled Tasks Phase 4D.0F execution feasibility gate
status: In Progress
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
updated_date: 2026-08-27 02:19
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
Baseline dev SHA: 2306c1939f3b460f9c62da8ae83a1aa47c02ee0d. ADR-040 is occupied by synchronized moodboards; ADR-041 is reserved for this task. Reusable evidence sources: Sandbox runtime capabilities/operator evidence/operator status; ACP sandbox bridge/runner and session persistence/admin service; MCP managed credential broker; existing ACP certification smoke helper. Confirmed limitations: ordinary ACP stores raw prompts; ACP sandbox creation has no stable dispatch token/idempotency binding; cancellation lacks the required ordered per-attempt evidence journal; MCP credential brokering is partial and not Scheduled Tasks grant/action-token binding; no deployment class is currently certified. Statically eligible runtimes must remain draft_only until all seven server-verified domains pass; ineligible runtimes are unsupported.

Focused pre-change baseline on the recorded SHA: 171 passed, 0 failed, 0 skipped, and 19 warnings in 78.28 seconds across Scheduled Tasks automation API, ACP sandbox/session/runner, Sandbox runtime capabilities/operator evidence, and MCP slot-status tests. This validates reusable primitives only and is not Phase 4D.0F certification evidence. Two unrelated untracked Watchlists templates appeared in the isolated worktree and are intentionally excluded from this task.

Stage 1 TDD evidence: the new pure-domain suite first failed during collection because execution_certification.py did not exist. The implemented immutable domain now covers canonical deployment/profile identity, the seven closed requirements, exact subject/freshness/verification checks, authoritative bundle receipt matching, unsupported boundary outcomes, bounded reason codes, a repository-characterization-only current resolver, and the independent execution-stack conjunction. GREEN rerun: 31 passed, 0 failed, 2 warnings in 7.97 seconds. Ruff and compileall passed. Bandit report /tmp/bandit_task_13129_stage1.json contains 0 findings and 0 errors across 510 lines. git diff --check passed.
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
