---
id: TASK-13126
title: Design Scheduled Tasks Phase 4D Agent Task execution
status: Done
created_date: 2026-08-24 06:01
labels:
- scheduled-tasks
- phase-4d
- design
- ux
- api-first
- agent-task
priority: high
references:
- Docs/superpowers/specs/2026-06-09-scheduled-tasks-phase4-recurring-question-agent-task-api-contract-design.md
- Docs/superpowers/specs/2026-06-09-scheduled-tasks-phase4b-backend-api-foundation-design.md
- Docs/superpowers/specs/2026-06-30-scheduled-tasks-phase4c-recurring-question-execution-design.md
- Docs/ADR/003-jobs-vs-scheduler-default.md
- Docs/superpowers/specs/2026-08-24-scheduled-tasks-phase4d-agent-task-execution-design.md
dependencies:
- TASK-13127
updated_date: 2026-08-24 17:12
documentation:
- Docs/superpowers/specs/2026-08-24-scheduled-tasks-phase4d-agent-task-execution-design.md
modified_files:
- Docs/superpowers/specs/2026-08-24-scheduled-tasks-phase4d-agent-task-execution-design.md
- backlog/tasks/task-13126 - Design-Scheduled-Tasks-Phase-4D-Agent-Task-execution.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create the API-first product/UX and backend dependency design for Scheduled Tasks Phase 4D: safe scheduled Agent Task execution. Define agent selection, message/prompt configuration, risk preview, approval policy, dispatch, monitoring, transcripts/outputs, audit, result surfacing, and enterprise client behavior. Preserve the existing standalone Agent Tasks and Watchlists jobs/personas; treat the WebUI and extension as reference/main enterprise clients rather than the product boundary.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Spec is grounded in current Scheduled Tasks, ACP/API agent, Agent Tasks, Jobs, approvals, result, audit, RBAC, and capability-discovery behavior visible in the repository.
- [x] #2 Spec defines the Agent Task lifecycle, agent/target selection, message/prompt handling, schedule, preview and risk classification, approval modes, dispatch, retries/cancellation, run/output/transcript/audit contracts, result visibility, retention, and recovery behavior.
- [x] #3 Spec preserves API-first ownership, keeps the WebUI/extension as reference clients, and explicitly separates Scheduled Agent Tasks from the existing standalone Agent Tasks and Watchlists personas/jobs.
- [x] #4 Spec identifies backend dependencies and migration/compatibility constraints without assuming unsupported execution or storing raw Agent Task messages inline.
- [x] #5 Spec includes alternatives, proposed defaults, risks, open questions, acceptance criteria, and staged implementation recommendations suitable for a follow-up implementation plan.
- [x] #6 User reviews and approves the written design before implementation planning begins.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Repository discovery found existing side-effect-free automation execution from TASK-13020/TASK-13021/TASK-13022/TASK-13110. Agent Task execution remains planned because raw messages are redacted at rest and tool approvals are not durable for unattended runs. Baseline verification: 71/72 focused tests passed; `test_missing_definition_skips` failed deterministically because `create_scheduled_task_run` enforces definition existence before the consumer's missing-definition check. Tracked as separate dependency TASK-13127.

Approved brainstorming decisions consolidated into the written Phase 4D design: Scheduled Tasks-owned direct adapter dispatch; provider-neutral target refs with ACP first; encrypted prompt references and audited reveal; revision-bound bounded authority; fail-closed drift; phase-aware retry and confirmation-only cancellation; canonical Results/Home projection; reference-client IA and accessibility; automatic fenced legacy ACP schedule migration; staged API-first rollout.

Final structured review completed across three independent tracks: API/security, product UX/HCI/accessibility, and migration/reliability. All blocking findings were incorporated and each track approved the current spec. Key remediation includes route-complete RBAC, attested authority precedence, secure prompt/output handling, typed authorize-and-run and uncertainty recovery, independent lifecycle/schedule/activity/multi-attention state, Results-versus-Attention IA, one-time outcome mapping, surfacing compatibility, all-path migration activation fencing, restartable rollback, per-attempt execution fences, archive-safe Jobs idempotency, and monotonic cancellation-race ordering.

A later cross-section review found seven additional issues. The spec now gates M2 on certified execution plus durable run/Result/Attention and operator-recovery workflows; defines non-duplicating `noteworthy_only`, `every_run`, and `history_only` Result semantics; adds normative cross-resource retention/deletion/evidence rules; separates prompt reveal, plaintext copy, encrypted clone, and destructive deletion permissions; makes isolation feasibility a deployment-class go/no-go ADR; defines extension result/attention behavior; and uses stable resource-specific OpenAPI path parameter names. A final consistency pass preserved the API-first boundary by requiring WebUI parity only for bundled enterprise deployments while allowing equivalent supported API/CLI evidence for headless deployments, and aligned the extension summary with its compact updates role.

Fresh documentation verification passed: `git diff --check`, Markdown table pipe consistency, duplicate-heading scan, and targeted scans for generic `{id}` parameters, retired permission names, TODO/TBD/FIXME markers, and prohibited punctuation. Referenced files were previously verified. Bandit remains not applicable because this design slice changes Markdown/Backlog records only.

The user explicitly approved the revised Phase 4D design on 2026-08-24. Implementation planning may now proceed from the approved contract and staged plan shape.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Approved the API-first Scheduled Tasks Phase 4D Agent Task execution design. The final contract defines safe provider-neutral scheduled agent execution, secure prompt and transcript handling, granular authority and destructive-operation permissions, durable runs/results/attention/recovery, retention and deletion evidence, deployment-class feasibility certification, fenced legacy ACP schedule migration, reference WebUI and extension behavior, accessibility acceptance, and preservation of Watchlists and standalone Agent Tasks. The design was hardened through API/security, UX/accessibility, migration/reliability, and final cross-section reviews; all identified findings were addressed before user approval.
<!-- SECTION:FINAL_SUMMARY:END -->
## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Current repository evidence and affected contracts documented
- [x] #3 Spec review loop completed with blocking findings addressed
- [x] #4 Documentation verification recorded
- [x] #5 Bandit applicability or skip documented
- [x] #6 Final summary and known deferrals recorded
<!-- DOD:END -->
