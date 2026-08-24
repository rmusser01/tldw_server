---
id: TASK-13112
title: Design Scheduled Tasks Phase 4D Agent Task execution
status: In Progress
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
- TASK-13113
updated_date: 2026-08-24 16:42
documentation:
- Docs/superpowers/specs/2026-08-24-scheduled-tasks-phase4d-agent-task-execution-design.md
modified_files:
- Docs/superpowers/specs/2026-08-24-scheduled-tasks-phase4d-agent-task-execution-design.md
- backlog/tasks/task-13112 - Design-Scheduled-Tasks-Phase-4D-Agent-Task-execution.md
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
- [ ] #6 User reviews and approves the written design before implementation planning begins.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Repository discovery found existing side-effect-free automation execution from TASK-13020/TASK-13021/TASK-13022/TASK-13110. Agent Task execution remains planned because raw messages are redacted at rest and tool approvals are not durable for unattended runs. Baseline verification: 71/72 focused tests passed; `test_missing_definition_skips` failed deterministically because `create_scheduled_task_run` enforces definition existence before the consumer's missing-definition check. Tracked as separate dependency TASK-13113.

Approved brainstorming decisions consolidated into the written Phase 4D design: Scheduled Tasks-owned direct adapter dispatch; provider-neutral target refs with ACP first; encrypted prompt references and audited reveal; revision-bound bounded authority; fail-closed drift; phase-aware retry and confirmation-only cancellation; canonical Results/Home projection; reference-client IA and accessibility; automatic fenced legacy ACP schedule migration; staged API-first rollout.

Final structured review completed across three independent tracks: API/security, product UX/HCI/accessibility, and migration/reliability. All blocking findings were incorporated and each track approved the current spec. Key remediation includes route-complete RBAC, attested authority precedence, secure prompt/output handling, typed authorize-and-run and uncertainty recovery, independent lifecycle/schedule/activity/multi-attention state, Results-versus-Attention IA, one-time outcome mapping, surfacing compatibility, all-path migration activation fencing, restartable rollback, per-attempt execution fences, archive-safe Jobs idempotency, and monotonic cancellation-race ordering.

Documentation verification: referenced files exist; Markdown table pipe counts are consistent; no TODO/TBD/FIXME, prohibited punctuation, or trailing-whitespace findings; `git diff --no-index --check` passed for the untracked spec. Bandit skipped because this design slice changes Markdown/Backlog records only and no Python scope. The spec is ready for the required user review; implementation planning remains gated on acceptance criterion #6.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [x] #2 Current repository evidence and affected contracts documented
- [x] #3 Spec review loop completed with blocking findings addressed
- [x] #4 Documentation verification recorded
- [x] #5 Bandit applicability or skip documented
- [ ] #6 Final summary and known deferrals recorded
<!-- DOD:END -->
