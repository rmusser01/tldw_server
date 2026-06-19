---
id: TASK-2392
title: Implement ACP support-safe Agent Tasks run summaries
status: In Progress
labels:
- ACP
- agent-orchestration
- github-2408
references:
- https://github.com/rmusser01/tldw_server/issues/2408
- https://github.com/rmusser01/tldw_server/issues/2398
modified_files:
- Docs/Development/ACP_Production_Readiness.md
- Docs/Development/Agent_Client_Protocol.md
- Docs/User_Guides/Integrations_Experiments/Getting_Started_with_ACP.md
- Docs/superpowers/plans/2026-06-19-acp-support-safe-task-run-summaries-plan.md
- Docs/superpowers/specs/2026-06-19-acp-support-safe-task-run-summaries-design.md
- apps/packages/ui/src/components/Option/AgentTasks/__tests__/AgentTasksPage.connection.test.tsx
- apps/packages/ui/src/components/Option/AgentTasks/index.tsx
- tldw_Server_API/app/api/v1/endpoints/agent_orchestration.py
- tldw_Server_API/tests/Agent_Orchestration/test_orchestration_api.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Track implementation for GitHub issue #2408: add a support-safe/redacted task run summary mode for Agent Tasks task detail so support/export surfaces can avoid prompt/result preview leakage while preserving operational run metadata.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Agent Tasks task detail can request redacted run summaries without changing default full-fidelity diagnostics.
- [ ] #2 Redacted run summaries preserve operational metadata including run status, stop reason, artifact count, diagnostic count, audit count, and session links.
- [ ] #3 Prompt/result previews and other run-summary free text are absent or replaced with a stable redacted sentinel in redacted mode.
- [ ] #4 Docs explain task-level redacted summaries versus ACP session redacted endpoints.
- [ ] #5 Backend and frontend contract tests cover the redacted summary behavior.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented GitHub issue #2408 for ACP support-safe Agent Tasks run summaries. `GET /api/v1/agent-orchestration/tasks/{task_id}` now accepts `run_summary_mode=full|redacted`, with full as the unchanged default. Redacted mode preserves operational metadata while replacing prompt/result previews, diagnostic messages/URIs, run errors/result summaries, failure-context free text, review-decision feedback, and top-level review feedback with `[redacted]`; ACP detail/events/artifacts links point to `?redacted=true` in redacted mode. Added frontend URL-builder support and docs/spec/plan updates.

Final verification after rebase onto `origin/dev`: `python -m pytest tldw_Server_API/tests/Agent_Orchestration/test_orchestration_api.py -q` passed 40 tests; `./node_modules/.bin/vitest run src/components/Option/AgentTasks/__tests__/AgentTasksPage.connection.test.tsx --maxWorkers=1 --no-file-parallelism` passed 12 tests after temporary `bun install` in `apps/` to repair local worktree dependency links, with generated Bun artifacts restored before staging; `python -m bandit -r tldw_Server_API/app/api/v1/endpoints/agent_orchestration.py -f json -o /tmp/bandit_acp_task_run_summaries_2408_post_rebase.json` reported zero findings; `git diff --check` passed.
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
