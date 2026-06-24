---
id: TASK-2403
title: Harden Agent Orchestration review findings
status: Done
assignee: []
created_date: 2026-06-23 18:10
updated_date: 2026-06-24 01:11
labels:
- backend
- acp
- hardening
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address current-code Agent_Orchestration review findings: rejected-review retry dispatch, per-user DB scoping, run terminal-state validation, ACP artifact size/count bounds, and legacy in-memory service cleanup or deprecation.

Review source: current Agent_Orchestration module review requested in Codex thread on 2026-06-23.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Rejected review retry tasks can be dispatched through a valid state-machine path without allowing duplicate active runs
- [x] #2 OrchestrationDB project/task/run/review and MCP server reads/writes are scoped to the owning user
- [x] #3 Run completion/failure updates reject missing runs and terminal-state rewrites
- [x] #4 ACP completion artifact parsing/promotion enforces explicit count and size limits
- [x] #5 Legacy in-memory service is removed or clearly isolated without stale architecture guidance
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implementation plan: Docs/superpowers/plans/2026-06-23-agent-orchestration-review-hardening-plan.md

Implemented all five review hardening stages from Docs/superpowers/plans/2026-06-23-agent-orchestration-review-hardening-plan.md. Verification: focused regression set passed cleanly with TEST_MODE=1 ULTRA_MINIMAL_APP=1 and --confcutdir; full Agent_Orchestration suite passed 204 tests, 2 warnings. Bandit touched backend scope reported 0 findings and 0 errors in /tmp/bandit_agent_orchestration_2403.json.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Hardened Agent Orchestration retry dispatch, per-user DB scoping, run terminal-state handling, ACP artifact bounds, and legacy service factory isolation. Full Agent_Orchestration tests passed and Bandit reported no findings.
Rebased PR #2438 onto latest `origin/dev` and addressed follow-up review comments by enforcing one running run per task in SQLite, making run terminal updates conditional, making workspace updates use static SQL inside transactions, moving active-run checks into `OrchestrationDB`, creating ACP sessions before running run rows, cleaning up failed sessions, bounding direct dict completion/review payloads, and adding focused regression coverage.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
- [x] #7 Focused pytest coverage demonstrates each fixed review finding
- [x] #8 Bandit runs on touched backend scope with no new findings
<!-- DOD:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Rebased PR #2438 onto latest origin/dev and addressing follow-up review comments: static workspace update SQL with transaction rollback, DB-enforced one-running-run invariant, session-before-run dispatch ordering with cleanup, bounded structured signal payloads, and focused regression tests.
PR #2438 follow-up verification after rebase: `TEST_MODE=1 ULTRA_MINIMAL_APP=1 python -m pytest --confcutdir=tldw_Server_API/tests/Agent_Orchestration tldw_Server_API/tests/Agent_Orchestration -q` passed 211 tests with 2 warnings. `python -m bandit -r tldw_Server_API/app/core/DB_Management/Orchestration_DB.py tldw_Server_API/app/api/v1/endpoints/agent_orchestration.py tldw_Server_API/app/core/Agent_Orchestration/completion_signals.py tldw_Server_API/app/core/Agent_Orchestration/artifact_promotion.py -f json -o /tmp/bandit_agent_orchestration_pr2438_followup.json` reported results=0 and errors=0. `git diff --check` passed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
