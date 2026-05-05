---
id: TASK-45.1
title: Create tldw_server design-system proof-surface implementation plan
status: Done
assignee: []
created_date: '2026-05-04 17:35'
updated_date: '2026-05-04 17:41'
labels:
  - docs
  - design-system
  - frontend
dependencies: []
documentation:
  - Docs/Design/tldw_web_design_system_contract.md
parent_task_id: TASK-45
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Write a repo-grounded implementation plan for the first design-system migration slice defined by the WebUI and extension design-system contract. The plan should be executable by a future agent without prior conversation context and should cover tokens, shared UI primitives, setup/recovery/admin-health proof surfaces, tests, and verification gates while keeping browser-extension compatibility in scope from day one.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A uniquely named implementation plan is added under Docs/superpowers/plans/ and follows the repo/superpowers plan format.
- [x] #2 The plan maps the v1 proof surface to concrete current files, expected component ownership, and sequencing.
- [x] #3 The plan includes TDD-style steps with exact focused test commands and expected results for each implementation stage.
- [x] #4 The plan explicitly includes WebUI and browser-extension compatibility checks, token aliasing, accessibility, and visual regression expectations.
- [x] #5 Docs-only verification is run and recorded; skipped runtime/security checks are justified if no executable code changes are made.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Inspect the approved design-system contract and current WebUI/shared-UI proof-surface files.
2. Write a single implementation plan document under Docs/superpowers/plans/ with exact file ownership, TDD steps, verification commands, and rollout checkpoints.
3. Self-review the plan against the contract and current repo topology; note that no reviewer subagent is used unless explicitly requested because tool policy restricts spawning agents.
4. Run docs-only verification on the plan and Backlog task, then update TASK-45.1 acceptance criteria and final summary.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Created Docs/superpowers/plans/2026-05-04-tldw-web-design-system-proof-surface-implementation-plan.md with a contract-bound proof-surface implementation sequence for state tokens, state registry, shared state primitives, recovery boundaries, setup/readiness gates, health/admin pages, WebUI compile, extension build, visual smoke checks, and final verification.

Self-reviewed the plan against Docs/Design/tldw_web_design_system_contract.md and current repo files under apps/packages/ui, apps/tldw-frontend, and apps/extension. No reviewer subagent was used because the available spawn-agent tool is restricted to explicit user requests for delegation/subagents.

Verification: git diff --check passed for the new implementation plan and TASK-45.1 Backlog task. rg checks confirmed required plan header, proof-surface paths, extension Tailwind inheritance, Button non-migration guidance, build verification, and Bandit skip guidance. Runtime tests and Bandit were not run because this task only added documentation/planning files.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added a concrete implementation plan for the tldw_server WebUI and browser-extension design-system proof surface. The plan sequences token aliases, a typed canonical state registry, shared state primitives, backend recovery, configuration/readiness gates, setup, health, /admin/server migration, parity checks, and final verification without broad Button migration or unrelated admin-route churn.
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
