---
id: TASK-418.7
title: Plan WebUI operations integrations implementation
status: Done
labels:
- ux
- design
- webui
- extension
- planning
- operations
- integrations
priority: High
parent_task_id: TASK-418
documentation:
- Docs/superpowers/specs/2026-05-17-webui-extension-ux-remediation-program-design.md
- Docs/superpowers/plans/2026-05-17-webui-extension-ux-remediation-implementation-plan.md
modified_files:
- Docs/superpowers/plans/2026-05-17-webui-operations-integrations-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Documentation-only child implementation plan for the approved WebUI/extension UX remediation program Task 10. Scope maps findings F4, F9, F12 support, F17 support, and F18 support into a reviewable plan for operations, automation, integration, source, workflow, admin, and MCP route jobs without product code changes in this task.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] Child plan saved at `Docs/superpowers/plans/2026-05-17-webui-operations-integrations-implementation-plan.md`.
- [x] Plan covers `/admin`, `/mcp-hub`, `/sources`, `/connectors`, `/integrations`, `/scheduled-tasks`, `/watchlists`, `/workflow-editor`, and `/skills`.
- [x] Plan maps findings `F4`, `F9`, `F12 support`, `F17 support`, and `F18 support` to route-level implementation tasks.
- [x] Plan distinguishes frontend-only state cleanup from backend capability-map work.
- [x] Plan preserves the documentation-only boundary for this task and does not modify product frontend or backend code.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Created a route-specific implementation plan for operations, automation, integrations, admin, MCP, sources, watchlists, workflow, connectors, and skills routes.
- Included an explicit route-job contract to test route ownership, capability mode, diagnostics policy, placeholder policy, and backend-gate boundaries before changing UI code.
- Split implementation into small reviewable tasks: route contract, sources and scheduled tasks, integrations and connectors, admin entry, MCP and workflow and skills, watchlists, and browser QA.
- Used code evidence from current route registry, Next admin and connector pages, route wrappers, Sources, Scheduled Tasks, Integrations, MCP Hub, Watchlists, Workflow Editor, Skills, and Server Admin components.
- Verification run:
  - `rg -n "T[O]D[O]|T[B]D|F[I]XME|\\.\\.\\.|\\bm[a]ybe\\b|\\bpr[o]bably\\b|\\bshould c[o]nsider\\b" Docs/superpowers/plans/2026-05-17-webui-operations-integrations-implementation-plan.md` exited 1 with no matches.
  - `rg -n "[[:blank:]]$|[^\\x00-\\x7F]" Docs/superpowers/plans/2026-05-17-webui-operations-integrations-implementation-plan.md` exited 1 with no matches.
  - `git diff --check -- Docs/superpowers/plans/2026-05-17-webui-operations-integrations-implementation-plan.md` exited 0.
  - Node route and file coverage check exited 0 with `coverage ok`.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed the documentation-only WP10 child implementation plan for the WebUI/extension UX remediation program. The plan defines route inventory, frontend-only versus backend-gated scope, non-goals, scoped files, route-job metadata, TDD tasks, Playwright browser QA, and verification commands for improving operations and integrations routes without changing product code in this task.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip: skipped because this task changed Markdown planning and Backlog documentation only.
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented: no product implementation or browser QA was run because this task only writes the implementation plan.
<!-- DOD:END -->
