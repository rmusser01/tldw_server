---
id: TASK-12148
title: Implement first-run MCP tool packs setup
status: In Progress
assignee: []
created_date: '2026-07-04 23:41'
labels:
  - implementation
  - mcp
  - setup
  - first-run
dependencies:
  - TASK-12132
references:
  - >-
    Docs/superpowers/plans/2026-07-04-first-run-mcp-tool-packs-implementation-plan.md
  - Docs/superpowers/specs/2026-07-04-first-run-mcp-tool-packs-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Execute the reviewed first-run MCP tool packs implementation plan using subagent-driven development. Scope includes backend catalog/apply/validate APIs, MCP Hub profile integration, frontend onboarding step, MCP Hub follow-up status, tests, verification, and commits.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Backend setup catalog, apply, validate, and admin recovery endpoints are implemented per plan.
- [ ] #2 Frontend onboarding MCP tools step and MCP Hub follow-up/recovery UI are implemented per plan.
- [ ] #3 Focused backend/frontend tests and touched-scope Bandit verification are recorded.
- [ ] #4 Implementation commits are reviewed with spec and code-quality subagent gates.
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
