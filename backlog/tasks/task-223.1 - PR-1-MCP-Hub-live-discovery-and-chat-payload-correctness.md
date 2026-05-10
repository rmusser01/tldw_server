---
id: TASK-223.1
title: 'PR 1: MCP Hub live discovery and chat payload correctness'
status: To Do
assignee: []
created_date: '2026-05-10 06:13'
labels:
  - mcp
  - webui
  - backend
  - chat
dependencies: []
parent_task_id: TASK-223
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the first PR-sized remediation slice from the MCP Hub walkthrough. This phase should remove the backend restart requirement after managed external server setup and make chat MCP selection match the actual request payload.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Managed external server create, update, and import can trigger live external tool discovery refresh without backend restart.
- [ ] #2 The existing external.tools.refresh MCP tool validates arguments and no longer fails the write-tool pre-exec validator for valid calls.
- [ ] #3 MCP Hub setup and catalog surfaces report refresh success, refresh failure, and runtime unavailable states clearly.
- [ ] #4 Chat request construction and raw request preview use the same effective MCP tool decision and expose the reason when tools are omitted.
- [ ] #5 The readiness gate allows degraded but usable health into the app while preserving blocking behavior for unreachable or unhealthy API states.
- [ ] #6 Focused backend, frontend, and readiness tests cover the changed behavior.
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
