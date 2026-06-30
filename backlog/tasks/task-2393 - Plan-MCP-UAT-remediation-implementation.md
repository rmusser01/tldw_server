---
id: TASK-2393
title: Plan MCP UAT remediation implementation
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-19 21:16'
labels:
  - mcp
  - uat
  - planning
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create the implementation plan for the approved MCP UAT JSON-RPC and transport remediation spec. The plan should sequence test-first work across mounted tldw_server MCP, standalone MCP transports, smoke harness alignment, auth/RBAC compatibility, policy resolver import-cycle remediation, and full UAT validation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Implementation plan is written under Docs/superpowers/plans and references the approved spec.
- [x] #2 Plan maps exact files to responsibilities before task breakdown.
- [x] #3 Plan decomposes work into TDD-friendly tasks with commands and expected outcomes.
- [x] #4 Plan covers mounted and standalone UAT validation plus Bandit.
- [x] #5 Plan review loop is completed or documented with follow-up guidance.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implementation plan written at Docs/superpowers/plans/2026-06-19-mcp-uat-jsonrpc-transport-remediation.md and references approved spec Docs/superpowers/specs/2026-06-19-mcp-uat-jsonrpc-transport-remediation-design.md. Plan maps file responsibilities before task breakdown, uses TDD-friendly task steps with commands and expected outcomes, covers mounted and standalone UAT validation including live HTTP/WebSocket/stdio plus Bandit, and completed the plan review loop. First review found three issues; plan was patched for standalone live HTTP/WebSocket smoke, mounted WebSocket JWT smoke, and non-forgeable trusted auth metadata. Second review approved. Verification: git diff --check passed for plan file. Bandit skipped because this is docs/task-record only.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implementation plan saved at Docs/superpowers/plans/2026-06-19-mcp-uat-jsonrpc-transport-remediation.md. The plan references the approved design spec, maps file responsibilities, decomposes the work into TDD-friendly tasks, covers mounted and standalone UAT including live HTTP/WebSocket/stdio plus Bandit, and passed the plan review loop after one revision. Bandit was not run because this task changed only documentation and Backlog task metadata.
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
