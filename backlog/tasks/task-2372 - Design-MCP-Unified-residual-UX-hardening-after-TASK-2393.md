---
id: TASK-2372
title: Design MCP Unified residual UX hardening after TASK-2393
status: In Progress
assignee: []
created_date: ''
updated_date: '2026-06-28 03:22'
labels:
  - mcp
  - ux
  - docs
  - security
  - standalone
dependencies: []
references:
  - Docs/superpowers/plans/2026-06-26-mcp-unified-standalone-ux-remediation.md
  - >-
    backlog/tasks/task-2393 -
    Plan-and-implement-MCP-Unified-standalone-UX-remediation.md
  - >-
    Docs/superpowers/specs/2026-05-26-mcp-unified-standalone-library-gateway-design.md
documentation:
  - >-
    Docs/superpowers/specs/2026-06-28-mcp-unified-residual-ux-hardening-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create a residual hardening design for the remaining Unified MCP standalone/embedded UX findings after completed TASK-2393. Scope includes package-local gateway truthfulness, safer default high-risk capability opt-in, docs/admin/Docker contracts, WebSocket auth copy, known-error recovery, package gateway readiness status, and smoke/docs consistency. Full standalone gateway serving remains out of scope.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Residual design references completed TASK-2393 and does not reopen its completed scope.
- [ ] #2 Design covers remaining latest-review findings with explicit in-scope and out-of-scope boundaries.
- [ ] #3 Design includes safer high-risk defaults with migration and explicit opt-in guidance.
- [ ] #4 Design includes compatibility guardrails for additive API/runtime changes and JSON-RPC compatibility.
- [ ] #5 Design is reviewed and approved before implementation planning starts.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- 2026-06-28: Drafted residual hardening design in Docs/superpowers/specs/2026-06-28-mcp-unified-residual-ux-hardening-design.md.
- 2026-06-28: Spec review iteration 1 found missing client-doc coverage, explicit-config migration semantics, and publishing-status ambiguity. Patched all three.
- 2026-06-28: Spec review iteration 2 found HTTP error-shape compatibility risk and status-path ambiguity. Patched additive HTTP metadata language and explicit /api/v1/mcp/status vs package /status requirements.
- 2026-06-28: Spec review iteration 3 approved the design with no blocking contradictions or unsafe ambiguity.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Design spec written and approved by spec review loop; awaiting user review before implementation planning.
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
