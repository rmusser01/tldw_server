---
id: TASK-2372
title: Design MCP Unified residual UX hardening after TASK-2393
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-30 03:04'
labels:
- mcp
- ux
- docs
- security
- standalone
dependencies: []
references:
- Docs/superpowers/plans/2026-06-26-mcp-unified-standalone-ux-remediation.md
- backlog/tasks/task-2393 - Plan-and-implement-MCP-Unified-standalone-UX-remediation.md
- Docs/superpowers/specs/2026-05-26-mcp-unified-standalone-library-gateway-design.md
- TASK-12054
- TASK-12055
- TASK-12064
- TASK-12065
- https://github.com/rmusser01/tldw_server/pull/2548
documentation:
- Docs/superpowers/specs/2026-06-28-mcp-unified-residual-ux-hardening-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create a residual hardening design for the remaining Unified MCP standalone/embedded UX findings after completed TASK-2393. Scope includes package-local gateway truthfulness, safer default high-risk capability opt-in, docs/admin/Docker contracts, WebSocket auth copy, known-error recovery, package gateway readiness status, and smoke/docs consistency. Full standalone gateway serving remains out of scope.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Residual design references completed TASK-2393 and does not reopen its completed scope.
- [x] #2 Design covers remaining latest-review findings with explicit in-scope and out-of-scope boundaries.
- [x] #3 Design includes safer high-risk defaults with migration and explicit opt-in guidance.
- [x] #4 Design includes compatibility guardrails for additive API/runtime changes and JSON-RPC compatibility.
- [x] #5 Design is reviewed and approved before implementation planning starts.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- 2026-06-28: Drafted residual hardening design in Docs/superpowers/specs/2026-06-28-mcp-unified-residual-ux-hardening-design.md.
- 2026-06-28: Spec review iteration 1 found missing client-doc coverage, explicit-config migration semantics, and publishing-status ambiguity. Patched all three.
- 2026-06-28: Spec review iteration 2 found HTTP error-shape compatibility risk and status-path ambiguity. Patched additive HTTP metadata language and explicit /api/v1/mcp/status vs package /status requirements.
- 2026-06-28: Spec review iteration 3 approved the design with no blocking contradictions or unsafe ambiguity.
- 2026-06-30: Finalized after PR #2548 merged. The approved residual design was converted into TASK-12054, implemented in TASK-12055, and verified through TASK-12064/TASK-12065 follow-ups. Full standalone gateway serving remains intentionally out of scope for this design/implementation line.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Residual MCP Unified UX hardening design is closed. The design referenced completed TASK-2393 without reopening its scope, covered the remaining standalone/embedded UX findings, specified safer high-risk module defaults and additive compatibility guardrails, and completed the review loop before implementation planning. The design was implemented by TASK-12055 and merged in PR #2548, with rebase/review/CI follow-up recorded in TASK-12064 and TASK-12065. No runtime verification or Bandit was required for this finalization because it only updates the design task record; implementation verification is recorded on the linked implementation tasks. Full standalone gateway serving remains out of scope.
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
