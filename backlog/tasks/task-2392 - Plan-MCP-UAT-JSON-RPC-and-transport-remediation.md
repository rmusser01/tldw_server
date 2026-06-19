---
id: TASK-2392
title: Plan MCP UAT JSON-RPC and transport remediation
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-19 20:52'
labels:
  - mcp
  - uat
  - design
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create the approved design/spec for a cohesive PR that fixes discovered MCP UAT issues across mounted tldw_server MCP and standalone MCP smoke surfaces. Scope covers JSON-RPC envelope serialization, notifications, malformed request handling, auth/RBAC consistency, policy resolver import cycle, smoke contract alignment, and validation sequencing.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Design spec is written under Docs/superpowers/specs with the approved remediation sequence.
- [x] #2 Spec references mounted tldw_server MCP and standalone MCP UAT surfaces.
- [x] #3 Spec captures non-goals, risks, testing matrix, and sequencing decisions.
- [x] #4 Spec review loop is completed or documented with follow-up guidance.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Spec written at Docs/superpowers/specs/2026-06-19-mcp-uat-jsonrpc-transport-remediation-design.md. Automated spec-review loop ran three passes. Pass 1 found seven gaps; pass 2 found four remaining gaps; pass 3 found five final tightening items. All concrete findings were incorporated into the spec. A fourth reviewer was not dispatched because the brainstorming workflow caps the loop at three iterations. Verification: git diff --check passed for the spec and backlog task. Bandit skipped because this is docs/backlog-only with no code changes.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Created the MCP UAT JSON-RPC and transport remediation design for one cohesive PR covering mounted tldw_server MCP and standalone MCP smoke/UAT surfaces. The final spec pins JSON-RPC envelope serialization, id/null semantics, notification outcomes, malformed batch behavior, auth/RBAC compatibility guards, policy resolver fail-closed coverage, smoke harness alignment, and validation sequencing.
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
