---
id: TASK-2392
title: Plan MCP UAT JSON-RPC and transport remediation
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-19 21:06'
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

User-requested design review pass incorporated seven additional constraints: post-auth JSON-RPC authorization status handling, intentional mounted batch contract update, raw envelope id-presence parsing, explicit trusted compatibility metadata names, FastAPI response-model adjustments, exact WebSocket keepalive allowlist, and bounded standalone scope. Focused reviewer pass then found three gaps, followed by two final gaps; all were patched. Final focused review approved the revised spec. Verification: git diff --check passed for the spec and task file. Bandit remains skipped because this revision is docs/backlog-only.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Created and revised the MCP UAT JSON-RPC and transport remediation design for one cohesive PR covering mounted tldw_server MCP and standalone MCP smoke/UAT surfaces. The final spec pins JSON-RPC envelope serialization, raw id-presence semantics, notification outcomes, malformed HTTP and batch behavior, HTTP status handling for post-auth JSON-RPC authorization failures, trusted compatibility auth metadata, response-model adjustments, WebSocket keepalive/failure handling, policy resolver fail-closed coverage, bounded standalone scope, smoke harness alignment, and validation sequencing.
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
