---
id: TASK-591
title: Implement standalone MCP gateway admin auth and credential grant management
status: To Do
assignee: []
created_date: ''
updated_date: '2026-06-02 02:05'
labels:
  - mcp-unified
  - standalone-gateway
  - security
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add a package-owned admin auth seam plus credential-grant metadata manager, FastAPI routes, and CLI commands for the standalone MCP gateway. Keep secret material out of persistence and responses.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Management routes can require standalone admin auth without importing tldw_Server_API.
- [ ] #2 JSON-RPC /request and /ws routes are not accidentally gated by admin auth.
- [ ] #3 Credential grants support list/show/create/patch/delete through manager, FastAPI, and CLI.
- [ ] #4 Credential grants persist only broker references, slots, scopes, metadata, and provenance; secret-looking values are rejected or omitted before persistence.
- [ ] #5 External-server delete guards continue to block deletion when enabled grants reference the server.
- [ ] #6 Focused tests and Bandit on touched package files are recorded.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-06-02-mcp-gateway-admin-auth-credential-grants.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->
<!-- SECTION:FINAL_SUMMARY:END -->

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
