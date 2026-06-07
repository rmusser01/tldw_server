---
id: TASK-2298
title: Add MCP effective permission preview surface
status: Done
labels:
- mcp
- policy
- filesystem
- admin
- followup
references:
- Docs/superpowers/specs/2026-06-07-mcp-fs-patch-write-safe-edit-tools-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add an admin/API/CLI surface that explains effective path permissions for a profile, tool, action, and workspace-relative path. The preview should reuse the path-enforcer decision contract and report safe fields such as requested action, normalized path, grant source, matched grant/effect, outcome, and denial reason without absolute paths or file content.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] Resolve the caller's effective MCP Hub policy for persona/group/org/team/workspace context.
- [x] Reuse the existing `McpHubPathEnforcementService` path-decision contract.
- [x] Return redacted preview fields for tool, action, workspace-relative path, outcome, reason, selected assignment/profile, grant source/outcome/effect, path-scope mode, workspace id, allowlist prefixes, and path decisions.
- [x] Avoid exposing absolute filesystem paths or file contents in the preview response.
- [x] Cover service-level redaction and API route wiring with regression tests.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Added `EffectivePermissionPreviewRequest` / `EffectivePermissionPreviewResponse`, `POST /api/v1/mcp/hub/effective-permission-preview`, and `McpHubPathEnforcementService.preview_effective_path_permission()`.

The preview wrapper normalizes workspace-relative paths, invokes the existing enforcer using a synthetic path-boundable filesystem tool definition, and emits only redacted workspace-relative decision metadata. Standalone gateway CLI/admin simulation remains a separate follow-up under `TASK-2303`.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the MCP Hub effective permission preview surface for path-scoped tool calls. The endpoint resolves the authenticated user's effective policy context, delegates path/action/path evaluation to the existing path enforcer, and returns a redacted operator/debug payload for policy troubleshooting.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded: `python -m pytest tldw_Server_API/tests/MCP_unified/test_mcp_hub_path_enforcement_service.py tldw_Server_API/tests/MCP_unified/test_mcp_hub_policy_api.py` -> 60 passed
- [x] #3 Documentation updated when relevant: schema and route tests cover this API-surface slice
- [x] #4 Bandit run for touched code: `python -m bandit -r <touched production files> -f json -o /tmp/bandit_mcp_effective_permission_preview.json` -> 0 findings
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented: standalone gateway CLI/admin simulation remains covered by `TASK-2303`
<!-- DOD:END -->
