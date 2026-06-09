---
id: TASK-2305
title: Expand MCP file policy action taxonomy and operation tools
status: Done
labels:
- mcp
- filesystem
- policy
- tools
- followup
references:
- Docs/superpowers/specs/2026-06-07-mcp-fs-patch-write-safe-edit-tools-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the reserved file-policy actions beyond the first `read`/`edit`/`write` slice. Define separate policy semantics and tools for delete, rename, move, share/export, chmod/admin, and lock so exfiltration and destructive operations are not bundled under generic write authority.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] Package-level file-policy action constants and metadata cover `read`, `edit`, `write`, `delete`, `rename`, `move`, `share`, `export`, `chmod`, `admin`, and `lock`.
- [x] Path-grant validation, path-scope candidates, and effective-permission previews accept the expanded action vocabulary while still rejecting malformed actions fail-closed.
- [x] Existing `fs.read`, `fs.patch`, `fs.write`, `fs.read_text`, and `fs.write_text` behavior remains unchanged.
- [x] Existing filesystem tool descriptors expose action-family metadata so clients can distinguish read, bounded edit, and whole-write authority.
- [x] Package/user documentation explains the expanded action taxonomy and makes clear that destructive, exfiltration, admin, and lock actions are reserved until dedicated tools land.
- [x] Focused regression tests cover action metadata, compiler validation, preview/explanation behavior, filesystem descriptor metadata, and backward compatibility.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Plan: `Docs/superpowers/plans/2026-06-08-mcp-file-policy-action-taxonomy-implementation-plan.md`

Touched files:
- `mcp_unified/interfaces/file_policy_actions.py`
- `mcp_unified/interfaces/__init__.py`
- `mcp_unified/interfaces/path_scope.py`
- `mcp_unified/profiles/path_grants.py`
- `mcp_unified/gateway/profiles.py`
- `mcp_unified/USER_GUIDE.md`
- `tldw_Server_API/app/services/mcp_hub_path_enforcement_service.py`
- `tldw_Server_API/app/core/MCP_unified/modules/implementations/filesystem_module.py`
- `tldw_Server_API/app/core/MCP_unified/tests/test_file_policy_actions.py`
- `tldw_Server_API/app/core/MCP_unified/tests/test_path_grant_authoring.py`
- `tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_module.py`
- `tldw_Server_API/app/core/MCP_unified/tests/test_gateway_profile_management.py`
- `tldw_Server_API/tests/MCP_unified/test_mcp_hub_path_enforcement_service.py`

Verification:
- `python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_file_policy_actions.py tldw_Server_API/app/core/MCP_unified/tests/test_path_grant_authoring.py tldw_Server_API/tests/MCP_unified/test_mcp_hub_path_enforcement_service.py tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_module.py::test_filesystem_tools_include_path_scope_metadata tldw_Server_API/app/core/MCP_unified/tests/test_gateway_profile_management.py::test_create_profile_permission_governor_flags_reserved_path_grant_risks tldw_Server_API/app/core/MCP_unified/tests/test_gateway_profile_management.py::test_create_profile_calls_permission_governor_with_redacted_summary -q` -> 31 passed.
- `python -m ruff check <touched Python files>` -> all checks passed.
- `python -m py_compile <touched Python files>` -> passed.
- `python -m bandit -r <touched production files> -f json -o /tmp/bandit_mcp_file_policy_action_taxonomy.json` -> 0 findings, 0 errors.
- `git diff --check` -> clean.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Expanded the MCP file-policy action taxonomy with package-level metadata for `read`, `edit`, `write`, `delete`, `rename`, `move`, `share`, `export`, `chmod`, `admin`, and `lock`. Path grants, path-scope candidates, and effective-permission previews now use the shared vocabulary, while existing filesystem tool behavior remains limited to the current safe read/edit/write tools. Filesystem descriptors include action-family metadata, governance risk flags distinguish destructive/exfiltration/admin/lock grants, and the package user guide documents reserved-action semantics.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented: reserved actions are policy/preview vocabulary only until dedicated safe operation tools land.
<!-- DOD:END -->
