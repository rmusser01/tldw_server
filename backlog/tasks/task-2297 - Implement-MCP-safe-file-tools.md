---
id: TASK-2297
title: Implement MCP safe file tools
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-07 15:39'
labels:
  - mcp
  - filesystem
  - security
  - implementation
dependencies: []
references:
  - >-
    Docs/superpowers/specs/2026-06-07-mcp-fs-patch-write-safe-edit-tools-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the approved MCP safe file tools design: action-aware path grants, module-derived path candidates, fs.read with hashes/read receipts, fs.patch unified-diff editing, fs.write guarded create/replace behavior, legacy tool compatibility metadata, profile guidance updates, observability redaction, tests, and docs.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-06-07-mcp-safe-file-tools-implementation-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented MCP safe file tools with action-aware path grants, module-derived path candidates, bounded fs.read with hashes/read receipts, unified-diff fs.patch, guarded fs.write create/replace behavior, legacy tool metadata, profile preset wiring, metadata-only reporting, package user-guide documentation, and focused regression coverage.

Validation recorded on 2026-06-07:
- `source ../../.venv/bin/activate && python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_module.py tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_patch_parser.py tldw_Server_API/app/core/MCP_unified/tests/test_protocol_path_scope_candidates.py tldw_Server_API/tests/MCP_unified/test_mcp_hub_path_enforcement_service.py tldw_Server_API/app/core/MCP_unified/tests/test_tool_use_reporting_protocol.py -q` -> 86 passed, 6 warnings.
- `source ../../.venv/bin/activate && python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_protocol_preexec_validation.py tldw_Server_API/app/core/MCP_unified/tests/test_write_tools_validators.py tldw_Server_API/app/core/MCP_unified/tests/test_tool_observability.py -q` -> 26 passed, 4 warnings.
- `source ../../.venv/bin/activate && python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_profile_presets.py tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_module.py tldw_Server_API/app/core/MCP_unified/tests/test_tool_use_reporting_protocol.py -q` -> 90 passed, 4 warnings.
- `source ../../.venv/bin/activate && python -m bandit -r <touched code paths> -f json -o /tmp/bandit_mcp_safe_file_tools.json` -> exit 0; JSON `results` array empty.

Known skips/blockers: none.
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
