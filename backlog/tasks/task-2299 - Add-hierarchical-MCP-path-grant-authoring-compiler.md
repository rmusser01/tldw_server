---
id: TASK-2299
title: Add hierarchical MCP path grant authoring compiler
status: Done
labels:
- mcp
- policy
- filesystem
- admin
- followup
references:
- Docs/superpowers/specs/2026-06-07-mcp-fs-patch-write-safe-edit-tools-design.md
- Docs/superpowers/plans/2026-06-07-mcp-hierarchical-path-grants-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add an optional authored-policy layer for org/workspace/folder/file inheritance with explicit deny overrides. The executable runtime contract should remain flat `policy_document.path_grants`; this task compiles hierarchical authoring data into normalized effective grants and exposes validation/preview tooling.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] Standalone `mcp_unified` helper compiles authored org/workspace/folder/file path rules into normalized flat `policy_document.path_grants`.
- [x] Compiler returns validation diagnostics and preview metadata without absolute paths or file content.
- [x] Runtime path enforcement consumes compiled authored grants only when explicit flat `path_grants` are absent.
- [x] Invalid authored grants fail closed and do not fall back to legacy `path_allowlist_prefixes`.
- [x] User guide documents the authored hierarchy shape and flat runtime contract.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-06-07-mcp-hierarchical-path-grants-implementation-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Added `mcp_unified.profiles.path_grants` as a standalone compiler for flat and authored path-grant policy. It supports `path_grant_authoring` levels `org`, `workspace`, `folders`, and `files`, normalizes workspace-relative prefixes, merges duplicate prefix/effect rules, and emits validation diagnostics plus preview metadata. The tldw path enforcer now uses compiled authored grants only when explicit flat `path_grants` are absent, preserving the flat runtime decision contract.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented hierarchical MCP path-grant authoring as an optional compiler layer over the existing flat `path_grants` runtime contract. Authored rules compile to normalized flat grants, explicit flat grants remain authoritative, invalid authored rules fail closed, and the user guide documents the authoring shape.

Verification:
- `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_path_grant_authoring.py tldw_Server_API/tests/MCP_unified/test_mcp_hub_path_enforcement_service.py tldw_Server_API/app/core/MCP_unified/tests/test_profile_presets.py -q` -> 45 passed, 6 warnings.
- `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m ruff check <touched python files>` -> all checks passed.
- `git diff --check` -> clean.
- `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m bandit -r mcp_unified/profiles/path_grants.py mcp_unified/profiles/__init__.py tldw_Server_API/app/services/mcp_hub_path_enforcement_service.py -f json -o /tmp/bandit_mcp_hierarchical_path_grants.json` -> 0 findings, 0 errors.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented: none
<!-- DOD:END -->
