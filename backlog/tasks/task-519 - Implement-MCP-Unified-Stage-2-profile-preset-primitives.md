---
id: TASK-519
title: Implement MCP Unified Stage 2 profile preset primitives
status: Done
assignee: []
created_date: ''
updated_date: '2026-05-27 06:32'
labels:
  - mcp
  - mcp-unified
  - standalone
  - stage2
  - profiles
dependencies: []
documentation:
  - >-
    Docs/superpowers/specs/2026-05-26-mcp-unified-standalone-library-gateway-design.md
  - >-
    Docs/superpowers/plans/2026-05-27-mcp-unified-profile-presets-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Package-local profile preset primitives are available without importing tldw_Server_API.
- [x] #2 Bundled role/mode presets cover the spec's initial mode list with stable ids and preset versions.
- [x] #3 Preset policies satisfy the safety baseline: no broad process execution, destructive filesystem action, credentials, or external network by default unless explicitly granted with provenance.
- [x] #4 Presets can be duplicated into editable MCPProfile instances with preset id/version provenance preserved.
- [x] #5 Tests cover package import isolation, preset lookup, duplication, and safety validation.
- [x] #6 Focused tests, Ruff/Mypy for mcp_unified, Bandit touched scope, and diff whitespace checks pass.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implementation plan: Docs/superpowers/plans/2026-05-27-mcp-unified-profile-presets-implementation-plan.md

Implemented package-local MCP profile preset primitives:
- Added mcp_unified.profiles.presets with immutable ProfilePreset templates, preset lookup/listing, duplication into editable MCPProfile objects, and safety validation helpers.
- Added all initial spec role/mode ids: orchestrator, product-owner, architect, merge-conflict-resolver, documentation-writer, project-researcher, deep-researcher, code-reviewer, devops-engineer, backend-engineer, frontend-engineer, qa-engineer, sdet, memory-keeper.
- Presets use conservative capabilities, no default credential grants, no broad process execution, and explicit external network provenance for deep-researcher.
- Exported preset primitives from mcp_unified.profiles.
- Added package-boundary tests for no tldw_Server_API imports, preset coverage, lookup, duplication/provenance, and safety rejection.

Verification:
- python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_profile_presets.py tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py -v -> 9 passed, 3 warnings
- python -m ruff check mcp_unified tldw_Server_API/app/core/MCP_unified/tests/test_profile_presets.py -> passed
- python -m mypy mcp_unified --config-file pyproject.toml -> passed
- python -m bandit -r mcp_unified -f json -o bandit_mcp_unified_profile_presets.json -> 0 findings
- git diff --check -> clean

Known skips/blockers: none for this slice.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added package-local built-in MCP profile preset primitives for the initial front-end role/mode set. Presets are conservative templates with stable ids/versioning, safety-baseline validation, and duplication into editable MCPProfile objects with preset provenance preserved. No tldw_server host MCP route, policy, approval, credential, or execution behavior changed.
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
