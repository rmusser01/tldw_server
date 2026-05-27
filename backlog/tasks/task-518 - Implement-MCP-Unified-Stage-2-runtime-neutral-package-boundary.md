---
id: TASK-518
title: Implement MCP Unified Stage 2 runtime-neutral package boundary
status: Done
assignee: []
created_date: ''
updated_date: '2026-05-27 05:21'
labels:
  - mcp
  - mcp-unified
  - standalone
  - stage2
dependencies: []
documentation:
  - >-
    Docs/superpowers/specs/2026-05-26-mcp-unified-standalone-library-gateway-design.md
  - >-
    Docs/superpowers/plans/2026-05-27-mcp-unified-stage2-package-boundary-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Top-level mcp_unified package is included by package discovery.
- [x] #2 Runtime-neutral interface contracts are available under mcp_unified.interfaces without tldw_Server_API imports.
- [x] #3 Existing tldw_Server_API.app.core.MCP_unified.interfaces imports remain compatibility shims to the same package contracts.
- [x] #4 Profile schema and resolver primitives are available under mcp_unified.profiles with safe defaults.
- [x] #5 Focused MCP boundary, extraction, and basic functionality tests pass.
- [x] #6 Bandit reports no findings for touched package/interface code.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-05-27-mcp-unified-stage2-package-boundary-implementation-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the Stage 2 package-boundary slice from Docs/superpowers/plans/2026-05-27-mcp-unified-stage2-package-boundary-implementation-plan.md.

Verification:
- python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py tldw_Server_API/app/core/MCP_unified/tests/test_extraction_contracts.py tldw_Server_API/app/core/MCP_unified/tests/test_basic_functionality.py -v -> 30 passed, 5 warnings
- python -m bandit -r mcp_unified tldw_Server_API/app/core/MCP_unified/interfaces -f json -o bandit_mcp_unified_stage2_package_boundary.json -> 0 findings
- git diff --check -> clean

Known skips/blockers: none for this slice.

Review-fix pass after PR #2077 feedback:
- Rebased on latest origin/dev; branch was already current.
- Made package-boundary test resolve mcp_unified from the imported package path instead of cwd.
- Preserved profile and policy extension fields with Pydantic extra=allow.
- Coerced explicit null list/dict profile fields to safe empty defaults.
- Enforced aware profile timestamps and added naive timestamp rejection coverage.
- Included MCP_unified tests in pytest discovery and mcp_unified in Ruff/Mypy target config.
- Kept RbacPolicy sync-or-async because current protocol consumers support both and legacy in-memory RBAC is synchronous while AuthNZ RBAC is asynchronous.

Review-fix verification:
- python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py tldw_Server_API/app/core/MCP_unified/tests/test_extraction_contracts.py tldw_Server_API/app/core/MCP_unified/tests/test_basic_functionality.py -v -> 31 passed, 5 warnings
- python -m ruff check mcp_unified tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py -> passed
- python -m mypy mcp_unified --config-file pyproject.toml -> passed
- python -m bandit -r mcp_unified tldw_Server_API/app/core/MCP_unified/interfaces -f json -o bandit_mcp_unified_stage2_package_boundary.json -> 0 findings
- git diff --check -> clean
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Created the first runtime-neutral mcp_unified package boundary with host-neutral interface contracts and profile primitives. Converted the in-repo MCP Unified interface modules into compatibility re-export shims so existing imports keep working while the new package contracts become the canonical definitions. Updated package discovery and added focused boundary coverage for package isolation, shim identity, and safe profile defaults.

Review-fix follow-up addressed PR feedback for cwd-independent boundary tests, extension-preserving profile models, null-safe profile defaults, aware timestamp validation, pytest discovery, and mcp_unified Ruff/Mypy coverage. The RBAC protocol remains sync-or-async by design because existing host implementations use both shapes and protocol consumers already normalize them.
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
