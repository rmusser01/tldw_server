---
id: TASK-601
title: Add MCP Unified standalone package install smoke
status: Done
labels:
- mcp-unified
- packaging
priority: medium
references:
- Docs/superpowers/specs/2026-05-26-mcp-unified-standalone-library-gateway-design.md
modified_files:
- Docs/MCP_UNIFIED_STANDALONE_GATEWAY_ADMIN.md
- Docs/superpowers/plans/2026-06-03-mcp-unified-standalone-package-install-smoke-plan.md
- mcp_unified/federation/manager.py
- mcp_unified/pyproject.toml
- mcp_unified/storage/__init__.py
- tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add a dedicated standalone MCP Unified package descriptor plus an offline clean-environment install smoke test so the package release gate proves mcp_unified can be installed independently from the root tldw-server dependency surface.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-06-03-mcp-unified-standalone-package-install-smoke-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
PR #2232 rebased cleanly onto origin/dev and review remediation completed. Verified Qodo's federation/SQLAlchemy finding with red import-leak tests, then made SQLiteMCPStore a lazy storage export and changed federation manager imports to storage.models. Verified Qodo's offline build concern by adding descriptor-driven build-system preflight, explicit wheel_dir creation, and lowering the standalone build-system floor to setuptools>=61.0 to match the root project and avoid an unnecessary wheel runtime requirement. Addressed Gemini's subprocess diagnostics by replacing check=True in the smoke path with explicit return-code assertions that include captured stdout/stderr. Verification after fixes: package-boundary file passed with 20 passed; focused package suite passed with 96 passed and 5 warnings; git diff --check passed; no generated mcp_unified build/egg-info artifacts remain; Bandit runtime scan found 0 issues; Bandit test scan found 0 issues with expected test-only B101/B603/B404/B105 excluded.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added standalone package install-smoke coverage and addressed PR #2232 review feedback. The final package boundary keeps storage/federation imports free of eager SQLAlchemy loading, preflights offline build-system requirements from the package descriptor, and reports captured subprocess output directly when the smoke build/install/import steps fail.
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
