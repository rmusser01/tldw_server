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
- mcp_unified/pyproject.toml
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
Implemented in isolated worktree codex/mcp-package-build-smoke. Added package-local mcp_unified/pyproject.toml, red/green package descriptor tests, and an offline wheel install smoke that builds from a temporary source copy, installs with --no-deps/--no-index into an isolated target directory, and imports via python -S outside the repository checkout. Nested venv interpreters abort under this local UV-managed Python, so the final smoke avoids that environment-specific path while preserving the clean import/install invariant. Verification: baseline 91 passed before edits; red run failed on missing mcp_unified/pyproject.toml; focused final tests passed with 93 passed and 5 warnings; git diff --check passed; Bandit on mcp_unified wrote /tmp/bandit_mcp_package_build_smoke.json with 0 findings.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added a standalone MCP Unified package descriptor and release-gate smoke coverage. The descriptor carries core dependency floors, expected extras, package-dir mappings, GPL-3.0-only license metadata, and the mcp-unified-gateway entry point. The package-boundary suite now validates descriptor alignment with mcp_unified.package_metadata and proves the built wheel can be installed/imported without root tldw-server dependencies.
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
