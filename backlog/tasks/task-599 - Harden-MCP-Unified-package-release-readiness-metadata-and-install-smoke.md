---
id: TASK-599
title: Harden MCP Unified package release readiness metadata and install smoke
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-03 01:15'
labels:
  - mcp-unified
  - packaging
  - standalone-gateway
dependencies: []
documentation:
  - >-
    Docs/superpowers/specs/2026-05-26-mcp-unified-standalone-library-gateway-design.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the next MCP Unified standalone-library release-readiness slice after the gateway admin/config surface: define explicit package metadata/extras and add minimal install/import smoke coverage so mcp_unified is not presented as third-party-ready without a tested dependency/license boundary.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 MCP Unified package/release metadata documents the current license decision and experimental/internal status without marketing unsupported standalone publishing.
- [x] #2 Dependency extras for MCP Unified core, fastapi, sqlite, federation, and gateway are declared or otherwise represented in a testable metadata surface separate from the full tldw-server dependency graph.
- [x] #3 A clean minimal-install/import smoke test proves mcp_unified core imports without importing tldw_Server_API or requiring media/RAG/STT/TTS dependencies.
- [x] #4 Focused tests cover metadata shape, extras membership, import-boundary behavior, and docs/CLI visibility for the release-readiness status.
- [x] #5 Bandit on touched Python source, focused tests, and git diff --check pass before PR.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-06-03-mcp-unified-package-release-readiness-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

Verification completed: focused red checks failed before implementation for missing package metadata, CLI package-info, and docs release gate; full focused pytest files passed 91 tests; Bandit on touched Python reported zero findings; git diff --check passed. No skips or blockers.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added a pure-stdlib mcp_unified.package_metadata release-readiness contract declaring the current internal-experimental/not-published status, GPL-3.0-only license expression, and dependency-extra groups for core, fastapi, sqlite, federation, gateway, and dev. Exposed the contract through mcp-unified-gateway package-info and documented that mcp_unified is an in-repo/internal experimental package boundary, not a separately published standalone package. Added regression coverage for metadata shape, heavy-stack-free extras, minimal import smoke, CLI visibility, and docs visibility. Verification: targeted red checks failed before implementation for the expected missing surfaces; full focused pytest files passed 91 tests; Bandit reported zero findings; git diff --check passed.
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
