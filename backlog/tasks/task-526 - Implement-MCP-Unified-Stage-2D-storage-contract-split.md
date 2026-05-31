---
id: TASK-526
title: Implement MCP Unified Stage 2D storage contract split
status: Done
labels:
- mcp-unified
- standalone
- stage2
- storage
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add package-local MCP Unified storage contract primitives for profile assignments, approval policies, credential grants, external server registry entries, and audit events without SQLite persistence, runtime enforcement, FastAPI routes, external process lifecycle, or gateway entrypoints.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Stage 2D implementation plan is added and tied to the standalone MCP design spec.
- [x] #2 mcp_unified exposes typed storage models and protocols for profile assignments, approval policies, credential grants, external registry entries, and audit events while preserving existing ProfileStore behavior.
- [x] #3 Package-boundary tests prove storage models/protocols import without tldw_Server_API and preserve safe defaults/copy-friendly payloads.
- [x] #4 Focused MCP package tests plus Ruff, Mypy, Bandit, and git diff checks pass.
- [x] #5 No runtime execution, FastAPI route, SQLite persistence, external-server lifecycle, or gateway entrypoint behavior changes are introduced.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Created Stage 2D implementation plan at Docs/superpowers/plans/2026-05-28-mcp-unified-stage2d-storage-contract-split-implementation-plan.md. RED test run failed as expected with missing mcp_unified.storage and missing split store protocols. Added package-local storage models for ProfileAssignment, ApprovalPolicyDocument, CredentialGrant, ExternalServerDefinition, and AuditEvent with aware timestamps, safe list/dict defaults, caller-owned copied payloads, and credential grant metadata that excludes embedded secret fields. Expanded mcp_unified.interfaces.storage with ProfileAssignmentStore, ApprovalPolicyStore, CredentialGrantStore, typed ExternalRegistryStore, and typed AuditStore while preserving ProfileStore. Updated package and host compatibility interface exports. Verification: focused MCP package regression passed with 52 passed, 3 warnings; Ruff passed; Mypy passed with no issues in 14 source files; Bandit over mcp_unified reported 0 findings; git diff --check passed. Draft PR: https://github.com/rmusser01/tldw_server/pull/2085.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented MCP Unified Stage 2D storage contract split in the standalone mcp_unified package. This adds typed storage payload and protocol contracts for future assignment, approval, credential, external registry, and audit stores without changing runtime execution, routes, SQLite persistence, external process lifecycle, or gateway entrypoints.
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
