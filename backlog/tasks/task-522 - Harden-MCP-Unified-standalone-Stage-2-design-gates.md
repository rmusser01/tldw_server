---
id: TASK-522
title: Harden MCP Unified standalone Stage 2 design gates
status: Done
assignee: []
created_date: ''
updated_date: '2026-05-27 15:48'
labels:
  - mcp
  - mcp-unified
  - standalone
  - stage2
  - design
dependencies: []
documentation:
  - >-
    Docs/superpowers/specs/2026-05-26-mcp-unified-standalone-library-gateway-design.md
  - >-
    Docs/superpowers/plans/2026-05-27-mcp-unified-profile-registry-resolver-implementation-plan.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address the post-review design issues before continuing MCP Unified standalone extraction: split Stage 2 into explicit gates, add structured profile/effective-policy result semantics, require workspace binding for write-capable presets, clarify storage-contract split, add packaging/license release gate, and gate external stdio process work behind adapter/audit policy readiness.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Stage 2 is split into explicit sub-stages that distinguish profile primitives from enforcement/runtime/gateway work.
- [x] #2 Spec defines structured profile/effective-policy resolution outcomes with machine-readable reason codes and provenance expectations before execution wiring.
- [x] #3 Write-capable presets require workspace/path-scope binding before executable use.
- [x] #4 Storage contract responsibilities are clarified before SQLite persistence work.
- [x] #5 Packaging/license/minimal-install gate is explicit before standalone publication or third-party embedding claims.
- [x] #6 External MCP stdio process lifecycle work is gated behind registry, credential, audit, path, and process-policy adapters.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the post-review design hardening before further MCP Unified standalone extraction. Updated the standalone design spec with explicit Stage 2 sub-stages, structured profile/effective-policy result contracts, workspace-binding requirements for write-capable profiles, split storage responsibilities, packaging/license/minimal-install release gate, and a non-spawning external federation gate before upstream stdio process lifecycle work. Updated the completed Stage 2B profile registry plan with continuation gates and corrected the amended commit hash. Added a Stage 2C structured resolution implementation plan as the next executable slice.

Verification: git diff --check passed. Design self-review confirmed no runtime/code behavior changes, no FastAPI route changes, no MCPProtocol/MCPServer wiring, no SQLite persistence, no external stdio process work, and no gateway entrypoints. Bandit skipped because this task touched Markdown/Backlog design artifacts only.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Hardened the MCP Unified standalone design and planning artifacts after review. The spec now prevents treating profile primitives as execution enforcement, requires structured reason/provenance results before runtime wiring, requires workspace/path binding for write-capable profiles, splits storage responsibilities beyond a generic ProfileStore, blocks standalone publication claims until packaging/license/minimal-install gates are proven, and defers real upstream stdio process spawning until policy, audit, credential, path, and process adapters are ready. Added the concrete Stage 2C structured resolution plan as the next code slice.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Design/plan changes reviewed for scope creep
- [x] #3 Verification recorded
- [x] #4 Final summary added
<!-- DOD:END -->
