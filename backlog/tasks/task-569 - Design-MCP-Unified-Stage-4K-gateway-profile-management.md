---
id: TASK-569
title: Design MCP Unified Stage 4K gateway profile management
status: Done
labels:
- mcp-unified
- stage-4
- design
- gateway
- profiles
documentation:
- Docs/superpowers/specs/2026-05-30-mcp-unified-stage4k-gateway-profile-management-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create the Stage 4K design spec for standalone gateway profile management over both CLI and FastAPI. The design should cover stored profile listing/inspection, duplicating built-in presets into persisted profiles, default profile get/set, runtime default resolution, shared manager behavior, error contracts, boundaries, and verification strategy.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Spec documents CLI and FastAPI profile-management scope for Stage 4K.
- [x] #2 Spec explains manager/service architecture and reuse of existing ProfileStore/ProfileAssignmentStore primitives.
- [x] #3 Spec defines success/error contracts, reason codes, and HTTP/CLI status mapping.
- [x] #4 Spec records out-of-scope items and focused test strategy.
- [x] #5 Design is reviewed and committed before implementation planning.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Design draft created at Docs/superpowers/specs/2026-05-30-mcp-unified-stage4k-gateway-profile-management-design.md. Scope covers CLI plus FastAPI profile management, default profile behavior, shared manager architecture, error contracts, boundaries, and focused tests.
Spec review iteration 1 found a blocking HTTP mapping gap for profile_disabled and default_profile_not_configured. Patched the FastAPI contract with deterministic reason-code-to-status mapping.
Spec review iteration 2 found a blocking CLI store-selection gap. Patched the CLI contract to require --config or MCP_UNIFIED_GATEWAY_CONFIG for store-backed commands, clarify offline store-management semantics, reject nonpersistent memory-store mutations by default, and define the multiple-default assignment tie-breaker.
Spec review iteration 3 approved the revised Stage 4K design. Advisory items for implementation planning: mark profile_id and name optional for POST /profiles/from-preset, and pin the exact memory-store nonpersistent metadata shape in CLI tests.

Verification:
- git diff --check (passed)
- Marker scan for TODO/TBD/FIXME/placeholder/question-mark placeholders in Docs/superpowers/specs/2026-05-30-mcp-unified-stage4k-gateway-profile-management-design.md (no matches)
- Bandit skipped: docs/task metadata only.
Final design worktree rebased onto origin/dev at b5bb29f2448a3d8b863428e1a38223372b84ca2d before commit.
Post-approval design hardening requested by user. Updated the spec to: require one ProfileAssignmentStore-compatible default path including memory-backed tests; mount management endpoints only when explicitly enabled/provided; add optional AuditStore lifecycle events; require FastAPI default changes to affect subsequent no-profile JSON-RPC reads without restart; mark POST /profiles/from-preset profile_id/name optional; and pin CLI memory-store metadata/mutation behavior.
Post-hardening spec review approved. Advisory items for implementation planning: make the app-factory route gate concrete, and pin exact FastAPI success response envelopes in tests if frontend work follows immediately.

Post-hardening verification:
- git diff --check (passed)
- Marker scan for TODO/TBD/FIXME/placeholder/question-mark placeholders in the Stage 4K spec (no matches)
- Bandit skipped: docs/task metadata only.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Created and reviewed the Stage 4K gateway profile management design spec. The spec defines a shared GatewayProfileManager architecture for CLI and FastAPI, store-backed profile list/show/preset duplication/default profile workflows, deterministic success/error contracts, CLI config-store selection, runtime default resolution precedence, boundaries, and focused verification strategy. Review passed on iteration 3 after adding deterministic HTTP reason-code mappings and explicit CLI store-selection semantics.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
- [x] #7 Spec path recorded in task notes.
- [x] #8 Review outcome recorded.
- [x] #9 Verification commands recorded.
<!-- DOD:END -->
