---
id: TASK-472
title: Draft Persona Tool Administration PRD
status: Done
labels:
- persona
- tools
- mcp
- policy
- prd
- docs
priority: Medium
references:
- https://github.com/rmusser01/tldw_server/issues/1922
- https://github.com/rmusser01/tldw_server/issues/1902
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Draft a repo-grounded PRD for Persona Tool Administration covering tool install/config lifecycle, MCP Unified alignment, Persona scope/policy boundaries, admin lifecycle, audit/revocation, and non-overlap with current minimal Persona-local tool discovery.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 PRD is grounded in current Persona scope/policy, MCP Unified, and Persona API/runtime contracts.
- [x] #2 Scope, non-goals, permission model, admin/config lifecycle, audit/revocation, risks, staged implementation, and validation plan are documented.
- [x] #3 Issue #1922 and tracker #1902 are referenced.
- [x] #4 Docs-only verification is recorded.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
['Inspect current Persona policy/scopes docs and code, MCP Unified docs/code, and existing future PRD boundaries.', 'Draft the PRD with scope, non-goals, current evidence, permission layering, tool lifecycle, staged delivery, risks, and validation.', 'Run docs-only verification and update the task status.']
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Created `Docs/Product/Persona_Tool_Administration_PRD.md`. Grounded the PRD in the current Persona PRD boundaries, Persona scope/policy schemas and endpoints, `policy_evaluator.py`, Persona runtime policy enforcement, MCP Unified production contract, MCP Hub management docs, tool catalog docs, and MCP Hub permission/credential/approval/governance plans.

The PRD keeps MCP Hub as the canonical governance/control plane and defines Persona Tool Administration as a Persona-aware composition layer for effective access preview, reference-backed assignments, runtime enforcement, audit, revocation, and safe blocked-state explanations. Bandit is not applicable because no executable code changed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Drafted the Persona Tool Administration PRD for issue #1922 and tracker #1902. The PRD defines goals, non-goals, current evidence, authority layering, Persona/MCP Hub boundaries, assignment model, tool lifecycle, approvals, credentials, audit/revocation, API/data direction, staged delivery, validation plan, risks, acceptance criteria, and open questions.
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
