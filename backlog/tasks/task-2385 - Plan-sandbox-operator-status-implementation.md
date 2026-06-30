---
id: TASK-2385
title: Plan sandbox operator status implementation
status: Done
labels:
- sandbox
- operator-ux
- planning
documentation:
- Docs/superpowers/specs/2026-06-18-sandbox-operator-status-consolidation-design.md
- Docs/superpowers/plans/2026-06-18-sandbox-operator-status-implementation-plan.md
modified_files:
- Docs/superpowers/plans/2026-06-18-sandbox-operator-status-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create the implementation plan for Slice 1 of sandbox operator/admin status consolidation: a portable read-only service projection, admin endpoint, schema, tests, and docs, without evidence-file ingestion or mutation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Plan decomposes Slice 1 into TDD implementation tasks with exact files and commands.
- [ ] #2 Plan preserves read-only/no-mutation boundaries from the design spec.
- [ ] #3 Plan includes verification, Bandit scope, and documentation steps.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Planning-only slice for Slice 1 implementation. The plan decomposes operator status into a pure projection module, SandboxService wrapper, Pydantic schema, admin endpoint, RBAC coverage, docs, focused pytest verification, Bandit, and final commit steps. Inline review tightened the plan to avoid broad exception swallowing, require schema extra handling for section-specific fields, add safe coercion helpers, avoid generated_at, and keep evidence ingestion out of Slice 1.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Created the Slice 1 implementation plan for sandbox operator/admin status. The plan uses TDD, creates a focused operator_status.py projection module, wires SandboxService.operator_status, adds Pydantic response schemas, exposes the admin-only GET /api/v1/sandbox/admin/operator-status endpoint, adds RBAC and endpoint tests, updates operator docs, and defines focused pytest plus Bandit verification. Inline review tightened the plan to avoid broad exception swallowing, require schema extra handling for section-specific fields, add safe coercion helpers, avoid generated_at, and keep evidence ingestion out of Slice 1. Verification: python -m pytest tldw_Server_API/tests/sandbox/test_runtime_capability_gate.py -q passed with 8 tests. No production code was changed in this planning slice.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Plan file is committed on the operator status branch.
- [ ] #2 Backlog task links to the design spec and implementation plan.
- [ ] #3 No production behavior changes are introduced by the planning slice.
<!-- DOD:END -->
