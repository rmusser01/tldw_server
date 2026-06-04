---
id: TASK-2253
title: Design canonical Workspaces manager and Project Workspace creation roadmap
status: Done
assignee: []
created_date: ''
updated_date: 2026-06-04 03:34
labels:
- workspaces
- design
- research-workspace
- project-workspace
dependencies: []
references:
- 'Spec review pass 1: Approved with advisory recommendations; pass 2: Approved with
  no serious planning blockers.'
documentation:
- Docs/superpowers/specs/2026-06-04-canonical-workspaces-manager-project-creation-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create an implementation-ready design spec for the canonical `/workspaces` management surface and sequential task roadmap. Scope includes server-backed Workspace creation/management/editing, Project Workspace upgrade, durable sandbox workspace-volume contract, Workspace-owned sandbox root provision-and-attach command, host-local root attach, local Research Workspace reconciliation, cross-surface links to Research Workspace/MCP/ACP/Sandbox, validation strategy, and task parallelization boundaries. Do not implement runtime code in this task.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Spec defines `/workspaces` as the canonical product-level Workspace manager while preserving Research Workspace, MCP Hub, ACP, and Sandbox ownership boundaries.
- [x] #2 Spec decomposes the roadmap into sequential reviewable tasks with explicit dependencies and parallelizable follow-up slices.
- [x] #3 Spec explicitly separates durable Sandbox workspace-volume mechanics from the Workspace-owned sandbox root provision-and-attach product command.
- [x] #4 Spec covers project creation partial-failure recovery, archive/delete safety, file inventory capability gating, and metadata-first local Research Workspace reconciliation.
- [x] #5 Spec identifies API/client gaps, UX acceptance criteria, error states, validation strategy, and live backend/WebUI/CDP verification expectations.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Design/spec task only. Write the approved Workspaces manager and Project Workspace creation roadmap as a design spec, review it, update the task record with verification, and stop before implementation planning until user review approves the written spec.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Created the canonical Workspaces manager and Project Workspace creation roadmap spec at Docs/superpowers/specs/2026-06-04-canonical-workspaces-manager-project-creation-design.md. The spec defines /workspaces as the canonical product manager, separates durable Sandbox workspace-volume mechanics from the Workspace-owned sandbox root provision-and-attach command, decomposes the roadmap into seven reviewable tasks, adds partial-failure recovery states, archive/delete safety, file inventory gating, metadata-first local Research Workspace reconciliation, cross-surface links, validation strategy, and parallelization boundaries.

Spec reviewer pass 1 approved with advisory recommendations. Followed up by adding a shared manager attention-state mapping, recommended Project Workspace hard-delete disablement until a cleanup contract exists, and a minimal local reconciliation marker requirement.

Spec reviewer pass 2 approved with no serious planning blockers. Remaining advisory planning constraints: resolve Sandbox volume storage/runtime support/cleanup retention/ACP diagnostics link target/reconciliation marker shape during task planning; map Sandbox volume states into Workspace root/mount/capability/attention projections; define Task 3 idempotency mechanics and whether provision-and-attach returns at queued/provisioning or ready state before coding.

User-requested spec review amendment completed before implementation planning. Added explicit Task 3 response semantics (`202` for queued/active provisioning, `200` for already-attached or synchronously complete roots), Workspace-owned idempotency ownership and `409` behavior, deterministic Sandbox-to-Workspace projection mapping, definitive V1 Project Workspace hard-delete disablement, Task 4/Task 5 Project creation boundary, and required planning-time decisions for reconciliation marker, idempotency storage, projections, response behavior, and hard-delete policy.

Verification: keyword scan for unresolved planning markers returned no matches; wording ambiguity scan returned no matches; git diff --check on the spec/task paths passed. Bandit skipped because this is documentation/backlog-only work with no Python code changes.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Created, reviewed, and amended the canonical Workspaces manager and Project Workspace creation roadmap spec. The latest amendment adds implementation guardrails for canonical Workspace versus ACP/MCP/prototype naming, Task 1 client normalization fields, V1 archive/unarchive-only manager behavior, deferred deleted-row restore and delete cleanup contracts, Workspace-owned sandbox provision operation polling, context recovery after refresh, and bounded idempotency records with 7-day default retention. This task is documentation-only; implementation planning should begin after user review approval.
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
