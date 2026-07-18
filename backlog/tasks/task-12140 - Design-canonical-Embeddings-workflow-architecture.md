---
id: TASK-12140
title: Design canonical Embeddings workflow architecture
status: Done
assignee: []
created_date: '2026-07-04 01:08'
updated_date: '2026-07-04 01:15'
labels:
  - embeddings
  - design
  - workflow
dependencies: []
references:
  - Docs/superpowers/specs/2026-07-03-embeddings-workflow-architecture-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Write the system design for a canonical dual-mode Embeddings workflow engine and the first API-inline workflow facade slice. Scope includes workflow state model, runner boundaries, trace safety, staged strangler roadmap, and explicit non-goals. No implementation code in this task.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Spec captures one canonical workflow engine reused across API, media jobs, vector-store batches, re-embed flows, and workers.
- [x] #2 Spec defines dual-mode execution with inline API runner first and future Jobs-backed durable runner.
- [x] #3 Spec records state model, component boundaries, data flow, error handling, trace safety, tests, rollout, non-goals, and staged roadmap.
- [x] #4 Spec self-review checks placeholders, contradictions, scope creep, and ambiguous requirements before completion.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Spec written at Docs/superpowers/specs/2026-07-03-embeddings-workflow-architecture-design.md. Self-review completed: no placeholder markers found; scope limited to architecture/spec work; first implementation slice remains API inline workflow facade only; no code, schema, metrics, logs, headers, or endpoint behavior changes included. Verification: git diff --check passed. Bandit not run because this task only adds documentation and Backlog task metadata.

Post-approval review reopened the design task to tighten trace collector bounds/overflow behavior, endpoint test seam wording, workflow id/item identity, and durable RG/billing boundary language before implementation planning.

Post-approval spec refinements completed: added safe workflow id/item identity language, clarified that slice one emits only truthfully derivable phases, required bounded fail-closed in-memory traces, allowed endpoint tests to monkeypatch runner construction without trace exposure, and clarified durable RG/billing boundary ownership. Verification: placeholder scan passed and git diff --check passed. Bandit remains not applicable because only documentation/task metadata changed.

Planning review found that the inline runner needs a boundary hook between prepare and execute so the endpoint can preserve ResourceGovernor reservation after token counting and before provider/cache execution. Reopening design task to document that hook before finalizing the implementation plan.

Planning review update completed: the spec now requires an optional async pre-execute hook so endpoint-owned ResourceGovernor reservation can remain after prepare/token counting and before cache/provider execution. Verification: placeholder scan passed and git diff --check passed. Bandit remains not applicable for documentation-only changes.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Designed the canonical Embeddings workflow architecture: one shared workflow definition with dual-mode inline/durable runners, request-level workflow state with item sub-states, optional redacted traces, Jobs-rooted future durable ownership, and a staged strangler roadmap starting with the feature-flagged /api/v1/embeddings inline runner facade.

Post-approval review refinements tightened trace bounds, test seam wording, workflow identity, and durable RG/billing boundaries before implementation planning.

Added pre-execute hook guidance to preserve current ResourceGovernor reservation ordering in the inline workflow runner slice.
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
