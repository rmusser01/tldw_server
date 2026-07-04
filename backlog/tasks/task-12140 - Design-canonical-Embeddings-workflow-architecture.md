---
id: TASK-12140
title: Design canonical Embeddings workflow architecture
status: Done
assignee: []
created_date: '2026-07-04 01:08'
updated_date: '2026-07-04 01:09'
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
Spec written at Docs/superpowers/specs/2026-07-03-embeddings-workflow-architecture-design.md. Self-review completed: no TODO/TBD/FIXME placeholders found; scope limited to architecture/spec work; first implementation slice remains API inline workflow facade only; no code, schema, metrics, logs, headers, or endpoint behavior changes included. Verification: git diff --check passed. Bandit not run because this task only adds documentation and Backlog task metadata.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Designed the canonical Embeddings workflow architecture: one shared workflow definition with dual-mode inline/durable runners, request-level workflow state with item sub-states, optional redacted traces, Jobs-rooted future durable ownership, and a staged strangler roadmap starting with the feature-flagged /api/v1/embeddings inline runner facade.
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
