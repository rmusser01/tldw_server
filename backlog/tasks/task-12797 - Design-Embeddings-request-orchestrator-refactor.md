---
id: TASK-12797
title: Design Embeddings request orchestrator refactor
status: Done
created_date: 2026-06-24 04:51
labels:
- embeddings
- design
- refactor
priority: Medium
updated_date: 2026-06-24 17:53
modified_files:
- Docs/superpowers/specs/2026-06-24-embeddings-request-orchestrator-refactor-design.md
- backlog/tasks/task-12014 - Design-Embeddings-request-orchestrator-refactor.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Write and review a design spec for refactoring the Embeddings create request path around an internal request orchestrator boundary. Scope covers API-to-provider orchestration, cache/batching/fallback/policy ownership, error mapping, observability, rollout controls, and migration tests. No implementation code changes in this task.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Design spec is written under Docs/superpowers/specs/ with the approved scope B request-orchestrator design.
- [x] #2 Spec covers architecture, request flow, component responsibilities, migration/testing strategy, error handling, observability, rollout controls, invariants, and non-goals.
- [x] #3 Spec self-review is completed for placeholders, contradictions, ambiguity, and scope drift.
- [x] #4 Backlog task records touched files, verification, and final summary.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Starting approved design spec for scope B Embeddings request-orchestrator refactor in isolated worktree `/Users/appledev/Documents/GitHub/tldw_server/.worktrees/embeddings-orchestrator-design`.
Wrote design spec and completed self-review. Verification: placeholder scan (`rg TBD|TODO|FIXME|??|placeholder|CHANGE ME`) returned no matches; `git diff --check` returned clean. Bandit is not applicable because this task touches documentation/task metadata only and no executable code.
Follow-up spec review found several design clarifications to patch before implementation planning: explicit shim owners, primitive request context, cache value semantics, RG/billing characterization, response formatting parity tests, and path casing in task metadata.
Patched follow-up design review findings: request context now excludes framework/dependency handles, compatibility shim owners are explicit, cache value semantics are defined, RG/billing behavior must be characterized before changes, response formatting parity tests were added, and task metadata now matches the committed `Docs/` path casing. Re-ran spec placeholder scan and `git diff --check`; both clean.
Second follow-up spec review found final clarifications to patch before implementation planning: cache read/short-circuit in request flow, raw-input-free request context, explicit domain-error-to-HTTP mapping table, and rollout flag environment matrix.
Patched final spec review findings: request flow now includes cache read/full-hit short-circuit/partial-hit execution, request context is raw-input-free, error handling includes an explicit domain-condition-to-HTTP-status table, rollout controls include an environment matrix, and test strategy includes full/partial cache-hit behavior. Re-ran placeholder scan and `git diff --check`; both clean.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Design spec written and follow-up review issues patched at `Docs/superpowers/specs/2026-06-24-embeddings-request-orchestrator-refactor-design.md`. The final spec captures the approved scope B architecture, staged component split, request flow, migration/test strategy, domain errors, explicit HTTP mapping, observability, rollout controls with environment matrix, invariants, non-goals, explicit shim ownership, raw-input-free request context, cache read/write/value semantics, RG/billing characterization, full/partial cache-hit behavior, and response formatting parity tests. Verification completed: spec placeholder scan found no matches and `git diff --check` was clean. Bandit skipped as non-code documentation/task metadata only.
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
