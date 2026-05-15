---
id: TASK-378
title: Plan cache-aware chat cost-control implementation
status: Done
assignee: []
created_date: '2026-05-15 07:12'
updated_date: '2026-05-15 07:17'
labels:
  - planning
  - chat
  - world-books
  - cost-control
  - llm-cache
dependencies:
  - TASK-377
documentation:
  - >-
    Docs/superpowers/specs/2026-05-15-chat-worldbook-cache-cost-control-design.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create the implementation plan that turns the approved chat/world-book cache cost-control design into staged, reviewable implementation work. The plan must decompose the work into slices for measurement, guardrails, provider billing cache controls, local vLLM/llama.cpp diagnostics, and reporting without starting code implementation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A repository implementation plan is created under Docs/superpowers/plans and references TASK-377 and the approved design spec.
- [x] #2 The plan decomposes the work into staged implementation slices with clear file ownership, tests, and verification commands.
- [x] #3 The plan calls out which slices should be separate PRs/tasks and preserves the measurement-before-behavior-change boundary.
- [x] #4 The plan includes vLLM and llama.cpp local inference diagnostics separately from paid provider billing cache controls.
- [x] #5 The Backlog task records the plan path, verification notes, and final summary.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Create a repository implementation plan under Docs/superpowers/plans that references TASK-377 and the approved design spec.
2. Decompose implementation into reviewable slices: prompt envelope primitives, world-book diagnostics/preview parity, usage normalization/persistence, guardrails, paid provider cache controls, local vLLM/llama.cpp diagnostics, and reporting.
3. Include exact file ownership, tests, verification commands, and PR/task boundaries so implementation can proceed incrementally without changing provider behavior before measurement is in place.
4. Update TASK-378 with the plan path and verification notes, then close it once the plan artifact is reviewed for internal consistency.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Plan written: Docs/superpowers/plans/2026-05-15-chat-worldbook-cache-cost-control-implementation-plan.md

Verification: git diff --check passed. Runtime tests and Bandit were not run because this is a docs/planning-only change with no executable code.

Self-review: plan preserves the measurement-before-behavior-change boundary, separates paid provider billing-cache controls from local vLLM/llama.cpp diagnostics, and requires fresh official provider documentation verification before Stage 6 adapter changes.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Created the cache-aware chat/world-book cost-control implementation plan under Docs/superpowers/plans. The plan decomposes the approved TASK-377 design into staged implementation slices for prompt envelopes, world-book diagnostics, usage normalization and persistence, guardrails, paid provider cache controls, local vLLM/llama.cpp diagnostics, and reporting, with tests and verification commands for each slice.
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
