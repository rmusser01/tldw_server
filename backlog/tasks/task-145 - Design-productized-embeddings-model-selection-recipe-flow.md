---
id: TASK-145
title: Design productized embeddings model selection recipe flow
status: Done
assignee:
  - Codex
created_date: '2026-05-09 03:37'
updated_date: '2026-05-09 03:49'
labels:
  - design
  - evaluations
  - embeddings
  - rag
  - webui
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create a design spec for upgrading the existing `embeddings_model_selection` evaluation recipe into a guided RAG-focused WebUI recipe flow. The approved direction is to keep the existing recipe/run/report machinery, make `/evaluations?tab=recipes` the primary surface, default to current RAG setup, support light labels through search-and-select expected sources with manual IDs in advanced mode, prefill current embedding model plus suggested candidates, and add an explicit one-click apply step after results.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A design document is written under Docs/superpowers/specs with the approved product flow, architecture, UX, safety, and testing sections.
- [x] #2 The spec is grounded in existing recipe framework, embeddings A/B test, shared WebUI/extension UI, and RAG/evaluations code paths rather than proposing a parallel eval surface.
- [x] #3 The spec defines non-goals and boundaries for V1, including no full RAG tuning merger and no auto-apply during eval execution.
- [x] #4 The spec identifies implementation risks, test coverage expectations, and open questions for the follow-up implementation plan.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Approved design-doc plan:
1. Write `Docs/superpowers/specs/2026-05-08-embeddings-rag-recipe-webui-design.md` from the approved brainstorming sections.
2. Ground the spec in existing code paths: `embeddings_model_selection`, recipe run Jobs/report APIs, embeddings A/B test execution, and shared `apps/packages/ui` Evaluations recipe UI.
3. Keep V1 scoped to improving the existing `/evaluations?tab=recipes` flow: current RAG setup defaults, light-label query/source selection, hybrid candidate selection, recommendation-first report, and explicit apply confirmation.
4. Include non-goals, safety rules, testing expectations, implementation risks, and open questions for the later implementation plan.
5. Run a spec review pass, update the task with verification notes, then ask the user to review the committed spec before implementation planning.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Created after user approved the product flow, architecture, UX, and execution/safety/testing sections during brainstorming. This task is for the design artifact only; implementation planning follows after user review.

Wrote `Docs/superpowers/specs/2026-05-08-embeddings-rag-recipe-webui-design.md`. Verification: `git diff --check -- Docs/superpowers/specs/2026-05-08-embeddings-rag-recipe-webui-design.md backlog/tasks/task-145 - Design-productized-embeddings-model-selection-recipe-flow.md` passed with no output. Spec review subagent returned APPROVED with no blocking issues. Bandit is not applicable because this task changed only markdown/design tracking files.

Commit attempt: `git commit --only -m "docs: design embeddings rag recipe flow" -- Docs/superpowers/specs/2026-05-08-embeddings-rag-recipe-webui-design.md "backlog/tasks/task-145 - Design-productized-embeddings-model-selection-recipe-flow.md"` failed with `fatal: cannot do a partial commit during a merge.` The repo already had unrelated staged/unmerged paths, so the design files remain staged but uncommitted to avoid including unrelated work.

Reopened for user-requested design hardening pass before implementation planning. Scope: patch the spec to clarify media-scoped V1 execution, expected source ID contracts, candidate readiness statuses, and staged apply-preview/apply boundaries.

Design hardening pass completed. Updated the spec to narrow V1 to media-backed RAG corpus execution where resolvable, make media IDs the V1 expected-source contract, require server-provided candidate runnable statuses and apply eligibility, document preview/copy-config fallback when config mutation is not yet safe, and split follow-up implementation into staged reviewable slices. Spec review subagent returned APPROVED. Verification: `git diff --check -- Docs/superpowers/specs/2026-05-08-embeddings-rag-recipe-webui-design.md "backlog/tasks/task-145 - Design-productized-embeddings-model-selection-recipe-flow.md"` passed with no output. Bandit remains not applicable because only markdown/task tracking changed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added a design-hardening pass to the embeddings RAG recipe WebUI spec before implementation planning. The spec now tightens V1 around the existing media-ID embeddings A/B execution path, explicitly keeps chunk/note labels out of scope until the backend schema supports them, requires server-owned candidate runnable statuses and apply eligibility, and documents preview/copy-config fallback behavior when live config mutation is not yet safe. It also adds a staged implementation breakdown so the follow-up work can land in reviewable slices: guided media-scoped UI, server hints/source helpers, recommendation/apply preview polish, then focused config apply. Verification was design-scoped: `git diff --check` passed on the touched spec/task files, and the spec-review subagent approved the hardened design. Bandit is not applicable because no Python source changed.
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
