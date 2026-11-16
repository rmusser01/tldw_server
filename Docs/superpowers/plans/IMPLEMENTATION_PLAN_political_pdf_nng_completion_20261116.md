# Political PDF NN/g interface completion

Design: [POLITICAL_PDF_NNG_COMPLETION_2026_11_16.md](../../Design/POLITICAL_PDF_NNG_COMPLETION_2026_11_16.md). Tasks: TASK-15505.13–15. PR #3480 against dev. User authorized implementation and PR update; no merge requested.

ADR required: no; existing Research Workspace (ADR-007), Media navigation and verified document-operation ownership are preserved. No persistence, public API or provider change.

## Stage 1: Freeze scope and regressions
**Goal**: Record the four bounded designs, callers and failure cases before implementation.
**Success Criteria**: Task records and design exist; each owner produces a failing behavioral test for new logic.
**Tests**: Saved/imported/missing identity; checkpoint reader and review lock; bookmark precedence and validated scoped fallback.
**Status**: Complete

## Stage 2: Implement and integrate
**Goal**: Deliver output identity/waiting guidance, live completed-chapter reader and generated PDF Contents.
**Success Criteria**: Existing guards and truthful unavailable states are retained; English locale and callers are connected; focused tests pass.
**Tests**: Focused tests plus connected Stage1/review/navigation suites, type checks and scoped lint.
**Status**: Complete

## Stage 3: Review and walkthrough
**Goal**: Independently review the change and verify actual desktop/mobile/keyboard behavior.
**Success Criteria**: Same saved PDF/Luna output checked; isolated running-checkpoint fixture verified; detector and peer findings resolved or recorded accurately.
**Tests**: Batched visual/keyboard walkthrough, scoped static checks, independent code review, no changes to accepted raw evidence.
**Status**: Complete

## Stage 4: Evidence and delivery
**Goal**: Update the NN/g ledger, tasks and existing draft PR with the final implementation and validation.
**Success Criteria**: Four UI outcomes have traceable evidence; semantic/human acceptance stays explicitly open; scoped hooks pass; committed/pushed changes and CI status reported accurately. Preserve the human Change summary gate.
**Tests**: Diff/secret/task-format checks, scoped pre-commit, required PR CI checks. Bandit is not applicable if the final change remains TypeScript/locale/docs only.
**Status**: In Progress

November16 evidence: [completion report](../../Reviews/POLITICAL_PDF_NNG_COMPLETION_2026_11_16.md). Four UX outcomes pass local regression and actual desktop/mobile/keyboard checks (running state is an isolated checkpoint fixture). Late full-matrix failures reopened TASK-15505.8; bounded fixes and current-head hosted qualification are tracked separately from the completed UI behavior. ADR required remains no. Provider reruns and human acceptance are not part of this UI proof.

Remove only this plan when all stages are complete; preserve the design and review evidence.
