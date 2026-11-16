# Luna synthesis repair implementation plan

**Goal:** Repair contributor continuity and overview citation reliability, rerun Luna, and publish verified evidence to PR #3480.
**Spec:** `Docs/Design/POLITICAL_PDF_LUNA_SYNTHESIS_REPAIR_2026_11_15.md`.
**ADR assessment:** No new rule; existing Jobs/Workspace/provider/provenance owners remain unchanged.
**Constraints:** No new dependency, no validator relaxation, no changed source offsets, no automatic paid retries, no overwriting original failed drafts or human labels.

## Stage 1: Reproduce and bound the repair
**Goal:** Identify source-context loss and capture an exact overview probe; inspect current-head CI.
**Success Criteria:** Source-backed boundaries and caller/test inventory recorded; unknown original rejection remains honestly unknown.
**Tests:** Exact saved-prompt probe, offline ID classification, CI log inspection.
**Status:** Complete

## Stage 2: Implement and independently verify
**Goal:** Add verified heading context, exact citation vocabulary/diagnostics and accurate failure wording; fix actionable CI failures.
**Success Criteria:** Regression tests first fail for demonstrated behavior, then pass without weakening scope/version guards.
**Tests:** Single/multiple contributor contexts, mixed batches, shared references, ambiguous nodes, bounded diagnostics, current-call IDs, old cache projections; connected tests, types/lint, Bandit and pre-commit.
**Status:** Complete — verified context and literal author notes now survive reduction independently; 227 connected backend tests and independent reviews pass. CI follow-through continues in Stage 4.

## Stage 3: Rerun Luna and source-audit outputs
**Goal:** Generate a separate full chapter output and rerun the frozen QA matrix.
**Success Criteria:** Preserve immutable first responses and failures; verify exact evidence and review selected authors/claims in source; no semantic pass inferred from structural success.
**Tests:** Existing CLI source/retrieval/QA/summary stages, actual Jobs checkpoint/readback, targeted source audit and review/reload.
**Status:** In Progress

## Stage 4: Publish and inspect CI
**Goal:** Update report, tasks and draft PR against dev with the tested final state.
**Success Criteria:** Tests/security/publication checks pass; new head and current CI inspected; unresolved external or human acceptance limits explicit.
**Tests:** Task format/IDs, secret/diff checks, relevant pre-commit hooks, GitHub checks.
**Status:** Not Started

Remove this task-owned plan when all stages are complete; retain design and result evidence.
