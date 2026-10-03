# Kanban card-detail contract repair

**Task:** TASK13260.280
**Goal:** Render empty and populated card checklists/comments using the canonical API contract.
**Approach:** Adapt API envelopes and name fields in the shared Kanban service. Preserve the existing component title/content interface. Load checklist items from the existing detail endpoint; propagate request failures.
**Scope:** `apps/packages/ui/src/services/kanban.ts`, its service regression tests, the ChecklistSection load-error state, and checklist rendering regressions. No backend, dependency, auth or frozen UAT changes.

## Stage 1: Reproduce
**Goal:** Demonstrate the envelope and field mismatch in the real service.
**Success Criteria:** Tests fail on the original service for empty/populated lists and canonical name-field mutations.
**Tests:** Vitest service contract tests with schema-shaped transport fixtures; no live API calls.
**Status:** Complete

- [x] Add independent literal checklist/item/comment fixtures and expected UI results.
- [x] Run the new tests before changing production code. All 12 failed on the original source, including the real checklist render crash. The initial reset-hook return was corrected before this run.

## Stage 2: Correct the shared boundary
**Goal:** Return renderable checklist/item/comment data and send canonical mutation fields.
**Success Criteria:** Preserve IDs, order, item checked state and current title/content UI fields; fail visibly if a detail request fails.
**Tests:** New service tests plus existing Kanban card-detail/date tests and a real checklist rendering test.
**Status:** Complete

- [x] Unwrap checklists/comments envelopes.
- [x] Fetch each checklist detail and map name to title/content.
- [x] Translate checklist/item create/update payloads to name and map responses.
- [x] Show checklist load failures and disable creation while the list is unavailable.
- [x] Run affected tests and formatting/type/lint checks supported by the existing environment.

## Stage 3: Review and qualify
**Goal:** Review the actual correction and complete separate-runtime UAT when authorized.
**Success Criteria:** Independent source review, normal local commit, and fresh distinct-fixture runtime validation before bug closure.
**Tests:** Source review; Bandit applicability recorded for TypeScript-only scope; live UAT remains pending browser recovery.
**Status:** In Progress

- [x] Review actual code and tests independently; resolved error visibility and Enter bypass. Final verdict: no blocking findings.
- [x] Record concise results in the task and existing UAT bug notes.
- [x] Commit locally after checks; do not push this tracking branch or alter the frozen runtime.
- [ ] Qualify a separate corrected runtime with a new fixture after browser recovery is authorized. Keep the S03 bug open until then.

Source checks: 25 affected tests passed; final test-selector change rechecked with all4 component tests passing. Formatting/diff checks pass; lint0errors/5inherited warnings. Focused TypeScript0touched diagnostics with3dependency diagnostics; no full-project pass. Bandit cannot parse the TypeScript scope and provides no security result. Live UAT and comment pagination beyond the first50 remain open.
