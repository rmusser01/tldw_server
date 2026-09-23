# UAT370: remove action requires actual selection

Task: TASK-13260.277.20. Root owns native acceptance, tracker, git integration, and the separate UAT362 count repair. Scope is MediaReviewReadingPane membership visibility and its real-pane regression tests.

## Approved diagnosis/design

The Compare/spread reading pane calls the same card renderer for selected items and an unselected preview. The remove action currently checks only view mode. Its callback correctly filters selected IDs without clearing preview, so clicking it in the native zero-selection preview has no effect. Show the action only if the rendered media ID belongs to selectedIds, using the existing idsEqual helper for numeric/string consistency. Keep the count, preview, action semantics, and labels unchanged.

## Stage 1: Causal tests
**Goal**: Reproduce the enabled no-op with the actual reading pane.
**Success Criteria**: Unselected preview regression fails before repair; selected-ID and removal controls pass.
**Tests**: Zero-selection preview with actual translated Unstack label; selected ID number/string equivalence; unrelated selection does not authorize removal.
**Status**: Complete

## Stage 2: Membership gate
**Goal**: Hide selection removal for unselected rendered IDs.
**Success Criteria**: All causal tests pass through the actual pane; selected removal receives the original media ID.
**Tests**: Reading-pane suite, selection/batch interaction controls.
**Status**: Complete

## Stage 3: Review and native handoff
**Goal**: Freeze verified source for independent review and native acceptance.
**Success Criteria**: Scoped tests, lint/type comparison, diff checks and security applicability recorded; independent review clear; root native candidate confirms behavior.
**Tests**: Installed Vitest and lint/type entrypoints; root browser acceptance.
**Status**: In Progress

## Evidence

Initial native failure: `.tmp/uat-frontend-repair1-20260920/sqlite-multi-044-clear-preview.txt`. No frozen candidate changes. Full read-only ancillary observations are `/tmp/uat-native-extra-observations-triage.md` and remain outside this implementation.

Causal RED: `/tmp/uat370-causal-red.log`, 3 failed / 7 passed. Both fallback and actual English locale expose the inactive action for an unselected preview; a rendered ID outside selectedIds also exposes it. Number/string selected-ID callback controls already pass. Implementation reuses `includesId`, which delegates to `idsEqual`, rather than adding a new helper.

GREEN: `/tmp/uat370-green.log`, 3 suites / 44 tests passed (reading pane, selection-limit interaction, and batch toolbar). The full page selection suite includes actual removal through the existing shared action; the new real-pane controls verify numeric/string membership and forwarding the original media ID. Matched ESLint: 0 errors / 17 existing warnings before and after, no added diagnostics (`/tmp/uat370-lint-comparison.json`). Touched source/test have no trailing whitespace. Bandit ran with the project virtualenv: 0 Python LOC in this TypeScript-only scope (`/tmp/bandit_uat370.json`), which is not TypeScript security coverage. Manual source review confirms the production change only limits visibility using the existing selection predicate.

Source frozen for independent review. Exact four-file scope: `/tmp/uat370-changed-files.txt`. Native acceptance remains root-owned; task stays In Progress and this plan remains until acceptance.

Matched frontend TypeScript comparison completed: 93 existing diagnostics before and after, no additions/removals or touched-file diagnostics (`/tmp/uat370-type-comparison.json`, `/tmp/uat370-type.log`). Used the same installed compiler/config with a read-only HEAD-content baseline for the one production file, `node --max-old-space-size=12288 /tmp/uat370-typecheck.cjs`; no git mutation. Independent review and native acceptance remain pending.

Root independent review cleared the narrow source diff and reran all 10 real-pane tests successfully (`/tmp/uat370-root-review.log`). Native acceptance remains pending.
