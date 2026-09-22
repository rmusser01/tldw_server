# UAT392 C-03 exact-match evaluation repair

Task: TASK-13260.278.5.2. PR #2979. Full/native UAT remains paused.

## Stage 1: Establish the execution contract
**Goal**: Trace the existing UI, dataset schema, runner and canonical result API; reproduce advertised option failures.
**Success Criteria**: A literal two-row dataset has one match and one mismatch; case-sensitive configuration is honored.
**Tests**: Actual EvaluationRunner and saved run results; case-sensitive/default controls.
**Status**: Complete

## Stage 2: Require real browser outcomes
**Goal**: Replace the empty recipe click with UI-created exact-match evaluation and canonical completed run.
**Success Criteria**: Input/expected values, two sample identities, scores, aggregate and saved IDs survive reload; failed or unavailable execution fails.
**Tests**: Focused result-oracle controls and strict existing journey; types and lint.
**Status**: In Progress

## Stage 3: Review and publish
**Goal**: Record exact execution evidence and remaining C-03/B-09 coverage limits.
**Success Criteria**: Scoped checks and security review; retained first-attempt CI; no generated captures in Git.
**Tests**: Remote browser check; export/batch/foreign-owner and B-09 draft commit remain separately accounted variants until implemented and executed.
**Status**: In Progress
