# C02 Queue Health uses canonical percentages

Associated task: TASK-13260.278.18.83.45. Root recorded the actual Studio header showing 10000% after a completed optimization before edits. The existing Jobs producer returns percentages, including 0.0 when no jobs finished; the shared UI helper multiplies those percentages again.

## Design and ADR assessment

ADR required: no. Docs/ADR/README.md explicitly excludes small local bug fixes without a changed durable rule. This correction consumes the existing public percentage contract; no API, producer, schema, persistence, worker ownership, dependency, threshold, grading or security rule changes. One shared helper already serves both displayed percentage and summary. Keep rounding and the existing failure/backlog/processing precedence. Canonical 1 means 1%, and an absent value uses canonical no-finished 0 semantics; no fractional heuristic.

Exact pre-unit bytes for the helper and two existing test modules are retained only in memory. QueueHealthWidget, StudioTabContainer, the status service and Jobs/status producers were read as connected consumers and remain unchanged. Two stage6 fixture values and the utility fixtures need canonical percentage values; their original assertions remain.

## Stage 1: Prove canonical values at both consumers
**Goal**: Causal helper and real mounted widget regressions before production edits.
**Success Criteria**: Literal 100/85/0/1/absent oracles expose the repeated conversion and false default; real AntD Tooltip and widget summary/colors retain backlog/failure/processing behavior.
**Tests**: Existing utility module plus a new mounted QueueHealthWidget percentage module using real React/AntD/i18next and canonical status DTOs. No transport, provider or application runtime.
**Status**: Complete

## Stage 2: Correct the shared helper and fixture contract
**Goal**: Smallest complete correction in existing shared code.
**Success Criteria**: One helper consumes percentages directly with absent 0; existing utility and stage6 fixtures use canonical values; every original assertion is retained. Widget/parent and backend source remain unchanged.
**Tests**: Selected causal failures and genuinely affected original utility/parent contracts in normal UI/Web configs; retain passing controls without repeated red runs.
**Status**: Complete

## Stage 3: Qualify and freeze
**Goal**: Accurate bounded qualification and independent review handoff.
**Success Criteria**: Natural exit results, no new lint/security findings, exact source/test hashes reported directly. Root owns configured types/build/live reload and actual 100% acceptance.
**Tests**: Node20.19.5/Vitest4.0.18 normal UI/Web configs, NODE_OPTIONS8192, outer600; scoped baseline/current configured ESLint and TypeScript syntax. Bandit is not applicable to TS-only changes.
**Status**: In Progress

## Attempt ledger

- Attempt 1, unchanged production: normal UI natural exit 1, 2.313s outer; 13 failed / 5 passed across 18 utility/widget cases. Five utility failures and four mounted canonical badge failures establish repeated conversion/default defects (100 visibly rendered 10000%). Four later tooltip cases were initially inconclusive because the captured error excerpt ended before their details.
- Attempt 2, only those four inconclusive mounted cases: natural exit 1, 1.544s outer; 4 failed / 5 explicitly unselected. Failure-priority summary is a genuine percentage/precedence red. Zero, backlog and processing controls asserted before the actual AntD tooltip became visible. Correct only the common prerequisite by awaiting the real tooltip's visibility; all substantive expectations remain. Five passed controls are retained without replay.
- Attempt 3, corrected source and actual Tooltip visibility prerequisite: UI natural exit 0, 2.529s outer; 15 passed / 22 explicitly unselected (13 prior failures plus two genuinely affected stage6 status-query cases). The five prior passes remain retained: UI 20 distinct qualified cases across bounded runs, not a single combined 20-pass execution.
- First Web qualification: natural exit 0, 2.573s outer; 20 passed / 17 explicitly unselected, comprising all nine utility cases, all nine real Widget cases and the two stage6 status-query cases. The remaining 17 stage6 cases were not replayed. Maximum three attempts per distinct issue; no older evaluation/history/provider suite replay.

## Scoped qualification and handoff

- Four changed source/test files parse with zero TypeScript syntax diagnostics. Configured ESLint baseline/current: shared helper and utility test 0 errors / 0 warnings; new mounted Widget test 0 / 0; existing stage6 module retains exactly the same 1 error / 17 warnings with no new diagnostic fingerprints. This is a scoped no-new-findings result, not a clean whole-module lint claim. Bandit is not applicable to this TS-only unit.
- Self-review covered the entire one-line production delta, all original utility assertions, five new canonical utility cases, all nine real Widget bodies, both changed stage6 fixture values, current Widget color/summary/Tooltip branches, Studio hydration and direct Settings consumer, and canonical Jobs/status producers. No producer/schema/security/threshold/backlog/error/status branch changed. Native AntD Tooltip is mounted and awaited; translations use real i18next default interpolation, and transport/runtime/provider are not exercised.
- Exact pre-unit/current bytes and review deltas remain only in memory. Root owns independent source review, configured full typing, production build/reload and actual Queue Health acceptance. Stage 3 remains open for those gates; no live or all-43 acceptance is claimed.
