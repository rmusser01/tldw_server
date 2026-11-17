# C02 canonical optimization transport and saved fields

Associated task: TASK-13260.278.18.83.45 (root-owned; separate optimization note recorded and read back before edits).

## Scope and ADR assessment

Reuse existing project-authorized optimization list/detail/iteration/cancel routes and canonical saved response fields. Repair the existing shared service and two existing consumers; remove the unsupported Delete action/service. Preserve the create provider/model/budget request byte-identically. No backend route, history storage, provider operation or runtime change.

ADR-020 governs existing owner-bound database access; ADR-025 governs the unchanged creation provider contract; ADR-055 governs the existing shared repository implementation. Canonical client alignment introduces no new durable architecture rule, so no new ADR is needed.

## Stage 1: Contracts and causal regression
**Goal**: Exercise the real service through apiSend and mounted existing list/detail views against backend-shaped responses.
**Success Criteria**: Exact current route/response mismatches fail before production edits; unsupported Delete is absent; refusal and zero/missing metric controls are meaningful.
**Tests**: New Optimization.transport.test.tsx with real QueryClient/AntD/store/components, external apiSend boundary only; root case/outer caps retained.
**Status**: Complete

## Stage 2: Minimal canonical repair
**Goal**: Align routes, pagination and saved field consumers without creating history or deriving model results.
**Success Criteria**: List/detail/iterations/cancel use existing canonical routes; empty iteration history is explicit; finite zero is retained and missing scores unavailable; unsupported Delete removed; create function and frozen C02 target callers unchanged.
**Tests**: Causal positives and failures, selected identity retirement, affected existing optimization design-system fixture.
**Status**: Complete

## Stage 3: Qualification and independent review
**Goal**: Qualify affected UI then Web, syntax/lint and independent changed-flow/new-body review, then freeze.
**Success Criteria**: Normal Node20.19.5/Vitest4.0.18, 8192MB, 600-second outer/default case caps; no new scoped lint findings; independent prompt-cookie review complete. Root owns combined types/build/live.
**Tests**: New transport module plus genuinely affected optimization design-system module only; no completed C02 provider/budget replay. Bandit N/A for TypeScript.
**Status**: In Progress

## Evidence ledger

- Read canonical endpoint/schema/repository and all affected client callers. Exact pre-unit bytes retained in functions memory only. Canonical list is `/list/{project_id}`, detail `/get/{optimization_id}`, iterations `/iterations/{optimization_id}`, cancel `/cancel/{optimization_id}` with optional JSON string reason. No DELETE endpoint exists. Creation remains the existing POST returning Jobs identity.
- Saved iterations are optional persisted rows; the current engine does not synthesize them. This unit must not invent iteration records, candidate IDs or scores.
- Maximum three distinct attempts; fixture/setup failures are recorded separately.
- Causal attempt 1: configured UI selected three route/body regressions; 3 failed, 12 filtered, natural 1, 4.230 s outer. Actual service sent unmounted base list URL, swapped cancel URL and object reason; mounted list falsely rendered No optimizations yet. No fixture/setup failure or production edit preceded this red.
- Attempt 2: first full affected UI command (new transport module plus existing optimization design-system module), 15 passed/3 failed, natural 1, 7.714 s outer/7.37 s Vitest. The three new failure-view controls exposed unsupported `Alert variant="danger"` in the repair; the real primitive accepts `error`. This is a production error, not fixture setup. Before the next edit: correct only those three new variants, retain all assertions and the 15 passed cases, then qualify only the three failed UI IDs and the first full affected Web scope.
- Attempt 3: only the three failed UI IDs, natural 0, 3 passed/12 intentionally filtered, 3.107 s outer/2.78 s Vitest. Retained previous 15 passes: 18 distinct affected UI cases, not a combined 18-green run. The first full affected Web command passed all 18, natural 0/no skips, 4.858 s outer/4.52 s Vitest. All used actual Node20.19.5/Vitest4.0.18, 8192 MB and Python subprocess timeout600 with unchanged default case timeouts.
- Exact affected modules: `apps/packages/ui/src/components/Option/Prompt/Studio/Optimizations/__tests__/Optimization.transport.test.tsx` (15 new cases) and `OptimizationDesignSystem.test.tsx` in that same directory (3 existing cases, fixture fields only aligned). UI failed-only filter: `shows an owner refusal|shows missing detail truthfully|distinguishes iteration retrieval failure`; Web ran both complete modules through its configured runner.
- Scoped configured ESLint: 5 paths, natural 0, 0 errors, 40 inherited warnings versus 45 before, 0 new normalized rule/message/source findings; new module 0 warnings. TypeScript parser: all 5 paths, 0 syntax diagnostics. Bandit not applicable to this TypeScript-only scope. Existing AntD deprecation warnings remained; no test guard or dependency change.
- Exact createOptimization function bytes match pre-unit; frozen C02 target callers and C03 recovery sources match their previous hashes. Self-review read every unit hunk, complete new bodies, all represented client callers, and actual endpoint/schema/repository ownership contracts. Independent prompt_cookie_sync_fix source review CLEAR: all five frozen hashes, full unit hunks, all15 new bodies/3 existing fixture alignments and connected canonical endpoint/schema/repository/engine/transport/Alert read without execution. Root combined types/build/live remain open; real HTTP/owner/worker/model and beyond-first-page records are outside this unit's evidence.

## Frozen scope

- `apps/packages/ui/src/services/prompt-studio.ts`: `0b68ffb223bb549a44ef4dd43950573a93c05004ed68259f822d09d0a12c60d8`
- `apps/packages/ui/src/components/Option/Prompt/Studio/Optimizations/OptimizationsTab.tsx`: `72c3d2c7ee37ca789520fc22b3a927a36317802ed8721ff63d3ffcc5d87263e2`
- `apps/packages/ui/src/components/Option/Prompt/Studio/Optimizations/OptimizationProgressPanel.tsx`: `9e694183c235875482b7b93c4d7933e9e907855144c31e5f67e106bfcf57304b`
- Existing `OptimizationDesignSystem.test.tsx`: `c5de856c13eef0027e18a136840e319b4003f8d5045284d9a943d8926027badd`
- New `Optimization.transport.test.tsx`: `f727e9865ebfad2ed19f5c34be3fce76cefd4c3b68fce1409860a37ddd90d386`
