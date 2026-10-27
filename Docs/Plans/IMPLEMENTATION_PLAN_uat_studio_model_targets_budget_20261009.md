# C02 Prompt Studio model targets, results, and optimization budget

Associated task: TASK-13260.278.18.83.45 (parent-owned; C02 note established before edits).

## Scope and ADR assessment

Repair the three existing Test Case Run, Evaluation, and Optimization callers and their existing service types. Reuse the unchanged Execute provider-catalog normalization and validity rules through a small shared field hook. Preserve Execute, recipe recovery, backend authorization, and transport behavior. Canonical run results retain actual/expected outputs and milliseconds; absence of a grade is explicit. Strategy defaults cannot overwrite the reviewed iteration cap. API refusals stay failures.

[ADR-025](../ADR/025-llm-provider-adapter-routing-and-overrides.md) governs the existing provider adapter contract; no new durable architecture rule or backend API is introduced. Existing resource ownership/routing contracts remain authoritative. No new ADR is needed.

## Stage 1: Connected contracts and causal regression
**Goal**: Exercise real mounted callers and the real service layer at the external apiSend boundary.
**Success Criteria**: Backend-shaped catalog/results and selected budget expose the current defects before production changes.
**Tests**: Three caller payloads, canonical actual/expected/time/ungraded display, selected budget vs defaults, refusal.
**Status**: Complete

## Stage 2: Minimal shared repair
**Goal**: Add provider/model selection using existing normalization; align request/results types and preserve chosen budget.
**Success Criteria**: Explicit targets follow each canonical backend shape; provider changes retire incompatible models; canonical/legacy display remains truthful; refusal cannot produce success.
**Tests**: Mounted positive/default/provider-change/invalid-catalog/empty-catalog and error controls; existing affected design-system checks.
**Status**: Complete

## Stage 3: Qualification and independent review
**Goal**: Qualify affected UI/Web tests, scoped lint/syntax, full changed-flow/new-body review, and freeze.
**Success Criteria**: No new scoped static findings; independent review complete; root retains combined types/build/live acceptance.
**Tests**: Configured UI/Web on original Node 20.19.5/Vitest 4.0.18, 8192 MB heap, 600-second outer cap; maximum three distinct attempts with setup failures distinguished. Bandit is not applicable to TypeScript.
**Status**: In Progress

## Evidence ledger

- BEFORE the compiler-only correction: root's combined configured UI/Web checks both exited naturally 2 (96.554 s / 72.290 s), with only TestCaseRunModal.tsx:56 TS2339. The canonical/legacy union is not narrowed in the falsy arm of the combined `payload && "results" in payload` ternary. Use an explicit truthy outer guard followed by the existing structural discriminator, preserving canonical-first selection, legacy fallback and array validation. No casts, new fallback or runtime-test replay; root will run fresh combined types after the coordinated C04 delta.
- Compiler-only correction frozen at TestCaseRunModal.tsx SHA256 `73ed0b3126f7e2f13503e8e06e2b4fbe194c38faf3bd27eac151188e201c812a`; other eight edited source/test hashes unchanged. Exact branch review retains undefined for absent payload, canonical results whenever its property exists, otherwise legacy data; unchanged array refusal. Configured apps-base ESLint natural 0, 0 errors/2 unchanged inherited any warnings; TypeScript syntax parse 0 errors. No unchanged test replay; current combined types remain root-owned.
- Compiler-only peer review independently CLEAR at verified corrected source hash: canonical-first, legacy and absent branches retain prior runtime semantics and unchanged array refusal; no cast, extra fallback or guard relaxation. No new execution evidence from this source review.

- BEFORE the narrow fixture continuation: expanded configured UI qualification returned 20 passed/2 failed, natural 1, 41.318 s outer. The design-system mutation fake still supplied the old envelope to the now-array success callback. The provider-change test attempted dispatch during the effect's incompatible-model clearing window; the production guard refused it. Root authorized only callback-array alignment and a bounded wait for actual old-model disappearance. Retain the 20 passes, qualify only the two failed IDs, then run the first affected Web command. No production guard or payload/refusal assertions change.

- Read current caller/service/backend request and result contracts plus existing Execute utility. Captured exact pre-unit source bytes in memory only. No runtime/provider/request or dependency action.
- Baseline configured ESLint: natural 0; four production paths have 35 inherited warnings (3/4/3/25), no errors.
- Two read-only capture commands hit tool-output truncation; neither ran tests nor changed files. Chunked source capture completed successfully.

- Causal attempt 1: configured UI new module 7 failed/0 passed, natural 1, 21.998 s outer; reached actual forms/transport. Missing provider controls, canonical results omission, and API refusals incorrectly succeeding.
- Causal budget-only continuation: 1 failed, 7 filtered, natural 1, 3.870 s outer; exact POST cap expected 1, actual 10.
- Configuration Forms unmount on Review; read preserved fields with getFieldsValue(true). Target watches preserve unregistered fields.
- Automatic review rejected full-file snapshot writer for concurrent-change risk. Verified unchanged current hashes; safer minimal apply_patch edits succeeded. No rejected write occurred.
- Initial causal cases green: UI 8/8, no skips, natural 0, 19.175 s outer.
- Exact affected files: new `Studio/__tests__/Studio.model-targets.test.tsx` (15 cases), `Studio/TestCases/__tests__/TestCaseRunModal.design-system.test.tsx` (2), `Studio/Evaluations/__tests__/EvaluationDesignSystem.test.tsx` (2), `Studio/Optimizations/__tests__/OptimizationDesignSystem.test.tsx` (3), under `apps/packages/ui/src/components/Option/Prompt/`.
- Root-authorized fixture continuation: UI only `-t 'clears an incompatible model|renders run result summary and status'` in the new module and Run modal test; 2 passed/15 filtered, natural 0, 3.325 s outer. Retained 20 + 2 = 22 distinct UI passes; no single combined 22-green UI run is claimed.
- First affected Web command used all four files through `../packages/ui/src/...`: 22/22 passed, no skips, natural 0, 35.813 s outer. Both configured commands used actual Node 20.19.5/Vitest 4.0.18, 8192 MB, 600-second outer cap and unchanged default case limits.
- Scoped final configured ESLint: natural 0, 0 errors, 54 inherited warnings versus 55 baseline across nine edited source/test paths; normalized rule/message/code fingerprints show 0 new findings. New shared field and new mounted module each have 0 warnings. All nine TypeScript syntax parses have 0 errors. Bandit N/A (TypeScript only).
- Self-review covered every production hunk, complete new 15 bodies, changed display-only fixtures, actual canonical schema/runner/endpoint shapes and connected consumer inventory. Independent prompt-cookie peer review COMPLETE/CLEAR at all nine frozen hashes: all production hunks/full helper/new bodies/DS deltas plus actual services, Form/store consumers, canonical schemas/owner-gated endpoints/TestRunner read; no substantiated priority finding. This source-only review adds no test execution or provider/auth acceptance. Source/tests remain frozen; root owns current combined types/build/live acceptance. No provider/model/request action performed here.
- Execute/utility and all three frozen C03 source/test hashes match pre-unit bytes. Removed only the completed own C03 recovery plan after root confirmed manual GET-only reopen and one reload of the exact existing run, both configured types, build and live acceptance.
