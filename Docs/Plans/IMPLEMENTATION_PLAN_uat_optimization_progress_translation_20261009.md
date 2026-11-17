# C02 optimization progress translation repair

Associated task: TASK-13260.278.18.83.45. Root recorded and read back this unit before edits under standing human all-43 fix/validate authorization.

## Design and boundaries

The actual pending optimization has max_iterations=1, iterations_completed=0 and null metrics. Its progress drawer crashes the Prompts page. The English settings resource at managePrompts.studio.optimizations.progress is an object; the existing progress.title leaf is a string. The actual repository ICU plugin passes the object through to React. Earlier transport tests replace translations with string defaults and do not exercise this boundary.

Change only the panel lookup to the existing progress.title leaf. Keep shared i18n, locales, transport, worker/configuration, auth/owners, polling, score truth, stored iterations and all prior frozen changes intact. Root separately cancelled the queued original and owns the worker opt-in/reload and actual recovery; queued status is distinct from this render defect.

ADR assessment: no new ADR required. This is a local consumer correction using an existing string resource, without any new durable format, dependency, authority or i18n policy. [ADR-001](../ADR/001-adr-workflow-and-governance.md) distinguishes these local bug fixes from durable decisions; [ADR-055](../ADR/055-prompt-studio-db-single-implementation.md) remains the underlying canonical Prompt Studio storage contract, unchanged here.

Production scope: apps/packages/ui/src/components/Option/Prompt/Studio/Optimizations/OptimizationProgressPanel.tsx, exactly one translation-key change. New test scope: its __tests__/OptimizationProgressPanel.i18n.test.tsx using real English resources, ICU, I18nextProvider, QueryClient, AntD Drawer/Statistic, shared primitives and the actual PromptPageErrorBoundary. Only the external apiSend seam is mocked.

## Stage 1: Real i18n causal regression
**Goal**: Reproduce the actual pending drawer crash before production edits.
**Success Criteria**: The original pending/max1/iter0/null-score fixture fails through the real page boundary, attributed to the object resource; canonical request paths are retained.
**Tests**: Sole pending case in the normal configured UI environment before production change.
**Status**: Complete

## Stage 2: Minimal repair and affected qualification
**Goal**: Use the existing string leaf and retain status/result/error contracts.
**Success Criteria**: New pending/running/completed/failed/detail-refusal/iteration-refusal cases pass in configured UI and first Web qualification; scoped ESLint has no new findings and syntax passes.
**Tests**: Only the new real-i18n module, normal Node20.19.5/Vitest4.0.18, 8192MB, outer600; no prior18 mounted controls replay. Bandit N/A: TypeScript only.
**Status**: Complete

## Stage 3: Review and actual recovery
**Goal**: Freeze source/tests for independent review and root-controlled final acceptance.
**Success Criteria**: Independent connected source review clear; root combined types/build/load and one later actual optimization recovery render succeeds.
**Tests**: Root-owned configured typing/build and live qualification only.
**Status**: In Progress

## Bounded attempt ledger

Maximum three distinct issue attempts; setup/operator failures are separate from causal source red. Retain pre-unit bytes and reports in memory, no proof files. No runtime/model/browser/profile/parent-task/Git actions. Source baseline SHA256 9e694183c235875482b7b93c4d7933e9e907855144c31e5f67e106bfcf57304b.

Attempt 1 (causal original-source UI): natural1/2.534s outer, one intended failed case/five pattern-filtered controls. The actual page boundary retained the exact React error: object keys title/currentIteration/bestScore/iterations/timeline are not valid children. Fixture setup/import/resource leaf precondition succeeded; no fixture failure. Production remained baseline throughout. The sole key correction follows this valid red. Baseline ESLint source0errors/3 inherited warnings (two no-explicit-any and unused query), plus existing apps-base pages-directory notice. An oversized source-bearing JSON response was truncated; bounded source-free reporting will be used for current comparison, with no test replay.

Attempt 2 (current-source UI and first Web): UI natural1/4.677s and Web natural1/5.007s outer, each five passes and one new fixture failure. Pending/running/failed/detail-refusal/iteration-refusal are qualified. Completed appears in both the real Badge and Descriptions label, so the new single-text selector is ambiguous; this is not a production failure or new causal red. Before attempt3, correct only that query to the actual Badge data-ds-component selector while preserving exact Completed text and every remaining assertion. Retain the five passes per client; attempt3 is only the previously unqualified completed case per client. No production or old-suite replay.

Attempt3 (failed-only completed control): UI natural0/1.677s and Web natural0/1.711s outer, each sole selected case passed/five pattern-filtered. Retain six distinct cases per client through split executions (five earlier plus one corrected); do not claim one full-six green execution. No unchanged prior18 replay. The selector correction preserves exact status, canonical completed progress100, stored finalscore75.0%, actual zero iteration score0.0%, evidence note and ICU-interpolated savedPrompt22 assertions.

Final scoped static qualification: source ESLint0errors/3 inherited warnings unchanged from baseline; new test0errors/0warnings including final selector. Same apps-base pages-directory notice. TypeScript transpile syntax passes both touched TSX paths; final changed test syntax passes. This is syntax only, not configured project typing. Bandit N/A for this TypeScript-only unit. Inherited AntD valueStyle/items.children deprecation notices are unchanged and outside this key-only correction.

Self-review read complete six bodies and fixture, real page boundary, connected canonical service/query/status/score/iteration consumers, actual ICU wrapper and English resources. The production byte comparison proves exactly the one translation-key change and no other hunk. Pending, running, completed, stored error and both request-refusal boundaries are qualified through normal mounted components and real i18n; external apiSend is the only mock. No actual HTTP/auth/owner denial, polling lifecycle timing, backend worker/provider output or browser/model/runtime acceptance is claimed.

Source/tests frozen for root independent review. Root has read the one-line consumer and all six bodies; final corrected Badge selector is now ready for narrow re-read. Root combined types/build/load and later single actual optimization recovery remain open; Stage3 remains In Progress. No parent task, private profile, worker, runtime, dependency, Git or unrelated source action.
