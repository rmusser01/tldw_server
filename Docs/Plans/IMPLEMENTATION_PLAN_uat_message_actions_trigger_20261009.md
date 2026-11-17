# A09 message action trigger investigation

Associated task: TASK-13260.278.18.83.45. Root recorded the actual saved grounded assistant reply's two normal More actions clicks and exact clicked IconButton before any edits. The settled clicked element is the real lucide-ellipsis Popover child, not the collapsed overflow chip. Root owns browser/runtime/build/live acceptance. No third live click is performed here.

## Design and ADR assessment

This is a bounded existing component/trigger boundary investigation. ADR required: no. No durable transport, ownership, persistence, permission, dependency or UI architecture rule changes; existing shared button and AntD contracts remain the authority. ADR inventory was searched. Root approved real mounted causal tests before any proposed production repair.

Installed AntD6.2.1 delegates Popover through tooltip1.4/trigger3.9, which injects click/hover/focus events and a DOM ref. Common/IconButton forwards onClick/ref but drops other DOM event and ARIA props. Click forwarding therefore does not yet explain the exact live failure. Message mounts the action bar while not processing/editing; hover controls CSS visibility, not component lifetime. The collapsed native More chip is inert, but root's retained clicked element excludes it; do not patch that independent seam or call it this failure.

New tests mount real AntD/ConfigProvider and actual Common/IconButton/MessageActionsBar, without trigger/Popover/Tooltip/source-extraction mocks. A native button control qualifies the installed trigger fixture. Completed assistant actions must open and dispatch Note/Flashcard choices exactly once; hover and explicit focus compare native and shared-button behavior independently. A green real click cannot establish live recovery. An attributable hover/focus red may justify a separately labelled small shared event-forwarding correction after root reviews the result/design; do not infer that fixes the live click.

Pre-unit IconButton and MessageActionsBar bytes retained only in memory. No proofs, live requests, runtime/config/build/task edits, providers, dependencies or Git actions.

## Stage 1: Qualify the actual trigger boundary
**Goal**: Distinguish real click behavior from injected-event loss.
**Success Criteria**: Native trigger controls succeed; any failure has a concrete observed real-component oracle and fixture failures are excluded from causal attribution.
**Tests**: NEW MessageActionsBar.real-popover.test.tsx under normal configured UI/Web suites; real native/assistant click and real native/shared hover/focus. No existing test edits.
**Status**: Complete

## Stage 2: Apply only an attributed correction
**Goal**: Minimal repair supported by causal red and connected caller review.
**Success Criteria**: Root agrees the source cause; corrected required behavior is green without changing More/menu/save callbacks, guards, styles, title, disabled/type/ref behavior or wire/backend scope. If click is green, live click remains explicitly unknown.
**Tests**: Failed source oracles and genuinely affected controls only, preserving original tests.
**Status**: Complete

## Stage 3: Qualify and freeze
**Goal**: Accurate bounded evidence and independent source handoff.
**Success Criteria**: Configured UI/Web qualification, scoped baseline/current ESLint and syntax, no new findings. Root independent review, combined types/build and actual live acceptance remain separate gates.
**Tests**: Exact Node20.19.5/Vitest4.0.18, NODE_OPTIONS8192, activated supported Python3.12 stdlib subprocess timeout600; normal configured plugins and workers. Bandit N/A to TypeScript. No global types/build here.
**Status**: In Progress

## Attempt ledger

Maximum three issue attempts; fixture/setup failures are not source red. Record actual natural exits and retain passing controls without repeating them for combined counts. Production remains unchanged for the initial causal qualification.

Attempt 1, unchanged production UI: normal configured Node20.19.5/Vitest4.0.18/8192MB, outer600, natural exit1 in4.282s (Vitest4.03s). Seven tests: native click/hover/focus controls and both real completed assistant More→Notes/Flashcards callbacks/close passed (5). Only shared IconButton hover and explicit keyboard-focus failed (2), with no tooltip DOM; keyboard focus itself succeeded. No fixture/setup failures. Original live More click is NOT reproduced and remains UNKNOWN. Five passes are retained; do not replay unchanged UI bodies to get a combined count.

Minimal separate source-backed proposal sent root before production: preserve original IconButton custom aliases/default type/style/ref/onClick while forwarding remaining standard native button props. Scope only this existing shared component plus new test and plan; no message bar/chip/provider/backend edits. Native click green does not support a claim that forwarding other events fixes the observed live click.

Root approved the minimal separate Tooltip repair after clean causal red. Source scope is only Common/IconButton.tsx: standard React button attributes in its type, rest-prop forwarding, and legacy aria-haspopup/expanded/controls precedence only when the alias is supplied (otherwise preserve the standard attribute). Mandatory ariaLabel and existing type/title/class/ref/click/disabled behavior remain explicit. No MessageActionsBar or collapsed-chip change. Root requested meaningful real tooltip dismissal/description, native default/ref/click/disabled and standard-vs-legacy ARIA guards; these are added to the new module. Initial five UI successes remain unchanged/retained. Baseline scoped ESLint:0errors/1 inherited no-explicit-any warning in IconButton; new test0/0. Existing apps-base Next lint emits its inherited pages-directory advisory.

Declared next phase: UI selects only the two failed shared-tooltip cases and three new preservation controls (5); genuinely affected original MessageActionsBar menu/feedback modules are qualified separately without unchanged initial UI replay. First configured Web executes the new10 and same genuinely affected originals. No CLI worker/cache overrides. Original actual live click remains UNKNOWN.

Attempt 2, corrected source selected UI plus genuinely affected originals: natural exit0 in1.371s (Vitest1.18s). The two formerly failing shared Tooltip cases and three new preservation controls passed; original menu5/feedback2 passed. Current command12 passed and5 intentionally filtered (reported skipped by Vitest), retaining those five unchanged baseline successes. UI17 distinct qualified by split5 retained +12 current; no one-run17 green claim and no fixture failure.

Attempt 3, first configured Web qualification: started2026-10-09T09:02:09.124681Z, natural exit0 in1.907s (Vitest1.66s). All new10 + original menu5/feedback2 passed,17/17, no skips. No worker/cache/config overrides and no browser/provider/runtime action.

Current scoped ESLint: both touched TypeScript paths0errors/0warnings, natural0; baseline source0errors/1 inherited no-explicit-any warning, new test0/0. The removed unnecessary existing ARIA cast accounts for the warning reduction. Existing apps-base Next pages-directory advisory remains inherited. Static TypeScript transpile syntax: both paths zero diagnostics, natural0; this is not semantic configured typing. Bandit N/A, TypeScript-only scope.

Self-review read the complete source and every current new body. Actual AntD native/shared hover/focus and dismissal, injected tooltip description, actual More→Notes/Flashcards callbacks/close, explicit label/alias/default type/ref/click/disabled guards remain substantive DOM oracles. Native/actual click passing does not explain the original live failure. MessageActionsBar production and the collapsed-chip implementation remain byte-identical to pre-unit memory. No old test bodies changed. No Critical/Important/Minor finding identified in this owned correction. Root independent source review, combined configured types/build and actual rebuilt live keyboard/menu recovery remain OPEN. Stage3 stays In Progress for these gates.
