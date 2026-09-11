# PR2761 React hook enforcement — TASK-12116

## Scope and baseline

This batch addresses the 8 `react-hooks/purity` and 11 `react-hooks/static-components` diagnostics in the full shared-UI/WebUI-page scan. Both rules are now errors in `apps/tldw-frontend/eslint.config.mjs`. The parent change separately fixed two `use-memo` findings and enabled that rule. The other four disabled compiler rules are outside this batch: the separate inventory contains 243 refs, 89 set-state-in-effect, 22 immutability, and 49 preserve-manual-memoization findings (403 total).

The pre-change scan contained 5,154 files and 1,402 errors. The component follow-up scan covered 5,158 files (4,996 shared UI source files including tests, plus 162 WebUI pages), with zero unsuppressed diagnostics from these three rules. At that point, the remaining 1,381 errors had exactly the same counts by rule as the unrelated baseline. Gate validation then exposed two obsolete unknown-rule references in SplashOverlay's inline jsx-a11y directive; removing that ineffective directive leaves 1,379 unrelated errors. This is a three-rule gate, not a claim that full shared ESLint is clean.

## Changes and behavior

| Area | Resolution |
| --- | --- |
| Theme advanced editor / health summary | Hoist ResetButton and Dot to module scope; component identity stays stable through parent renders. |
| Knowledge QA source card | Select directly between imported MessageSquare and FileText components. |
| ACP permission modal | Use the lazy Date.now state initializer; retain the existing one-second update loop. |
| Moderation undo | Maintain clock state and schedule a timeout for the undo deadline. Undo expires while the bar remains open, preserves the original strict-before comparison, reschedules when the deadline changes, and clears timers on unmount. Long deadlines are capped to the platform timer limit and rescheduled. |
| Workflow execution | Advance elapsed wall time once per second during an unfinished run; freeze at completedAt and clean up the timer. Paused time continues to count as before. |
| Quiz results | Refresh the selected relative date window once per minute while active, so old attempts leave the window without another query/filter change. No interval runs for the all-dates filter. |
| Legacy prompt timestamps | Missing createdAt uses the existing unknown-date sentinel 0 instead of inventing a new creation time during render. |
| Speech download | Use the render artifact ID as the filename suffix. A download remains associated with that artifact across rerenders instead of acquiring a new render-time timestamp. |

Three single-line dispositions retain existing behavior and each has a runtime characterization test:

- ProviderIcon: the registry returns existing module-level components. The real provider SVG DOM node retains identity on rerender. Direct registry indexing still triggers the compiler diagnostic, so moving the lookup does not remove the false positive.
- TableBlock: Date.now runs in the CSV download click handler. The test verifies that rendering performs no download and clicking later uses the click-time timestamp.
- Writing session save: Date.now runs in TanStack Query's mutation success callback. A real mutation with a deferred service response verifies that lastSavedAt remains unset during rendering and the pending save, then records completion time.

These are local documented exceptions, not file-wide or rule-wide suppressions.

## Required gate

`bun scripts/check-shared-hooks.mjs` runs from `apps/tldw-frontend` in the existing `frontend-required` job when `tldw_frontend_changed` is true, which includes shared-UI changes. The existing `bun run lint` step is preserved. The new checker enumerates every JS/JSX/TS/TSX/MJS/CJS file in `apps/packages/ui/src` and `apps/tldw-frontend/pages`, using the same ESLint configuration. It fails on the three hook rules, parser errors, unknown rule definitions, configuration failures, ignored source files, disabled/downgraded required rules, or an empty required scope. It reports unrelated error counts explicitly. It does not add dependencies, exclude shared files, or alter unrelated rule severities.

## Validation

- Clock regressions reproduced before implementation: 3 failing / 10 passing; after fixes 13 passing.
- Legacy prompt timestamp and stable artifact filename regressions reproduced before implementation: 2 failing / 2 passing; after fixes 4 passing.
- Save-completion clock characterization: 1 passing.
- Combined runtime suite: **52 tests passing in 10 files**, including existing theme, source-card, ACP permission, and speech suites. Existing i18next/CSS/antd warnings remain.
- Checker behavioral suite: **12 tests passing** with the real ESLint configuration. Fixtures cover all three compiler diagnostics in both roots, parser failure, invalid configuration, unknown inline rule definitions, ignored files, disabled file-specific rules, empty scopes, and explicit unrelated-error/nonfatal-warning handling.
- Required workflow contract suite: **10 passing**; Actionlint validates both frontend-required and container-build-check. The new checker and its tests pass ESLint; the Python contract test passes Ruff and Black.
- Existing full frontend lint: **793 files, zero errors, 169 warnings**, exit 0. This preserves the existing full lint gate alongside the new shared-source coverage.
- WebUI TypeScript check: exit 0 with no diagnostics (`--noEmit --pretty false --incremental false`). This is the existing WebUI typecheck, not a claim of full shared strict-mode coverage.
- Full follow-up ESLint audit: zero unsuppressed purity/static-components/use-memo findings across 5,158 files; 1,381 unrelated errors retained.
- Final CI command `bun scripts/check-shared-hooks.mjs`: **exit 0, 5,158 files, zero failures, 1,379 unrelated errors** after the obsolete unknown-rule directive cleanup. Existing Next pages-directory and Babel large-generated-file notices remain nonfatal.
- Bandit over changed frontend component scope: no Python input, no findings. Bandit does not analyze TypeScript. The Python workflow contract test reports only 47 B101 low-severity findings for ordinary pytest assertions; no runtime-security findings.

Local detailed evidence (not committed): `/tmp/pr2761-hooks-runtime-full.log`, `/tmp/pr2761-hooks-typecheck-final.log`, `/tmp/pr2761-hooks-checker-green.log`, `/tmp/pr2761-hooks-shared-gate-final.log`, `/tmp/pr2761-hooks-ci-contract-green.log`, `/tmp/pr2761-hooks-full-final.json`, and `/tmp/pr2761-hooks-full-summary.json`. The full follow-up inventory JSON is compact diagnostic-only output, without source text.

SHA-256 of the full follow-up diagnostic inventory: `1bb22b797291082286c2c27d792b259da46deece15fd70548217ef4bc85f0f73`.
