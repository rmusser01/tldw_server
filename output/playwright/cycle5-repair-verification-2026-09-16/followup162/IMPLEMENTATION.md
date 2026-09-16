# UAT162 — named Advanced Timeouts controls

Task: TASK-13260.99. Base HEAD: 413be51cc87f81971729a070202caddf89b4b788. Scope frozen for independent review.

## Result and cause
Each of the eight timeout inputs now has a semantic association to its existing translated visible label. One React.useId prefix and eight descriptive suffixes keep input identities unique between component instances and stable across rerenders. Existing timeout handlers, values, presets and persistence are unchanged.

Original native evidence: .tmp/uat152-153-final-native-20260916/balanced-fields-and-labels.txt shows all eight IDs empty, labels arrays empty, and no aria-label. Native original152/153 behavior had already passed; this repair preserves that behavior.

## Owned files
- apps/packages/ui/src/components/Option/Settings/TldwTimeoutSettings.tsx
- apps/packages/ui/src/components/Option/Settings/__tests__/TldwTimeoutSettings.accessibility.test.tsx (new)

No other production/test edits. No runtime/browser, staging or commit actions. Root owns tracker, native acceptance and integration.

## Permanent RED → GREEN
The new suite mounts the actual component with real Ant Design Collapse/Input controls and actual English i18next resources. Every timeout field is queried by its visible translated accessible name and by associated label; clicking that label must focus the actual input. Two simultaneously mounted instances check sixteen unique IDs, correct second-instance focus targets, and stable IDs after a controlled-value rerender.

Before production editing: red.log reports nine failing tests — eight missing spinbutton accessible names and one empty-ID assertion. The initial fixture run used an overly exact Collapse button name and failed before reaching the fields; test-setup-first-run.log retains that non-regression failure. Correcting the query to accept Ant Design's collapsed icon name produced the meaningful nine-failure RED without changing production.

After the minimal production edit: green.log reports 18/18 tests passing in 2/2 suites: nine new accessibility tests plus nine unchanged full Settings timeout integration tests, including Custom→Balanced save/reload, Extended/reset, explicit override preservation, default budgets and actual delayed request timeout behavior.

## Exact checks
From apps/tldw-frontend:

    bunx vitest run ../packages/ui/src/components/Option/Settings/__tests__/TldwTimeoutSettings.accessibility.test.tsx
    bunx vitest run ../packages/ui/src/components/Option/Settings/__tests__/TldwTimeoutSettings.accessibility.test.tsx ../packages/ui/src/components/Option/Settings/__tests__/tldw.timeouts.form.test.tsx
    bun run typecheck

From repo root:

    apps/tldw-frontend/node_modules/.bin/eslint -c apps/tldw-frontend/eslint.config.mjs apps/packages/ui/src/components/Option/Settings/TldwTimeoutSettings.tsx apps/packages/ui/src/components/Option/Settings/__tests__/TldwTimeoutSettings.accessibility.test.tsx
    git diff --check -- apps/packages/ui/src/components/Option/Settings/TldwTimeoutSettings.tsx apps/packages/ui/src/components/Option/Settings/__tests__/TldwTimeoutSettings.accessibility.test.tsx
    source .venv/bin/activate
    python -m bandit apps/packages/ui/src/components/Option/Settings/TldwTimeoutSettings.tsx -f json -o .tmp/uat162-timeout-labels-20260916/bandit.json

ESLint and diff exit0. ESLint has zero rule errors/warnings; it prints the same repo-root Next pages-path notice for baseline and current. The initial lint from frontend cwd ignored out-of-base-path UI source and is retained as eslint-baseline.log; eslint-baseline-corrected.log and eslint.log are the actual scoped checks.

Full compiler exits2 with90 diagnostics, none in either owned file. Comparing filename/code/message (line positions normalized) against retained .tmp/uat159-first-run-auth-20260916/typecheck.log gives90 prior/90 current, zero added/removed. This is an existing retained baseline comparison, not a fresh HEAD replay. See typecheck-comparison.json.

The test runner emits known Ant Design/jsdom CSS parse warnings, Node experimental localStorage warning, and the existing form test's Flashcards-transfer cleanup warning; there are no failed tests or unhandled errors. Bandit cannot parse TSX (one AST error, zero findings); no TypeScript security assurance is claimed. The repair adds only native ID/label attributes.

## Remaining acceptance
Independent read-only review and root-owned native accessibility/label-focus checks remain pending. Suggested native check: reopen Advanced Timeouts at the established route, query eight spinbuttons by visible label, inspect their labels arrays, click a label and confirm focus, retain preset/value controls. Root owns task closure after acceptance.

Private design and staged progress: DESIGN_PLAN.md. Exact source/test copies: review-snapshot/. Frozen hashes: owned-manifest.json. The unchanged existing outer section anchor is outside this repair; input IDs themselves are unique across instances.
