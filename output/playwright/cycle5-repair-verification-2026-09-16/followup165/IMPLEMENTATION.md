# UAT165 / TASK-13260.102 — long Default model identifier wrapping

## Scope and cause

Baseline HEAD: 588d6c3cef7bb29ec9a4fef9f726b65f9330fb5f.
Only production file: apps/packages/ui/src/components/Option/Models/index.tsx.
The retained native before screenshot (.tmp/uat164-native-20260916/pg-multi-ready.png) shows an unbroken path segment crossing from Default model into Configured providers. Existing hyphenated segments wrap naturally. The shared Models implementation, rather than a frontend/src copy, owns the tile.

## Change

Add Tailwind break-all to the Default model value div at line 684. This follows existing Settings URL/scope identifier rendering in ServicePromptsSettings.tsx and tldw.tsx. Full defaultModelLabel content remains in the DOM, selectable and readable; the grid, default-provider/model resolution, saved preference, readiness state, and selection behavior are unchanged. No truncation or maximum-height clipping was introduced.

Source SHA256: f736239e73323ba5756f28526d4fc0229730a00511ae9b41a4c21e25bd0457c6.

## Verification performed

- Existing Models regressions: 21 tests / 3 suites PASS. Command from apps/tldw-frontend: bunx vitest run ../packages/ui/src/components/Option/Models/__tests__. Receipt: models-tests.log.
- Scoped ESLint: zero errors, zero warnings in the touched file. Command from repository root: node apps/tldw-frontend/node_modules/eslint/bin/eslint.js --config apps/tldw-frontend/eslint.config.mjs apps/packages/ui/src/components/Option/Models/index.tsx -f json. Receipts: eslint.json and eslint-stderr.txt. The latter retains the existing root-cwd Next pages-directory configuration notice; no source lint finding.
- git diff --check for the owned file: PASS.
- Bandit attempted with the project virtual environment: source .venv/bin/activate && python -m bandit apps/packages/ui/src/components/Option/Models/index.tsx -f json -o .tmp/uat165-repair-20260916/bandit.json. Bandit reports one TSX syntax/AST parsing error and performs no meaningful TypeScript security analysis. This is a coverage limitation, not a clean security-scan claim. Receipts: bandit.json and bandit.log.
- No implementation-mirroring CSS class test was added, per the approved bounded scope. No full compiler rerun for this class-string-only change.

## Acceptance handoff

Root owns native desktop and narrow viewport geometry/screenshots and independent source review. Those acceptance checks were not performed by this author and remain separate from the regression results above. No browser, runtime, model-provider call, tracker, plan, git staging, or commit action was performed. The production source is frozen for review; exact source and diff are retained in index.review.tsx and owned.patch.
