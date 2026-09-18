# UAT272 static receipts

## Whitespace

```sh
git diff --check -- apps/packages/ui/src/services/tldw/TldwApiClient.ts apps/packages/ui/src/services/tldw/domains/characters.ts apps/packages/ui/src/services/tldw/domains/chat-rag.ts apps/packages/ui/src/services/tldw/__tests__/TldwApiClient.request-scope.test.ts apps/packages/ui/src/components/Option/WorldBooks/Manager.tsx apps/packages/ui/src/components/Option/WorldBooks/WorldBookDetailPanel.tsx apps/packages/ui/src/components/Option/WorldBooks/__tests__/WorldBooksManager.attachmentStage1.test.tsx apps/packages/ui/src/components/Option/WorldBooks/__tests__/WorldBookDetailPanel.test.tsx .superpowers/sdd/IMPLEMENTATION_PLAN_fresh_matrix_231_246_repairs/task-272-brief.md .superpowers/sdd/IMPLEMENTATION_PLAN_fresh_matrix_231_246_repairs/task-272-report.md
```

Exit: `0`.

## TypeScript compiler

```sh
cd apps/packages/ui && bunx tsc --noEmit -p tsconfig.json
```

Exit: `134`. The process exhausted its Node heap before diagnostics. This broad compiler check is not reported as passing; no retry was made because it is not a source-specific diagnostic.

## Frontend typecheck

```sh
cd apps/tldw-frontend && bun run typecheck
```

Exit: `2`. The established repository baseline is 90 TypeScript diagnostics. This receipt contains 90 diagnostics and none references an owned source or test path.

## ESLint

```sh
node apps/tldw-frontend/node_modules/eslint/bin/eslint.js --config apps/tldw-frontend/eslint.config.mjs <seven owned repo-relative paths> --format json
```

Exit: `1`. ESLint assessed all seven paths with no ignored-file messages and found 1 error plus 728 warnings. The error is the pre-existing `@typescript-eslint/no-non-null-asserted-optional-chain` result for `Manager.tsx`'s `openEntries?.id!`. Running the same configuration with the HEAD version of that file finds the same error at the same expression (its line moved from 1986 to 2025), with 104 baseline warnings and 1 baseline error. Therefore this receipt is not clean, but no new error is attributed to this diff. The Next pages-directory informational warning is retained in `frontend-eslint-stderr.txt`.

## Bandit

```sh
source .venv/bin/activate && python -m bandit -r apps/packages/ui/src/components/Option/WorldBooks apps/packages/ui/src/services/tldw -f json -o .tmp/uat-repairs-231-246/worldbook272/bandit-ui.json
```

Exit: `0`. The JSON result has 0 findings and 0 errors, but reports 0 scanned TypeScript lines. This is a required scanner receipt only; Bandit provides no language-level security assurance for this TypeScript-only scope.

## Disabled-link follow-up static receipts

### Frontend typecheck

```sh
cd apps/tldw-frontend && bun run typecheck
```

Exit: `2`. The receipt has the established 90 TypeScript diagnostics and none names an owned World Books, client, chat-RAG domain, or focused-test path.

### ESLint

```sh
node apps/tldw-frontend/node_modules/eslint/bin/eslint.js --config apps/tldw-frontend/eslint.config.mjs <eight owned repo-relative paths> --format json
```

Exit: `1`. ESLint assessed eight files with no ignored-file messages, reporting 1 pre-existing Manager error (`openEntries?.id!`) and 963 warnings. Root retains the baseline comparison; this receipt is not reported as clean.

### Bandit

```sh
source .venv/bin/activate && python -m bandit -r apps/packages/ui/src/components/Option/WorldBooks apps/packages/ui/src/services/tldw/TldwApiClient.ts apps/packages/ui/src/services/tldw/domains/chat-rag.ts -f json -o .tmp/uat-repairs-231-246/worldbook272/bandit-ui-review2.json
```

Exit: `0`, with 0 findings and 2 parser errors: Bandit cannot parse `TldwApiClient.ts` or `domains/chat-rag.ts` as Python. This is a TypeScript-scanner limitation and provides no clean security assurance.
