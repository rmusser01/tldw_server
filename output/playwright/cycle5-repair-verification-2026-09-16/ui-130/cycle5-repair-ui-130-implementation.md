# UAT130 / TASK13260.70

Frozen 2026-09-16T09:49:10.374Z.

## Root cause and correction

AnalysisModal dropped catalog provider metadata and exact-matched a qualified preference to bare model IDs, silently selecting the first row. Catalog entries now retain their raw dispatch model and recognized provider-derived key. Equivalent provider/model selection resolves once; ambiguous, foreign, providerless and contradictory qualified metadata remain unselected. Known-provider parser reused without registry edits. Removed unqualified-model fallback, loading behavior and explicit shared-owner writes preserved.

## Evidence

- Permanent RED7 failed/2 passed: /private/tmp/cycle5-repair-ui-130-red.log (real selected-model owner/storage, actual parser/resolver, delayed catalog, controlled completion).
- Additional contradictory descriptor RED1/control1: -conflict-red.log.
- GREEN74/5 suites: -green.log, AnalysisModal model-owner/stage3/cancel + resolve-api-provider + TldwModels. Qualified and bare shapes, known alias, exact filesystem path, unknown colon prefix, ambiguous two-provider choice recovery, newer explicit choice and unavailable unqualified fallback covered.
- Command from apps/packages/ui: bun run test src/components/Media/__tests__/AnalysisModal.model-owner.test.tsx src/components/Media/__tests__/AnalysisModal.stage3.regression.test.tsx src/components/Media/__tests__/AnalysisModal.stage1.cancel.test.tsx src/utils/__tests__/resolve-api-provider.test.ts src/services/tldw/__tests__/TldwModels.test.ts --maxWorkers=1 --no-file-parallelism
- Scoped lint: exact HEAD baseline1 error (existing require-yield in stage1.cancel) and18 warnings, zero added. -lint-comparison.json plus before/after JSON retain proof. The adjacent test mocks now preserve the actual parser export; no behavior/assertions removed. git diff --check passed.
- No whole typecheck; parent integrated baseline90 comparison pending. Bandit not applicable to TypeScript-only.

## Limits

No live catalog/backend/browser/inference exercised during implementation; transport controlled in mounted tests. Native first-open configured identity and resulting request require parent acceptance. No auth/catalog scope expansion, runtime changes, staging or commits.

## Exact files

- apps/packages/ui/src/components/Media/AnalysisModal.tsx
- apps/packages/ui/src/components/Media/__tests__/AnalysisModal.model-owner.test.tsx
- apps/packages/ui/src/components/Media/__tests__/AnalysisModal.stage1.cancel.test.tsx
- apps/packages/ui/src/components/Media/__tests__/AnalysisModal.stage3.regression.test.tsx
- backlog/tasks/task-13260.70 - Preserve-configured-model-identity-when-Media-analysis-loads-its-catalog.md

Byte hashes: /private/tmp/cycle5-repair-ui-130-manifest.json.
