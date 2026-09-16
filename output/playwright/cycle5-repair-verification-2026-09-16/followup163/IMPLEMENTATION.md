# UAT163 — persist explicit source retrieval activation

Task TASK-13260.100. Base HEAD 223591ac4fb2520290634c74310351ce7d8b18ae. Author source frozen for independent review.

## Cause and repair
Home's accepted rag_media handoff sets chatMode=rag, selected media IDs and fileRetrievalEnabled=true. The session snapshot and persisted store omitted the last field. Cold option state defaults false, so the real shouldUseRagForTurn gate sent an ordinary completion even when the same conversation and media IDs restored.

The existing session store now carries an explicit boolean, default false, through its existing partialize/save/rehydrate path. The persistence hook subscribes to and snapshots that flag; restore applies only a literal true under the existing sourceSelectionRevision guard. Explicit false and absent legacy values remain false. No consent is inferred from mode or media IDs. Existing owner validation, restoreRevision cancellation, selected-chat reset, and assistant persistence are unchanged.

A second real callback RED proved a newer explicit disable could lose to delayed restore. The Form's actual onFileRetrievalChange callback now marks accepted source selection intent before invoking the existing setter. Raw store setters and session replay do not mark intent. This follows the Form's existing handoff guard and protects a false→true→false sequence, where comparing values would miss the newer choice.

## Owned scope
Production three files:
- apps/packages/ui/src/store/playground-session.tsx: type/default/partialize field.
- apps/packages/ui/src/hooks/usePlaygroundSessionPersistence.tsx: subscription/snapshot/dependencies and guarded replay.
- apps/packages/ui/src/components/Option/Playground/PlaygroundForm.tsx: intent mark on explicit retrieval toggle.

Tests two existing files:
- apps/packages/ui/src/components/Option/Playground/__tests__/PlaygroundForm.openui-mode.test.tsx: real cold persistence/rehydrate→Send controls and actual Form callback race.
- apps/packages/ui/src/hooks/__tests__/usePlaygroundSessionPersistence.test.tsx: deterministic flag reset and strengthened selected-chat cancellation control.

Design rationale recorded before edits in DESIGN.md and official Backlog notes. Read existing Home handoff intent, session restore guards, server-chat selection cancellation, and new-chat reset paths. No shared plan/tracker, dependency, browser/runtime, staging or commit edits.

## Meaningful RED and GREEN
- red-cold.log: 2 failed/4 passed/29 filtered tests. Both true activation cases (fresh selection and prior controlled retrieval failure) cold-rehydrated the actual serialized session, then the real Send failed to issue RAG. Explicit false, missing legacy flag, different account and different server controls passed.
- green-cold.log after store+hook repair: all six selected cases pass.
- red-toggle.log before Form guard: 2 failed/35 filtered. Toggling on then off through the real Form callback before or during delayed restore still issued scoped RAG, contrary to the newer disable.
- green.log after the narrow callback guard: 57/57 tests pass, 4/4 suites, no filtered/skipped tests. Existing handoff ordering, same-value intent, account changes, later requested restore, queue restore, assistant restore, and explicit selected-chat cancellation controls also pass.

The source-flow fixture preserves real Form, submit/action router, persistence hook and Zustand localStorage serialization/rehydrate, RAG pipeline and request serializer. It mocks DB/network/model transport and unrelated UI. The KnowledgeSection fixture invokes the actual Form callback through a checkbox; this is a Form user-action boundary test, not a claim that the complete KnowledgePanel or useMessageOption wrapper was mounted. The resulting request asserts include_media_ids [42], downstream model context contains the returned Rowan facts, and the assistant source receipt contains those facts. No real inference accuracy claim is made.

## Reproducible commands
From apps/tldw-frontend:

    bunx vitest run --config ../../.tmp/uat163-repair-20260916/vitest.config.ts ../packages/ui/src/components/Option/Playground/__tests__/PlaygroundForm.openui-mode.test.tsx -t cold
    bunx vitest run --config ../../.tmp/uat163-repair-20260916/vitest.config.ts ../packages/ui/src/components/Option/Playground/__tests__/PlaygroundForm.openui-mode.test.tsx -t 'newer explicit retrieval toggle'
    bunx vitest run --config ../../.tmp/uat163-repair-20260916/vitest.config.ts ../packages/ui/src/components/Option/Playground/__tests__/PlaygroundForm.openui-mode.test.tsx ../packages/ui/src/hooks/__tests__/usePlaygroundSessionPersistence.test.tsx ../packages/ui/src/store/__tests__/playground-session-store.test.ts ../packages/ui/src/hooks/__tests__/playground-session-restore.test.ts
    bun run typecheck

From repo root, ESLint uses the explicit frontend config on the five owned paths; git diff --check also uses those paths. Full path list: owned-paths.json.

## Static results and environment limits
- ESLint exit0, 0 errors/91 existing warnings. Baseline HEAD versus current rule/message/severity arrays match exactly for all five files (eslint-comparison.json). Repo-root Next pages-path notice is unchanged.
- Diff check exit0.
- Full tsc exits2 with90 diagnostics. None concern the added persistence field or tests. Prior retained UAT162 baseline versus current filename/code/message comparison (line positions normalized): 90/90, zero added/removed. This is not a fresh baseline replay; see typecheck-comparison.json.
- Bandit invoked from activated .venv on all three production TSX files: no findings, three unsupported TSX AST parse errors. No TypeScript security assurance claimed.
- First collection attempts failed before tests because the optional pa-tesseract.js package was unresolved from shared UI. The package is already installed in Bun cache; the retained private Vitest config merges the real frontend config and adds only an alias to that existing package, matching prior UAT103/157 verification. No installed module/package/lockfile mutation and no final OCR mock. test-collection-ocr.log retains the initial non-regression failure.

## Remaining acceptance
Independent review and root-owned real browser cold reload/scoped retrieval acceptance remain pending. Native original artifacts are in .tmp/uat013-103-156-final-native-20260916; the original native source013 wrong-answer-with-facts and103 diagnostic history semantics remain separate. This repair does not change backend RAG answers, request wire contracts or canonical message promotion.

Frozen source copies and exact SHA256 manifest are in review-snapshot/ and owned-manifest.json; complete owned diff in owned.patch.
