# UAT151 / TASK13260.90 implementation

## Status

Author corrected candidate frozen 2026-09-16T18:13:44.885Z for independent rereview. Final author run: **200 tests / 14 suites pass**. Native acceptance and parent combined compiler remain pending. No browser/runtime/inference/staging/commit actions were performed.

Independent review reproduced a removed newly created deck remaining selected. Permanent controls for both panels reproduced 2 RED; the temporary proof now expires at a newer successful React Query list revision, captured at create acknowledgment. Original unchanged private probe passes 1/1 and full 200/14 regression passes. Rereview remains pending. Initial manifests are preserved as `cycle5-uat151-initial-{code,production}-freeze.json`.

## Root cause and repair

Both creation panels reused an unscoped React Query deck cache. Their initial-selection effects only handled null values, so an Alice ID/label survived a Bob empty-list response. Numeric save targets were accepted without list membership. Generation writes already captured authority, while image occlusion uploaded/saved/retained Undo under mutable current credentials.

The two panels now use the existing Manager snapshot for opt-in deck reads. Their cache uses the separate `flashcards:decks:scoped` prefix, excluding it from the existing legacy updater's exact-prefix cache writes. Query cancellation combines its own signal with the captured authority signal and rejects late results. Unresolved/aborted panel scope exposes no cached list. Omitted scope preserves legacy callers.

Both panels reconcile selections only after a successful current list and validate numeric save targets. A captured new-deck acknowledgment permits immediate save before refetch observes it. Occlusion forwards scope to upload, create, bulk and versioned Undo read/delete, and checks captured owner/mount before and after awaited work. Already acknowledged work is not replayed or compensated after invalidation.

The exact added policy routes are GET `/api/v1/flashcards/decks`, POST `/api/v1/flashcards/assets`, POST `/api/v1/flashcards/bulk`, and GET/DELETE `/api/v1/flashcards/{canonical-uuid}`. Existing POST generation/deck/card routes remain unchanged. Default callers, payload arrays, workspace filters, expected-version queries and asset bytes are preserved.

## Owned production files (7)

- `apps/packages/ui/src/components/Flashcards/hooks/useFlashcardQueries.ts`
- `apps/packages/ui/src/components/Flashcards/tabs/ImportExportTab.tsx`
- `apps/packages/ui/src/components/Flashcards/tabs/ImportExport/GeneratePanel.tsx`
- `apps/packages/ui/src/components/Flashcards/tabs/ImageOcclusionTransferPanel.tsx`
- `apps/packages/ui/src/services/flashcards.ts`
- `apps/packages/ui/src/services/flashcard-assets.ts`
- `apps/packages/ui/src/services/tldw/service-prompt-scope-error.ts`

Owned tests (6): actual Manager deck-authority integration (new); actual Manager private-handoff integration (only mock adds successful-query status); existing ImageOcclusionTransferPanel; ImportExportTab.deck-creation; services flashcards.private-scope; services flashcard-assets. Full paths and SHA-256 values are in `cycle5-uat151-code-freeze.json`, including TASK13260.90. The production-only subset is `cycle5-uat151-production-freeze.json`. Baseline source hashes use commit `12b683dc49e853ee1731134c11ba36c0ecbe87c3`.

## RED and GREEN evidence

| Evidence | Result and boundary |
| --- | --- |
| `cycle5-uat151-valid-red.log` | 2 actual mounted failures: Alice selector after Bob empty list, and same-owner removed deck. Real QueryClient, real Manager, actual AntD selectors; external authority/config/transport and unrelated children mocked. |
| `cycle5-uat151-transport-red.log` | 18 failed / 23 passed before scope forwarding/policy/cancellation correction; two suites. |
| `cycle5-uat151-occlusion-red.log` | 7 failed / 5 passed before captured-owner continuation correction. |
| `cycle5-uat151-save-controls.log` | 29 / 3 pass: expanded real Manager and panel boundary controls. Overlaps final run. |
| `cycle5-uat151-final-green.log` | Initial 198 / 14 pass; overlaps corrected run. |
| `cycle5-uat151-created-proof-red.log` | Permanent 2 RED / 24 unselected after independent review: newly created deck removed from later catalogue, actual Manager and actual Occlusion. |
| `cycle5-uat151-created-original-green.log` | Original unchanged independent probe 1 / 1 pass (9 other tests unselected). |
| `cycle5-uat151-corrected-final-green.log` | 200 / 14 pass after proof expiry; overlaps final typed run. |
| `cycle5-uat151-final-typed-green.log` | **200 / 14 pass**, authoritative final author regression run after truthful masked-data typing, exit 0. |

The root's first selector attempt and author early selector attempts were harness failures, not valid product RED. AntD 6 forwards the test ID and retains leaving dropdowns; its test mode also reuses listbox IDs. The final selection helper limits the actual visible non-leaving popup before clicking its option.

Intermediate candidate runs are retained (`first-green`, `controls-green`, `final-focused`) and are not final acceptance. One author null-check regression in legacy omitted-scope rendering was corrected before final verification. The unresolved-authority control initially counted Manager's separate legacy initial-summary query; it now checks the two scoped panel reads and actual masked panel labels. The actual private-handoff suite's old mock omitted `isSuccess`; adding that truthful status retains its source/provenance and delayed-save assertions. The new abort transport mock was corrected to honor the Fetch signal, matching real Fetch; no transport production expansion was made for it.

Final coverage includes account/server change, colliding IDs, delayed scoped list across A→B→A, legacy cache-writer isolation, benign same-owner event/draft preservation, successful empty/removal, loading/error refusal, new-deck immediate save, canvas/upload/create/bulk cancellation, unmount, retained Undo plus cancellation between read/delete, exact payload/version/query/scope fields, unscoped defaults and canonical route-negative controls. Existing source handoff, generation, query-reference and import decomposition controls remain green.

## Reproduce

Working directory: `/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui`.

Exact installed-local command: `/private/tmp/cycle5-uat151-test-command.txt`. Its 14-path inventory is `/private/tmp/cycle5-uat151-test-paths.json`. It runs `./node_modules/.bin/vitest run` with six touched suites and eight relevant adjacent suites, without mocks/servers for native acceptance.

Root-scoped ESLint uses `apps/tldw-frontend/node_modules/.bin/eslint --config apps/tldw-frontend/eslint.config.mjs` on all 13 code/test paths. Evidence: `cycle5-uat151-eslint-final.json`, `-baseline.json`, `-comparison.json`, and reproducible comparison script `cycle5-uat151-lint-compare.mjs`.

- Current: **0 errors, 18 warnings** across 13 code/test paths.
- Baseline: 0 errors, 26 warnings across 12 tracked predecessor paths (new test has no predecessor).
- **0 added diagnostics, 8 removed**, normalizing line-number-only hook messages and retaining multiplicity.
- Scoped `git diff --check` passes.
- Bandit: not applicable; only TypeScript/TSX changed, no Python production/test paths.
- Whole TypeScript: parent initially observed 90→102 diagnostics (12 new inferred-unknown errors in Manage/Scheduler). Explicit `useQuery<Deck[]>` alone did not fix it. A fresh TypeScript checker showed masked `data: undefined` widened to any under frontend `strict: false`. The hook now explicitly types `data: Deck[] | undefined` and returns one merged shape; a fresh checker reports Deck[]. Before/after evidence and script: `cycle5-uat151-query-types*`. Consumers are unchanged. Parent-owned final compiler comparison remains pending. Initial broad UI run was stopped; it is not a pass. Web332/17 passed independently of this change, per parent.

## Limits and pending acceptance

- Independent review and original native Alice→Bob route/account switch remain pending. Unit/integration checks use real UI owners and transport code with external services mocked; they do not constitute browser/native permission verification.
- Other Flashcards tabs and Manager's initial-deck summary still use their existing legacy queries. This unit isolates the two reported creation panels and their actual mutations; it does not certify all Flashcards cache ownership.
- Existing JSDOM/AntD CSS parsing, browser-storage advisory and root pages-directory lint advisory are retained in logs; they are not hidden or counted as product/native failures.
- Task AC1/AC2 checked from the reproducible regression and focused controls. AC3 remains open for independent/static/native integration.
