# Independent source review: UAT271 / TASK13260.212

**Verdict: CLEAR for the frozen source change.** No blocking findings. Original Character4 native retry remains pending.

## Causal boundary and implementation

The retained native Character4 request omitted `expected_version` and returned422 with validation location `query.expected_version`. The editor already had version state and an update mutation that forwarded it, but opening a record never populated that state.

The fix passes the existing `setEditVersion` setter into `useCharacterCrud` and captures the loaded record's non-negative integer version when opening the editor. Missing or malformed versions become null. Every edit overwrites the previous value, and the existing close paths clear it. The update transport preserves zero and omits only nullish values.

This preserves optimistic concurrency: Save uses the version the user loaded. It does not fetch a newer version at save time or bypass the unchanged backend409 check. No new auth, ownership, credential, network or database behavior is introduced. The implementation is appropriately local; concurrent World Books and character-list route UAT272 edits are excluded from this review.

## Independent evidence

| Check | Result |
| --- | --- |
| Maintained Manager/form regression, normal source | 1 passed; 98 other tests deliberately deselected |
| Same regression with baseline hook loaded by a review-only Vite overlay | Exact causal failure: expected version3, received undefined; 1 failed, 98 deselected |
| Normal source after overlay | 1 passed; 98 deselected |
| Maintained real-client update/delete transport file | 4 passed |
| ESLint applied from repository root to all four files | 0 errors, 236 warnings; identical per-file diagnostic multisets to baseline |
| Scoped diff whitespace check | Passed |
| Local evidence audit | 18 checks passed, 37 hashed inputs |

The Manager test exercises the real component, hook, state and shared form; query/transport dependencies are mocked. It verifies the edited payload, captured version3 and absence of `getCharacter` calls. The real client regression independently verifies `?expected_version=0` and omission when no version was supplied. Together they test the actual repaired state-to-request boundary.

The author's original RED was described in `verification.json`, but no retained original RED log was present when reviewed. It is not counted as retained causal evidence. The independent overlay supplies that proof without modifying production files: only `useCharacterCrud.tsx` is supplied from baseline commit `8764bf5b6c686e362d21a52d4cc6fc0627241827` during the selected test. Overlay, baseline source, test logs and hashes are retained under the review directory.

## Exact frozen scope

| File | SHA-256 |
| --- | --- |
| `Characters/Manager.tsx` | `7299c9a070126e392d7c4eb3d4cf5ea3c325008ed3434e3be7713e90bc5e1fc9` |
| `Characters/hooks/useCharacterCrud.tsx` | `0373edec234c4a758ff87e0efa515f915ed4c72ea1f53c080f7335a5d838470c` |
| `Characters/__tests__/Manager.first-use.test.tsx` | `51dd43d68f386b21a19c88eb50bbc4d78b2b11bbb4685f9b8a2ce4d3721a3b5c` |
| `services/__tests__/tldw-api-client.characters-delete.test.ts` | `31eacb13865faf0fd6496bdbb954aa30a923ccb0df09aa666f95db4290fa9030` |

All paths are under `apps/packages/ui/src/components/Option/` except the final `services/` path under `apps/packages/ui/src/`. All four hashes still match the author's frozen inventory.

## Limits

- No native browser retry or real backend concurrency test was run by this review. The stale409 contract was inspected in source; native original-card Save/association readback still belongs to the acceptance gate.
- Missing/malformed version handling was inspected directly. The maintained tests cover the native positive path and explicit-zero/omitted transport behavior, not an exhaustive malformed-input matrix.
- Author compiler logs contain90 frontend and380 direct-UI diagnostics, with no scoped-file diagnostic. These are not a clean build or an independently established whole-project baseline comparison. The expensive compiler runs were not repeated.
- The author's full Manager file run was interrupted and is excluded. This report does not claim a full suite pass.
- Bandit records two TypeScript parse errors. Zero findings is not TypeScript security assurance. The author ESLint command ignored these paths; the independent corrected-root run above actually analyzed them. Next's missing-root-pages informational warning remains. Whole-file Prettier warnings are not represented as clean formatting.
- Two initial review command-path mistakes prevented those attempts from starting the intended tests/overlay; corrected commands and final complete logs are the evidence. No product, tests, browser, runtime, model, DB, Git or Backlog changes were made.

Evidence: `.tmp/uat-repairs-231-246/character271-review/audit.json`, `audit.mjs`, `reviewed.diff`, `causal-red.config.ts`, `baseline-useCharacterCrud.tsx`, and retained focused logs/lint comparison.
