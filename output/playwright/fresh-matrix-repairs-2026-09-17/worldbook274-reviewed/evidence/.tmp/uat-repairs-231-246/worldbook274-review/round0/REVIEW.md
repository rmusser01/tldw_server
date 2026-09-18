# UAT274 / TASK13260.215 — independent source review

**NEEDS CHANGE: one narrow compatibility-contract finding.** The canonical numeric-ID repair is minimal and fixes the observed manager boundary; independent focused tests pass17/17. No product/source files were changed by this reviewer.

## Finding

**[P2] Reject malformed canonical ID types before coercion** — `apps/packages/ui/src/components/Option/WorldBooks/worldBookEntryUtils.ts:20`.

The brief requires that missing/invalid canonical values not invent a usable ID, but `Number(record.id)` accepts unrelated JSON types. Running the frozen production helper directly gives `{id:true} → {id:true,entry_id:1}` and `{id:[91]} → {id:[91],entry_id:91}`. The manager uses this new ID for edit, direct delete and bulk operations. Add a primitive type guard before conversion and maintained controls for boolean/array/object/missing values; continue preserving present legacy `entry_id` values, including0. If numeric strings are intentionally compatible, limit coercion to numbers/strings and retain an explicit numeric-string control.

This is a bounded malformed-data/spec issue. The current backend response schema declares numeric `id:int` and the actual native response had `id:1`; no native wrong-record operation, ownership bypass or security failure is demonstrated. The existing native274 failure and the UAT275 count issue remain separate from this review finding.

## What is correct

- Normalization occurs once in the manager's memoized list boundary, for object-wrapped and legacy array query data. The raw Base/domain transports remain unchanged.
- Present legacy `entry_id` wins, including0. Missing/null/zero/negative/fractional/non-numeric IDs do not derive a new field in the independent probe. Numeric canonical91 becomes internal91 without mutating the input.
- Edit, direct delete, row keys, selection, bulk operations and relationship lookup consume normalized manager entries. Move-source IDs use normalized entries; move-destination deduplication uses content/keywords only.
- The new maintained component regression invokes the actual manager `queryFn`, provides an id-only response, then drives normal edit, direct delete and selected bulk-delete callbacks with91. React Query/mutations/service are mocked, so this is a meaningful component regression, not an actual HTTP/DB acceptance test. Direct deletion does not remove the mocked row; subsequent bulk action proves ID propagation, not persistence sequencing.
- The retained baseline manager is byte-identical to repository HEAD. Its test differs from the final maintained test only in annotations; TypeScript transpilation yields identical JavaScript. The causal RED reports update called withundefined instead of91. The utility RED only proves the new function was absent and is weaker causal evidence.

## Independent verification

- Canonical UI config, existing installed Vitest, four focused files: **17 passed**, four files passed, no skips (`focused-canonical.log`).
- Correct-root scoped ESLint actually inspected all four files: **0 errors,52 warnings**, no ignored files. Warnings match the author's pre-existing warning count. Author retained line-level warning comparison.
- `identifier-probe.mjs/json` loads and runs the frozen utility after TypeScript erasure; it retains the exact malformed-type evidence and valid/legacy/missing/invalid controls.
- Reviewed author direct UI compiler receipt:382 diagnostics, none in the four touched paths. This is not a clean project typecheck and was not rerun broadly.
- Reviewed Bandit receipt:0 findings,2 TypeScript parse errors; no TypeScript security assurance.
- Initial reviewer Vitest command combined `--root` with a root-prefixed config, so config resolution failed before tests. That failed receipt is preserved as `focused.log`; corrected canonical working-directory run is separate.

The production and test hashes remain the author's frozen `hashes.sha256` values. No browser/runtime/model/DB/Git/Backlog mutation, response interception, dependency installation or new native claim. A native repeat of the preserved original book3/entry1 is still needed after source review/commit/upgrade.
