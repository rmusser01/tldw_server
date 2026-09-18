# TASK13260.215 / UAT274 — World Book entry identifier normalization

## Outcome

The World Book entry manager now normalizes the canonical backend `id` into its established internal `entry_id` field at the single manager data boundary. A list response such as `{ id: 91, world_book_id: 1 }` therefore reaches edit, direct delete, and selected bulk delete with `91`, rather than producing an `undefined` URL segment.

The client transports were inspected and intentionally left unchanged: both Base and chat-rag `listWorldBookEntries` methods return the backend response without entry mapping. Normalizing there would duplicate a manager-specific compatibility concern.

## Causal evidence

The original manager implementation, isolated from the repository baseline source, fails the maintained UI regression when it receives the canonical API shape. The retained failure expects `updateWorldBookEntry(91, ...)` and records the original call as `updateWorldBookEntry(undefined, ...)`:

- [manager baseline RED log](../../../.tmp/uat-repairs-231-246/worldbook274/manager-baseline-causal-red.log)
- [utility baseline RED log](../../../.tmp/uat-repairs-231-246/worldbook274/baseline-causal-red.log)

The focused manager regression executes the component's actual query function, supplies an `id`-only service response, then verifies edit, direct delete, and selected bulk delete all use `91`. Utility controls preserve a present legacy `entry_id`, including `0`, and decline missing, zero, or non-numeric canonical IDs.

## Validation

- `bunx vitest run ...worldBookEntryUtils.test.ts ...WorldBookEntryManager.budget.test.tsx ...worldBookRelationshipUtils.test.ts ...worldBookBulkActionUtils.test.ts --reporter=dot` — **17 passed**, exit 0. Receipt: [focused-final.log](../../../.tmp/uat-repairs-231-246/worldbook274/focused-final.log).
- Scoped ESLint on the four changed TypeScript files — exit 0, 52 warnings and 0 errors. The retained line-level comparison records no warning on the added manager regression lines; the 52 warnings remain elsewhere in the targeted files. Receipt: [eslint-final.log](../../../.tmp/uat-repairs-231-246/worldbook274/eslint-final.log), [eslint-delta.json](../../../.tmp/uat-repairs-231-246/worldbook274/eslint-delta.json).
- Direct UI TypeScript compiler invocation with an 8 GiB heap — exit 2 with 382 repository diagnostics, with no diagnostic referencing any of the four touched paths. This does not establish a clean project typecheck. Receipt: [ui-tsc-final.log](../../../.tmp/uat-repairs-231-246/worldbook274/ui-tsc-final.log).
- `git diff --check` for the four changed files — exit 0.
- Python Bandit on the two TypeScript production paths — exit 0 with 0 findings and 2 parse errors. Bandit does not parse TypeScript, so this is not TypeScript security assurance. Receipt: [bandit-final.json](../../../.tmp/uat-repairs-231-246/worldbook274/bandit-final.json).

## Scope and limits

Changed production scope is only `WorldBookEntryManager.tsx` and `worldBookEntryUtils.ts`; changed tests cover the utility contract and the real manager query path. Relationship and bulk utility regressions remain in the focused run. The move destination list consumes entry content and keywords only; source selection and relationship calls consume the normalized manager entries. No UAT275 summary invalidation changes were started.

Exact file and evidence hashes are retained in [hashes.sha256](../../../.tmp/uat-repairs-231-246/worldbook274/hashes.sha256) and [verification.json](../../../.tmp/uat-repairs-231-246/worldbook274/verification.json).
