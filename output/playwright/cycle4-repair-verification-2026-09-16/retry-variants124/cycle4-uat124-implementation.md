# UAT124 / TASK-13260.64 — failed Retry variant restoration

## Outcome and scope

Frozen for independent review; native verification remains root-owned and pending. One production argument changed in `apps/packages/ui/src/hooks/chat-modes/chatModePipeline.ts`: the general catch forwards `resolvedAssistantParentMessageId`, matching live message construction and the existing success/stream-interruption saves.

Base: `53ef4bf4d53add604bfff76f532a6d6c5fb2e027`. Code/test hashes and freeze timestamp: `/private/tmp/cycle4-uat124-owned-manifest.json`. No backend, schema, storage migration, dependency, runtime, browser or inference change.

## Root cause and evidence

The live normal pipeline groups regenerations as variants, and resolves their parent to the existing user. Its general error catch alone saved the raw optional `assistantParentMessageId`, normally absent. `saveMessageOnError` and real Dexie `saveMessage` correctly stored that received null as `parent_message_id`. `formatToMessage` intentionally groups assistant rows only when they share a nonnull parent. Reload therefore restored three separate errors, while the live UI had one assistant with three alternatives.

Native evidence inspected: `/private/tmp/cycle4-native122-final-first-identity.json`, `-retry-identity.json`, `-reload-identity.json`, and `-reload-snapshot.txt`. User `pa_6783-8ca1-144-53ea` and PNG SHA256 `314f71d711b39db674e4679f96307ef75f4da2ab93c3c8947d6e5c3ec1d0f1f7` remain identical. Initial assistant `pa_d6af-2303-5c9-dffa`, retry `pa_20e2-a070-ae0-c143`, and later retry `pa_6fd7-026e-185-1b21` become separate bubbles after reload. Root reports no Chat POST; this repair does not claim server duplication or data loss.

A private actual pipeline/save-helper/PageAssistDatabase/Dexie/formatter probe confirms the source diagnosis. Without an explicit caller parent, all three persisted assistant parents are null and reopening yields three assistants. Its positive control supplies the same resolved parent and restores one assistant with all three variants. After the production fix both cases pass. The probe uses installed-only fake-indexeddb and does not access the user's browser/database.

## Approved design and implementation stages

1. **Complete:** trace native identities, compare working success/interruption paths, reproduce actual error-save/restore boundary.
2. **Complete:** change only the general-catch argument. Add permanent behavioral regression through actual Retry handler, pipeline, error helper, saveMessage, PageAssistDatabase and restore formatter, controlling only the storage table adapter and model failure.
3. **Code verification complete; review/native pending:** focused regression, exploratory real Dexie round trip, explicit-root lint comparison, hash freeze and independent handoff.

AC1 means the latest generated failed Retry variant remains active after reload. Manually selecting an older variant is currently memory-only; persisting arbitrary swipe selection is outside this repair. Existing legacy unparented rows are not guessed, merged, deleted or rewritten. That limitation is covered explicitly.

## Changed files

- `apps/packages/ui/src/hooks/chat-modes/chatModePipeline.ts` — one argument.
- `apps/packages/ui/src/hooks/chat-modes/__tests__/chatModePipeline.error-variants.persistence.test.ts` — three permanent controls: three failures then restore; restore between retries; leave unparented legacy records independent. User local ID, exact stored image string, variant IDs/order and latest active index asserted. No text-based deduplication.
- Official Backlog task64 plan/notes updated through CLI.

## RED / GREEN

Run from `apps/packages/ui`:

```sh
./node_modules/.bin/vitest run src/hooks/chat-modes/__tests__/chatModePipeline.error-variants.persistence.test.ts
```

Before production: **2 failed / 1 passed**, correct restoration failures: three assistants rather than one, and two assistants when a reload occurs between retries. `/private/tmp/cycle4-uat124-permanent-red.log`.

Final focused command:

```sh
./node_modules/.bin/vitest run src/hooks/chat-modes/__tests__/chatModePipeline.error-variants.persistence.test.ts src/hooks/chat-modes/__tests__/chatModePipeline.abort-lifecycle.test.ts src/hooks/chat/__tests__/useChatActions.saved-normal.integration.test.tsx src/hooks/chat-helper/__tests__/saveMessageOnError.test.ts src/utils/__tests__/message-variants.test.ts src/hooks/handlers/__tests__/messageHandlers.regenerate.test.ts
```

**102 passed / 6 files**, exit0: `/private/tmp/cycle4-uat124-focused-green.log`. Includes the new3 plus existing Retry, acknowledged identity, image refusal/owner, cancellation and abort controls; counts overlap rather than sum with later independent/combined checks.

Private real-Dexie command, same working directory:

```sh
./node_modules/.bin/vitest run --config /private/tmp/cycle4-uat124-variant-persistence-probe.config.mts
```

Probe/config: `/private/tmp/cycle4-uat124-variant-persistence-probe.test.ts` and `.config.mts`. Baseline **1 expected failed / 1 passed** `/private/tmp/cycle4-uat124-variant-persistence-probe.log`; unchanged probe after fix **2 passed** `/private/tmp/cycle4-uat124-real-dexie-green.log`. Earlier exploratory configuration and missing caller pre-trim fixture mistakes were corrected before this retained red result; they are not product failures. This is actual Dexie over an in-memory IndexedDB shim, not native browser acceptance or a committed dependency.

## Static checks and limits

Repository root:

```sh
apps/tldw-frontend/node_modules/.bin/eslint --config apps/tldw-frontend/eslint.config.mjs apps/packages/ui/src/hooks/chat-modes/chatModePipeline.ts apps/packages/ui/src/hooks/chat-modes/__tests__/chatModePipeline.error-variants.persistence.test.ts -f json -o /private/tmp/cycle4-uat124-eslint-final.json
node /private/tmp/cycle4-uat124-lint-compare.mjs
```

**0 errors / 6 unchanged baseline warnings; zero added or removed diagnostic signatures.** New test has0diagnostics. `/private/tmp/cycle4-uat124-eslint-{final,baseline,comparison}.json` and `.log`. Baseline uses exact filenames/root config and `git show` of the base SHA. Existing Next pages-directory advisory retained, not counted as new source diagnostic. Scoped `git diff --check` clean. Whole TypeScript/integrated verification belongs to root; no new full compiler was run by this agent. Bandit not applicable to TypeScript-only scope; no Python changed.

Latest generated active variant is preserved using existing formatter behavior. Existing historical unparented records remain flattened, intentionally avoiding speculative repair. No claim of native success yet. Permanent tests control the lowest storage adapter; the separately retained real-Dexie probe and pending native test disclose that distinction.
