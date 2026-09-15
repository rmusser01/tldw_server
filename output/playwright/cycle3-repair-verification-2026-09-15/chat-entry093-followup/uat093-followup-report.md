# TASK13260.33 / UAT093 native entry follow-up

Status: production and six source/test paths frozen at 2026-09-15T22:11:17Z for independent review. No browser/runtime/commit actions. Native saved Robot acceptance remains pending.

## Root cause and correction

The original .33 removal fixed the nested-update loop, but the later native Robot entry still stalled: old Cedar4 remained selected while validated metadata described Robot5. The loader published metadata readiness before optional profile selection, so Playground's deliberate-picker mismatch effect cleared the target/messages. Its captured settings return could then reassert the old route. Multiple actual consumers mount this loader, allowing another metadata response to publish readiness while the first canonical selection was still queued in storage.

The bounded correction:

- Hydrates the validated minimal tracked identity before metadata readiness. Optional profile enrichment retains the original request authority and controller; messages become ready without waiting for that enrichment.
- Reads the existing synchronous selectedAssistant operation revision so newer picker intent wins before React rerenders, including during queued profile writes.
- Waits for stable settlement of the existing selection commit chain before metadata publication, then rechecks original owner/controller. This includes a later queued picker and adds no new state or queue.
- Marks a settings-return handoff consumed only after accepted application. Later deliberate selection cannot reassert the old mount target; cancelled/failed local returns remain unaccepted.

## Scope

Exact six source/test paths and hashes: `/private/tmp/uat093-followup-frozen-manifest.json`; path-only list: `/private/tmp/uat093-followup-owned-files.json`.

Production: `Playground.tsx`, `useServerChatLoader.ts`, `useSelectedAssistant.ts`.
Tests: actual `Playground.coordinator.integration.test.tsx`; existing loader scope/mirror integration fixtures only gain the new selection-revision/wait export contract and operation counter.
Backlog: TASK13260.33 updated through CLI. No .34/title, transport, global docs, schema, runtime or original data edits.

## Behavioral verification

The permanent coordinator harness runs actual Playground, option store, server loader, effective-assistant resolver and useSelectedAssistant. Storage and HTTP boundaries are controlled. Held-storage tests model installed Plasmo's await-storage-before-render contract. Coverage includes delayed metadata/messages/profile, simultaneous loaders, a newer queued picker, same-turn picker/profile completion, account/replacement/unmount cancellation, and existing local/Timeline handoffs.

- Fresh broad run: **128 tests / 9 suites passed**, `/private/tmp/uat093-followup-final-broader.log`.
- Held-storage RED: `/private/tmp/uat093-shared-loader-storage-red.log`; read-only `/private/tmp/uat093-shared-storage-prior.config.ts` removes only the final wait/guard at transform time. Both final current/picker variants pass in `/private/tmp/uat093-shared-loader-storage-green.log` and the broad run.
- Same-turn picker/profile RED: `/private/tmp/uat093-metadata-sameturn-profile-red.log`. This is a permanent coordinator test, not a separate temporary config. The crucial ordering starts picker selection, releases profile, then awaits the picker.
- Other RED evidence: `uat093-metadata-permanent-red.log`, `uat093-metadata-real-selection-red.log`, `uat093-metadata-consumed-route-red.log`, `uat093-metadata-profile-completion-red.log`.
- Scoped ESLint: **0 errors, 23 pre-existing warnings, 0 added/removed warning signatures**. `/private/tmp/uat093-followup-eslint.json` and `uat093-followup-eslint-comparison.json`. The shared-package run also emits the existing Next pages-directory advisory.
- TypeScript: exit2 with **90 exact merged-baseline diagnostics, 0 added/removed signatures**. `/private/tmp/uat093-followup-final-typecheck.log` and `uat093-followup-final-typecheck-comparison.json`. Compiler started22:07:08.533UTC, finished22:07:47.421UTC, after root's Trash wrapper freeze22:06:08UTC. No clean-typecheck claim.
- `git diff --check` passed. Bandit does not apply to these TypeScript-only changes.

## Retained original probes and compatibility

Original configs remain unchanged and are hashed in the manifest:

- `uat093-metadata-selection.config.ts`: stale Cedar RED.
- `uat093-metadata-clear-race.config.ts`: transient target-clear RED.
- `uat093-metadata-held-messages.config.ts`: original trace showed readiness/clear/repin but eventually recovered; do not describe its original log as a failing assertion.
- `uat093-concurrent-metadata.config.ts`: two real concurrent-loader failures.

The first three pass on final production via named `*-compat.config.mjs` runners (one each). These retain the prior committed coordinator fixture and adapt only the selection revision/wait mock contract. They are explicitly compatibility runs, not unchanged harnesses. Logs are named `*-final-green.log`.

The concurrent runner `uat093-concurrent-metadata-compat.config.mjs` restores only the expected Form stub string before applying the original untouched transform. The original transform still inserts two real loaders and holds both metadata responses. The final compatibility run passes both original cases (2/2) in `uat093-concurrent-metadata-final-green.log`. The permanent held-storage tests provide the stronger current fixture control.

## Exact commands

From `apps/packages/ui`:

```sh
npx vitest run src/components/Option/Playground/__tests__/Playground.coordinator.integration.test.tsx src/hooks/__tests__/useLoadLocalConversation.test.tsx src/hooks/__tests__/usePlaygroundSessionPersistence.test.tsx src/hooks/__tests__/useServerChatLoader.test.ts src/hooks/__tests__/useServerChatLoader.scope.test.tsx src/hooks/__tests__/useServerChatLoader.mirror.integration.test.tsx src/hooks/__tests__/useSelectedAssistant.test.tsx src/hooks/__tests__/useMessageOption.assistant-overlay.test.tsx src/components/Notes/__tests__/NotesManagerPage.stage26.backlink-labels.test.tsx --maxWorkers=1
npx vitest run src/components/Option/Playground/__tests__/Playground.coordinator.integration.test.tsx -t 'shared selection storage|same turn' --maxWorkers=1
npx vitest run --config /private/tmp/uat093-concurrent-metadata-compat.config.mjs -t 'keeps a saved target over a previous character' --maxWorkers=1
npx vitest run --config /private/tmp/uat093-shared-storage-prior.config.ts -t 'shared selection storage' --maxWorkers=1
```

The last command intentionally restores the pre-correction behavior and must fail. The first three metadata compatibility configs each use `-t 'UAT093 keeps the explicit saved target'`.

ESLint from `apps/packages/ui`:

```sh
node ../../node_modules/.bun/eslint@9.39.2+288993669ddeca06/node_modules/eslint/bin/eslint.js --config ../../tldw-frontend/eslint.config.mjs src/components/Option/Playground/Playground.tsx src/components/Option/Playground/__tests__/Playground.coordinator.integration.test.tsx src/hooks/chat/useServerChatLoader.ts src/hooks/useSelectedAssistant.ts src/hooks/__tests__/useServerChatLoader.scope.test.tsx src/hooks/__tests__/useServerChatLoader.mirror.integration.test.tsx --format json
```

TypeScript from `apps/tldw-frontend`: `npx tsc --noEmit --incremental false`.

## Native evidence and limits

Parent retains `/private/tmp/uat093-native-repaired-entry-timeout-snapshot.txt`, `uat093-repaired-entry-stalled.png`, `uat093-repaired-stall-normal-reload.txt`. Its actual auth/me verified Alice and backend observed successful Robot metadata/messages/profile requests. This agent has not performed a new native acceptance run. Original saved data, authorization, offline local-history contract, and other agents' files remain outside this correction.
