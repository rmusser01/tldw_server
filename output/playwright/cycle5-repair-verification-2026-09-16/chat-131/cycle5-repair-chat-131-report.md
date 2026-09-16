# UAT131 / TASK13260.71 — promotion receipt identity

## Result

Ready for independent review. **183 permanent tests / 7 suites PASS**, plus **3 real Dexie controls PASS**. No browser, runtime, inference, configuration, staging or commit actions. Native acceptance and combined TypeScript belong to the parent. Exact frozen source/test/task bytes are in cycle5-repair-chat-131-manifest.json; production subset in cycle5-repair-chat-131-production-freeze.json.

## Root cause and repair

Promotion saved role/content snapshots and retained server receipts only in its private prefix-recovery array. It never attached those receipts to the source local greeting. The newly published server conversation could therefore load its canonical greeting beside the unacknowledged local greeting; later inference included both. Equal text does not establish identity.

The Form now supplies visible message identities to the existing persistence hook. It captures the ordered source once, pairs only aligned role/content/attachment-free rows with unique local IDs, and applies each actual receipt or verified recovered-prefix receipt to those captured IDs. Later content edits survive; a conflicting existing canonical identity is rejected. Existing local rows receive transactional ACK updates with owner/history/role and cancellation checks. A synthetic greeting without a previous durable row retains its acknowledged local ID when the first owned mirror is written. Distinct rows with equal content remain distinct.

The existing pending-promotion helper exposes an optional loader-only wait until successful completion. The same waiter survives an incomplete/ambiguous copy and explicit Retry, and is released on operation clear/abort. Ordinary send retains its existing immediate incomplete-save error. Loader reads the current store after waiting, avoiding a render ref from before the final ACK. The persistence operation also captures existing restoreRevision: a route-away/back with identical IDs cannot claim its older receipt. No new ownership or request framework.

## Exact file scope

- apps/packages/ui/src/components/Option/Playground/hooks/usePlaygroundPersistence.tsx
- apps/packages/ui/src/components/Option/Playground/PlaygroundForm.tsx
- apps/packages/ui/src/db/dexie/server-chat-mirror.ts
- apps/packages/ui/src/hooks/chat/useServerChatLoader.ts
- apps/packages/ui/src/services/pending-chat-promotion.ts

Permanent tests:

- apps/packages/ui/src/hooks/chat/__tests__/useChatActions.saved-normal.integration.test.tsx
- apps/packages/ui/src/db/dexie/__tests__/server-chat-mirror.test.ts

## RED / GREEN evidence

- cycle5-repair-chat-131-red.log: actual promotion + mounted loader/mirror first reproduces **3 RED** (immediate ACK, delayed ACK, ambiguous committed ACK).
- cycle5-repair-chat-131-baseline.config.ts / baseline-red.log: final strengthened promotion tests, unchanged ba5233a5a9 production modules supplied by read-only Vite source replay, **3 expected RED / 70 unselected**. No source checkout or file swap.
- cycle5-repair-chat-131-interleaving.log: delayed loader shows stale pre-ACK render projection and ambiguous-copy Retry initially leaves a failed loader; preserved diagnostic before the bounded current-store/success-wait correction.
- cycle5-repair-chat-131-generation-red.log: **1 RED** proves same-history route roundtrip could attach an old receipt before existing restoreRevision was captured.
- cycle5-repair-chat-131-final-green.log: **178 PASS / 6 suites**, including8 new actual-owner cases,9 mirror/receipt controls and existing normal/failed retry, image, cancellation, queued turn, partial0/1copy, ambiguous prefix, readiness, account ABA, character and mirror coverage. Counts overlap earlier runs; they are not additive.
- Main actual-owner tests execute real persistence hook, loader, mirror, normal action/pipeline and ChatTldw acknowledgement parsing. Controlled server transport and in-memory IndexedDB adapter remain test seams. They verify exact Pirate system, one original greeting/local+canonical ID, one user/reply, outbound greeting once, and remount; no real model call is claimed.
- Separate cycle5-repair-chat-131-real-dexie.{test.ts,config.mts,log}: **3 PASS**, actual repository Dexie schema/transactions/formatters, database close/reopen, exact local-ID retention, edit preservation, rollback after ownership changes during the write. Installed fake-indexeddb supplies only the browser IndexedDB substrate. This is not native browser evidence and adds no committed dependency.
- Two existing delayed old-owner-correlation expectations were made more exact: a replacement owner's successful new autosave now legitimately receives its own ACK. They retain draft/content/count checks, match that owner's actual server receipt and explicitly reject the delayed old ID. No failure assertion was disabled.

## Reproduce focused permanent checks

From repository root:

```sh
bun run --cwd apps/packages/ui test \
  src/hooks/chat/__tests__/useChatActions.saved-normal.integration.test.tsx \
  src/db/dexie/__tests__/server-chat-mirror.test.ts \
  src/components/Option/Playground/hooks/__tests__/usePlaygroundPersistence.test.tsx \
  src/hooks/__tests__/useServerChatLoader.mirror.integration.test.tsx \
  src/hooks/__tests__/useServerChatLoader.scope.test.tsx \
  src/hooks/__tests__/useServerChatLoader.test.ts
```

Private Dexie control: bun run --cwd apps/packages/ui test --config /private/tmp/cycle5-repair-chat-131-real-dexie.config.mts.

## Static checks and limits

Repository-root ESLint with explicit apps/tldw-frontend/eslint.config.mjs covers all8 changed code/test files: **0 errors / 99 existing warnings; 0 added/removed**, comparison normalizes only embedded moved line numbers. Artifacts eslint-final.json, eslint-baseline.json, eslint-comparison.json, lint-compare.mjs. Scoped diff-check clean. Bandit N/A: TypeScript-only unit; no backend files changed. Parent owns full compiler; no clean-typecheck claim.

No retrospective matching/deletion of already-damaged unacknowledged records. Unaligned legacy snapshots or image-bearing promotion sources are not assigned guessed IDs; the existing text-only promotion payload is unchanged. This unit repairs the demonstrated ordinary greeting promotion, not attachment serialization or general history migration. Browser-native acceptance remains pending.

## Independent review correction and final refreeze

The reviewer identified a terminal waiter path that the mounted controls did not exercise: the owning scope can resolve already aborted before its abort listener is installed. The helper removed that operation without releasing its saved promise. Replacing an owner also orphaned its previous saved waiter.

Permanent actual-helper tests in apps/packages/ui/src/services/__tests__/pending-chat-promotion.test.ts reproduced **3 RED / 2 GREEN** (waiters-red.log). The bounded helper correction releases saved waiters in the terminal catch and clears/aborts a previous operation before tracking its replacement. Existing incomplete-send behavior, explicit retry continuity and cancelling a loader without cancelling its owner remain covered.

Final command is the six-suite command above plus src/services/__tests__/pending-chat-promotion.test.ts: **183 PASS / 7 suites** in waiters-final-green.log. Eight code/test paths linted with **0 errors / 99 unchanged warnings; 0 added/removed**. Only pending-chat-promotion.ts changed among the five production paths in this refreeze. Previous hashes retained in before-waiter-fix-manifest.json; current authoritative hashes remain manifest.json and production-freeze.json. The three Dexie controls and other four production modules are unchanged. No native acceptance claim.
