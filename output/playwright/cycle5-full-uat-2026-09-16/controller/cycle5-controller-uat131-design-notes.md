# UAT131 / TASK13260.71 — saved greeting copied twice on the first ordinary turn

## Outcome

Confirmed separate failure boundary from reopened UAT068. The earlier Character replacement failure helped create the local state, but UAT131 is an identity loss during ordinary chat promotion and reload. A saved greeting and its unacknowledged local source coexist; the first ordinary send includes both. Do not fix this by deduplicating equal assistant text or by changing provider routing.

Read-only diagnosis at HEAD `d9c8854ec07a92bf753e6be4bff49636710082e9`. No repository/source/test/task edits, browser actions, API calls, runtime changes or inference performed. Private probe/config/log only. Product hashes retained in `/private/tmp/cycle5-uat131-source-hashes.txt` (seven source/test entries plus probe config/log).

## Native evidence and chronology

- `/private/tmp/cycle5-multi-native-pirate-new-saved.txt`: actual New saved chat action.
- `/private/tmp/cycle5-multi-native-pirate-context.txt`: immediately before clearing, local conversation has one `Hello! How can I help you today?` row marked greeting and no saved history.
- `/private/tmp/cycle5-multi-native-pirate-clear-assistant.txt`: actual explicit Clear assistant click, resulting snapshot timestamp 08:56:30.099Z.
- `/private/tmp/cycle5-multi-native-pirate-standard.txt`: Standard chat, Server chat / History linked, already TWO greeting texts **before weather Send**. One is an ordinary Assistant row; the other is Default Assistant with a greeting badge.
- `/private/tmp/cycle5-multi-native-pirate-send-ready.txt`: same two-row condition before dispatch.
- `/private/tmp/cycle5-multi-native-pirate-canonical.txt`: reload GET200 at 08:58:04.598Z; complete five-row listing (`has_more:false`) for `bc7a816c-17d1-4da9-a6c0-b98080a4f997`:
  1. `053a01b5-ed24-4438-9a7d-b7390a375c2e`, system Pirate instructions, 08:56:28.803Z.
  2. `68450214-62ad-48cc-9da8-f443eb5d916b`, greeting, 08:56:28.974Z, metadata null.
  3. `2988cf09-a990-4897-8f19-d77087773103`, identical greeting, 08:57:06.483Z, assistant role/name metadata.
  4. `76168f81-753e-4ac1-abf7-76c220f66274`, weather question, 08:57:06.485Z, correlation `pa_a7fc-b4ea-f61-243c`.
  5. `def930d0-b5ca-4007-be97-703fb2104eef`, successful Pirate response containing ARRR, 08:57:10.368Z.

Native author confirmed one Clear assistant and one weather Send. Prior actions included New saved chat, Modes/Explore, switching to the other saved TestBot tab and the failed Default Assistant selection, then returning to this tab. No creation/message observer was installed during Clear assistant. Therefore the exact create/message POST cause is source-backed inference, not independently captured native POST evidence. Canonical duplicate identities and the pre-send double display are directly captured.

`Helpful_AI_Assistant` on the second greeting/final reply is not proof of wrong routing. Backend persistence deliberately assigns assistant sender names from its resolved character-card context; the system text and successful ARRR response remain valid positive evidence. No model/provider request mismatch was established here.

## Causal source boundary

1. `apps/packages/ui/src/components/Option/Playground/Playground.tsx:2691`: Clear assistant clears selection/server identity and sets Standard workflow; it retains current history. Retaining the selected Pirate system instruction and legitimate earlier transcript is consistent with preserving history.
2. `.../hooks/usePlaygroundPersistence.tsx:275`: promotion snapshots role/content only, losing local identity/type. The guard permits ordinary neutral history after tracked Character workflow is cleared; auto-save starts at line501. Creation sets the server chat ID before the copy completes (line363).
3. Same hook line414–424: each `addChatMessage` receipt is checked, but its ID is stored only in closure-local `acknowledgedIds`. It is not attached to the source local message or its durable mirror. The hook contract receives history but no local message identity/ACK projection callback.
4. `apps/packages/ui/src/hooks/chat/useServerChatLoader.ts:388`: a leading assistant is inferred as a greeting only when `characterId != null`. This neutral saved conversation has no such identity and its saved row has no explicit greeting type.
5. `apps/packages/ui/src/db/dexie/server-chat-mirror.ts:125`: the canonical row has its server ID, while the local greeting has only a local ID. The synthetic-greeting special case applies only if an incoming row is also explicitly/inferred greeting. Thus both rows remain. Loader lines1035–1037 and1098–1107 publish both into the visible transcript **and inference history**.
6. `apps/packages/ui/src/utils/generate-history.ts` sends both assistant entries. Backend `tldw_Server_API/app/core/Chat/chat_service.py:4400` trims an overlapping saved prefix, not arbitrary duplicate content; the additional local greeting after that prefix is persisted at line4470 onward. This explains the later second canonical greeting beside the new user turn. Do not broaden backend content deduplication: equal assistant turns can be legitimate.

The trigger does not require a stale canonical Character route or UAT068's route-reassertion loop. A neutral saved promotion of a local synthetic greeting is sufficient. Repair 068 first if ownership overlaps, then retain separate 131 acceptance.

## Private interacting production-boundary probe

- Config: `/private/tmp/cycle5-uat131-greeting-probe.config.ts`
- Output: `/private/tmp/cycle5-uat131-greeting-probe.log`
- Command, working directory `apps/packages/ui`:

  `bun run test --config /private/tmp/cycle5-uat131-greeting-probe.config.ts`

A Vite read-only transform adds four cases to the existing real adapter → mounted `useServerChatLoader` → mirror → formatter integration fixture. Production loader, client adapter, merge and visible/inference stores execute. Synthetic transport and in-memory DB fixture remain the existing test seams; this is not another native run.

Result: **1 RED / 3 GREEN**:

- RED: neutral conversation + unacknowledged local greeting → two visible rows and two inference-history entries; expected one.
- GREEN: same neutral conversation with that local row's canonical ACK ID → one row/history entry.
- GREEN: tracked Character conversation retains its existing inferred-greeting behavior → one row/history entry.
- GREEN: two genuinely separate canonical assistant rows with equal content → both remain.

The test does not execute promotion, a real backend completion, or browser selection. Promotion receipt loss and subsequent backend prefix handling are source-traced; the native artifacts independently demonstrate their end-to-end outcome. No new ordinary request wire capture is claimed.

## Smallest safe post-freeze correction

Own the promotion ACK boundary and its existing caller/tests; reuse existing owner-aware message/mirror reconciliation. Capture stable source local IDs alongside the promotion snapshot, then apply each verified save/reconciled-prefix receipt to its exact captured source row under the same authority, history and operation generation. Attach canonical ID/version in mounted state and durable mirror as appropriate. Preserve current edited content according to existing revision rules; a successful save receipt identifies the original row, not permission to replace later edits. Do not zip receipts onto a changing live message array.

For ambiguous commit/network failure, retain current exact fresh-prefix retry validation and recover its IDs onto those same captured source identities. Existing readiness pauses, owner/target invalidation and no duplicate mutation replay remain required. Early or late loader completion around create/copy must converge on those IDs.

No new backend text heuristic, forced Character identity, unconditional first-assistant deletion or broad synthetic-greeting deletion is justified. Clearing the assistant need not erase an already shown legitimate greeting; for this retained-history scenario the expected canonical shape is system + one greeting + user + reply (four rows). A separate new empty chat should remain empty according to its current product contract. Do not silently rewrite the already-corrupted native conversation without explicit repair policy.

## Permanent regressions required

1. Actual mounted promotion → acknowledged writes → real neutral loader/mirror → ordinary send → reload: preserve exact Pirate system instruction and exactly one original greeting, one user and one final reply. Verify canonical IDs and local IDs, not just text counts.
2. Interleave loader before/after greeting ACK and completion; account/target/history A→B→A or unmount during copy cannot attach old receipts or write into replacement mirror. No extra completion dispatch.
3. Partial copy, ambiguous committed write, readiness pause and explicit Retry recover the exact prefix once; two equal assistant rows in captured source retain distinct identities/order.
4. Preserve unrelated unsent drafts and edits made during save/readback, other saved rows with equal text, and genuine new identical turns.
5. Reload/remount and mirror reopen retain ACK mapping; tracked Character greeting behavior remains green. Keep 068 picker/confirmation controls, 113 canonical Character reload, and 123 ordinary Note/History mode controls.

The native incident proves canonical duplication; it does not prove all plain-chat promotion variants or a wrong-provider defect. No broader scope is proposed.
