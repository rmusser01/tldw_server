# UAT103 / TASK13260.44 — read-only diagnosis

## Finding

Confirmed missing acknowledgment of the normal Chat user message, followed by correct preservation of that apparently unsynced local row during server reconciliation. This is one saved conversation and one linked local history, not another server conversation, mode change, or assistant sender-projection defect.

The runner's `output/playwright/cycle4-full-uat-2026-09-16/multi/uat103-read-only-mirror.txt` at 2026-09-16T00:46:21.046Z confirms history `pa_1ce6-0e93-f32-a6ee` belongs to normal conversation `a9feabc7-38aa-4b06-b08d-0b673b069819` and holds exactly seven rows:

| Role | Original local ID | Canonical server ID | Content length |
|---|---|---|---|
| First user, original local | pa_c2f0-84a4-fe2-10dd | null | 71 |
| First user, added mirror | pa_1ce6-0e93-f32-a6ee:server:9ab80484-2cc9-44b4-ae63-109b0afa0013 | 9ab80484-2cc9-44b4-ae63-109b0afa0013 | 71 |
| First assistant | pa_ff7f-88e7-a45-4ae2 | 6da03e47-d9b9-46c5-a683-d9fad8c85abf | 8 |
| Second user, original local | pa_c743-b596-7e5-f183 | null | 66 |
| Second user, added mirror | pa_1ce6-0e93-f32-a6ee:server:5cf2af2b-c6fb-4a67-abfe-bc68b181c87b | 5cf2af2b-c6fb-4a67-abfe-bc68b181c87b | 66 |
| Second assistant | pa_787e-118d-d3c-6141 | bdc63deb-6c67-4b56-8488-1d2775ab8850 | 8 |

The seventh row is the single canonical system message `dafcaad4-b891-4035-a7c0-bbccc8e0dd8e`. Independent server reread `normal-messages-settled.json` has exactly five rows. Visible rows have the same IDs as persistent rows. The snapshot establishes the duplicated text; the bounded database probe deliberately returns no content/settings/auth values.

## Exact causal boundary

1. `apps/packages/ui/src/hooks/chat-modes/chatModePipeline.ts:225` creates a separate local user ID. Its completion publication at 826–836 attaches `modelClient.serverMessageId` only to the assistant, and success payload at 918–930 carries only `assistantServerMessageId`.
2. `apps/packages/ui/src/hooks/chat-helper/index.ts:556–579` saves the user with local ID and no server ID. The assistant write at 580–596 has `serverMessageId: assistantServerMessageId`. `types/chat-modes.ts:36` likewise has only the assistant acknowledgment field.
3. `apps/packages/ui/src/models/ChatTldw.ts:181–184` consumes only `tldw_message_id`. Backend `core/Chat/streaming_utils.py:1333–1351` exposes the saved assistant ID; nonstream `core/Chat/chat_service.py:7100–7104` does likewise. `build_context_and_messages` at 4343–4354 persists current input messages but discards the returned saved ID. Thus this is not simply an existing user-ID field forgotten by the formatter.
4. `apps/packages/ui/src/db/dexie/server-chat-mirror.ts:49–53,73–80` keys by acknowledged server ID (or old local ID already equal to a server ID), preserving unmatched local rows. Persistent reconciliation at 99–101 similarly cannot associate the random user ID with the server UUID, so it adds a qualified canonical row. It returns all old/new rows, as required to preserve genuine unsynced work.
5. `apps/packages/ui/src/hooks/chat/useServerChatLoader.ts:1020–1022` merges the server snapshot; at 1076–1092 it persists and republishes the mirror. Both paths therefore retain the two unacknowledged users. The later mirror stage explains how a correct initial five-row snapshot can settle at seven, though the exact scheduling order was not instrumented live.

Additional recovery constraint: all current native parentMessageId values are null. The normal pipeline initially defaults the assistant parent to its local user ID (`chatModePipeline.ts:238–242`), but mirror projection at `server-chat-mirror.ts:113` replaces it with `remote.parentMessageId ?? null`. We cannot rely on surviving local parent links to repair an already-reconciled legacy mirror, and this capture does not independently establish its pre-reload parent value.

## Controlled reproduction and existing test gap

No repository source/test changes or browser/runtime operations by this investigator. A private Vite transform appends a diagnostic to the existing real adapter → loader → mirror → formatter test module. Only its pre-existing storage/transaction and authority/transport seams are mocked. It feeds the actual retained five-row server fixture and four local rows shaped like normal success persistence: random user IDs without acknowledgment, assistant rows with canonical IDs.

- `/private/tmp/uat103-mirror-probe.config.ts` and `uat103-mirror-probe-red.log`: one expected failure, expected5/actual7. Ten original cases are selection-filtered, not disabled or claimed rerun. Actual loader, mirror and formatters reproduce both duplicate users and unique assistants.
- `/private/tmp/uat103-mirror-controls.config.ts` and `uat103-mirror-controls.log`: two controls pass. Adding correct user acknowledgments yields five rows; adding a genuinely unsynced equal-text user draft yields six and preserves that draft. This establishes why blanket text deduplication would be wrong; no product fix was applied.
- Initial private runner startup used an unresolvable absolute package-export path (`uat103-mirror-probe.log`). Corrected to the installed Vitest config entry before the behavioral run. Startup failure is not product RED.

Existing `useChatActions.saved-normal.integration.test.tsx:600–619` mocks the model/server pair without canonical message IDs and never remounts through the server loader. Existing `useServerChatLoader.mirror.integration.test.tsx:65` uses old user ID `question`, equal to the server ID, so it exercises the legacy identity shortcut rather than the actual random-ID normal-persistence shape. Existing helper scope test intentionally avoids assigning the assistant ID to the user, but does not establish a separate acknowledged user ID.

Three useful existing patterns: the tracked-character user write records `createdUser.id` before continuing (`useChatActions.ts:2216–2239`); normal assistant completion already carries the server acknowledgment through the model/pipeline/helper; the mirror already reconciles acknowledged/legacy-equal IDs while preserving a true draft. Reuse those boundaries rather than adding a render-only filter.

## Minimal durable repair contract after freeze

1. Preserve the actual server user acknowledgment for each successful saved normal turn and attach it to that same local user row in memory and Dexie before reconciliation. Carry it alongside the assistant acknowledgment through the existing model → pipeline → save payload → helper chain. Keep the existing local primary key; do not replace or move other histories' rows. The server currently does not expose that acknowledgment, so implementation requires an explicit bounded choice: additive persisted-user metadata in existing completion responses is the strongest correlation; a scoped readback must instead be anchored to the known committed assistant/turn and verified exact request content, never simply the last user or a global equal-text match. Do not silently invent an ID if metadata/readback fails.
2. Preserve local turn lineage when a server row has no corresponding lineage value. Existing contaminated mirrors need a separate conservative recovery rule. This live mirror already lost parent links; equal content/timestamps alone cannot prove an unknown row is not a genuinely unsynced draft. Do not delete, hide or rewrite ambiguous rows merely to force the count to five. Define recovery using defensible saved-turn provenance; if unavailable, preserve the original and make uncertainty explicit. This is a repair-design constraint, not permission to leave the newly created acknowledged path broken.
3. Retain owner/generation guards before and after acknowledgment/readback and local transaction. A delayed A→B→A completion or switched conversation must not associate IDs with another owner/turn. Temporary/promoted histories, canceled/failed sends and explicit retry require controls because they create local rows without the ordinary normal-completion acknowledgment flow.

## Required regression boundary

Primary regression should join actual normal send pipeline + actual save helper + owned persistence adapter with actual reload loader/mirror, using a transport fixture that assigns different local and server UUIDs. Two completed turns, system row, unmount/remount, awaited mirror completion and second reload must remain five visible/persistent rows, with one conversation and Standard identity.

Extend existing files rather than a hook-only UI filter test:

- `hooks/chat/__tests__/useChatActions.saved-normal.integration.test.tsx`: make server metadata realistic and include final persisted ID assertions; connect its produced rows to actual loader/formatter rather than fabricating already-acknowledged rows.
- `hooks/__tests__/useServerChatLoader.mirror.integration.test.tsx` and `db/dexie/__tests__/server-chat-mirror.test.ts`: real random local IDs, current contaminated mirror fixture, repeat reload, ambiguous equal-text draft preservation and global-PK/other-owner controls.
- `models/__tests__/ChatTldw.stream-metadata.test.ts`, `hooks/chat-modes/__tests__/chatModePipeline.conversation-id.test.ts`, `hooks/chat-helper/__tests__/saveMessageOnSuccess.scope.test.ts`: full distinct user/assistant metadata propagation, no ack on temporary/failed paths, delayed owner changes.
- If completion metadata is extended, cover streamed and nonstreamed backend response contracts, save failures/continuation/retry, and run scoped Bandit. Native real IndexedDB two-turn send/reload remains required after implementation; the private probe is not a browser database test.

Meaningful scenarios: (a) ordinary two turns and queued second turn with delayed link; (b) same-text repeated legitimate turns plus equal-text unsynced draft; (c) current contaminated mirror and two settled reloads; (d) temporary→Saved promotion and failure/retry; (e) delayed acknowledgment/storage/restore across conversation switch and A→B→A. Character reload, newer local edits and failed/canceled turns must remain intact.

No further live evidence is needed for the confirmed cause. Capturing a future unchanged raw final completion metadata frame or pre-reload local IDs would strengthen protocol implementation tests, but is optional and must not require another inference during frozen UAT. The runner already supplied the exact bounded read-only evidence requested.
