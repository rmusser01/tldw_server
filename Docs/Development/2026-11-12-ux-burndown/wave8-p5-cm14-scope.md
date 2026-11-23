# Wave 8 scope: finishing D7 P5 and CM-14

Read-only pass against dev @ ab04a4b0e8 (2026-11-23), after the owner parked
#3071/#3159 [21]. Paths use `ui/` for `apps/packages/ui/src/` and `api/` for
`tldw_Server_API/app/`. Owner decisions [22] and [23] are in the plan.

## Findings that changed the plan

1. **Edit and send needs no server work.** Only the gate at
   `ui/hooks/chat-modes/chatModePipeline.ts:898-900` (`!historyTurn.historyEdit`)
   keeps it client-settled.
   - `admitHistoryInput` already sends `tldw_history_edit_v1` for server
     persistence (`:685-694`).
   - `beforeDispatch` already picks completion or messages
     (`normalChatMode.ts:1358-1361`).
   - The server accepts edits on completions
     (`api/core/Chat/chat_request_schemas.py:1013-1024`,
     `chat_service.py:3227-3236`). The tests are in
     `tests/Chat_NEW/integration/test_history_branch_admission_api.py:203-370`.
   - Every server with `native_completion_v1` accepts the edit field.
2. **Existing-selection turns are local-only** (`useChatActions.ts:3804-3811`,
   `useMessage.tsx:2975-2981`). Local owners keep model, generation info and
   sources in Dexie. Not part of P5.
3. **Plain server chats have no Retry for an unanswered saved question.** This
   happens after a Stop before the first token, or a provider error after
   admission.
   - `canRetryHistoryTurnRecovery` excludes server persistence
     (`ui/services/history-turn-retry.ts:44-50`).
   - "Review saved question" is wired only for character chats
     (`Playground.tsx:4124`, `useMessage.tsx:4140`; `useChatActions.ts:3954-3961`
     and `:4434-4436` need `tracked_character`).
   - The CM-02 harness test passes only because the fake server defaults to
     `nativeCompletion: false`.
4. **The existing-input contract forbids current messages**
   (`chat_request_schemas.py:1001-1008`, B02 "Compatibility"). Decision [22]
   amends it to allow system-only messages.
5. **A pre-existing gap: the default non-RAG prompt.** Plain native Regenerate
   and Continue send only `settings.systemPrompt`
   (`native-history-character-send.ts:630-636`), so they drop the default
   non-RAG prompt, which only `normalChatMode.ts:532` and
   `continueChatMode.ts:66` read. Lane 8B's Retry must not inherit this.
   Filed as a follow-up for Regenerate and Continue.

## Turn kinds

| Turn kind | Decided at | Wave 8 |
|---|---|---|
| Edit and send (native) | `chatModePipeline.ts:898-901` | **Server-settle** (8A) |
| Retry of a server-saved unanswered question, plain chat | `history-turn-retry.ts:44-50`; `Playground.tsx:4124`; `useMessage.tsx:4140` | **Server-settle** through `tldw_history_server_existing_input_v1` (8B) |
| Retry with `retryAdmission` (client-managed recovery) | `chatModePipeline.ts:900`, `:698-704` | Client-settled: it follows its original turn |
| Actor that rewrites history | `normalChatMode.ts:744-754`, `:768` | Client-settled: needs an actor contract (L) |
| OCR | `normalChatMode.ts:768` | Client-settled: model input differs from saved input (M–L) |
| `messageForModel` (document snippets, context composition) | `normalChatMode.ts:768` | Client-settled (M–L) |
| `historyForModel` (`clientManagedContext`) | `normalChatMode.ts:1678`, `:769` | Client-settled (L) |
| Saved system or error rows in the selection | `normalChatMode.ts:770-772` | Client-settled. The server would include rows the client drops, which changes every server-settled turn. Follow-up |
| Reasoning rows on old servers | `normalChatMode.ts:762-764`, `:771-772` | Client-settled. Current servers support them |
| Sync v2 owners; servers without `native_completion_v1` | `chatModePipeline.ts:899`; `chat.py:8645-8650`, `:4825-4830` | Client-settled until the outbox (#3182) |
| Image questions on Retry | existing input is text only (`history_context.py:37-38`) | Unchanged |
| Local and temporary chats | — | Dexie keeps the metadata |

Under decision [23], client-settled replies keep losing the output-limit marker
and model after a reload. That is what remains of CM-03 (a) and CM-04 (1). It
is documented and revisited with #3182.

CM-04 (2) is already on dev. `formatSelectedHistory` restores `sources` from
`workspace_rag_sources_v1` (`ui/db/dexie/helpers.ts:414-417`, `31598212f4`).

CM-04 (3) is a follow-up, size M. Sources for server-settled and web-search
turns need:
- a new allow-listed request field (web sources have no `mediaId`);
- writing it in `completion_save_message`;
- lifting the refusals at `chatModePipeline.ts:937-938` and
  `normalChatMode.ts:1666-1670`;
- a design note, because D7 §6.4 lists sources only for P8.

## Lanes

The lanes run in parallel. Shared files are touched only in disjoint hunks.
Merge order: 8A, 8B, 8C.

### 8A — P5 Edit and send (S)

- Drop `!historyTurn.historyEdit` from the `nativeCompletion` gate
  (`chatModePipeline.ts:898-901`). Keep every other condition, especially
  `native_completion_v1.supported`.
- `isDefinitiveNativeEditRejection` and `isDefinitiveNativeBranchRejection`
  (`ui/utils/history-selection.ts:228-305`) follow `.cause`. TldwChat wraps
  errors (`TldwChat.ts:729`), so a wrapped 409 `history_edit_target_changed`
  must be classified as refused, not unknown.
- Files: `chatModePipeline.ts` (that gate only), `ui/utils/history-selection.ts`,
  and new tests.
- Tests (red first):
  - A Playground harness test with `createFakeTldwServer({ nativeCompletion:
    true })`. Saved chat, Edit → Save & Send, reply with `finishReason:
    "length"`.
    - The completion body has `save_to_db: true`, `tldw_history_branch: true`
      and `tldw_history_edit_v1`.
    - No `POST /chats/{id}/messages`.
    - The server row has `metadata_extra.generation_status: "length"`.
    - "Output limit reached" shows after the view follows the reply.
  - A refusal variant of `normalChatMode.history-selection.integration.test.tsx`
    (around `:6830-7009`): `native_completion_v1` in the capture,
    `tldwChat.streamMessage` throws a wrapped 409. Expect state "refused" and
    the edited text kept.
  - A unit test in `chatModePipeline.history-admission.test.ts`: a
    `historyEdit` turn with native completion gets `saveToDb: true` and no
    `clientManagedHistory`.
  - Re-run `Playground.message-actions` and
    `normalChatMode.history-selection.integration`.

### 8B — P5 plain-chat saved-question Retry (M frontend, S backend)

- **Backend [22].** Existing-input requests may carry system-only current
  messages.
  - The validator (`chat_request_schemas.py:1001-1008`) rejects user,
    assistant and tool roles.
  - `_existing_input_validation_request` and the completion path
    (`chat.py:1099-1129`, `:4622-4644`, `:4866-4895`): the provider sees the
    system messages, then the saved selection and question once. No system row
    is saved, and moderation still checks the saved question.
  - Add a capability sub-flag on `native_completion_v1` (`chat.py:8645-8650`) so
    clients know a server accepts it, with an ADR-068 note.
  - Pytest in `test_history_selection_api.py`:
    - system message plus existing input: the provider sees both, the
      question once, and no system row is saved;
    - a user or assistant message alongside existing input is rejected with 422;
    - moderation still runs on the saved question.
- **Frontend.**
  - Route plain `historyServerExistingInput` to
    `sendNativeHistoryCharacter({ existingInput })`, the way `historyRegenerate`
    is routed (`useChatActions.ts:4327-4349`). Put the logic in a new module:
    `useChatActions.ts` and `useMessage.tsx` are frozen with almost no headroom,
    so keep their edits to routing lines.
  - Carry the System and Prompt instructions (including the default non-RAG
    prompt) when the server advertises the sub-flag.
  - On an older server, refuse before sending while an instruction is active,
    and keep the question.
  - Remove the character-only gates (`Playground.tsx:4124`, `useMessage.tsx:4140`).
  - Write plain-chat wording for `HistorySelectionReview.tsx:1001-1030`, which
    now says "saved character context".
  - Cover both the full page and the side panel.
  - Harness (`fake-tldw-server.ts`): support existing-input capture (now a 422
    at `:675`); settle the reply under the existing question in
    `persistCompletionTurn` (`:361-421`); write `model_id` from `body.model`.
- **Docs.** In the D7 P5 section:
  - the residual client-settled list [23];
  - sync-v2 Q2 answered: Sync v2 owners stay client-managed until the outbox;
  - update the re-triage rows CM-03, CM-04, P5 and CM-14.
- **Tests (red first).**
  - The CM-02 harness test with `nativeCompletion: true`, full page and side
    panel. Stop before any text, then "Review saved question" → "Retry saved
    question".
    - One completion with `tldw_history_server_existing_input_v1` and no
      user or assistant messages.
    - One copy of the question on the server, and the reply's parent is the
      question.
  - A Retry reply at the output limit keeps its marker and model.
  - With an instruction active, the instruction is carried; an old server
    refuses instead.

### 8C — CM-14 first-send bubble (M)

- In `runChatModeWithHistory`, before `captureNormalHistoryTurn`
  (`normalChatMode.ts:1616-1624`), for ordinary new input only (not
  regenerate, `editResend`, `editAndSend` or `retryAdmission`, and the message
  not empty):
  - Fix `userMessageId` (`params.userMessageId ?? generateID()`).
  - Publish the user bubble idempotently, as `reserveDocumentProcessingTurn`
    does (`PlaygroundForm.tsx:4300-4327`).
  - Pass the id to the pipeline, which replaces the list with the same id.
  - Remove the bubble by id if the turn exits before `runChatPipeline`.
- Add `quietLoad?: true` to `HistoryLoadTarget` (`useHistorySelection.ts:71-81`)
  and set it on the first-send target (`normalChatMode.ts:920-925`).
  - `open()` publishes it.
  - `install()` skips `onCapture` for a quiet load, so there is no bubble →
    empty → bubble flash.
- `HistorySelectionReview.tsx:1092-1098`: hide the loading panel for a quiet
  load; otherwise show it only after 300 ms.
- `chatModePipeline.ts` does not change. Character first sends and the side
  panel are out of scope.
- Files: `normalChatMode.ts` (the target block and `runChatModeWithHistory`),
  `useHistorySelection.ts`, `HistorySelectionReview.tsx:1092-1098`,
  `HistorySelectionReview.test.tsx`, a new
  `Playground.first-send.integration.test.tsx`, and a request-gate helper in
  the `fake-tldw-server.ts` router.
- Tests (red first):
  - Hold `POST /api/v1/chats/`, type, press Enter. The bubble is visible, there
    is no "Loading selected history", and the composer is empty. Release:
    exactly one user bubble, and the right server rows.
  - A failed create removes the bubble. The text stays in D7 P4's "Server save
    unconfirmed" recovery, as on dev, not in the composer (see the Wave 8 log).
  - A later send in a saved chat shows no duplicate bubble.
  - A Stop right after the quiet load leaves no stale history.
  - Fake timers: the panel is hidden before 300 ms, shown after, and hidden for
    a quiet load.
  - Update `HistorySelectionReview.test.tsx:726-742` on purpose: it expects the
    panel at once.

## Risks

- The harness defaults to client-managed turns, so P5 tests must set
  `nativeCompletion: true`.
- Every P5 change keeps the `native_completion_v1.supported` check, so Sync v2
  owners stay client-managed.
- `useChatActions.ts` and `useMessage.tsx` are frozen with little headroom.
- CM-14: remove the bubble only by its id. A Stop right after the quiet load
  must not leave stale history.

## Follow-ups

- The default non-RAG prompt in plain native Regenerate and Continue (finding 5).
- CM-04 (3): sources at server settlement.
- Saved system and error rows in the server's model context.
- Flip the harness default to `nativeCompletion: true`.
