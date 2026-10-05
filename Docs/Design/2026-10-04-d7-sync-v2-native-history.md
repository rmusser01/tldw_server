# D7: native history for Sync v2 owners

- **Date:** 2026-10-04
- **Status:** §3 is implemented on `feat/chat-sync-v2-native-history`. §5.2 and §5.3 are fixed on `fix/sync-v2-unenrolled-delete` (#3181, #3182); §5.1 is open. §4 is proposed and needs the owner's decisions (§6).
- **Tracking:** epic #3101 · E2 #3126 · D7 design PR #3165, §8 Q4 ("Sync v2 users are in scope")
- **Builds on:** P3 (PR #3174, client-supplied chat id). Independent of P1 (#3171) and P2; see §8 for the merge.
- **Code base:** `origin/feat/chat-idempotent-create-p3 @ a96114f238`. Line numbers are on that commit.
- **Path shorthand:** `api/` = `tldw_Server_API/app/`; `ui/` = `apps/packages/ui/src/`.

---

## Summary

A user with an active Sync v2 profile (the Chatbook desktop client plus the WebUI on one account) could not use the WebUI's native history owner at all. Capture, admission and settlement returned 409 `sync_owner_unsupported`, and a chat could not be created with a client id (409 `sync_client_chat_id_unsupported`). After D1 (P4) that user could not send from the WebUI.

**What works now for a Sync v2 owner**

| Step | Route | Result |
|---|---|---|
| Create the chat with a client id (P3) | `POST /chats/` with `id` | Same replay (200), conflict (409) and trash (410) rules as without Sync. Publishes one `chat.conversation` upsert. |
| Capture a selection | `POST /chat/conversations/{id}/history/selection` | Allowed, except for a chat bound to a character (below). A read; publishes nothing. |
| Confirm a reviewed legacy path | `POST /chat/conversations/{id}/history/legacy-projection` | Allowed. Server-local; publishes nothing. |
| Admit the user turn | `POST /chats/{id}/messages` with `tldw_history_selection_v1` | Publishes one `chat.message` append. |
| Settle the reply | `POST /chats/{id}/messages` with `tldw_history_admission_v1` | Publishes one `chat.message` append. |

This is the client-managed turn the WebUI uses for ordinary chats today: capture, admit, a stateless `/chat/completions` (`save_to_db: false`, no conversation id), settle.

**What is still refused for a Sync v2 owner**

- A **server-settled turn**: `/chat/completions` with `tldw_history_selection_v1`. It still returns 409 `sync_owner_unsupported`, before any write. The WebUI uses this path for chats bound to a character today, and P5 moves every chat to it. §4.1 says what blocks it.
- **Capture of a chat bound to a character.** Such a chat takes server-settled turns, so capture still returns 409 `sync_owner_unsupported` for it. Character chats therefore behave exactly as before: the WebUI stops at the handshake and dispatches nothing. Without this, the send would fail one step later and the WebUI would keep an "unknown outcome" recovery record for a turn the server refused (`ui/hooks/chat/native-history-character-send.ts`, the `dispatched` branch).
- Versioned messages with an image (400 `sync_v2_binary_message_unsupported`, the existing Sync v2 M1 limit).
- The character routes `complete-v2` and `completions/persist`, message edit, chat restore and permanent delete. These were already refused under Sync v2 and are unchanged.

**Three defects found on the base branch** are listed in §5. The first two together could leave a user's Sync v2 dataset blocked. The second (the block itself) and the third are fixed on `fix/sync-v2-unenrolled-delete`. The first is open: its rows are no longer a risk to the dataset, but other devices still do not receive them.

---

## 1. How Sync v2 writes a chat today

Sync v2 is an append-only log of envelopes per dataset, plus a projection of that log into the owner's ChaChaNotes database.

- **A profile is active** when the user has the default personal Chatbook dataset (`api/core/Sync/v2/server_origin.py:293-308`). The check is per user. Nothing records which conversations are in the dataset.
- **An ordinary API write is routed through the log.** With an active profile, `POST /chats/`, `PUT /chats/{id}`, `DELETE /chats/{id}`, `POST /chats/{id}/messages` and `DELETE /messages/{id}` call `capture_server_origin_mutation` (`server_origin.py:103-197`) for chats in the global scope. Workspace chats are never routed. A delete is routed only for rows the dataset holds (§5.2).
  1. It appends an accepted envelope (`chat.conversation` upsert or tombstone, `chat.message` append or tombstone) from the reserved device `server-origin`.
  2. A materializer writes the row (`api/core/Sync/v2/materializers/chat.py:18-256`, using `upsert_conversation_from_sync` and `append_message_from_sync`, `api/core/DB_Management/chacha/conversation_store.py:636`, `chacha/message_store.py:1675`).
  3. It records object state (revision, payload hash, cursor) and marks the envelope applied.
- **The log comes first.** If step 2 fails the envelope stays pending or failed, and repair replays it later. The projection is designed to be replayable from the envelope alone.
- **One fence per dataset.** Projection takes a dataset-wide lock (`api/core/DB_Management/Sync_DB.py:2579-2642`). An envelope is not projected while an earlier one is unapplied (`:2815-2859`), and no new envelope is accepted while a projected conflict is unresolved (`:7539-7561`).
- **Other devices** pull accepted envelopes in cursor order (`api/core/Sync/v2/service.py:10772-10824`) and push their own, which go through the same materializers.
- **A Sync-written message carries Sync metadata.** `append_message_from_sync` stamps `message_metadata.extra_json.sync_v2` (stable id, payload hash, revision) on the row (`message_store.py:1931-1958`).

## 2. Why history selection was refused

The refusal was deliberate. The H1 design says (`Docs/Design/2026-09-16-chatbook-h1-history-selection-design.md:165-167`):

> Existing sync routing must be respected, but cannot be used as permission to materialize H1 state natively. Before mutation, route through an adapter that supports retained selection/admission, or return `unsupported_history_capability` without writes. In H1 that adapter is not implemented.

So each versioned entry point checked for an active profile and stopped before any write:

| Entry point | Refusal on the base commit |
|---|---|
| Admission and settlement through `MessageCreate` | `api/api/v1/endpoints/character_messages.py:424-427` |
| Capture | `api/api/v1/endpoints/chat.py:8299-8300` |
| Legacy projection confirm | `chat.py:8340-8341` |
| Server-settled completion | `chat.py:4615-4616` |
| Client chat id (P3) | `api/api/v1/endpoints/character_chat_sessions.py:4642-4650, 4745-4747` |

Capture is a read, but it is also the capability handshake: a client that cannot capture sends no versioned write. That is why it was refused too.

Three properties of a versioned write make the ordinary route (log first, materializer second) unusable for it:

1. **It is validated and written in one owner transaction.** `append_selected_history_input` re-resolves the selection under the conversation lock and appends in the same transaction (`message_store.py:700-782`). A refusal (`stale_selection`, and P1's `history_branch_changed`) must leave nothing behind. An envelope accepted first could not be taken back, and an envelope that cannot be projected blocks every later one in the dataset.
2. **The materializer cannot write owner provenance.** `history_admission_json` is owner-only (`message_store.py:648-653`: "public CRUD and sync never call this seam"), and the envelope must not carry it.
3. **Sync metadata would break settlement.** The admission stores a digest of the input row that includes its metadata (`message_store.py:655-683, 778-781`). If the materializer stamped `sync_v2` metadata on the row afterwards, the digest would change and settlement would fail with `stale_parent` (`:904-908`).

The client-id refusal had a similar cause: the id is claimed, and the create fingerprint stored, by the conversation INSERT (P3). The Sync materializer upserts and knows nothing of the fingerprint.

## 3. What this change does

### 3.1 The mechanism: applied capture

The order is reversed for these writes. The owner write commits first, inside the Sync fence, and is then recorded in the log as an envelope that is already applied.

`capture_applied_server_origin_write` (`api/core/Sync/v2/server_origin.py`) does this in one Sync transaction (`SyncV2Store.projection_fence`):

1. **Take the dataset projection fence.** No other envelope can be accepted or projected for the dataset until the transaction ends.
2. **Check before writing.**
   - The dataset policy allows server-origin writes, and the domain is enrolled.
   - No id the write is about to create has Sync history that is not applied.
   - If one of those ids is not in the log yet, no projected conflict is unresolved. A pure replay skips this check, so an idempotent retry still answers while the dataset is blocked.
3. **Run the owner write.** It commits in ChaChaNotes. If it raises, nothing is recorded and the error reaches the client unchanged.
4. **Record what it wrote.** For each new object: one accepted envelope, its object state, and `apply_status = applied`. No materializer runs, so the row keeps its owner provenance and gets no Sync metadata. An object that already has Sync history is skipped.

The helper is general. The three callers are the message admission, the message settlement and the client-id create.

### 3.2 Envelopes

Each write emits the same envelope an ordinary Sync-routed write would, so another device needs no new handling.

| Write | Domain / operation | `object_id` | `parent_id` | Payload |
|---|---|---|---|---|
| Client-id create | `chat.conversation` / `upsert` | the client's chat id | none | The same fields as the ordinary Sync create: `title`, `root_id`, assistant identity, `state`, `topic_label`, `cluster_id`, `source`, `external_ref`, `rating`, `client_id`, `scope_type`, `workspace_id` |
| Admission | `chat.message` / `append` | the client's message id | the chat id | `conversation_id`, `parent_message_id` (the last selected message, or null), `sender`, `content`, `timestamp`, `client_id` |
| Settlement | `chat.message` / `append` | the client's reply id | the chat id | The same fields; `parent_message_id` is the admitted input |

Every envelope has `device_id = server-origin`, `object_revision = 1`, no base, `status = accepted`, `apply_status = applied`, and object state with the payload hash. The payload is read back from the stored row, so it carries what the owner transaction wrote: the parent and timestamp of a message, the normalized state of a chat.

**Not in any envelope:** the selection, the admission, `history_admission_json`, the create fingerprint, and reviewed legacy projections. These are the server owner's own state.

### 3.3 Ordering, idempotency and conflicts

- **Ordering.** A chat's envelope precedes its messages, and an input precedes its reply, because each is published before its request returns. Settlement also publishes the admitted input first if that input is not in the log (see the failure window below).
- **Idempotency.** The message id (or chat id) is the identity. A repeated admission, settlement or create runs the same idempotent owner write and finds the object already in the log, so it publishes nothing twice. The envelope id is derived from the object id, so a second insert could not create a duplicate either.
- **Later Sync writes.** Because object state is recorded, a later rename, message delete or chat delete finds its base and applies normally. A device push into the conversation applies normally.
- **Id collision with another device.** A device that pushes a message id the server already published gets a conflict at push time, as for any reused id. Nothing is projected and the dataset is not blocked.
- **Blocked dataset.** If a projected conflict is unresolved, new versioned writes return 503 `sync_server_origin_append_failed` before the owner write. This matches ordinary Sync-routed writes. The one conflict the server clears by itself is the stranded tombstone described in §5.2.
- **Busy fence.** If the fence cannot be taken (a lock timeout), the request returns the same 503 before the owner write.
- **Replay and repair.** `repair` re-runs applied envelopes. The materializer sees matching object state and does nothing, so the owner's row is not rewritten.

### 3.4 The one failure window

The owner commit and the Sync commit are two commits. If the process dies between them, or the Sync database fails at that moment, the message exists on the server and is not in the log.

- The client gets an error (503, or a dropped connection), never a success.
- **The retry repairs it.** Admission, settlement and client-id create are idempotent, and the same request sent again publishes the row it finds.
- Settlement publishes an unpublished input before the reply, even if the admission was never retried.
- A replayed client-id create publishes a chat whose first capture was lost.

What is not repaired: a client that never retries, or that retries with a new message id (which adds a second message). The first row then stays out of the log until something else publishes it. It can still be deleted (§5.2). §4.2 proposes the durable fix.

A server-origin write that fails after its owner commit is logged at warning level with the dataset and object ids.

### 3.5 A plain chat stays plain when Sync upserts it

A chat created with a client id and no assistant is stored as a plain chat. On the base branch, the first rename through Sync would have rewritten it as persona `sync-v2` (§5.3). The conversation materializer no longer invents that persona: an upsert that names no assistant is stored with none, for a chat that is already stored and for one first seen through Sync.

### 3.6 Capture still refuses a chat bound to a character

The H1 design gives character chats to server-settled turns: the server composes the character context and owns the input and result writes. Those turns are not available to a Sync v2 owner yet (§4.1), so for a chat with a character the handshake keeps answering 409 `sync_owner_unsupported`. This is the base behaviour for those chats, kept on purpose. It goes away with §4.1.

### 3.7 Behaviour without Sync v2

Unchanged. With no active profile the same code path runs the owner write directly and never opens the Sync store. Workspace chats are never routed through Sync.

---

## 4. Deferred work

### 4.1 Server-settled turns (`/chat/completions` with a selection)

**Blocked.** The request admits the user turn and saves the reply itself.

| Problem | Detail |
|---|---|
| No retry can publish a lost capture | The server generates the ids of the input chain and the reply (`message_store.py:833-838`; `chat.py:4661`), and a repeated request is refused with `selection_already_consumed` (`message_store.py:806-818`). The repair in §3.4 depends on an idempotent retry, which this path does not have. |
| The reply is saved inside the stream's finalizer | P2 will also save a partial reply there after a disconnect, inside a cancelled task (D7 §4.2 "Uncertainty"). A capture cut short after a committed save is exactly the window in §3.4, with nothing to close it. |
| Rows that Sync v2 M1 cannot carry | Inputs can have several images, tool rows and tool-call metadata. The `chat.message` payload is text only. |
| `persona_not_found` on plain chats | Fixed for new chats (§5.3). A chat stored as persona `sync-v2` before the fix still returns 404 from `/chat/completions` with its conversation id until it is repaired. |

**Options**

- **A. Applied capture at both points, no repair.** Small, but a lost capture is permanent and can later block the dataset (§5.2). Not acceptable.
- **B. Applied capture plus the outbox in §4.2.** The outbox row is written in the owner transaction, so a lost capture is published by the next request or by a sweep. Needs §4.2 first.
- **C. Route the turn through envelopes.** Would need an envelope that can be refused after acceptance. Sync v2 has no such state.

**Recommendation: B**, after §4.2 and the repair of existing placeholder rows (§5.3), and after P2 lands so the settlement code is stable. The wiring is then small: wrap `accept_history` (`api/core/Chat/chat_service.py:4147-4166`) and the settlement closure (`chat.py:4657-4667`), refuse images before admission, and remove the capture refusal for character chats.

### 4.2 A durable record of unpublished rows (transactional outbox)

**Problem.** §3.4 relies on a client retry. Three cases have none: a client that goes away, server-settled turns (§4.1), and the unversioned completion save (§5.1).

**Proposal.** A small table in ChaChaNotes, written in the same transaction as the message or conversation whenever the owner has an active profile:

```
chat_sync_outbox(object_domain, object_id, conversation_id, created_at)
```

- The applied capture deletes the outbox row after it records the envelope.
- A row that survives is published by the next Sync-routed request for that owner, and by a periodic sweep. Publication is idempotent (§3.3).
- This closes the window for every path, and it is what makes §4.1 safe.

**Cost.** One schema migration on both backends, and a drain step. It was left out of this change because it needs a migration while P1–P3 are also in flight, and because the client-managed turn already repairs itself.

### 4.3 Conversations that predate the profile (enrollment)

**Problem.** Nothing publishes chats that existed before the profile was activated. Their new messages are published with a parent and a conversation the other device has never seen. This is also true of ordinary Sync-routed writes on the base branch.

**Options**

- **A. Leave it.** New turns in an old chat reach the other device without their history.
- **B. Enroll on first write.** Publish the conversation and every live message, parents first, the first time a Sync-routed write touches the chat. Complete, but a long chat is thousands of envelopes in one request.
- **C. An explicit backfill.** A job the user starts ("Sync my existing chats"), using the outbox from §4.2 as its queue.

**Recommendation: C**, built on §4.2. The H1 design assigned enrollment to "H4"; this is that work.

Two things the backfill has to decide, both found while fixing §5.2:

- A chat that is not enrolled can now be deleted, and the delete publishes nothing. Its rows stay in the database as soft-deleted rows with no Sync history.
- A device can enrol a row the server already holds by pushing the same id with the same content. The adoption does not look at `deleted`, so a row deleted on the server before the device's push stays deleted there while the log, and every other device, has it live. The backfill should either skip such rows or publish their tombstones.

### 4.4 Messages from other devices are unversioned

**Problem.** A message the Chatbook client pushes has no owner provenance. The history snapshot treats it as legacy data (`message_store.py:284-305`):

- A chat written only by the device, as one parent chain, is read as a graph. The WebUI can continue it without a review.
- Device messages without parent ids, or a device message whose parent is a WebUI message, make the chat `legacy_review_required`. The user must confirm a path (the review this change now allows under Sync) before the next WebUI send, and again after the device appends more.

So two devices can take turns in one conversation, but each hand-over from the device to the WebUI costs a review. All three cases are covered by tests (§7).

**Options**

- **A. Leave it.** Correct, but clumsy.
- **B. Mark a pushed message that names its parent as graph data.** The materializer would write the interpretation tag only (`{"interpretation": {"kind": "parent_graph_v1"}}`), not an admission. H1 forbids Sync from writing owner authority, so this needs the H1 owner's agreement. It also needs the Chatbook client to always send `parent_message_id`.
- **C. Loosen the classifier** to accept an unversioned row whose parent is versioned. This weakens the check for real legacy data.

**Recommendation: B**, as its own change with its own contract tests.

### 4.5 Character routes and edits

`complete-v2`, `completions/persist` (`character_chat_sessions.py:6420, 8505`) and message edit (`character_messages.py:1014-1020`) stay refused under Sync v2. For D7's CM-01, Edit & resend is an admission at the edited message's parent, which this change supports. An in-place edit is still refused. No change is proposed here.

---

## 5. Defects found on the base branch

These existed before this change. §5.2 and §5.3 are fixed on `fix/sync-v2-unenrolled-delete` (#3181, #3182). §5.1 is open, for the reason given there.

### 5.1 An unversioned completion save publishes nothing (open, #3182)

With an active profile, `POST /chat/completions` with `save_to_db: true` and no selection writes the system, user and assistant rows directly. No envelope and no object state are written. Verified with a probe: three rows, an empty log. `chat.py` has no Sync capture on this path; only the character routes are guarded.

Effect: other devices never see these messages. Since §5.2 they can be deleted like any other row, so they are no longer a risk to the dataset.

Neither of the two possible fixes was shipped as a bug fix.

| Fix | Can it lose or duplicate data? | Why it was not shipped |
|---|---|---|
| Publish the rows with an applied capture | Yes, both | The server generates the message ids, so the request is not idempotent. A capture lost after the owner commit is never published, and the client's retry saves a second set of messages. The reply is saved in the stream's finalizer, where nothing retries a capture that was cut short. Rows with images or tool calls do not fit the M1 `chat.message` payload. This is §4.1 again and needs the outbox (§4.2). |
| Refuse the save with 409, before the provider call and any write | No | It turns a send that works today into an error. The UI package sends `save_to_db: true` without a selection whenever no history turn is in play (`ui/models/index.ts:146-149`, where the default is `!temporaryChat`; `ui/hooks/chat-modes/chatModePipeline.ts:716` forces `false` only for a client-managed turn), and so can any OpenAI-compatible client. Which of those flows a Sync v2 owner reaches was not audited. A refusal first needs a signal the client can read before it sends (§6 Q2). |

Refusal is the only one of the two that cannot lose or duplicate data, so it is the fix to ship once the clients can handle it. The place is `chat.py`, before `_build_context_and_messages_compat`: `build_context_and_messages` (`api/core/Chat/chat_service.py`) both decides `should_persist` and writes the user turn. Publication becomes possible with §4.2.

### 5.2 Deleting a row Sync never saw blocked the dataset (fixed, #3181)

**The defect.** With an active profile, `DELETE /messages/{id}` (or deleting the chat) for a message with no object state appended a tombstone, which the materializer rejected as `message_base_conflict` / `missing_server_message` (`materializers/chat.py:363-380`). The request returned 503, the conflict stayed unresolved, and every later Sync-routed write for that user returned 503 `sync_server_origin_append_failed`. A chat with no object state could not be deleted at all: the store refuses a `chat.conversation` tombstone without a base at insert (`_validate_envelope_contract` in `api/core/DB_Management/Sync_DB.py`), so that request returned 503 without blocking anything.

Rows this applied to: any chat that predates the profile (§4.3), the rows from §5.1, and a row left by the window in §3.4.

**The rule now.** `delete_unenrolled_server_origin_objects` (`api/core/Sync/v2/server_origin.py`) checks each row a delete names, inside the dataset projection fence.

| The dataset holds | The delete |
|---|---|
| No current head and no object state for the row | Direct: the same soft delete as without a profile. Nothing is appended. |
| Anything else | A tombstone through the log, as before. |

- **Message delete.** One row.
- **Chat delete.** The chat and its live messages are checked in one pass. Unpublished messages are deleted directly, in one transaction. Published messages are tombstoned in order. The chat goes last: by a tombstone if it is published, directly if it is not.
- **A chat with both kinds of row.** Only the published rows produce envelopes. If an unresolved conflict blocks the dataset and any row needs a tombstone, the request is refused before anything is deleted.
- **A row whose envelope is accepted but not applied** counts as published, because a pull delivers pending envelopes. Its delete is refused by the head check at insert and appends nothing.
- **The chat row of a published chat** is not touched by a direct message delete. Without a profile a message delete also bumps the chat's version; with one, only the log changes a published chat.
- **A client-private dataset** cannot be written by the server front end, so the delete is refused there as before (409).

**Why delete directly, and not publish first.** The alternative was to publish the row with an applied capture and then tombstone it.

| | Direct delete | Publish, then tombstone |
|---|---|---|
| What other devices receive | Nothing. They never had the row. | The content of the deleted message, then its tombstone. |
| The log | Unchanged. | Keeps the deleted content. The log is append-only. |
| A chat that predates the profile | Deleted with no envelope. | Needs the chat and every message published first, two envelopes per row in one request. This is enrollment (§4.3) as a side effect of a delete. |
| Rows M1 cannot carry (images, tool calls) | Deleted. | Cannot be published, so they still could not be deleted. |
| While another conflict blocks the dataset | Works. Nothing is appended. | Refused. |
| A device that holds a copy by another route | Not told. See route 2 below. | Told, by the tombstone. |

**Can a device hold a copy of such a row?** Three routes were checked.

1. *Through this dataset.* Only if an envelope for the row exists. A pull also delivers envelopes that are accepted but not applied, which is why the rule asks for no current head and not only for no object state.
2. *Under the same id, without Sync.* Yes. `append_message_from_sync` (`api/core/DB_Management/chacha/message_store.py`) adopts a stored row that has no Sync metadata when a device pushes the same id with the same content, so a device can enrol a row the server already has. If the server deleted that row directly first, the device's later push is applied and the other devices receive the message, while the server row stays deleted and object state says it is live. Nothing conflicts and the dataset is not blocked. The same already happens to any row deleted before the profile existed, because the adoption does not look at `deleted`. Pinned by `test_a_device_that_publishes_its_own_copy_of_a_directly_deleted_row_does_not_block_the_dataset`. What a first sync should do with rows deleted on one side belongs to §4.3.
3. *Through another dataset of the same user.* This one is from reading the code; it was not run. Server-origin writes go to the default personal dataset only, and the check is against that dataset. A row that a device pushed into another dataset is deleted on the server with no tombstone there. Before the fix its delete blocked the default dataset. Workspace chats were already deleted directly.

**A dataset that is already blocked** recovers on the next server-origin write: a send, a chat create or rename, a delete, a note write. The stranded conflict is resolved with `skip` when all of these hold.

- The conflict is `chat.message` / `message_base_conflict` with reason `missing_server_message`.
- Its envelope is a tombstone from `server-origin` with no base, accepted, with `apply_status = conflict`.
- The object still has no object state.

`skip` marks the envelope `superseded` (`sync_conflict_skipped`), restores the object's head and dismisses the conflict with the note `server_origin_tombstone_of_object_without_sync_history`. Nothing is lost by it:

- no device received the tombstone, because a pull withholds conflicted and superseded envelopes and stops at the blocking one;
- nothing was projected from it;
- its request returned 503, so no client was told the delete happened.

The row it named is left as it is, and deleting it again now works. Every other conflict is left for review, including a device's own tombstone of an unknown message, which is that device's claim.

A device push does not trigger the recovery. A user who writes only from devices can clear the conflict by hand:

1. `GET /api/v1/sync/conflicts?dataset_id=…&status=unresolved`
2. `POST /api/v1/sync/conflicts/resolve` with `{"dataset_id": …, "device_id": …, "resolutions": [{"conflict_id": …, "action": "skip"}]}`

**Not changed.** Both verified with probes.

- A device that pushes a tombstone for a message the server does not have still leaves the same projection conflict, and it blocks the dataset until someone resolves it. The device is told: the push result lists the conflict. The store accepts a `chat.message` tombstone without a base at insert, unlike a `chat.conversation` or `notes.note` one; refusing it there would close this route.
- A server-origin `notes.note` tombstone for a note with no object state is refused at insert ("tombstones require base metadata"). Nothing is blocked, but such a note cannot be deleted through the API while the profile is active. Whether notes that predate the profile are enrolled when it is created was not checked.

### 5.3 A Sync upsert stored a plain chat as persona `sync-v2` (fixed for new upserts, #3182)

**The defect.** The conversation materializer filled in `assistant_kind = persona`, `assistant_id = sync-v2` when a payload had no assistant identity. No such persona exists, so persona admission (`require_current_persona`, `api/core/Persona/conversation_admission.py`) answered 404 `persona_not_found` for the chat, on `POST /chat/completions` with its conversation id among other routes.

- On the base branch this also rewrote an existing plain chat on any later upsert, such as a rename. The WebUI reads the stored identity to choose how to send (`ui/hooks/chat/effective-assistant-state.ts`), so a renamed chat would have been treated as a persona chat for a persona that does not exist. This is from reading the code; it was not run in a browser.
- `feat/chat-sync-v2-native-history` fixed that case only: an already plain chat stayed plain.

**Now.** An upsert is the whole chat. One that names no assistant is stored with none, whether the chat is new or already stored. The special case for an already plain chat is gone with the placeholder. A payload that names an assistant is stored as named.

- A chat created by an id-less `POST /chats/`, or by a device push with no assistant, is a plain chat.
- An upsert without an assistant on a chat that has one clears it. It did before too, to the placeholder. The server's own upserts are built from the stored row and always carry the identity. A device must send it as well.

**Rows that already hold the placeholder** are not rewritten by this change.

- They are not read as plain. The stored identity is read in many places: persona admission, the completion context in `chat_service.py`, the chat response, the WebUI's choice of send path. Admission is strict on purpose and never falls back to a plain assistant. Reading the placeholder as plain would need an exception in each of them.
- Such a row becomes plain on its next upsert that names no assistant, for example a rename pushed by the device that created it.
- A rename through the server API does not repair it. That payload is built from the stored row, so it carries `persona` / `sync-v2`, which is stored as named.
- To repair the rows in one step, run this on the user's ChaChaNotes database after checking that the user has no persona with that id:

```sql
UPDATE conversations
   SET assistant_kind = NULL, assistant_id = NULL, persona_memory_mode = NULL
 WHERE assistant_kind = 'persona' AND assistant_id = 'sync-v2' AND character_id IS NULL
   AND NOT EXISTS (SELECT 1 FROM persona_profiles WHERE id = 'sync-v2');
```

Object state is not affected by the repair: it holds the hash of the payload, not of the row. A migration that does the same is the durable fix. It was left out for the reason given in §4.2: a schema version bump while P1–P3 are in flight.

The client-managed turn sends no conversation id to `/chat/completions`, so it never met this defect. A server-settled turn does (§4.1).

---

## 6. Questions for the owner

1. **§4.2 outbox.** Approve a small ChaChaNotes table and migration to make publication durable? It is the prerequisite for server-settled turns under Sync v2.
2. **§4.1 timing.** P5 moves every chat to server-settled turns. Should P5 wait for §4.2 and §4.1, or should P5 keep the client-managed turn for Sync v2 owners until they land? The second needs a signal the client can read before it dispatches. The 409 on `/chat/completions` comes too late: the WebUI has already marked the turn as dispatched and would keep an "unknown outcome" record. The capture response is the natural place for that signal.
3. **§4.4.** Is it acceptable for the Sync materializer to mark a pushed message as graph data when it names its parent? Does the Chatbook client send `parent_message_id` on every message?
4. **§5.1.** §5.2 and §5.3 are fixed (#3181, #3182). For the unversioned completion save, should the server refuse it for Sync v2 owners once the client has the signal from question 2, or leave it until §4.2 can publish it?
5. **§5.3.** Approve a small migration that clears the `sync-v2` placeholder from existing chats? Until then those chats are repaired by hand or by their next upsert from a device.

## 7. Tests

- `tldw_Server_API/tests/Sync/test_sync_v2_applied_server_origin_capture.py`: the helper. One applied envelope and object state per object, in order; a refused write records nothing and its own error is passed through; replay, including while the dataset is blocked; a lost capture and a failed Sync commit repaired by the retry; a partly recorded capture rolls back as a whole; refusals before the write (blocked dataset, unavailable fence, unapplied history on a claimed id, client-private policy, missing dataset, unsupported adapter version); tombstone and pull afterwards; full repair leaves the row untouched.
- `tldw_Server_API/tests/Sync/test_sync_v2_native_history_capture.py`: the routes, with a real Sync service and a separate materializer handle on the same database, as the service factory wires it. Capture, admit and settle; capture of a character chat still refused; a second turn; a second device pulls and applies the envelopes into its own database; replays; refusals; both repair paths; a blocked dataset; image refusal; chat delete; a rename that keeps a plain chat plain; a device push into the conversation (and the review it then requires); a turn that continues a chat the device wrote; a device push that reuses a published id; legacy review; the client-id create rules (201, 200 replay, rename then replay, 409, 410, lost capture, concurrent duplicates, character chat, workspace chat); and the same turn with no profile.
- `tldw_Server_API/tests/Sync/test_sync_v2_chat_materializer.py`: an upsert that names no assistant keeps an existing plain chat plain, creates a plain chat, and clears a stored placeholder; one that names an assistant is stored as named (§5.3).
- `tldw_Server_API/tests/Sync/test_sync_v2_unenrolled_delete.py` (§5.2): the delete routes and the helper. A message and a chat Sync never saw are deleted with no envelope, and later writes still succeed; a row whose capture was lost; chats with both kinds of row, in both directions; a published delete still tombstones and reaches a second device; a row with a pending envelope is not deleted directly; a blocked dataset (direct delete still works, a needed tombstone refuses the whole request); stale versions; what follows a direct delete (a retried admission, a device that publishes its own copy); recovery of a dataset blocked by the old behaviour, from an ordinary write, a retried delete and a versioned write, and the conflicts it must leave alone; and the same deletes with no profile.
- `tldw_Server_API/tests/Sync/test_sync_v2_unenrolled_delete_postgres_contract.py`: the same routes with the chat database on PostgreSQL and with the Sync store on PostgreSQL.
- `tldw_Server_API/tests/Chat_NEW/integration/test_history_selection_api.py`: the Sync owner now captures; the server-settled completion is still refused before any write.

## 8. Merge notes

- **P1 (#3171)** edits the same lines of `character_messages.send_message`. Resolve by adding `history_branch=message_data.tldw_history_branch` to the `append_selected_history_input` call inside `history_write`, and `**exc.details` to the 409. One closure serves both the Sync and the non-Sync path, so the leaf check then applies to Sync v2 owners too. A `history_branch_changed` refusal is raised inside the owner write, so it records nothing.
- **OpenAPI fingerprint.** `POST /chats/` changed its description and its 409 description; nothing else in the schema changed. Regenerate the fingerprint after merging with any other PR that touches it.
- **`sync_client_chat_id_unsupported`** no longer exists. P4 needs no special case for Sync v2 owners.
- **`fix/sync-v2-unenrolled-delete`** (§5.2, §5.3) is based on `feat/chat-sync-v2-native-history`. It changes `character_messages.delete_message`, `character_chat_sessions.delete_chat_session`, `server_origin.py` and the conversation materializer. No route, schema or docstring changed, so the OpenAPI fingerprint and the privilege snapshot are the same.
