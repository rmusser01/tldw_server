# D7: native history for Sync v2 owners

- **Date:** 2026-10-04
- **Status:** §3 is implemented on `feat/chat-sync-v2-native-history`. §5.2 and §5.3 are fixed on `fix/sync-v2-unenrolled-delete` (#3181, #3182). The migration in §5.3 and the tombstone rule in §5.4 are on `fix/sync-v2-placeholder-and-device-tombstone`. §5.1 is open and waits for §4.2. §4 is proposed; §6 records what the owner has decided and what is still open.
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

**Four defects found on the base branch** are listed in §5. The second (a delete that blocked the dataset), the third (the placeholder persona, including the rows that already hold it) and the fourth (a device's delete that blocked the dataset) are fixed. The first is open: its rows are no longer a risk to the dataset, but other devices still do not receive them.

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
| `persona_not_found` on plain chats | Fixed (§5.3): new chats are stored plain, and a migration repairs the chats stored as persona `sync-v2` before the fix. |

**Options**

- **A. Applied capture at both points, no repair.** Small, but a lost capture is permanent and can later block the dataset (§5.2). Not acceptable.
- **B. Applied capture plus the outbox in §4.2.** The outbox row is written in the owner transaction, so a lost capture is published by the next request or by a sweep. Needs §4.2 first.
- **C. Route the turn through envelopes.** Would need an envelope that can be refused after acceptance. Sync v2 has no such state.

**Recommendation: B**, after §4.2, and after P2 lands so the settlement code is stable. The wiring is then small: wrap `accept_history` (`api/core/Chat/chat_service.py:4147-4166`) and the settlement closure (`chat.py:4657-4667`), refuse images before admission, and remove the capture refusal for character chats.

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

Three things the backfill has to decide, found while fixing §5.2 and §5.4:

- A chat that is not enrolled can be deleted on the server, and that delete publishes nothing (§5.2). Its rows stay in the database as soft-deleted rows with no Sync history.
- A device can enrol a row the server already holds by pushing the same id with the same content. The adoption does not look at `deleted`, so a row deleted on the server before the device's push stays deleted there while the log, and every other device, has it live. The backfill should either skip such rows or publish their tombstones.
- The other direction is covered: a device that deletes a message the server holds outside Sync pushes a tombstone, and the server deletes its copy (§5.4).

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

These existed before this change. §5.2 and §5.3 are fixed on `fix/sync-v2-unenrolled-delete` (#3181, #3182); the migration in §5.3 and §5.4 are fixed on `fix/sync-v2-placeholder-and-device-tombstone`. §5.1 is open, for the reason given there.

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

Since §5.4 there is a third option. A tombstone for such a row is now applied instead of rejected, so the server could publish a tombstone alone, which carries the id and no content, for a message it deletes directly. That would also reach a device holding a copy by route 2. It was not done: it reverses the decision above for rows nobody else has, adds one envelope per message of a chat that predates the profile, and the chat's own tombstone is still refused when appended (§5.4). It is a question for the owner (§6).

**Can a device hold a copy of such a row?** Three routes were checked.

1. *Through this dataset.* Only if an envelope for the row exists. A pull also delivers envelopes that are accepted but not applied, which is why the rule asks for no current head and not only for no object state.
2. *Under the same id, without Sync.* Yes. `append_message_from_sync` (`api/core/DB_Management/chacha/message_store.py`) adopts a stored row that has no Sync metadata when a device pushes the same id with the same content, so a device can enrol a row the server already has. If the server deleted that row directly first, the device's later push is applied and the other devices receive the message, while the server row stays deleted and object state says it is live. Nothing conflicts and the dataset is not blocked. The same already happens to any row deleted before the profile existed, because the adoption does not look at `deleted`. Pinned by `test_a_device_that_publishes_its_own_copy_of_a_directly_deleted_row_does_not_block_the_dataset`. What a first sync should do with rows deleted on one side belongs to §4.3.
3. *Through another dataset of the same user.* This one is from reading the code; it was not run. Server-origin writes go to the default personal dataset only, and the check is against that dataset. A row that a device pushed into another dataset is deleted on the server with no tombstone there. Before the fix its delete blocked the default dataset. Workspace chats were already deleted directly.

**A dataset that is already blocked** by this defect recovers on the next server-origin write (a send, a chat create or rename, a delete, a note write) and on the next push from any device. `SyncV2Service.settle_stranded_tombstones` resolves the stranded conflict with `skip` when all of these hold.

- The conflict is `chat.message` / `message_base_conflict` with reason `missing_server_message`.
- Its envelope is a tombstone from `server-origin` with no base, accepted, with `apply_status = conflict`.
- The object still has no object state.

`skip` marks the envelope `superseded` (`sync_conflict_skipped`), restores the object's head and dismisses the conflict with the note `server_origin_tombstone_of_object_without_sync_history`. Nothing is lost by it:

- no device received the tombstone, because a pull withholds conflicted and superseded envelopes and stops at the blocking one;
- nothing was projected from it;
- its request returned 503, so no client was told the delete happened.

The row it named is left as it is, and deleting it again now works. A device's stranded tombstone is settled differently: it is applied (§5.4). Every other conflict is left for review.

A conflict can still be cleared by hand:

1. `GET /api/v1/sync/conflicts?dataset_id=…&status=unresolved`
2. `POST /api/v1/sync/conflicts/resolve` with `{"dataset_id": …, "device_id": …, "resolutions": [{"conflict_id": …, "action": "skip"}]}`

**Not changed.** Verified with a probe: a server-origin `notes.note` tombstone for a note with no object state is refused at insert ("tombstones require base metadata"). Nothing is blocked, but such a note cannot be deleted through the API while the profile is active. Whether notes that predate the profile are enrolled when it is created was not checked. See §5.4 for the same contract on the device side.

### 5.3 A Sync upsert stored a plain chat as persona `sync-v2` (fixed, #3182)

**The defect.** The conversation materializer filled in `assistant_kind = persona`, `assistant_id = sync-v2` when a payload had no assistant identity. No such persona exists, so persona admission (`require_current_persona`, `api/core/Persona/conversation_admission.py`) answered 404 `persona_not_found` for the chat, on `POST /chat/completions` with its conversation id among other routes.

- On the base branch this also rewrote an existing plain chat on any later upsert, such as a rename. The WebUI reads the stored identity to choose how to send (`ui/hooks/chat/effective-assistant-state.ts`), so a renamed chat would have been treated as a persona chat for a persona that does not exist. This is from reading the code; it was not run in a browser.
- `feat/chat-sync-v2-native-history` fixed that case only: an already plain chat stayed plain.

**New upserts.** An upsert is the whole chat. One that names no assistant is stored with none, whether the chat is new or already stored. The special case for an already plain chat is gone with the placeholder. A payload that names an assistant is stored as named.

- A chat created by an id-less `POST /chats/`, or by a device push with no assistant, is a plain chat.
- An upsert without an assistant on a chat that has one clears it. It did before too, to the placeholder. The server's own upserts are built from the stored row and always carry the identity. A device must send it as well.

**Rows that already hold the placeholder** are repaired by a migration (owner decision, §6): ChaChaNotes SQLite v75 → v76, PostgreSQL v79 → v80.

```sql
UPDATE conversations
   SET assistant_kind = NULL,
       assistant_id = NULL,
       persona_memory_mode = NULL,
       assistant_startup_json = NULL
 WHERE assistant_kind = 'persona'
   AND assistant_id = 'sync-v2'
   AND character_id IS NULL
   AND NOT EXISTS (
       SELECT 1 FROM persona_profiles AS profile
        WHERE profile.id = 'sync-v2'
          AND profile.user_id = conversations.client_id
   )
```

- **Which rows.** Only the exact placeholder: both values are compared exactly. A persona that is only *named* `sync-v2` has another id and is not matched, and neither is `sync-v2-custom` or `SYNC-V2`. A chat whose owner has, or had, a persona profile with id `sync-v2` (a deleted profile counts) is that owner's own binding and is left alone; another owner's profile with that id does not protect it. Chats in trash are repaired too, so a restored chat is plain.
- **What changes on the row.** The four columns that make up a binding. A plain chat cannot keep a memory mode (`_normalize_conversation_assistant_identity` rejects it), and startup provenance belongs to a binding the chat never really had. `character_id` is NULL by the predicate. `version` and `last_modified` do not move: nothing was edited, and under Sync the row version is the object revision.
- **Other tables.** None holds a chat's binding. A Workspace startup receipt holds a digest of it, and a replay compares that digest with the current row. The placeholder was written by `upsert_conversation_from_sync`, which invalidates the chat's receipts in the same transaction when the binding changes, and an invalidated receipt stays invalid. So the repair does not bring a receipt back, and it leaves the table alone.
- **A turn in flight across the upgrade.** From reading the code, not run: a history selection binds a digest of the stored identity (`storage_context_digest`, `chacha/message_store.py`). One captured on such a chat before the upgrade no longer matches after it and is refused as stale.
- **Readers.** After the repair the row is a plain chat for every reader (persona admission, the completion context, the chat response, the WebUI's send path), with no special case in any of them.
- **PostgreSQL.** One database holds every owner's chats, and `conversations` has a forced tenant policy. A role that bypasses row-level security (the documented superuser setup) sees all rows. A table owner that does not bypass it would repair only the rows its session sees and then record the version. So the migration takes an ACCESS EXCLUSIVE lock on the table, lifts the forced policy for the owner, repairs, and restores the policy before the version bump, as the PostgreSQL migrations v56 to v60 do for `notes`. A role that is neither owner nor exempt is refused with a `SchemaError`.
- **Idempotent.** A second run matches nothing. The version is bumped in the same transaction, and a failure rolls back both.

**No envelope is appended, and Sync state is untouched.** The repair is a server-local correction of a server-local artefact.

- The placeholder was invented while projecting. The envelope that created such a chat names no assistant, on the server-origin route and on a device push alike. Clearing the row makes the projection agree with the log.
- Object state holds the payload hash and cursor of the last envelope, not anything read from the row. The next upsert finds its base as before.
- The placeholder did reach the log in one way. A rename through the server API was built from the stored row, so that envelope carries `persona` / `sync-v2`, and a device that received it can echo it. A corrective envelope cannot come from a schema migration: the migration has no Sync store (another file on SQLite, another database on PostgreSQL), cannot take the dataset fence, and the dataset may be blocked or client-private. It is not needed either: the conversation materializer now reads that identity in a payload as no assistant, under the same rule as the migration (`ConversationStore.is_retired_sync_placeholder_assistant`), so a replay or an echo projects a plain chat, and the next server-origin upsert of the chat publishes the corrected payload on the current base.

The client-managed turn sends no conversation id to `/chat/completions`, so it never met this defect. A server-settled turn did (§4.1).

### 5.4 A device's tombstone of a message the server does not have blocked the dataset (fixed)

**The defect.** Verified with a probe. A device that pushed a `chat.message` tombstone for a message the dataset holds no state for got `message_base_conflict` / `missing_server_message`, the same rejection as §5.2. The conflict blocked the dataset, and no other device received the delete. A device has such a message when it predates the profile, or when it was created and deleted offline.

**Owner decision.** Do not reject it at insert: "that would indicate a sync issue and blocking it wouldn't help fix things, at least with a delete it'd be deleted both sides."

**The rule.** The end state the tombstone asks for is "gone". When the dataset has no object state for the message, that already holds, so `ChatMessageMaterializer` applies the tombstone.

| For that id the server holds | Result |
|---|---|
| Nothing | Applied as a no-op. |
| A row with no Sync history (a copy kept outside Sync) in a chat this owner holds | The row is soft-deleted, and the tombstone is applied. |
| Object state | Unchanged: the base must match, otherwise it is a conflict. |
| A row that carries Sync history, but no object state in this dataset | Still a conflict (`missing_server_message`). The row was projected from other Sync history and is not this tombstone's to delete. |

- The envelope is accepted with `apply_status = applied`. No conflict is recorded and the dataset is not blocked.
- Object state records the message as tombstoned: the envelope's revision, hash and cursor, `deleted = true`.
- Only chats the projection's owner holds are searched. The lookup is by message id, and on PostgreSQL one database holds every owner's messages; with no object state tying the id to this owner, the chat's owner is the only thing that does.
- The rule does not depend on where the tombstone came from. The server's delete routes do not produce one for such a row (§5.2).
- The rule is for a tombstone with no base, which is what a device sends for a message it never synced. One that names a base while the dataset has no head for the message claims server history that does not exist. The head check refuses it at push as `stale_base_state`, as before. That is a push conflict, so nothing is blocked, and the same delete sent without a base is applied.

**Convergence.** Every replica that applies the tombstone under this rule ends at "gone", whatever it started with.

| Replica | Before | After the tombstone |
|---|---|---|
| The device that pushed it | Already deleted locally | Its push is accepted and applied. |
| The server | No row, or a copy outside Sync | No live row. |
| A third device that never had the message | Nothing | A pull delivers the tombstone; applying it is a no-op. |
| A third device that kept a copy outside Sync | A live copy | A pull delivers the tombstone; the same rule deletes the copy. |
| A third device that created the same id offline and pushes it later | A live, unpublished message | Its append has no base, and the tombstone is the object's head, so the head check refuses it as a push conflict. That is not a projection conflict and blocks nothing. The device pulls the tombstone and deletes its copy. |

The delete reaches the other devices because the tombstone is an ordinary applied envelope. Under the old behaviour it was a conflicted one, which a pull withholds. Nothing is resurrected: object state is tombstoned, and an append that got past the head check would meet `message_deleted_conflict` in the materializer.

The two "third device" rows that hold a copy are tested with a second projection that runs this materializer on its own database. A device client that projects envelopes with other code has to follow the same rule; a client that rejects the tombstone instead keeps its copy and diverges.

**A dataset that is already blocked** by a device's tombstone recovers on the next push from any device and on the next server-origin write. `SyncV2Service.settle_stranded_tombstones` applies the stranded tombstone under the rule above, so the delete the device asked for is honoured and delivered, not discarded. Its conflict record is closed as `resolved` with the action `apply_on_retry`. A device that pushes its stranded tombstone again also gets it applied. The server's own stranded tombstone is still resolved with `skip` (§5.2), and every other conflict is left for review.

**Conversations and notes: not changed.** Verified with probes at the service level, for a device that pushes a tombstone of a chat or a note the server does not have:

| The tombstone | Today |
|---|---|
| Has no base | `push` raises `SyncStoreError` ("tombstones require base metadata") from the store contract. The whole call fails, including the envelopes after it in the batch. |
| Has a base | A push conflict (`stale_base_state`): the head check finds no head. |

Neither blocks the dataset, which is why this was less urgent than messages. Under the owner's reasoning both are still unhelpful: the delete never reaches the other devices, and the device gets an error for a state that already holds. The recommendation is the same rule. It was left out because it is four changes that only work together:

1. `_validate_envelope_contract` (`api/core/DB_Management/Sync_DB.py`): accept a tombstone without a base for `chat.conversation` and `notes.note`.
2. `_require_expected_current_head`: accept a tombstone that carries a base when the object has no head. This check is shared by every domain.
3. `ChatConversationMaterializer`: with no object state, a tombstone that carries a base is a `missing_server_object` conflict, which blocks the dataset, and one without a base raises ("Conversation not found for Sync v2 tombstone.") and leaves a *failed* envelope, which holds up every later projection in the dataset. Both need the no-op.
4. `NotesMaterializer`: the same two paths ("Note not found for Sync v2 tombstone."), and the note's attachments, links and tasks need checking against a tombstoned state that no upsert preceded.

Doing 1 and 2 without 3 and 4 would turn an error that blocks nothing into one that blocks the dataset. Separately, a contract violation in one envelope should become a rejection of that envelope instead of raising out of `push`.

---

## 6. Questions for the owner

1. **§4.2 outbox.** Approve a small ChaChaNotes table and migration to make publication durable? It is the prerequisite for server-settled turns under Sync v2.
2. **§4.1 timing.** P5 moves every chat to server-settled turns. Should P5 wait for §4.2 and §4.1, or should P5 keep the client-managed turn for Sync v2 owners until they land? The second needs a signal the client can read before it dispatches. The 409 on `/chat/completions` comes too late: the WebUI has already marked the turn as dispatched and would keep an "unknown outcome" record. The capture response is the natural place for that signal.
3. **§4.4.** Is it acceptable for the Sync materializer to mark a pushed message as graph data when it names its parent? Does the Chatbook client send `parent_message_id` on every message?
4. **§5.1.** Decided: wait. The unversioned completion save is not refused now; it waits for the outbox (§4.2).
5. **§5.3.** Decided: migrate. Shipped (§5.3).
6. **§5.4, messages.** Decided: a device's tombstone of a message the server does not have is not rejected and must not block. Shipped (§5.4).
7. **§5.4, conversations and notes.** Apply the same rule to a device's tombstone of a chat or a note the server does not have? It needs the four changes listed in §5.4.
8. **§5.2.** Should the server's own delete of a message Sync never saw now publish a tombstone alone? It would reach a device that holds a copy outside Sync, at the cost of one envelope per such message.

## 7. Tests

- `tldw_Server_API/tests/Sync/test_sync_v2_applied_server_origin_capture.py`: the helper. One applied envelope and object state per object, in order; a refused write records nothing and its own error is passed through; replay, including while the dataset is blocked; a lost capture and a failed Sync commit repaired by the retry; a partly recorded capture rolls back as a whole; refusals before the write (blocked dataset, unavailable fence, unapplied history on a claimed id, client-private policy, missing dataset, unsupported adapter version); tombstone and pull afterwards; full repair leaves the row untouched.
- `tldw_Server_API/tests/Sync/test_sync_v2_native_history_capture.py`: the routes, with a real Sync service and a separate materializer handle on the same database, as the service factory wires it. Capture, admit and settle; capture of a character chat still refused; a second turn; a second device pulls and applies the envelopes into its own database; replays; refusals; both repair paths; a blocked dataset; image refusal; chat delete; a rename that keeps a plain chat plain; a device push into the conversation (and the review it then requires); a turn that continues a chat the device wrote; a device push that reuses a published id; legacy review; the client-id create rules (201, 200 replay, rename then replay, 409, 410, lost capture, concurrent duplicates, character chat, workspace chat); and the same turn with no profile.
- `tldw_Server_API/tests/Sync/test_sync_v2_chat_materializer.py`: an upsert that names no assistant keeps an existing plain chat plain, creates a plain chat, and clears a stored placeholder; one that names an assistant is stored as named; one that names the retired placeholder is stored plain, unless the owner has that persona (§5.3).
- `tldw_Server_API/tests/DB_Management/test_sync_placeholder_assistant_migration.py` (§5.3), on SQLite and PostgreSQL: the placeholder is cleared, in trash too, with its memory mode and startup provenance; every other identity is untouched, including look-alikes and a real persona with that id, live or deleted; version and timestamps do not move; idempotent on reopen and when the migration runs twice; an interrupted run rolls back the repair and the version. PostgreSQL only: all owners are repaired in one run, another owner's persona does not protect a chat, and under a role that does not bypass row-level security the hidden rows are repaired and the forced policy is restored.
- `tldw_Server_API/tests/ChaChaNotesDB/test_chacha_conversation_store.py`: which identity counts as the retired placeholder.
- `tldw_Server_API/tests/Sync/test_sync_v2_satisfied_tombstone.py` (§5.4): a device's tombstone of a message the server never had is applied with tombstoned object state and no conflict; later writes succeed; a replayed push; the server's copy outside Sync is deleted; a third device receives the tombstone; a replica that never had the message and one that kept a copy both end at "gone"; a later append of the same id is refused without blocking; a tombstone that disagrees with a held message, one that names a base the dataset never issued, a row with other Sync history and another owner's message are left alone; recovery of a blocked dataset from a server write, a push, a re-push and a versioned write, and the conflicts it must leave.
- `tldw_Server_API/tests/ChaChaNotesDB/test_chacha_message_store.py`: the store method behind the rule (absent, deleted, replayed, other Sync history, another owner).
- `tldw_Server_API/tests/Sync/test_sync_v2_unenrolled_delete.py` (§5.2): the delete routes and the helper. A message and a chat Sync never saw are deleted with no envelope, and later writes still succeed; a row whose capture was lost; chats with both kinds of row, in both directions; a published delete still tombstones and reaches a second device; a row with a pending envelope is not deleted directly; a blocked dataset (direct delete still works, a needed tombstone refuses the whole request); stale versions; what follows a direct delete (a retried admission, a device that publishes its own copy); recovery of a dataset blocked by the old behaviour, from an ordinary write, a retried delete and a versioned write, and the conflicts it must leave alone; and the same deletes with no profile.
- `tldw_Server_API/tests/Sync/test_sync_v2_unenrolled_delete_postgres_contract.py`: the delete routes and the device tombstone rule with the chat database on PostgreSQL and with the Sync store on PostgreSQL, and a tombstone that names another owner's message in the shared database.
- `tldw_Server_API/tests/Chat_NEW/integration/test_history_selection_api.py`: the Sync owner now captures; the server-settled completion is still refused before any write.

## 8. Merge notes

- **P1 (#3171)** edits the same lines of `character_messages.send_message`. Resolve by adding `history_branch=message_data.tldw_history_branch` to the `append_selected_history_input` call inside `history_write`, and `**exc.details` to the 409. One closure serves both the Sync and the non-Sync path, so the leaf check then applies to Sync v2 owners too. A `history_branch_changed` refusal is raised inside the owner write, so it records nothing.
- **OpenAPI fingerprint.** `POST /chats/` changed its description and its 409 description; nothing else in the schema changed. Regenerate the fingerprint after merging with any other PR that touches it.
- **`sync_client_chat_id_unsupported`** no longer exists. P4 needs no special case for Sync v2 owners.
- **`fix/sync-v2-unenrolled-delete`** (§5.2, §5.3) is based on `feat/chat-sync-v2-native-history`. It changes `character_messages.delete_message`, `character_chat_sessions.delete_chat_session`, `server_origin.py` and the conversation materializer. No route, schema or docstring changed, so the OpenAPI fingerprint and the privilege snapshot are the same.
- **`fix/sync-v2-placeholder-and-device-tombstone`** (§5.3 migration, §5.4) is based on the P3 stack (`feat/chat-idempotent-create-p3`). It takes the next ChaChaNotes schema versions, SQLite v76 and PostgreSQL v80; another branch that adds a migration must renumber. `SyncV2Service` gains `recover_stranded_tombstones` and `settle_stranded_tombstones`; `push` calls the first before its loop, which costs one conflict lookup per push when nothing is blocked. No route, schema or docstring changed, so the OpenAPI fingerprint and the privilege snapshot are the same.
