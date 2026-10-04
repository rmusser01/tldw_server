# D7: native history for Sync v2 owners

- **Date:** 2026-10-04
- **Status:** §3 is implemented on `feat/chat-sync-v2-native-history`. §4 is proposed and needs the owner's decisions (§6).
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

**Three defects found on the base branch** are listed in §5. The first two together can leave a user's Sync v2 dataset blocked. The third is fixed here for the case this change depends on.

---

## 1. How Sync v2 writes a chat today

Sync v2 is an append-only log of envelopes per dataset, plus a projection of that log into the owner's ChaChaNotes database.

- **A profile is active** when the user has the default personal Chatbook dataset (`api/core/Sync/v2/server_origin.py:293-308`). The check is per user. Nothing records which conversations are in the dataset.
- **An ordinary API write is routed through the log.** With an active profile, `POST /chats/`, `PUT /chats/{id}`, `DELETE /chats/{id}`, `POST /chats/{id}/messages` and `DELETE /messages/{id}` call `capture_server_origin_mutation` (`server_origin.py:103-197`) for chats in the global scope. Workspace chats are never routed.
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
- **Blocked dataset.** If a projected conflict is unresolved, new versioned writes return 503 `sync_server_origin_append_failed` before the owner write. This matches ordinary Sync-routed writes.
- **Busy fence.** If the fence cannot be taken (a lock timeout), the request returns the same 503 before the owner write.
- **Replay and repair.** `repair` re-runs applied envelopes. The materializer sees matching object state and does nothing, so the owner's row is not rewritten.

### 3.4 The one failure window

The owner commit and the Sync commit are two commits. If the process dies between them, or the Sync database fails at that moment, the message exists on the server and is not in the log.

- The client gets an error (503, or a dropped connection), never a success.
- **The retry repairs it.** Admission, settlement and client-id create are idempotent, and the same request sent again publishes the row it finds.
- Settlement publishes an unpublished input before the reply, even if the admission was never retried.
- A replayed client-id create publishes a chat whose first capture was lost.

What is not repaired: a client that never retries, or that retries with a new message id (which adds a second message). The first row then stays out of the log until something else publishes it. §4.2 proposes the durable fix.

A server-origin write that fails after its owner commit is logged at warning level with the dataset and object ids.

### 3.5 A plain chat stays plain when Sync upserts it

A chat created with a client id and no assistant is stored as a plain chat. On the base branch, the first rename through Sync would have rewritten it as persona `sync-v2` (§5.3). The conversation materializer now leaves a chat that is already plain as it is when an upsert names no assistant. A chat first seen through Sync is projected as before.

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
| `persona_not_found` on plain chats | See §5.3. A plain chat first created through the ordinary Sync route is stored as persona `sync-v2`, and `/chat/completions` with that conversation id returns 404. |

**Options**

- **A. Applied capture at both points, no repair.** Small, but a lost capture is permanent and can later block the dataset (§5.2). Not acceptable.
- **B. Applied capture plus the outbox in §4.2.** The outbox row is written in the owner transaction, so a lost capture is published by the next request or by a sweep. Needs §4.2 first.
- **C. Route the turn through envelopes.** Would need an envelope that can be refused after acceptance. Sync v2 has no such state.

**Recommendation: B**, after §4.2 and the rest of the §5.3 fix, and after P2 lands so the settlement code is stable. The wiring is then small: wrap `accept_history` (`api/core/Chat/chat_service.py:4147-4166`) and the settlement closure (`chat.py:4657-4667`), refuse images before admission, and remove the capture refusal for character chats.

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

These exist without this change. They are reported here because they shape §4.

### 5.1 An unversioned completion save publishes nothing

With an active profile, `POST /chat/completions` with `save_to_db: true` and no selection writes the system, user and assistant rows directly. No envelope and no object state are written. Verified with a probe: three rows, an empty log. `chat.py` has no Sync capture on this path; only the character routes are guarded.

Effect: other devices never see these messages, and each is a row that §5.2 applies to.

### 5.2 Deleting a row Sync never saw blocks the dataset

With an active profile, `DELETE /messages/{id}` (or deleting the chat) for a message that has no object state appends a tombstone, which the materializer rejects as `message_base_conflict` / `missing_server_message` (`materializers/chat.py:363-380`). The request returns 503, the conflict stays unresolved, and every later Sync-routed write for that user returns 503 `sync_server_origin_append_failed` until someone resolves it. Verified with a probe for the message delete; the chat delete tombstones each message through the same call.

Rows this applies to: any chat that predates the profile (§4.3), the rows from §5.1, and a row left by the window in §3.4.

Suggested fix, separate from this work: before tombstoning, treat a row with no object state as not enrolled and delete it directly, or publish it first.

### 5.3 A Sync upsert turns a plain chat into persona `sync-v2` (partly fixed here)

The conversation materializer fills in `assistant_kind = persona`, `assistant_id = sync-v2` when a payload has no assistant identity (`materializers/chat.py:466-473`; pinned by `tests/Sync/test_sync_v2_chat_materializer.py:174`).

- On the base branch this also rewrote an existing plain chat on any later upsert, such as a rename. The WebUI reads the stored identity to choose how to send (`ui/hooks/chat/effective-assistant-state.ts`), so a renamed chat would have been treated as a persona chat for a persona that does not exist. This is from reading the code; it was not run in a browser.
- **Fixed in this change:** an upsert that names no assistant leaves a chat that is already plain as it is. A chat created with a client id is plain and stays plain when renamed.
- **Not changed:** a chat first created through the ordinary Sync route (an id-less `POST /chats/`, or a device push) with no assistant is still stored as persona `sync-v2`. `POST /chat/completions` with such a conversation id returns 404 `persona_not_found`. Verified with a probe.

The client-managed turn sends no conversation id to `/chat/completions`, so it is not affected by the remaining case. A server-settled turn is (§4.1).

---

## 6. Questions for the owner

1. **§4.2 outbox.** Approve a small ChaChaNotes table and migration to make publication durable? It is the prerequisite for server-settled turns under Sync v2.
2. **§4.1 timing.** P5 moves every chat to server-settled turns. Should P5 wait for §4.2 and §4.1, or should P5 keep the client-managed turn for Sync v2 owners until they land? The second needs a signal the client can read before it dispatches. The 409 on `/chat/completions` comes too late: the WebUI has already marked the turn as dispatched and would keep an "unknown outcome" record. The capture response is the natural place for that signal.
3. **§4.4.** Is it acceptable for the Sync materializer to mark a pushed message as graph data when it names its parent? Does the Chatbook client send `parent_message_id` on every message?
4. **§5.1, §5.2 and the rest of §5.3.** Should these be fixed as separate bug-fix PRs now? §5.2 can block a user's dataset today.

## 7. Tests

- `tldw_Server_API/tests/Sync/test_sync_v2_applied_server_origin_capture.py`: the helper. One applied envelope and object state per object, in order; a refused write records nothing and its own error is passed through; replay, including while the dataset is blocked; a lost capture and a failed Sync commit repaired by the retry; a partly recorded capture rolls back as a whole; refusals before the write (blocked dataset, unavailable fence, unapplied history on a claimed id, client-private policy, missing dataset, unsupported adapter version); tombstone and pull afterwards; full repair leaves the row untouched.
- `tldw_Server_API/tests/Sync/test_sync_v2_native_history_capture.py`: the routes, with a real Sync service and a separate materializer handle on the same database, as the service factory wires it. Capture, admit and settle; capture of a character chat still refused; a second turn; a second device pulls and applies the envelopes into its own database; replays; refusals; both repair paths; a blocked dataset; image refusal; chat delete; a rename that keeps a plain chat plain; a device push into the conversation (and the review it then requires); a turn that continues a chat the device wrote; a device push that reuses a published id; legacy review; the client-id create rules (201, 200 replay, rename then replay, 409, 410, lost capture, concurrent duplicates, character chat, workspace chat); and the same turn with no profile.
- `tldw_Server_API/tests/Sync/test_sync_v2_chat_materializer.py`: an upsert that names no assistant keeps an existing plain chat plain.
- `tldw_Server_API/tests/Chat_NEW/integration/test_history_selection_api.py`: the Sync owner now captures; the server-settled completion is still refused before any write.

## 8. Merge notes

- **P1 (#3171)** edits the same lines of `character_messages.send_message`. Resolve by adding `history_branch=message_data.tldw_history_branch` to the `append_selected_history_input` call inside `history_write`, and `**exc.details` to the 409. One closure serves both the Sync and the non-Sync path, so the leaf check then applies to Sync v2 owners too. A `history_branch_changed` refusal is raised inside the owner write, so it records nothing.
- **OpenAPI fingerprint.** `POST /chats/` changed its description and its 409 description; nothing else in the schema changed. Regenerate the fingerprint after merging with any other PR that touches it.
- **`sync_client_chat_id_unsupported`** no longer exists. P4 needs no special case for Sync v2 owners.
