# H2 native fork source audit — 2026-09-18

Task: TASK-13261.2. This is source inspection, not H2 runtime qualification. Implementation baseline is H1 PR [#2968](https://github.com/rmusser01/tldw_server/pull/2968), `ac76c4bc5b035561bf816009c1326a114e87def9`. Server `dev` was checked at `59049e094e0845a4611ea725ae19b7c1754ea709`; Chatbook `dev` at `e89f28d751bc8a5b4f4545b8894b87437252c657`. No H2 production or test code exists in this audit.

The complete independent [asset audit](CHATBOOK_H2_NATIVE_ASSET_AUDIT_2026_09_18.md) and [character audit](CHATBOOK_H2_NATIVE_CHARACTER_AUDIT_2026_09_18.md) are preserved alongside this record. Their proposals are inputs; the focused [design](../Design/2026-09-18-chatbook-h2-native-fork-design.md) decides the resulting contract. Paths and line numbers below refer to the immutable H1 baseline.

## Native transaction and authorization

| Existing source | Verified behavior | H2 consequence |
|---|---|---|
| `core/Chat/persistence_service.py:144` | `native_history_owner_key` hashes normalized ASGI origin/base path and authenticated user. | Reuse this owner key; do not trust a request-supplied user/server namespace or a forwarded header. Bind workspace separately. Origin alias changes fail closed. |
| `api/v1/endpoints/chat.py:6410,8121,8159` | Ownership, workspace/global scope, expected-user and active Sync routing precede H1 capture/legacy projection. | Reuse guards. Receipt resolution is a separate route with no live-source lookup dependency. A new uncommitted native operation must not bypass enrolled Sync ownership. |
| `core/DB_Management/ChaChaNotes_DB.py:703,8411,44676` | Schema is 68; repository transactions use SQLite immediate write admission and managed PostgreSQL connections. Nested caller-connection writes do not independently commit. | Reserve migration 69 only after checking current head. Insert child graph, settings/snapshot, owned references and receipt on the same connection. |
| `core/DB_Management/chacha/conversation_store.py:387` | `add_conversation` accepts caller connection; fresh IDs are available. | Use a dedicated bound insertion seam, never normal character creation or per-message HTTP requests. |
| `core/DB_Management/chacha/message_store.py:64,78,340,526,602` | Owner lock, coherent snapshot, selection validation and selected-input insertion already exist. | Reuse coherent reads/pure graph resolution, adding fork-purpose context projection rather than weakening send validation. |
| `core/DB_Management/chacha/message_store.py:942,962` | Existing message/metadata edits can lock old rows before advancing the conversation fence. | Fork must not lock old message rows after acquiring the conversation lock. Test actual writers, not only synthetic fence changes. |
| `core/DB_Management/chacha/conversation_store.py:111` | Settings writes lock conversation before settings. | Read/copy settings and accepted snapshot under that same conversation lock. |
| `core/DB_Management/chacha/conversation_store.py:1248,1389` | Soft/hard deletion mutate the conversation; hard deletion cascades child-owned data. | Durable receipt/key history cannot have a cascading source/child FK. Every deletion route must burn the operation's replayable child outcome transactionally. |
| `core/DB_Management/backends/pg_rls_policies.py:820` | H1 projection RLS depends on a live conversation. | Receipt RLS instead binds directly to authenticated owner; reauthorize child access before disclosing IDs/map. |

Source paths in this table are under `tldw_Server_API/app/`.

## Comparable implementations and their limits

1. `chacha/note_graph_suggestion_store.py:304,2086,2116,2169` supplies canonical request fingerprints, receipt admission and completion CAS. Its schema's source-note cascade and expiring lifecycle do **not** meet deletion-independent H2 replay. Reuse the small transaction pattern, not the table or retention policy.
2. `chacha/message_store.py:602,984` supplies selected-history admission and caller-connection image/message insertion. It is the closest single-owner commit model. H1's raw settings digest includes excluded summary data and cannot be used unchanged for fork retention.
3. `Sharing/clone_service.py:188` and `media_db/repositories/clone_snapshot_repository.py` illustrate operation-owned staging and conservative reconciliation. Their multi-store publication and media/chunk/transcript snapshot do not already copy original binary files or supply the H2 chat transaction.
4. `Sync/v2/blob_store.py:36` supplies verified bounded chunks and immutable namespace/hash publication independent of Sync enrollment. A native adapter needs its own root and ChaCha ownership/cleanup claims; calling the Sync coordinator or GC would introduce the wrong authority.

All paths in this list are under `tldw_Server_API/app/core/DB_Management/` except `Sharing/` and `Sync/`, which are under `tldw_Server_API/app/core/`.

## Client and protocol audit

`apps/packages/ui/src/db/dexie/fork-operations.ts` persists a client dispatch guard. It is not a native receipt: only `prepared` can claim first dispatch, and a lost-response legacy copy cannot safely be retried. H2 needs an explicit protocol discriminator; absent discriminator means the existing legacy protocol. A legacy pending/unknown record must never be sent to the new atomic endpoint.

`apps/packages/ui/src/services/chat-history-selection.ts:637` deliberately limits native copying to plain history. `:779` creates the conversation and messages in separate requests, so `legacy_completed`, partial and unknown outcomes are truthful. The H2 native adapter replaces this path only after explicit capability negotiation. A 404 from a mutation is not evidence an old server is safe to fall back to.

Scoped transport seams are `services/tldw/domains/chat-rag.ts:1230`, `services/tldw/service-prompt-scope-error.ts:95`, shared `entries/background.ts:1420,1684`, `services/background-proxy.ts:2231` and `services/tldw/request-core.ts`, all under `apps/packages/ui/src/`. `apps/extension/entrypoints/background.ts` is only a re-export of the shared entry. New methods require exact path/method allowlisting, expected-user guarding, frozen origin/account/workspace routing and malformed-path negatives. No widening of the entire conversations API is needed.

`components/Common/Playground/HistorySelectionReview.tsx:128` retains operation outcomes, and the H1 handler distinguishes a committed copy from failure to open or save the local result record. Reuse those boundaries. Add same-key recovery only for the atomic protocol and preserve owner leases across every asynchronous transition.

## Issues the focused design must close

| ID | Risk if left unspecified | Required resolution |
|---|---|---|
| A1 | External Media/Collections/AuthNZ lookup then chat commit races revocation/deletion. | Native manifest authority before accepted fork review, or a real common guard honored by all relevant mutators; never describe an unguarded re-read as atomic. |
| A2 | Generated text event may have no retained bytes; a document citation is not its original. | Typed representation inventory and exact selected revision; missing bytes are strict failure or explicit degraded review. |
| A3 | Reclamation races adoption or deletes another attempt's bytes. | Per-attempt namespaces and transactional non-adoptable reclaiming state before byte deletion. |
| C1 | Copied live-card FK cascades on card hard deletion. | Child-owned protected snapshot binding with null card FK, plus list/detail/edit/send readers. |
| C2 | Normal completion reloads live card/preset/book despite a copied snapshot. | Frozen rich composer admitted before input persistence; cold reopen and next provider payload qualification. |
| C3 | Excluded summary/prefill changes stale a fork or silently survive inside duplicated controls. | Canonical retained projection, scrub every supported representation, preserve declarative policy, rebind projected snapshot digest. |
| R1 | Receipt expires or cascades, permitting a second child. | Permanent accepted-key tombstone and source-independent lookup. |
| R2 | New client replays an old partially completed legacy copy. | Protocol-tagged records; legacy never upgrades into atomic retry. |
| R3 | Stale view/account updates show or mutate another owner's operation. | Existing leases plus frozen original owner/scope for capture, dispatch, lookup and result adoption. |
| R4 | PostgreSQL conversation→message lock inversion deadlocks with real edits. | Preserve established writer order; coherent MVCC old-row read and source revalidation. |

These are design obligations, not claims of bugs introduced by the H1 PR. No test results from H1 qualify a new H2 implementation.
