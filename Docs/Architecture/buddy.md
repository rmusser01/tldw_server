# Buddy

This document describes the server-side companion subsystem: independent artwork snapshots with preserved attribution, per-client-slot attachments to conversations or workspaces, and bounded process-owned FIFO turns that drive the ordinary authenticated Chat API in-process. Buddy is not an LLM client — there is no provider code in `app/core/Buddy/`.

## Authoritative files

All paths under `tldw_Server_API/` unless noted.

| Concern | File | Key symbols |
| --- | --- | --- |
| Service | `app/core/Buddy/service.py` | `BuddyService` — create (starter or persona-pack source), asset reads with sha256 re-verification, target resolution (ownership-checked), roleplay-resume effective completion ("no server defaults apply"), attachments, activity projection, acknowledgements; snapshot limits 256 assets / 64 MiB |
| Turns runtime | `app/core/Buddy/turns.py` | `BuddyTurnRuntime` — capacity 64 (16 per user), 120 s timeout, idempotency digests, per-(user, conversation) FIFO drain, 5 s lease heartbeats, credentials cleared after execution |
| Publication | `publication.py` | the `current_buddy_publication` ContextVar — "cannot be supplied in HTTP input" |
| Profile DB | `app/core/DB_Management/Buddy_DB.py` | `buddy_profiles/assets/attachments/result_acknowledgements`; optimistic `expected_version` updates |
| Turns DB | `app/core/DB_Management/Buddy_Turns_DB.py` | the `buddy_turn_owners` lease table (30 s), turn rows with `UNIQUE(user_id, client_request_id)`, claim/expire/stop, the `assert_publication` conditional-update fence |
| Endpoints | `app/api/v1/endpoints/buddies.py`, `buddy_turns.py` | CRUD, targets, attachments (per `client_slot`), activity/acknowledgements; `POST /buddies/turns` → **202** with a bounded auth-header allowlist forwarded; stop; router lifespan closes the runtime |
| Chat hooks | `app/api/v1/endpoints/chat.py`, `app/core/Chat/chat_service.py` | publication captured **before** the executor (ContextVars don't inherit into threads); single-transaction persist + publication fence + receipt (the effect-replay ledger); fail-closed on revocation |

## Turn lifecycle (dataflow)

1. **Accept** (202, admission shielded from caller disconnect): dedupe by `client_request_id` (same key + same request digest → replay the stored row; different digest → 409) → capacity checks → prepare on a thread (attachment version match else 409; attachment/conversation containment else 404; provider/model from the request or the conversation's roleplay-resume settings, neither → 422) → SQL lease claim + row insert → FIFO enqueue.
2. **Execute**: publication fence → ContextVar set → in-process `httpx` ASGI POST to `/api/v1/chat/completions` with the requester's forwarded headers and a bounded non-streaming payload (`stream=False`, `tool_choice="none"`, save-to-db). Chat persists through its normal path; **each persisted message re-fences the publication and records a receipt in the same transaction**. The response must carry the exact conversation id and a message id matching the recorded receipt, else `stale_target`.
3. **Stop**: SQL stop first (revokes publication authority — any worker can Stop), then in-process cancellation. Provider effects are never undone or retried.
4. **Observe**: `activity` projects persisted results only for conversations inside the attached target; acknowledgements mark rows read.

## Boundaries

- Buddy reuses Chat admission, moderation, rate limits, and accounting by calling the real ASGI app with the requester's own credentials and a fixed internal URL — never a client URL, never a server master credential (the ADR-046 rule).
- Only the principal's own content DB opens; foreign/shared targets are refused.
- Turn bodies and forwarded credentials live only in retained runtime task memory, cleared post-execution.
- One process owns a user's queue; losing the lease is terminal — expired work fails (`interrupted_unknown`, or `completed` if a receipt was already recorded), never replayed against the provider.

## Failure behaviors (verified)

| Case | Behavior |
| --- | --- |
| Unknown buddy / foreign target / missing attachment | 404 |
| Version mismatch / key reuse with different body | 409 |
| Runtime closed / capacity exceeded | 429 |
| Another process holds the lease | 503 + `Retry-After: 5` |
| No resolvable provider/model | 422 |
| Chat non-2xx / stop observed at publish / lease expiry | `failed` with `chat_http_<status>` / `stale_target` / `interrupted_unknown` |
| Artwork checksum mismatch (create or read) | 422 at create; 404 at read — tampered files are 404s, not 500s |

## Config keys

None — all constants are code (snapshot limits, runtime capacity/timeout, lease/heartbeat intervals, the `client_slot` charset constraint).

## Verified gotchas

1. Turn acceptance returns 202 with the queued row, never the completion — results surface only through Chat persistence + activity.
2. The publication ContextVar is captured before `run_in_executor` because executor threads don't inherit ContextVars — the classic refactor breakage.
3. Lazy reconciliation on every list/get: expired owners' turns resolve as they're read, and can flip to `completed` posthumously when a receipt exists.
4. Attachments are per `client_slot`, not per Buddy — one Buddy can serve several slots.
5. The runtime lives on `app.state` per process: multi-worker deployments get owner affinity for new sends, while Stop stays cross-worker via the SQL fence.
6. Streaming is impossible by construction (`stream=False` is forced).

## Governing decisions and docs

No Buddy-specific ADR; ADR-046 records the governing boundary rule Buddy implements. No module README — this doc plus `Docs/Code_Documentation/Persona_Ambient_Companion.md` (current, client-side behavior) and `Persona_Visual_Packs.md` (pack review/activation semantics) are the references. Tests: `tests/Persona/test_persona_buddy_*`, `tests/ChaChaNotesDB/test_persona_buddy_db.py`.

## Related docs

- [chat-character.md](./chat-character.md) — the Chat boundary Buddy drives
- [authnz.md](./authnz.md) — the forwarded credentials
