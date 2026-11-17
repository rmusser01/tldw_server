# Chat, character chat, and chatbooks

This document describes the conversation APIs the chatbook client consumes: general chat completions with history selection and branch fencing, character sessions with streaming + separate persistence, character/world-book CRUD with V1–V3 import, chatbook ZIP export/import with quotas and signed downloads, macros, and workflows.

## Authoritative files

All paths under `tldw_Server_API/` unless noted.

| Concern | File | Key symbols |
| --- | --- | --- |
| General chat routes | `app/api/v1/endpoints/chat.py` | `POST /chat/completions`, conversations CRUD + tree + share links + analytics, RAG-context, citations, history selection |
| Orchestration | `app/core/Chat/` | `chat_service.py` (the dispatch seam), completion/response pipelines, moderation, persistence, tool execution + auto-exec, streaming pipeline/utils, rate limiter, request queue, prompt templates, provider manager, history selection, native fork projection |
| Character sessions | `app/api/v1/endpoints/character_chat_sessions.py` | `/api/v1/chats`: create/list/get, `complete` + `complete-v2` (streaming SSE), `completions/persist`, prompt preview, presets, greetings, author-note, lorebook diagnostics, export/restore |
| Character core | `app/core/Character_Chat/` | session engine, card IO (V1/V2/V3 import), `ccv3_parser.validate_v3_card`, world-book manager, dictionaries, grammar, per-user rate limiter/limits, regex safety |
| Characters CRUD | `app/api/v1/endpoints/characters_endpoint.py` | `/api/v1/characters`: import, CRUD, versions/diff/revert, exemplars, world books + entries; **export emits `chara_card_v2`** |
| Chatbooks | `app/core/Chatbooks/` (`chatbook_service.py`, `chatbook_models.py`, `chatbook_format_v1_1.py`, `quota_manager.py`, `services/jobs_worker.py`, `import_adapters/openwebui.py`) | export/import, v1.1 file inventory + envelopes, 100 MB cap, Jobs contract (domain `chatbooks`, export/import), OpenWebUI import with hard caps |
| Chatbooks routes | `app/api/v1/endpoints/chatbooks.py` | export (+continue), import, preview, OpenWebUI hydration, jobs, `download/{job_id}`, cleanup |
| Macros | `app/core/Chat_Macros/` + routes | YAML-defined macros under the user DB base; catalog CRUD, validate, run (202 Accepted), cancel, clone |
| Workflows | `app/core/Chat_Workflows/` | dialogue orchestration (debate + moderator), question rendering modes |

## General chat turn (dataflow)

Endpoint → payload validation (413 on size/length) → moderation → **history selection** (`tldw_history_selection_v1`, ADR-049): a stale branch (`tldw_history_branch:false`) returns **409 `history_branch_changed` with the current leaf ids before any provider call** → provider stream (see [llm-providers.md](./llm-providers.md)) → terminal `stream_end` event carries the conversation id, success, and saved message ids → single `[DONE]`.

## Character chat turn (dataflow)

1. Ownership check; persona sessions refuse generation with 409 `persona_session_generation_unsupported`.
2. LLM quota then the per-user character rate limiter.
3. Context resolution (directed character, participants, aliases) and effective generation settings.
4. Provider call with streaming caps (max chunks/bytes; exceeding yields an SSE error frame and stop) and background stream cleanup.
5. **Streaming replies are not persisted even with `save_to_db=true`** — clients must call `POST /{chat_id}/completions/persist` afterwards.

## Chatbook export/import (dataflow)

- Export: validate + quota admission (UserProfiles `limits.chatbooks_*`; off-switch) → sync mode writes manifest+files and zips into the per-user exports dir with a completed ExportJob (`download_url`, `expires_at`); async mode enqueues a chatbooks-domain job. v1.1 (`format_version: "1.1.0"`) adds sha256 `file_inventory` and content envelopes.
- Download: job-id format + ownership + COMPLETED checks; expiry enforced (410) unless disabled; optional HMAC signed URLs whose message binds `job_id:exp:user_id` so tokens cannot cross accounts.
- Import: ≤100 MB multipart → temp save → ZIP validation → safe extraction (path traversal refused) → per-content-type import; OpenWebUI source format supported with hard caps (10k chats, 20k messages/chat, 200k total).

## Boundaries

- Separate routers/files per surface (`/chat`, `/chats`, `/characters`, `/chatbooks`, `/chat/macros`); all persist into per-user ChaChaNotes DBs.
- Character cards: server stores canonical rows; import accepts V1/V2/V3 while **export emits V2**.
- Chatbooks never ingests media directly — binaries ride the ZIP; per-content-type DBs are touched through their owners.
- **`discovery_owner` does not exist server-side** — it is a chatbook-client partition. The server equivalents are the `character_id`/`character_scope=all|character|non_character` conversation filters and the `assistant_kind` (`character|regular|persona`) + `persona_memory_mode` normalization.

## Failure behaviors (verified)

| Case | Behavior |
| --- | --- |
| Oversized payload | 413 |
| Stale history branch | 409 `history_branch_changed` + current leaf ids |
| Persona-session generation | 409 |
| Character stream caps | SSE error frame + stop |
| Chatbook download expired | 410 |
| Chatbook file > 100 MB | 413 |
| Invalid job id / not owner / not completed | 400 / 403 / 400 |
| OpenWebUI import over caps | Rejected by the adapter |

## Config keys (selection)

`[Chat-Module]` (see [llm-providers.md](./llm-providers.md)); chatbooks env: `CHATBOOKS_DISABLE_QUOTAS`, `CHATBOOKS_ENFORCE_EXPIRY`, `CHATBOOKS_SIGNED_URLS` + `CHATBOOKS_SIGNING_SECRET`, `CHATBOOKS_JOBS_QUEUE`; quota values in UserProfiles `limits.chatbooks_*`.

## Verified gotchas

1. Character streaming is fire-and-forget — the separate persist call is mandatory, not optional polish.
2. V3 cards import but export as V2.
3. `CHARACTER_CHAT_API_DOCUMENTATION.md` is stale (documents `/complete` only; missing `/complete-v2` and `/completions/persist`); `Chat_Developer_Guide.md`'s dispatch entrypoint wording is stale; the fresh references are `Chatbook_Developer_Guide.md`, `Guides/Chatbooks_Code_Guide.md`, and `Guides/Evaluations_Code_Guide.md` (all October).
4. The chatbooks module README's route line references have drifted.

## Governing decisions

ADR-049 (`Docs/ADR/049-chat-history-selection-ownership.md`), ADR-050 (native chat fork storage lifecycle), ADR-054 (chatops shared shell), ADR-057 (workspace chat strict startup receipts), ADR-009 (quick-chat docs assistant modes).

## Related docs

- [llm-providers.md](./llm-providers.md) — the provider seam underneath
- [notes-sync.md](./notes-sync.md) — chat domains on Sync v2
