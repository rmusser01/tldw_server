# UAT118 / TASK-13260.58 — read-only attachment recovery design

No production/test edits, browser actions, runtime changes or inference were performed for this assessment. The separately reviewed text UAT108 unit remains frozen.

## Confirmed boundaries

1. `chacha/message_store.py:get_messages_for_conversation` already loads ordered `message_images` with each owned message; `get_message_images` preserves position and the legacy primary `image_data`/`image_mime_type` columns remain available. Existing write limits constrain attachment bytes. No new database storage is needed.
2. `character_messages.py:_convert_db_message_to_response` and `chat_session_schemas.py:MessageResponse` expose only `has_image`. Neither standard list nor individual message GET returns attachment bytes; completion-formatted listing also emits only text. Consequently `domains/chat-rag.ts:listChatMessages` discards attachment presence/bytes, and `useServerChatLoader.ts:mapServerChatMessagesToPlaygroundMessages` emits empty images for ordinary users. Its separate generated-assistant-image envelope is already supported.
3. `chat.py:_save_message_turn_to_db` substitutes `<Image attachment xN>` when successful attachments have no text. That synthetic body has no distinguishing metadata today. Actual `TldwChat` removes empty text parts before sending, so real image-only input reaches this branch.
4. `generate-history.ts` emits image then text. The current backend explicit Retry matcher reconstructs stored text then images and compares arrays. Thus even exact text+image can conflict solely due representation order; the existing unit fixture was text-first and does not establish real frontend compatibility. This is source-confirmed; no native image inference was run.

## Existing patterns compared

- **Owned paginated message listing:** retain `_verify_conversation_access`, per-user DB, existing scope/permission/deletion and pagination rules. This is the smallest place to expose stored bytes without a second fetch per image or new access route.
- **Character avatar listing:** existing `include_image_base64=false` default demonstrates explicit opt-in binary expansion. Use the same opt-in principle for messages, leaving current default payloads light.
- **Canonical DB history to provider:** `chat_service.py` and `chat_helpers.py` encode ordered stored image bytes as data URLs, with a primary-blob fallback for old rows. Reuse this representation and ordering rather than persisting another copy of image data.
- **Frontend generated-image mirror:** existing mapper/data-URL rendering and Dexie `images:string[]` already support displaying/persisting image URLs. Preserve generated-assistant-event precedence and existing content/owner guards.

## Recommended minimal contract

Add `include_images=false` to the existing authenticated `GET /api/v1/chats/{id}/messages`, with optional `images` data URLs on the standard response schema. When requested, serialize the already-loaded ordered attachments; fall back to the legacy primary blob only when the ordered list is empty. Preserve the default response and existing completion-format behavior. No public/binary route, new capability, metadata copy of base64 content or credential-bearing URL is needed. Use existing validated image MIME formats; malformed stored attachment data must not create a false match. Request page limits still apply.

Pass this optional field through the current client domain adapter and type, and have the active owned loader request it. Feed returned images to the ordinary message mapper. The current exact correlation/role/content/images checks should remain unchanged. A cross-owner or stale result continues to fail the existing scope/transaction guard.

For new image-only writes, use the existing `content_placeholder_reason` metadata key with a distinct `image_attachment` reason, set only by the server when it actually synthesized the placeholder. This avoids another copy of original input text or images. The loader may project empty user text only when that marker, the exact generated placeholder and the valid attachment count agree. Never strip a matching-looking literal user string based on text alone. Apply the same projection when the backend validates an explicit Retry.

For the actual frontend retry shape, compare the exact persisted text and ordered decoded image bytes/MIME as separate stored components, rather than literal mixed-part array order. At minimum cover the actual zero-or-one-text-part frontend shape; do not introduce whitespace trimming, image substitution, equal-text identity, or broad multipart normalization. Preserve strict correlation, canonical tail, ordinary-repeat and leftover-user409 checks. Multiple image order remains significant. If arbitrary multipart segmentation cannot be represented exactly by the current DB projection, fail closed rather than generalizing silently.

## Why not a new digest identity

A new input digest alone does not restore the visible attachment. It also needs a shared canonicalization contract and invalidation on text/image edits; otherwise stale hashes can acknowledge edited canonical rows. That expands both scope and lifecycle risk. Returning existing owned bytes through an opt-in listing plus explicit server-generated placeholder provenance is smaller and independently inspectable.

## Exact proposed production scope (approval required)

1. `tldw_Server_API/app/api/v1/schemas/chat_session_schemas.py`: optional attachment response field.
2. `tldw_Server_API/app/api/v1/endpoints/character_messages.py`: opt-in serialization after existing conversation authorization.
3. `tldw_Server_API/app/api/v1/endpoints/chat.py`: mark only new synthetic image-only content.
4. `tldw_Server_API/app/core/Chat/chat_service.py`: exact stored-component failed Retry comparison and image-only marker interpretation.
5. `apps/packages/ui/src/services/tldw/TldwApiClient.ts`: response type only.
6. `apps/packages/ui/src/services/tldw/domains/chat-rag.ts`: preserve the optional image array.
7. `apps/packages/ui/src/hooks/chat/useServerChatLoader.ts`: request images and map actual owned attachment content.

No database migration, shared auth/transport default, correlation-helper weakening, model/provider selection or new endpoint allowlist is required. `server-chat-mirror.ts` should remain unchanged if the mapper supplies truthful content/images.

## Behavioral test contract

- **Same text + same image:** actual frontend serialization → real endpoint/SQLite persistence → provider failure → actual client/list mapper/mounted loader → Retry → canonical reload retains one user and exact attachment bytes plus final assistant. Capture the real image-first wire order. Do not merely handcraft an `images` mirror row.
- **Same image + changed text:** no ACK of newer local draft; explicit Retry conflicts before writes/provider. Original server and local draft remain intact.
- **Same text + changed image / changed image order:** same fail-closed behavior; distinguish identical MIME with different bytes and repeated images with changed order.
- **Image-only:** server-generated placeholder marker permits exact empty-text reconstruction; persisted/returned attachment remains visible through reload and Retry. A literal user text `<Image attachment x1>` is preserved. Invalid/failed images keep existing failure-placeholder behavior.
- **Old records:** legacy primary-blob images can render without migration. Existing canonical ACKs remain stable. An old unacknowledged image-only placeholder without provenance/correlation must remain unmatched/actionable; do not silently guess or delete. A versioned/user-edited placeholder must not be erased from display.
- **Unauthorized or delayed owner:** actual other-user conversation GET remains404/403 under current permission contract, scoped stale requests and A→B→A cannot hydrate images, and foreign local PKs cannot be repointed. The opt-in flag adds no access path.
- **Compatibility:** default list does not expand attachment bodies; pagination and both server scopes preserved; tracked-character placeholder rendering and generated-assistant-image events unchanged; ordinary repeats and successful-answer regeneration remain separate operations.

## Existing test locations suited to these boundaries

- Backend real owner/persistence: `tests/Chat/integration/test_persona_backed_chat_conversations.py` plus existing character-message authorization/listing tests selected after task implementation begins.
- Backend Retry: `tests/Chat/unit/test_chat_history_and_streaming.py`; retain exact DB transaction and changed-tail controls.
- Client wire: `services/tldw/__tests__/TldwApiClient.request-scope.test.ts` and existing domain chat-list tests.
- Mounted action/loader: `hooks/chat/__tests__/useChatActions.saved-normal.integration.test.tsx`; use real `ChatTldw` serialization and a real-shaped listing adapter.
- Mapping/ownership: `hooks/__tests__/useServerChatLoader.scope.test.tsx` and existing loader mirror/image-event controls; preserve prior fail-closed tests where image bytes are omitted.

OpenAPI schema/fingerprint refresh may be needed for the additive query/response fields; coordinate with root rather than changing generated contract artifacts independently. No product/test write window has been granted yet.

## Refinement after independent review (authoritative over the earlier sketch)

Reviewed `/private/tmp/cycle4-uat118-independent-design-review.md`. All four requested refinements are accepted. The following bounds avoid a new storage/identity framework.

### 1. Placeholder proof is version-bound

Use the conservative existing message version, rather than changing every message-edit entry point. Set `content_placeholder_reason: image_attachment` only **after** successful image normalization actually creates the synthetic body, within the same message/metadata transaction. Empty-text projection is permitted only for a user row at **version1**, with that marker, the exact generated placeholder/count and a complete validated attachment list. Backend provider-history projection, backend explicit Retry matching and frontend mapping use the same conditions. Unknown versions fail closed.

Any later version retains its stored literal text, including a pin/metadata-only edit or an edit-away-and-back to the same placeholder. This is intentionally conservative: an unrelated version bump may require the user to resolve the edited failed turn, but cannot silently erase user intent. Old image-only rows without the marker are rendered as stored; their images can restore, but unacknowledged local empty text is not guessed into an ACK. No migration or blanket placeholder stripping.

### 2. One bounded equivalence for explicit Retry tail and overlap

Reuse a small local comparison/projection helper in `chat_service.py`, restricted to explicit failed Retry. It accepts plain text or zero/one text part plus ordered image parts. It preserves exact text, ordered image bytes/MIME, and existing user identity. Only validated base64 data URLs are decoded locally; never fetch a URL. Reject unknown part keys/types, multiple text segments, non-default image detail, invalid base64 and unsupported MIME. Normalize omitted/default `detail:auto` only, as the existing Pydantic/request contract already does.

Apply that same bounded representation to both the actual-tail check **and** the existing overlap signatures. Keep roles, tool metadata and the residual unmatched-user409 check. Ordinary sends retain their existing distinct-turn behavior. This explicitly handles the real frontend's image-first order versus the DB's separate text/ordered-image fields without discarding data. New unrepresentable explicit Retry input rejects before writes/provider dispatch.

Canonical provider history must use the version-bound empty-text projection too. Otherwise a valid image-only Retry would still send `<Image attachment xN>` to the model. The corresponding earlier successful image-only turn must also project correctly. Metadata is already read for historical messages; reuse that read. No new general serializer replaces the existing provider pipeline.

### 3. Strict reads must distinguish missing data from read failure

Add one narrowly scoped storage option in `chacha/message_store.py`: strict attachment reading for the opt-in listing and explicit Retry, with the default preserving existing callers. `get_message_images` must propagate its existing database error in strict mode. `get_messages_for_conversation` passes that option through. A failed ordered-image read must **not** fall back to the primary blob. Only a successful, genuinely empty ordered list permits legacy primary-blob fallback.

Validate every returned attachment before serialization. Empty/corrupt entries, invalid MIME, invalid positions or an incomplete sequence fail the entire opt-in read; do not return a subset or silently collapse duplicate positions. The client adapter must likewise reject malformed present attachment arrays, rather than filtering bad entries to an apparently complete empty/subset array. A absent optional field from a legacy/default server remains absent; this does not claim complete image recovery. Existing loader error handling preserves the prior local transcript and exposes retryable load failure, with no partial mirror reconciliation.

This adds **one production path** to the earlier seven: `tldw_Server_API/app/core/DB_Management/chacha/message_store.py`. No migration, other DB facade rewrite or broad exception behavior change is intended.

### 4. Bounded expansion preserves pagination and abort semantics

Keep the existing requested message limit/offset/order. Do not shorten a page to meet a byte limit; this loader interprets a short page as end-of-history. Do not drop over-budget attachments or reconcile a partial page.

Proposed concrete caps: **32MiB decoded attachment bytes per opt-in page** and **64MiB encoded attachment characters across the full frontend load**. These are simple read/expansion caps, not new user configuration or write limits. Reuse the current per-image write/validation limits and MIME helper as well; the page cap cannot override those. The encoded client cap is checked after each page and before appending/reconciling it. Existing per-page abort/owner checks remain; explicitly check an already-aborted signal before and after every awaited page.

To bound the read as well as base64 expansion, the strict optional DB listing path selects the exact page, computes stored attachment byte lengths and conditionally projects blobs in one SQL statement. Count ordered attachments, falling back to the primary column only when the ordered table is genuinely empty; avoid counting the duplicate primary copy twice. The single statement supplies one snapshot on SQLite and PostgreSQL, without changing shared transaction isolation. A CTE identifies the page and size; CASE-gated primary/ordered blob columns remain absent when over budget. Reconstruct the existing rows and validate attachment positions without follow-up image queries. This query belongs in the storage layer. Exceeding the cap fails visibly without fetching/serializing the binary payload. Recheck actual accumulated decoded lengths before base64 serialization. The default listing path and pagination contract are unchanged. No background downloader, byte-range protocol or lazy-image framework is added.

Independent review found that the ordinary PostgreSQL transaction is READ COMMITTED, so a separate preflight and blob query would not establish the required common snapshot. The bounded single-statement strict path is the approved correction; PostgreSQL verification uses existing fixtures and records genuine availability limits.

If the existing transaction helper cannot provide the required stable read without changing default behavior, stop and report that concrete implementation constraint before expanding this design.

### Clarified response compatibility

`include_images=false` must not expand image bytes on create/update/search/individual GET or the normal default listing. The new field is optional and absent when not requested (do not accidentally inject null/empty images via the shared conversion helper). Apply the opt-in explicitly in both standard listing branches, with/without character context. Completion-formatted output retains its current contract. The unauthenticated share resolver is untouched. Active loader requests remain scoped and bypass shared cache; query-specific cache keys distinguish other opt-in callers.

### Additional permanent boundaries required before implementation freeze

1. An earlier successful image-bearing pair followed by a failed image turn: actual image-first request, tail reuse **and** overlap succeed; provider sees each user once, without synthetic empty-image text.
2. Version1 synthetic image-only row versus edit-away-and-back, literal lookalike, unknown/later version and metadata-only/pin version changes. Only the proven unedited row projects empty text.
3. Strict ordered-image read error with a valid primary blob must fail; no fallback. Corrupt second attachment must not turn a two-image server message into a one-image local ACK.
4. Both standard list branches, unchanged default/completion payloads, exact page boundary and multi-page history, cap exceedance before blob expansion, cumulative client cap and abort/ABA between pages. No partial mirror writes.
5. Existing authorization, wrong workspace, deleted conversation/message, old primary-blob and generated-assistant-image controls remain green.

The actual endpoint/DB, real client adapter, mounted loader/mirror and exact transport order must participate in the regressions. Pure handcrafted image arrays remain safety/unit controls only. Implementation remains paused pending root's explicit production write window.

## Bounded implementation extension after the approved source window

The actual mounted transport/loader regression exposed an existing local-history conversion in `hooks/chat-modes/normalChatMode.ts`: an already qualified PNG URL is rewritten with a JPEG MIME prefix while the outbound request retains PNG. The exact canonical match then correctly refuses the mislabeled mirror. TASK13260.58 also owns the minimal correction to preserve qualified image data URLs while retaining the existing raw-payload fallback. Keep MIME/byte equality strict, and retain the actual mounted failure and correction as evidence. This adds one production path to the approved eight-path unit; it does not change the attachment storage or authorization design.

The same actual image-only flow exposed `hooks/handlers/messageHandlers.ts` returning before reading an attachment whenever user text is empty. Its bounded correction reads the image first, accepts text or image, and uses trimming only for empty-input validation while preserving the original text for matching. This is the tenth production path. Both existing loader-to-history mappings must retain the supported image field. The current frontend history accepts one image string; endpoint tests with ordered image arrays do not certify general multiple-image frontend Retry. Preserve extra canonical attachments and reject unrepresentable/ambiguous work instead of silently discarding images or expanding unrelated APIs.
