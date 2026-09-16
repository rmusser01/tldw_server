# UAT118 independent design review

Reviewed `/private/tmp/cycle4-uat118-image-recovery-design.md` against frozen source `4bd4e2dfda246bd83650dc922789551032a5af52`. Read-only source/design review; no repository edits, browser, runtime changes or inference.

## Outcome

**Support the existing authenticated opt-in listing and component-based Retry comparison, with the refinements below.** Existing ordered image storage, data-URL consumers and scoped listing make this smaller than new storage, digest identity or image-fetch routes. Preserve the strict mirror correlation checks. The seven proposed files cover most of the correction; strict attachment-read failure handling may require one narrow storage change.

## Required design refinements

1. **Bind synthetic-placeholder provenance to unedited content.** `character_messages.py:1033` updates content without clearing metadata.extra; `message_store.py:1221` increments the message version. A server marker plus exact placeholder/count alone still strips a user edit back to the identical literal placeholder. Require a demonstrably unedited version on both client and backend projection, or invalidate the marker transactionally on content edits. A conservative version-1 rule is small but must explicitly retain literal text after later versions, including metadata-only edits. Set the marker after actual image normalization: the current marker serialization at `chat.py:3116` precedes image-placeholder creation at `3150`.

2. **Apply exact component comparison throughout explicit Retry reconciliation.** Fixing only `chat_service.py:4278` leaves overlap signatures at `4332` sensitive to stored text-first versus frontend image-first order. With an earlier image-bearing user/assistant pair and a later failed image-bearing turn, overlap can remain zero and the residual-user guard at `4398` rejects an otherwise exact Retry. Reuse bounded component equivalence for explicit Retry overlap, retaining the extra-user rejection and ordinary-repeat behavior. Also project proven image-only placeholders in canonical provider history (`4172–4224`); otherwise the provider still receives synthetic text even when the matcher reconstructs empty user text. Reject unrepresentable multiple text segments, unsupported part keys and non-default image-detail semantics rather than silently discarding them. Decode only validated data URLs, without fetching external URLs.

3. **Do not turn incomplete attachments into a complete empty/subset list.** The mirror compares ordinary image arrays exactly (`db/dexie/server-chat-mirror.ts:49–62`). Silently dropping corrupt attachments can falsely ACK a locally edited subset. Prefer an explicit failed opt-in read over partial images, preserving the local draft. Moreover, `message_store.py:401–422` catches image-read errors and returns `[]`; that is indistinguishable from a genuine legacy row and could trigger primary-blob fallback. If completeness is promised, distinguish this read failure from a truly empty ordered table; keep old default callers compatible.

4. **Bound binary expansion without changing message pagination silently.** The loader requests 200 rows across up to 100 pages (`useServerChatLoader.ts:77–78,162–192`). Per-image write limits do not bound total serialized response/client memory; one allowed image per row can already make a page very large. Specify an opt-in byte/read bound and fail visibly before partial reconciliation, or an equivalent bounded strategy. If the endpoint silently returns fewer rows to fit bytes, this loader treats that short page as end-of-history. Preserve offsets/order and abort checks on every page.

## Compatibility and access controls

- Keep `include_images=false` lightweight on all existing callers. `_convert_db_message_to_response` is also used by create, individual GET, update and search; do not expand those accidentally. Define omitted versus null optional fields explicitly.
- Apply the opt-in to both standard listing branches, with and without character context (`character_messages.py:788,859`); retain completion-format behavior and placeholder-rendering defaults.
- Existing `_verify_conversation_access` (`220–267`), user-scoped DB and global/workspace checks are appropriate. Do not add attachment expansion to the unauthenticated share resolver. Verify deleted-message and deleted-conversation behavior.
- Query-specific cache keys already separate opt-in requests, and the active loader supplies requestScope, bypassing shared cache. Preserve generation/owner guards across later pages and A→B→A.
- The actual runtime method is the domain override applied to `TldwApiClient.prototype`; changing its domain mapper plus the exported response type is sufficient. Preserve generated-assistant-image precedence.

## Minimum acceptance tests

- Real image-first frontend serialization → authorized endpoint/SQLite → provider failure → actual client mapper/mounted loader/mirror → Retry → reload, both text+image and image-only. Assert one canonical user, exact image bytes and exactly one provider user turn.
- Add a prior successful image turn; changed text, changed bytes/MIME/order, literal placeholder, edit-away-and-back, versioned/pinned placeholder, legacy primary blob, and missing provenance must remain safe.
- Corrupt attachment, failed ordered-image read with a primary blob, unsupported MIME, non-default detail and unrepresentable multipart inputs must not yield a false ACK or provider dispatch.
- Cross-page image recovery, oversized opt-in response, both standard listing branches, default/completion responses, unauthorized/wrong workspace/deleted resources and delayed-owner/ABA controls.

No additional native failure is asserted by this design review. UAT118 remains an automated mapper-to-mirror finding until the completed implementation receives targeted native verification.

Source references are relative to `/Users/macbook-dev/Documents/GitHub/tldw_server2/`: backend paths under `tldw_Server_API/app/` and frontend paths under `apps/packages/ui/src/` as identified in the proposed design.
