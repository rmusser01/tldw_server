# H1: selected chat history and fork ownership

Date: 2026-09-16. Design tracking: TASK-13261. Implementation tracking: TASK-13261.1.

Status: focused specification following the reviewed D1–D6 decisions. This document specifies H1; it does not claim implemented parity. The [review closure](2026-09-16-chatbook-chat-parity-review-closure.md) remains the authority for the full program. The [implementation plan](../../IMPLEMENTATION_PLAN_chatbook_h1_history_selection.md) turns this specification into five deliveries.

Implementation checkpoint, 2026-09-17: the shared contract, native/local owner adapters, mounted selection/review/restore UI, ordinary/overlay sends and supported native tracked-character sends have passed task review and required fix reviews. Safe fork projection, uncertain fork outcomes and final real-browser/owner qualification remain in progress or unstarted; H1 is not complete. The [verification record](../Reviews/CHATBOOK_H1_HISTORY_SELECTION_VERIFICATION_2026_09_17.md) distinguishes reviewed evidence and outstanding acceptance checks.

## 1. Outcome and scope

Choosing an assistant variant must change the history used by the next normal send and by Fork. Those actions capture stable message identities, a boundary and source revisions. They cannot reconstruct ancestry from the currently rendered array, a display index, matching text or a timestamp sort.

H1 delivers this behavior in the shared WebUI/full-page extension UI and its compact sidepanel controls. It repairs the current local fork's source isolation and prevents an uncertain server fork from automatically becoming another fork or changing storage owner. It provides the common result contract used by H2 and H3.

H1 does not implement atomic native forks (H2), complete local/temporary settings and asset recovery (H3), sync publication/enrollment (H4), independent model connection/cold WebUI installation (F02), browser execution hosts (D3), ACP controls or queue/voice lifecycle repairs (D5/D6). Their contracts constrain H1, but their implementation proofs are not H1 prerequisites. An unavailable mode stays explicitly unavailable; that is not parity credit.

### Global constraints

- Full parity targets are the WebUI and extension full-page UI; the compact sidepanel uses the same selection/action services and expands into the extension's own full-page UI.
- `tldw-agent` executes server-directed OS/tool actions in external environments. It has no Chatbook runtime, chat storage, synchronization, browser hosting or model-hosting responsibility.
- Preserve independent local ownership and optional synchronization. A server/account change cannot silently reassign a local conversation or operation.
- Preserve the client-managed context composer from TASK-188; do not introduce a monolithic server context-preview service.
- Reuse the repository's Python, Pydantic, DB backend, Dexie, TypeScript, Vitest and Playwright toolchains; add no runtime dependency for H1.
- Preserve existing request-scope cancellation and persistence rollback from TASK-13023, and integrate without overwriting TASK-13260.15 or the active UAT chat fixes.
- No normal-chat action may use a rendered index, timestamp order or text equality as message ancestry or mutation identity.
- H1 cannot advertise H2/H3 receipt recovery, rich fork fidelity, H4 sync compatibility or F02 independent generation as implemented.

## 2. Source baseline and existing seams

Reviewed dev pins: server `59049e094e0845a4611ea725ae19b7c1754ea709`; Chatbook `24094f23d59c7a9d3cfac964c19fd263bc0393b2`. The working checkout currently belongs to unrelated UAT work, so implementation must start in isolation and recheck dev before integration. References below describe the reviewed pin, not every unmerged checkout change. The previous temporary audit checkout has been removed; immutable Git objects remain the source of evidence.

On 2026-09-17, server dev was reverified unchanged and Chatbook dev advanced to `1c0327b3bb3d95b61e3e1b9a83b30e7030453ad6`. The [source refresh](2026-09-17-chatbook-chat-parity-source-refresh.md) records its model/settings/readiness delta. No identified H1 contract changes were introduced; the source references and original audit results below retain their historical pin.

| Existing path | Evidence and implication |
|---|---|
| [PlaygroundChat variant control](https://github.com/rmusser01/tldw_server/blob/59049e094e0845a4611ea725ae19b7c1754ea709/apps/packages/ui/src/components/Option/Playground/PlaygroundChat.tsx#L1149), [sidepanel control](https://github.com/rmusser01/tldw_server/blob/59049e094e0845a4611ea725ae19b7c1754ea709/apps/packages/ui/src/components/Sidepanel/Chat/body.tsx#L80) | Swiping changes rendered messages only. Both controls must update the shared selection. |
| [Full-page submit](https://github.com/rmusser01/tldw_server/blob/59049e094e0845a4611ea725ae19b7c1754ea709/apps/packages/ui/src/hooks/chat/useChatActions.ts#L3324), [sidepanel submit](https://github.com/rmusser01/tldw_server/blob/59049e094e0845a4611ea725ae19b7c1754ea709/apps/packages/ui/src/hooks/useMessage.tsx#L2901) | Separate message/history arrays currently enter the live normal mode. Update these mounted callers, not only unused extracted hooks. |
| [Dexie formatting](https://github.com/rmusser01/tldw_server/blob/59049e094e0845a4611ea725ae19b7c1754ea709/apps/packages/ui/src/db/dexie/helpers.ts#L242), [variant identity](https://github.com/rmusser01/tldw_server/blob/59049e094e0845a4611ea725ae19b7c1754ea709/apps/packages/ui/src/utils/message-variants.ts#L21) | Formatting picks the latest sibling; variant deduplication can merge distinct IDs with identical text. Neither behavior can be the selected path resolver. |
| [Local forks](https://github.com/rmusser01/tldw_server/blob/59049e094e0845a4611ea725ae19b7c1754ea709/apps/packages/ui/src/db/dexie/branch.ts#L5) | Existing transactions can be reused, but whole-record spreads, parent identity and the `sessionId` file key need repair. |
| [Server branch handler](https://github.com/rmusser01/tldw_server/blob/59049e094e0845a4611ea725ae19b7c1754ea709/apps/packages/ui/src/hooks/handlers/messageHandlers.ts#L423) | Current creation and message copying are separate requests; failure can trigger a second local implementation. H1 cannot make this an atomic fork merely by adding a client operation ID. |
| [Complete local history reader](https://github.com/rmusser01/tldw_server/blob/59049e094e0845a4611ea725ae19b7c1754ea709/apps/packages/ui/src/db/dexie/chat.ts#L124), [server display loader](https://github.com/rmusser01/tldw_server/blob/59049e094e0845a4611ea725ae19b7c1754ea709/apps/packages/ui/src/hooks/chat/useServerChatLoader.ts#L157) | Local rows can be read completely; the server display helper can stop at its page cap and must not certify a complete source. |
| [Normal mode composer](https://github.com/rmusser01/tldw_server/blob/59049e094e0845a4611ea725ae19b7c1754ea709/apps/packages/ui/src/hooks/chat-modes/normalChatMode.ts#L525) | Feed selected history into the existing model/context assembly. Preserve explicit comparison overrides and provider filtering. |
| [MessageStore](https://github.com/rmusser01/tldw_server/blob/59049e094e0845a4611ea725ae19b7c1754ea709/tldw_Server_API/app/core/DB_Management/chacha/message_store.py#L65), [resume fences](https://github.com/rmusser01/tldw_server/blob/59049e094e0845a4611ea725ae19b7c1754ea709/tldw_Server_API/app/core/DB_Management/chacha/conversation_resume_store.py#L454) | Reuse transactional history versioning and owner/scope fences; ordinary chat must not require character readiness. |
| [User persistence endpoint](https://github.com/rmusser01/tldw_server/blob/59049e094e0845a4611ea725ae19b7c1754ea709/tldw_Server_API/app/api/v1/endpoints/character_messages.py#L325), [normal completion schema](https://github.com/rmusser01/tldw_server/blob/59049e094e0845a4611ea725ae19b7c1754ea709/tldw_Server_API/app/api/v1/schemas/chat_request_schemas.py#L885) | Both client-managed persistence and server-owned completion need versioned selection admission. |

Related work: [TASK-188 context design](../superpowers/specs/2026-05-09-conversation-context-workflow-design.md), TASK-13023 scope rollback, TASK-13260.15 character/mirror reconciliation. The implementation baseline must also inspect changes to chronology, user acknowledgement and provider routing made by the active UAT work.

## 3. Ownership and shared contract

Four identities remain separate: authenticated persistence owner; global/workspace routing (`ChatScope`); view/client-session identity; model connection. The current `ChatScope` type represents only workspace routing. Use the existing verified server/account scope key facilities to construct an opaque `owner_key`; never put credentials in a bookmark, digest, log or provenance object. A server derives authorization from the request and database, and verifies any supplied owner descriptor rather than trusting it.

A durable local conversation is bound to its local profile and local owner ID. A mirrored server conversation remains server-owned and binds its server/account/dataset context as applicable. Temporary owners use memory only. A model connection is not part of storage authority. H1 uses current model routing and leaves independent connection setup to F02.

Existing unscoped data must not acquire authority from whichever account happens to be logged in during migration. Existing purely local histories remain under their browser-local profile. A legacy mirror carrying only `server_chat_id`, without a verifiable server/account binding, is an unbound mirror: retain read/export access, but require an explicit connection binding and authorized owner lookup before a server mutation. Do not treat an ID match as proof of ownership or upload its contents to establish the binding. Reuse already verified scope metadata where present.

Use these wire names in the new shared TypeScript and strict Pydantic types. The types are new interfaces; existing message/storage types remain adapters around them.

```typescript
type HistoryCursorV1 =
  | { kind: "after_message"; message_id: string }
  | { kind: "before_message"; message_id: string }
  | { kind: "empty" }

type HistoryInterpretationV1 =
  | { kind: "parent_graph_v1" }
  | { kind: "legacy_linear_v1"; projection_id: string }

type HistoryMessageRevisionV1 = { id: string; revision: string }

type HistorySelectionV1 = {
  version: 1
  owner_key: string
  conversation_id: string
  interpretation: HistoryInterpretationV1
  cursor: HistoryCursorV1
  selection_revision: number
  purpose: "send" | "fork"
  messages: HistoryMessageRevisionV1[] // ordered retained path, not visible rows
  fences: {
    conversation: string
    history: string
    settings: string
  }
  storage_context_digest: string
  request_context_digest: string
  selection_digest: string
}
```

`HistorySelectionSnapshotV1` contains `version`, `owner_key`, `conversation_id`, the three fences, the complete lightweight ordered source manifest, `source_digest`, and interpretation status. Manifest rows include ID, revision, parent ID, role, settled/live state, and identities/revisions of required metadata/assets. Content is loaded for the selected rows through the same coherent owner snapshot. Never inline full binary assets into a legacy-review manifest. Local revisions are stable canonical content/metadata digests; server revisions include the row version and relevant metadata/asset revision identities. This avoids pretending `createdAt` is a mutation revision.

`selection_digest` hashes canonical JSON containing owner/conversation, interpretation, cursor, purpose, ordered IDs/revisions and both context digests. Selection revision is carried as the originating view fence; it does not create a global server cursor. Use a documented canonical array/tuple encoding with explicit nulls and deterministic field ordering, SHA-256 via existing Python `hashlib` and `@noble/hashes`, and cross-language fixtures. No digest authorizes a read or write by itself.

The exact selection tuple is `[1, owner_key, conversation_id, [interpretation.kind, projection_id_or_null], [cursor.kind, message_id_or_null], purpose, [[message_id, revision], ...], storage_context_digest, request_context_digest]`. Serialize compact JSON as UTF-8 with literal Unicode, no whitespace and no Unicode normalization. All variable tuple leaves are strings or null, so floating-point and object-key ordering are absent from this shared encoding. Each owner defines its context projection/hash over its own normalized inputs; the other owner treats that context hash as opaque rather than reserializing a foreign settings object.

`storage_context_digest` covers settings, accepted character behavior and required asset revisions authoritative in the persistence owner. That owner recomputes and validates it. `request_context_digest` records the client's exact prepared composer/provider inputs and requested copy policy; the server treats it as client provenance, not evidence that it has validated local prompts, overlays or credentials. The client binds these inputs to its settings/model-connection/request-scope lease and revalidates that lease before admission and again before provider dispatch. Server-owned completions validate their request context from actual supplied fields plus the owner snapshot. This separation preserves the client composer without asking the server to reconstruct client-only settings.

For fork, both digests cover only the retained fork projection/policy; source drafts, prefill and old summary/compaction content are excluded. Separate a changed row/settings counter from a changed relevant projection: reread and compare relevant values when a fence changes. Source deletion, changed retained content or ancestry, incompatible required context, and changed required asset identity reject an unaccepted action. An unrelated view's selection change does not. Check the originating selection revision after asynchronous preparation, immediately before dispatching owner admission. Once dispatched, admission is an immutable pending intent: a remote owner cannot observe subsequent local swipes, so a later swipe cannot be reported as proof that admission was rejected or rolled back.

### View selection and bookmarks

`HistoryViewSelectionV1` is `{ view_session_id, owner_key, conversation_id, interpretation, cursor, selection_revision }`. Each live view owns a monotonic revision. A new explicit cursor choice increments it, including choosing before-first or empty. A persisted reopen bookmark is keyed by `[profile_id, client_session_id, owner_key, conversation_id]`; a new independent view gets its own client-session identity. A synced hint may initialize an absent bookmark only. It does not update existing bookmarks or open views.

Restore selection before deriving rendered variants and normal history. Missing/deleted bookmark targets produce an explicit stale-selection state with branch choice; they do not silently choose latest. On an uninitialized valid graph, a documented initial tip/default hint may initialize the bookmark. A legacy transcript requires the review below before a history-dependent mutation. `before_message(first_root)` and `empty` resolve to `[]`, even when other messages exist.

Swiping a variant selects that stable variant ID as the boundary. Descendants belonging to a different variant are not spliced into its path. Selecting a known descendant resolves its own ancestor chain. All alternatives remain available in branch/variant controls. Completion persists using the accepted parent, and advances the originating view only if owner, conversation, view/session and selection revision still match. It leaves another view's selection unchanged.

## 4. Complete snapshots and legacy review

New versioned writes use explicit parent relationships and protected interpretation provenance. A complete, valid unique chain can be interpreted automatically. An already versioned valid graph may branch and is resolved by its cursor. Unversioned parentless, mixed, conflicting, cyclic or orphaned rows are not converted into ancestry by sorting. A single live root or an empty source is unambiguous. Cross-owner/conversation parents are invalid, never imported by following their IDs.

For ambiguous legacy data, reading, browsing and export continue. On the first dependent send/fork/rewind-continuation, show **Review conversation history**: the complete owner-ordered row list, clear alternatives and the proposed included path. The user can select an ordered subset, with omissions visibly retained as alternatives. Default chronological presentation is not confirmation. The review can be virtualized, but confirmation covers all source identities/revisions, not only rendered rows.

Server presentation order is `timestamp, last_modified, id`; local order is `createdAt, id`. Fetch the complete lightweight manifest in one coherent owner read. If selected content is fetched separately, bind it to that manifest and reject drift before admission. Explicit resource exhaustion returns an error, never an apparently complete truncated manifest. The acceptance test includes more than 20,000 rows to exceed current display-pagination assumptions; binary payloads are not loaded to render that review.

`LegacyHistoryProjectionConfirmV1` carries version, owner/conversation, complete source digest and fences, the complete ordered ID/revision manifest, chosen ordered path IDs, tagged cursor and originating selection revision. Reject duplicate or missing IDs, cross-conversation members and a cursor outside its chosen path. In one owner transaction, recheck the complete source, then insert an immutable `legacy_linear_v1` projection. Do not rewrite message IDs/content or delete excluded alternatives.

Use a small `conversation_history_projections` table in the existing native chat database, and a corresponding Dexie table for local owners. Fields: immutable projection ID; conversation and owner identity; interpretation version; complete reviewed source digest/history fence; reviewed source member revisions; ordered path; projection digest; creation time. Unique keys bind owner/conversation/projection ID. Multiple reviewed projections may coexist. This table stores legacy interpretation only; it is not a generic turn store or a settings extension.

Browser bookmarks reference the accepted projection only after owner commit. Local projection and bookmark commit together in Dexie. For a server owner, the server stores the immutable projection transactionally; the browser then commits its bookmark locally. A browser failure after server confirmation leaves an unused projection, not a changed global leaf; retrying the same projection confirmation must return the same matching record. Freeze a client-generated projection ID before confirmation and reject same-ID/different-content reuse. After authorization, look up a matching existing confirmation before applying fresh-source CAS; unrelated later appends cannot turn a successful confirmation's replay into a new review. Reauthorize access to the conversation before returning it. No network call spans a database transaction.

New descendants bind the accepted `projection_id` in protected admission metadata. Their ancestry follows that immutable base plus explicit new edges. If two reviews select different paths through a shared legacy row, one view's later confirmation does not replace the other's base. Reopen and continuation must select the matching base; do not combine edges from different interpretations. Existing accepted projections remain interpretable after append, while edited/deleted retained members invalidate a new mutation until the user resolves that change.

Keep legacy interpretation separate from generic settings writes and sync imports. No public metadata/settings body can manufacture an accepted projection or acceptance record. H4 supplies versioned cross-owner transfer later. An existing sync-routed owner without that capability must reject dependent H1 mutations before any legacy acceptance or message write, rather than bypassing its sync owner.

## 5. Server admission and settlement

Add two narrow routes alongside existing conversation history routes in `endpoints/chat.py`:

| Route | Request and result |
|---|---|
| `POST /api/v1/chat/conversations/{conversation_id}/history/selection` | Resolve a cursor/interpretation/purpose under request owner/workspace scope. Return a capture containing the complete snapshot, selected rows/IDs and storage-context digest, or structured `legacy_review_required`, `stale_selection`, `unsupported_history_capability` or resource error. The client finalizes the request digest after composition; this read does not accept a turn or call a provider. |
| `POST /api/v1/chat/conversations/{conversation_id}/history/legacy-projection` | Accept `LegacyHistoryProjectionConfirmV1`; return the immutable projection and accepted cursor after CAS. No generation or content rewrite. |

Successful versioned selection resolution is the capability handshake. The new clients do not send versioned mutations after a missing/unsupported route or malformed/older response. Rollout enables this behavior only for a server deployment whose serving nodes support the same version. Existing unversioned API clients keep their current behavior; they cannot overwrite protected H1 interpretation/admission metadata.

Add caller-connection-aware methods to `MessageStore`, exposed by `CharactersRAGDB`: complete snapshot read, selection validation, legacy confirmation, selected user append and accepted assistant settlement. Reuse `db.transaction()` and the backend's owner/scope checks. A PostgreSQL snapshot uses one joined statement or transaction-bound reads with a conversation fence. Respect the existing message-before-conversation lock order for edits; admission must not hold a conversation lock then lock old message/metadata rows. It inserts new rows under the fence. Test an actual racing edit, not only a mocked lock call.

### Client-managed persistence

Extend `MessageCreate` in `chat_session_schemas.py` with optional `tldw_history_selection_v1` for user admission and `tldw_history_admission_v1` for assistant settlement, with role/field validation. For a versioned user append, the owner validates the source selection, derives the parent from the accepted path, inserts the user message and protected admission metadata atomically, and returns its ID/version plus `HistoryAdmissionV1`. If an explicit `parent_message_id` conflicts, reject it. Empty/before-first selections derive null, never the most recent row.

The mounted ordinary-chat path currently calls generic `addChatMessage` for both user and assistant after generation. Move the owned user admission before inference and route the generic assistant append through the same accepted-settlement store operation as character streaming. Resolve/create a required server conversation using existing scoped preparation first; a locally owned conversation stays local. Suppress the old duplicate post-generation user append once admission succeeded. Test ordinary chats with no character as well as character conversations; changing only the character streaming endpoint misses normal chat.

`HistoryAdmissionV1` contains version, owner/conversation, accepted selection digest and immutable manifest, input message ID/version and originating view revision. The server generates it inside admission. A supplied digest is not evidence of acceptance: settlement reads the stored protected input metadata. The client freezes the request composed from that accepted snapshot; no second timestamp history fetch or content-signature dedup occurs. If the client's composer/connection lease changed while user admission was in flight, do not dispatch a provider or silently recompute under a new connection. Retain the accepted unsent user turn for explicit continuation; existing scope rollback may remove only its own demonstrably uncommitted mirror effects.

Extend `CharacterChatStreamPersistRequest` with the same versioned admission reference. Both generic assistant append and character stream persistence call the shared insertion/update operation: verify live owned conversation, input parent identity/version, matching protected admission, and the expected assistant identity for any retry. Persist the assistant against that input and copy inert accepted-path provenance. An edited/deleted accepted parent returns `stale_parent`; keep the generated client result available for explicit recovery. Unrelated branches and another view's cursor cannot redirect or reject settlement. H1 adds no automatic resend after an uncertain user append or assistant save.

Preserve existing scope leases and stable-ID retry guards. Public arbitrary message metadata updates cannot create or alter the protected H1 namespace, including through ordinary completion payloads. Existing sync materialization may carry ordinary data, but imported admission-shaped values are not local owner authority.

Keep uncertain admission, accepted-but-unsent input and generated-but-unsaved text in narrow recovery records alongside the originating scoped browser bookmark. These are client observations, not accepted transcript rows or owner receipts. Persist dispatch intent before calling the owner; preserve immutable operation/owner/conversation identity and credential-free result text through navigation. Reopening under the same profile and verified owner can inspect or copy those records without replaying them. Bookmark and legacy-confirmation updates must preserve recovery records; account changes cannot expose or reassign them. Do not create a native mirror or insert a synthetic assistant row merely to retain an uncertain result.

### Server-owned completion

Extend `ChatCompletionRequest` with the same optional `tldw_history_selection_v1`. In `build_context_and_messages`, freeze validated selected rows into the existing `continuation_runtime` before dispatch. Persist the new current-turn input chain with explicit parents and admission metadata atomically; its final input becomes `assistant_parent_message_id`. Use the selected rows as prior history and preserve downstream system/character/worldbook/prompt composition. Validate the versioned request's current-turn messages as new input, not a second copy of historical rows.

Route all resulting assistant/tool/continuation writes through the accepted parent binding, including the existing continuation path that currently initializes its parent to null. Recheck parent authority during settlement. Exclude the new internal selection/admission fields from provider call parameters and public provider extension forwarding.

For an already client-composed inference request, use the existing stateless path: `save_to_db=false`, complete authorized provider messages, and no local/foreign conversation ID or persistence selection IDs. The inference endpoint is not asked to resolve another owner's storage. Independent model selection remains F02.

The current tracked-character browser branch is server-managed and remains separate from the ordinary/overlay client composer. Integrate it with this versioned server-owned completion path using only new current inputs and frozen explicit request values. The existing saved-behavior adapter supplies supported native character context inside owner admission; a mutable card read or legacy timestamp stream cannot substitute for it. The server owns the input/result writes, so the browser must not perform the separate append/settlement used by client-managed inference. Stage3 includes this distinct integration and cannot finish with only its ordinary-chat task complete.

Versioned streaming must acknowledge successful settlement with the canonical saved result ID even when optional legacy stream metadata is disabled. This acknowledgement is emitted only after the owner save succeeds and is bound to the original admitted operation. Provider-supplied admission or result identity fields cannot become owner acknowledgements. A stop marker, successful provider completion or finished stream without that identity is not proof of persistence; retain the known admission and generated text as uncertain. Preserve the metadata flag's behavior for unversioned requests.

### Active sync and compatibility

The current default-personal-dataset helpers do not prove individual conversation enrollment. Existing sync routing must be respected, but cannot be used as permission to materialize H1 state natively. Before mutation, route through an adapter that supports retained selection/admission, or return `unsupported_history_capability` without writes. In H1 that adapter is not implemented; read/export remain available. H4 adds enrollment and transfer. Do not silently downgrade a versioned request to old timestamp history or bypass an enrolled owner.

## 6. Shared browser integration

Introduce one pure history selection resolver, one owner adapter service, and one view-selection hook. The owner adapter reads complete local rows in Dexie or calls the server selection endpoint. It does not add a new runtime host. Rendered messages and normal provider history are projections of the selected source; persisted source rows retain alternatives.

Preparation has two explicit phases. `captureHistorySnapshot` returns `HistorySelectionCaptureV1`: complete owner snapshot, resolved path/rows, originating view fence, purpose and owner-validated storage-context digest. It does not yet claim a final request digest. The existing composer then prepares the exact payload from those rows and its captured inputs. `finalizeHistorySelection` validates the captured view/preparation lease and combines the capture with that immutable prepared request (or fork policy) to produce `HistorySelectionV1`. Only this finalized value is submitted for owner admission. Neither phase reads ambient draft/global state to fill a missing input. Legacy confirmation yields an accepted interpretation and a refreshed capture, not an already-composed request.

Update both live submission paths (`useChatActions`, `useMessage`), both variant controls, all local/server hydration paths, and the playground/sidepanel session snapshots. Selection capture precedes asynchronous attachments/context work; a changed originating revision before admission dispatch produces `stale_selection` without dispatch. After dispatch, reconcile the actual returned or unknown owner outcome and retain the immutable binding even if the view navigates. View selection changes alone do not invalidate the already dispatched intent; independent auth/connection/context lease invalidation still prevents unauthorized provider work as described above. Completion updates only a still-matching view.

The provider adapter explicitly maps local `images` and historical singular `image` forms without changing model capabilities. Preserve current filtering for system rows and image-generation turns; record that behavior in regression tests. Identity-based selection replaces content dedup for versioned history, so identical text with different IDs stays distinct.

Replace the branch handler's `(index: number) => Promise<string | null>` contract with `(request: ForkRequestV1) => Promise<ForkResultV1>`. A control translates its clicked row to a stable boundary immediately, including rendered greeting offsets. Selection and owner services validate IDs; no downstream index slicing is permitted.

Adjacent H1 controls use scoped stable IDs for plain local edits and leaf deletion. The deletion transaction rejects a message with persisted children, including hidden alternatives; it does not cascade or reparent them. Only the initiating explicit-graph view may step from a successfully deleted boundary to its captured parent or empty root, while other views keep explicit stale bookmarks. Invalidated reviewed legacy interpretations require review. Native selected-message edit/delete and sibling-producing regenerate/edit-and-resend remain capability-gated before requests, copies or display mutation until their owner-safe mutation/preparation seams are implemented. These limits are broader parity work, not delivered action parity.

Comparison keeps an explicit model-qualified input: common prompts plus that model's responses and per-model prompts. `CompareHistorySelectionV1` carries version, owner/conversation, model ID, cluster/boundary, ordered IDs/revisions, source fences and the same two context digests; its digest includes the model/cluster selection. Validate those IDs and their order against the existing comparison semantics, not normal `parent_graph_v1` ancestry. In a two-model transcript a retained common prompt can currently parent to the other model's last response, which is intentionally absent from this projection.

Materializing that comparison selection into an ordinary child creates an explicit linear chain in the validated projection order. Original compare edges are inert provenance only; do not dereference an omitted model's parent or reject a valid comparison copy for omitting it. `ForkRequestV1` tags its input as normal selection or comparison selection. Update the live compare branch in `useChatActions` and its utility, as well as comparison controls. Do not route comparison through normal visible history or count an unused extracted hook as production coverage.

The compact sidepanel uses the same failure/legacy/pending states. A large review may expand into the extension full-page UI carrying only the scoped conversation/bookmark/projection reference. H1 does not promise active-stream host continuity across this expansion; D3 owns that work. No server/local or account reassignment accompanies navigation.

## 7. Safe current local fork projection

One allowlisted copy builder consumes a validated ordered path and builds an independent child within the existing Dexie transaction. Allocate every new message ID before rewriting references. Reject missing/duplicate members and a changed source; never silently filter IDs, reorder them by timestamp or try a second snapshot copier after failure. Retain supported settled complete/stopped/failed content; live/runtime-only state cannot be copied as settled history.

| State | H1 treatment |
|---|---|
| Conversation | Fresh local ID/time, branch title, unpinned state and supported declarative prompt/model display data. `server_chat_id`, server mutation versions and sync enrollment are absent. Child root is independent. Source ancestry is a scoped inert provenance descriptor, never mutation or upload authority. |
| Messages | Fresh IDs/history ID; preserve retained role/name/text/time and explicitly supported presentation/content. Remap every retained parent, recompute depth and reset server IDs/versions. A reviewed legacy path materializes as an explicit child chain. |
| Variants/comparison | Copy only the selected path. Retained references point to child IDs. For comparison, construct an explicit chain in validated model-projection order rather than remapping cross-model parents. Normalize comparison-only runtime tags when creating an ordinary child; never copy compare-state control maps by spread. |
| Prompt/settings | Preserve only currently representable declarative prompt text, with no dereferenceable source prompt authority. Reset source drafts, one-shot/pinned prefill and old summary/compaction contents. Required non-default character/settings snapshots that H1 cannot atomically retain block this fork before writing; H3 implements them. |
| Files | Write `SessionFiles.sessionId = childId`. Allocate fresh local file identities and deep-own in-record bytes/text. Preserve safe completed derived content and `retrievalEnabled`; remove source draft/job/batch/idempotency/processing-control references. Never copy a `history_id` property as a substitute key. |
| Images/documents/citations | Preserve self-contained content or independently owned references supported by the existing adapter. Remap local file links. A remote protected URL, source-only document ID or path is not ownership. Required unavailable/unsupported assets block before write in H1; reviewed degraded and staged rich copies are H3. |
| Rich metadata | Explicit allowlist only. Retain safe content/presentation where represented; omit source execution IDs, active tools, approvals, secrets, queue, live effects and trace/action authority. Historical usage may be displayed as source provenance and is not charged as child usage. |

H1 does not copy arbitrary external Plasmo settings and call it atomic. Read required settings before preparation, reject unsupported required state, then revalidate relevant captured state before commit. Local acceptance authority must be in the owner transaction; any external settings read that cannot be fenced cannot certify support. This intentionally bounds H1's current safe-copy mode. H3 establishes canonical local settings and cache recovery for broader modes.

The strict settings path must establish that its actual storage instance uses a persistent backend. A silent in-memory fallback or an extension client whose reads/writes are no-ops cannot prove persistent settings absent and cannot initialize an empty baseline. Browser API presence alone does not certify an instance constructed earlier against a fallback. Ordinary fallback construction and server-side rendering remain compatible; strict fork certification fails explicitly when persistence is unavailable.

The existing local settings writer may maintain a small write guard on its owning history record to fence this eligibility check. The guard contains no settings payload and must account for every in-flight writer; an older writer completing after a newer one cannot disappear from the fence. Failed or interrupted writes leave fork eligibility explicitly unavailable while ordinary settings edits remain possible. A settings-read error is not proof of absence. Changes only to excluded summary/compaction contents require a relevant-policy reread, not rejection solely because a write counter advanced. Full repair of interrupted external-state writes remains H3 work.

Imports strip incoming owner-control guards. Replacing an existing history preserves that destination's guard within the replacement transaction; it cannot erase a pending write or inherit an imported initialization marker. Ordinary history deletion/clear also rejects a pending settings guard before removing any messages, so delete/undo cannot restore an older snapshot without the live guard. Settled deletion and import remain supported. An interrupted write can therefore also prevent deletion until its uncertainty is resolved; H1 does not invent a recovery receipt or silently discard that fence.

Local per-conversation settings use the browser-local storage area: implicit browser-sync writes cannot participate in that local owner fence. Existing browser-sync values migrate once through the settings seam without deleting the legacy copy or losing required context. An initialized empty local baseline is distinct from unreadable/uninitialized state, and later browser-sync values cannot override it. Migration and ordinary edits share the guard; stale migration reads cannot overwrite newer local edits. Server/scratch/global preferences retain their existing storage behavior. Explicit server/local synchronization remains H4 work.

Source and child deletion/edit/reopen tests must prove isolation, including removal of a copied file that previously carried ingestion control IDs. H1 uses only asset representations whose existing storage ownership is demonstrably independent; it does not add a second upload/ingestion pipeline.

## 8. Fork operation and uncertain outcomes

`ForkRequestV1` contains `operation_id`, owner namespace, the immutable selection (or explicit comparison projection), destination owner and canonical request digest. Allocate an unpredictable operation ID and record the immutable request before any side effect. A local transaction claims the operation and prevents another view from dispatching the same prepared action. A pending operation is not a server receipt.

```typescript
type ForkResultV1 =
  | { state: "committed"; operation_id: string; owner_key: string;
      child_id: string; message_map: Record<string, string> }
  | { state: "legacy_completed"; operation_id: string; owner_key: string;
      child_id: string } // acknowledged old multi-request path; no receipt claim
  | { state: "rejected" | "blocked"; operation_id: string;
      owner_key: string; code: string }
  | { state: "unknown" | "partial"; operation_id: string;
      owner_key: string; candidate_child_id?: string; code: string }
```

H1's supported local transaction may return `committed` after its database commit. This is an observed commit result, not H3's durable receipt/recovery guarantee. The current multi-request server adapter may return `legacy_completed` only after every required request acknowledges its limited supported projection; it cannot advertise atomic or rich parity. H2 replaces it with native receipt-backed completion. Versioned selection validation is necessary but does not make that old sequence an owner-atomic source copy.

The limited native adapter requires affirmative context eligibility from the existing owner capture. Derive this narrow proof from the same coherent settings/assistant/behavior snapshot and bind it to the returned storage-context digest; a digest alone says nothing about representability. Do not infer a plain source from unavailable browser caches or unscoped metadata reads. Missing proof from an older server is unsupported for this copy. Required unsupported context blocks before creation, while supported plain content uses explicit empty assistant identity so workspace defaults cannot change the child. This read-only eligibility addition is neither a fork receipt nor a settings-copy service; normal send capture keeps its existing semantics.

Record states `prepared`, `dispatching`, `partial`, `unknown`, `completed` or `rejected` under the original namespace. Before dispatch, capability/source rejection is definite. Once creation could have reached the server, timeout, disconnect, cancellation or failed lookup is unknown. A known created child with a later failed copy is partial. Persist any known candidate child ID, the original operation ID/digest and failure stage; preserve the source view and storage owner. Do not navigate to a partial child as a successful fork, delete it automatically, retry with a new ID or create a local fallback.

On restart, a `dispatching` record becomes unknown and is not resent. Show **Fork status unknown** or **Fork incomplete**, retain any known server link for inspection, and explain that reliable result reconciliation requires H2. A user may deliberately start another operation after reviewing that state; automatic retry/fallback is prohibited. Scope changes hide operations from the new account without reassigning or erasing their original owner record.

For local copy, a definite transaction abort returns rejection. An uncertain commit observation stays unknown. The handler must not try the existing second snapshot copier. Temporary requests use in-memory pending state; forking alone must not create durable history or operation records. H3 supplies complete temporary fork/Save semantics.

## 9. Acceptance and release evidence

| ID | Required behavior | Delivery evidence |
|---|---|---|
| H1-A | Two views choose different variants; normal sends, fork boundaries and late completion retain the correct path. Same-text distinct IDs remain distinct. | Pure resolver fixtures, mounted submit/control tests, both browser shells. |
| H1-B | Before-first, empty, deleted bookmark, equal timestamps and reopened sessions have explicit behavior. | Resolver, bookmark and hydration tests. |
| H1-C | Complete legacy review covers alternatives beyond 20,000 rows; CAS rejects source drift; two immutable reviews coexist; subsequent explicit descendants reopen correctly. | Real owner transactions on Dexie and SQLite; PostgreSQL owner/transaction integration. |
| H1-D | Client-managed and server-owned normal admission bind user/assistant/tool continuation to accepted parent and preserve the existing composer. Forged admission and stale parent fail. | Endpoint/service tests, call-parameter regression, scope invalidation and held-completion tests. |
| H1-E | Local fork remaps exact membership/order, parents and files; child actions cannot mutate source; unsupported required settings/assets cause no writes. | Real IndexedDB browser transactions plus pure allowlist tests and reopened source/child assertions. |
| H1-F | Lost response/partial copy retains owner/op/digest across reload; no local fallback, second implementation or automatic replay. | Deferred API failure tests, operation store/reload tests, visible browser state. |
| H1-G | Comparison remains model-qualified, including two common rounds and a per-model follow-up when the other model completed last; the ordinary child has a complete explicit chain. Sidepanel/full-page controls share contracts and account/workspace routing remains distinct. | Live call-path integration and existing compare/scope regression suites. |
| H1-H | Old server and unsupported sync-routed modes fail before dependent writes; ordinary unversioned clients retain compatibility; new internal fields never reach providers. | API/client capability, forged metadata and serialization tests. |

Implementation completion requires these behavior proofs, relevant TypeScript/Python checks, Bandit on touched Python, an independent design/code review and recorded skips/limitations. Docs-only review does not satisfy them. H1 success leaves the program's other parity rows open until their own implementation and live qualification pass.

## 10. Focused specification review record

The concrete spec and plan received separate server/DB and browser/local-owner reviews, followed by correction reviews. Four P2 findings were resolved:

| Finding | Adopted repair |
|---|---|
| Owner cannot validate client-only composer settings from one hash. | Separate owner storage context from client prepared-request context and validate each under its own authority/lease. |
| A remote owner cannot see a local swipe while admission is in flight. | Freeze at admission dispatch; reconcile accepted/unknown results without treating later navigation as rejection. |
| Valid comparison projections can exclude a stored cross-model parent. | Validate comparison order separately and materialize an ordinary child chain in that order. |
| A capture function cannot finalize a request digest before receiving composed inputs. | Capture snapshot/rows first; compose; finalize with explicit immutable inputs and lease checks. |

The final review also verified the ordinary, character-free `addChatMessage` path is covered before inference and during assistant settlement. Root review clarified unbound legacy mirror migration and fixed the exact cross-language selection tuple. The final independent correction reviews reported **zero remaining P1/P2 findings** in their scopes; root checked the combined requirements and task coverage.

Document checks passed for the two new artifacts: 15 pinned source links, five local links, all eight H1 gates, five unstarted implementation stages, matching global constraints, code-fence/whitespace checks and explicit distinction between existing and proposed files. Both remote dev heads were rechecked and still match the pins in section 2. These are source/document checks; no application tests or implementation acceptance ran. Bandit is not applicable to this Markdown-only change. TASK-13261.1 remains To Do.
