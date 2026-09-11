# TASK-12020.40 Recipient Shared-Workspace Data Plane for Research Workspace

**Task:** TASK-12020.40

**Date:** 2026-08-21

**Status:** Approved for implementation

## Executive Decision

`/research-workspace?shared={share_id}` will use a dedicated recipient-facing
shared-workspace data plane. A valid share renders the owner's canonical
workspace identity and source membership while storing chat history in the
recipient's conversation database. Invalid, unauthorized, deleted, and revoked
shares fail closed and never fall back to a recipient-local workspace.

The implementation keeps the existing Research Workspace route and existing
`/api/v1/sharing/shared-with-me/{share_id}` API family. It does not create a
parallel workspace identity, redirect to another route, or hydrate the local
Research Workspace store while shared mode is unresolved or active.

Shared mode in this task is read and grounded-chat only. Existing share access
tiers remain policy ceilings, but source and workspace mutations are disabled
until a later collaboration task implements and validates them. Durable clone
status remains TASK-12020.41.

## Problem

The current route reads share metadata but then initializes the ordinary local
Research Workspace. This can display the recipient's unrelated sources and chat
under a shared-workspace banner. The current shared chat endpoint also invokes
the owner's RAG pipeline without an explicit media allowlist derived from the
shared workspace membership. That permits unrelated owner media to enter
retrieval or citations.

Additional verified gaps affect the same trust boundary:

- Sharing endpoints apply `rbac_rate_limit(...)`, but rate limiting is not an
  authorization check.
- `SharedWorkspaceDBResolver` rejects missing or revoked shares but does not
  establish team or organization membership itself.
- The shared media endpoint addresses owner media by raw `media_id` and returns
  unbounded content.
- Shared source listing is unbounded.
- `LLMGenerator` can fall back to source excerpts after provider failure, which
  makes a failed model call look like a generated answer.
- Existing conversation source-reference lookup is not unique, so it cannot by
  itself guarantee one conversation per recipient and share.
- Existing RAG message-metadata helpers are best-effort and cannot guarantee
  atomic answer-and-citation persistence.

## Goals

1. Render only the canonical shared workspace identity, source set, and
   recipient-owned shared-chat history.
2. Enforce global permission, share membership, access tier, revocation, and
   source scope before reading owner data or invoking a model.
3. Guarantee that retrieval and citations are constrained to queryable media
   currently attached to the shared workspace.
4. Provide bounded source listing, source inspection, chat history, and
   diagnostic responses suitable for large workspaces.
5. Make shared chat idempotent under retries and safe under concurrent requests,
   process failure, source mutation, and revocation.
6. Preserve recipient chat continuity across reloads without copying owner
   notes, chats, or workspace records into the recipient database.
7. Verify behavior against SQLite, PostgreSQL, a real backend, the real WebUI,
   and Chrome DevTools Protocol automation.

## Non-Goals

- Recipient source, workspace, note, or artifact mutation.
- Real-time collaborative editing.
- Streaming shared-chat responses.
- Durable clone execution or clone status; TASK-12020.41 owns that work.
- Browser-extension capture into a shared workspace.
- MCP, ACP, sandbox, project-root, or tool execution in recipient shared mode.
- Refactoring the complete local Research Workspace component.
- Merging Research Workspace with the separate `/research` product.
- Adding a route alias, redirect, or compatibility route.

## Canonical Workspace Alignment

This design follows
`Docs/Design/Research_Workspace_Shared_Workspace_Model_Contract_2026_05.md`.
Research Workspace remains the research UI shell over a canonical Workspace.
Sharing owns share records, access tiers, shared-with-me discovery, and
cross-user database resolution. Workspace Core remains the source of workspace
identity and source membership. MCP Hub, ACP, and Sandbox remain core dimensions
of the broader Workspace model but do not participate in this recipient data
plane.

The data ownership split is:

| Data | Authoritative owner |
| --- | --- |
| Share record, scope, access tier, revocation | AuthNZ Sharing repository |
| Workspace identity and source membership | Owner ChaChaNotes database |
| Media text, chunks, and source readiness | Owner Media database and Jobs projection |
| Embeddings | Owner embedding namespace |
| Provider credentials and model choice | Recipient or server configuration resolved for the authenticated recipient, never inferred from content ownership |
| Shared-chat conversation, messages, citations, and receipts | Recipient ChaChaNotes database |

Stored recipient references to a share, owner, or owner workspace are historical
identifiers only. They never authorize access and have no cross-database foreign
key to owner or AuthNZ records.

## Architecture

The backend adds three focused units:

1. `SharedWorkspaceAccessService` resolves a share, enforces permission and
   membership, computes allowed actions, sanitizes recipient-visible metadata,
   and opens owner databases only after authorization.
2. `SharedWorkspaceChatService` resolves source scope, performs retrieval,
   validates provenance, invokes generation, and coordinates revalidation and
   persistence.
3. `SharedWorkspaceChatStore`, implemented inside `core/DB_Management`, owns
   recipient-side thread mapping, request receipts, fenced leases, and strict
   transactional message/citation writes. Sharing services contain no raw SQL.

The frontend adds a lightweight `ResearchWorkspaceRouteGate` shared by both
route wrappers and a dedicated `SharedResearchWorkspace` surface. The existing
local `ResearchWorkspace` remains lazy-loaded and unchanged except for removal
of obsolete fail-open shared-state handling.

```text
/research-workspace?shared=42
          |
          v
ResearchWorkspaceRouteGate
          |
          +-- no shared parameter --> lazy local ResearchWorkspace
          |
          +-- invalid shared id ----> scoped invalid-share state
          |
          +-- valid shared id ------> SharedResearchWorkspace
                                           |
                                           v
                    /api/v1/sharing/shared-with-me/42/...
                                           |
                     +---------------------+---------------------+
                     |                                           |
             owner source databases                       recipient chat DB
                     |                                           |
             owner embedding namespace                 thread + receipts + messages
```

## Authorization and Disclosure Boundaries

Every shared workspace read or chat endpoint has both:

- `require_permissions("sharing.read")`; and
- the existing sharing rate limit dependency.

Clone retains its separate `sharing.clone` permission and remains outside this
task's UI.

The access service performs these checks in order:

1. Validate the authenticated principal and global permission.
2. Resolve the unrevoked share and current accessor membership with one
   authoritative AuthNZ repository query.
3. Collapse missing, revoked, inactive-scope, suspended-membership, and
   out-of-scope results to the same denial.
4. Do not use token or request-state team/organization claims as the current
   membership authority.
5. Resolve the owner workspace before opening media or conversation data.
6. Compute server-authoritative `allowed_actions` from the access tier and
   implementation availability.

A missing global permission returns `403`. A missing share, revoked share,
deleted workspace, or out-of-scope accessor returns the same neutral `404`
response. This prevents share enumeration. Operational failures after access is
established return a typed `503` rather than masquerading as a missing share.

`GET /sharing/shared-with-me` uses the same authoritative active-membership
repository query rather than iterating team/organization IDs from token claims,
so a suspended or removed member stops seeing stale share entries immediately.

The owner may open their own share URL, but the URL still renders the recipient
shared view. Editing requires the ordinary local workspace route.

Authorization is re-resolved before provider generation and again immediately
before recipient persistence. Source membership and content snapshots are
revalidated at the same boundaries. Because share state and recipient messages
live in different databases, revocation cannot be made perfectly serializable
without a cross-database locking protocol. The contract therefore guarantees
checks at every controllable disclosure boundary, but does not claim to recall
content already sent to a model or saved by a recipient while authorized.

## API Design

All recipient endpoints remain under:

```text
/api/v1/sharing/shared-with-me/{share_id}
```

All new response models are explicit Pydantic schemas with unknown request
fields forbidden. The API never returns owner database paths, raw provider
configuration, owner notes, owner conversations, raw internal diagnostics, or
owner `media_id` values.

### Bootstrap

```text
GET /api/v1/sharing/shared-with-me/{share_id}/workspace
```

This becomes the single bounded bootstrap envelope:

```json
{
  "schema_version": 1,
  "generated_at": "2026-08-21T20:00:00Z",
  "share": {
    "share_id": 42,
    "access_level": "view_chat",
    "allow_clone": false,
    "owner_display_name": "Research owner",
    "shared_at": "2026-08-20T18:00:00Z"
  },
  "workspace": {
    "workspace_id": "workspace-alpha",
    "name": "Evidence review",
    "description": "Review set"
  },
  "allowed_actions": {
    "inspect_sources": {"allowed": true, "reason_code": null},
    "ask_grounded_questions": {"allowed": true, "reason_code": null},
    "add_sources": {"allowed": false, "reason_code": "shared_write_not_available"},
    "edit_workspace": {"allowed": false, "reason_code": "shared_write_not_available"},
    "clone_workspace": {"allowed": false, "reason_code": "clone_deferred"}
  },
  "generation_default": {
    "provider": "llama",
    "model": "configured-model",
    "ready": true,
    "reason_code": null
  },
  "source_summary": {
    "total": 2,
    "queryable": 2,
    "processing": 0,
    "failed": 0
  },
  "sources": {
    "items": [],
    "pagination": {"offset": 0, "limit": 50, "total": 2, "has_more": false}
  },
  "conversation": {
    "conversation_id": null,
    "messages": [],
    "next_before": null
  },
  "partial_errors": []
}
```

The first page contains at most 50 sources and the latest 30 messages. Share
authorization, owner workspace identity, and owner source membership are
critical bootstrap dependencies. Their failure rejects the whole envelope.
Optional provider readiness, Jobs progress enrichment, or recipient history
failure appears as a bounded `partial_errors` entry and disables only the
dependent action. `generation_default` is resolved by the server for the
recipient and exact current share scope. It contains only provider, model,
readiness, and a stable reason code; it never exposes credential source, key
presence, endpoint, or raw provider diagnostics. `provider` and `model` are
non-empty only when `ready=true`; an unavailable default uses null values and a
non-empty stable `reason_code`.

Each partial error has exactly `area`, `code`, `message`, and `retryable`
fields. The bootstrap contains at most eight partial errors. Messages are
recipient-safe copy selected from stable codes rather than raw exception text.

`allowed_actions` is the only frontend permission/action authority. Client
defaults are deny, not allow. Form validity, source selection, and the
server-projected generation readiness may further disable submission, but can
never grant an action. The ordinary model catalog is discovery metadata rather
than credential or authorization proof; every submitted provider/model is
resolved again by the backend. `view_chat_add` and `full_edit` remain visible as
the granted tier, but write actions remain disabled with
`shared_write_not_available` in this slice.

### Source Listing

```text
GET /api/v1/sharing/shared-with-me/{share_id}/sources
```

Supported query parameters are `offset`, `limit`, `q`, and `state`. The default
limit is 50 and the maximum is 200. Results use the repository's canonical
offset pagination envelope and a stable `(position, source_id)` ordering.

Each source contains:

- canonical `source_id`;
- title and source type;
- sanitized HTTP or HTTPS origin when safe;
- readiness state and bounded reason code;
- citation and retrieval capability flags; and
- position and added timestamp.

File paths are never returned. HTTP and HTTPS origins omit user information,
query strings, and fragments; when removing those parts would make the URL
misleading, the response contains only an `origin_host` label. Pending, failed,
and missing sources remain visible but are not selectable for chat.

### Source Preview

```text
GET /api/v1/sharing/shared-with-me/{share_id}/sources/{source_id}/preview
```

The preview is addressed by canonical workspace `source_id`, never owner
`media_id`. It revalidates share access and source membership, then returns a
bounded text preview using the existing local workspace preview machinery. The
default is 3,000 characters, the hard maximum is 12,000 characters, and no
response includes more than 10 chunks. An optional `chunk_index` query value
centers a citation preview on that validated chunk and its nearest available
neighbors without increasing those bounds.

The old recipient-facing full-content
`/shared-with-me/{share_id}/media/{media_id}` endpoint is removed rather than
retained as an alias or redirect.

### Chat History

```text
GET /api/v1/sharing/shared-with-me/{share_id}/chat/messages
```

The endpoint accepts an opaque `before` cursor and `limit`. The default limit is
30 and the maximum is 100. It authorizes the current share before resolving the
recipient thread and returns messages in stable chronological order with
bounded citation metadata attached to assistant messages. The response includes
`next_before` when older messages remain. An absent thread returns an empty page
rather than creating a conversation.

### Shared Chat Request

```text
POST /api/v1/sharing/shared-with-me/{share_id}/chat
```

```json
{
  "request_id": "de305d54-75b4-431b-adb2-eb6b9e546014",
  "query": "What evidence supports this conclusion?",
  "source_scope": {
    "mode": "include",
    "source_ids": ["source-a", "source-b"]
  },
  "provider": "llama",
  "model": "configured-model"
}
```

Rules:

- `request_id` is a UUID generated once per composer submission.
- `query` is non-empty and at most 10,000 characters.
- `source_scope.mode` is `all` or `include`.
- `all` resolves all currently queryable shared sources.
- `include` requires one or more unique canonical source IDs.
- Every included source must currently belong to the shared workspace and be
  retrieval-capable.
- An explicit scope contains at most 500 sources.
- `all` on a workspace with more than 500 queryable sources returns
  `409 source_subset_required`; it never truncates silently.
- `provider` and `model` are optional overrides resolved under recipient
  credentials and server policy.
- The request has no `system_message`, owner database identifier, conversation
  ID, media ID, tool, or arbitrary RAG-configuration field.

The canonical request fingerprint is computed after validation from the exact
trimmed query, normalized source mode, sorted and deduplicated source IDs, and
normalized requested provider/model. The effective provider/model and resolved
source snapshot are stored separately. A retry of `mode=all` reuses its original
snapshot; it does not silently expand to newly added sources.

### Shared Chat Response

```json
{
  "schema_version": 1,
  "request_id": "de305d54-75b4-431b-adb2-eb6b9e546014",
  "conversation_id": "recipient-conversation-id",
  "turn": {
    "user_message": {
      "message_id": "user-message-id",
      "role": "user",
      "content": "What evidence supports this conclusion?",
      "created_at": "2026-08-21T20:00:00Z"
    },
    "assistant_message": {
      "message_id": "assistant-message-id",
      "role": "assistant",
      "content": "The evidence indicates...",
      "created_at": "2026-08-21T20:00:01Z"
    }
  },
  "citations": [
    {
      "citation_id": "citation-1",
      "source_id": "source-a",
      "source_title": "Primary report",
      "locator": {"chunk": 4},
      "quote": "Bounded supporting passage",
      "score": 0.87
    }
  ],
  "generation": {
    "provider": "llama",
    "model": "configured-model"
  },
  "source_scope": {
    "mode": "include",
    "effective_source_count": 2
  },
  "replay": {"replayed": false}
}
```

Canonical citations are constructed only from verified retrieval output. Model
citation labels must resolve to that verified set. Unknown labels are never
turned into citation records. A generated answer must have at least one verified
evidence citation. Empty retrieval returns `409 no_relevant_evidence` without a
generation call or persisted turn.

Prompt budgeting produces an immutable evidence subset containing exactly the
labels and text sent to the provider. Response labels and citation quotes are
resolved only against that subset; dropped evidence and text trimmed from the
last retained item cannot appear in a citation.

At most 20 citations are returned or persisted. Each quote is limited to 1,000
characters and total persisted quote text is limited to 16,000 characters per
assistant message.

### Typed Errors

Errors use FastAPI's `detail` field with one typed object:

```json
{
  "detail": {
    "code": "request_in_progress",
    "message": "This question is still processing.",
    "retryable": true,
    "recovery_action": "retry",
    "retry_after_ms": 1500
  }
}
```

`recovery_action` and `retry_after_ms` are omitted when they do not apply. The
neutral `404` body is identical for missing, revoked, deleted, and unauthorized
shares. Required mappings are:

| HTTP | Code | Meaning |
| --- | --- | --- |
| 401 | `authentication_required` | No valid principal |
| 403 | `sharing_permission_required` | Principal lacks global permission |
| 404 | `shared_workspace_not_found` | Missing, revoked, deleted, or out-of-scope share |
| 409 | `request_in_progress` | Matching receipt has an active lease |
| 409 | `request_id_conflict` | Request ID was reused with another fingerprint |
| 409 | `source_subset_required` | `all` exceeds the source cap |
| 409 | `shared_source_changed` | Frozen source membership or content changed |
| 409 | `no_relevant_evidence` | Retrieval found no verified evidence |
| 422 | `invalid_shared_workspace_request` | A recipient read path or query value is invalid |
| 422 | `invalid_shared_chat_request` | Request shape or values are invalid |
| 422 | `shared_chat_context_too_large` | The question and minimum evidence cannot fit the selected model context |
| 429 | `shared_workspace_rate_limited` | Recipient read rate limit exceeded |
| 429 | `shared_chat_rate_limited` | Recipient/share rate limit exceeded |
| 503 | `shared_workspace_unavailable` | Authorized owner data cannot be read |
| 503 | `retrieval_unavailable` | RAG retrieval failed |
| 503 | `no_provider_configured` | Recipient has no usable provider |
| 503 | `generation_failed` | Provider failed without fallback |

No error causes local workspace rendering. Operational errors expose no owner
paths, SQL, model credentials, query content, or retrieved excerpts.

## Recipient Persistence Model

The recipient ChaChaNotes schema gains two dedicated tables for SQLite and
PostgreSQL.

The active `CharactersRAGDB.client_id`, normalized to a non-empty string, is
the sole recipient tenant identity for this persistence layer. Recipient and
historical owner IDs are stored as `TEXT`, matching ChaChaNotes' canonical
tenant representation and avoiding a dependency on AuthNZ's current numeric ID
implementation. The store derives the recipient key from its bound database;
Sharing services and HTTP payloads cannot provide or override it.

### `shared_workspace_chat_threads`

The table contains:

- `recipient_user_id`;
- `share_id`;
- `conversation_id` with a local foreign key to `conversations`;
- historical `owner_user_id` and `workspace_id` references;
- creation and update timestamps; and
- a unique key on `(recipient_user_id, share_id)` plus a composite unique key
  that lets receipts reference the exact thread/conversation mapping.

On PostgreSQL, the table has forced row-level security. Its `USING` and `WITH
CHECK` predicates require the recipient key to equal `app.current_user_id` and
the mapped conversation to be live and owned by that same recipient.

Conversation creation and thread mapping occur in one transaction. An insert
race reloads the winning mapping. The conversation uses:

- `source = "shared_workspace"`;
- `external_ref = "share:{share_id}"`;
- `scope_type = "global"`; and
- `workspace_id = NULL`.

The conversation cannot use `scope_type = "workspace"` because the owner's
workspace row does not exist in the recipient database. The thread table, not
`external_ref`, guarantees uniqueness. A new share record receives a new share
ID and therefore a new conversation even when it targets the same workspace.

Deleting the recipient conversation cascades the thread and its request
receipts. The next authorized chat creates a fresh conversation.

### `shared_workspace_chat_requests`

The table contains:

- recipient user, share, and request IDs;
- canonical request fingerprint;
- conversation ID;
- status: `in_progress`, `retryable`, `completed`, or `conflicted`;
- monotonic lease epoch, random lease token, and lease expiry;
- frozen canonical source IDs as bounded JSON;
- frozen source-snapshot hash;
- effective provider and model;
- user and assistant message references after completion;
- bounded error code and timestamps; and
- a unique key on `(recipient_user_id, share_id, request_id)`.

A composite foreign key to
`shared_workspace_chat_threads(recipient_user_id, share_id, conversation_id)`
prevents a receipt from pairing one valid share mapping with another valid
conversation. The thread's conversation foreign key remains the cascade root.

On PostgreSQL, the table also has forced row-level security. Its read/write
predicate requires the same recipient key, a matching recipient-visible thread
for the share and conversation, and a recipient-owned conversation. Any
non-null message references must point to messages in that conversation owned
by the same recipient. These policies are installed by the canonical ChaCha RLS
builder during migration and normal initialization; store predicates remain
defense in depth.

The source-ID JSON contains at most 500 canonical IDs and no owner media IDs or
content. It allows `mode=all` retries to resolve the original source set rather
than a newer set. The table otherwise stores references and bounded codes, not
queries, answers, excerpts, prompts, credentials, or raw diagnostic payloads.
Completed receipts remain until conversation deletion because one receipt row
per turn is a reasonable cost for durable replay. Conflicted receipts expire
after 24 hours. Retryable receipts retain their fingerprint and may be reclaimed
by the same request.

The lease duration derives from the configured provider timeout plus a bounded
grace period, with a five-minute minimum and thirty-minute maximum. This avoids
lease expiry during a valid model call while still permitting recovery after a
dead process.

### Claim and Completion Semantics

1. A new request atomically inserts an `in_progress` receipt with epoch 1 and a
   random lease token.
2. The same request ID with another fingerprint returns
   `request_id_conflict`.
3. A matching completed receipt returns the stored assistant response without
   reserving generation capacity, retrieval, or generation, but only after
   current share authorization.
4. A matching unexpired in-progress receipt returns `request_in_progress` and
   bounded retry timing.
5. A matching retryable or expired receipt can be reclaimed by an atomic
   compare-and-swap that increments the lease epoch and replaces the token.
6. Only a new or reclaimed claimant reserves recipient/share generation rate
   capacity; a rejected reservation returns the receipt to `retryable`.
7. The initial claimant resolves and stores canonical source IDs plus the
   snapshot hash before retrieval. A claimant that crashed before this update
   leaves no frozen set, so its fenced successor may perform the initial
   resolution.
8. Every source snapshot update, failure transition, and completion includes
   the current epoch and token in its update predicate.
9. A stale request whose token no longer matches cannot persist messages or
   complete the receipt.

Transient retrieval, provider, backend, or disconnect failures move the receipt
to `retryable` when no successful turn was persisted. A frozen source mismatch
moves it to `conflicted`; the client refreshes source state and submits a new
request ID.

`workspace_operations` and Jobs are not used for synchronous chat. The former
has workspace-owned foreign-key and command semantics, while the latter is for
user-visible background work. Shared chat needs a small recipient-local
idempotency receipt instead.

### Strict Turn Persistence

Generation runs outside a database transaction. After final authorization and
source revalidation, one recipient transaction:

1. verifies the receipt lease epoch and token;
2. verifies the mapped conversation created with the initial claim;
3. inserts the user message;
4. inserts the assistant message;
5. writes bounded citation metadata under the established RAG message metadata
   shape; and
6. marks the receipt completed with both message references.

Any failure rolls back the whole turn and receipt completion. This path does not
call the existing best-effort `set_message_rag_context()` helper. The dedicated
store validates metadata and performs strict writes through the active
transaction for both database backends.

No user message is saved for a failed request. The frontend retains the draft
and selected sources until success.

## Source Authorization Snapshot

`workspace_sources.version` is not a content revision. It changes for selection,
ordering, review state, and metadata while owner media content can change
independently. It must not be the sole consistency token.

For each selected source, the service freezes:

- canonical `source_id`;
- internal owner `media_id` used only server-side;
- `Media.uuid`;
- `Media.content_hash`;
- deletion and trash state; and
- current retrieval/citation readiness class.

The sorted snapshot is hashed and recorded on the request receipt. Revalidation
requires the same source-to-media mapping, media UUID, content hash, live state,
and retrieval capability. Title, ordering, selected flag, and review metadata
changes do not abort a request because they do not change authorization or
evidence content.

A removed source, remapped source, changed content hash, deleted media row,
trashed media row, or loss of queryability returns `shared_source_changed`.
Unrelated workspace source changes do not abort the request.

## Retrieval and Generation Flow

1. Authorize the recipient without opening owner content databases.
2. Resolve or create the recipient chat thread and claim the request receipt.
3. Resolve canonical source IDs to the frozen source authorization snapshot.
4. Retrieve using only the owner Media database and owner embedding namespace.
5. Set the RAG source set to media only, pass explicit non-empty media IDs, and
   apply a locked retrieval-only policy that disables cache, profiles,
   provider-backed query transforms/reranking, adaptive execution, external
   retrieval, generation, and fallback.
6. Reject the complete retrieval result if the pipeline reports an error,
   returns a generated answer or external-source marker, or any passage lacks
   verifiable `media_db` provenance inside the frozen snapshot.
7. Revalidate share authorization and selected source snapshots.
8. Resolve a server-owned context budget and recipient credentials, limiting
   scoped credential lookup to the share's authoritative team/organization
   scope, then generate with a server-owned grounding prompt, no tools, and no
   fallback generator.
9. Validate returned citation labels against the verified evidence set.
10. Revalidate share authorization and selected source snapshots again.
11. Persist the turn atomically in the recipient database.
12. Return the stored response and emit a bounded audit event.

An empty media ID list is never passed to RAG because the current pipeline
treats an empty include list as unconstrained. No owner ChaChaNotes database is
supplied to retrieval, which prevents owner notes and conversations from
entering the context. Cache is explicitly disabled to avoid cross-share or
cross-recipient reuse.

The shared service passes no caller-owned resolved request, retrieval plan,
profile, metadata, or extra RAG kwargs. A signature-contract test requires every
security-sensitive pipeline parameter to be explicitly pinned or reviewed as
inert. Runtime postconditions still reject generation or source broadening, so
configuration or future default drift fails closed rather than silently
widening retrieval.

Source content is treated as untrusted input. Shared chat cannot request tools,
MCP, ACP, sandbox execution, or provider-side function calls. The server-owned
prompt separates instructions from a JSON-serialized evidence array and tells
the model not to follow instructions embedded in sources. Source text is never
interpolated into hand-written delimiters that it could terminate or forge.

The complete serialized prompt is budgeted before credential use or generation.
A positive context window comes from server-owned model/runtime metadata.
Known values from 2,048 through 1,000,000 tokens are accepted, larger values
are capped at 1,000,000, a known smaller model is rejected for this flow, and
unknown or invalid metadata uses a conservative 4,096-token window. The service
reserves 256-1,200 output tokens and a ten-percent-or-256-token safety margin,
caps evidence input at 12,000 tokens, and rejects a question that leaves no
room for one non-empty evidence item. It never silently truncates the user's
question. Counting uses a local `tiktoken` encoding when available and
otherwise a UTF-8 byte upper bound. It must not call provider-native or
commercial tokenizer endpoints with shared content.

Credential resolution receives `request=None` plus explicit team and
organization ID lists containing only the authoritative share scope. Trusted
base-URL eligibility is derived separately from the authenticated request.
This preserves recipient BYOK -> current share scope -> server fallback order
without allowing stale request-state active-scope claims to narrow or replace
the share scope. Credentials are never selected because the share owner owns
the source data. When the owner opens their own share URL, their user credential
is eligible only because the authenticated owner is also the recipient.

Provider failures are returned as typed errors. `FallbackGenerator` is disabled
for this path so a provider failure cannot be displayed as an answer assembled
from excerpts.

## Revocation and Historical Data

A share observed as revoked blocks bootstrap, source listing, preview,
retrieval, new chat, and completed-receipt replay through the shared route. The
shared route returns neutral `404` and does not return its recipient chat
history after revocation. The disclosure-boundary caveat in the authorization
section still applies to a request already between two checks.

Already persisted messages remain recipient-owned and may continue to appear in
the general Chats product. Their bounded citation quotes remain part of the
saved transcript, but citation preview cannot fetch owner content after
revocation. If an individual source is removed while the share remains active,
historical citations display `This source is no longer shared` when preview is
requested.

While the share remains active, replaying an already completed receipt returns
the recipient-owned saved turn without revalidating its historical source
snapshot. New requests use current membership, and every preview performs
current authorization. This matches the treatment of the same saved turn in
the general Chats product.

Owner-facing share copy must state:

> Revoking access prevents future workspace reads and questions. It does not
> erase content or answers recipients saved while they had access. Recipients
> may use their own configured model provider, which can receive selected shared
> passages when they ask a question.

This is presented in the existing share-management flow, not as another banner
inside Research Workspace.

## Frontend Route and State Design

Both Research Workspace route wrappers import the same
`ResearchWorkspaceRouteGate`. The gate uses reactive search parameters and
classifies the route into exactly one state:

- unresolved;
- local;
- shared-invalid; or
- shared-valid.

The shared query value must occur exactly once and be a positive base-10 safe
integer. While unresolved, the gate renders stable loading geometry and does not
mount either workspace. Local mode lazy-loads the existing local component.
Shared-invalid renders a scoped recovery state. Shared-valid lazy-loads
`SharedResearchWorkspace` keyed by share ID.

Changing or removing the share ID aborts active requests and clears the prior
share data before resolving the new route. An invalid, failed, or revoked share
never mounts the local workspace, never reads local Zustand workspace state, and
never performs local workspace API calls.

The obsolete shared context/banner path is removed from the local component.
Any compatibility context that must temporarily remain defaults every action to
denied.

## Shared Research Workspace Interaction Design

Desktop uses two workspace panes: Sources and Chat. Mobile uses Sources and Chat
tabs. Source preview opens in a workspace-contained drawer on desktop and a
full-screen sheet on mobile.

The compact header contains:

- Back to Shared with me;
- workspace name;
- `Shared by {owner}`;
- access-tier badge with an explanatory tooltip; and
- no additional trust banner, migration banner, workspace banner, onboarding
  prompt, or local-storage status bar.

For `view_chat_add` and `full_edit`, the access tooltip explains that the tier
is the policy ceiling and shared editing is not yet available. The interface
does not render disabled mutation toolbars, Studio, General Chat, notes, MCP,
ACP, sandbox, artifacts, or local workspace controls.

### Sources Pane

- All retrieval-capable sources start selected.
- Each row shows a checkbox, title, type, and concise readiness state.
- Pending, failed, missing, or non-retrieval-capable sources are disabled with
  a specific reason.
- `Select all queryable` and `Clear selection` support bulk scope changes.
- Search and state filtering execute against the paginated server endpoint.
- More than 500 queryable sources requires an explicit subset before chat.
- Source activation opens the bounded preview.
- Citation activation opens the same preview and focuses the cited excerpt when
  the locator is available.

### Chat Pane

- One recipient-owned conversation is used per share.
- The bootstrap contains the latest message page; older history loads upward
  through an opaque `before` cursor based on stable message ordering.
- The current source scope count appears directly above the composer.
- Provider and model controls display the effective generation destination.
- The server `generation_default` seeds selection ahead of local/global model
  preferences. The generic catalog may offer alternatives but does not prove
  credential availability; submission remains server-authoritative.
- The submit action generates one request UUID and reuses it for network retry.
- A source-state conflict refreshes the source set and creates a new request ID
  only after the user retries.
- Citations are keyboard-operable and have descriptive accessible names.
- Provider and retrieval failures preserve the draft and selected sources.
- The draft clears only after the successful response is persisted and shown.

### Loading, Empty, and Error States

The shared surface uses in-pane states rather than stacked global banners:

| State | UI behavior |
| --- | --- |
| Bootstrap loading | Stable two-pane or tab skeleton |
| No sources | Sources empty state; chat disabled |
| No queryable sources | Readiness explanation and source list; chat disabled |
| Partial status failure | Affected status shown unavailable; unrelated pane remains usable |
| Provider unavailable | Composer keeps draft; provider settings action shown |
| Rate limited | Submit disabled until bounded retry time |
| Request in progress | Current submission remains pending; retry uses the same ID |
| Source changed | Refresh source set and offer retry with a new request ID |
| Unauthorized or revoked | Neutral full-surface not-found recovery |
| Owner backend unavailable | Scoped unavailable state; no local fallback |

On route changes, focus moves to the shared workspace heading. Loading and
submission changes use appropriate live regions. Pane controls, source rows,
tabs, preview, citations, pagination, and composer are reachable and operable by
keyboard. Text and controls must not overflow at mobile or desktop widths.

## Browser Extension Boundary

TASK-12020.40 changes only the extension-hosted Research Workspace route wrapper
so it uses the same fail-closed `ResearchWorkspaceRouteGate` as the WebUI. It
does not change extension capture, destination, or handoff contracts and does
not add shared-workspace capture. The shared bootstrap exposes no writable
destination capability, and the shared WebUI exposes no add-source request path
or writable workspace handoff metadata. An extension must continue to require
an explicitly selected recipient-owned local workspace before capture. Full
extension destination rejection, access-tier handling, capture receipts, and
save-and-open behavior remain separate work.

## Privacy, Security, and Audit

- Source previews and citations are bounded.
- Unsafe source URLs, local filesystem paths, secrets, and raw diagnostics are
  removed from recipient responses.
- Client `system_message`, tool, and arbitrary RAG overrides are forbidden.
- Recipient credentials are used for generation; owner provider credentials are
  never used or returned.
- Retrieval cannot access owner notes, conversations, or media outside the
  selected canonical source set.
- Shared chat is non-streaming so final authorization and source checks occur
  before disclosure.
- Rate limits are keyed to the recipient and share in addition to global
  sharing policy.
- Audit events record share ID, actor, effective source count, provider/model,
  outcome code, replay status, and timings. They do not record the query,
  answer, excerpts, credentials, or system prompt.
- Audit persistence remains best-effort and cannot turn a successful grounded
  response into a failure.
- Diagnostics use existing operation redaction bounds: at most 16 keys and 320
  characters per string after secret and host-path removal.

## Verification Strategy

### Database Tests

Run equivalent focused tests against SQLite and PostgreSQL:

- concurrent first chat creates one thread and conversation;
- matching request IDs return the same receipt;
- mismatched fingerprints conflict;
- active leases return bounded retry timing;
- expired and retryable receipts are reclaimed with a new epoch and token;
- stale lease holders cannot update snapshots, messages, or completion;
- completed receipt replay returns the stored message;
- strict metadata failure rolls back both messages and receipt completion;
- conversation deletion cascades thread and receipt rows; and
- SQLite and backend abstraction errors map to the same domain responses;
- PostgreSQL policy catalogs show RLS enabled and forced for both tables; and
- a second recipient on the same PostgreSQL backend cannot read or mutate the
  first recipient's thread or receipts even with known identifiers.

### Backend and Security Tests

- Explicit permission dependencies reject principals without `sharing.read`.
- Missing, revoked, deleted-target, and nonmember shares return the same neutral
  `404` body.
- Authorized operational failures return typed `503` without owner details.
- Bootstrap and source responses enforce all bounds and sanitization rules.
- Preview accepts source IDs and rejects media IDs or nonmember sources.
- `all`, `include`, duplicate IDs, empty IDs, unknown IDs, nonqueryable IDs, and
  the 500-source cap follow the contract.
- A sentinel owner media item outside the workspace is never retrieved, sent to
  generation, returned, or cited.
- Owner notes and conversations cannot enter retrieval.
- Any out-of-scope retriever result aborts before generation.
- Revocation, membership loss, source removal, media replacement, content-hash
  change, deletion, trashing, and readiness loss at each revalidation boundary
  abort without persisted messages.
- Unrelated source changes do not abort the request.
- Empty retrieval does not call generation.
- Provider failure never invokes fallback generation.
- Known and unknown model context windows enforce deterministic local-only
  prompt budgets; oversized questions fail before provider or tokenizer HTTP
  calls.
- Completed replay authorizes before reading recipient receipt data.
- Removed-source and revoked-share citation preview fails closed.

### Frontend Tests

- The shared query accepts exactly one positive safe integer.
- Parameter changes clear prior shared data synchronously.
- Shared loading, failure, and valid states never mount local Research
  Workspace or hydrate its Zustand store.
- Shared mode performs no local workspace, Studio, notes, MCP, ACP, sandbox, or
  mutation requests.
- Allowed actions drive every visible control with deny-by-default behavior.
- Source selection, select-all, clear, search, filters, pagination, and the
  500-source state send canonical source IDs.
- History survives reload and older messages paginate upward without duplicate
  rendering.
- Provider errors preserve draft and source selection.
- Citation activation opens the source preview and handles removed sources.
- Desktop and mobile layouts pass keyboard, focus, live-region, overflow, and
  responsive tests.

### Live Backend and CDP UAT

Live acceptance uses an actual multi-user backend and actual WebUI. Browser
interaction uses Chrome DevTools Protocol, not computer control or mocked
network responses.

The fixture contains:

- an owner with a shared canonical workspace containing two ingested sources;
- an unrelated owner media item containing a unique sentinel phrase;
- an authorized team member with unrelated local workspace content; and
- a nonmember account.

The run uses the configured local OpenAI-compatible provider at
`127.0.0.1:9099` when available. Another configured provider is acceptable when
the effective provider and model are recorded. File ingestion must be allowed
to finish until the canonical source projection reports queryable before chat
acceptance begins.

The CDP walkthrough verifies:

1. The member sees only the owner's shared identity and source set.
2. Asking across all sources returns verified citations.
3. Asking against a subset cites only that subset.
4. The unrelated owner sentinel never appears in context, answer, or citations.
5. Citation preview shows bounded evidence.
6. Reload restores the recipient-owned conversation.
7. The owner opening the share URL receives the same recipient-style surface.
8. Malformed IDs and direct nonmember access reveal no workspace details.
9. Revocation clears the active surface and blocks subsequent reads and chat.
10. Previously saved chat remains in Chats while revoked previews cannot read
    owner content.
11. Clean desktop and mobile screenshots capture the valid source/chat surface,
    grounded answer with evidence, and revoked state without unrelated debug or
    banner bars.

An API-level live probe additionally races matching request IDs, verifies
completed replay, and confirms mismatched fingerprints conflict. The 501-source
limit case is covered by deterministic integration tests rather than expensive
live ingestion.

## Rollout and Compatibility

This is a direct repair of a pre-1.0 shared data plane. The existing WebUI and
tests move to the typed envelopes in the same change. Current active API docs
and schema examples are updated.

The insecure full-media recipient endpoint is removed with no alias or redirect.
The Research Workspace URL remains `/research-workspace?shared={share_id}`;
there is no new route and no legacy Research Workspace route restoration.

Schema migrations create recipient thread and request tables for SQLite and
PostgreSQL. The feature does not backfill conversations because the current
recipient route does not persist canonical shared chat. Existing unrelated
conversations remain untouched.

## Alternatives Rejected

### Reuse the Local Research Workspace Store

Hydrating owner data into recipient local Zustand state would be a smaller UI
change, but it preserves the current risk of local/shared identity mixing,
optimistic mutations, and accidental local API calls. It is rejected.

### Build a Generic Workspace Data Adapter First

A generic adapter could eventually reduce duplication between local, shared,
cloned, and project workspaces. Building it before the security boundary is
proven would require refactoring the existing large local component and widen
the change substantially. The dedicated shared surface establishes a narrow
contract that a later adapter can implement.

### Use Synthetic Recipient Workspace Rows

Copying or projecting an owner's workspace into the recipient database would
create a second workspace identity, cross-user lifecycle ambiguity, and
revocation cleanup problems. It conflicts with the canonical Workspace model.

### Reuse Jobs or `workspace_operations` for Chat Receipts

Shared chat is synchronous, has no user-facing background lifecycle, and writes
to a recipient database that does not contain the owner's workspace row. A
small dedicated receipt is simpler and correctly scoped.

### Stream Shared Chat

Streaming would expose content before final source and revocation checks and
would complicate atomic persistence and replay. It is deferred until a later
design introduces a secure streaming disclosure protocol.

### Retain Full Retrieved RAG Context

Persisting complete retrieved documents is convenient for debugging but copies
more owner content than the recipient needs, increases database size, and makes
revocation expectations harder to explain. Bounded citations are sufficient for
evidence inspection and transcript continuity.

## Implementation Boundaries

Implementation planning should preserve five reviewable stages:

1. Add a fail-closed frontend route gate and disable the legacy unconstrained
   shared chat/full-media paths before adding the replacement data plane.
2. Add the recipient persistence schema and concurrency-safe store.
3. Add access, bootstrap, history, preview, retrieval, generation, and typed API
   contracts.
4. Add the dedicated recipient Research Workspace UI.
5. Complete focused tests, security checks, and live backend/CDP acceptance.

The first stage may temporarily show a scoped unavailable state for valid
shares. Every subsequent commit must keep the route and shared chat fail closed;
no intermediate commit may render recipient-local data or run unconstrained
owner retrieval when a shared parameter is present.

## Acceptance Mapping

| Task criterion | Design coverage |
| --- | --- |
| Valid share renders shared identity and sources | Bootstrap envelope, route gate, dedicated shared surface |
| Permissions drive controls and requests | Explicit permission dependency, membership resolution, `allowed_actions` |
| Canonical shared chat cites only shared media | Frozen source snapshot, explicit media allowlist, provenance checks, bounded citations |
| Invalid and revoked shares fail closed | Neutral 404, no local mount, authorization-before-replay, revocation checks |
| Focused tests and live UAT | SQLite/PostgreSQL, API, frontend, security, and CDP matrices |
