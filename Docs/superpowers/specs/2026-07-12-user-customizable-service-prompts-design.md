# User-Customizable Service Prompts Design

**Status:** Approved for implementation planning on 2026-07-15

**Date:** 2026-07-13

**Backlog:** TASK-13142 (legacy ID: TASK-13013; reconciles historical TASK-12955, TASK-12956, and TASK-12958)

**Authoritative source:** approved commit `1a038599753e780f32f62243871026ca9b6d2c06`

## Summary

Service Prompts gives an authenticated user one Settings page where they can inspect and customize a curated set of content-generation prompts. A valid save becomes active immediately for that user and is used consistently by the supported WebUI, browser extension, and registered server consumers.

The feature is intentionally small. It uses a static allowlist, one table in the existing per-user prompts database, one two-source resolver, four API operations, and a deterministic client-side preview. It does not introduce revision approval, signed manifests, OS-keyring requirements, deployment policy states, a second prompt database, or prompt-specific asynchronous machinery.

## Product outcome

A user can:

1. Open **Settings → Workflow prompts** in either the WebUI or extension options page.
2. See which workflows a prompt affects and which variables it must contain.
3. Edit and preview the prompt as plain text.
4. Save it and have the change apply immediately to future requests prepared by supported consumers.
5. Reset it to the packaged server default.

On a supported server, the same authenticated account sees the same effective prompt in the supported WebUI and extension. Browser-local prompt values are explicit migration input, not a second source of truth.

## Goals

- Make selected content-generation prompts discoverable and editable without exposing security- or control-plane instructions.
- Keep overrides owner-scoped in both single-user and multi-user deployments.
- Preserve current default text and assembly semantics when no override exists. Intentional corrections are keeping placeholder-looking text and JavaScript replacement metasequences inside runtime data literal, plus isolating the Document Chat rewrite call from tools and persistence.
- Make save, client-side preview, reset, conflict, corruption, disconnected, and legacy-migration behavior understandable.
- Reuse the existing per-user prompt storage, authentication, database transactions, client scope key, and Settings shell.
- Deliver one useful vertical slice before migrating more prompt domains.

## Non-goals

- Editing authentication, authorization, moderation, safety, routing, tool-selection, tool-permission, judge, grading, or machine-protocol prompts.
- Creating arbitrary prompt IDs at runtime.
- Per-save administrator approval or a review inbox.
- Signed mutable manifests, anti-rollback anchors, key rotation, or OS-keyring integration.
- A prompt-specific deployment enable/disable state matrix in v1.
- Full revision history, restore, diffing, bulk editing, or portable import/export.
- A live LLM test from Settings. Preview is deterministic and makes no server or provider call.
- Migrating every candidate in the existing inventory before release.
- Registering an asynchronous consumer before that domain receives its own snapshot-persistence design.

## Eligibility boundary

A definition may be registered only when all of the following are true:

- It changes user-visible generated or analyzed content.
- Its entire editable template can be shown to the user.
- Independent code—not prompt wording—enforces authorization, routing, tool availability and permissions, persistence, and output validation.
- A concrete runtime consumer and owning workflow are identified.
- Its variables and default template are finite and testable.

A definition is excluded when it can grant or alter authorization, moderation, safety policy, provider or model selection, available tools, tool permissions, toolChoice, agent control, evaluation scores, retrieval grading, or a machine-readable protocol that is not independently enforced. Like ordinary user text, an eligible content prompt may influence a model's response within capabilities already enabled by code; it cannot change those capabilities. User-owned wording that rewrites the user's own search query is content transformation, not service routing; source access, provider selection, and retrieval policy remain locked in code.

Prompts containing hidden or locked instruction fragments are not eligible for v1. They may be reconsidered only after the consumer has a clear boundary between editable content guidance and independently enforced control behavior.

The static code registry is the allowlist. Database rows never create definitions.

## V1 vertical slice

V1 migrates the three existing browser-local prompt settings that have live runtime consumers and exposes one previously internal synchronous backend prompt:

| Stable ID | Settings label | Editable parts and required variables | Current local key | Reachable runtime consumers |
| --- | --- | --- | --- | --- |
| chat.rag.answer | RAG answer | template: context, question | systemPromptForRag | Main Chat ragMode, tabChatMode, documentChatMode, and legacy Sidepanel RAG in useMessage |
| chat.rag.question_rewrite | RAG follow-up rewrite | template: chat_history, question | questionPromptForRag | Main Chat ragMode, documentChatMode, and legacy Sidepanel RAG |
| chat.web_search.answer | Web-search answer | template: current_date_time, search_results | webSearchPrompt | normalChatMode, including each branch of Compare when web search is enabled |
| media.text.translation | Text translation | system: literal; user_template: target_language, text | — | Synchronous POST /translate |

No-override compatibility preserves the current value supplied to each variable:

| Consumer path | Compatibility rule |
| --- | --- |
| Main Chat ragMode final answer | question is the original current user message |
| Tab Chat final answer | question is the original current user message; Tab Chat does not rewrite the question |
| Document Chat final answer | question is the original current user message |
| Legacy Sidepanel RAG final answer | question is the rewritten standalone retrieval query |
| Main Chat, Document Chat, and legacy Sidepanel rewrite calls | question is the current follow-up and chat_history is that path's current serialized history |
| Main and Compare web-search answers | current_date_time and normalized search_results keep their current formatting |
| Translation | target_language and text keep their current values; provider/model fields remain code-controlled |

These RAG-path differences are deliberate compatibility, not new semantic preferences. The implementation plan must either preserve each path with separate golden tests or prove a path unreachable and remove it before consolidation. Ordinary no-override messages remain byte-equivalent; fixtures containing placeholder-looking runtime data or JavaScript replacement metasequences assert the intentional single-pass correction.

Before chat.rag.question_rewrite is eligible to activate, documentChatMode must match the existing isolated rewrite calls: toolChoice is forced to none, saveToDb is false, and the rewrite call cannot attach or invoke tools. Its output remains a string used only as the retrieval query. This is a required v1 hardening change, not optional cleanup.

webSearchFollowUpPrompt is not included because repository-wide caller tracing found no runtime consumer; it is currently a no-op Settings value. It remains untouched in legacy storage and is not advertised or imported. A future definition requires a real consumer and the normal eligibility tests.

The first slice does not migrate the broader server prompt inventory. That inventory remains research material and a prioritized backlog for later vertical slices.

## Architecture

### 1. Static registry

A plain immutable mapping of ServicePromptDefinition values describes each allowed prompt:

- stable dotted ID;
- English fallback label and description;
- stable affected-workflow IDs plus English fallback text;
- packaged default parts;
- an ordered set of editable parts, each with a stable key, fallback label, literal-or-template mode, and exact required variable names.

The client localizes known stable definition, part, and workflow IDs. It falls back to server-provided English text so a newer server definition remains understandable to an older supported client.

The three Chat definitions contain one template part. Translation contains an atomic literal system part and templated user_template part. Consumer code owns provider-message roles and assembly; the registry does not invent a generic assembly language. It also does not include lifecycle modes, approval metadata, deployment sources, cryptographic digests, or arbitrary assembly policies.

Defaults are canonical on a supported server. Because an updated extension may connect to an older server, the client retains clearly named legacy compatibility defaults solely for the explicit unsupported-server fallback. Supported servers never use those copies. Initial golden tests require server defaults and compatibility copies to match; later server-default changes do not alter old-server behavior.

### 2. Per-user override store

Overrides live in the existing per-user prompts database returned by the standard authenticated prompts-database dependency. The API never accepts a target user ID or database path.

PromptsDatabase moves from schema v5 to v6. Fresh initialization and a new transactional _apply_schema_v6 migration both create exactly one table and advance schema_version from 5 to 6:

    CREATE TABLE IF NOT EXISTS ServicePromptOverrides (
        definition_id TEXT PRIMARY KEY,
        parts_json TEXT NOT NULL,
        revision TEXT NOT NULL
    );

For each definition, parts_json contains exactly its registered keys; for example, a Chat row is {"template": "..."} and Translation contains system and user_template. Reads reject missing, extra, non-string, malformed, or otherwise invalid parts.

Ownership is provided by the per-user database itself. Store methods use the existing database abstraction and transaction helper. Each content-changing save replaces revision with a new opaque UUID. An exact no-op or identical retry returns the existing row unchanged. Reset deletes the row. Compare-and-swap on revision prevents stale writes even when an override is reset and recreated. There are no revision-history, event, receipt, approval, or catalog-generation tables.

A corrupt parts_json value does not prevent reading the separate definition_id and revision columns needed for a safe reset.

Physical backup of the per-user prompts database covers this table. Chatbooks and the current portable backup flow do not export Service Prompt overrides in v1. User-facing copy must say **Backup supported account data** and list Service Prompt overrides among the exclusions. The migration panel repeats that these overrides are server/account scoped and are not in the portable backup. Portable export/import remains deferred.

### 3. Resolver, request snapshot, and rendering

For a registered definition, the resolver has exactly two sources:

1. the authenticated user's valid saved override;
2. the packaged registry default.

It returns the selected parts and non-sensitive provenance: definition ID, source (user or packaged), and revision when a saved override exists.

Existing request-local controls keep their current semantics outside this resolver. For example, systemPromptAppendix remains a literal suffix; it is not another template source. The RAG template is rendered first and the literal appendix is appended afterward, so braces or placeholder-looking text in the appendix remain literal. Provider, model, source, retrieval, and tool controls stay in consumer code.

Server-side Translation calls the resolver directly. The shared TypeScript client obtains the three Chat definitions through authenticated detail reads. Before preflight, each top-level Chat invocation:

1. determines the definitions needed by that mode;
2. fetches those details concurrently once;
3. freezes a small request-scoped snapshot in the invocation's existing context; and
4. carries it through preflight, retrieval, and provider-message preparation.

Pipeline modes carry the snapshot in ChatModeContext. Legacy useMessage keeps one local immutable snapshot because it does not use chatModePipeline. Main Chat ragMode, normalChatMode, tabChatMode, documentChatMode, and legacy Sidepanel RAG must consume their applicable definitions from those snapshots or be proven unreachable and removed. Repeated helpers within an invocation never reload a definition.

A Compare submission is one top-level invocation. It resolves the applicable normalChatMode definitions once before the Promise.all model fan-out and gives every branch the same snapshot; individual model branches do not resolve independently.

Prompt resolution is outside best-effort provider/search catches. In particular, failure to resolve chat.web_search.answer from a supported server aborts request preflight; an ordinary web-search provider failure may retain its existing fallback behavior. A supported-server prompt failure is never silently converted into a Chat request without search context.

Detail responses use Cache-Control: no-store. Runtime prompt resolution performs fresh authenticated detail reads for each top-level invocation rather than reusing the Settings query cache. Client query keys, migration state, and editor drafts are additionally scoped by normalized server URL plus a resolved authenticated principal ID. Hosted WebUI sessions resolve /api/auth/session and pass its user ID into the existing scope-key builder; authenticated state must never fall back to user:anonymous merely because no access token is stored. Queries and mutations remain disabled until that identity resolves. A save invalidates the matching scoped Settings data. A server/account change aborts in-flight reads, invalidates the old queries, clears migration state, and makes any old-scope draft ineligible to save. HTTP cache headers alone are not treated as account isolation.

The renderer is deterministic and single-pass:

- Parse only the authored template into literal and placeholder tokens.
- Concatenate runtime values into those tokens exactly once.
- Never parse or replace text inside an inserted value.
- Preserve runtime values byte-for-byte, including braces, newlines, dollar-sign replacement metasequences, and placeholder-looking text.
- Treat escaped braces as literal braces.

Sequential replacement loops are forbidden. The TypeScript implementation must not use replacement-string semantics that reinterpret $&, $', or similar sequences. Token concatenation or callback substitution is required.

Every required variable appears exactly once in an accepted v1 template. This prevents user-authored repetition from amplifying a bounded runtime input into a much larger provider prompt while preserving every current default. Existing consumer input/context limits still apply. If a future use case genuinely requires repeated variables, it must introduce and test an explicit rendered-size policy first.

No asynchronous definition is part of v1. Snapshot persistence for a future asynchronous consumer is designed with that concrete domain, using existing Jobs infrastructure only if it fits.

### 4. Template validation and client-side preview

The server applies one validator to packaged defaults and saved overrides:

- parts must contain exactly the keys declared by the definition;
- each value must be a string, must contain at least one non-whitespace character, and must be at most 20,000 Unicode code points; the TypeScript client counts code points rather than UTF-16 code units;
- literal parts do not parse braces;
- Python string.Formatter.parse identifies replacement fields in template parts;
- fields must be simple ASCII identifiers matching [A-Za-z_][A-Za-z0-9_]*;
- attribute access, index access, conversions, and format specifications are rejected;
- every registered variable must occur exactly once;
- unknown, missing, and repeated variables are rejected;
- escaped literal braces remain supported; and
- invalid input is rejected, never truncated or silently repaired.

The client mirrors those rules for immediate editor feedback, but PUT remains authoritative validation.

Preview is local to the Settings client. It calls the production TypeScript renderer with visible [variable_name] marker values and returns an ordered display for each part. Literal-part content remains unchanged. Preview performs no API request and no LLM call.

One language-neutral fixture set is consumed by Python and TypeScript tests. It covers escaped braces, Unicode, newlines, $&, $', backslashes, values containing {question} or other registered names, and exact rendered bytes. Packaged defaults must pass the same fixtures and validator during tests and registry initialization.

### 5. API

All operations use normal authentication and implicit current-user ownership. API-key requests require read scope for GET and write scope for PUT/DELETE; JWT behavior follows the existing authenticated-user rules.

| Method and path | Success response | Purpose |
| --- | --- | --- |
| GET /api/v1/service-prompts | 200 catalog | Return metadata summaries without prompt bodies |
| GET /api/v1/service-prompts/{definition_id} | 200 detail | Return default, saved, and effective parts |
| PUT /api/v1/service-prompts/{definition_id} | 200 detail | Validate, atomically save, and immediately activate |
| DELETE /api/v1/service-prompts/{definition_id} | 200 detail | Atomically reset and return the packaged-default state |

The router is always registered in supported builds and is not hidden by a generic route_key switch. A catalog 404 is therefore reserved for an older server that does not implement Service Prompts. Catalog 401, 403, 5xx, invalid responses, and network failures are explicit errors and never enable browser-local compatibility mode.

Catalog summaries contain IDs, localized-fallback metadata, part schemas, and affected-workflow metadata, but no prompt bodies.

The canonical detail representation contains:

- id and definition metadata;
- default_parts;
- saved_parts, or null;
- effective_parts;
- source, either user or packaged; and
- revision, or null when no saved row exists.

PUT accepts {"parts": {...}, "expected_revision": <UUID-or-null>}. Null means the caller observed no saved row. Existing-row updates use one compare-and-swap statement whose WHERE clause includes revision. New-row inserts rely on the primary key and map a race to identical-retry success or 409 Conflict. If validated parts equal the parsed stored parts, PUT returns the current detail without generating a new revision.

DELETE accepts optional expected_revision as a query parameter; omission means the caller observed no saved row. An absent row with omitted expected_revision is an idempotent success. Every other mismatch, including a UUID supplied after another client reset the row, returns 409. A matching revision can reset a corrupt row because revision is stored outside parts_json.

Recognized Service Prompt endpoint/domain failures use the FastAPI envelope below:

    {
      "detail": {
        "code": "stable_machine_code",
        "message": "user-safe message",
        "...": "status-specific safe fields"
      }
    }

This custom envelope applies to unknown definitions, semantic parts/template validation, revision conflicts, corrupt rows, and transactional store failures handled after dependency resolution. Authentication and scope denials, malformed request bodies or query values, and prompts-database dependency/initialization failures keep the repository's existing FastAPI contracts because they occur before domain handling. The client normalizes known domain failures from direct HTTP and extension-proxy transports while retaining a safe generic path for those existing framework/dependency errors. Prompt bodies never appear in error details.

Status-specific fields and behavior:

- 404 service_prompt_unknown_definition for an unregistered definition;
- 409 service_prompt_revision_conflict with current_revision, which may be null;
- 422 service_prompt_validation_failed with field_errors keyed by part;
- 500 service_prompt_corrupt_override with revision and can_reset: true when the stored row cannot be safely resolved; and
- 500 service_prompt_store_failed for transactional store failures handled by the endpoint, with no partial write.

The field is named revision rather than revision_token so the extension proxy's generic secret redaction does not destroy the non-secret concurrency value. Extension transport tests must prove that detail, conflict, corrupt-row reset, and revision values survive the round trip.

Invalid packaged defaults are code defects: registry initialization fails rather than marking a definition conditionally available or sending invalid text to a model.

## Shared Settings experience

The existing /settings/prompt route remains under **Preferences & Workflow** and becomes **Workflow prompts** in both the hosted WebUI and extension options app. Existing Omni Search, Settings index, Prompt Search, and tests that currently describe /settings/prompt as the reusable Prompt Library are retargeted to /prompts. The new page retains a secondary **Open reusable Prompts workspace** link. The unreachable legacy editor is not resurrected.

V1 uses the existing Settings content width. It does not add a navigation group or redesign the shell. The route provides:

- a plain list of the four v1 definitions;
- a detail view addressable with ?prompt=<definition_id>;
- **Server default** or **Customized** status;
- affected-workflow text and explicit server/account scope;
- required-variable reference chips;
- one labeled plain-text editor per registered part;
- local **Preview**, **Save changes**, and **Reset to default** actions; and
- an unsaved-change guard when navigating away.

Known stable IDs are localized in the client, with server English text as fallback. On narrow layouts, selection navigates from list to detail. A permanent two-column editor and side-by-side diff are not required.

Literal parts show no variable controls; template parts show only their registered variables. Dirty state, validation, save, reset, and conflict apply atomically to the complete parts object, including Translation's system and user_template fields. Reset requires confirmation naming the definition and stating that its saved customization will be permanently removed because v1 has no history or undo. The same confirmation is required when resetting a corrupt row.

The first-class states are loading, server default, customized, dirty, saving, validation error, edit conflict, corrupt/unavailable, unsupported older server, and disconnected. A conflict preserves the complete draft and offers to reload the current server value; it never overwrites automatically. A corrupt override offers reset using the safe revision returned by the API. An older-server notice explains that existing local runtime behavior remains active and that server-synced editing requires a server update; v1 does not resurrect a second local-only editor.

The editor's query, detail cache, draft, selected definition, migration panel, and mutations all include the current server/account scope. If that scope changes after a draft loads, saving is disabled until the page loads the new scope; the old draft is not silently rebound.

Unsaved-change protection is host-aware. It covers Settings navigation, prompt selection through the query string, browser back/forward, and beforeunload in both hosts. The implementation must not assume the current Next router shim's no-op useBlocker provides this protection.

Prompt text is rendered only in text controls or escaped code blocks. The client does not interpret it as HTML or Markdown.

## Legacy browser-local migration

The server catalog is the capability boundary. Migration detection reads raw storage areas directly rather than calling default-producing helpers or the helper that automatically moves legacy sync data:

- On a supported catalog response, the three mapped local keys stop participating in runtime resolution. They are migration input only.
- Only a catalog 404 activates existing local behavior for an older server.
- Authentication, authorization, server, protocol, and network failures show an error or disconnected state and perform no migration or mutation.
- For the RAG keys, current local storage wins when both local and legacy sync contain a value; sync is used only when local is absent. Web search reads its current local area.
- Migration never synthesizes a compatibility default as if it were a saved value.

When supported raw values exist, /settings/prompt shows a one-time migration panel naming the connected server/account and offering **Import to this server** or **Discard local values**. It states that imported Service Prompt overrides are server/account scoped and are not included in portable Backup supported account data.

Reaching any Chat path that would consume one of the three mapped local keys before resolving that panel stops request preflight and shows an actionable **Review workflow prompts** notice linked to /settings/prompt. This includes Main RAG, Tab Chat, Document Chat, legacy Sidepanel RAG, main web search, and Compare web search as applicable. It does not silently use the server default or legacy value. Translation is unaffected because it has no browser-local predecessor.

Import uses ordinary GET detail and PUT operations. If a target already has a different saved customization, confirmation names the prompts that will be replaced; no sampled or hidden diff is used. Legacy text that fails the new rules remains visible and editable per key so the user can repair, copy, or discard it. It is never destroyed by failed validation.

Each successfully saved value is removed from both current local and legacy sync storage areas only after the server confirms it. A partial failure preserves every unconfirmed raw value and reports exactly which imports remain. Discard requires confirmation and removes only the three mapped keys from both areas.

Migration state is keyed by server/account and is cleared when that scope changes. There is no automatic import because browser storage may have been created for a different server or account. There is no permanent local-precedence mode because it would make supported clients and the server disagree.

## Security and privacy

- Registry eligibility—not approval ceremony—is the primary safety boundary.
- Standard authentication chooses the per-user database; callers cannot select another owner.
- API-key read/write scopes protect reads and mutations in addition to owner resolution.
- API schemas, strict part validation, bounded placeholder occurrence, and parameterized database operations protect the write boundary.
- Service Prompt bodies are not logged, included in catalog summaries, or placed in error details.
- The UI treats prompt bodies as plain text.
- Workflows continue enforcing authorization, routing, provider/model selection, tool availability and permissions, retrieval scope, and output contracts outside prompt text.
- Every question-rewrite model call, including Document Chat, forces toolChoice none, disables persistence with saveToDb false, and cannot attach or invoke tools. A hostile customized rewrite may change only the resulting query string. Authenticated corpus ownership, source IDs, retrieval options, provider/model, and tool configuration remain code-controlled and are verified by integration tests for every rewrite consumer.
- Existing Context Integrity protection for packaged files is unchanged. User-owned overrides are ordinary authenticated user data, comparable to explicit prompt input already accepted by eligible workflows.

No new cryptographic trust infrastructure is justified for this feature.

## Failure behavior

| Condition | Required behavior |
| --- | --- |
| Unknown definition | Return structured 404; do not create a row |
| Invalid, repeated-variable, or oversized template | Return field-specific 422; preserve the current override |
| Concurrent edit/reset | Return 409 with current_revision; preserve both the server value and local draft |
| Database write failure | Roll back; the previous override remains active |
| Corrupt active override | Return a safe error with revision and allow conditional reset; never substitute the default silently |
| Older server catalog 404 | Keep existing browser-local behavior |
| Supported server with unresolved legacy value | Block only the affected Chat workflow and link to the migration panel |
| Authentication, server, protocol, or network failure | Show the appropriate error/disconnected state; do not infer old-server capability or mutate storage |
| Supported-server prompt detail failure | Fail request preflight with a retryable user-safe error; do not use stale or local text |
| Web-search provider failure after prompt resolution | Preserve the current best-effort search fallback without swallowing prompt errors |
| Server/account scope changes with a draft | Abort old work and disable save; never send the draft to the new scope |

## Verification strategy

### Backend unit tests

- Registry IDs are unique and every packaged default passes its validator.
- Placeholder parsing accepts escaped braces and rejects traversal, indexing, conversions, format specs, unknown, missing, and repeated variables.
- Shared fixtures prove single-pass literal insertion for braces, newlines, Unicode, backslashes, $&, $', and placeholder-looking runtime values.
- Resolver precedence is user override then packaged default.
- PromptsDatabase fresh initialization creates v6 and an existing v5 database migrates to v6 without losing existing data.
- Store create, identical retry, compare-and-swap update, insert race, conflict, reset, and corrupt-row behavior are deterministic.

### API and integration tests

- Read-only API keys can read but cannot PUT/DELETE; write-scoped keys and JWT users follow the normal contract.
- User A cannot read, mutate, or reset User B's override through any request field.
- Catalog 404 is the only old-server signal; supported routers are present in minimal route configurations.
- Recognized GET/PUT/DELETE domain failures use the specified wire shapes; structural FastAPI, auth, and dependency failures retain and test their existing contracts.
- Save is immediately visible through detail and the consuming workflow.
- Translation resolves both editable parts atomically while provider/model request fields keep their current behavior.
- Every RAG rewrite call is non-persistent and tool-disabled. A hostile rewrite can change only query text; corpus ownership, source IDs, retrieval options, provider/model, and tool configuration remain unchanged.
- Single-user and multi-user authentication modes exercise the same owner-scoped contract.

### Frontend tests

- /settings/prompt is Workflow prompts in both hosts; former Prompt Library links target /prompts and the secondary workspace link remains.
- The editor handles server-default, customized, dirty, conflict, corrupt, unavailable, unsupported-older-server, disconnected, and scope-changed states accessibly.
- Local preview and reset do not invoke an LLM; preview also makes no Service Prompt API request.
- The TypeScript renderer passes the shared single-pass fixtures and never reinterprets inserted values.
- Main Chat modes, Tab Chat, Document Chat, and legacy Sidepanel RAG consume one immutable request snapshot per top-level invocation; shared modes use ChatModeContext and legacy useMessage uses a local snapshot.
- Compare resolves one snapshot before model fan-out, and every branch receives that same revision.
- Separate goldens preserve original-question semantics in Main RAG, Tab Chat, and Document Chat and rewritten-question semantics in legacy Sidepanel RAG.
- Web-search prompt resolution errors abort preflight while ordinary search-provider failures retain their current fallback.
- Query, draft, migration, and mutation state are isolated by normalized server and resolved session user ID; hosted authenticated sessions never share an anonymous key, and scope changes abort and invalidate old work.
- Unsaved-change guards cover Settings links, query selection, browser history, and beforeunload in both hosts.
- Normal and corrupt-row reset require the permanent-removal confirmation before DELETE.
- Direct and extension-proxy transports normalize the same detail envelope and preserve revision for conflict and corrupt reset.
- Migration probes raw local and legacy sync values, exposes invalid legacy text for repair, and clears only confirmed values.
- Unresolved mapped values gate Main RAG, Tab Chat, Document Chat, legacy Sidepanel RAG, main web search, and Compare web search before prompt preparation.
- Supported servers never allow legacy local values to shadow server state; non-404 capability failures never enable fallback.
- Localized known IDs use client strings and unknown IDs fall back to safe server English.
- Backup copy and migration disclosure explicitly exclude Service Prompt overrides.
- webSearchFollowUpPrompt is not presented as effective without a consumer.

### End-to-end checks

- Saving in the WebUI is visible from the extension options app for the same account/server.
- Changing server or account never exposes or saves the prior scope's prompt or draft.
- Each of the four v1 definitions affects its named consumers.
- With no override and ordinary runtime values, every reachable workflow's provider message remains golden-equivalent to pre-feature output; placeholder-looking runtime data and JavaScript replacement metasequences instead prove the deliberate single-pass correction.
- Reset from either client returns both clients and the workflow to the packaged default.
- A corrupt override can be reset through the extension without losing its revision in transport.

Implementation completion also requires formatter/linter checks, focused backend and frontend tests, git diff --check, and Bandit on touched Python paths.

## Incremental rollout

1. Ship the registry, v6 one-table store, two-source resolver, four API operations, shared Settings replacement, migration panel, three live Chat definitions, and synchronous Translation definition as one end-to-end slice.
2. Observe actual use and fix cross-surface or resolution problems before adding another domain.
3. Add later synchronous definitions one workflow at a time, with eligibility evidence, a named consumer, golden no-override coverage, and localized metadata.
4. Design snapshot persistence only when the first concrete asynchronous definition is proposed.

No release gate requires cataloging or migrating every internal prompt. The inventory is a discovery backlog, not runtime authority.

## Rejected approaches

- **Per-revision operator approval:** rejected because eligible prompt text grants no authority and equivalent explicit input already executes without approval.
- **Mutable signed manifest and OS-keyring anchor:** rejected because it blocks recommended container deployments and solves a different high-assurance integrity problem.
- **Explicit/deployment resolver tiers:** rejected from v1 because none of the four consumers has such a registered full-replacement source.
- **Prompt-specific deployment switch:** rejected from v1 because it creates dormant and migration states without a concrete operator requirement.
- **Server Preview endpoint:** rejected because the production client renderer can provide deterministic marker preview and PUT remains authoritative.
- **Prompt-specific protected Jobs ledger or async policy:** rejected because v1 has no asynchronous definition; the first concrete domain should design only what it needs.
- **Browser-local overrides as permanent precedence:** rejected because supported clients and server would disagree.
- **Broad migration before first release:** rejected because it delays user value and multiplies integration risk.
- **Revision history and portable backup product:** deferred until actual demand justifies more storage and UI.

## Acceptance criteria

- An authenticated user can inspect, preview locally, save, and reset each of the four v1 definitions from /settings/prompt in both supported hosts.
- A valid save activates immediately for that user without administrator approval, signing, keyring setup, or a deployment policy state.
- The same account/server resolves the same saved override from supported WebUI, extension, and registered server consumers.
- No request accepts a target user ID or database path; unknown definition IDs cannot persist rows, undeclared parts are rejected, and API-key scopes protect mutations.
- Rendering is single-pass and literal; each required variable occurs exactly once and inserted content is never reparsed.
- Fresh and existing per-user prompt databases reach schema v6 safely.
- Invalid templates, conflicts, corruption, disconnection, scope changes, and legacy migration are explicit and non-destructive.
- Reset and legacy discard are the only intentional destructive actions and require prompt-specific confirmation.
- Only catalog 404 enables old-server local compatibility; supported-server failures never fall back to local values.
- Legacy mapped values are imported or discarded explicitly and cannot shadow server state on supported servers.
- Every reachable Chat consumer uses one request snapshot, Compare shares one snapshot across its fan-out, and current no-override variable semantics are preserved.
- Every RAG rewrite call is non-persistent and tool-disabled; customization cannot change authenticated source scope, provider/model selection, retrieval policy, or tool configuration.
- Prompt Library links remain correct, Workflow prompts retains a reusable-workspace link, and portable-backup exclusions are honest.
- webSearchFollowUpPrompt is not exposed as functional until it has a runtime consumer.
- No-override provider messages remain golden-equivalent for ordinary runtime values, with the documented single-pass correction for placeholder-looking data and JavaScript replacement metasequences.
- No new approval service, cryptographic manifest, deployment-state matrix, server Preview endpoint, Jobs status, Jobs table, or prompt-specific reconciler is introduced.
- Additional synchronous domains can be added independently without making the complete inventory a release dependency.

## Superseded planning artifacts

The previous foundation, broad-domain, Context Integrity approval, and protected-job-pinning plans were based on the rejected architecture and must not be executed. They remain superseded. Human review approved this revision on 2026-07-15; the replacement implementation plan is tracked by TASK-13142.
