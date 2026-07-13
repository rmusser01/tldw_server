# User-Customizable Service Prompts Design

**Date:** 2026-07-12

**Status:** Approved in brainstorming; pending independent spec review

**Backlog:** TASK-12112

**Related:** TASK-2341 (broader shared prompt-registry follow-up)

## Summary

tldw_server should let each authenticated user customize a curated set of prompts that backend services apply automatically when creating or analyzing content. The capability will appear in a dedicated **Service Prompts** settings page shared by the WebUI and browser-extension options app.

The selected architecture is a typed `ServicePromptRegistry` backed by immutable server defaults and versioned per-user overrides. Services resolve prompts through one governed resolver rather than reading files or embedding strings directly. The design preserves existing explicit request overrides and deployment prompt files, validates prompt variables, supports deterministic previews, keeps revision history, pins queued work to an immutable prompt revision, and integrates with the existing context-integrity system.

This feature is deliberately separate from:

- The reusable Prompt Library and Prompt workspace
- Prompt Studio projects, experiments, and evaluations
- Conversation-specific system prompts and `preferences.chat.system_prompt`
- MCP prompt-catalog exposure
- Security, authorization, routing, judge, and machine-protocol prompts

The broad content-facing rollout is one product initiative but must be delivered as multiple Backlog tasks and reviewable pull requests.

## Context

The current codebase has several overlapping prompt mechanisms:

- `tldw_Server_API/Config_Files/Prompts` contains editable YAML and Markdown defaults loaded through `prompt_loader`.
- `TLDW_PROMPT_FILE_<MODULE>__<KEY>` environment variables can replace individual prompt assets at deployment time.
- User-authored reusable prompts live in per-user Prompts databases and are managed through `/api/v1/prompts` and the Prompt workspace.
- Prompt Studio has a separate project, prompt, test, evaluation, and revision model.
- Some service prompts remain embedded directly in Python or TypeScript.
- Many request schemas already accept explicit `system_prompt` or `custom_prompt` values.
- Context-integrity enforcement now protects prompt files and database prompt versions at use time.

The WebUI and extension already share route, settings, service, and component code under `apps/packages/ui/src`. The extension options application and Next.js WebUI can therefore use one implementation. The narrow extension sidepanel should deep-link to the full options editor rather than host the editor itself.

## Goals

1. Give each user one active override for every eligible service-prompt definition.
2. Apply that override to the user's WebUI, extension, API, synchronous, scheduled, and background activity on the same server.
3. Keep shipped and deployment-managed defaults immutable from the user settings surface.
4. Preserve explicit per-request prompts as the highest-precedence user choice.
5. Validate required variables and locked output contracts before activation.
6. Provide safe preview, comparison, reset, history, restore, and upstream-default awareness.
7. Keep already queued work reproducible after later edits or server upgrades.
8. Migrate the broad set of eligible content-facing prompts, including currently hardcoded prompts.
9. Preserve multi-user isolation, context-integrity enforcement, and operational privacy.
10. Maintain no-override behavior at byte-equivalent LLM-provider message boundaries unless a behavior change is separately approved.

## Non-Goals

- A user-facing editor for authentication, authorization, safety policy, tool permissions, provider routing, evaluator/judge, or prompt-enforced machine-protocol instructions
- Administrator or organization-wide prompt overrides in this release
- Multiple named variants per service slot
- Binding service slots to reusable Prompt Library records
- Editing Prompt Studio artifacts
- Live LLM test calls or side-by-side model evaluations from the settings editor
- Automatic merges between a user override and a changed server default
- Persisting unsaved drafts automatically on a browser device
- Treating prompt text as secret or credential storage

## Product Decisions

- Exposure uses a curated allowlist, not a denylist.
- Ownership is per-user. Existing deployment-file overrides remain compatible but are not a new admin UI feature.
- Each definition has one active user override.
- Coordinated prompt parts are edited and versioned atomically.
- Required variables are strict; softer quality diagnostics are warnings.
- Preview uses deterministic safe sample values and never calls an LLM.
- Saved overrides are server-side and account-scoped.
- Full revision history is retained subject to normal storage quotas and lifecycle policy.
- User overrides remain pinned when the server default changes.
- Queued work pins its prompt at enqueue time.
- The first product release targets the broad content-facing eligibility set.
- The settings route is `/settings/service-prompts`.

## Architecture

### ServicePromptRegistry

`ServicePromptRegistry` is the canonical allowlist of editable service-prompt definitions. A definition represents one workflow contract, not necessarily one string. For example, `web.article_summary` may contain coordinated `system` and `user` parts.

Each typed definition declares:

- Stable definition ID and schema version
- Localization label and description keys with English fallbacks
- Primary category, searchable tags, and affected workflow identifiers
- Named prompt parts and their assembly order
- Immutable packaged/default asset references for each part
- Compatibility mappings for existing prompt-loader module/key pairs and environment override variables
- Required and optional variables
- Safe deterministic sample values
- Per-template, per-variable, repetition, and assembled-size budgets
- Whether oversize runtime values are rejected or deterministically truncated
- Editable sections and server-managed locked sections
- Locked-section visibility: `visible` or `hidden`
- Output-contract classification and eligibility evidence
- Sensitivity, deprecation, and replacement metadata

Startup validation rejects duplicate IDs, missing assets, invalid part mappings, undeclared placeholders, invalid sample values, contradictory size budgets, or incomplete replacement metadata.

### Atomic definitions and locked assembly

System/user pairs and other coordinated fragments are one definition and one revision. Users cannot save only half of a coordinated contract.

Editable text is not always the complete prompt. The server may prepend or append locked fragments. Visible contract fragments are shown read-only. Hidden server-managed fragments are represented in preview and provenance without returning their bodies.

A structured-output workflow is eligible only when the output schema is enforced independently, such as by provider response-format support or deterministic validation and retry. If downstream correctness relies primarily on prompt wording, the prompt remains fully locked.

### PromptExecutionContext

Consumers resolve prompts using a typed context rather than a bare user ID. It contains:

- Canonical owner identity, when user-owned
- Operation/workflow identifier
- Optional explicit override and its kind: `literal` or `template`
- Optional immutable prompt pin
- Request, trace, and job identifiers as safe metadata

Userless maintenance activity deliberately uses the server default. Store errors or missing ownership never cause an accidental cross-user or default fallback.

### Resolution precedence

The resolver selects content in this order:

1. Explicit request override
2. Authorized pinned revision or snapshot
3. Active per-user override
4. Deployment default, including existing environment/file overrides
5. Packaged default

All parts are resolved before the resolver computes the atomic server-default bundle digest. That bundle digest drives revision provenance and upstream-change detection.

The immutable `ResolvedServicePrompt` contains rendered or render-ready named parts plus safe provenance:

- Definition ID and schema version
- Source kind
- User revision or snapshot reference, when applicable
- Server-default bundle digest
- Canonical content digest
- Assembly order and locked-section markers

The resolver runs once per request, batch, or job. Lower-level loops receive the immutable result and do not perform repeated database lookups.

### Constrained renderer

The renderer accepts only simple declared placeholders such as `{context}`. It supports escaped literal braces but rejects:

- Attribute access
- Indexing
- Expressions
- Format specifiers
- Filters
- Function calls
- Arbitrary code or template control flow

It uses a linear parser, reports part/variable/line/column diagnostics without echoing surrounding prompt text, and enforces template, variable, repetition, and final assembly budgets. Save, preview, synchronous execution, and worker execution use the same validation and rendering implementation.

## Persistence

Service-prompt records live behind a dedicated repository using each user's existing Prompts database boundary. They are not Prompt Library rows and do not appear in ordinary Prompt search, Prompt Studio, MCP prompt listings, or Prompt exports.

The conceptual data model contains the following record types. The implementation plan may consolidate physical tables if it preserves these contracts.

### Service prompt state

One mutable state record per definition contains:

- Definition ID
- Active revision ID or null
- Monotonically increasing generation
- Last acknowledged server-default digest
- Updated timestamp

The generation changes for save, restore, reset, and default-change acknowledgement. It is the optimistic-concurrency token even when the active revision remains null.

### Immutable revisions

Each revision contains:

- Revision ID and sequence number
- Definition ID
- Atomic editable parts
- Action: save, restore, or reset
- Registry schema version
- Base server-default bundle digest
- Canonical content digest
- Creation timestamp
- Creator identity metadata

Revision content is immutable. Integrity/activation status is tracked separately so the content row never changes.

Reset clears the active pointer but preserves history. Restore revalidates an older revision against the current definition and creates a new revision; it never rewinds the pointer. An incompatible historical revision returns an actionable compatibility report.

### Content-addressed snapshots

Active user revisions are already immutable, so queued jobs can pin them directly. Content-addressed snapshots are created only when an immutable stored source does not already exist, notably:

- Effective server defaults that must survive a later deployment change
- Explicit per-request prompt overrides used by queued work

Snapshots contain prompt templates and safe provenance only. They do not contain rendered source documents, runtime variable values, credentials, or other job input.

### Prompt pins and mutation receipts

A prompt pin binds a definition, owner, immutable revision/snapshot reference, and digest to queued work. It begins pending and is bound to the returned job UUID after enqueue.

Mutation receipts map a client mutation ID to the completed result so safe network retries do not create duplicate revisions or misleading concurrency errors.

### Concurrency and integrity activation

Mutations use `expected_generation`. A stale generation returns a conflict and preserves the client's draft.

Activation follows:

1. Validate the draft against the current registry contract.
2. Append an immutable pending revision.
3. Register and verify the exact canonical digest with context integrity.
4. Advance the active state and generation only after acceptance.

If integrity registration fails, the prior effective revision remains active. In hardened deployments requiring an external trust decision, the new revision may remain pending approval.

### Cross-database enqueue protocol

Prompt storage and Jobs storage do not share a transaction. Enqueue therefore uses a small saga:

1. Commit the prompt pin.
2. Enqueue the owner-scoped job referencing the pin ID and digest.
3. Bind the pin to the returned job UUID.

A reconciler repairs binding failures and garbage-collects unreferenced pins only after a grace period and a check of active and retained jobs. A job never becomes runnable before its referenced pin exists. Workers verify owner, definition, digest, and integrity state before use.

Pinned jobs are retained according to Jobs retention. Referenced revisions and snapshots cannot be removed. Revision history is cursor-paginated and subject to account storage quotas.

## API Design

Add an authenticated `/api/v1/service-prompts` namespace. All ownership comes from the authenticated principal; endpoints never accept a target user ID.

### Capability contract

The server exposes:

```json
{
  "service_prompts": {
    "enabled": true,
    "mode": "enabled",
    "contract_version": 1
  }
}
```

Clients use this to distinguish supported, unavailable, read-only, bypass, and incompatible servers.

### Endpoints

- `GET /catalog`
  - Metadata and current-user state summaries
  - Search/filter inputs as needed
  - No prompt bodies
  - Private composite ETag based on registry digest and user-state generation

- `GET /{definition_id}`
  - Server-default, active, and effective editable parts when visible and trusted
  - Structured provenance, validation contract, generation, and upstream-change state
  - Hidden or quarantined bodies omitted
  - `Cache-Control: private, no-store`

- `POST /{definition_id}/preview`
  - Validates draft parts
  - Renders deterministic safe samples
  - Returns named rendered parts, variable diagnostics, locked-section markers, and assembly order
  - Never invokes an LLM

- `PUT /{definition_id}/override`
  - Saves all editable parts atomically
  - Requires expected generation and client mutation ID

- `POST /{definition_id}/reset`
  - Clears the active override after validation and concurrency checks

- `POST /{definition_id}/acknowledge-default`
  - Records acknowledgement of the current server-default digest without rewriting prompt content

- `GET /{definition_id}/revisions`
  - Cursor-paginated metadata history without bodies

- `GET /{definition_id}/revisions/{revision_id}`
  - Owner-scoped revision detail with private/no-store caching

- `POST /{definition_id}/revisions/{revision_id}/restore`
  - Validates the historical revision against the current contract and creates a new active revision

Preview and mutation endpoints share runtime validation, body limits, rate limits, idempotency behavior, and stable machine-readable error codes.

### API failure semantics

- `422`: invalid draft with part-specific diagnostics
- `409`: stale generation or incompatible restore
- `413`: template or request body too large
- `403`: authenticated principal lacks access
- `404`: capability/definition is unavailable to this principal
- `429`: mutation or preview rate limit
- `503`: expected prompt store or integrity service is temporarily unavailable

Quarantined or incompatible expected content fails closed with a recovery code. Responses, logs, and telemetry do not echo prompt bodies.

## Runtime and Job Behavior

Synchronous services resolve once at their public service boundary. Existing explicit request prompt fields remain supported and take precedence. A migration must preserve whether an existing field is literal final text or a template.

Queued endpoints commit a pin before enqueue. Job payloads and job-status responses contain only safe references and digests, never raw prompt bodies. Workers distinguish:

- Temporary prompt-store unavailability: retryable
- Missing or digest-mismatched pin: permanent integrity failure
- Override execution held by operator mode: held/retryable without substituting a different prompt

Editing, resetting, acknowledging, or upgrading defaults does not alter already pinned work.

## WebUI and Browser Extension

### Information architecture

Add `/settings/service-prompts` as a dedicated advanced page under AI & Models. Keep `/settings/prompt` as the link to the reusable Prompt workspace.

The page introduction explains the difference among:

- Service Prompts: automatic backend workflow defaults
- Prompt Library: reusable prompts selected by the user
- Conversation system prompts: chat-specific behavior

The implementation belongs in the shared UI package. The WebUI and extension options app render the same route. The narrow extension sidepanel uses the existing platform navigation abstraction to open the full options route.

### Catalog experience

The catalog provides:

- Search across names, descriptions, tags, and affected workflows
- Primary categories with secondary tags
- Filters for Customized, Server default, Upstream changed, and Needs attention
- Status summaries without loading prompt bodies
- Clear workflow-impact descriptions

Categories include summarization, RAG, media/audio, documents/web, extraction/chunking, and reports/digests, but eligibility is governed by the registry matrix rather than category alone.

### Editor experience

The editor shows:

- Definition description and affected workflows
- Named editable parts in lightweight accessible text areas
- Required/optional variable chips with cursor insertion
- Visible locked sections as read-only text
- Hidden locked sections as labeled server-managed markers
- Server-default, customized, and effective provenance
- Part-by-part plain-text comparison with unchanged regions collapsed
- Deterministic preview and assembly order
- Save, reset, acknowledgement, history, and restore actions

The UI does not add a heavyweight code-editor dependency. Prompt text and diffs are escaped plain text and are never rendered as rich Markdown or HTML.

### Interaction states

- Inline validation plus an accessible error summary
- Dirty-state navigation and native close protection
- In-memory draft preservation during connection errors
- Retry and Copy Draft actions; no automatic device persistence
- Conflict handling that preserves the local draft and offers reload/copy, never automatic merge
- Upstream-change banner with Compare, Keep override, and Use server default actions
- Success messaging that changes apply to new work while queued jobs keep their pins
- Explicit disconnected, unsupported-server, read-only, bypass, pending-approval, quarantined, and incompatible states

“Keep override” stores the acknowledged default digest. “Use server default” clears the override rather than copying the default, so later server-default updates continue to flow through.

### Responsive and accessible behavior

Desktop may use a master-detail layout. Narrow screens use list-first navigation followed by a full-width editor with a clear Back action. The route must not create page-level horizontal scrolling at 390 px or extension-option widths.

Keyboard operation, focus restoration, labeled controls, screen-reader status updates, touch targets, and validation focus are required. Localization uses client translation keys with server fallbacks; placeholder identifiers remain literal and untranslated.

## Eligibility and Security

### Eligible prompts

Eligible definitions are user-visible content-generation or content-analysis instructions whose modification does not control authorization or an unenforced machine protocol. The inventory must record:

- Source and runtime consumer
- Data owner and workflow owner
- Variables and assembly behavior
- Output dependency
- Locked fragments
- Visibility
- Eligibility decision and reason

### Excluded prompts

The following remain locked:

- Authentication and authorization instructions
- Safety and moderation policy
- Tool and MCP permission/policy prompts
- Provider/model routing and tool-selection control prompts
- Evaluator, judge, grader, and security-review prompts
- Instructions whose output is parsed as a machine protocol without independent enforcement
- Hidden provider- or deployment-sensitive instructions

### Authorization boundary

Prompt text never grants authority. Tools, MCP calls, network access, filesystem access, data retrieval, and provider actions remain governed by existing RBAC, scope, policy, and argument validation on every call.

### Privacy

Prompt content is not secret storage. The UI warns against credentials. Bodies and rendered values are excluded from:

- Ordinary logs
- Audit message text
- Metrics labels
- Job payload/status summaries
- Notifications
- Validation/error responses

Audit metadata includes actor, definition ID, action, revision/digest, result, and timestamp.

Private account backups include service-prompt state and history. Shareable Chatbooks, ordinary Prompt exports, and MCP prompt catalogs exclude service prompts by default unless a later explicitly approved portability design changes that boundary.

### Context integrity

Server defaults use existing exact-byte file verification. User revisions and snapshots use canonical asset identities scoped by owner, definition, and immutable version. At use time, the resolver verifies the exact bytes or row version consumed.

Out-of-band mutations are quarantined. Normal-mode owner saves can activate after local integrity acceptance. Hardened deployments may require external approval and keep a revision pending while the previous effective revision remains active.

Review and settings surfaces never render untrusted content as rich HTML or feed it into model-assisted review.

## Compatibility

### Existing prompt loader and deployment overrides

The legacy `prompt_loader` remains for locked and not-yet-migrated prompts. A migrated workflow cannot mix registry-managed and direct-loaded versions of the same definition.

Existing module/key and `TLDW_PROMPT_FILE_*` mappings become deployment-default providers for eligible parts. The registry computes the atomic bundle after resolving every part. Hidden deployment defaults reveal only safe provenance and digest state.

### Chat and reusable prompts

`preferences.chat.system_prompt`, composer prompt selection, and conversation-specific system prompts remain under Chat settings. Prompt Library and Prompt Studio records retain their existing APIs and user experiences. No new precedence is introduced between those systems and Service Prompts.

### Operator modes

Expose an operator-controlled mode:

- `enabled`: normal read/write and runtime behavior
- `read_only`: existing overrides execute, but mutations are blocked
- `bypass_new_work`: new work visibly uses server defaults; jobs pinned to user content are held rather than silently substituted

Mode changes are visible through capabilities, UI state, logs, and safe audit metadata. They never rewrite or delete history.

## Rollout Plan

This umbrella design is implemented through separate Backlog tasks and reviewable changes.

### Slice 1: Inventory and eligibility matrix

- Inventory centralized and hardcoded prompt sources.
- Trace runtime consumers and prompt assembly.
- Assign stable definition IDs.
- Classify eligible, locked, or deferred with reasons.
- Identify exact explicit-override compatibility behavior.

### Slice 2: Registry and resolver foundation

- Typed definitions and startup validation
- Default-provider compatibility
- Constrained renderer and budgets
- Execution context and immutable resolver result
- No-override provider-message goldens

### Slice 3: Persistence, integrity, and API

- Per-user repository and migrations
- Revisions, generations, acknowledgement, receipts, snapshots, and pins
- Two-phase integrity activation
- Cross-database enqueue reconciliation
- Authenticated API and capability contract

### Slice 4: Shared settings experience

- Shared route, catalog, editor, preview, compare, history, restore, conflicts, and capability states
- WebUI and extension-options integration
- Responsive, accessibility, and localization coverage

### Slice 5: Broad domain migrations

Migrate in independent domain units:

1. Summarization and media/audio analysis
2. Document and web analysis
3. User-visible RAG generation
4. Reports, digests, watchlists, and output creation
5. Eligible extraction/chunking flows with independently enforced contracts

Each domain change includes registry entries, consumer migrations, eligibility-matrix updates, default-message goldens, integration tests, and documentation together.

### Launch and completion gates

- The settings route launches only when the capability contract and at least one complete domain are available.
- A domain is complete only when every eligible call site uses the resolver and remaining prompt sources are explicitly locked or deferred.
- Broad-release completion requires all approved content-facing domains.
- Existing no-override provider messages are byte-equivalent unless separately approved.
- Existing deployment file/environment overrides remain effective.
- Rollback modes are visible and non-destructive.

## Verification Strategy

### Backend unit tests

- Registry uniqueness, asset presence, definition schema, deprecation, and eligibility contracts
- Placeholder parsing, escaped braces, invalid syntax, deterministic rendering, budgets, and locked assembly
- Resolution precedence, literal/template explicit overrides, server-default bundles, userless activity, and hidden defaults
- Revision immutability, generation conflicts, idempotency receipts, acknowledgement, reset, and compatible/incompatible restore
- Two-phase integrity activation and prior-effective-state preservation
- Pin creation, binding, reconciliation, retention, and garbage collection
- Operator modes and stable failure codes

### Property-based tests

- Arbitrary brace and placeholder inputs never execute code or crash the parser
- Rendering is deterministic for identical definitions and variables
- Unknown or malformed expressions are always rejected
- Size and repetition budgets hold across generated templates and values

### API and isolation tests

- Principal-derived ownership and cross-user denial
- Catalog body omission and private composite ETags
- Detail/revision no-store behavior
- Preview/save validator parity
- Stale generation, idempotent retry, limits, rate limits, and safe errors
- Hidden, quarantined, pending, read-only, bypass, and unsupported states

### Runtime and job integration tests

- Two users receive their own overrides for the same definition
- Explicit request prompts retain precedence
- Reset returns to the current server default
- Default changes do not rewrite user revisions
- Queued work stays pinned after edits, reset, acknowledgement, or upgrade
- Missing/tampered pins fail permanently; temporary stores retry
- Provider message arrays are byte-equivalent with no override

### Frontend tests

- Shared routing and capability detection
- Search, categories, filters, and status summaries
- Atomic multi-part editing, variable insertion, preview, and locked sections
- Save, reset, acknowledgement, history, restore, conflict, and pending approval
- Offline draft recovery without persistent storage
- Escaped rendering and Copy Draft
- No horizontal overflow at 390 px and extension widths
- Keyboard, focus, screen-reader, and touch-target behavior

### End-to-end tests

- WebUI user edits a service prompt and the affected workflow uses it
- Packaged extension options edits the same account override and WebUI observes it
- Background job pins the selected revision and remains reproducible
- Upstream server-default change produces compare/keep/reset behavior
- Unsupported and disconnected extension states provide recovery actions

### Security and operational validation

- Cross-user access and path-like definition IDs
- XSS and rich-rendering regression tests
- Prompt-body log, metric, error, notification, and job-status redaction
- Context-integrity tamper and quarantine behavior
- Storage quota and mutation abuse
- Fresh and migrated per-user Prompts databases
- Private backup/restore and older-version ignore behavior
- Bounded-cardinality metrics for definition ID, source kind, validation outcome, and latency
- Touched-scope Bandit, frontend lint/type checks, and focused test suites

## Implementation Planning Decisions

The implementation plan should decide, without changing this product contract:

- Exact Python module and repository class names
- Physical table consolidation and migration numbering
- Default storage quota and snapshot-retention durations
- Exact capability endpoint integration point
- Domain-by-domain stable ID catalog from the completed eligibility inventory
- Which existing diff component is reused by the settings editor
- The concrete Jobs held/retry representation for `bypass_new_work`

## Acceptance Criteria

The design is successfully implemented when:

1. Authenticated users can discover and customize every approved service-prompt definition from the shared WebUI/extension settings route.
2. Overrides are validated, previewed, versioned, restorable, server-side, owner-isolated, and context-integrity checked.
3. Runtime resolution obeys the approved precedence and never silently falls back after an expected-source failure.
4. Queued work references immutable prompt content and remains reproducible.
5. Structured-output, security, routing, judge, and permission boundaries remain protected.
6. Existing deployment defaults, explicit request overrides, chat system prompts, Prompt Library, Prompt Studio, and MCP prompt behavior remain compatible.
7. Broad content-facing domains are migrated through documented eligibility decisions and byte-equivalent default-provider message tests.
8. The settings experience is responsive, accessible, explicit about capability/failure states, and does not persist drafts locally without user action.
